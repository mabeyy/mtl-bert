"""
MTL-BERT Inference -- Weighted Late Fusion for Cyberbullying Detection
(Unified Dataset variant: 6-class emotion)

Loads a trained MTL-BERT model (from mtl_bert_equal_weight_one_dataset.py)
and performs inference using weighted late fusion:

  1. Shared BERT encoder produces contextual embeddings
  2. Three task heads output probabilities:
     - Sarcasm:  p(yes)                (1 value, sigmoid)
     - Harm:     p(yes)                (1 value, sigmoid)
     - Emotion:  [p_sad, ..., p_surp]  (6 values, softmax)
  3. Scale each head's output by domain-driven weights:
     - w_sarcasm = 0.20  (modifier signal)
     - w_harm    = 0.50  (primary indicator)
     - w_emotion = 0.30  (contextual signal)
  4. Concatenate into weighted feature vector z (8 values)
  5. Output: p(cyberbullying), p(not cyberbullying)

Usage:
    python inference_unified.py
    python inference_unified.py --model results/unified-dataset/mtl-equal-weight-one-dataset/mtl_equal_one_dataset_seed456.pt
    python inference_unified.py --text "I hope he fails"
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
import argparse
import os
import json
from typing import Dict, List


# ============================================================
# Config
# ============================================================

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

TASK_CONFIGS = {
    "sarc": 2,
    "intent": 2,
    "emotion": 6,
}

# Domain-driven inference weights (Sec 3.2.6)
INFERENCE_WEIGHTS = {
    "sarc": 0.20,
    "intent": 0.50,
    "emotion": 0.30,
}

EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


# ============================================================
# MTL Model (must match training architecture)
# ============================================================

class MultitaskModel(nn.Module):
    def __init__(self, model_name: str, task_configs: Dict[str, int], dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            out_size = 1 if num_classes == 2 else num_classes
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, out_size),
            )
        self.task_configs = task_configs

    def forward(self, input_ids, attention_mask, task_name):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        return self.task_heads[task_name](pooled)


# ============================================================
# Fusion Classifier (inference only -- not part of MTL training)
# ============================================================

class CyberbullyingFusionClassifier:
    """
    Weighted late fusion for cyberbullying detection.

    Takes the 3 task heads' outputs, scales them by domain-driven
    weights, concatenates into a feature vector, and produces a
    final cyberbullying prediction.
    """

    def __init__(self, model, tokenizer, device, weights=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.weights = weights or INFERENCE_WEIGHTS
        self.model.eval()

        # Fusion dense layer: 8 weighted features -> 2 logits (not cyberbullying, cyberbullying)
        # Input size: sarc(1) + harm(1) + emotion(6) = 8
        self.fusion_layer = nn.Linear(8, 2).to(device)

        # Initialize weights to reflect domain knowledge
        with torch.no_grad():
            self.fusion_layer.weight.zero_()
            self.fusion_layer.bias.zero_()

            # Map: z = [sarc_yes, harm_yes, sad, joy, love, angry, fear, surprise]
            #  idx:      0         1        2    3    4     5      6     7

            # Cyberbullying logit (index 1): positive weights for harmful signals
            self.fusion_layer.weight[1, 0] = 1.0   # sarc_yes -> weak signal
            self.fusion_layer.weight[1, 1] = 6.0   # harm_yes -> strongest signal
            self.fusion_layer.weight[1, 2] = 1.0   # sad -> weak signal
            self.fusion_layer.weight[1, 5] = 1.5   # angry -> moderate signal
            self.fusion_layer.weight[1, 6] = 1.5   # fear -> moderate signal
            self.fusion_layer.bias[1] = -1.5        # require evidence to classify as cyberbullying

            # Not-cyberbullying logit (index 0): positive weights for safe signals
            self.fusion_layer.weight[0, 0] = -1.0  # sarc_yes -> slight push away
            self.fusion_layer.weight[0, 1] = -6.0  # harm_yes -> strong push away
            self.fusion_layer.weight[0, 3] = 3.0   # joy -> not cyberbullying
            self.fusion_layer.weight[0, 4] = 3.0   # love -> not cyberbullying
            self.fusion_layer.weight[0, 7] = 1.5   # surprise -> not cyberbullying
            self.fusion_layer.bias[0] = 1.5         # default toward not cyberbullying

    def get_task_probabilities(self, text):
        """Run text through all 3 task heads and return softmax probabilities."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        task_probs = {}
        with torch.no_grad():
            for task_name, num_classes in TASK_CONFIGS.items():
                logits = self.model(input_ids, attention_mask, task_name)

                if num_classes == 2:
                    prob_yes = torch.sigmoid(logits.squeeze(-1)).item()
                    task_probs[task_name] = [prob_yes]
                else:
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
                    task_probs[task_name] = probs

        return task_probs

    def build_weighted_vector(self, task_probs):
        """
        Scale each task's probabilities by its weight and concatenate.

        Returns:
            z: weighted feature vector (8 values for 1+1+6)
            breakdown: dict with per-task scaled values
        """
        breakdown = {}
        z = []

        for task_name in ["sarc", "intent", "emotion"]:
            w = self.weights[task_name]
            probs = task_probs[task_name]
            scaled = [p * w for p in probs]
            breakdown[task_name] = {
                "raw_probs": probs,
                "weight": w,
                "scaled": scaled,
            }
            z.extend(scaled)

        return z, breakdown

    def predict(self, text):
        """
        Full inference pipeline:
        1. Get task probabilities from all 3 heads
        2. Scale each by domain-driven weights
        3. Concatenate into weighted feature vector z (8 values)
        4. Pass z through a dense layer to get 2 logits
        5. Apply softmax to get final [p_not, p_cyberbullying]
        """
        task_probs = self.get_task_probabilities(text)
        z, breakdown = self.build_weighted_vector(task_probs)

        z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.fusion_layer(z_tensor)

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        p_cyberbullying = float(probs[1])
        p_not = float(probs[0])

        return {
            "text": text,
            "p_cyberbullying": p_cyberbullying,
            "p_not_cyberbullying": p_not,
            "task_probabilities": task_probs,
            "weighted_vector_z": z,
            "logits": logits.squeeze().cpu().numpy().tolist(),
            "breakdown": breakdown,
        }

    def predict_batch(self, texts):
        """Run inference on a list of texts."""
        return [self.predict(text) for text in texts]


# ============================================================
# Display helpers
# ============================================================

def print_prediction(result):
    """Pretty-print a single prediction result."""
    text = result["text"]
    display_text = f'"{text[:60]}..."' if len(text) > 60 else f'"{text}"'

    print(f"\n  Input: {display_text}")
    print(f"  {'─' * 60}")

    tp = result["task_probabilities"]
    print(f"  Task Head Outputs:")
    print(f"    Sarcasm:  p(yes) = {tp['sarc'][0]:.4f}")
    print(f"    Harm:     p(yes) = {tp['intent'][0]:.4f}")

    emo_str = ", ".join(f"{EMOTION_CLASSES[i]}={tp['emotion'][i]:.3f}" for i in range(len(EMOTION_CLASSES)))
    print(f"    Emotion:  [{emo_str}]")

    z = result["weighted_vector_z"]
    print(f"\n  Weighted Feature Vector z ({len(z)} values):")
    labels = ["sarc_yes", "harm_yes", "sad", "joy", "love", "angry", "fear", "surprise"]
    for i, (label, val) in enumerate(zip(labels, z)):
        print(f"    z[{i}] {label:>10s} = {val:.4f}")

    logits = result["logits"]
    print(f"\n  Logits (dense layer output):")
    print(f"    o = [{logits[0]:.4f}, {logits[1]:.4f}]")
    print(f"        [not_cyber,  cyber]")

    print(f"\n  Softmax -> Final Prediction:")
    print(f"    p(cyberbullying)     = {result['p_cyberbullying']:.4f}")
    print(f"    p(not cyberbullying) = {result['p_not_cyberbullying']:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MTL-BERT Inference (Unified Dataset)")
    parser.add_argument("--model", type=str,
                        default="results/unified-dataset/mtl-equal-weight-one-dataset/mtl_equal_one_dataset_seed456.pt",
                        help="Path to trained MTL model .pt file")
    parser.add_argument("--text", type=str, default=None,
                        help="Single text to classify (if not provided, uses sample texts)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MTL-BERT Inference -- Weighted Late Fusion (Unified Dataset)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading model: {args.model}")
    model = MultitaskModel(MODEL_NAME, TASK_CONFIGS)

    if os.path.exists(args.model):
        state_dict = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print("  Model loaded successfully.")
    else:
        print(f"  WARNING: {args.model} not found. Using untrained model.")

    model.to(device)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    classifier = CyberbullyingFusionClassifier(model, tokenizer, device)

    print(f"\nInference weights:")
    for task, w in INFERENCE_WEIGHTS.items():
        print(f"  {task}: {w}")

    if args.text:
        texts = [args.text]
    else:
        texts = [
            # Bullying + Not Sarcastic
            "Everyone hates you, just give up already",
            "You are so stupid, how did you even pass",
            # Bullying + Sarcastic
            "Oh great, another genius who thinks they can bully people online",
            "Wow, you really showed everyone how smart you are, idiot",
            "Sure, keep talking, we all love hearing your dumb opinions",
            # Not Bullying + Not Sarcastic
            "Happy birthday! Hope you have an amazing day!",
            "The weather is really nice today, perfect for a walk",
            # Not Bullying + Sarcastic
            "Oh sure, because waking up early on Monday is my favorite thing",
            "Great, another meeting that could have been an email",
            "Fantastic, my phone died right when I needed it most",
        ]

    print(f"\n{'=' * 60}")
    print(f"  Analyzing {len(texts)} text(s)")
    print(f"{'=' * 60}")

    results = classifier.predict_batch(texts)
    for result in results:
        print_prediction(result)

    print(f"\n{'=' * 60}")
    print(f"  Done -- {len(results)} text(s) analyzed")
    print(f"{'=' * 60}")

    if results:
        z = results[0]["weighted_vector_z"]
        print(f"\n  Example weighted feature vector z (first text):")
        print(f"    z = {[f'{v:.4f}' for v in z]}")
        print(f"    Length: {len(z)} values (sarc:1 + harm:1 + emotion:6)")

    save_path = os.path.join(os.path.dirname(args.model) or "results", "inference_results.json")
    serializable = []
    for r in results:
        serializable.append({
            "text": r["text"],
            "task_probabilities": r["task_probabilities"],
            "weighted_vector_z": [round(v, 6) for v in r["weighted_vector_z"]],
            "logits": [round(v, 6) for v in r["logits"]],
            "p_cyberbullying": round(r["p_cyberbullying"], 6),
            "p_not_cyberbullying": round(r["p_not_cyberbullying"], 6),
            "breakdown": {
                task: {
                    "raw_probs": [round(p, 6) for p in info["raw_probs"]],
                    "weight": info["weight"],
                    "scaled": [round(s, 6) for s in info["scaled"]],
                }
                for task, info in r["breakdown"].items()
            },
        })
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
