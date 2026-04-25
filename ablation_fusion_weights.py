"""
Fusion Weight Ablation Study for SA-MTL Cyberbullying Detection

Evaluates different fusion weight configurations on the test set
across all 3 seeds (42, 123, 456) to determine the impact of
varying w_harm, w_sarcasm, and w_emotion on the final cyberbullying
classification.

The fusion weights control how each task head's output is scaled
before being passed to the dense classification layer (Section 3.2.6).

Usage:
    python ablation_fusion_weights.py
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import csv
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List


# ============================================================
# Config (must match training)
# ============================================================

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
SEEDS = [42, 123, 456]
DATA_PATH = os.path.join("data", "cyberbully_train_ready.csv")

EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_TO_IDX = {name: i for i, name in enumerate(EMOTION_CLASSES)}

TASK_CONFIGS = {
    "sarc": 2,
    "intent": 2,
    "emotion": 6,
}

MODEL_DIR = "results/unified-dataset/mtl-equal-weight-one-dataset"

# ============================================================
# Fusion weight configurations to ablate
# ============================================================

# Fusion weight configurations derived from experimental results.
#
# F1-Proportional (Config 1): weights proportional to each task head's
# F1 score from Table 4.3 (sarc=0.5978, harm=0.6465, emotion=0.3896).
# Each weight = task_F1 / sum(all_F1), so higher-performing heads
# receive proportionally more influence in the fusion.
#
# Equal (Config 2): all three tasks weighted equally (1/3 each),
# serving as a neutral reference with no task prioritization.
#
# Harm-Dominant (Config 3): assigns majority weight to the harm head
# since harmful intent is the most direct indicator of cyberbullying.
# Suggested by panelists as an example configuration.
#
# Emotion-Prioritized (Config 4): increases emotion weight to test
# whether contextual affective signals improve classification.
#
# Sarcasm-Reduced (Config 5): minimizes sarcasm weight given it is
# the weakest-performing head (F1=0.5978, Table 4.3), redistributing
# its share to the stronger harm and emotion heads.

FUSION_CONFIGS = {
    "Config 1 (F1-Proportional)": {"sarc": 0.37, "intent": 0.40, "emotion": 0.24},
    "Config 2 (Equal)":           {"sarc": 0.33, "intent": 0.34, "emotion": 0.33},
    "Config 3 (Harm-Dominant)":   {"sarc": 0.20, "intent": 0.60, "emotion": 0.20},
    "Config 4 (Emotion-Prioritized)": {"sarc": 0.20, "intent": 0.50, "emotion": 0.30},
    "Config 5 (Sarcasm-Reduced)": {"sarc": 0.10, "intent": 0.50, "emotion": 0.40},
}


# ============================================================
# Model (must match training architecture exactly)
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
        pooled = outputs.last_hidden_state[:, 0]
        return self.task_heads[task_name](pooled)


# ============================================================
# Data loading (must match training split logic)
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_path):
    samples = []
    skipped = 0
    with open(data_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text", "") or "").strip()
            if not text:
                skipped += 1
                continue
            try:
                sarc = int(row["sarcasm"])
                intent = int(row["harm"])
            except (ValueError, KeyError):
                skipped += 1
                continue
            emotion_label = (row.get("emotion", "") or "").strip()
            if emotion_label not in EMOTION_TO_IDX:
                skipped += 1
                continue

            # cyberbullying ground truth label
            try:
                cyberbullying = int(row["cyberbullying"])
            except (ValueError, KeyError):
                skipped += 1
                continue

            samples.append({
                "text": text,
                "sarc": sarc,
                "intent": intent,
                "emotion": EMOTION_TO_IDX[emotion_label],
                "cyberbullying": cyberbullying,
            })
    if skipped > 0:
        print(f"  Skipped {skipped} rows (missing/invalid labels)")
    return samples


def get_test_split(all_samples, seed):
    """Reproduce the exact test split used during training."""
    set_seed(seed)
    shuffled = all_samples.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    s1, s2 = int(0.8 * n), int(0.9 * n)
    return shuffled[s2:]


# ============================================================
# Fusion classifier with configurable weights
# ============================================================

def build_fusion_layer(device):
    """Build the fixed dense classification layer (same as inference.py)."""
    fusion_layer = nn.Linear(8, 2).to(device)
    with torch.no_grad():
        fusion_layer.weight.zero_()
        fusion_layer.bias.zero_()

        # Cyberbullying logit (index 1)
        fusion_layer.weight[1, 0] = 1.0   # sarc_yes
        fusion_layer.weight[1, 1] = 6.0   # harm_yes
        fusion_layer.weight[1, 2] = 1.0   # sadness
        fusion_layer.weight[1, 5] = 1.5   # anger
        fusion_layer.weight[1, 6] = 1.5   # fear
        fusion_layer.bias[1] = -1.5

        # Not-cyberbullying logit (index 0)
        fusion_layer.weight[0, 0] = -1.0  # sarc_yes
        fusion_layer.weight[0, 1] = -6.0  # harm_yes
        fusion_layer.weight[0, 3] = 3.0   # joy
        fusion_layer.weight[0, 4] = 3.0   # love
        fusion_layer.weight[0, 7] = 1.5   # surprise
        fusion_layer.bias[0] = 1.5

    return fusion_layer


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  Fusion Weight Ablation Study")
    print("  SA-MTL Cyberbullying Detection")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load full dataset
    print(f"Loading dataset: {DATA_PATH}")
    all_samples = load_dataset(DATA_PATH)
    print(f"  Total samples: {len(all_samples)}")

    # Build fusion layer (same for all configs — only the input scaling changes)
    fusion_layer = build_fusion_layer(device)

    # Store results: {config_name: {seed: metrics}}
    all_results = {}

    # Load model once per seed, run all configs on it, then free it
    # This avoids OOM from loading BERT 15 times
    seed_cache = {}  # seed -> {test_samples, task_probs per sample}

    for seed in SEEDS:
        test_samples = get_test_split(all_samples, seed)

        model_path = os.path.join(MODEL_DIR, f"mtl_equal_one_dataset_seed{seed}.pt")
        if not os.path.exists(model_path):
            print(f"  WARNING: {model_path} not found, skipping seed {seed}")
            continue

        print(f"\n  Loading model for seed {seed}...")
        model = MultitaskModel(MODEL_NAME, TASK_CONFIGS)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Pre-compute all task head probabilities once (these don't depend on fusion weights)
        print(f"  Computing task head outputs for {len(test_samples)} test samples...")
        cached_probs = []
        for sample in test_samples:
            encoding = tokenizer(
                sample["text"], truncation=True, padding="max_length",
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                sarc_logits = model(input_ids, attention_mask, "sarc")
                harm_logits = model(input_ids, attention_mask, "intent")
                emo_logits = model(input_ids, attention_mask, "emotion")

                sarc_prob = torch.sigmoid(sarc_logits.squeeze(-1)).item()
                harm_prob = torch.sigmoid(harm_logits.squeeze(-1)).item()
                emo_probs = torch.softmax(emo_logits, dim=-1).squeeze().cpu().numpy().tolist()

            cached_probs.append({
                "sarc": sarc_prob,
                "harm": harm_prob,
                "emotion": emo_probs,
                "cyberbullying": sample["cyberbullying"],
            })

        seed_cache[seed] = cached_probs

        # Free GPU memory
        del model, state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  Done. Model freed.")

    # Now run all fusion weight configs using cached probabilities (no model needed)
    for config_name, weights in FUSION_CONFIGS.items():
        print(f"\n{'-' * 70}")
        print(f"  {config_name}")
        print(f"  w_harm={weights['intent']:.2f}, w_sarcasm={weights['sarc']:.2f}, w_emotion={weights['emotion']:.2f}")
        print(f"{'-' * 70}")

        config_results = []

        for seed in SEEDS:
            if seed not in seed_cache:
                continue

            cached_probs = seed_cache[seed]
            all_preds = []
            all_labels = []

            for cp in cached_probs:
                # Build weighted feature vector z
                z = []
                z.append(cp["sarc"] * weights["sarc"])
                z.append(cp["harm"] * weights["intent"])
                for ep in cp["emotion"]:
                    z.append(ep * weights["emotion"])

                z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = fusion_layer(z_tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                    pred_cyber = 1 if probs[1] >= 0.5 else 0

                all_preds.append(pred_cyber)
                all_labels.append(cp["cyberbullying"])

            # Compute metrics
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

            config_results.append({
                "seed": seed,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "n_test": len(cached_probs),
            })

            print(f"  Seed {seed}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  (n={len(cached_probs)})")

        if config_results:
            mean_acc = np.mean([r["accuracy"] for r in config_results])
            mean_prec = np.mean([r["precision"] for r in config_results])
            mean_rec = np.mean([r["recall"] for r in config_results])
            mean_f1 = np.mean([r["f1"] for r in config_results])

            std_acc = np.std([r["accuracy"] for r in config_results])
            std_prec = np.std([r["precision"] for r in config_results])
            std_rec = np.std([r["recall"] for r in config_results])
            std_f1 = np.std([r["f1"] for r in config_results])

            print(f"\n  Mean:  Acc={mean_acc:.4f}+/-{std_acc:.4f}  Prec={mean_prec:.4f}+/-{std_prec:.4f}  "
                  f"Rec={mean_rec:.4f}+/-{std_rec:.4f}  F1={mean_f1:.4f}+/-{std_f1:.4f}")

            all_results[config_name] = {
                "weights": weights,
                "per_seed": config_results,
                "mean": {
                    "accuracy": round(mean_acc, 4),
                    "precision": round(mean_prec, 4),
                    "recall": round(mean_rec, 4),
                    "f1": round(mean_f1, 4),
                },
                "std": {
                    "accuracy": round(std_acc, 4),
                    "precision": round(std_prec, 4),
                    "recall": round(std_rec, 4),
                    "f1": round(std_f1, 4),
                },
            }

    # ── Summary table ──
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY: Fusion Weight Ablation Results")
    print(f"  (Cyberbullying classification, mean +/- std across 3 seeds)")
    print(f"{'=' * 70}")
    print(f"\n  {'Config':<28s} {'w_h':>4s} {'w_s':>4s} {'w_e':>4s}  {'Accuracy':>14s}  {'Precision':>14s}  {'Recall':>14s}  {'F1':>14s}")
    print(f"  {'-' * 120}")

    best_f1 = -1
    best_config = None

    for config_name, data in all_results.items():
        w = data["weights"]
        m = data["mean"]
        s = data["std"]
        row = (f"  {config_name:<28s} {w['intent']:.2f} {w['sarc']:.2f} {w['emotion']:.2f}  "
               f"{m['accuracy']:.4f}+/-{s['accuracy']:.4f}  "
               f"{m['precision']:.4f}+/-{s['precision']:.4f}  "
               f"{m['recall']:.4f}+/-{s['recall']:.4f}  "
               f"{m['f1']:.4f}+/-{s['f1']:.4f}")
        print(row)

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_config = config_name

    print(f"\n  Configuration that yielded the highest F1: {best_config} (F1={best_f1:.4f})")

    # Save results
    save_dir = "results/unified-dataset/fusion-weight-ablation"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ablation_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
