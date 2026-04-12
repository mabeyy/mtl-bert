"""
Attention Visualization for Interpretability (Sec 3.5.6)

Inspects attention weight distributions from the shared BERT encoder,
focusing on [CLS] token attention to identify tokens that receive higher
importance during prediction.

For each sample, examines the same representation across all three task heads
to observe how shared contextual information supports different task outputs.

Qualitative criteria (Sec 3.5.6):
  1. Relevance  - emphasized tokens correspond to meaningful linguistic cues
  2. Alignment  - attention patterns are consistent with the predicted label
  3. Consistency - related cues appear coherently across task predictions

Usage:
  python attention_viz.py --model results/mtl-bert.pt
  python attention_viz.py  (uses default path)
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse


# ============================================================
# Model (must match training architecture)
# ============================================================

class MultitaskModel(nn.Module):
    def __init__(self, model_name, task_configs, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        hidden_size = self.encoder.config.hidden_size
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            out = 1 if num_classes == 2 else num_classes
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, out),
            )
        self.task_configs = task_configs

    def forward(self, input_ids, attention_mask, task_name):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True,
        )
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.task_heads[task_name](pooled)
        return logits, outputs.attentions


# ============================================================
# Visualization
# ============================================================

def get_cls_attention(attentions, layer=-1):
    """
    Extract [CLS] token attention from a specific layer.
    Averages across all attention heads.

    Args:
        attentions: tuple of (batch, heads, seq_len, seq_len) per layer
        layer: which layer to use (-1 = last)

    Returns: (seq_len,) array of attention weights from [CLS] to each token
    """
    attn = attentions[layer]  # (batch, heads, seq_len, seq_len)
    # Average across heads, take [CLS] row (index 0)
    cls_attn = attn[0].mean(dim=0)[0]  # (seq_len,)
    return cls_attn.cpu().numpy()


def predict_all_tasks(model, input_ids, attention_mask, task_configs, device):
    """Run model on all task heads and return predictions + attention."""
    model.eval()
    results = {}

    with torch.no_grad():
        for task_name, num_classes in task_configs.items():
            logits, attentions = model(
                input_ids.to(device), attention_mask.to(device), task_name,
            )
            if num_classes == 2:
                prob = torch.sigmoid(logits.squeeze(-1)).item()
                pred = 1 if prob > 0.5 else 0
                conf = prob if pred == 1 else 1 - prob
            else:
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                conf = probs.max().item()

            cls_attn = get_cls_attention(attentions, layer=-1)

            results[task_name] = {
                "prediction": pred,
                "confidence": conf,
                "cls_attention": cls_attn,
            }

    return results


def plot_attention_heatmap(tokens, task_results, task_configs, save_path,
                           title="Attention Visualization"):
    """
    Plot [CLS] attention weights for all three tasks side by side.
    Highlights which tokens the shared encoder emphasizes per task.
    """
    task_names = list(task_configs.keys())
    task_labels = {
        "sarc": ["Non-Sarcastic", "Sarcastic"],
        "intent": ["Not Bullying", "Bullying"],
        "emotion": ["Sad", "Joy", "Love", "Angry", "Fear", "Surprise"],
    }

    fig, axes = plt.subplots(len(task_names), 1, figsize=(14, 3 * len(task_names)))
    if len(task_names) == 1:
        axes = [axes]

    for ax, task_name in zip(axes, task_names):
        result = task_results[task_name]
        attn = result["cls_attention"]
        pred = result["prediction"]
        conf = result["confidence"]
        pred_label = task_labels[task_name][pred]

        # Only show real tokens (exclude padding)
        n_tokens = len(tokens)
        attn_display = attn[:n_tokens]

        # Normalize for display
        attn_display = attn_display / attn_display.max() if attn_display.max() > 0 else attn_display

        # Bar chart
        colors = plt.cm.Blues(attn_display)
        ax.bar(range(n_tokens), attn_display, color=colors)
        ax.set_xticks(range(n_tokens))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Attention Weight")
        ax.set_title(
            f"{task_name.upper()}: {pred_label} (conf={conf:.3f})",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlim(-0.5, n_tokens - 0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attention_comparison(tokens, task_results, task_configs, save_path):
    """
    Overlay [CLS] attention from all tasks on a single plot to show
    consistency of shared representations (Sec 3.5.6 criterion 3).
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    n_tokens = len(tokens)
    width = 0.25
    x = np.arange(n_tokens)
    task_names = list(task_configs.keys())
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for i, task_name in enumerate(task_names):
        attn = task_results[task_name]["cls_attention"][:n_tokens]
        attn = attn / attn.max() if attn.max() > 0 else attn
        ax.bar(x + i * width, attn, width, label=task_name.upper(),
               color=colors[i], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Attention")
    ax.set_title("Cross-Task Attention Comparison (Shared Encoder)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="results/unified-dataset/mtl-equal-weight-one-dataset/mtl_equal_one_dataset_seed456.pt",
                        help="Path to trained MTL model .pt file")
    args = parser.parse_args()

    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 128
    task_configs = {"sarc": 2, "intent": 2, "emotion": 6}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}...")
    model = MultitaskModel(MODEL_NAME, task_configs)

    if os.path.exists(args.model):
        state_dict = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print("  Model loaded successfully.")
    else:
        print(f"  WARNING: {args.model} not found. Using untrained model.")

    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Sample texts matching inference.py (Sec 4.10)
    # Same texts used in the prototype demonstration
    samples = [
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

    output_dir = os.path.join("figures", "attention")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAnalyzing {len(samples)} samples...")
    print("=" * 60)

    for idx, text in enumerate(samples):
        print(f"\nSample {idx+1}: \"{text[:60]}...\"" if len(text) > 60
              else f"\nSample {idx+1}: \"{text}\"")

        # Tokenize
        encoding = tokenizer(
            text, truncation=True, padding=False,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        # Get predictions and attention for all tasks
        task_results = predict_all_tasks(
            model, encoding["input_ids"], encoding["attention_mask"],
            task_configs, device,
        )

        # Print per-task predictions
        task_labels = {
            "sarc": ["Non-Sarcastic", "Sarcastic"],
            "intent": ["Not Bullying", "Bullying"],
            "emotion": ["Sad", "Joy", "Love", "Angry", "Fear", "Surprise"],
        }
        for task, result in task_results.items():
            label = task_labels[task][result["prediction"]]
            print(f"  {task:>8s}: {label} ({result['confidence']:.3f})")

        # Fusion prediction (matching inference.py pipeline)
        from inference import CyberbullyingFusionClassifier, INFERENCE_WEIGHTS
        # Build task_probs in inference.py format
        sarc_conf = task_results["sarc"]["confidence"]
        sarc_pred = task_results["sarc"]["prediction"]
        sarc_prob = sarc_conf if sarc_pred == 1 else 1 - sarc_conf

        intent_conf = task_results["intent"]["confidence"]
        intent_pred = task_results["intent"]["prediction"]
        intent_prob = intent_conf if intent_pred == 1 else 1 - intent_conf

        emo_logits = model(encoding["input_ids"].to(device),
                           encoding["attention_mask"].to(device), "emotion")[0]
        emo_probs = torch.softmax(emo_logits, dim=-1).squeeze().detach().cpu().numpy().tolist()

        # Build weighted vector z (same as inference.py)
        z = [
            sarc_prob * INFERENCE_WEIGHTS["sarc"],
            intent_prob * INFERENCE_WEIGHTS["intent"],
        ]
        for e in emo_probs:
            z.append(e * INFERENCE_WEIGHTS["emotion"])

        # Run through fusion dense layer
        fusion = CyberbullyingFusionClassifier.__new__(CyberbullyingFusionClassifier)
        fusion.fusion_layer = nn.Linear(8, 2).to(device)
        with torch.no_grad():
            fusion.fusion_layer.weight.zero_()
            fusion.fusion_layer.bias.zero_()
            fusion.fusion_layer.weight[1, 0] = 1.0
            fusion.fusion_layer.weight[1, 1] = 6.0
            fusion.fusion_layer.weight[1, 2] = 1.0
            fusion.fusion_layer.weight[1, 5] = 1.5
            fusion.fusion_layer.weight[1, 6] = 1.5
            fusion.fusion_layer.bias[1] = -1.5
            fusion.fusion_layer.weight[0, 0] = -1.0
            fusion.fusion_layer.weight[0, 1] = -6.0
            fusion.fusion_layer.weight[0, 3] = 3.0
            fusion.fusion_layer.weight[0, 4] = 3.0
            fusion.fusion_layer.weight[0, 7] = 1.5
            fusion.fusion_layer.bias[0] = 1.5

            z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device)
            logits_fusion = fusion.fusion_layer(z_tensor)
            probs_fusion = torch.softmax(logits_fusion, dim=-1).squeeze().cpu().numpy()

        p_cyber = float(probs_fusion[1])
        p_not = float(probs_fusion[0])
        fusion_label = "Cyberbullying" if p_cyber > p_not else "Not Cyberbullying"
        print(f"  {'FUSION':>8s}: {fusion_label} (p_cyber={p_cyber:.3f}, p_not={p_not:.3f})")

        # Per-sample attention heatmap
        heatmap_path = os.path.join(output_dir, f"attention_sample_{idx+1}.png")
        plot_attention_heatmap(
            tokens, task_results, task_configs, heatmap_path,
            title=f"Sample {idx+1}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"",
        )

        # Cross-task comparison
        compare_path = os.path.join(output_dir, f"attention_compare_{idx+1}.png")
        plot_attention_comparison(
            tokens, task_results, task_configs, compare_path,
        )

    # Interpretation table (Table 3.12)
    print(f"\n{'=' * 60}")
    print("  Interpretation of Prediction Combinations (Table 3.12)")
    print(f"{'=' * 60}")
    print(f"  {'Sarcasm':>10s}  {'Harmful':>10s}  {'Meaning'}")
    print(f"  {'Yes':>10s}  {'Yes':>10s}  Masked cyberbullying")
    print(f"  {'Yes':>10s}  {'No':>10s}  Harmless humor")
    print(f"  {'No':>10s}  {'Yes':>10s}  Direct cyberbullying")
    print(f"  {'No':>10s}  {'No':>10s}  Neutral")

    print(f"\nAll visualizations saved to '{output_dir}/'")
    print("Done!")


if __name__ == "__main__":
    main()
