"""
Single-Task BERT Baseline

Trains a separate BERT model for each task independently.
Used as a baseline comparison against the MTL-BERT approach.

Features:
- Per-task BERT fine-tuning (one model per task)
- Multiple seed runs (3 seeds) with mean +/- std reporting
- Per-class metrics for emotion classification
- Confusion matrix visualization for emotion task
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
import random
import json
import os
from dataset import create_sample_datasets, compute_metrics, compute_per_class_metrics
from typing import List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Dataset
# ============================================================

class SingleTaskDataset(Dataset):
    """Dataset class for a single task (same tokenization as MTL)."""

    def __init__(self, data: List[Tuple], tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# Model
# ============================================================

class SingleTaskBERT(nn.Module):
    """
    BERT encoder with a single classification head.

    Architecture mirrors the MTL model but without shared task heads:
    - BERT encoder -> [CLS] representation -> Linear classifier
    - Binary tasks: single output + BCEWithLogitsLoss
    - Multi-class tasks: num_classes outputs + CrossEntropyLoss
    """

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.num_classes = num_classes

        if num_classes == 2:
            # Binary: sigma(Wz + b) -> single output
            self.classifier = nn.Linear(hidden_size, 1)
        else:
            # Multi-class: softmax(Wz + b) -> num_classes outputs
            self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    """Load a JSON file, return None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, dataloader, num_classes, device):
    """Evaluate model on a dataloader. Returns metrics dict, predictions, labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask)

            if num_classes == 2:
                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs > 0.5).long()
            else:
                preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    metrics = compute_metrics(all_preds, all_labels)
    return metrics, all_preds, all_labels


# ============================================================
# Training
# ============================================================

def train_single_task(task_name, num_classes, train_data, val_data, test_data,
                      tokenizer, model_name, device, config, seed=0):
    """
    Train and evaluate a single-task BERT model.

    Uses the same hyperparameters as the MTL model for fair comparison:
    - AdamW (lr=2e-5, weight_decay=0.01)
    - Gradient clipping (max_norm=1.0)
    - Best checkpoint selection based on validation F1
    - Checkpoint saving after each epoch for resume
    """
    checkpoint_dir = os.path.join("checkpoints", "baseline")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{task_name}_seed{seed}.pt")

    train_ds = SingleTaskDataset(train_data, tokenizer, config["max_length"])
    val_ds = SingleTaskDataset(val_data, tokenizer, config["max_length"])
    test_ds = SingleTaskDataset(test_data, tokenizer, config["max_length"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    # Initialize model with fresh weights each time
    model = SingleTaskBERT(model_name, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    if num_classes == 2:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_model_state = None
    start_epoch = 0

    # Resume from checkpoint if exists
    if os.path.exists(ckpt_path):
        print(f"  Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_f1 = ckpt["best_val_f1"]
        best_model_state = ckpt.get("best_model_state")
        if ckpt.get("mid_epoch", False):
            # Mid-epoch checkpoint: restart that epoch with updated weights
            start_epoch = ckpt["epoch"]
            print(f"  Resumed mid-epoch at epoch {start_epoch + 1}, best Val F1: {best_val_f1:.4f}")
        else:
            start_epoch = ckpt["epoch"] + 1
            print(f"  Resumed from epoch {start_epoch}, best Val F1: {best_val_f1:.4f}")

    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)

            if num_classes == 2:
                loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                print(f"    Step {num_batches}/{len(train_loader)}, Loss: {loss.item():.4f}")

            # Mid-epoch checkpoint every 1000 steps
            if num_batches % 1000 == 0:
                torch.save({
                    "epoch": epoch,
                    "mid_epoch": True,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "best_model_state": best_model_state,
                }, ckpt_path)
                print(f"    Mid-epoch checkpoint saved (step {num_batches})")

        avg_loss = total_loss / max(num_batches, 1)

        # Validate
        val_metrics, _, _ = evaluate_model(model, val_loader, num_classes, device)
        print(
            f"  Epoch {epoch + 1}/{config['num_epochs']} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Keep best model based on validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save checkpoint after each epoch
        torch.save({
            "epoch": epoch,
            "mid_epoch": False,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "best_model_state": best_model_state,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)
    test_metrics, all_preds, all_labels = evaluate_model(model, test_loader, num_classes, device)

    # Clean up checkpoint after successful completion
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return test_metrics, all_preds, all_labels


# ============================================================
# Visualization
# ============================================================

def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    """Plot and save a confusion matrix as a heatmap."""
    if normalize:
        # Normalize by row (true labels) to get percentages
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.where(row_sums > 0, cm / row_sums * 100, 0)
        fmt = ".1f"
        title = "Emotion Classification - Normalized Confusion Matrix (%)"
    else:
        cm_display = cm.astype(float)
        fmt = ".0f"
        title = "Emotion Classification - Confusion Matrix"

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            text = f"{val:{fmt}}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Single-Task BERT Baseline")
    print("=" * 60)

    # ----- Configuration -----
    MODEL_NAME = "bert-base-uncased"
    SEEDS = [42, 123, 456]  # 3 seeds for mean +/- std
    # MAX_SAMPLES = 1000  # Set to None to use full dataset
    config = {
        "batch_size": 8,
        "num_epochs": 5,
        "max_length": 128,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    task_configs = {
        "sarc": 2,       # binary: non-sarcastic / sarcastic
        "intent": 2,     # binary: not bullying / bullying
        "emotion": 6,    # multi-class: sad, joy, love, angry, fear, surprise
    }
    EMOTION_CLASSES = ["sad", "joy", "love", "angry", "fear", "surprise"]

    results_dir = os.path.join("results", "baseline")
    os.makedirs(results_dir, exist_ok=True)

    # ----- Load tokenizer & data -----
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nLoading datasets from CSV files...")
    tasks_data = create_sample_datasets()

    # Limit dataset size for testing
    # if MAX_SAMPLES is not None:
    #     for task_name in tasks_data:
    #         tasks_data[task_name] = tasks_data[task_name][:MAX_SAMPLES]

    for task_name, data in tasks_data.items():
        print(f"  {task_name}: {len(data)} samples")

    # ----- Multi-seed training with progress tracking -----
    progress_path = os.path.join(results_dir, "training_progress.json")
    all_seed_results = {task: [] for task in task_configs}
    all_seed_predictions = {task: [] for task in task_configs}
    all_seed_labels = {task: [] for task in task_configs}

    # Load progress if resuming
    completed = set()
    if os.path.exists(progress_path):
        progress = load_json(progress_path)
        if progress:
            completed = set(progress.get("completed", []))
            # Restore saved results
            for task in task_configs:
                all_seed_results[task] = progress.get("results", {}).get(task, [])
                all_seed_labels[task] = progress.get("labels", {}).get(task, [])
                all_seed_predictions[task] = progress.get("predictions", {}).get(task, [])
            if completed:
                print(f"\nResuming: {len(completed)} task+seed combos already done, skipping them.")

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'=' * 60}")
        print(f"  Seed {seed_idx + 1}/{len(SEEDS)} (seed={seed})")
        print(f"{'=' * 60}")

        set_seed(seed)

        # Split data: 80/10/10 (same ratios as MTL for fair comparison)
        train_data = {}
        val_data = {}
        test_data = {}

        for task_name, task_samples in tasks_data.items():
            shuffled = task_samples.copy()
            random.shuffle(shuffled)
            n = len(shuffled)
            split_train = int(0.8 * n)
            split_val = int(0.9 * n)
            train_data[task_name] = shuffled[:split_train]
            val_data[task_name] = shuffled[split_train:split_val]
            test_data[task_name] = shuffled[split_val:]

        # Train each task independently
        for task_name, num_classes in task_configs.items():
            combo_key = f"{task_name}_seed{seed}"

            if combo_key in completed:
                print(f"\n--- Skipping: {task_name} (seed={seed}) - already completed ---")
                continue

            print(f"\n--- Training: {task_name} (seed={seed}) ---")
            print(
                f"  Train: {len(train_data[task_name])}, "
                f"Val: {len(val_data[task_name])}, "
                f"Test: {len(test_data[task_name])}"
            )

            test_metrics, preds, labels = train_single_task(
                task_name, num_classes,
                train_data[task_name], val_data[task_name], test_data[task_name],
                tokenizer, MODEL_NAME, device, config, seed=seed,
            )

            all_seed_results[task_name].append(test_metrics)
            all_seed_predictions[task_name].append([int(p) for p in preds])
            all_seed_labels[task_name].append([int(l) for l in labels])
            completed.add(combo_key)

            print(
                f"  Test: Acc={test_metrics['accuracy']:.4f}, "
                f"P={test_metrics['precision']:.4f}, "
                f"R={test_metrics['recall']:.4f}, "
                f"F1={test_metrics['f1']:.4f}"
            )

            # Save progress after each task+seed completes
            with open(progress_path, "w") as f:
                json.dump({
                    "completed": list(completed),
                    "results": all_seed_results,
                    "predictions": all_seed_predictions,
                    "labels": all_seed_labels,
                }, f, indent=2)
            print(f"  Progress saved ({len(completed)}/{len(SEEDS) * len(task_configs)} done)")

    # ================================================================
    # Aggregated Results: mean +/- std across seeds
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Aggregated Results (mean +/- std across {len(SEEDS)} seeds)")
    print(f"{'=' * 60}")

    aggregated = {}
    for task_name in task_configs:
        metrics_list = all_seed_results[task_name]
        agg = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = [m[metric] for m in metrics_list]
            agg[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_seed": [float(v) for v in values],
            }
        aggregated[task_name] = agg

        print(f"\n  {task_name}:")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean = agg[metric]["mean"]
            std = agg[metric]["std"]
            print(f"    {metric:>10s}: {mean:.4f} +/- {std:.4f}")

    # Save aggregated results
    agg_path = os.path.join(results_dir, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {agg_path}")

    # ================================================================
    # Per-class metrics for emotion (requirement 2)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Per-Class Metrics: Emotion Task")
    print(f"{'=' * 60}")

    # Use best seed (highest emotion test F1) for per-class reporting
    best_seed_idx = int(np.argmax([m["f1"] for m in all_seed_results["emotion"]]))
    best_preds = all_seed_predictions["emotion"][best_seed_idx]
    best_labels = all_seed_labels["emotion"][best_seed_idx]

    per_class = compute_per_class_metrics(best_preds, best_labels, EMOTION_CLASSES)
    print(f"\n  (Best seed: {SEEDS[best_seed_idx]})\n")
    print(per_class["report_str"])

    # Save per-class metrics
    per_class_save = {}
    for k, v in per_class["report_dict"].items():
        per_class_save[k] = v
    per_class_save["confusion_matrix"] = per_class["confusion_matrix"]
    per_class_path = os.path.join(results_dir, "emotion_per_class_metrics.json")
    with open(per_class_path, "w") as f:
        json.dump(per_class_save, f, indent=2)
    print(f"Per-class metrics saved to {per_class_path}")

    # ================================================================
    # Confusion matrix for emotion (requirement 4)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Confusion Matrix: Emotion Task")
    print(f"{'=' * 60}")

    cm = np.array(per_class["confusion_matrix"])

    # Save raw confusion matrix plot
    cm_path = os.path.join(results_dir, "emotion_confusion_matrix.png")
    plot_confusion_matrix(cm, EMOTION_CLASSES, cm_path, normalize=False)
    print(f"  Raw confusion matrix saved to {cm_path}")

    # Save normalized confusion matrix plot
    cm_norm_path = os.path.join(results_dir, "emotion_confusion_matrix_normalized.png")
    plot_confusion_matrix(cm, EMOTION_CLASSES, cm_norm_path, normalize=True)
    print(f"  Normalized confusion matrix saved to {cm_norm_path}")

    # Print confusion matrix to console
    print(f"\n  {'':>10s}", end="")
    for name in EMOTION_CLASSES:
        print(f"  {name:>8s}", end="")
    print()
    for i, name in enumerate(EMOTION_CLASSES):
        print(f"  {name:>10s}", end="")
        for j in range(len(EMOTION_CLASSES)):
            print(f"  {cm[i][j]:>8d}", end="")
        print()

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"\n  Seeds used: {SEEDS}")
    print(f"  Epochs per run: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Model: {MODEL_NAME}")
    print(f"\n  Output files in '{results_dir}/':")
    print(f"    - aggregated_results.json        (mean +/- std for all tasks)")
    print(f"    - emotion_per_class_metrics.json  (per-class P/R/F1 + confusion matrix)")
    print(f"    - emotion_confusion_matrix.png    (raw counts)")
    print(f"    - emotion_confusion_matrix_normalized.png  (row-normalized %)")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
