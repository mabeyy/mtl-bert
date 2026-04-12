"""
Single-Task BERT Baseline (Unified Dataset)

Trains a separate BERT model for each task independently,
using the unified multi-label dataset (cyberbully_train_ready.csv).

Features:
- Per-task BERT fine-tuning (one model per task)
- Shared 80/10/10 data split across all tasks (same samples, different labels)
- Multiple seed runs (3 seeds) with mean +/- std reporting
- Per-class metrics for emotion classification
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
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "utils"))
from dataset import load_unified_dataset, compute_metrics, compute_per_class_metrics, EMOTION_CLASSES
from typing import List, Dict


# ============================================================
# Dataset
# ============================================================

class SingleTaskDataset(Dataset):
    """Dataset wrapper for a single task from the unified multi-label data."""

    def __init__(self, samples: List[Dict], task_name: str, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(sample[self.task_name], dtype=torch.long),
        }


# ============================================================
# Model
# ============================================================

class SingleTaskBERT(nn.Module):
    """
    BERT encoder with a single classification head.

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
            self.classifier = nn.Linear(hidden_size, 1)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, dataloader, num_classes, device):
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

def train_single_task(task_name, num_classes, train_samples, val_samples, test_samples,
                      tokenizer, model_name, device, config, seed=0):
    """Train and evaluate a single-task BERT model."""
    checkpoint_dir = os.path.join("checkpoints", "unified-dataset", "stl-bert")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{task_name}_seed{seed}.pt")

    train_ds = SingleTaskDataset(train_samples, task_name, tokenizer, config["max_length"])
    val_ds = SingleTaskDataset(val_samples, task_name, tokenizer, config["max_length"])
    test_ds = SingleTaskDataset(test_samples, task_name, tokenizer, config["max_length"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

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
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_val_f1 = ckpt["best_val_f1"]
            best_model_state = ckpt.get("best_model_state")
            if ckpt.get("mid_epoch", False):
                start_epoch = ckpt["epoch"]
                print(f"  Resumed mid-epoch at epoch {start_epoch + 1}, best Val F1: {best_val_f1:.4f}")
            else:
                start_epoch = ckpt["epoch"] + 1
                print(f"  Resumed from epoch {start_epoch}, best Val F1: {best_val_f1:.4f}")
        except Exception as e:
            print(f"  WARNING: Checkpoint corrupted ({e}). Starting from scratch.")
            os.remove(ckpt_path)

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

        val_metrics, _, _ = evaluate_model(model, val_loader, num_classes, device)
        print(
            f"  Epoch {epoch + 1}/{config['num_epochs']} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        torch.save({
            "epoch": epoch,
            "mid_epoch": False,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "best_model_state": best_model_state,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)
    test_metrics, all_preds, all_labels = evaluate_model(model, test_loader, num_classes, device)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return test_metrics, all_preds, all_labels


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Single-Task BERT Baseline (Unified Dataset)")
    print("=" * 60)

    MODEL_NAME = "bert-base-uncased"
    SEEDS = [42, 123, 456]
    config = {
        "batch_size": 8,
        "num_epochs": 5,
        "max_length": 128,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    task_configs = {
        "sarc": 2,
        "intent": 2,
        "emotion": 6,
    }

    results_dir = os.path.join("results", "unified-dataset", "stl-bert")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nLoading unified dataset...")
    all_samples = load_unified_dataset()
    print(f"  Total samples: {len(all_samples)}")

    # ----- Multi-seed training with progress tracking -----
    progress_path = os.path.join(results_dir, "training_progress.json")
    all_seed_results = {task: [] for task in task_configs}
    all_seed_predictions = {task: [] for task in task_configs}
    all_seed_labels = {task: [] for task in task_configs}

    completed = set()
    if os.path.exists(progress_path):
        progress = load_json(progress_path)
        if progress:
            completed = set(progress.get("completed", []))
            for task in task_configs:
                all_seed_results[task] = progress.get("results", {}).get(task, [])
                all_seed_labels[task] = progress.get("labels", {}).get(task, [])
                all_seed_predictions[task] = progress.get("predictions", {}).get(task, [])
            if completed:
                print(f"\nResuming: {len(completed)} task+seed combos already done.")

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'=' * 60}")
        print(f"  Seed {seed_idx + 1}/{len(SEEDS)} (seed={seed})")
        print(f"{'=' * 60}")

        set_seed(seed)

        # Shared 80/10/10 split — same samples for all tasks
        shuffled = all_samples.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        s1, s2 = int(0.8 * n), int(0.9 * n)
        train_samples = shuffled[:s1]
        val_samples = shuffled[s1:s2]
        test_samples = shuffled[s2:]

        print(f"  Split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

        for task_name, num_classes in task_configs.items():
            combo_key = f"{task_name}_seed{seed}"

            if combo_key in completed:
                print(f"\n--- Skipping: {task_name} (seed={seed}) - already completed ---")
                continue

            print(f"\n--- Training: {task_name} (seed={seed}) ---")

            test_metrics, preds, labels = train_single_task(
                task_name, num_classes,
                train_samples, val_samples, test_samples,
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

            with open(progress_path, "w") as f:
                json.dump({
                    "completed": list(completed),
                    "results": all_seed_results,
                    "predictions": all_seed_predictions,
                    "labels": all_seed_labels,
                }, f, indent=2)
            print(f"  Progress saved ({len(completed)}/{len(SEEDS) * len(task_configs)} done)")

    # ================================================================
    # Aggregated Results
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

    agg_path = os.path.join(results_dir, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {agg_path}")

    # ================================================================
    # Per-class Emotion Analysis
    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Per-Class Metrics: Emotion Task")
    print(f"{'=' * 60}")

    best_seed_idx = int(np.argmax([m["f1"] for m in all_seed_results["emotion"]]))
    best_preds = all_seed_predictions["emotion"][best_seed_idx]
    best_labels = all_seed_labels["emotion"][best_seed_idx]

    per_class = compute_per_class_metrics(best_preds, best_labels, EMOTION_CLASSES)
    print(f"\n  (Best seed: {SEEDS[best_seed_idx]})\n")
    print(per_class["report_str"])

    per_class_save = {}
    for k, v in per_class["report_dict"].items():
        per_class_save[k] = v
    per_class_save["confusion_matrix"] = per_class["confusion_matrix"]
    per_class_path = os.path.join(results_dir, "emotion_per_class_metrics.json")
    with open(per_class_path, "w") as f:
        json.dump(per_class_save, f, indent=2)

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"\n  Dataset: cyberbully_train_ready.csv (unified)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"\n  Output files in '{results_dir}/':")
    print(f"    - aggregated_results.json")
    print(f"    - emotion_per_class_metrics.json")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
