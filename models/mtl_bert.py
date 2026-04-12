"""
MTL-BERT Equal Weights — Single Multi-Label Dataset

Uses the Cyberbully_corrected_emotion_sentiment dataset where each sample
has labels for all 3 tasks (sarcasm, cyberbullying, emotion) on the SAME text.

Tasks:
  - sarc:    binary   (Yes=1, No=0)
  - intent:  binary   (Harm=1, No Harm=0)
  - emotion: 6-class  (sadness, joy, love, anger, fear, surprise)

Usage:
    python mtl_bert_equal_weight_one_dataset.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
import random
import json
import csv
import os
from typing import Dict, List, Tuple


# ============================================================
# Config
# ============================================================

MODEL_NAME = "bert-base-uncased"
SEEDS = [42, 123, 456]
BATCH_SIZE = 8
NUM_EPOCHS = 5
MAX_LENGTH = 128

DATA_PATH = os.path.join("data", "cyberbully_train_ready.csv")

EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_TO_IDX = {name: i for i, name in enumerate(EMOTION_CLASSES)}

TASK_CONFIGS = {
    "sarc": 2,
    "intent": 2,
    "emotion": len(EMOTION_CLASSES),
}


# ============================================================
# Helpers
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def compute_metrics(predictions, labels):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def compute_per_class_metrics(predictions, labels, class_names=None):
    report_str = classification_report(labels, predictions, target_names=class_names, zero_division=0)
    report_dict = classification_report(labels, predictions, target_names=class_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(labels, predictions)
    return {"report_str": report_str, "report_dict": report_dict, "confusion_matrix": cm.tolist()}


# ============================================================
# Data Loading — Single CSV, Multi-Label
# ============================================================

def load_dataset(data_path):
    """
    Load the cleaned multi-label dataset (cyberbully_train_ready.csv).
    Columns: text, cyberbullying (0/1), sarcasm (0/1), emotion (str), harm (0/1)

    Returns list of dicts: [{"text": str, "sarc": int, "intent": int, "emotion": int}, ...]
    """
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
            emotion = EMOTION_TO_IDX[emotion_label]

            samples.append({
                "text": text,
                "sarc": sarc,
                "intent": intent,
                "emotion": emotion,
            })

    if skipped > 0:
        print(f"  Skipped {skipped} rows (missing/invalid labels)")

    return samples


# ============================================================
# Dataset — Per-Task Views of the Same Data
# ============================================================

class SingleTaskDataset(Dataset):
    """Wraps multi-label samples for a single task."""

    def __init__(self, samples: List[dict], task_name: str, tokenizer, max_length=128):
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
# Trainer
# ============================================================

class MultitaskTrainer:
    def __init__(self, model, tokenizer, task_configs, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.task_configs = task_configs
        self.device = device
        self.ckpt_dir = None
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)

        self.loss_fns = {}
        for task, num_classes in task_configs.items():
            self.loss_fns[task] = (
                nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
            )

        # Equal weights for all tasks
        self.task_weights = {t: 1.0 for t in task_configs}
        print("  Task weights: EQUAL (1.0 for all tasks)")

        self.history = {
            "train_loss": [],
            "task_losses": {t: [] for t in task_configs},
            "val_metrics": {t: [] for t in task_configs},
        }
        self.start_epoch = 0
        self.best_val_f1 = 0.0
        self.best_model_state = None

    def save_checkpoint(self, epoch, checkpoint_dir, mid_epoch=False):
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "mid_epoch": mid_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "task_weights": self.task_weights,
            "best_val_f1": self.best_val_f1,
            "best_model_state": self.best_model_state,
        }
        if not mid_epoch:
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(ckpt, path)
            print(f"  Checkpoint saved: {path}")
        latest = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        torch.save(ckpt, latest)
        if mid_epoch:
            print(f"  Mid-epoch checkpoint saved")

    def load_checkpoint(self, checkpoint_dir):
        self.ckpt_dir = checkpoint_dir
        latest = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        if not os.path.exists(latest):
            return False
        print(f"\nFound checkpoint at {latest}, resuming training...")
        try:
            ckpt = torch.load(latest, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.history = ckpt["history"]
            self.best_val_f1 = ckpt.get("best_val_f1", 0.0)
            self.best_model_state = ckpt.get("best_model_state", None)
            if ckpt.get("mid_epoch", False):
                self.start_epoch = ckpt["epoch"]
                print(f"  Resumed mid-epoch {ckpt['epoch']+1}.")
            else:
                self.start_epoch = ckpt["epoch"] + 1
                print(f"  Resumed after epoch {ckpt['epoch']+1}.")
            if self.best_val_f1 > 0:
                print(f"  Best Val F1 so far: {self.best_val_f1:.4f}")
        except Exception as e:
            print(f"  WARNING: Checkpoint corrupted ({e}). Starting from scratch.")
            os.remove(latest)
            return False
        return True

    def update_best_model(self, val_metrics):
        avg_f1 = np.mean([m["f1"] for m in val_metrics.values()])
        if avg_f1 > self.best_val_f1:
            self.best_val_f1 = avg_f1
            self.best_model_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }
            print(f"  New best model: avg Val F1 = {avg_f1:.4f}")
        return avg_f1

    def load_best_model(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
            print(f"  Loaded best model (avg Val F1 = {self.best_val_f1:.4f})")
        else:
            print("  No best model saved, using final epoch model")

    def train_epoch(self, task_dataloaders, epoch):
        self.model.train()
        total_loss = 0
        task_losses = {t: 0 for t in self.task_configs}
        task_counts = {t: 0 for t in self.task_configs}
        task_iters = {t: iter(dl) for t, dl in task_dataloaders.items()}
        task_names = list(task_dataloaders.keys())
        max_batches = max(len(dl) for dl in task_dataloaders.values())
        total_steps = max_batches * len(task_names)
        step, task_idx = 0, 0

        while step < total_steps:
            task_name = task_names[task_idx % len(task_names)]
            task_idx += 1
            try:
                batch = next(task_iters[task_name])
            except StopIteration:
                task_iters[task_name] = iter(task_dataloaders[task_name])
                batch = next(task_iters[task_name])

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(input_ids, attention_mask, task_name)

            if self.task_configs[task_name] == 2:
                loss = self.loss_fns[task_name](logits.squeeze(-1), labels.float())
            else:
                loss = self.loss_fns[task_name](logits, labels)

            weighted_loss = loss * self.task_weights[task_name]
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += weighted_loss.item()
            task_losses[task_name] += loss.item()
            task_counts[task_name] += 1
            step += 1

            if step % 10 == 0:
                print(
                    f"Epoch {epoch}, Step {step}/{total_steps}, "
                    f"Task: {task_name}, Loss: {loss.item():.4f}"
                )
            if step % 1000 == 0 and self.ckpt_dir:
                self.save_checkpoint(epoch, checkpoint_dir=self.ckpt_dir, mid_epoch=True)

        avg_total = total_loss / max(step, 1)
        avg_tasks = {t: task_losses[t] / max(task_counts[t], 1) for t in self.task_configs}
        self.history["train_loss"].append(avg_total)
        for t, l in avg_tasks.items():
            self.history["task_losses"][t].append(l)
        return avg_total, avg_tasks

    def evaluate(self, task_dataloaders):
        self.model.eval()
        preds = {t: [] for t in self.task_configs}
        lbls = {t: [] for t in self.task_configs}
        with torch.no_grad():
            for task_name, dl in task_dataloaders.items():
                for batch in dl:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    logits = self.model(input_ids, attention_mask, task_name)
                    if self.task_configs[task_name] == 2:
                        predictions = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
                    else:
                        predictions = torch.argmax(logits, dim=-1)
                    preds[task_name].extend(predictions.cpu().numpy())
                    lbls[task_name].extend(batch["label"].numpy())

        task_metrics = {}
        for t in self.task_configs:
            if preds[t]:
                m = compute_metrics(preds[t], lbls[t])
                task_metrics[t] = m
                self.history["val_metrics"][t].append(m)
        return task_metrics


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  MTL-BERT Equal Weights — Single Dataset (Multi-Label)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_dir = os.path.join("results", "unified-dataset", "mtl-equal-weight-one-dataset")
    os.makedirs(results_dir, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load single multi-label dataset
    print(f"\nLoading dataset: {DATA_PATH}")
    all_samples = load_dataset(DATA_PATH)
    print(f"  Total valid samples: {len(all_samples)}")

    # Print label distributions
    from collections import Counter
    sarc_dist = Counter(s["sarc"] for s in all_samples)
    intent_dist = Counter(s["intent"] for s in all_samples)
    emotion_dist = Counter(s["emotion"] for s in all_samples)
    print(f"  Sarcasm:       No={sarc_dist[0]}, Yes={sarc_dist[1]}")
    print(f"  Cyberbullying: Nonbully={intent_dist[0]}, Bully={intent_dist[1]}")
    print(f"  Emotion:       {len(EMOTION_CLASSES)} classes")
    for idx, name in enumerate(EMOTION_CLASSES):
        print(f"    {name}: {emotion_dist[idx]}")

    # Multi-seed training with progress tracking
    progress_path = os.path.join(results_dir, "training_progress.json")
    all_seed_results = {task: [] for task in TASK_CONFIGS}
    all_seed_predictions = {task: [] for task in TASK_CONFIGS}
    all_seed_labels = {task: [] for task in TASK_CONFIGS}
    all_seed_histories = []
    completed = set()

    if os.path.exists(progress_path):
        progress = load_json(progress_path)
        if progress:
            completed = set(progress.get("completed", []))
            all_seed_results = progress.get("results", all_seed_results)
            all_seed_predictions = progress.get("predictions", all_seed_predictions)
            all_seed_labels = progress.get("labels", all_seed_labels)
            all_seed_histories = progress.get("histories", [])
            if completed:
                print(f"\nResuming: {len(completed)} seed(s) already done.")

    for seed_idx, seed in enumerate(SEEDS):
        seed_key = f"seed{seed}"
        if seed_key in completed:
            print(f"\n--- Skipping seed {seed} (already done) ---")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Seed {seed_idx+1}/{len(SEEDS)} (seed={seed})")
        print(f"{'=' * 60}")

        set_seed(seed)

        # Split: 80/10/10 — same split for ALL tasks (since it's one dataset)
        shuffled = all_samples.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        s1, s2 = int(0.8 * n), int(0.9 * n)
        train_samples = shuffled[:s1]
        val_samples = shuffled[s1:s2]
        test_samples = shuffled[s2:]

        print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

        # Create per-task DataLoaders from the SAME data split
        train_loaders, val_loaders, test_loaders = {}, {}, {}
        for task_name in TASK_CONFIGS:
            train_loaders[task_name] = DataLoader(
                SingleTaskDataset(train_samples, task_name, tokenizer, MAX_LENGTH),
                batch_size=BATCH_SIZE, shuffle=True,
            )
            val_loaders[task_name] = DataLoader(
                SingleTaskDataset(val_samples, task_name, tokenizer, MAX_LENGTH),
                batch_size=BATCH_SIZE, shuffle=False,
            )
            test_loaders[task_name] = DataLoader(
                SingleTaskDataset(test_samples, task_name, tokenizer, MAX_LENGTH),
                batch_size=BATCH_SIZE, shuffle=False,
            )

        model = MultitaskModel(MODEL_NAME, TASK_CONFIGS)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        trainer = MultitaskTrainer(model, tokenizer, TASK_CONFIGS, device)

        ckpt_dir = os.path.join("checkpoints", "unified-dataset", "mtl-equal-weight-one-dataset", seed_key)
        trainer.load_checkpoint(ckpt_dir)

        print(f"\n  Training for {NUM_EPOCHS} epochs (equal weights, single dataset)...")
        for epoch in range(trainer.start_epoch, NUM_EPOCHS):
            print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}")

            train_loss, task_losses = trainer.train_epoch(train_loaders, epoch)
            print(
                f"  Train Loss: {train_loss:.4f}  "
                f"(sarc={task_losses['sarc']:.4f}, "
                f"intent={task_losses['intent']:.4f}, "
                f"emotion={task_losses['emotion']:.4f})"
            )

            val_metrics = trainer.evaluate(val_loaders)
            for task, m in val_metrics.items():
                print(f"    {task}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")

            trainer.update_best_model(val_metrics)
            trainer.save_checkpoint(epoch, ckpt_dir)

        # Load best model for test evaluation
        trainer.load_best_model()

        print(f"\n  Test Evaluation (seed={seed}):")
        test_metrics = trainer.evaluate(test_loaders)

        model.eval()
        seed_preds = {task: [] for task in TASK_CONFIGS}
        seed_lbls = {task: [] for task in TASK_CONFIGS}
        with torch.no_grad():
            for task_name, dataloader in test_loaders.items():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    logits = model(input_ids, attention_mask, task_name)
                    if TASK_CONFIGS[task_name] == 2:
                        preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
                    else:
                        preds = torch.argmax(logits, dim=-1)
                    seed_preds[task_name].extend(preds.cpu().numpy().tolist())
                    seed_lbls[task_name].extend(batch["label"].numpy().tolist())

        for task, m in test_metrics.items():
            all_seed_results[task].append(m)
            all_seed_predictions[task].append(seed_preds[task])
            all_seed_labels[task].append(seed_lbls[task])
            print(
                f"    {task}: Acc={m['accuracy']:.4f}, P={m['precision']:.4f}, "
                f"R={m['recall']:.4f}, F1={m['f1']:.4f}"
            )

        all_seed_histories.append(trainer.history)
        completed.add(seed_key)

        with open(progress_path, "w") as f:
            json.dump(
                {
                    "completed": list(completed),
                    "results": all_seed_results,
                    "predictions": all_seed_predictions,
                    "labels": all_seed_labels,
                    "histories": all_seed_histories,
                },
                f,
                indent=2,
            )
        print(f"  Progress saved ({len(completed)}/{len(SEEDS)} seeds done)")

        model_path = os.path.join(results_dir, f"mtl_equal_one_dataset_{seed_key}.pt")
        torch.save(model.state_dict(), model_path)

    # ================================================================
    # Aggregated Results
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Aggregated Results (mean +/- std, {len(SEEDS)} seeds)")
    print(f"{'=' * 60}")

    aggregated = {}
    for task in TASK_CONFIGS:
        metrics_list = all_seed_results[task]
        agg = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = [m[metric] for m in metrics_list]
            agg[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_seed": [float(v) for v in values],
            }
        aggregated[task] = agg
        print(f"\n  {task}:")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean = agg[metric]["mean"]
            std = agg[metric]["std"]
            print(f"    {metric:>10s}: {mean:.4f} +/- {std:.4f}")

    agg_path = os.path.join(results_dir, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {agg_path}")

    # ================================================================
    # Per-class emotion analysis (best seed by F1)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Per-Class Metrics: Emotion Task")
    print(f"{'=' * 60}")

    best_idx = int(np.argmax([m["f1"] for m in all_seed_results["emotion"]]))
    best_preds = all_seed_predictions["emotion"][best_idx]
    best_labels = all_seed_labels["emotion"][best_idx]

    per_class = compute_per_class_metrics(best_preds, best_labels, EMOTION_CLASSES)
    print(f"\n  (Best seed: {SEEDS[best_idx]})\n")
    print(per_class["report_str"])

    per_class_path = os.path.join(results_dir, "emotion_per_class_metrics.json")
    per_class_save = {}
    for k, v in per_class["report_dict"].items():
        per_class_save[k] = v
    per_class_save["confusion_matrix"] = per_class["confusion_matrix"]
    with open(per_class_path, "w") as f:
        json.dump(per_class_save, f, indent=2)
    print(f"Per-class metrics saved to {per_class_path}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"\n  Dataset: {DATA_PATH}")
    print(f"  Samples: {len(all_samples)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Task weighting: EQUAL (1.0 for all tasks)")
    print(f"  Emotion classes: {len(EMOTION_CLASSES)}")
    print(f"\n  Output files in '{results_dir}/':")
    print(f"    - aggregated_results.json")
    print(f"    - emotion_per_class_metrics.json")
    print(f"    - mtl_equal_one_dataset_seed*.pt")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
