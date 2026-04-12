"""
Pipeline-Based Baseline for Sarcasm-Aware Cyberbullying Detection (Sec 3.5.1)
(Unified Dataset variant)

Implements a true sequential pipeline where:
  Stage 1: Train a BERT encoder for sarcasm detection
  Stage 2: Freeze the sarcasm encoder. Use its [CLS] representation + sarcasm
           probability as input to a harm classification head. The harm
           classifier has NO separate BERT encoder -- it fully depends on
           the sarcasm encoder's representations.
  Stage 3: Emotion is trained independently (same as single-task baseline)

This design ensures that upstream sarcasm errors propagate into harm predictions
through both the shared representation and the sarcasm probability signal.

Computes Cascade Error Rate (CER, Sec 3.4.1):
  CER = (harm errors caused by upstream sarcasm errors) / (total harm predictions)

Uses the unified multi-label dataset (cyberbully_train_ready.csv).
Multi-seed (3 seeds) for mean +/- std reporting.
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
from dataset import load_unified_dataset, compute_metrics, EMOTION_CLASSES
from typing import List, Dict


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Datasets
# ============================================================

class SingleTaskDataset(Dataset):
    """Single-task dataset wrapper for the unified multi-label data."""

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
            sample["text"], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(sample[self.task_name], dtype=torch.long),
        }


# ============================================================
# Models
# ============================================================

class SarcasmBERT(nn.Module):
    """
    Stage 1: BERT encoder fine-tuned for sarcasm detection.
    Also provides [CLS] embeddings for downstream use.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_emb)
        return logits

    def get_embedding_and_prediction(self, input_ids, attention_mask):
        """Return both [CLS] embedding and sarcasm probability."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_emb)
        sarc_prob = torch.sigmoid(logits.squeeze(-1))
        return cls_emb, sarc_prob


class PipelineHarmHead(nn.Module):
    """
    Stage 2: Harm classification head that operates on the FROZEN
    sarcasm encoder's output.

    Input: [CLS] embedding (768) concatenated with sarcasm probability (1) = 769
    Architecture: Linear(769, 384) -> ReLU -> Dropout -> Linear(384, 1)

    No separate BERT encoder -- fully depends on sarcasm encoder representations.
    """

    def __init__(self, input_size: int = 769, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, 1),
        )

    def forward(self, cls_emb, sarc_prob):
        combined = torch.cat([cls_emb, sarc_prob.unsqueeze(-1)], dim=-1)
        return self.head(combined)


# ============================================================
# Training helpers
# ============================================================

def train_sarcasm_model(model, train_loader, val_loader, device, num_epochs=5):
    """Train the sarcasm BERT encoder (Stage 1)."""
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_f1 = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()

            logits = model(input_ids, attention_mask).squeeze(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask).squeeze(-1)
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["label"].numpy())

        val_metrics = compute_metrics(all_preds, all_labels)
        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"    Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val_f1


def extract_pipeline_features(sarc_model, samples, tokenizer, device,
                              max_length=128, batch_size=32):
    """
    Extract [CLS] embeddings and sarcasm probabilities from the frozen
    sarcasm encoder for all samples.
    """
    sarc_model.eval()
    all_embeddings = []
    all_sarc_probs = []
    texts = [s["text"] for s in samples]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoding = tokenizer(
            batch_texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            cls_emb, sarc_prob = sarc_model.get_embedding_and_prediction(
                input_ids, attention_mask
            )

        all_embeddings.append(cls_emb.cpu())
        all_sarc_probs.extend(sarc_prob.cpu().numpy().tolist())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings, all_sarc_probs


class PipelineFeatureDataset(Dataset):
    """Dataset of pre-extracted [CLS] embeddings + sarcasm probs for harm training."""

    def __init__(self, embeddings, sarc_probs, samples):
        self.embeddings = embeddings
        self.sarc_probs = sarc_probs
        self.labels = [s["intent"] for s in samples]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "cls_emb": self.embeddings[idx],
            "sarc_prob": torch.tensor(self.sarc_probs[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_harm_head(harm_head, train_loader, val_loader, device, num_epochs=5):
    """Train the harm classification head on frozen sarcasm encoder features (Stage 2)."""
    optimizer = optim.AdamW(harm_head.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_f1 = 0.0
    best_state = None

    for epoch in range(num_epochs):
        harm_head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            cls_emb = batch["cls_emb"].to(device)
            sarc_prob = batch["sarc_prob"].to(device)
            labels = batch["label"].to(device).float()

            logits = harm_head(cls_emb, sarc_prob).squeeze(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(harm_head.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validate
        harm_head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                cls_emb = batch["cls_emb"].to(device)
                sarc_prob = batch["sarc_prob"].to(device)
                logits = harm_head(cls_emb, sarc_prob).squeeze(-1)
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["label"].numpy())

        val_metrics = compute_metrics(all_preds, all_labels)
        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"    Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in harm_head.state_dict().items()}

    return best_state, best_val_f1


def evaluate_harm_head(harm_head, dataloader, device):
    """Evaluate the harm classification head."""
    harm_head.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            cls_emb = batch["cls_emb"].to(device)
            sarc_prob = batch["sarc_prob"].to(device)
            logits = harm_head(cls_emb, sarc_prob).squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    metrics = compute_metrics(all_preds, all_labels)
    return metrics, all_preds, all_labels


def compute_cer(preds_with_sarc, preds_without_sarc, labels):
    """
    Cascade Error Rate (Sec 3.4.1).
    CER = (harm errors CAUSED by sarcasm signal) / (total predictions)

    An error is "caused" by sarcasm if:
      - the prediction is WRONG with the actual sarcasm signal, AND
      - the prediction would be RIGHT without the sarcasm signal (sarc_prob=0)
    """
    total = len(labels)
    caused_errors = 0
    for i in range(total):
        wrong_with = (preds_with_sarc[i] != labels[i])
        right_without = (preds_without_sarc[i] == labels[i])
        if wrong_with and right_without:
            caused_errors += 1
    cer = caused_errors / total if total > 0 else 0.0
    return cer


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Pipeline-Based Baseline (Unified Dataset)")
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

    results_dir = os.path.join("results", "unified-dataset", "pipeline-baseline")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nLoading unified dataset...")
    all_samples = load_unified_dataset()
    print(f"  Total samples: {len(all_samples)}")

    # --- Multi-seed training ---
    progress_path = os.path.join(results_dir, "training_progress.json")
    all_results = {"sarc": [], "intent": [], "cer": []}
    completed = set()

    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            progress = json.load(f)
        completed = set(progress.get("completed", []))
        all_results = progress.get("results", all_results)
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

        # Shared 80/10/10 split
        shuffled = all_samples.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        s1, s2 = int(0.8 * n), int(0.9 * n)
        train_samples = shuffled[:s1]
        val_samples = shuffled[s1:s2]
        test_samples = shuffled[s2:]

        print(f"  Split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

        # ============================
        # Stage 1: Train sarcasm BERT
        # ============================
        print(f"\n  [Stage 1] Training sarcasm encoder...")

        sarc_model = SarcasmBERT(MODEL_NAME).to(device)

        sarc_train_ds = SingleTaskDataset(train_samples, "sarc", tokenizer,
                                          config["max_length"])
        sarc_val_ds = SingleTaskDataset(val_samples, "sarc", tokenizer,
                                        config["max_length"])
        sarc_test_ds = SingleTaskDataset(test_samples, "sarc", tokenizer,
                                         config["max_length"])

        sarc_train_dl = DataLoader(sarc_train_ds, batch_size=config["batch_size"],
                                   shuffle=True)
        sarc_val_dl = DataLoader(sarc_val_ds, batch_size=config["batch_size"],
                                 shuffle=False)
        sarc_test_dl = DataLoader(sarc_test_ds, batch_size=config["batch_size"],
                                  shuffle=False)

        best_sarc_state, _ = train_sarcasm_model(
            sarc_model, sarc_train_dl, sarc_val_dl, device,
            num_epochs=config["num_epochs"],
        )
        sarc_model.load_state_dict(best_sarc_state)
        sarc_model.to(device)

        # Evaluate sarcasm on test
        sarc_model.eval()
        sarc_preds, sarc_labels = [], []
        with torch.no_grad():
            for batch in sarc_test_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = sarc_model(input_ids, attention_mask).squeeze(-1)
                preds = (torch.sigmoid(logits) > 0.5).long()
                sarc_preds.extend(preds.cpu().numpy())
                sarc_labels.extend(batch["label"].numpy())

        sarc_test_metrics = compute_metrics(sarc_preds, sarc_labels)
        print(f"    Sarcasm test: Acc={sarc_test_metrics['accuracy']:.4f}, "
              f"F1={sarc_test_metrics['f1']:.4f}")

        # ============================
        # Stage 2: Extract features from FROZEN sarcasm encoder,
        #          then train harm head on top
        # ============================
        print(f"\n  [Stage 2] Extracting features from frozen sarcasm encoder...")

        # Freeze sarcasm model -- no more updates
        sarc_model.eval()
        for param in sarc_model.parameters():
            param.requires_grad = False

        # Extract [CLS] embeddings + sarcasm probs for all splits
        train_embs, train_sarc_probs = extract_pipeline_features(
            sarc_model, train_samples, tokenizer, device,
            config["max_length"],
        )
        val_embs, val_sarc_probs = extract_pipeline_features(
            sarc_model, val_samples, tokenizer, device,
            config["max_length"],
        )
        test_embs, test_sarc_probs = extract_pipeline_features(
            sarc_model, test_samples, tokenizer, device,
            config["max_length"],
        )

        sarc_pos_rate = np.mean([1 if p > 0.5 else 0 for p in test_sarc_probs])
        print(f"    Sarcasm positive rate on test: {sarc_pos_rate:.2%}")

        # Create feature datasets for harm training
        harm_train_ds = PipelineFeatureDataset(train_embs, train_sarc_probs, train_samples)
        harm_val_ds = PipelineFeatureDataset(val_embs, val_sarc_probs, val_samples)
        harm_test_ds = PipelineFeatureDataset(test_embs, test_sarc_probs, test_samples)

        harm_train_dl = DataLoader(harm_train_ds, batch_size=config["batch_size"],
                                   shuffle=True)
        harm_val_dl = DataLoader(harm_val_ds, batch_size=config["batch_size"],
                                 shuffle=False)
        harm_test_dl = DataLoader(harm_test_ds, batch_size=config["batch_size"],
                                  shuffle=False)

        # Train harm head
        print(f"\n  [Stage 2] Training harm classification head...")
        harm_head = PipelineHarmHead(input_size=769).to(device)
        best_harm_state, _ = train_harm_head(
            harm_head, harm_train_dl, harm_val_dl, device,
            num_epochs=config["num_epochs"],
        )
        harm_head.load_state_dict(best_harm_state)
        harm_head.to(device)

        # ============================
        # Evaluate pipeline + CER
        # ============================
        print(f"\n  [CER Analysis] Computing Cascade Error Rate...")

        # Evaluate with actual sarcasm predictions
        harm_metrics, preds_with, harm_labels = evaluate_harm_head(
            harm_head, harm_test_dl, device
        )
        print(f"    Harm test (pipeline): Acc={harm_metrics['accuracy']:.4f}, "
              f"F1={harm_metrics['f1']:.4f}")

        # Evaluate with sarcasm zeroed out (no sarcasm signal)
        harm_test_ds_clean = PipelineFeatureDataset(
            test_embs, [0.0] * len(test_samples), test_samples
        )
        harm_test_dl_clean = DataLoader(harm_test_ds_clean,
                                        batch_size=config["batch_size"],
                                        shuffle=False)
        harm_metrics_clean, preds_without, _ = evaluate_harm_head(
            harm_head, harm_test_dl_clean, device
        )
        print(f"    Harm test (no sarc):  Acc={harm_metrics_clean['accuracy']:.4f}, "
              f"F1={harm_metrics_clean['f1']:.4f}")

        cer = compute_cer(preds_with, preds_without, harm_labels)
        print(f"    CER = {cer:.4f} ({cer:.2%} of predictions corrupted by sarcasm)")

        total = len(harm_labels)
        wrong_with = sum(1 for i in range(total) if preds_with[i] != harm_labels[i])
        wrong_without = sum(1 for i in range(total)
                           if preds_without[i] != harm_labels[i])
        print(f"    Errors with sarcasm: {wrong_with}/{total}")
        print(f"    Errors without sarcasm: {wrong_without}/{total}")

        # Save seed results
        all_results["sarc"].append(sarc_test_metrics)
        all_results["intent"].append(harm_metrics)
        all_results["cer"].append({
            "cer": cer,
            "harm_f1_with_sarc": harm_metrics["f1"],
            "harm_f1_without_sarc": harm_metrics_clean["f1"],
        })
        completed.add(seed_key)

        with open(progress_path, "w") as f:
            json.dump({"completed": list(completed), "results": all_results}, f,
                      indent=2)
        print(f"  Progress saved ({len(completed)}/{len(SEEDS)} seeds done)")

    # ================================================================
    # Aggregated results
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Pipeline Baseline: Aggregated Results ({len(SEEDS)} seeds)")
    print(f"{'=' * 60}")

    aggregated = {}
    for task in ["sarc", "intent"]:
        metrics_list = all_results[task]
        agg = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = [m[metric] for m in metrics_list]
            agg[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        aggregated[task] = agg
        print(f"\n  {task}:")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            print(f"    {metric:>10s}: {agg[metric]['mean']:.4f} "
                  f"+/- {agg[metric]['std']:.4f}")

    cer_values = [c["cer"] for c in all_results["cer"]]
    aggregated["cer"] = {
        "mean": float(np.mean(cer_values)),
        "std": float(np.std(cer_values)),
        "per_seed": all_results["cer"],
    }
    print(f"\n  Cascade Error Rate (CER):")
    print(f"    CER: {np.mean(cer_values):.4f} +/- {np.std(cer_values):.4f}")
    print(f"    (MTL CER = 0 by design, no sequential dependency)")

    aggregated["emotion_note"] = (
        "Emotion is trained independently in the pipeline (same as single-task "
        "baseline). Refer to STL baseline results for emotion metrics."
    )

    agg_path = os.path.join(results_dir, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nResults saved to {agg_path}")

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"\n  Dataset: cyberbully_train_ready.csv (unified)")
    print(f"  Pipeline: sarcasm encoder (frozen) -> harm head")
    print(f"  CER: {np.mean(cer_values):.4f} "
          f"(sarcasm errors corrupt {np.mean(cer_values):.2%} of harm predictions)")
    print(f"  MTL CER: 0.0000 (no sequential dependency by architecture)")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
