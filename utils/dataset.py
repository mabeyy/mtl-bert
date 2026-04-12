import csv
import os
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_TO_IDX = {name: i for i, name in enumerate(EMOTION_CLASSES)}


def load_unified_dataset(data_path=None):
    """
    Load the unified multi-label dataset (cyberbully_train_ready.csv).
    Columns: text, cyberbullying (0/1), sarcasm (0/1), emotion (str), harm (0/1)

    Returns list of dicts: [{"text": str, "sarc": int, "intent": int, "emotion": int}, ...]
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "cyberbully_train_ready.csv")

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


def compute_metrics(predictions, labels):
    """Compute accuracy, precision, recall, and F1 score (weighted average)."""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def compute_per_class_metrics(predictions, labels, class_names=None):
    """
    Compute per-class precision, recall, F1 and confusion matrix.
    Returns dict with classification report (str + dict) and confusion matrix.
    """
    report_str = classification_report(
        labels, predictions, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        labels, predictions, target_names=class_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(labels, predictions)
    return {
        "report_str": report_str,
        "report_dict": report_dict,
        "confusion_matrix": cm.tolist(),
    }
