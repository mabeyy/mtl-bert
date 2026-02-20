import csv
import os
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def create_sample_datasets():
    """
    Load datasets from CSV files in data/ directory.
    Returns dict mapping task name -> list of (text, label) tuples.

    Tasks:
    1. Sarcasm (sarc): 2 classes (0=non-sarcastic, 1=sarcastic)
    2. Cyberbullying (intent): 2 classes (0=not bullying, 1=bullying)
    3. Emotion (emotion): 6 classes (0=sad, 1=joy, 2=love, 3=angry, 4=fear, 5=surprise)
    """
    datasets = {}

    # Task 1: Sarcasm - CSV format: id,class,text
    sarc_path = os.path.join(DATA_DIR, "sarcasm", "sarcasm.csv")
    sarc_data = []
    with open(sarc_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 3:
                try:
                    label = int(row[1])
                    text = row[2].strip()
                    if text:
                        sarc_data.append((text, label))
                except ValueError:
                    continue
    datasets["sarc"] = sarc_data

    # Task 2: Cyberbullying - CSV format: id,class,text
    cyber_path = os.path.join(DATA_DIR, "cyberbullying", "cyberbullying.csv")
    intent_data = []
    with open(cyber_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 3:
                try:
                    label = int(row[1])
                    text = row[2].strip()
                    if text:
                        intent_data.append((text, label))
                except ValueError:
                    continue
    datasets["intent"] = intent_data

    # Task 3: Emotions - CSV format: class,text
    emo_path = os.path.join(DATA_DIR, "emotions", "emotions.csv")
    emotion_data = []
    with open(emo_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    label = int(row[0])
                    text = row[1].strip()
                    if text:
                        emotion_data.append((text, label))
                except ValueError:
                    continue
    datasets["emotion"] = emotion_data

    return datasets


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
