"""
Thesis Visualization Script

Generates publication-quality figures from MTL-BERT and Single-Task BERT results.
All figures are saved to the figures/ directory at 300 DPI.

Usage:
    python visualize.py

Figures generated:
    1. dataset_distribution.png        - Class distribution for all 3 tasks
    2. mtl_vs_stl_comparison.png       - MTL vs STL grouped bar chart
    3. training_loss_curves.png        - Per-task training loss across epochs
    4. validation_f1_curves.png        - Per-task validation F1 across epochs
    5. emotion_per_class_f1.png        - Per-class P/R/F1 for emotion task
    6. emotion_confusion_matrix.png    - Normalized confusion matrix for emotion
    7. results_summary_table.png       - Full results table for thesis
"""

import json
import os
import csv
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# Config
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline")
DATA_DIR = os.path.join(BASE_DIR, "data")

DPI = 300

# Consistent color palette
COLORS = {
    "mtl": "#2563EB",       # blue
    "stl": "#F97316",       # orange
    "sarc": "#8B5CF6",      # purple
    "intent": "#10B981",    # green
    "emotion": "#EF4444",   # red
    "total": "#1F2937",     # dark gray
}

TASK_LABELS = {
    "sarc": "Sarcasm",
    "intent": "Cyberbullying",
    "emotion": "Emotion",
}

EMOTION_CLASSES = ["sad", "joy", "love", "angry", "fear", "surprise"]

# Clean academic style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    """Load a JSON file, return None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_figure(fig, filename):
    """Save figure to figures/ directory."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 1: Dataset Class Distribution
# ============================================================

def plot_dataset_distribution():
    """Bar charts showing class distribution for all 3 tasks."""
    print("\n[1/7] Dataset Distribution...")

    # Load class counts from CSVs
    distributions = {}

    # Sarcasm: id,class,text
    sarc_path = os.path.join(DATA_DIR, "sarcasm", "sarcasm.csv")
    if os.path.exists(sarc_path):
        labels = []
        with open(sarc_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3:
                    try:
                        labels.append(int(row[1]))
                    except ValueError:
                        continue
        distributions["sarc"] = Counter(labels)

    # Cyberbullying: id,class,text
    cyber_path = os.path.join(DATA_DIR, "cyberbullying", "cyberbullying.csv")
    if os.path.exists(cyber_path):
        labels = []
        with open(cyber_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3:
                    try:
                        labels.append(int(row[1]))
                    except ValueError:
                        continue
        distributions["intent"] = Counter(labels)

    # Emotions: class,text
    emo_path = os.path.join(DATA_DIR, "emotions", "emotions.csv")
    if os.path.exists(emo_path):
        labels = []
        with open(emo_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    try:
                        labels.append(int(row[0]))
                    except ValueError:
                        continue
        distributions["emotion"] = Counter(labels)

    if not distributions:
        print("  Skipped: no CSV data found.")
        return

    class_labels = {
        "sarc": {0: "Non-sarcastic", 1: "Sarcastic"},
        "intent": {0: "Not Bullying", 1: "Bullying"},
        "emotion": {0: "Sad", 1: "Joy", 2: "Love", 3: "Angry", 4: "Fear", 5: "Surprise"},
    }

    task_colors = {
        "sarc": ["#C4B5FD", "#7C3AED"],
        "intent": ["#6EE7B7", "#059669"],
        "emotion": ["#93C5FD", "#3B82F6", "#F9A8D4", "#FCA5A5", "#FCD34D", "#A5F3FC"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (task, dist) in zip(axes, distributions.items()):
        sorted_keys = sorted(dist.keys())
        names = [class_labels[task][k] for k in sorted_keys]
        counts = [dist[k] for k in sorted_keys]
        colors = task_colors[task][:len(sorted_keys)]

        bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(TASK_LABELS.get(task, task), fontweight="bold")
        ax.set_ylabel("Number of Samples")

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", va="bottom", fontsize=8,
            )

        if task == "emotion":
            ax.tick_params(axis="x", rotation=35)

    fig.suptitle("Dataset Class Distribution", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "dataset_distribution.png")


# ============================================================
# Figure 2: MTL vs STL Comparison
# ============================================================

def plot_mtl_vs_stl_comparison():
    """Grouped bar chart comparing MTL vs STL per task."""
    print("\n[2/7] MTL vs STL Comparison...")

    mtl_metrics = load_json(os.path.join(RESULTS_DIR, "test_metrics.json"))
    stl_metrics = load_json(os.path.join(BASELINE_DIR, "aggregated_results.json"))

    if mtl_metrics is None or stl_metrics is None:
        print("  Skipped: need both results/test_metrics.json and results/baseline/aggregated_results.json")
        return

    tasks = ["sarc", "intent", "emotion"]
    metrics_to_plot = ["accuracy", "f1"]
    metric_labels = {"accuracy": "Accuracy", "f1": "F1 Score"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        x = np.arange(len(tasks))
        width = 0.3

        # MTL values
        mtl_vals = [mtl_metrics[t][metric] for t in tasks]

        # STL values (mean ± std)
        stl_means = [stl_metrics[t][metric]["mean"] for t in tasks]
        stl_stds = [stl_metrics[t][metric]["std"] for t in tasks]

        bars1 = ax.bar(x - width / 2, mtl_vals, width, label="MTL-BERT",
                       color=COLORS["mtl"], edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, stl_means, width, label="STL-BERT",
                       color=COLORS["stl"], edgecolor="white", linewidth=0.5,
                       yerr=stl_stds, capsize=4, error_kw={"linewidth": 1.2})

        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
        ax.set_ylim(0, 1.1)
        ax.legend()

        # Add value labels on bars
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for bar, std in zip(bars2, stl_stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("MTL-BERT vs Single-Task BERT", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "mtl_vs_stl_comparison.png")


# ============================================================
# Figure 3: Training Loss Curves
# ============================================================

def plot_training_loss_curves():
    """Per-task and overall training loss across epochs."""
    print("\n[3/7] Training Loss Curves...")

    history = load_json(os.path.join(RESULTS_DIR, "training_history.json"))
    if history is None:
        print("  Skipped: results/training_history.json not found.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, history["train_loss"], marker="o", linewidth=2,
            color=COLORS["total"], label="Total Loss", markersize=6)

    for task in ["sarc", "intent", "emotion"]:
        if task in history["task_losses"]:
            ax.plot(epochs, history["task_losses"][task], marker="s", linewidth=1.5,
                    color=COLORS[task], label=f"{TASK_LABELS[task]} Loss", markersize=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss per Epoch", fontweight="bold")
    ax.legend()
    ax.set_xticks(list(epochs))

    fig.tight_layout()
    save_figure(fig, "training_loss_curves.png")


# ============================================================
# Figure 4: Validation F1 Curves
# ============================================================

def plot_validation_f1_curves():
    """Per-task validation F1 score across epochs."""
    print("\n[4/7] Validation F1 Curves...")

    history = load_json(os.path.join(RESULTS_DIR, "training_history.json"))
    if history is None:
        print("  Skipped: results/training_history.json not found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for task in ["sarc", "intent", "emotion"]:
        if task in history["val_metrics"]:
            f1_scores = [m["f1"] for m in history["val_metrics"][task]]
            epochs = range(1, len(f1_scores) + 1)
            ax.plot(epochs, f1_scores, marker="o", linewidth=2,
                    color=COLORS[task], label=f"{TASK_LABELS[task]}", markersize=6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("Validation F1 Score per Epoch", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(epochs))

    fig.tight_layout()
    save_figure(fig, "validation_f1_curves.png")


# ============================================================
# Figure 5: Emotion Per-Class F1
# ============================================================

def plot_emotion_per_class_f1():
    """Grouped bar chart of per-class precision, recall, F1 for emotion."""
    print("\n[5/7] Emotion Per-Class F1...")

    per_class = load_json(os.path.join(BASELINE_DIR, "emotion_per_class_metrics.json"))
    if per_class is None:
        print("  Skipped: results/baseline/emotion_per_class_metrics.json not found.")
        return

    classes = EMOTION_CLASSES
    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        if cls in per_class:
            precisions.append(per_class[cls]["precision"])
            recalls.append(per_class[cls]["recall"])
            f1s.append(per_class[cls]["f1-score"])
        else:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, precisions, width, label="Precision", color="#3B82F6", edgecolor="white")
    ax.bar(x, recalls, width, label="Recall", color="#10B981", edgecolor="white")
    ax.bar(x + width, f1s, width, label="F1 Score", color="#EF4444", edgecolor="white")

    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Score")
    ax.set_title("Emotion Classification - Per-Class Metrics (STL-BERT)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_ylim(0, 1.1)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "emotion_per_class_f1.png")


# ============================================================
# Figure 6: Emotion Confusion Matrix
# ============================================================

def plot_emotion_confusion_matrix():
    """Normalized confusion matrix heatmap for emotion classification."""
    print("\n[6/7] Emotion Confusion Matrix...")

    per_class = load_json(os.path.join(BASELINE_DIR, "emotion_per_class_metrics.json"))
    if per_class is None or "confusion_matrix" not in per_class:
        print("  Skipped: confusion matrix data not found.")
        return

    cm = np.array(per_class["confusion_matrix"])

    # Normalize by row (true labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums * 100, 0)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Percentage (%)", fontsize=10)

    classes = [c.capitalize() for c in EMOTION_CLASSES]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Emotion Classification - Normalized Confusion Matrix",
    )
    ax.set_title("Emotion Classification - Normalized Confusion Matrix", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotations
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax.text(
                j, i, f"{val:.1f}%",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9, fontweight="bold" if i == j else "normal",
            )

    fig.tight_layout()
    save_figure(fig, "emotion_confusion_matrix.png")


# ============================================================
# Figure 7: Results Summary Table
# ============================================================

def plot_results_summary_table():
    """Render a full results table as an image for thesis insertion."""
    print("\n[7/7] Results Summary Table...")

    mtl_metrics = load_json(os.path.join(RESULTS_DIR, "test_metrics.json"))
    stl_metrics = load_json(os.path.join(BASELINE_DIR, "aggregated_results.json"))

    if mtl_metrics is None or stl_metrics is None:
        print("  Skipped: need both test_metrics.json and aggregated_results.json")
        return

    tasks = ["sarc", "intent", "emotion"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    # Build table data
    headers = ["Task", "Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    rows = []

    for task in tasks:
        task_label = TASK_LABELS[task]

        # MTL row
        mtl_row = [task_label, "MTL-BERT"]
        for m in metrics:
            mtl_row.append(f"{mtl_metrics[task][m]:.4f}")
        rows.append(mtl_row)

        # STL row
        stl_row = ["", "STL-BERT"]
        for m in metrics:
            mean = stl_metrics[task][m]["mean"]
            std = stl_metrics[task][m]["std"]
            stl_row.append(f"{mean:.4f} ± {std:.4f}")
        rows.append(stl_row)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header style
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#1F2937")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(len(rows)):
        for j in range(len(headers)):
            cell = table[i + 1, j]
            if i % 4 < 2:
                cell.set_facecolor("#F3F4F6")
            else:
                cell.set_facecolor("white")

    # Bold MTL rows
    for i in range(len(rows)):
        if rows[i][1] == "MTL-BERT":
            for j in range(len(headers)):
                table[i + 1, j].set_text_props(fontweight="bold")

    ax.set_title("MTL-BERT vs Single-Task BERT - Test Results",
                 fontweight="bold", fontsize=13, pad=20)

    fig.tight_layout()
    save_figure(fig, "results_summary_table.png")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 50)
    print("  Generating Thesis Figures")
    print("=" * 50)

    plot_dataset_distribution()
    plot_mtl_vs_stl_comparison()
    plot_training_loss_curves()
    plot_validation_f1_curves()
    plot_emotion_per_class_f1()
    plot_emotion_confusion_matrix()
    plot_results_summary_table()

    print(f"\n{'=' * 50}")
    print(f"  All figures saved to: {FIGURES_DIR}/")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
