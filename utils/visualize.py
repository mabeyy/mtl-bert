"""
Thesis Visualization Script

Generates all publication-quality figures from training results.
Reads from:
  - results/unified-dataset/stl-bert/                              (Single-Task BERT)
  - results/unified-dataset/mtl-equal-weight-one-dataset/          (MTL-BERT, equal weights)
  - results/unified-dataset/mtl-equal-weight-one-dataset-augmented/ (MTL-BERT + MLM augmentation)
  - results/unified-dataset/pipeline-baseline/                     (Pipeline baseline)

Figures saved to figures/ at 300 DPI.

Usage:
    python visualize.py
"""

import json
import os
import csv
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Result directories (unified dataset)
BASELINE_DIR = os.path.join(RESULTS_DIR, "unified-dataset", "stl-bert")
MTL_EQUAL_DIR = os.path.join(RESULTS_DIR, "unified-dataset", "mtl-equal-weight-one-dataset")
MTL_AUGMENTED_DIR = os.path.join(RESULTS_DIR, "unified-dataset", "mtl-equal-weight-one-dataset-augmented")
PIPELINE_DIR = os.path.join(RESULTS_DIR, "unified-dataset", "pipeline-baseline")

DPI = 300

COLORS = {
    "baseline": "#F97316",       # orange
    "mtl_equal": "#2563EB",      # blue
    "mtl_augmented": "#10B981",  # green
    "pipeline": "#8B5CF6",       # purple
    "sarc": "#8B5CF6",           # purple
    "intent": "#10B981",         # green
    "emotion": "#EF4444",        # red
    "total": "#1F2937",          # dark gray
}

MODEL_LABELS = {
    "baseline": "STL-BERT",
    "mtl_equal": "MTL-BERT",
    "mtl_augmented": "MTL-BERT (Augmented)",
    "pipeline": "Pipeline",
}

TASK_LABELS = {
    "sarc": "Sarcasm",
    "intent": "Harm",
    "emotion": "Emotion",
}

EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

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
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_metrics(results_dir):
    """
    Load test metrics from a results directory.
    Handles two formats:
      - test_metrics.json:        {"sarc": {"accuracy": 0.79, ...}, ...}
      - aggregated_results.json:  {"sarc": {"accuracy": {"mean": 0.79, "std": ...}, ...}, ...}
    Returns (flat_metrics, std_metrics) where flat is always mean values.
    """
    flat = load_json(os.path.join(results_dir, "test_metrics.json"))
    if flat is not None:
        return flat, None

    agg = load_json(os.path.join(results_dir, "aggregated_results.json"))
    if agg is not None:
        flat = {}
        stds = {}
        for task, task_data in agg.items():
            if not isinstance(task_data, dict):
                continue
            flat[task] = {}
            stds[task] = {}
            for metric, metric_data in task_data.items():
                if isinstance(metric_data, dict) and "mean" in metric_data:
                    flat[task][metric] = metric_data["mean"]
                    stds[task][metric] = metric_data.get("std", 0.0)
                else:
                    flat[task][metric] = metric_data
                    stds[task][metric] = 0.0
        return flat, stds

    return None, None


def load_training_history(results_dir):
    """Load training history from progress or history JSON."""
    history = load_json(os.path.join(results_dir, "training_history.json"))
    if history is not None:
        return history

    # Try extracting from training_progress.json (best seed)
    progress = load_json(os.path.join(results_dir, "training_progress.json"))
    if progress and "histories" in progress and progress["histories"]:
        return progress["histories"][0]

    return None


def save_figure(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 1: Dataset Class Distribution
# ============================================================

def plot_dataset_distribution():
    print("\n[1/8] Dataset Distribution...")

    distributions = {}

    unified_path = os.path.join(DATA_DIR, "cyberbully_train_ready.csv")
    if os.path.exists(unified_path):
        sarc_labels = []
        intent_labels = []
        emotion_labels = []
        with open(unified_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sarc_labels.append(int(row["sarcasm"]))
                except (ValueError, KeyError):
                    pass
                try:
                    intent_labels.append(int(row["harm"]))
                except (ValueError, KeyError):
                    pass
                try:
                    emotion_labels.append(row["emotion"])
                except KeyError:
                    pass
        if sarc_labels:
            distributions["sarc"] = Counter(sarc_labels)
        if intent_labels:
            distributions["intent"] = Counter(intent_labels)
        if emotion_labels:
            distributions["emotion"] = Counter(emotion_labels)

    if not distributions:
        print("  Skipped: no CSV data found.")
        return

    class_labels = {
        "sarc": {0: "Non-sarcastic", 1: "Sarcastic"},
        "intent": {0: "Not Harmful", 1: "Harmful"},
        "emotion": {
            "sadness": "Sadness", "joy": "Joy", "love": "Love",
            "anger": "Anger", "fear": "Fear", "surprise": "Surprise",
        },
    }

    task_colors = {
        "sarc": ["#C4B5FD", "#7C3AED"],
        "intent": ["#6EE7B7", "#059669"],
        "emotion": ["#93C5FD", "#3B82F6", "#F9A8D4", "#FCA5A5", "#FCD34D", "#A5F3FC"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (task, dist) in zip(axes, distributions.items()):
        if task == "emotion":
            sorted_keys = [k for k in EMOTION_CLASSES if k in dist]
        else:
            sorted_keys = sorted(dist.keys())
        names = [class_labels[task][k] for k in sorted_keys]
        counts = [dist[k] for k in sorted_keys]
        colors = task_colors[task][:len(sorted_keys)]

        bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(TASK_LABELS.get(task, task), fontweight="bold")
        ax.set_ylabel("Number of Samples")

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", va="bottom", fontsize=8,
            )

        if task == "emotion":
            ax.tick_params(axis="x", rotation=35)

    fig.suptitle("Dataset Class Distribution", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "dataset_distribution.png")


# ============================================================
# Figure 2: Three-Model Comparison (Accuracy & F1)
# ============================================================

def plot_model_comparison():
    print("\n[2/8] Model Comparison...")

    baseline, baseline_std = load_metrics(BASELINE_DIR)
    mtl_eq, mtl_eq_std = load_metrics(MTL_EQUAL_DIR)
    mtl_aug, mtl_aug_std = load_metrics(MTL_AUGMENTED_DIR)

    models = {}
    model_stds = {}
    if baseline:
        models["baseline"] = baseline
        model_stds["baseline"] = baseline_std
    if mtl_eq:
        models["mtl_equal"] = mtl_eq
        model_stds["mtl_equal"] = mtl_eq_std
    if mtl_aug:
        models["mtl_augmented"] = mtl_aug
        model_stds["mtl_augmented"] = mtl_aug_std

    if len(models) < 2:
        print("  Skipped: need at least 2 models with metrics.")
        return

    tasks = ["sarc", "intent", "emotion"]
    metrics_to_plot = ["accuracy", "precision", "recall"]
    metric_labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, metric in zip(axes, metrics_to_plot):
        x = np.arange(len(tasks))
        n_models = len(models)
        width = 0.7 / n_models

        for i, (model_key, metrics) in enumerate(models.items()):
            vals = [metrics.get(t, {}).get(metric, 0) for t in tasks]
            stds = model_stds[model_key]
            yerr = [stds[t][metric] for t in tasks] if stds else None
            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(
                x + offset, vals, width * 0.9,
                label=MODEL_LABELS[model_key],
                color=COLORS[model_key],
                edgecolor="white", linewidth=0.5,
                yerr=yerr, capsize=3,
                error_kw={"linewidth": 1.0} if yerr else {},
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Model Comparison: STL-BERT vs MTL-BERT vs MTL-BERT (Augmented)",
        fontweight="bold", fontsize=13, y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "model_comparison.png")


# ============================================================
# Figure 3: Training Loss Curves (MTL models side by side)
# ============================================================

def plot_training_loss_curves():
    print("\n[3/8] Training Loss Curves...")

    history_eq = load_training_history(MTL_EQUAL_DIR)

    histories = {}
    if history_eq and "train_loss" in history_eq:
        histories["MTL (Equal)"] = history_eq

    if not histories:
        print("  Skipped: no training history found.")
        return

    task_colors = {"sarc": "#8B5CF6", "intent": "#10B981", "emotion": "#EF4444"}
    n_panels = len(histories)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    for idx, (label, history) in enumerate(histories.items()):
        ax = axes[0][idx]
        n_epochs = len(history["train_loss"])
        epochs = range(1, n_epochs + 1)

        ax.plot(epochs, history["train_loss"], marker="o", linewidth=2,
                color="#1F2937", label="Total Loss", markersize=6)

        for task in ["sarc", "intent", "emotion"]:
            if task in history.get("task_losses", {}):
                ax.plot(epochs, history["task_losses"][task], marker="s",
                        linewidth=1.5, color=task_colors[task],
                        label=f"{TASK_LABELS[task]} Loss", markersize=5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xticks(list(epochs))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Loss per Epoch", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "training_loss_curves.png")


# ============================================================
# Figure 4: Validation F1 Curves (MTL models side by side)
# ============================================================

def plot_validation_f1_curves():
    print("\n[4/8] Validation F1 Curves...")

    history_eq = load_training_history(MTL_EQUAL_DIR)
    baseline, _ = load_metrics(BASELINE_DIR)

    histories = {}
    if history_eq and "val_metrics" in history_eq:
        histories["MTL (Equal)"] = history_eq

    if not histories:
        print("  Skipped: no training history with val_metrics found.")
        return

    task_colors = {"sarc": "#8B5CF6", "intent": "#10B981", "emotion": "#EF4444"}
    n_panels = len(histories)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    for idx, (label, history) in enumerate(histories.items()):
        ax = axes[0][idx]

        n_train = len(history.get("train_loss", []))
        for task in ["sarc", "intent", "emotion"]:
            if task in history["val_metrics"]:
                f1s = [m["f1"] for m in history["val_metrics"][task]]
                # If val has one more entry than train, first is pre-training baseline
                if len(f1s) == n_train + 1:
                    epochs = range(0, len(f1s))
                else:
                    epochs = range(1, len(f1s) + 1)
                ax.plot(list(epochs), f1s, marker="o", linewidth=2,
                        color=task_colors[task],
                        label=f"{TASK_LABELS[task]}", markersize=6)

            # Add STL baseline as dashed reference line
            if baseline and task in baseline:
                ax.axhline(y=baseline[task]["f1"], color=task_colors[task],
                           linestyle="--", linewidth=1, alpha=0.5,
                           label=f"{TASK_LABELS[task]} STL" if idx == 0 else "")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Validation F1 Score per Epoch (dashed = STL baseline)",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "validation_f1_curves.png")


# ============================================================
# Figure 5: Per-Metric Breakdown (all models, all tasks)
# ============================================================

def plot_per_metric_breakdown():
    print("\n[5/8] Per-Metric Breakdown...")

    baseline, _ = load_metrics(BASELINE_DIR)
    mtl_eq, _ = load_metrics(MTL_EQUAL_DIR)

    models = {}
    if baseline:
        models["baseline"] = baseline
    if mtl_eq:
        models["mtl_equal"] = mtl_eq

    if not models:
        print("  Skipped: no metrics found.")
        return

    tasks = ["sarc", "intent", "emotion"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = {"accuracy": "Acc", "precision": "Prec", "recall": "Rec", "f1": "F1"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, task in zip(axes, tasks):
        x = np.arange(len(metrics))
        n_models = len(models)
        width = 0.7 / n_models

        for i, (model_key, data) in enumerate(models.items()):
            vals = [data.get(task, {}).get(m, 0) for m in metrics]
            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(
                x + offset, vals, width * 0.9,
                label=MODEL_LABELS[model_key],
                color=COLORS[model_key],
                edgecolor="white", linewidth=0.5,
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )

        ax.set_title(TASK_LABELS[task], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[m] for m in metrics])
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.2)

        if task == "sarc":
            ax.legend(fontsize=8)
        ax.set_ylabel("Score" if task == "sarc" else "")

    fig.suptitle(
        "Per-Task Metrics: All Models",
        fontweight="bold", fontsize=14, y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "per_metric_breakdown.png")


# ============================================================
# Figure 6: Emotion Per-Class F1 (from best available model)
# ============================================================

def plot_emotion_per_class_f1():
    print("\n[6/8] Emotion Per-Class F1...")

    # Try MTL weighted first, then baseline
    per_class = None
    source = None
    for label, dir_path in [("MTL (Equal)", MTL_EQUAL_DIR),
                             ("STL Baseline", BASELINE_DIR)]:
        pc = load_json(os.path.join(dir_path, "emotion_per_class_metrics.json"))
        if pc is not None:
            per_class = pc
            source = label
            break

    if per_class is None:
        print("  Skipped: no emotion_per_class_metrics.json found.")
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
    ax.set_title(f"Emotion Classification - Per-Class Metrics ({source})", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_ylim(0, 1.1)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "emotion_per_class_f1.png")


# ============================================================
# Figure 7: Emotion Confusion Matrix
# ============================================================

def plot_emotion_confusion_matrix():
    print("\n[7/8] Emotion Confusion Matrix...")

    per_class = None
    source = None
    for label, dir_path in [("MTL (Equal)", MTL_EQUAL_DIR),
                             ("STL Baseline", BASELINE_DIR)]:
        pc = load_json(os.path.join(dir_path, "emotion_per_class_metrics.json"))
        if pc is not None and "confusion_matrix" in pc:
            per_class = pc
            source = label
            break

    if per_class is None:
        print("  Skipped: no confusion matrix data found.")
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
    )
    ax.set_title(f"Emotion - Normalized Confusion Matrix ({source})", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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

    # Also save raw (unnormalized) version
    fig2, ax2 = plt.subplots(figsize=(8, 6.5))
    im2 = ax2.imshow(cm.astype(float), interpolation="nearest", cmap="Blues")
    ax2.figure.colorbar(im2, ax=ax2, shrink=0.8)

    ax2.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted Label",
        ylabel="True Label",
    )
    ax2.set_title(f"Emotion - Confusion Matrix ({source})", fontweight="bold")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh2 = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(
                j, i, f"{int(cm[i, j])}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh2 else "black",
                fontsize=9,
            )

    fig2.tight_layout()
    save_figure(fig2, "emotion_confusion_matrix_raw.png")


# ============================================================
# Figure 8: Results Summary Table
# ============================================================

def plot_results_summary_table():
    print("\n[8/8] Results Summary Table...")

    baseline, baseline_std = load_metrics(BASELINE_DIR)
    mtl_eq, mtl_eq_std = load_metrics(MTL_EQUAL_DIR)

    models = {}
    model_stds = {}
    if baseline:
        models["baseline"] = baseline
        model_stds["baseline"] = baseline_std
    if mtl_eq:
        models["mtl_equal"] = mtl_eq
        model_stds["mtl_equal"] = mtl_eq_std

    if not models:
        print("  Skipped: no metrics found.")
        return

    tasks = ["sarc", "intent", "emotion"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    headers = ["Task", "Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    rows = []

    for task in tasks:
        task_label = TASK_LABELS[task]
        first = True
        for model_key, data in models.items():
            row_label = task_label if first else ""
            first = False
            row = [row_label, MODEL_LABELS[model_key]]
            stds = model_stds.get(model_key)
            for m in metrics:
                val = data.get(task, {}).get(m, 0)
                if stds and task in stds and m in stds[task] and stds[task][m] > 0:
                    row.append(f"{val:.4f} \u00b1 {stds[task][m]:.4f}")
                else:
                    row.append(f"{val:.4f}")
            rows.append(row)

    fig, ax = plt.subplots(figsize=(13, 2 + 0.4 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header style
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#1F2937")
        cell.set_text_props(color="white", fontweight="bold")

    # Row shading by task group
    model_count = len(models)
    for i in range(len(rows)):
        task_group = i // model_count
        for j in range(len(headers)):
            cell = table[i + 1, j]
            if task_group % 2 == 0:
                cell.set_facecolor("#F3F4F6")
            else:
                cell.set_facecolor("white")

    # Bold best F1 per task
    for task_idx, task in enumerate(tasks):
        f1_values = {}
        for model_key, data in models.items():
            f1_values[model_key] = data.get(task, {}).get("f1", 0)
        best_model = max(f1_values, key=f1_values.get)
        best_row_offset = task_idx * model_count + list(models.keys()).index(best_model)
        for j in range(len(headers)):
            table[best_row_offset + 1, j].set_text_props(fontweight="bold")

    ax.set_title(
        "Test Results: STL-BERT vs MTL (Equal) vs MTL (Weighted)",
        fontweight="bold", fontsize=13, pad=20,
    )

    fig.tight_layout()
    save_figure(fig, "results_summary_table.png")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 55)
    print("  Generating Thesis Figures")
    print("=" * 55)
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Figures dir: {FIGURES_DIR}")

    # Show what data is available
    for label, path in [("STL Baseline", BASELINE_DIR),
                        ("MTL Equal", MTL_EQUAL_DIR),
                        ("MTL Augmented", MTL_AUGMENTED_DIR),
                        ("Pipeline", PIPELINE_DIR)]:
        has_agg = os.path.exists(os.path.join(path, "aggregated_results.json"))
        has_prog = os.path.exists(os.path.join(path, "training_progress.json"))
        print(f"  {label:>16s}: results={'YES' if has_agg else 'no ':>3s}  "
              f"progress={'YES' if has_prog else 'no ':>3s}")

    plot_dataset_distribution()
    plot_model_comparison()
    plot_training_loss_curves()
    plot_validation_f1_curves()
    plot_per_metric_breakdown()
    plot_emotion_per_class_f1()
    plot_emotion_confusion_matrix()
    plot_results_summary_table()

    print(f"\n{'=' * 55}")
    print(f"  All figures saved to: {FIGURES_DIR}/")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
