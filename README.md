# MTL-BERT: Multi-Task Learning with BERT for Cyberbullying Detection

A multi-task learning framework that leverages shared BERT representations across three auxiliary tasks — sarcasm detection, harm/intent classification, and emotion recognition — to improve cyberbullying detection through weighted late fusion.

## Architecture

```
                         Input Text
                             |
                      [BERT Encoder] (shared)
                             |
                 +-----------+-----------+
                 |           |           |
           [Sarcasm]   [Harm/Intent]  [Emotion]
            binary       binary       6-class
              |             |            |
              v             v            v
           p(yes)       p(yes)     [sad, joy, love,
            x0.20        x0.50      anger, fear, surprise]
                                       x0.30
                 \          |          /
                  +----+----+----+----+
                       |
               Weighted Feature Vector z (8-dim)
                       |
                  [Dense Layer]
                       |
              p(cyberbullying)
```

## Project Structure

```
mtl-bert/
├── models/
│   ├── mtl_bert_equal_weight_one_dataset.py          # MTL-BERT (equal weight, unified dataset)
│   ├── mtl_bert_equal_weight_one_dataset_augmented.py # MTL-BERT + contextual MLM augmentation
│   ├── stl_bert.py                                    # Single-task BERT baselines
│   └── pipeline_baseline.py                           # Sequential pipeline baseline
├── utils/
│   ├── dataset.py            # Data loading & metric computation
│   ├── visualize.py          # Publication-quality figure generation
│   └── attention_viz.py      # Attention distribution visualization
├── inference.py              # Weighted late fusion inference
├── prototype/
│   └── app.py                # Tkinter GUI for live inference
├── data/
│   └── cyberbully_train_ready.csv   # Unified multi-label dataset
├── results/                  # Training results & metrics
├── figures/                  # Generated plots
├── Colab/                    # Colab notebook experiments
├── paper/                    # Thesis document
└── requirements.txt
```

## Models

| Model | Description |
|-------|-------------|
| **MTL-BERT (Equal Weight)** | Multi-task BERT with equal loss weighting across sarcasm, harm, and emotion tasks. Trained on unified dataset with shared encoder. |
| **MTL-BERT (Augmented)** | Same as above + contextual MLM augmentation (15% masking) and class-balanced oversampling for emotion classes. |
| **STL-BERT** | Single-task baseline: one independent BERT model per task. |
| **Pipeline Baseline** | Sequential cascade where sarcasm feeds into harm classification. Tracks Cascade Error Rate (CER). |

## Setup

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch >= 2.10.0
- Transformers >= 5.1.0
- scikit-learn >= 1.8.0

## Training

Each training script supports multiple seed runs (42, 123, 456) and saves results to `results/`.

```bash
# MTL-BERT (equal weight)
python models/mtl_bert_equal_weight_one_dataset.py

# MTL-BERT with augmentation
python models/mtl_bert_equal_weight_one_dataset_augmented.py

# Single-task baselines
python models/stl_bert.py

# Pipeline baseline
python models/pipeline_baseline.py
```

## Inference

Run inference using the trained MTL-BERT model with weighted late fusion:

```bash
# Default model
python inference.py

# Specify model path
python inference.py --model results/unified-dataset/mtl-equal-weight-one-dataset/mtl_equal_one_dataset_seed456.pt

# Single text
python inference.py --text "I hope he fails"
```

### Fusion Weights

| Task | Weight | Role |
|------|--------|------|
| Harm/Intent | 0.50 | Primary indicator |
| Emotion | 0.30 | Contextual signal |
| Sarcasm | 0.20 | Modifier signal |

## Visualization

```bash
# Generate comparison figures across all models
python utils/visualize.py

# Attention distribution analysis
python utils/attention_viz.py
```

## GUI Prototype

A Tkinter-based desktop app for live cyberbullying detection:

```bash
python prototype/app.py
```

### Screenshots

| Default | Not Cyberbullying | Cyberbullying Detected |
|:-------:|:-----------------:|:----------------------:|
| ![Default](figures/screenshots/gui_default.png) | ![Not Cyberbullying](figures/screenshots/gui_not_cyberbullying.png) | ![Cyberbullying Detected](figures/screenshots/gui_cyberbullying.png) |

The GUI displays:
- **Final prediction** with confidence score
- **Task head outputs** — sarcasm, harm/intent, and emotion probabilities
- **Fusion vector z** — the 8-dim weighted feature vector used for the final decision

## Dataset

The unified dataset (`data/cyberbully_train_ready.csv`) contains multi-label annotations with the following columns:

| Column | Type | Values |
|--------|------|--------|
| `text` | string | Input text |
| `cyberbullying` | binary | 0 / 1 |
| `sarcasm` | binary | 0 / 1 |
| `harm` | binary | 0 / 1 |
| `emotion` | categorical | sadness, joy, love, anger, fear, surprise |

Split: 80% train / 10% validation / 10% test
