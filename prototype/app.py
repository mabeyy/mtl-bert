"""
SA-MTL Cyberbullying Detection Prototype
Tkinter-based interface for the Sarcasm-Aware Multi-Task Learning framework.

Loads the trained MTL-BERT model and performs inference using weighted late fusion:
  1. Shared BERT encoder produces contextual embeddings
  2. Three task heads output probabilities (sarcasm, harm, emotion)
  3. Weighted fusion produces final cyberbullying prediction

Usage:
    python prototype/app.py
    python prototype/app.py --model results/unified-dataset/mtl-equal-weight-one-dataset/mtl_equal_one_dataset_seed456.pt
"""

import tkinter as tk
from tkinter import ttk
import threading
import os
import sys
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict


# ============================================================
# Config
# ============================================================

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

TASK_CONFIGS = {
    "sarc": 2,
    "intent": 2,
    "emotion": 6,
}

# Fusion weights: Config 2 (Equal) from ablation study (Section 4.5.1)
# This configuration yielded the highest cyberbullying classification F1 (0.7139)
INFERENCE_WEIGHTS = {
    "sarc": 0.33,
    "intent": 0.34,
    "emotion": 0.33,
}

EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# ── Color palette ──
BG           = "#0F172A"
BG_SURFACE   = "#1E293B"
BG_ELEVATED  = "#263548"
BG_INPUT     = "#334155"
BORDER       = "#334155"
BORDER_LIGHT = "#475569"

TEXT         = "#F8FAFC"
TEXT_SEC     = "#CBD5E1"
TEXT_DIM     = "#94A3B8"
TEXT_FAINT   = "#64748B"

ACCENT       = "#3B82F6"
ACCENT_HOVER = "#2563EB"

DANGER       = "#F87171"
DANGER_STRONG = "#EF4444"
DANGER_BG    = "#451A1A"
SAFE         = "#34D399"
SAFE_STRONG  = "#10B981"
SAFE_BG      = "#0A3D2E"
WARNING      = "#FBBF24"
PURPLE       = "#A78BFA"


# ============================================================
# Model (same architecture as training)
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
        pooled = outputs.last_hidden_state[:, 0]
        return self.task_heads[task_name](pooled)


# ============================================================
# Fusion Classifier
# ============================================================

class CyberbullyingFusionClassifier:
    def __init__(self, model, tokenizer, device, weights=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.weights = weights or INFERENCE_WEIGHTS
        self.model.eval()

        self.fusion_layer = nn.Linear(8, 2).to(device)

        with torch.no_grad():
            self.fusion_layer.weight.zero_()
            self.fusion_layer.bias.zero_()

            self.fusion_layer.weight[1, 0] = 1.0
            self.fusion_layer.weight[1, 1] = 6.0
            self.fusion_layer.weight[1, 2] = 1.0
            self.fusion_layer.weight[1, 5] = 1.5
            self.fusion_layer.weight[1, 6] = 1.5
            self.fusion_layer.bias[1] = -1.5

            self.fusion_layer.weight[0, 0] = -1.0
            self.fusion_layer.weight[0, 1] = -6.0
            self.fusion_layer.weight[0, 3] = 3.0
            self.fusion_layer.weight[0, 4] = 3.0
            self.fusion_layer.weight[0, 7] = 1.5
            self.fusion_layer.bias[0] = 1.5

    def predict(self, text):
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        task_probs = {}
        with torch.no_grad():
            for task_name, num_classes in TASK_CONFIGS.items():
                logits = self.model(input_ids, attention_mask, task_name)
                if num_classes == 2:
                    prob_yes = torch.sigmoid(logits.squeeze(-1)).item()
                    task_probs[task_name] = [prob_yes]
                else:
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
                    task_probs[task_name] = probs

        z = []
        for task_name in ["sarc", "intent", "emotion"]:
            w = self.weights[task_name]
            for p in task_probs[task_name]:
                z.append(p * w)

        z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.fusion_layer(z_tensor)

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        return {
            "p_cyberbullying": float(probs[1]),
            "p_not_cyberbullying": float(probs[0]),
            "sarc_prob": task_probs["sarc"][0],
            "harm_prob": task_probs["intent"][0],
            "emotion_probs": task_probs["emotion"],
            "emotion_label": EMOTION_CLASSES[int(max(range(6), key=lambda i: task_probs["emotion"][i]))],
            "weighted_vector": z,
        }


# ============================================================
# Tkinter Application
# ============================================================

class CyberbullyingDetectorApp:
    def __init__(self, root, classifier):
        self.root = root
        self.classifier = classifier
        self.root.title("SA-MTL Cyberbullying Detection")
        self.root.configure(bg=BG)
        self.root.minsize(860, 700)

        w, h = 880, 720
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self._build_ui()

    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=32, pady=24)

        # ── Header ──
        header = tk.Frame(outer, bg=BG)
        header.pack(fill=tk.X, pady=(0, 16))

        title_frame = tk.Frame(header, bg=BG)
        title_frame.pack(side=tk.LEFT)
        tk.Label(title_frame, text="SA-MTL", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=ACCENT).pack(side=tk.LEFT)
        tk.Label(title_frame, text="  Cyberbullying Detection", font=("Segoe UI", 20),
                 bg=BG, fg=TEXT).pack(side=tk.LEFT)

        tk.Label(header, text="Weighted Late Fusion",
                 font=("Segoe UI", 9), bg=BG, fg=TEXT_FAINT).pack(side=tk.RIGHT)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 16))

        # ── Input ──
        tk.Label(outer, text="INPUT TEXT", font=("Segoe UI", 8, "bold"),
                 bg=BG, fg=TEXT_FAINT).pack(anchor="w", pady=(0, 5))

        input_wrap = tk.Frame(outer, bg=BORDER_LIGHT, padx=1, pady=1)
        input_wrap.pack(fill=tk.X)

        self.text_input = tk.Text(
            input_wrap, height=2, font=("Segoe UI", 11),
            bg=BG_INPUT, fg=TEXT, insertbackground=TEXT,
            relief="flat", wrap=tk.WORD, padx=12, pady=8,
            selectbackground=ACCENT, selectforeground=TEXT,
        )
        self.text_input.pack(fill=tk.X)
        self.text_input.bind("<Return>", self._on_enter)

        self._placeholder_active = True
        self.text_input.insert("1.0", "Type or paste social media text here...")
        self.text_input.config(fg=TEXT_FAINT)
        self.text_input.bind("<FocusIn>", self._clear_placeholder)
        self.text_input.bind("<FocusOut>", self._restore_placeholder)

        # ── Buttons ──
        btn_row = tk.Frame(outer, bg=BG)
        btn_row.pack(fill=tk.X, pady=(10, 0))

        self.analyze_btn = tk.Button(
            btn_row, text="Analyze", font=("Segoe UI", 10, "bold"),
            bg=ACCENT, fg="white", activebackground=ACCENT_HOVER,
            activeforeground="white", relief="flat", cursor="hand2",
            padx=24, pady=7, command=self._on_analyze, border=0,
        )
        self.analyze_btn.pack(side=tk.LEFT)

        self.clear_btn = tk.Button(
            btn_row, text="Clear", font=("Segoe UI", 9),
            bg=BG_SURFACE, fg=TEXT_DIM, activebackground=BORDER,
            activeforeground=TEXT, relief="flat", cursor="hand2",
            padx=16, pady=7, command=self._on_clear, border=0,
        )
        self.clear_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="")
        tk.Label(btn_row, textvariable=self.status_var, font=("Segoe UI", 9),
                 bg=BG, fg=TEXT_FAINT).pack(side=tk.LEFT, padx=(12, 0))

        tk.Frame(outer, bg=BORDER, height=1).pack(fill=tk.X, pady=(14, 0))

        # ── Scrollable results (hidden scrollbar — mouse wheel only) ──
        self.results_canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, bd=0)

        self.results_frame = tk.Frame(self.results_canvas, bg=BG)
        self.results_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )

        self.results_canvas.create_window((0, 0), window=self.results_frame,
                                          anchor="nw", tags="rf")

        self.results_canvas.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.results_canvas.bind("<Configure>",
            lambda e: self.results_canvas.itemconfig("rf", width=e.width))
        self.results_canvas.bind_all("<MouseWheel>",
            lambda e: self.results_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self._show_placeholder_results()

    # ── placeholder ──

    def _clear_placeholder(self, event=None):
        if self._placeholder_active:
            self.text_input.delete("1.0", tk.END)
            self.text_input.config(fg=TEXT)
            self._placeholder_active = False

    def _restore_placeholder(self, event=None):
        if not self.text_input.get("1.0", tk.END).strip():
            self._placeholder_active = True
            self.text_input.insert("1.0", "Type or paste social media text here...")
            self.text_input.config(fg=TEXT_FAINT)

    def _show_placeholder_results(self):
        for w in self.results_frame.winfo_children():
            w.destroy()
        tk.Label(self.results_frame, text="Results will appear here after analysis.",
                 font=("Segoe UI", 10), bg=BG, fg=TEXT_FAINT).pack(pady=40)

    # ── actions ──

    def _on_enter(self, event):
        if not event.state & 0x1:
            self._on_analyze()
            return "break"

    def _on_clear(self):
        self._placeholder_active = False
        self.text_input.delete("1.0", tk.END)
        self._restore_placeholder()
        self._show_placeholder_results()
        self.status_var.set("")

    def _on_analyze(self):
        if self._placeholder_active:
            return
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            return

        self.analyze_btn.config(state=tk.DISABLED, text="Analyzing...")
        self.status_var.set("Running inference...")

        def run():
            try:
                result = self.classifier.predict(text)
                self.root.after(0, lambda: self._display_results(text, result))
            except Exception as e:
                self.root.after(0, lambda: self._show_error(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _show_error(self, msg):
        self.analyze_btn.config(state=tk.NORMAL, text="Analyze")
        self.status_var.set(f"Error: {msg}")

    # ── render results ──

    def _display_results(self, text, result):
        self.analyze_btn.config(state=tk.NORMAL, text="Analyze")
        self.status_var.set("Inference complete")

        for w in self.results_frame.winfo_children():
            w.destroy()

        p_cyber = result["p_cyberbullying"]
        is_cyber = p_cyber > 0.5
        confidence = p_cyber if is_cyber else (1 - p_cyber)

        # ── 1. Verdict banner ──
        verdict_bg = DANGER_BG if is_cyber else SAFE_BG
        verdict_fg = DANGER if is_cyber else SAFE
        verdict_text = "CYBERBULLYING DETECTED" if is_cyber else "NOT CYBERBULLYING"

        banner = tk.Frame(self.results_frame, bg=verdict_bg, padx=20, pady=14)
        banner.pack(fill=tk.X, pady=(6, 10))

        left = tk.Frame(banner, bg=verdict_bg)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(left, text=verdict_text, font=("Segoe UI", 14, "bold"),
                 bg=verdict_bg, fg=verdict_fg).pack(anchor="w")
        # Inline probability summary
        p_line = f"p(cyberbullying) = {result['p_cyberbullying']:.4f}    p(not) = {result['p_not_cyberbullying']:.4f}"
        tk.Label(left, text=p_line, font=("Consolas", 9),
                 bg=verdict_bg, fg=TEXT_DIM).pack(anchor="w", pady=(4, 0))

        right = tk.Frame(banner, bg=verdict_bg)
        right.pack(side=tk.RIGHT, padx=(12, 0))
        tk.Label(right, text=f"{confidence:.1%}", font=("Segoe UI", 24, "bold"),
                 bg=verdict_bg, fg=verdict_fg).pack()
        tk.Label(right, text="confidence", font=("Segoe UI", 8),
                 bg=verdict_bg, fg=TEXT_DIM).pack()

        # ── 2. Task heads — compact two-column layout ──
        heads_frame = tk.Frame(self.results_frame, bg=BG)
        heads_frame.pack(fill=tk.X, pady=(0, 10))
        heads_frame.columnconfigure(0, weight=1, uniform="col")
        heads_frame.columnconfigure(1, weight=1, uniform="col")

        # Left column: Sarcasm + Harm stacked (grid to match right column height)
        left_col = tk.Frame(heads_frame, bg=BG)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_col.rowconfigure(0, weight=1, uniform="taskrow")
        left_col.rowconfigure(1, weight=1, uniform="taskrow")
        left_col.columnconfigure(0, weight=1)

        # Sarcasm card
        sarc_val = result["sarc_prob"]
        sc = tk.Frame(left_col, bg=BG_SURFACE, padx=16, pady=12)
        sc.grid(row=0, column=0, sticky="nsew", pady=(0, 3))

        sc_top = tk.Frame(sc, bg=BG_SURFACE)
        sc_top.pack(fill=tk.X)
        tk.Label(sc_top, text="SARCASM", font=("Segoe UI", 8, "bold"),
                 bg=BG_SURFACE, fg=TEXT_FAINT).pack(side=tk.LEFT)
        tk.Label(sc_top, text=f"w = 0.33", font=("Segoe UI", 8),
                 bg=BG_SURFACE, fg=TEXT_FAINT).pack(side=tk.RIGHT)

        sc_body = tk.Frame(sc, bg=BG_SURFACE)
        sc_body.pack(fill=tk.X, pady=(6, 0))
        sarc_color = WARNING if sarc_val > 0.5 else SAFE
        sarc_label = "Sarcastic" if sarc_val > 0.5 else "Not Sarcastic"
        tk.Label(sc_body, text=sarc_label, font=("Segoe UI", 13, "bold"),
                 bg=BG_SURFACE, fg=sarc_color).pack(side=tk.LEFT)
        tk.Label(sc_body, text=f"{sarc_val:.4f}", font=("Consolas", 12),
                 bg=BG_SURFACE, fg=TEXT_SEC).pack(side=tk.RIGHT)

        # Harm card
        harm_val = result["harm_prob"]
        hc = tk.Frame(left_col, bg=BG_SURFACE, padx=16, pady=12)
        hc.grid(row=1, column=0, sticky="nsew", pady=(3, 0))

        hc_top = tk.Frame(hc, bg=BG_SURFACE)
        hc_top.pack(fill=tk.X)
        tk.Label(hc_top, text="HARMFUL INTENT", font=("Segoe UI", 8, "bold"),
                 bg=BG_SURFACE, fg=TEXT_FAINT).pack(side=tk.LEFT)
        tk.Label(hc_top, text=f"w = 0.34", font=("Segoe UI", 8),
                 bg=BG_SURFACE, fg=TEXT_FAINT).pack(side=tk.RIGHT)

        hc_body = tk.Frame(hc, bg=BG_SURFACE)
        hc_body.pack(fill=tk.X, pady=(6, 0))
        harm_color = DANGER_STRONG if harm_val > 0.5 else SAFE_STRONG
        harm_label = "Harmful" if harm_val > 0.5 else "Not Harmful"
        tk.Label(hc_body, text=harm_label, font=("Segoe UI", 13, "bold"),
                 bg=BG_SURFACE, fg=harm_color).pack(side=tk.LEFT)
        tk.Label(hc_body, text=f"{harm_val:.4f}", font=("Consolas", 12),
                 bg=BG_SURFACE, fg=TEXT_SEC).pack(side=tk.RIGHT)

        # Right column: Emotion card
        emo_probs = result["emotion_probs"]
        top_idx = int(max(range(6), key=lambda i: emo_probs[i]))

        ec = tk.Frame(heads_frame, bg=BG_SURFACE, padx=16, pady=12)
        ec.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        ec_top = tk.Frame(ec, bg=BG_SURFACE)
        ec_top.pack(fill=tk.X)
        tk.Label(ec_top, text="EMOTIONAL TONE", font=("Segoe UI", 8, "bold"),
                 bg=BG_SURFACE, fg=TEXT_FAINT).pack(side=tk.LEFT)
        tk.Label(ec_top, text=f"w = 0.33", font=("Segoe UI", 8),
                 bg=BG_SURFACE, fg=TEXT_FAINT).pack(side=tk.RIGHT)

        tk.Label(ec, text=EMOTION_CLASSES[top_idx].capitalize(),
                 font=("Segoe UI", 13, "bold"), bg=BG_SURFACE,
                 fg=PURPLE).pack(anchor="w", pady=(6, 8))

        for i, (cls, p) in enumerate(zip(EMOTION_CLASSES, emo_probs)):
            erow = tk.Frame(ec, bg=BG_SURFACE)
            erow.pack(fill=tk.X, pady=1)

            is_top = (i == top_idx)
            fg = TEXT if is_top else TEXT_FAINT
            wt = "bold" if is_top else "normal"

            tk.Label(erow, text=cls.capitalize(), font=("Segoe UI", 9, wt),
                     bg=BG_SURFACE, fg=fg, width=9, anchor="w").pack(side=tk.LEFT)

            track = tk.Frame(erow, bg=BORDER, height=8)
            track.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
            track.pack_propagate(False)
            fill_color = PURPLE if is_top else TEXT_FAINT
            fill = tk.Frame(track, bg=fill_color)
            fill.place(relx=0, rely=0, relwidth=max(p, 0.01), relheight=1.0)

            tk.Label(erow, text=f".{int(p*1000):03d}", font=("Consolas", 9, wt),
                     bg=BG_SURFACE, fg=fg, width=5, anchor="e").pack(side=tk.LEFT)

        # ── 3. Weighted vector z — grouped by task ──
        zcard = tk.Frame(self.results_frame, bg=BG_SURFACE, padx=16, pady=10)
        zcard.pack(fill=tk.X, pady=(0, 6))

        tk.Label(zcard, text="FUSION VECTOR z", font=("Segoe UI", 8, "bold"),
                 bg=BG_SURFACE, fg=TEXT_FAINT).pack(anchor="w", pady=(0, 8))

        z = result["weighted_vector"]

        groups = [
            ("SARCASM",  WARNING,       ["sarc"],    z[0:1]),
            ("HARM",     DANGER_STRONG,  ["harm"],    z[1:2]),
            ("EMOTION",  PURPLE,         ["sad", "joy", "love", "anger", "fear", "surp"], z[2:8]),
        ]

        zrow = tk.Frame(zcard, bg=BG_SURFACE)
        zrow.pack(fill=tk.X)
        zrow.columnconfigure(0, weight=1, uniform="zg")
        zrow.columnconfigure(1, weight=1, uniform="zg")
        zrow.columnconfigure(2, weight=6, uniform="zg")

        for col, (group_name, color, labels, vals) in enumerate(groups):
            gframe = tk.Frame(zrow, bg=BG_ELEVATED, padx=8, pady=6)
            gframe.grid(row=0, column=col, sticky="nsew",
                        padx=(0 if col == 0 else 3, 0))

            # Group header with colored accent
            gh = tk.Frame(gframe, bg=BG_ELEVATED)
            gh.pack(fill=tk.X, pady=(0, 5))
            tk.Frame(gh, bg=color, width=3, height=10).pack(side=tk.LEFT, padx=(0, 6))
            tk.Label(gh, text=group_name, font=("Segoe UI", 7, "bold"),
                     bg=BG_ELEVATED, fg=color).pack(side=tk.LEFT)
            wt = INFERENCE_WEIGHTS[["sarc", "intent", "emotion"][col]]
            tk.Label(gh, text=f"x{wt}", font=("Segoe UI", 7),
                     bg=BG_ELEVATED, fg=TEXT_FAINT).pack(side=tk.RIGHT)

            # Values row
            vrow = tk.Frame(gframe, bg=BG_ELEVATED)
            vrow.pack(fill=tk.X)
            for j, (label, val) in enumerate(zip(labels, vals)):
                vcell = tk.Frame(vrow, bg=BG_ELEVATED)
                vcell.pack(side=tk.LEFT, expand=True, fill=tk.X)
                tk.Label(vcell, text=f"{val:.3f}", font=("Consolas", 10, "bold"),
                         bg=BG_ELEVATED, fg=TEXT).pack()
                tk.Label(vcell, text=label, font=("Segoe UI", 7),
                         bg=BG_ELEVATED, fg=TEXT_FAINT).pack()

    # ── helpers ──

    def _card(self, parent):
        card = tk.Frame(parent, bg=BG_SURFACE, padx=16, pady=12)
        card.pack(fill=tk.X, pady=(0, 8))
        return card


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SA-MTL Cyberbullying Detection Prototype")
    parser.add_argument("--model", type=str,
                        default="results/unified-dataset/mtl-equal-weight-one-dataset/mtl_equal_one_dataset_seed456.pt",
                        help="Path to trained MTL model .pt file")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    print("=" * 50)
    print("  SA-MTL Cyberbullying Detection Prototype")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model: {model_path}")
    model = MultitaskModel(MODEL_NAME, TASK_CONFIGS)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print("  Model loaded successfully.")
    else:
        print(f"  WARNING: {model_path} not found. Using untrained model.")
    model.to(device)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    classifier = CyberbullyingFusionClassifier(model, tokenizer, device)
    print("Ready. Launching interface...\n")

    root = tk.Tk()
    app = CyberbullyingDetectorApp(root, classifier)
    root.mainloop()


if __name__ == "__main__":
    main()
