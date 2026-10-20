"""
make_workflow_figure.py -- schematic of the study design and the leakage-safe pipeline (Fig_workflow.png / .pdf in <output_folder>/figures)
Usage: python make_workflow_figure.py [output_folder]      (the numbers shown are read from model_input_v2.csv)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs_v2")
FG = os.path.join(OUT, "figures")
os.makedirs(FG, exist_ok=True)
df = pd.read_csv(os.path.join(HERE, "model_input_v2.csv"), usecols=["YEAR", "DENGUE"])
n, nd = len(df), int(df.DENGUE.sum())
tr_t, te_t = int((df.YEAR <= 2022).sum()), int((df.YEAR >= 2023).sum())
d_tr = df.DENGUE[df.YEAR <= 2022].mean() * 100
d_te = df.DENGUE[df.YEAR >= 2023].mean() * 100
n_te = int(np.ceil(n * 0.2))
n_tr = n - n_te

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.6, 7.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

BLUE, ORANGE, GREEN, RED, GREY = "#DCE9F5", "#FBE7C6", "#D8EFE6", "#F7D9CF", "#EEEEEE"
EDGE = "#4A4A4A"


def box(x, y, w, h, text, fc, fs=7.4, bold=False, ec=EDGE, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)


def panel(x, y, w, h, title, fc, ec, tc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec=ec, lw=1.3))
    ax.text(x + 2, y + h - 2.6, title, fontsize=7.8, fontweight="bold", color=tc, va="center")


def arrow(x1, y1, x2, y2, color=EDGE, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=lw, color=color, linestyle=ls))


# 1  data
box(2, 87, 96, 11, "Kolkata Municipal Corporation line-lists, 2014–2025 (24 files: dengue and malaria for each year)\n"
    f"merged and cleaned  →  139,468 records  →  complete cases: {n:,}  (dengue {nd:,} = {nd / n * 100:.1f}%;  malaria {n - nd:,})\n"
    "six fields: ward, borough, month, age, sex, place of treatment", GREY, fs=7.6)
arrow(30, 86.5, 26, 82.8)
arrow(70, 86.5, 74, 82.8)

# 2  splits
box(2, 74, 47, 8, f"Five random 80/20 splits (seeds 1–5)\ntrain {n_tr:,}  |  test {n_te:,}  |  dengue 33.0% in both", BLUE)
box(51, 74, 47, 8, f"Temporal split\ntrain 2014–2022: {tr_t:,} ({d_tr:.1f}% dengue)\ntest 2023–2025: {te_t:,} ({d_te:.1f}% dengue)", BLUE)
arrow(25, 73.4, 25, 68.6)
arrow(75, 73.4, 75, 68.6)

# 3  training panel
panel(2, 36, 96, 32, "TRAINING DATA ONLY", "#F8FBFE", "#2E74B5", "#1F4E79")
box(4.5, 49.5, 29, 13.5, "Encoding fitted on the\ntraining data only:\nward target-encoded\n(5-fold cross-fitted)", BLUE, fs=7.2)
box(35.5, 49.5, 29, 13.5, "Imbalance arm\n(training data only)\nBaseline · class weight · SMOTE\nBorderline-SMOTE · ADASYN", ORANGE, fs=7.2)
box(66.5, 49.5, 29, 13.5, "Seven classifiers, tuned once\non the baseline arm (random\nsearch, 3-fold CV, ROC-AUC)", GREEN, fs=7.2)
arrow(33.7, 56.2, 35.3, 56.2)
arrow(64.7, 56.2, 66.3, 56.2)
box(4.5, 38, 44, 9.2, "Decision threshold: 0.5, or tuned to maximise\nMacro-F1 on a 25% validation slice\nof the training data", RED, fs=7.2)
box(51.5, 38, 44, 9.2, "Field importance (SHAP / permutation) → ranked fields\n→ smallest k within 0.005 of the best\ncross-validated Macro-F1", RED, fs=7.2)
arrow(35, 49.2, 30, 47.6, lw=0.8)
arrow(65, 49.2, 70, 47.6, lw=0.8)

# 4  test panel
arrow(50, 35.5, 50, 32.6, lw=1.4, color="#C77700")
panel(2, 17.5, 96, 15.5, "HELD-OUT TEST DATA, USED ONCE", "#FFFDF7", "#C77700", "#8A5200")
box(4.5, 19.3, 91, 7.6, "AUC · Macro-F1 · recall · precision · calibration (slope, intercept) · decision curves · bootstrap 95% CIs · McNemar test\n"
    "precision at matched recall · subgroups · ward-level calibration map · temporal split by year · field ablations", ORANGE, fs=6.9)

# 5  leakage audit
panel(2, 1, 96, 15, "LEAKAGE AUDIT  (earlier workflow reproduced on today's data)", "#FFF8F6", "#B3412A", "#8E2E1B")
box(4.5, 3, 36, 8.4, "Ward encoded with all labels;\nwhole dataset oversampled\nbefore the train/test split", RED, fs=7.0)
arrow(40.8, 7.2, 43.6, 7.2, color="#B3412A")
box(44.5, 3, 51, 8.4, "Scored four ways (3 splits, 4 classifiers):\n1 correct pipeline · 2 ward leak only ·\n"
    "3 both leaks, real test cases ·\n4 both leaks, balanced half-synthetic test set", GREY, fs=6.8)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FG, f"Fig_workflow.{ext}"))
print("saved", os.path.join(FG, "Fig_workflow.png"))
