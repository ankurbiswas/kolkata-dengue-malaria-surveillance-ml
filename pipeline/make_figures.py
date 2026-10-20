"""
make_figures.py -- publication figures for the model results, from the tables written by analyze_results.py and extra_analysis.py

Usage:   python make_figures.py [output_folder]       (default: outputs_v2)       writes  <output_folder>/figures/*.png (300 dpi) and *.pdf
Figures  (numbering is provisional):
  Fig_importance      RQ1  SHAP share of each field (4 classifiers) and permutation importance (all classifiers)
  Fig_parsimony       RQ2  Macro-F1 against the number of fields kept
  Fig_oversampling    RQ3  precision-recall curve of the baseline with every arm's operating point; matched-recall precision gain
  Fig_calibration     reliability curves of every arm; calibration-in-the-large by classifier and arm
  Fig_leakage         leakage audit: accuracy under the four scenarios
  Fig_ablation        change in AUC when a field is removed / only a family of fields is used
  Fig_decision_curve  net benefit of the baseline and oversampling arms
  Fig_temporal        temporal split by test year
Figures whose input table is missing are skipped.
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs_v2")
AN = os.path.join(OUT, "analysis")
FG = os.path.join(OUT, "figures")
DATA = os.environ.get("DATA") or os.path.join(HERE, "model_input_v2.csv")
os.makedirs(FG, exist_ok=True)

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
FIELDS = ["WARD", "AGE", "MONTH", "SEX", "PLACE", "BOROUGH"]
FIELD_LABEL = {"WARD": "Ward", "AGE": "Age", "MONTH": "Month", "SEX": "Sex", "PLACE": "Place of treatment", "BOROUGH": "Borough"}
FIELD_COL = {"WARD": "#0072B2", "AGE": "#E69F00", "MONTH": "#009E73", "SEX": "#CC79A7", "PLACE": "#D55E00", "BOROUGH": "#56B4E9"}
ARMS = ["Baseline", "ClassWeight", "SMOTE", "BorderlineSMOTE", "ADASYN"]
ARM_LABEL = {"Baseline": "Baseline", "ClassWeight": "Class weight", "SMOTE": "SMOTE", "BorderlineSMOTE": "Borderline-SMOTE", "ADASYN": "ADASYN"}
ARM_COL = {"Baseline": "#222222", "ClassWeight": "#E69F00", "SMOTE": "#0072B2", "BorderlineSMOTE": "#009E73", "ADASYN": "#D55E00"}
CLFS = ["DecisionTree", "RandomForest", "XGBoost", "LogisticRegression", "KNN", "MLP", "GaussianNB"]
CLF_LABEL = {"DecisionTree": "Decision Tree", "RandomForest": "Random Forest", "XGBoost": "XGBoost", "LogisticRegression": "Logistic Regression",
             "KNN": "k-NN", "MLP": "MLP", "GaussianNB": "Gaussian NB"}
SCEN = ["CORRECT", "WARD_LEAK", "ORIGINAL_REAL_CASES", "ORIGINAL_AS_REPORTED"]
SCEN_LABEL = {"CORRECT": "Correct pipeline", "WARD_LEAK": "Ward encoded with all labels", "ORIGINAL_REAL_CASES": "Both leaks, real test cases",
              "ORIGINAL_AS_REPORTED": "Both leaks, balanced half-synthetic test set"}
SCEN_COL = {"CORRECT": "#0072B2", "WARD_LEAK": "#E69F00", "ORIGINAL_REAL_CASES": "#D55E00", "ORIGINAL_AS_REPORTED": "#9A9A9A"}


def table(name):
    p = os.path.join(AN, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FG, f"{name}.{ext}"))
    plt.close(fig)
    print("saved", name, flush=True)


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------ importance
def fig_importance():
    T = table("T5_importance.csv")
    if T is None:
        return
    clfs = [c for c in CLFS if c in set(T.classifier)]
    shap_clfs = [c for c in clfs if T[(T.classifier == c) & T.shap_share_pct_mean.notna()].shape[0]]
    fig = plt.figure(figsize=(7.4, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    left = np.zeros(len(shap_clfs))
    for f in FIELDS:
        v = np.array([float(T[(T.classifier == c) & (T.field == f)].shap_share_pct_mean.iloc[0]) for c in shap_clfs])
        ax.barh(range(len(shap_clfs)), v, left=left, color=FIELD_COL[f], edgecolor="white", linewidth=0.6, label=FIELD_LABEL[f])
        for i, (a, b) in enumerate(zip(left, v)):
            if b >= 6:
                ax.text(a + b / 2, i, f"{b:.0f}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        left += v
    ax.set_yticks(range(len(shap_clfs)))
    ax.set_yticklabels([CLF_LABEL[c] for c in shap_clfs])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total mean |SHAP value| (%)")
    ax.set_title("SHAP share of each field")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=7.5, columnspacing=1.0, handlelength=1.2)
    panel(ax, "A")
    ax2 = fig.add_subplot(gs[1])
    cols = FIELDS + ["SPATIAL(WARD+BOROUGH)"]
    M = np.array([[float(T[(T.classifier == c) & (T.field == f)].perm_auc_drop_mean.iloc[0]) for f in cols] for c in clfs])
    im = ax2.imshow(M, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.12, np.nanmax(M[:, :6])))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax2.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7, color="black" if M[i, j] < 0.6 * np.nanmax(M[:, :6]) else "white")
    ax2.set_xticks(range(len(cols)))
    ax2.set_xticklabels([FIELD_LABEL.get(c, "Ward +\nBorough") for c in cols], rotation=45, ha="right")
    ax2.set_yticks(range(len(clfs)))
    ax2.set_yticklabels([CLF_LABEL[c] for c in clfs])
    ax2.spines[:].set_visible(False)
    ax2.set_title("Permutation importance (fall in AUC)")
    panel(ax2, "B")
    save(fig, "Fig_importance")


# ------------------------------------------------------------------ parsimony
def fig_parsimony():
    T = table("T4a_sweep_curves.csv")
    S = table("T4b_parsimony_selected.csv")
    if T is None:
        return
    clfs = [c for c in CLFS if c in set(T.classifier)]
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.1), sharex=True)
    axes = axes.ravel()
    ymin = T.test_macro_f1_mean.min() - 0.03
    ymax = T.test_macro_f1_mean.max() + 0.015
    for ax, c in zip(axes, clfs):
        d = T[T.classifier == c].sort_values("k")
        ax.plot(d.k, d.test_macro_f1_mean, "-o", color="#0072B2", ms=3.5, lw=1.4, label="test")
        if d.test_macro_f1_sd.notna().any():
            ax.fill_between(d.k, d.test_macro_f1_mean - d.test_macro_f1_sd.fillna(0), d.test_macro_f1_mean + d.test_macro_f1_sd.fillna(0), color="#0072B2", alpha=0.18, lw=0)
        ax.plot(d.k, d.cv_macro_f1_mean, "--s", color="#888888", ms=3, lw=1, label="training CV")
        if S is not None and c in set(S.classifier):
            k = int(S[S.classifier == c].k_most_common.iloc[0])
            ax.axvline(k, color="#D55E00", lw=1, ls=":")
            ax.text(k + 0.08, ymin + 0.004, f"k = {k}", color="#D55E00", fontsize=7.5)
        ax.set_title(CLF_LABEL[c], fontsize=8.5)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(range(1, 7))
    for ax in axes[len(clfs):-1]:
        ax.axis("off")
    for ax in axes[4:]:
        ax.set_xlabel("Fields kept (k)")
    for ax in axes[::4]:
        ax.set_ylabel("Macro-F1")
    h = [Line2D([0], [0], color="#0072B2", marker="o", ms=3.5, label="Test (mean ± SD, 5 splits)"),
         Line2D([0], [0], color="#888888", ls="--", marker="s", ms=3, label="Training cross-validation"),
         Line2D([0], [0], color="#D55E00", ls=":", label="Selected k (smallest within 0.005 of best CV)")]
    axes[-1].axis("off")
    axes[-1].legend(handles=h, loc="center", frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "Fig_parsimony")


# ------------------------------------------------------------------ oversampling vs threshold shift
def load_preds(split, clf):
    pr = pd.read_csv(os.path.join(OUT, "preds", f"{split}__{clf}.csv.gz"))
    meta = json.load(open(os.path.join(OUT, "meta", f"{split}__{clf}.json"), encoding="utf-8"))
    df = pd.read_csv(DATA).set_index("ROW_ID")
    return pr, meta, df.loc[pr["ROW_ID"].to_numpy(), "DENGUE"].to_numpy()


def fig_oversampling():
    T3 = table("T3_matched_recall.csv")
    fig = plt.figure(figsize=(7.6, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    split, clf = "seed1", "XGBoost"
    if os.path.exists(os.path.join(OUT, "preds", f"{split}__{clf}.csv.gz")):
        pr, meta, y = load_preds(split, clf)
        pb = pr["p_Baseline"].to_numpy(float)
        prec, rec, _ = precision_recall_curve(y, pb)
        ax.plot(rec, prec, color="#222222", lw=1.6, label="Baseline, all thresholds", zorder=1)
        for arm in ARMS:
            if "p_" + arm not in pr.columns:
                continue
            p = pr["p_" + arm].to_numpy(float)
            for mode, thr, filled in (("0.5", 0.5, True), ("tuned", meta["thresholds"][arm], False)):
                pred = p >= thr
                tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
                r, pcs = tp / (tp + fn), tp / max(tp + fp, 1)
                ax.scatter([r], [pcs], s=46 if filled else 52, facecolor=ARM_COL[arm] if filled else "white", edgecolor=ARM_COL[arm], linewidth=1.4, zorder=3)
        ax.set_xlabel("Recall (sensitivity) for dengue")
        ax.set_ylabel("Precision for dengue")
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.55, 1.0)
        ax.set_title(f"{CLF_LABEL[clf]}, test split {split[-1]}")
        h = [Line2D([0], [0], color="#222222", lw=1.6, label="Baseline curve")] + [Line2D([0], [0], marker="o", ls="", color=ARM_COL[a], label=ARM_LABEL[a], ms=5) for a in ARMS]
        h += [Line2D([0], [0], marker="o", ls="", markerfacecolor="black", markeredgecolor="black", ms=5, label="threshold 0.5 (filled)"),
              Line2D([0], [0], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", ms=5, label="tuned threshold (open)")]
        ax.legend(handles=h, frameon=False, fontsize=7, loc="lower left")
    panel(ax, "A")
    ax2 = fig.add_subplot(gs[1])
    if T3 is not None and len(T3):
        clfs = [c for c in CLFS if c in set(T3.classifier)]
        arms = [a for a in ARMS if a != "Baseline"]
        w = 0.19
        vals = []
        for j, a in enumerate(arms):
            d = T3[T3.arm == a].set_index("classifier")
            xs = [i + (j - 1.5) * w for i, c in enumerate(clfs) if c in d.index]
            v = [d.loc[c, "precision_gain_vs_threshold_shift_mean"] for c in clfs if c in d.index]
            s = [d.loc[c, "precision_gain_vs_threshold_shift_sd"] if "precision_gain_vs_threshold_shift_sd" in d.columns else 0 for c in clfs if c in d.index]
            vals += [(x - np.nan_to_num(e), x + np.nan_to_num(e)) for x, e in zip(v, s)]
            ax2.errorbar(xs, v, yerr=np.nan_to_num(s), fmt="o", color=ARM_COL[a], ms=4.5, capsize=2, lw=1, label=ARM_LABEL[a])
        ax2.axhline(0, color="#444444", lw=0.9)
        ax2.axhspan(-0.01, 0.01, color="#999999", alpha=0.15, lw=0)
        lo_v, hi_v = min([a for a, _ in vals] + [-0.012]), max([b for _, b in vals] + [0.012])
        ax2.set_ylim(lo_v - 0.004, hi_v + 0.012)
        ax2.set_xticks(range(len(clfs)))
        ax2.set_xticklabels([CLF_LABEL[c] for c in clfs], rotation=35, ha="right")
        ax2.set_ylabel("Precision gain over the baseline\nmoved to the same recall")
        ax2.set_title("Does the method beat a threshold shift?")
        ax2.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", columnspacing=0.9, handletextpad=0.2)
    panel(ax2, "B")
    fig.subplots_adjust(bottom=0.27)
    save(fig, "Fig_oversampling")


# ------------------------------------------------------------------ calibration
def fig_calibration():
    B = table("T9_calibration_bins.csv")
    C = table("T11_calibration.csv")
    if B is None:
        return
    fig = plt.figure(figsize=(7.6, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, ls="--")
    clf = "XGBoost"
    for arm in ARMS:
        d = B[(B.classifier == clf) & (B.arm == arm)]
        if len(d):
            ax.plot(d.mean_predicted, d.observed, "-o", color=ARM_COL[arm], ms=3.5, lw=1.3, label=ARM_LABEL[arm])
    ax.set_xlabel("Mean predicted probability of dengue (decile bins)")
    ax.set_ylabel("Observed share of dengue")
    ax.set_title(f"Reliability curves, {CLF_LABEL[clf]}")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "A")
    ax2 = fig.add_subplot(gs[1])
    if C is not None:
        R = C[C.split.str.startswith("seed")].groupby(["classifier", "arm"])[["citl"]].agg(["mean", "std"])
        clfs = [c for c in CLFS if c in set(C.classifier)]
        w = 0.16
        for j, a in enumerate(ARMS):
            xs, v, s = [], [], []
            for i, c in enumerate(clfs):
                if (c, a) in R.index:
                    xs.append(i + (j - 2) * w)
                    v.append(R.loc[(c, a), ("citl", "mean")])
                    s.append(R.loc[(c, a), ("citl", "std")])
            ax2.errorbar(xs, v, yerr=np.nan_to_num(s), fmt="o", color=ARM_COL[a], ms=4.2, capsize=2, lw=1, label=ARM_LABEL[a])
        ax2.axhline(0, color="#444444", lw=0.9)
        ax2.set_xticks(range(len(clfs)))
        ax2.set_xticklabels([CLF_LABEL[c] for c in clfs], rotation=35, ha="right")
        ax2.set_ylabel("Calibration-in-the-large (log-odds)\nnegative = over-predicts dengue")
        ax2.set_title("Calibration-in-the-large")
    panel(ax2, "B")
    fig.subplots_adjust(bottom=0.27)
    save(fig, "Fig_calibration")


# ------------------------------------------------------------------ leakage audit
def fig_leakage():
    A = table("leakage_audit_summary.csv")
    if A is None:
        return
    clfs = [c for c in CLFS if c in set(A.classifier)]
    arms = ["Baseline", "SMOTE", "BorderlineSMOTE", "ADASYN"]
    fig, axes = plt.subplots(1, len(clfs), figsize=(1.9 * len(clfs) + 0.6, 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.2
    lo = A.accuracy_mean.min() - 0.03
    for ax, c in zip(axes, clfs):
        for j, s in enumerate(SCEN):
            vals = []
            for a in arms:
                x = A[(A.classifier == c) & (A.arm == a) & (A.scenario == s)]
                vals.append(float(x.accuracy_mean.iloc[0]) if len(x) else np.nan)
            ax.bar(np.arange(len(arms)) + (j - 1.5) * w, vals, width=w, color=SCEN_COL[s], label=SCEN_LABEL[s])
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(["Base-\nline", "SMOTE", "B-\nSMOTE", "ADA-\nSYN"], fontsize=7)
        ax.set_title(CLF_LABEL[c], fontsize=8.5)
        ax.set_ylim(lo, min(1.0, A.accuracy_mean.max() + 0.03))
    axes[0].set_ylabel("Accuracy")
    h = [matplotlib.patches.Patch(color=SCEN_COL[s], label=SCEN_LABEL[s]) for s in SCEN]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "Fig_leakage")


# ------------------------------------------------------------------ ablation
def fig_ablation():
    T = table("T10c_ablation_difference_ci.csv")
    if T is None or T.empty:
        return
    order = ["drop_WARD", "drop_MONTH", "drop_AGE", "drop_SEX", "drop_PLACE", "drop_BOROUGH", "only_WARD+BOROUGH", "only_WARD+MONTH",
             "only_MONTH+AGE+SEX+PLACE", "only_MONTH+AGE+SEX"]
    label = {"drop_WARD": "Without ward", "drop_MONTH": "Without month", "drop_AGE": "Without age", "drop_SEX": "Without sex",
             "drop_PLACE": "Without place of treatment", "drop_BOROUGH": "Without borough", "only_WARD+BOROUGH": "Ward + borough only (where)",
             "only_WARD+MONTH": "Ward + month only (where, when)", "only_MONTH+AGE+SEX+PLACE": "Month, age, sex, place only (no location)",
             "only_MONTH+AGE+SEX": "Month, age, sex only (no location, no place)"}
    clf_col = {"XGBoost": "#0072B2", "LogisticRegression": "#E69F00", "RandomForest": "#009E73"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for j, c in enumerate([k for k in clf_col if k in set(T.classifier)]):
        for grp, filled in (("random", True), ("temporal", False)):
            d = T[(T.classifier == c) & (T.group == grp)].set_index("fieldset")
            ys = [len(order) - 1 - i + (j - 1) * 0.2 + (0.07 if grp == "temporal" else -0.07) for i, f in enumerate(order) if f in d.index]
            x = [d.loc[f, "d_auc"] for f in order if f in d.index]
            lo = [d.loc[f, "d_auc"] - d.loc[f, "d_auc_lo"] for f in order if f in d.index]
            hi = [d.loc[f, "d_auc_hi"] - d.loc[f, "d_auc"] for f in order if f in d.index]
            ax.errorbar(x, ys, xerr=[lo, hi], fmt="o", ms=4.2, color=clf_col[c], mfc=clf_col[c] if filled else "white", capsize=1.8, lw=1)
    ax.axvline(0, color="#444444", lw=0.9)
    ax.axhline(3.5 + 0.0, color="#BBBBBB", lw=0.6, ls="-")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label[f] for f in order][::-1])
    ax.set_xlabel("Change in AUC relative to the model with all six fields (95% bootstrap interval)")
    h = [Line2D([0], [0], marker="o", ls="", color=clf_col[c], label=CLF_LABEL[c], ms=5) for c in clf_col if c in set(T.classifier)]
    h += [Line2D([0], [0], marker="o", ls="", mfc="black", color="black", label="five random splits", ms=5), Line2D([0], [0], marker="o", ls="", mfc="white", color="black", label="temporal split", ms=5)]
    ax.legend(handles=h, frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "Fig_ablation")


# ------------------------------------------------------------------ decision curves
def fig_decision_curve():
    D = table("T12_decision_curve.csv")
    if D is None:
        return
    clf = "XGBoost"
    groups = [g for g in ("random_mean", "temporal") if g in set(D.group)]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.9 * len(groups), 3.3), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        for arm in ("Baseline", "SMOTE", "ADASYN"):
            d = D[(D.group == g) & (D.classifier == clf) & (D.arm == arm)].sort_values("threshold")
            if len(d):
                ax.plot(d.threshold, d.net_benefit, color=ARM_COL[arm], lw=1.6, label=ARM_LABEL[arm])
        d = D[(D.group == g) & (D.classifier == clf) & (D.arm == "Baseline")].sort_values("threshold")
        ax.plot(d.threshold, d.net_benefit_treat_all, color="#888888", lw=1.1, ls="--", label="Treat all as dengue")
        ax.axhline(0, color="#444444", lw=0.9, label="Treat none")
        ax.set_xlabel("Threshold probability")
        ax.set_title("Five random splits (mean)" if g == "random_mean" else "Temporal split (2023-2025)")
        ax.set_ylim(-0.05, max(0.34, float(d.net_benefit.max()) * 1.15 if len(d) else 0.34))
    axes[0].set_ylabel("Net benefit")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "Fig_decision_curve")


# ------------------------------------------------------------------ temporal split by year
def fig_temporal():
    T = table("T14_temporal_by_year.csv")
    if T is None or T.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))
    d = T[(T.classifier == "XGBoost") & (T.arm == "Baseline") & (T["mode"] == "0.5")].sort_values("year")
    x = np.arange(len(d))
    axes[0].bar(x - 0.18, d.observed_prevalence * 100, width=0.36, color="#888888", label="Observed")
    axes[0].bar(x + 0.18, d.mean_predicted * 100, width=0.36, color="#0072B2", label="Mean predicted (XGBoost)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(d.year)
    axes[0].set_ylabel("Dengue share (%)")
    axes[0].set_title("Prevalence shift")
    axes[0].legend(frameon=False, fontsize=7)
    panel(axes[0], "A")
    for c in CLFS:
        e = T[(T.classifier == c) & (T.arm == "Baseline") & (T["mode"] == "0.5")].sort_values("year")
        if len(e):
            axes[1].plot(e.year, e.auc, "-o", ms=3.5, lw=1.2, label=CLF_LABEL[c])
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Discrimination by test year")
    axes[1].set_xticks(sorted(T.year.unique()))
    axes[1].legend(frameon=False, fontsize=6.5, ncol=1, loc="upper right")
    panel(axes[1], "B")
    for mode, ls in (("0.5", "-"), ("tuned", "--")):
        e = T[(T.classifier == "XGBoost") & (T.arm == "Baseline") & (T["mode"] == mode)].sort_values("year")
        axes[2].plot(e.year, e.recall, ls, color="#0072B2", marker="o", ms=3.5, label=f"Recall, thr {mode}")
        axes[2].plot(e.year, e.precision, ls, color="#D55E00", marker="s", ms=3.5, label=f"Precision, thr {mode}")
    axes[2].set_xticks(sorted(T.year.unique()))
    axes[2].set_title("XGBoost, baseline")
    axes[2].legend(frameon=False, fontsize=6.5)
    panel(axes[2], "C")
    fig.tight_layout()
    save(fig, "Fig_temporal")


for fn in (fig_importance, fig_parsimony, fig_oversampling, fig_calibration, fig_leakage, fig_ablation, fig_decision_curve, fig_temporal):
    try:
        fn()
    except Exception as e:   # a missing input must not stop the other figures
        print(f"!! {fn.__name__} failed: {type(e).__name__}: {e}", flush=True)
print("figures in", FG)
