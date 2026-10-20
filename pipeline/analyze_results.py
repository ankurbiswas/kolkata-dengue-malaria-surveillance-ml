"""
analyze_results.py -- turns the predictions saved by run_revised.py into the tables for the paper.

Usage:   python analyze_results.py [output_folder]        (default: outputs_v2; BOOT=1000 bootstrap resamples)
Writes to <output_folder>/analysis/ :
  T1_overall.csv            mean and SD over the random splits, per classifier x arm x threshold mode (0.5 or tuned)
  T2_paired_differences.csv difference vs Baseline with paired-bootstrap 95% CI, plus McNemar p-values (default threshold)
  T3_matched_recall.csv     precision of each arm versus the Baseline moved to the SAME recall (does oversampling beat threshold shifting?)
  T4a_sweep_curves.csv      Macro-F1 by number of fields kept, per classifier (training CV and test)
  T4b_parsimony_selected.csv  fields chosen by training-CV, and the test difference to the full model
  T5_importance.csv         SHAP share and permutation importance per field, per classifier
  T6_temporal.csv           train 2014-2022 / test 2023-2025 results
  T7_subgroups.csv          performance by age band, sex, borough and treatment place
  T8_ward_validation.csv    per ward: observed dengue share vs mean predicted P(Dengue)  (join to the ward map in QGIS)
  T9_calibration_bins.csv   reliability-curve points
  SUMMARY.txt               the main findings in plain sentences
All probabilities are P(Dengue). Dengue is the positive class.
"""
import os
import sys
import glob
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from fastboot import Scorer, cm_from_counts, draw_weights

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs_v2")
DATA = os.environ.get("DATA") or os.path.join(HERE, "model_input_v2.csv")
BOOT = int(os.environ.get("BOOT", "1000"))
AN = os.path.join(OUT, "analysis")
os.makedirs(AN, exist_ok=True)
ARMS = ["Baseline", "ClassWeight", "SMOTE", "BorderlineSMOTE", "ADASYN"]
CLF_ORDER = ["DecisionTree", "RandomForest", "XGBoost", "LogisticRegression", "KNN", "MLP", "GaussianNB"]
FIELDS = ["WARD", "MONTH", "AGE", "SEX", "PLACE", "BOROUGH"]
METRICS = ["accuracy", "macro_f1", "recall", "precision", "auc", "brier"]
rng = np.random.default_rng(12345)


def say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ metrics
def count_metrics(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    f1d = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    f1m = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else np.nan
    return {"accuracy": (tp + tn) / len(y), "macro_f1": (f1d + f1m) / 2,
            "recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan, "dengue_f1": f1d}


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def prob_metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p)}


# ------------------------------------------------------------------ loading
df = pd.read_csv(DATA)
by_id = df.set_index("ROW_ID")
units = {}
for mp in sorted(glob.glob(os.path.join(OUT, "meta", "*.json"))):
    with open(mp, encoding="utf-8") as f:
        meta = json.load(f)
    key = os.path.basename(mp)[:-5]
    pr = pd.read_csv(os.path.join(OUT, "preds", key + ".csv.gz"))
    units[(meta["split"], meta["classifier"])] = {"meta": meta, "pr": pr, "y": by_id.loc[pr["ROW_ID"].to_numpy(), "DENGUE"].to_numpy(),
                                                  "info": by_id.loc[pr["ROW_ID"].to_numpy()].reset_index()}
if not units:
    sys.exit("no finished units found in " + OUT)
SPLITS_RANDOM = sorted({s for s, _ in units if s.startswith("seed")})
CLFS = [c for c in CLF_ORDER if any(k[1] == c for k in units)]
say(f"loaded {len(units)} units | random splits: {SPLITS_RANDOM} | temporal: {any(s == 'temporal' for s, _ in units)} | classifiers: {CLFS}")


def arms_of(u):
    return [a for a in ARMS if "p_" + a in u["pr"].columns]


def prob(u, arm):
    return u["pr"]["p_" + arm].to_numpy(float)


def thr(u, arm, mode):
    return 0.5 if mode == "0.5" else float(u["meta"]["thresholds"][arm])


def sel_prob(u):
    k = u["meta"]["selected_k"]
    return prob(u, "Baseline") if k >= len(FIELDS) else u["pr"][f"p_k{k}"].to_numpy(float)


# ------------------------------------------------------------------ T1 / T6
rows = []
for (split, clf), u in units.items():
    y = u["y"]
    for arm in arms_of(u):
        p = prob(u, arm)
        pm = prob_metrics(y, p)
        for mode in ("0.5", "tuned"):
            t = thr(u, arm, mode)
            rows.append({"split": split, "classifier": clf, "arm": arm, "mode": mode, "threshold": t,
                         **count_metrics(y, (p >= t).astype(int)), **pm})
M = pd.DataFrame(rows)
vals = ["threshold", "accuracy", "macro_f1", "recall", "precision", "specificity", "dengue_f1", "auc", "pr_auc", "brier", "ece"]
R = M[M["split"].str.startswith("seed")]
if len(R):
    g = R.groupby(["classifier", "arm", "mode"])[vals]
    T1 = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(R.groupby(["classifier", "arm", "mode"]).size().rename("n_splits")).reset_index()
    T1["classifier"] = pd.Categorical(T1["classifier"], CLF_ORDER, ordered=True)
    T1.sort_values(["classifier", "mode", "arm"]).round(4).to_csv(os.path.join(AN, "T1_overall.csv"), index=False)
Tm = M[M["split"] == "temporal"]
if len(Tm):
    Tm.round(4).to_csv(os.path.join(AN, "T6_temporal.csv"), index=False)
say("T1/T6 written")

# ------------------------------------------------------------------ T2 (paired bootstrap)


def variants(u):
    v = {}
    for arm in arms_of(u):
        p = prob(u, arm)
        v[(arm, "0.5")] = (p, 0.5, arm)
        v[(arm, "tuned")] = (p, thr(u, arm, "tuned"), arm)
    v[("SEL", "0.5")] = (sel_prob(u), 0.5, "SEL")
    return v


def scorers_of(y, vs):
    """One fast bootstrap scorer per distinct score vector (the arm, or the selected-fields model)."""
    sc = {}
    for p, t, tag in vs.values():
        if tag not in sc:
            sc[tag] = Scorer(y, p)
    return sc


def variant_metrics(sc, vs, w):
    """Metrics of every variant for the bootstrap resample w (w = number of times each case is drawn; all ones = the original sample)."""
    aucs, out = {}, {}
    for key, (p, t, tag) in vs.items():
        s = sc[tag]
        cm = cm_from_counts(*s.counts(t, w))
        if tag not in aucs:
            aucs[tag] = s.auc(w)
        out[key] = np.array([cm["accuracy"], cm["macro_f1"], cm["recall"], cm["precision"], aucs[tag], s.brier(w)])
    return out


def comparisons(arms):
    c = []
    for a in arms:
        if a != "Baseline":
            c.append((f"{a}@0.5 minus Baseline@0.5", (a, "0.5"), ("Baseline", "0.5")))
    for a in arms:
        if a != "Baseline":
            c.append((f"{a}@tuned minus Baseline@tuned", (a, "tuned"), ("Baseline", "tuned")))
    c.append(("Baseline@tuned minus Baseline@0.5", ("Baseline", "tuned"), ("Baseline", "0.5")))
    c.append(("SelectedFields@0.5 minus AllFields@0.5", ("SEL", "0.5"), ("Baseline", "0.5")))
    return c


t2_rows = []
for clf in CLFS:
    us = [units[(s, clf)] for s in SPLITS_RANDOM if (s, clf) in units]
    if not us:
        continue
    arms = [a for a in ARMS if all(a in arms_of(u) for u in us)]
    comps = comparisons(arms)
    vsl = [variants(u) for u in us]
    scl = [scorers_of(u["y"], vs) for u, vs in zip(us, vsl)]
    point = {c[0]: [] for c in comps}
    for u, vs, sc in zip(us, vsl, scl):
        m = variant_metrics(sc, vs, np.ones(len(u["y"])))
        for name, A, Bk in comps:
            point[name].append(m[A] - m[Bk])
    boot = {c[0]: np.zeros((BOOT, len(METRICS))) for c in comps}
    for b in range(BOOT):
        per = {c[0]: [] for c in comps}
        for u, vs, sc in zip(us, vsl, scl):
            m = variant_metrics(sc, vs, draw_weights(rng, len(u["y"])))
            for name, A, Bk in comps:
                per[name].append(m[A] - m[Bk])
        for name in per:
            boot[name][b] = np.nanmean(per[name], axis=0)
    for name, A, Bk in comps:
        est = np.nanmean(point[name], axis=0)
        lo, hi = np.nanpercentile(boot[name], [2.5, 97.5], axis=0)
        mc_p, mc_n = np.nan, np.nan
        if name.endswith("@0.5 minus Baseline@0.5") and A[0] != "SEL":
            ps = []
            for u in us:
                y, pb, pa = u["y"], prob(u, "Baseline"), prob(u, A[0])
                cb, ca = ((pb >= 0.5).astype(int) == y), ((pa >= 0.5).astype(int) == y)
                n10, n01 = int(np.sum(cb & ~ca)), int(np.sum(~cb & ca))
                ps.append(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)
            mc_p, mc_n = float(np.median(ps)), int(np.sum(np.array(ps) < 0.05))
        for j, mname in enumerate(METRICS):
            t2_rows.append({"classifier": clf, "comparison": name, "metric": mname, "difference": est[j], "ci_low": lo[j], "ci_high": hi[j],
                            "ci_excludes_zero": int(lo[j] > 0 or hi[j] < 0), "mcnemar_median_p": mc_p, "mcnemar_splits_p_below_0.05": mc_n,
                            "n_splits": len(us)})
    say(f"  T2 bootstrap done: {clf}")
if t2_rows:
    pd.DataFrame(t2_rows).round(5).to_csv(os.path.join(AN, "T2_paired_differences.csv"), index=False)

# ------------------------------------------------------------------ T3 (matched recall)
t3 = []
for clf in CLFS:
    for s in SPLITS_RANDOM:
        u = units.get((s, clf))
        if not u:
            continue
        y, pb = u["y"], prob(u, "Baseline")
        prec, rec, _ = precision_recall_curve(y, pb)
        for arm in arms_of(u):
            if arm == "Baseline":
                continue
            ca = count_metrics(y, (prob(u, arm) >= 0.5).astype(int))
            if not np.isfinite(ca["recall"]) or not np.isfinite(ca["precision"]):
                continue
            ok = np.where(rec >= ca["recall"] - 1e-12)[0]
            if len(ok):
                t3.append({"classifier": clf, "split": s, "arm": arm, "arm_recall": ca["recall"], "arm_precision": ca["precision"],
                           "baseline_precision_at_same_recall": float(prec[ok.max()])})
t3df = pd.DataFrame(t3)
if t3:
    t3df["precision_gain_vs_threshold_shift"] = t3df["arm_precision"] - t3df["baseline_precision_at_same_recall"]
    g = t3df.groupby(["classifier", "arm"])[["arm_recall", "arm_precision", "baseline_precision_at_same_recall", "precision_gain_vs_threshold_shift"]]
    T3 = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
    T3.round(4).to_csv(os.path.join(AN, "T3_matched_recall.csv"), index=False)
say("T3 written")

# ------------------------------------------------------------------ T4 (parsimony) and T5 (importance)
sel_rows = []
sw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "sweep", "*.csv")))], ignore_index=True)
swr = sw[sw["split"].str.startswith("seed")]
if len(swr):
    a = swr.groupby(["classifier", "k"]).agg(fields_most_common=("fields", lambda s: s.value_counts().index[0]),
                                            cv_macro_f1_mean=("cv_macro_f1", "mean"), cv_macro_f1_sd=("cv_macro_f1", "std"),
                                            test_macro_f1_mean=("test_macro_f1", "mean"), test_macro_f1_sd=("test_macro_f1", "std"),
                                            test_auc_mean=("test_auc", "mean"), test_recall_mean=("test_dengue_recall", "mean"),
                                            times_selected=("selected", "sum"), n_splits=("split", "nunique")).reset_index()
    a["classifier"] = pd.Categorical(a["classifier"], CLF_ORDER, ordered=True)
    a.sort_values(["classifier", "k"]).round(4).to_csv(os.path.join(AN, "T4a_sweep_curves.csv"), index=False)
    for clf in CLFS:
        sel = swr[(swr.classifier == clf) & (swr.selected == 1)]
        if sel.empty:
            continue
        full = swr[(swr.classifier == clf) & (swr.k == len(FIELDS))].set_index("split")
        row = {"classifier": clf, "k_selected_per_split": ",".join(str(int(k)) for k in sel.sort_values("split")["k"]),
               "k_most_common": int(sel["k"].mode().iloc[0]), "fields_most_common": sel["fields"].mode().iloc[0],
               "rank_method": sel["rank_method"].iloc[0], "test_macro_f1_selected_mean": sel["test_macro_f1"].mean(),
               "test_macro_f1_full_mean": full["test_macro_f1"].mean()}
        if t2_rows:
            q = [r for r in t2_rows if r["classifier"] == clf and r["comparison"].startswith("SelectedFields") and r["metric"] == "macro_f1"]
            if q:
                row.update({"macro_f1_difference_vs_full": q[0]["difference"], "ci_low": q[0]["ci_low"], "ci_high": q[0]["ci_high"]})
        sel_rows.append(row)
    pd.DataFrame(sel_rows).round(4).to_csv(os.path.join(AN, "T4b_parsimony_selected.csv"), index=False)
imp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "importance", "*.csv")))], ignore_index=True)
impr = imp[imp["split"].str.startswith("seed")]
if len(impr):
    g = impr.groupby(["classifier", "field"])
    T5 = g[["shap_share_pct", "perm_auc_drop", "rank_used"]].mean().add_suffix("_mean").join(g[["shap_share_pct", "perm_auc_drop"]].std().add_suffix("_sd")).reset_index()
    T5["classifier"] = pd.Categorical(T5["classifier"], CLF_ORDER, ordered=True)
    T5.sort_values(["classifier", "rank_used_mean"]).round(4).to_csv(os.path.join(AN, "T5_importance.csv"), index=False)
say("T4/T5 written")

# ------------------------------------------------------------------ T7 subgroups, T8 ward validation, T9 calibration
bands = [(0, 15, "0-14"), (15, 30, "15-29"), (30, 45, "30-44"), (45, 60, "45-59"), (60, 200, "60+")]


def band(a):
    for lo, hi, lab in bands:
        if lo <= a < hi:
            return lab
    return "unknown"


t7 = []
for clf in [c for c in ("XGBoost", "LogisticRegression") if c in CLFS]:
    for s in SPLITS_RANDOM:
        u = units.get((s, clf))
        if not u:
            continue
        info = u["info"].copy()
        info["AGE_BAND"] = info["AGE"].map(band)
        y, p = u["y"], prob(u, "Baseline")
        for mode in ("0.5", "tuned"):
            t = thr(u, "Baseline", mode)
            for gcol in ("AGE_BAND", "SEX", "BOROUGH", "PLACE OF TREATMENT"):
                for gv, idx in info.groupby(gcol).indices.items():
                    yy, pp = y[idx], p[idx]
                    cm = count_metrics(yy, (pp >= t).astype(int))
                    t7.append({"classifier": clf, "split": s, "mode": mode, "group": gcol, "level": gv, "n": len(idx), "dengue_prevalence": yy.mean(),
                               "mean_predicted": pp.mean(), "recall": cm["recall"], "precision": cm["precision"], "accuracy": cm["accuracy"],
                               "auc": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan})
if t7:
    d = pd.DataFrame(t7)
    d.groupby(["classifier", "mode", "group", "level"])[["n", "dengue_prevalence", "mean_predicted", "recall", "precision", "accuracy", "auc"]].mean().reset_index().round(4).to_csv(
        os.path.join(AN, "T7_subgroups.csv"), index=False)
t8, t9 = [], []
for clf in CLFS:
    parts = []
    for s in SPLITS_RANDOM:
        u = units.get((s, clf))
        if u:
            d = pd.DataFrame({"WARD": u["info"]["WARD"].to_numpy(), "y": u["y"], "p": prob(u, "Baseline")})
            parts.append(d)
    if not parts:
        continue
    d = pd.concat(parts)
    w = d.groupby("WARD").agg(n_test_predictions=("y", "size"), observed_dengue_share=("y", "mean"), mean_predicted_p_dengue=("p", "mean")).reset_index()
    w["residual_pred_minus_obs"] = w["mean_predicted_p_dengue"] - w["observed_dengue_share"]
    w.insert(0, "classifier", clf)
    t8.append(w)
    for arm in ARMS:
        ps = [(units[(s, clf)]["y"], prob(units[(s, clf)], arm)) for s in SPLITS_RANDOM if (s, clf) in units and arm in arms_of(units[(s, clf)])]
        if not ps:
            continue
        y = np.concatenate([a for a, _ in ps])
        p = np.concatenate([b for _, b in ps])
        bins = pd.qcut(p, 10, duplicates="drop")
        c = pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)
        c.insert(0, "arm", arm)
        c.insert(0, "classifier", clf)
        t9.append(c)
if t8:
    W = pd.concat(t8)
    W.round(5).to_csv(os.path.join(AN, "T8_ward_validation.csv"), index=False)
if t9:
    pd.concat(t9).round(5).to_csv(os.path.join(AN, "T9_calibration_bins.csv"), index=False)
say("T7/T8/T9 written")

# ------------------------------------------------------------------ SUMMARY
L = ["SUMMARY (random splits unless stated; Dengue = positive class)", ""]
if len(R):
    r05 = T1[T1["mode"] == "0.5"].copy()
    best = r05.loc[r05.groupby("classifier", observed=True)["accuracy_mean"].idxmax()]
    nb = int((best["arm"] == "Baseline").sum())
    L.append(f"1. At the default threshold, Baseline has the highest mean accuracy in {nb} of {len(best)} classifiers.")
    for c in CLFS:
        x = r05[r05.classifier == c].set_index("arm")
        if "Baseline" in x.index:
            b = x.loc["Baseline"]
            parts = [f"{a}: acc {x.loc[a, 'accuracy_mean']:.4f}, recall {x.loc[a, 'recall_mean']:.3f}, prec {x.loc[a, 'precision_mean']:.3f}, AUC {x.loc[a, 'auc_mean']:.4f}" for a in ARMS if a in x.index]
            L.append(f"   {c}: " + " | ".join(parts))
if t2_rows:
    T2 = pd.DataFrame(t2_rows)
    q = T2[(T2.metric == "auc") & T2.comparison.str.contains("@0.5 minus Baseline@0.5")]
    L += ["", f"2. AUC difference versus Baseline (arms other than Baseline): CI excludes zero in {int(q.ci_excludes_zero.sum())} of {len(q)} comparisons; "
              f"positive (better) in {int(((q.difference > 0) & (q.ci_excludes_zero == 1)).sum())}, negative (worse) in {int(((q.difference < 0) & (q.ci_excludes_zero == 1)).sum())}."]
    q = T2[(T2.metric == "macro_f1") & T2.comparison.str.contains("@tuned minus Baseline@tuned")]
    L.append(f"3. After tuning the decision threshold of every arm, Macro-F1 difference versus Baseline: CI excludes zero in {int(q.ci_excludes_zero.sum())} of {len(q)}; "
             f"arm better in {int(((q.difference > 0) & (q.ci_excludes_zero == 1)).sum())}, arm worse in {int(((q.difference < 0) & (q.ci_excludes_zero == 1)).sum())}.")
if t3:
    L.append(f"4. Matched-recall check: mean precision gain of an oversampling/class-weight arm over the Baseline moved to the same recall = "
             f"{t3df['precision_gain_vs_threshold_shift'].mean():+.4f} (near zero means the arm behaves like a threshold shift).")
if sel_rows:
    L.append("5. Parsimony (fields kept, chosen on training CV): " + "; ".join(f"{r['classifier']} k={r['k_most_common']} ({r['fields_most_common']})" for r in sel_rows))
if len(Tm):
    z = Tm[(Tm["mode"] == "0.5") & (Tm["arm"] == "Baseline")].set_index("classifier")
    L.append("6. Temporal split (train 2014-2022, test 2023-2025), Baseline accuracy: " + "; ".join(f"{c} {z.loc[c, 'accuracy']:.4f}" for c in z.index))
with open(os.path.join(AN, "SUMMARY.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
say("\n".join(L))
say(f"done -> {AN}")
