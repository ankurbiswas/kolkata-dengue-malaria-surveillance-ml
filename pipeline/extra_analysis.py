"""
extra_analysis.py -- additional analyses for the paper, built on the predictions saved by run_revised.py

  A  Field ablations (retrains the Baseline arm on field subsets, with the tuned hyperparameters of the main run):
       leave-one-field-out, and field families ("where", "when", "who").  Answers RQ1 without relying on SHAP, and checks that the
       result survives dropping Place of treatment (recorded at notification, so it may partly encode the diagnosis).
  B  Calibration slope and calibration-in-the-large (TRIPOD+AI items), per classifier x arm x split.
  C  Decision curves (net benefit) for XGBoost, LogisticRegression and RandomForest, every arm.
  D  Operating points of the Baseline arm: sensitivity at fixed specificity, specificity at fixed sensitivity, PPV, NPV.
  E  The temporal split broken down by test year (2023, 2024, 2025): prevalence shift versus discrimination.

Run AFTER run_revised.py:        python extra_analysis.py
Settings (environment variables, optional):
  SKIP_ABLATION=1                 skip part A (the only part that retrains models)
  ABL_CLASSIFIERS=XGBoost,...     default: DecisionTree,RandomForest,XGBoost,LogisticRegression (k-NN and MLP refits are slow and add little)
  ABL_SPLITS=seed1,temporal       default: all splits
  ABL_BOOT=1000                   bootstrap resamples for the ablation confidence intervals (PERF_BOOT=1000 for T15/T16)
Writes to <outputs>/analysis/:
  T10_ablation_summary.csv  T10b_ablation_long.csv  T11_calibration.csv  T12_decision_curve.csv
  T13_operating_points.csv  T14_temporal_by_year.csv  T15_primary_ci.csv  T16_classifier_comparison.csv  EXTRA_SUMMARY.txt
and <outputs>/ablation/<split>__<classifier>.csv.gz (predictions of every ablated model; the run can be resumed).
  F  (T15) bootstrap confidence intervals for the primary measures of the Baseline arm; G (T16) paired comparison of every classifier with
     the best one (largest AUC).   Setting: PERF_BOOT=1000
  H  (T17) the five-field model (no place of treatment) applied to the 7,665 cases excluded only because place of treatment is missing
     (7,578 of them from 2014; needs excluded_place_missing.csv, built by make_excluded_input.py).
All probabilities are P(Dengue). Dengue is the positive class.
"""
import os
import glob
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
import run_revised as R   # noqa: E402  (same encoders, models and settings as the main run)
from fastboot import Scorer, cm_from_counts, draw_weights as _draw_weights   # noqa: E402

OUT = R.OUT
AN = os.path.join(OUT, "analysis")
ABD = os.path.join(OUT, "ablation")
os.makedirs(AN, exist_ok=True)
os.makedirs(ABD, exist_ok=True)
SKIP_ABLATION = os.environ.get("SKIP_ABLATION", "0") == "1"
ABL_CLFS = [c for c in os.environ.get("ABL_CLASSIFIERS", "DecisionTree,RandomForest,XGBoost,LogisticRegression").split(",") if c.strip()]
ABL_SPLITS = [c for c in os.environ.get("ABL_SPLITS", "").split(",") if c.strip()]
ABL_BOOT = int(os.environ.get("ABL_BOOT", "1000"))
BOOT_CLFS = ["XGBoost", "LogisticRegression", "RandomForest"]
DC_CLFS = ["XGBoost", "LogisticRegression", "RandomForest"]
ARMS = R.ARMS
CLF_ORDER = R.ALL_CLASSIFIERS
rng = np.random.default_rng(2024)
T0 = time.time()

ABLATIONS = {
    "drop_WARD": ["MONTH", "AGE", "SEX", "PLACE", "BOROUGH"],
    "drop_MONTH": ["WARD", "AGE", "SEX", "PLACE", "BOROUGH"],
    "drop_AGE": ["WARD", "MONTH", "SEX", "PLACE", "BOROUGH"],
    "drop_SEX": ["WARD", "MONTH", "AGE", "PLACE", "BOROUGH"],
    "drop_PLACE": ["WARD", "MONTH", "AGE", "SEX", "BOROUGH"],
    "drop_BOROUGH": ["WARD", "MONTH", "AGE", "SEX", "PLACE"],
    "only_WARD+BOROUGH": ["WARD", "BOROUGH"],                      # where
    "only_WARD+MONTH": ["WARD", "MONTH"],                          # where and when
    "only_MONTH+AGE+SEX+PLACE": ["MONTH", "AGE", "SEX", "PLACE"],  # no location at all
    "only_MONTH+AGE+SEX": ["MONTH", "AGE", "SEX"],                 # no location, no place of treatment
}


def say(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def count_metrics(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    f1d = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    f1m = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else np.nan
    return {"accuracy": (tp + tn) / len(y), "macro_f1": (f1d + f1m) / 2,
            "recall": tp / (tp + fn) if (tp + fn) else np.nan, "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan, "npv": tn / (tn + fn) if (tn + fn) else np.nan}


def draw_weights(n):
    return _draw_weights(rng, n)


# ------------------------------------------------------------------ loading
df = pd.read_csv(R.DATA)
if R.SUBSAMPLE < 1.0:
    df = df.sample(frac=R.SUBSAMPLE, random_state=0).sort_values("ROW_ID")
df = df.reset_index(drop=True)
by_id = df.set_index("ROW_ID")
SPL = np.load(os.path.join(OUT, "splits.npz"))
units = {}
for mp in sorted(glob.glob(os.path.join(OUT, "meta", "*.json"))):
    with open(mp, encoding="utf-8") as f:
        meta = json.load(f)
    key = os.path.basename(mp)[:-5]
    pr = pd.read_csv(os.path.join(OUT, "preds", key + ".csv.gz"))
    ids = pr["ROW_ID"].to_numpy()
    units[(meta["split"], meta["classifier"])] = {"meta": meta, "pr": pr, "key": key, "y": by_id.loc[ids, "DENGUE"].to_numpy(),
                                                  "info": by_id.loc[ids].reset_index()}
if not units:
    raise SystemExit("no finished units in " + OUT)
RANDOM = sorted({s for s, _ in units if s.startswith("seed")})
CLFS = [c for c in CLF_ORDER if any(k[1] == c for k in units)]
say(f"loaded {len(units)} units | random splits {RANDOM} | temporal {any(s == 'temporal' for s, _ in units)} | classifiers {CLFS}")


def arms_of(u):
    return [a for a in ARMS if "p_" + a in u["pr"].columns]


def prob(u, arm):
    return u["pr"]["p_" + arm].to_numpy(float)


# ------------------------------------------------------------------ A. field ablations
def run_ablation():
    X, y_all = df[R.RAWCOLS], df["DENGUE"].to_numpy()
    for (split, clf), u in units.items():
        if (ABL_CLFS and clf not in ABL_CLFS) or (ABL_SPLITS and split not in ABL_SPLITS):
            continue
        path = os.path.join(ABD, u["key"] + ".csv.gz")
        if os.path.exists(path):
            continue
        meta = u["meta"]
        tr, te = SPL[f"{split}_train"], SPL[f"{split}_test"]
        assert np.array_equal(df["ROW_ID"].iloc[te].to_numpy(), u["pr"]["ROW_ID"].to_numpy()), "test rows differ from the saved predictions"
        Xtr, ytr, Xte = X.iloc[tr].reset_index(drop=True), y_all[tr], X.iloc[te].reset_index(drop=True)
        out = pd.DataFrame({"ROW_ID": u["pr"]["ROW_ID"].to_numpy()})
        t0 = time.time()
        for name, fields in ABLATIONS.items():
            pipe = R.make_pipe(clf, meta["params"], fields, meta["seed"], "Baseline", meta["scale_pos_weight"]).fit(Xtr, ytr)
            out["p_" + name] = pipe.predict_proba(Xte)[:, 1].astype("float32")
        chk = ""
        if clf in ("DecisionTree", "LogisticRegression", "XGBoost"):   # reproducibility check of the full-field model
            pipe = R.make_pipe(clf, meta["params"], R.FIELDS, meta["seed"], "Baseline", meta["scale_pos_weight"]).fit(Xtr, ytr)
            chk = f" | refit of the full model differs from the saved one by at most {np.abs(pipe.predict_proba(Xte)[:, 1] - u['pr']['p_Baseline'].to_numpy()).max():.1e}"
        out.to_csv(path, index=False, compression="gzip", float_format="%.5f")
        say(f"ablation {split:9s} {clf:18s} {time.time() - t0:5.0f}s{chk}")


def ablation_tables():
    rows, store = [], {}
    for (split, clf), u in units.items():
        path = os.path.join(ABD, u["key"] + ".csv.gz")
        if not os.path.exists(path):
            continue
        ab = pd.read_csv(path)
        y = u["y"]
        sets = {"ALL": prob(u, "Baseline")}
        for c in ab.columns[1:]:
            sets[c[2:]] = ab[c].to_numpy(float)
        store[(split, clf)] = (y, sets)
        for name, p in sets.items():
            cm = count_metrics(y, (p >= 0.5).astype(int))
            rows.append({"split": split, "classifier": clf, "fieldset": name, **cm, "auc": roc_auc_score(y, p), "brier": float(np.mean((p - y) ** 2))})
    if not rows:
        return None
    Lg = pd.DataFrame(rows)
    Lg.round(5).to_csv(os.path.join(AN, "T10b_ablation_long.csv"), index=False)
    order = ["ALL"] + list(ABLATIONS)
    out = []
    for clf in [c for c in CLF_ORDER if c in set(Lg.classifier)]:
        sub = Lg[Lg.classifier == clf]
        rnd = sub[sub.split.str.startswith("seed")]
        tmp = sub[sub.split == "temporal"]
        for name in order:
            r = {"classifier": clf, "fieldset": name}
            a = rnd[rnd.fieldset == name]
            if len(a):
                for m in ("auc", "macro_f1", "accuracy", "recall", "precision"):
                    r[f"{m}_mean"], r[f"{m}_sd"] = a[m].mean(), a[m].std()
                r["n_random_splits"] = len(a)
            t = tmp[tmp.fieldset == name]
            if len(t):
                r["temporal_auc"], r["temporal_macro_f1"] = float(t["auc"].iloc[0]), float(t["macro_f1"].iloc[0])
            out.append(r)
    S = pd.DataFrame(out)
    # paired bootstrap of the difference to ALL (selected classifiers only)
    ci = []
    for clf in BOOT_CLFS:
        for grp, splits_ in (("random", [s for s in RANDOM if (s, clf) in store]), ("temporal", ["temporal"] if ("temporal", clf) in store else [])):
            if not splits_:
                continue
            names = [n for n in order if n != "ALL"]
            sc = {s: {n: Scorer(store[(s, clf)][0], p) for n, p in store[(s, clf)][1].items()} for s in splits_}

            def one(w_by):   # (dAUC, dMacro-F1) of every field set against ALL, averaged over the splits
                acc = np.zeros((len(names), 2))
                for s in splits_:
                    w, base = w_by[s], sc[s]["ALL"]
                    a_auc, a_f1 = base.auc(w), cm_from_counts(*base.counts(0.5, w))["macro_f1"]
                    for i, n in enumerate(names):
                        acc[i] += np.array([sc[s][n].auc(w) - a_auc, cm_from_counts(*sc[s][n].counts(0.5, w))["macro_f1"] - a_f1]) / len(splits_)
                return acc
            point = one({s: np.ones(len(store[(s, clf)][0])) for s in splits_})
            boot = np.array([one({s: draw_weights(len(store[(s, clf)][0])) for s in splits_}) for _ in range(ABL_BOOT)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            for i, n in enumerate(names):
                ci.append({"classifier": clf, "fieldset": n, "group": grp, "d_auc": point[i, 0], "d_auc_lo": lo[i, 0], "d_auc_hi": hi[i, 0],
                           "d_macro_f1": point[i, 1], "d_macro_f1_lo": lo[i, 1], "d_macro_f1_hi": hi[i, 1], "n_splits": len(splits_)})
            say(f"  ablation bootstrap done: {clf} ({grp})")
    if ci:
        C = pd.DataFrame(ci)
        C.round(5).to_csv(os.path.join(AN, "T10c_ablation_difference_ci.csv"), index=False)
    S.round(4).to_csv(os.path.join(AN, "T10_ablation_summary.csv"), index=False)
    return S


# ------------------------------------------------------------------ B. calibration slope and calibration-in-the-large
def calibration_stats(y, p):
    q = np.clip(p, 1e-3, 1 - 1e-3)
    lp = np.log(q / (1 - q))
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500).fit(lp.reshape(-1, 1), y)
    a = 0.0                                          # calibration-in-the-large: intercept with the slope fixed at 1
    for _ in range(60):
        pr = 1 / (1 + np.exp(-(a + lp)))
        step = np.sum(y - pr) / (-np.sum(pr * (1 - pr)))
        a -= step
        if abs(step) < 1e-9:
            break
    return float(m.coef_[0, 0]), float(m.intercept_[0]), float(a)


def calibration_table():
    rows = []
    for (split, clf), u in units.items():
        y = u["y"]
        for arm in arms_of(u):
            p = prob(u, arm)
            slope, icpt, citl = calibration_stats(y, p)
            rows.append({"split": split, "classifier": clf, "arm": arm, "observed_prevalence": y.mean(), "mean_predicted": p.mean(),
                         "calibration_slope": slope, "calibration_intercept": icpt, "citl": citl})
    Cb = pd.DataFrame(rows)
    Cb.round(4).to_csv(os.path.join(AN, "T11_calibration.csv"), index=False)
    return Cb


# ------------------------------------------------------------------ C. decision curves
def decision_curve_table():
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    rows = []
    for (split, clf), u in units.items():
        if clf not in DC_CLFS:
            continue
        y = u["y"]
        n, prev = len(y), y.mean()
        for arm in arms_of(u):
            p = prob(u, arm)
            for t in grid:
                pred = p >= t
                tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
                w = t / (1 - t)
                rows.append({"split": split, "classifier": clf, "arm": arm, "threshold": t, "net_benefit": tp / n - fp / n * w,
                             "net_benefit_treat_all": prev - (1 - prev) * w, "prevalence": prev})
    Dc = pd.DataFrame(rows)
    Dc["group"] = np.where(Dc["split"].str.startswith("seed"), "random_mean", "temporal")
    S = Dc.groupby(["group", "classifier", "arm", "threshold"])[["net_benefit", "net_benefit_treat_all", "prevalence"]].mean().reset_index()
    S.round(5).to_csv(os.path.join(AN, "T12_decision_curve.csv"), index=False)
    return S


# ------------------------------------------------------------------ D. operating points (Baseline arm)
def operating_points_table():
    rows = []
    for (split, clf), u in units.items():
        y, p = u["y"], prob(u, "Baseline")
        fpr, tpr, thr = roc_curve(y, p)
        spec = 1 - fpr
        P, N = int(y.sum()), int((1 - y).sum())

        def at(i):
            tp, fp = tpr[i] * P, fpr[i] * N
            return {"ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": (spec[i] * N) / (spec[i] * N + (1 - tpr[i]) * P)}
        r = {"split": split, "classifier": clf}
        for s in (0.90, 0.95):
            ok = np.where(spec >= s - 1e-12)[0]
            i = ok[np.argmax(tpr[ok])]
            r[f"sens_at_spec{int(s * 100)}"] = float(tpr[i])
            r[f"ppv_at_spec{int(s * 100)}"] = at(i)["ppv"]
        for s in (0.80, 0.90):
            ok = np.where(tpr >= s - 1e-12)[0]
            i = ok[np.argmax(spec[ok])]
            r[f"spec_at_sens{int(s * 100)}"] = float(spec[i])
            r[f"ppv_at_sens{int(s * 100)}"] = at(i)["ppv"]
            r[f"npv_at_sens{int(s * 100)}"] = at(i)["npv"]
        cm = count_metrics(y, (p >= 0.5).astype(int))
        r["npv_at_0.5"] = cm["npv"]
        t = float(u["meta"]["thresholds"]["Baseline"])
        cm = count_metrics(y, (p >= t).astype(int))
        r.update({"tuned_threshold": t, "recall_at_tuned": cm["recall"], "precision_at_tuned": cm["precision"], "specificity_at_tuned": cm["specificity"],
                  "npv_at_tuned": cm["npv"], "macro_f1_at_tuned": cm["macro_f1"], "accuracy_at_tuned": cm["accuracy"]})
        rows.append(r)
    Op = pd.DataFrame(rows)
    Op["group"] = np.where(Op["split"].str.startswith("seed"), "random_mean", "temporal")
    num = [c for c in Op.columns if c not in ("split", "classifier", "group")]
    g = Op.groupby(["group", "classifier"])[num]
    S = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
    S.round(4).to_csv(os.path.join(AN, "T13_operating_points.csv"), index=False)
    return S


# ------------------------------------------------------------------ E. temporal split by test year
def temporal_by_year_table():
    rows = []
    for (split, clf), u in units.items():
        if split != "temporal":
            continue
        info, y = u["info"], u["y"]
        for arm in arms_of(u):
            p = prob(u, arm)
            for mode, t in (("0.5", 0.5), ("tuned", float(u["meta"]["thresholds"][arm]))):
                for yr, idx in info.groupby("YEAR").indices.items():
                    yy, pp = y[idx], p[idx]
                    cm = count_metrics(yy, (pp >= t).astype(int))
                    rows.append({"classifier": clf, "arm": arm, "mode": mode, "threshold": t, "year": int(yr), "n": len(idx), "observed_prevalence": yy.mean(),
                                 "mean_predicted": pp.mean(), "auc": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan, "accuracy": cm["accuracy"],
                                 "macro_f1": cm["macro_f1"], "recall": cm["recall"], "precision": cm["precision"], "specificity": cm["specificity"]})
    if not rows:
        return None
    Ty = pd.DataFrame(rows)
    Ty.round(4).to_csv(os.path.join(AN, "T14_temporal_by_year.csv"), index=False)
    return Ty


# ------------------------------------------------------------------ F. primary measures with bootstrap CIs, G. classifier comparison
PERF_BOOT = int(os.environ.get("PERF_BOOT", "1000"))
PERF_METRICS = ["accuracy", "macro_f1", "recall", "precision", "specificity", "auc"]


def primary_and_comparison_tables():
    rows, comp = [], []
    for grp, splits_ in (("random_mean", RANDOM), ("temporal", ["temporal"])):
        present = [c for c in CLFS if splits_ and all((s, c) in units for s in splits_)]
        if not present:
            continue
        for s in splits_:   # the test rows must be identical, and in the same order, for every classifier (paired bootstrap)
            ids = units[(s, present[0])]["pr"]["ROW_ID"].to_numpy()
            assert all(np.array_equal(ids, units[(s, c)]["pr"]["ROW_ID"].to_numpy()) for c in present), "test rows differ between classifiers"
        thr = {(s, c): float(units[(s, c)]["meta"]["thresholds"]["Baseline"]) for s in splits_ for c in present}
        sc = {(s, c): Scorer(units[(s, c)]["y"], prob(units[(s, c)], "Baseline")) for s in splits_ for c in present}

        def metrics_for(w_by):
            out = {}
            for c in present:
                per05, pert = [], []
                for s in splits_:
                    w, scr = w_by[s], sc[(s, c)]
                    auc = scr.auc(w)
                    a, b = cm_from_counts(*scr.counts(0.5, w)), cm_from_counts(*scr.counts(thr[(s, c)], w))
                    per05.append([a["accuracy"], a["macro_f1"], a["recall"], a["precision"], a["specificity"], auc])
                    pert.append([b["accuracy"], b["macro_f1"], b["recall"], b["precision"], b["specificity"], auc])
                out[c] = (np.mean(per05, axis=0), np.mean(pert, axis=0))
            return out
        n_by = {s: len(units[(s, present[0])]["y"]) for s in splits_}
        point = metrics_for({s: np.ones(n_by[s]) for s in splits_})
        ref = max(present, key=lambda c: point[c][0][5])
        b05 = {c: np.zeros((PERF_BOOT, 6)) for c in present}
        bt = {c: np.zeros((PERF_BOOT, 6)) for c in present}
        for b in range(PERF_BOOT):
            m = metrics_for({s: draw_weights(n_by[s]) for s in splits_})   # the same resample is used for every classifier (paired)
            for c in present:
                b05[c][b], bt[c][b] = m[c]
        for c in present:
            for mode, Bm, Pm in (("0.5", b05[c], point[c][0]), ("tuned", bt[c], point[c][1])):
                lo, hi = np.percentile(Bm, [2.5, 97.5], axis=0)
                for j, mname in enumerate(PERF_METRICS):
                    rows.append({"group": grp, "classifier": c, "mode": mode, "metric": mname, "estimate": Pm[j], "ci_low": lo[j], "ci_high": hi[j], "n_splits": len(splits_)})
            if c != ref:
                for j, mname in ((1, "macro_f1"), (5, "auc")):
                    d = b05[c][:, j] - b05[ref][:, j]
                    comp.append({"group": grp, "reference": ref, "classifier": c, "metric": mname, "difference": point[c][0][j] - point[ref][0][j],
                                 "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
        say(f"  primary-measure bootstrap done ({grp}); reference classifier for comparisons: {ref}")
    if rows:
        pd.DataFrame(rows).round(5).to_csv(os.path.join(AN, "T15_primary_ci.csv"), index=False)
        pd.DataFrame(comp).round(5).to_csv(os.path.join(AN, "T16_classifier_comparison.csv"), index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ H. cases excluded only for missing place of treatment (almost all from 2014)
EXCL = os.environ.get("EXCLUDED") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "excluded_place_missing.csv")
H_CLFS = ["XGBoost", "LogisticRegression", "RandomForest"]


def excluded_records_table():
    """The five-field model (no place of treatment) applied to cases the main analysis never used: month, age, sex, ward and borough are
    complete, only place of treatment is missing. 7,578 of the 7,665 are 2014 notifications (8.2% dengue)."""
    if not os.path.exists(EXCL):
        say("excluded_place_missing.csv not found: part H skipped")
        return None
    ex = pd.read_csv(EXCL)
    ex["PLACE OF TREATMENT"] = "HOME TREATMENT"   # placeholder column; the five-field models never use it
    y_ex = ex["DENGUE"].to_numpy()
    masks = {"all excluded cases": np.ones(len(ex), bool), "2014 only": (ex["YEAR"] == 2014).to_numpy()}
    X_all, y_all = df[R.RAWCOLS], df["DENGUE"].to_numpy()
    pred = {}
    for (split, clf), u in units.items():
        if clf not in H_CLFS:
            continue
        meta, tr = u["meta"], SPL[f"{split}_train"]
        pipe = R.make_pipe(clf, meta["params"], ABLATIONS["drop_PLACE"], meta["seed"], "Baseline", meta["scale_pos_weight"]).fit(X_all.iloc[tr].reset_index(drop=True), y_all[tr])
        pred[(split, clf)] = pipe.predict_proba(ex[R.RAWCOLS])[:, 1]
        say(f"  excluded-records check: {split} {clf}")
    rows = []
    for grp, splits_ in (("random_mean", RANDOM), ("temporal", ["temporal"])):
        for clf in H_CLFS:
            ss = [s for s in splits_ if (s, clf) in pred]
            if not ss:
                continue
            for sname, m in masks.items():
                if m.sum() < 50 or y_ex[m].sum() == 0:
                    continue
                scs = {s: Scorer(y_ex[m], pred[(s, clf)][m]) for s in ss}
                n = int(m.sum())

                def vec(w):
                    out = []
                    for s in ss:
                        sc_ = scs[s]
                        cm = cm_from_counts(*sc_.counts(0.5, w))
                        out.append([sc_.auc(w), cm["accuracy"], cm["macro_f1"], cm["recall"], cm["precision"], cm["specificity"], float(np.sum(w * sc_.p) / np.sum(w))])
                    return np.mean(out, axis=0)
                pt = vec(np.ones(n))
                bt = np.array([vec(draw_weights(n)) for _ in range(PERF_BOOT)])
                lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
                for j, mn in enumerate(["auc", "accuracy", "macro_f1", "recall", "precision", "specificity", "mean_predicted"]):
                    rows.append({"group": grp, "classifier": clf, "subset": sname, "n": n, "observed_prevalence": float(y_ex[m].mean()), "metric": mn,
                                 "estimate": pt[j], "ci_low": lo[j], "ci_high": hi[j], "n_splits": len(ss)})
    if rows:
        pd.DataFrame(rows).round(5).to_csv(os.path.join(AN, "T17_excluded_records.csv"), index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ run
if not SKIP_ABLATION:
    run_ablation()
Sab = ablation_tables()
say("ablation tables done" if Sab is not None else "no ablation results yet")
Cb = calibration_table()
say("T11 calibration written")
Dc = decision_curve_table()
say("T12 decision curves written")
Op = operating_points_table()
say("T13 operating points written")
Ty = temporal_by_year_table()
say("T14 temporal by year written" if Ty is not None else "no temporal results yet")
Pc = primary_and_comparison_tables()
say("T15 primary measures with CIs and T16 classifier comparison written")
Hx = excluded_records_table()
say("T17 excluded-records check written" if Hx is not None and len(Hx) else "no excluded-records results")

L = ["EXTRA ANALYSES (Dengue = positive class)", ""]
if Sab is not None:
    for clf in [c for c in BOOT_CLFS if c in set(Sab.classifier)]:
        s = Sab[Sab.classifier == clf].set_index("fieldset")
        L.append(f"A. {clf}: AUC (mean over random splits) " + " | ".join(f"{n} {s.loc[n, 'auc_mean']:.4f}" for n in s.index if "auc_mean" in s.columns and pd.notna(s.loc[n, "auc_mean"])))
if len(Cb):
    r = Cb[Cb.split.str.startswith("seed")].groupby(["classifier", "arm"])[["calibration_slope", "citl"]].mean()
    for clf in [c for c in BOOT_CLFS if c in CLFS]:
        L.append(f"B. {clf}: calibration slope / CITL by arm " + " | ".join(f"{a} {r.loc[(clf, a), 'calibration_slope']:.2f} / {r.loc[(clf, a), 'citl']:+.2f}" for a in ARMS if (clf, a) in r.index))
with open(os.path.join(AN, "EXTRA_SUMMARY.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
say("\n" + "\n".join(L))
say(f"done -> {AN}")
