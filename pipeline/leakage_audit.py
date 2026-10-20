"""
leakage_audit.py -- how much did the leaks in the ORIGINAL notebooks inflate the results?  (controlled experiment, current data)

The original per-classifier notebooks (1) target-encoded Ward on the WHOLE dataset, train AND test labels together
(encoded_dataset.csv), and (2) oversampled the WHOLE dataset (SMOTE / Borderline-SMOTE / ADASYN) BEFORE the 80/20 split
(Imbalance_*.ipynb -> *_balanced_dataset.csv -> classifier notebooks, which only then call train_test_split).
This script reproduces those workflows on today's data and compares them with the correct pipeline.

  CORRECT               split -> ward encoding fitted on the training part only -> oversample the training part only -> test on real cases
                        (taken from the main run, run_revised.py)
  WARD_LEAK             ward encoded with ALL labels -> split -> oversample the training part only -> test on real cases
  ORIGINAL_REAL_CASES   ward encoded with ALL labels -> oversample ALL data -> split -> scored on the REAL cases of the test part
                        (natural class mix; comparable with CORRECT; this isolates the leakage)
  ORIGINAL_AS_REPORTED  same model, scored on the whole test part (about half synthetic, 50/50 classes) - what the old notebooks reported;
                        NOT comparable with real-world performance

Same classifiers, hyperparameters (from the main run) and random splits throughout.
Run AFTER run_revised.py has finished at least seed 1.     python leakage_audit.py
Settings: AUDIT_SEEDS=1,2,3   AUDIT_CLASSIFIERS=DecisionTree,RandomForest,XGBoost,LogisticRegression
Writes  <outputs>/analysis/leakage_audit_long.csv, leakage_audit_summary.csv, leakage_audit.txt
"""
import os
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import run_revised as R

AUDIT_SEEDS = [int(s) for s in os.environ.get("AUDIT_SEEDS", "1" if R.QUICK else "1,2,3").split(",") if s.strip()]
AUDIT_CLFS = [c for c in os.environ.get("AUDIT_CLASSIFIERS", "DecisionTree,RandomForest,XGBoost,LogisticRegression").split(",") if c.strip()]
ARMS = ["Baseline", "SMOTE", "BorderlineSMOTE", "ADASYN"]
AN = os.path.join(R.OUT, "analysis")
os.makedirs(AN, exist_ok=True)
T0 = time.time()


def say(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    f1d = 2 * tp / max(2 * tp + fp + fn, 1)
    f1m = 2 * tn / max(2 * tn + fn + fp, 1)
    return {"accuracy": (tp + tn) / len(y), "macro_f1": (f1d + f1m) / 2, "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1), "auc": float(roc_auc_score(y, p))}


df = pd.read_csv(R.DATA)
if R.SUBSAMPLE < 1.0:
    df = df.sample(frac=R.SUBSAMPLE, random_state=0).sort_values("ROW_ID")
df = df.reset_index(drop=True)
y_all = df["DENGUE"].to_numpy()

# ward encoding that uses EVERY label (the original, leaky way): smoothed dengue rate per ward, smoothing 10
agg = df.groupby("WARD")["DENGUE"].agg(["mean", "count"])
te = (agg["mean"] * agg["count"] + y_all.mean() * 10.0) / (agg["count"] + 10.0)
leak = df[R.RAWCOLS].copy()
leak["WARD_TE_LEAK"] = df["WARD"].map(te).to_numpy()
prep = ColumnTransformer([
    ("ward_te", StandardScaler(), ["WARD_TE_LEAK"]),
    ("month", R.MonthCyclicEncoder(), ["MONTH"]),
    ("age", StandardScaler(), ["AGE"]),
    ("sex", R.BinaryMapEncoder({"F": 0, "M": 1}, "SEX"), ["SEX"]),
    ("place", R.BinaryMapEncoder({"HOME TREATMENT": 0, "HOSPITAL TREATMENT": 1}, "PLACE_OF_TREATMENT"), ["PLACE OF TREATMENT"]),
    ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["BOROUGH"]),
])
X_leak = prep.fit_transform(leak)
say(f"data {len(df)} rows | leaky matrix {X_leak.shape} | seeds {AUDIT_SEEDS} | classifiers {AUDIT_CLFS}")

rows = []
for seed in AUDIT_SEEDS:
    tr, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y_all)
    for clf in AUDIT_CLFS:
        mp = os.path.join(R.OUT, "meta", f"seed{seed}__{clf}.json")
        if not os.path.exists(mp):
            say(f"skip seed {seed} {clf}: main run not finished for it")
            continue
        with open(mp, encoding="utf-8") as f:
            params = json.load(f)["params"]
        pr = pd.read_csv(os.path.join(R.OUT, "preds", f"seed{seed}__{clf}.csv.gz"))
        yt = df.set_index("ROW_ID").loc[pr["ROW_ID"].to_numpy(), "DENGUE"].to_numpy()
        for arm in ARMS:
            rows.append({"scenario": "CORRECT", "seed": seed, "classifier": clf, "arm": arm,
                         **metrics(yt, pr["p_" + arm].to_numpy(float))})
        spw = float((y_all[tr] == 0).sum() / max((y_all[tr] == 1).sum(), 1))
        for arm in ARMS:
            ta = time.time()
            # WARD_LEAK: leaky ward feature, split first, oversample the training part only
            Xtr, ytr = X_leak[tr], y_all[tr]
            if arm != "Baseline":
                Xtr, ytr = R.RESAMPLERS[arm](seed).fit_resample(Xtr, ytr)
            est = R.make_est(clf, params, seed, spw=spw).fit(Xtr, ytr)
            rows.append({"scenario": "WARD_LEAK", "seed": seed, "classifier": clf, "arm": arm,
                         **metrics(y_all[te_idx], est.predict_proba(X_leak[te_idx])[:, 1])})
            # ORIGINAL workflow: leaky ward feature, oversample ALL data, then split. Two ways of scoring the same model:
            #   ORIGINAL_AS_REPORTED   on the whole test part (about half synthetic, 50/50 classes) - what the old notebooks reported
            #   ORIGINAL_REAL_CASES    on the real cases of the test part only (natural class mix) - comparable with CORRECT
            if arm == "Baseline":
                rows.append({**rows[-1], "scenario": "ORIGINAL_AS_REPORTED"})
                rows.append({**rows[-1], "scenario": "ORIGINAL_REAL_CASES"})
            else:
                Xr, yr = R.RESAMPLERS[arm](seed).fit_resample(X_leak, y_all)   # imblearn keeps the real rows first
                a, b = train_test_split(np.arange(len(yr)), test_size=0.2, random_state=seed, stratify=yr)
                est = R.make_est(clf, params, seed, spw=spw).fit(Xr[a], yr[a])
                pb = est.predict_proba(Xr[b])[:, 1]
                real = b < len(y_all)
                rows.append({"scenario": "ORIGINAL_AS_REPORTED", "seed": seed, "classifier": clf, "arm": arm, **metrics(yr[b], pb)})
                rows.append({"scenario": "ORIGINAL_REAL_CASES", "seed": seed, "classifier": clf, "arm": arm, **metrics(yr[b][real], pb[real])})
            say(f"seed {seed} | {clf:18s} | {arm:16s} | {time.time() - ta:5.0f}s")

L = pd.DataFrame(rows)
L.round(5).to_csv(os.path.join(AN, "leakage_audit_long.csv"), index=False)
g = L.groupby(["classifier", "arm", "scenario"])[["accuracy", "macro_f1", "recall", "precision", "auc"]]
S = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
S.round(4).to_csv(os.path.join(AN, "leakage_audit_summary.csv"), index=False)

lines = ["LEAKAGE AUDIT (same data, same hyperparameters; mean over seeds " + ",".join(map(str, AUDIT_SEEDS)) + ")", ""]
for clf in AUDIT_CLFS:
    s = S[S.classifier == clf]
    if s.empty:
        continue
    lines.append(clf)
    for arm in ARMS:
        x = s[s.arm == arm].set_index("scenario")
        order = ["CORRECT", "WARD_LEAK", "ORIGINAL_REAL_CASES", "ORIGINAL_AS_REPORTED"]
        if set(order) <= set(x.index):
            lines.append(f"  {arm:16s} accuracy " + " | ".join(f"{k[:14]} {x.loc[k, 'accuracy_mean']:.4f}" for k in order)
                         + "   recall " + " / ".join(f"{x.loc[k, 'recall_mean']:.3f}" for k in order))
with open(os.path.join(AN, "leakage_audit.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
say("\n" + "\n".join(lines))
