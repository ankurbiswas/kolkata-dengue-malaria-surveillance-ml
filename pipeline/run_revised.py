"""
run_revised.py -- revised leakage-safe benchmark (v2), designed to run on a laptop.

WHAT CHANGED COMPARED WITH THE FIRST PIPELINE (script 05)
  1. Feature importance (SHAP / permutation) is computed from TRAINING data only. The number of fields kept is chosen by
     cross-validation on the training data, and the test set is used once.
  2. Selection works on FIELDS (Ward, Month, Age, Sex, Treatment place, Borough), not on the 22 encoded columns.
  3. Every test-case prediction is saved, so confidence intervals, significance tests, calibration, subgroup analysis and
     the ward-level validation map can be produced afterwards (analyze_results.py).
  4. New arm "ClassWeight". Every arm is also evaluated at a decision threshold tuned on a validation slice of the training data.
  5. Five random splits plus one temporal split (train 2014-2022, test 2023-2025).
  6. The ward encoding is cross-fitted inside the training data (sklearn TargetEncoder, cv=5, smoothing 10).
  7. Hyperparameters are tuned once per split on the baseline arm (random search, 3-fold CV, ROC-AUC) and shared by all arms.
  8. Target: DENGUE = 1, MALARIA = 0, so every probability in the outputs is P(Dengue).

SETTINGS (environment variables, all optional)
  SEEDS=1,2,3,4,5   TEMPORAL=1   CLASSIFIERS=DecisionTree,RandomForest,XGBoost,LogisticRegression,KNN,MLP,GaussianNB
  THREADS=6 (default: logical cores - 2)   QUICK=1 (smoke test on a 6% sample)   OUT=<folder>
The run is resumable: finished (split, classifier) units are skipped. Create a file called STOP in the output folder to stop
cleanly after the current unit.
"""
import os
import sys
import json
import math
import time
import warnings
import platform

THREADS = int(os.environ.get("THREADS", max(1, (os.cpu_count() or 4) - 2)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS)
warnings.filterwarnings("ignore")

if os.name == "nt":
    try:
        import ctypes
        if os.environ.get("LOW_PRIORITY", "1") == "1":   # keep the laptop responsive while this runs
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        # stop Windows going to sleep while the run is active (reverts automatically when the script ends)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except Exception:
        pass

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from xgboost import XGBClassifier
import shap

# ------------------------------------------------------------------ settings
QUICK = os.environ.get("QUICK", "0") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT") or os.path.join(HERE, "outputs_quick" if QUICK else "outputs_v2")
DATA = os.environ.get("DATA") or os.path.join(HERE, "model_input_v2.csv")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1" if QUICK else "1,2,3,4,5").split(",") if s.strip()]
TEMPORAL = os.environ.get("TEMPORAL", "1") == "1"
ALL_CLASSIFIERS = ["DecisionTree", "RandomForest", "XGBoost", "LogisticRegression", "KNN", "MLP", "GaussianNB"]
CLASSIFIERS = [c for c in os.environ.get("CLASSIFIERS", ",".join(ALL_CLASSIFIERS)).split(",") if c.strip()]
SUBSAMPLE = float(os.environ.get("SUBSAMPLE", "0.06" if QUICK else "1.0"))
SHAP_N = int(os.environ.get("SHAP_N", "400" if QUICK else "3000"))
PERM_N = int(os.environ.get("PERM_N", "2000" if QUICK else "10000"))
PERM_REPEATS = int(os.environ.get("PERM_REPEATS", "2" if QUICK else "3"))
CV_FOLDS = int(os.environ.get("CV_FOLDS", "3"))
TIE_TOLERANCE = 0.005          # a smaller set is accepted if its CV Macro-F1 is within this of the best
ARMS = ["Baseline", "ClassWeight", "SMOTE", "BorderlineSMOTE", "ADASYN"]
FIELDS = ["WARD", "MONTH", "AGE", "SEX", "PLACE", "BOROUGH"]
RAWCOL = {"WARD": "WARD", "MONTH": "MONTH", "AGE": "AGE", "SEX": "SEX", "PLACE": "PLACE OF TREATMENT", "BOROUGH": "BOROUGH"}
RAWCOLS = [RAWCOL[f] for f in FIELDS]
PREFIX_TO_FIELD = {"ward_te": "WARD", "month": "MONTH", "age": "AGE", "sex": "SEX", "place": "PLACE", "borough": "BOROUGH"}
CW_SUPPORTED = {"DecisionTree", "RandomForest", "XGBoost", "LogisticRegression", "GaussianNB"}
SHAP_MODELS = {"DecisionTree", "RandomForest", "XGBoost", "LogisticRegression"}
THR_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)

SPACE = {
    "DecisionTree": ({"max_depth": [5, 8, 10, 15, 20, None], "min_samples_leaf": [1, 2, 5, 10, 20], "criterion": ["gini", "entropy"]}, 8),
    # depth is capped at 15: deeper trees add nothing here (only ~6 fields) and make exact SHAP values extremely slow
    "RandomForest": ({"max_depth": [8, 10, 12, 15], "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.5]}, 6),
    "XGBoost": ({"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7, 9], "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.8, 1.0]}, 8),
    "LogisticRegression": ({"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}, 6),
    "KNN": ({"n_neighbors": [5, 15, 31]}, 3),
    "MLP": ({"hidden_layer_sizes": [(50,), (100,), (64, 32)], "alpha": [1e-4, 1e-3, 1e-2]}, 4),
    "GaussianNB": ({"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}, 7),
}
if QUICK:
    SPACE = {k: (v[0], min(v[1], 2)) for k, v in SPACE.items()}

T_START = time.time()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T_START) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(OUT, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ------------------------------------------------------------------ preprocessing
class MonthCyclicEncoder(BaseEstimator, TransformerMixin):
    MONTHS = {m: i + 1 for i, m in enumerate(["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
                                              "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"])}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        m = pd.Series(np.asarray(X).ravel()).map(self.MONTHS).astype(float)
        return np.column_stack([np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])

    def get_feature_names_out(self, input_features=None):
        return np.array(["MONTH_sin", "MONTH_cos"])


class BinaryMapEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping, output_name="FEATURE"):
        self.mapping = mapping
        self.output_name = output_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(np.asarray(X).ravel()).map(self.mapping).astype(float).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([self.output_name])


def build_prep(fields, seed):
    t = []
    if "WARD" in fields:
        t.append(("ward_te", SkPipeline([("te", TargetEncoder(target_type="binary", smooth=10.0, cv=5, shuffle=True, random_state=seed)),
                                         ("sc", StandardScaler())]), ["WARD"]))
    if "MONTH" in fields:
        t.append(("month", MonthCyclicEncoder(), ["MONTH"]))
    if "AGE" in fields:
        t.append(("age", StandardScaler(), ["AGE"]))
    if "SEX" in fields:
        t.append(("sex", BinaryMapEncoder({"F": 0, "M": 1}, "SEX"), ["SEX"]))
    if "PLACE" in fields:
        t.append(("place", BinaryMapEncoder({"HOME TREATMENT": 0, "HOSPITAL TREATMENT": 1}, "PLACE_OF_TREATMENT"), ["PLACE OF TREATMENT"]))
    if "BOROUGH" in fields:
        t.append(("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["BOROUGH"]))
    return ColumnTransformer(t)


def column_field(colname):
    return PREFIX_TO_FIELD[colname.split("__")[0]]


# ------------------------------------------------------------------ models
def make_est(name, params, seed, cw=False, spw=1.0):
    p = dict(params)
    if "hidden_layer_sizes" in p:
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    bal = "balanced" if cw else None
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=seed, class_weight=bal, **p)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, n_jobs=THREADS, random_state=seed, class_weight=bal, **p)
    if name == "XGBoost":
        return XGBClassifier(tree_method="hist", n_jobs=THREADS, random_state=seed, eval_metric="logloss",
                             scale_pos_weight=spw if cw else 1.0, **p)
    if name == "LogisticRegression":
        return LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed, class_weight=bal, **p)
    if name == "KNN":
        return KNeighborsClassifier(n_jobs=THREADS, **p)
    if name == "MLP":
        return MLPClassifier(max_iter=300, early_stopping=True, random_state=seed, **p)
    if name == "GaussianNB":
        return GaussianNB(priors=[0.5, 0.5] if cw else None, **p)
    raise ValueError(name)


RESAMPLERS = {"SMOTE": lambda s: SMOTE(random_state=s), "BorderlineSMOTE": lambda s: BorderlineSMOTE(random_state=s),
              "ADASYN": lambda s: ADASYN(random_state=s)}


def make_pipe(name, params, fields, seed, arm, spw):
    est = make_est(name, params, seed, cw=(arm == "ClassWeight"), spw=spw)
    steps = [("prep", build_prep(fields, seed))]
    if arm in RESAMPLERS:
        return ImbPipeline(steps + [("res", RESAMPLERS[arm](seed)), ("clf", est)])
    return SkPipeline(steps + [("clf", est)])


def macro_f1_at(y, p, t):
    return float(f1_score(y, (p >= t).astype(int), average="macro"))


def best_threshold(y, p):
    scores = [macro_f1_at(y, p, t) for t in THR_GRID]
    i = int(np.argmax(scores))
    return float(THR_GRID[i]), float(scores[i])


def test_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"test_macro_f1": macro_f1_at(y, p, 0.5), "test_auc": float(roc_auc_score(y, p)),
            "test_dengue_recall": tp / max(tp + fn, 1), "test_dengue_precision": tp / max(tp + fp, 1)}


# ------------------------------------------------------------------ stages
def tune(name, Xtr, ytr, seed):
    space, n_iter = SPACE[name]
    n_iter = min(n_iter, math.prod(len(v) for v in space.values()))
    configs = list(ParameterSampler(space, n_iter=n_iter, random_state=seed))
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for a, b in skf.split(Xtr, ytr):
        prep = build_prep(FIELDS, seed).fit(Xtr.iloc[a], ytr[a])
        folds.append((prep.transform(Xtr.iloc[a]), ytr[a], prep.transform(Xtr.iloc[b]), ytr[b]))
    rows = []
    for cfg in configs:
        sc = []
        for Xa, ya, Xb, yb in folds:
            est = make_est(name, cfg, seed)
            est.fit(Xa, ya)
            sc.append(roc_auc_score(yb, est.predict_proba(Xb)[:, 1]))
        rows.append({"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                     "cv_auc": float(np.mean(sc)), "cv_auc_sd": float(np.std(sc))})
    best = max(rows, key=lambda r: r["cv_auc"])
    return best["params"], rows


def shap_field_share(pipe, Xtr, name, seed):
    prep, est = pipe.named_steps["prep"], pipe.named_steps["clf"]
    rs = np.random.RandomState(seed)
    n_rows = min(1500 if name == "RandomForest" else SHAP_N, len(Xtr))
    idx = rs.choice(len(Xtr), n_rows, replace=False)
    Xs = prep.transform(Xtr.iloc[idx])
    cols = list(prep.get_feature_names_out())
    if name == "LogisticRegression":
        sv = shap.LinearExplainer(est, Xs).shap_values(Xs)
    else:
        sv = shap.TreeExplainer(est).shap_values(Xs, check_additivity=False)
    sv = np.asarray(sv[1]) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = np.abs(sv).mean(axis=0)
    fld = np.array([column_field(c) for c in cols])
    share = {f: float(imp[fld == f].sum()) for f in FIELDS}
    tot = sum(share.values()) or 1.0
    return {f: share[f] / tot * 100 for f in FIELDS}


def permutation_field_importance(pipe_val, Xval, yval, seed):
    rng = np.random.RandomState(seed)
    if len(Xval) > PERM_N:
        sel = rng.choice(len(Xval), PERM_N, replace=False)
        Xval, yval = Xval.iloc[sel].reset_index(drop=True), yval[sel]
    base = roc_auc_score(yval, pipe_val.predict_proba(Xval)[:, 1])
    groups = {f: [RAWCOL[f]] for f in FIELDS}
    groups["SPATIAL"] = [RAWCOL["WARD"], RAWCOL["BOROUGH"]]
    out = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(PERM_REPEATS):
            perm = rng.permutation(len(Xval))
            Xp = Xval.copy()
            Xp[cols] = Xval[cols].iloc[perm].to_numpy()
            drops.append(base - roc_auc_score(yval, pipe_val.predict_proba(Xp)[:, 1]))
        out[g] = float(np.mean(drops))
    return out


def run_unit(split, tr, te, seed, name, df):
    t0 = time.time()
    X_all, y_all = df[RAWCOLS], df["DENGUE"].to_numpy()
    Xtr, ytr = X_all.iloc[tr].reset_index(drop=True), y_all[tr]
    Xte, yte = X_all.iloc[te].reset_index(drop=True), y_all[te]
    stage = {}

    t = time.time()
    params, tune_rows = tune(name, Xtr, ytr, seed)
    stage["tune_s"] = time.time() - t
    log(f"   tuned ({stage['tune_s']:.0f}s): {params}")

    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    fit_i, val_i = train_test_split(np.arange(len(ytr)), test_size=0.25, random_state=seed + 100, stratify=ytr)
    yval = ytr[val_i]
    preds, thresholds, val_f1 = {}, {}, {}
    base_pipe = base_val_pipe = None
    t = time.time()
    for arm in ARMS:
        if arm == "ClassWeight" and name not in CW_SUPPORTED:
            continue
        ta = time.time()
        try:
            pipe = make_pipe(name, params, FIELDS, seed, arm, spw).fit(Xtr, ytr)
            p_te = pipe.predict_proba(Xte)[:, 1]
            pipe_v = make_pipe(name, params, FIELDS, seed, arm, spw).fit(Xtr.iloc[fit_i], ytr[fit_i])
            p_val = pipe_v.predict_proba(Xtr.iloc[val_i])[:, 1]
        except Exception as e:
            log(f"   !! arm {arm} failed: {type(e).__name__}: {str(e)[:150]}")
            continue
        thr, f1v = best_threshold(yval, p_val)
        preds["p_" + arm], thresholds[arm], val_f1[arm] = p_te, thr, f1v
        if arm == "Baseline":
            base_pipe, base_val_pipe = pipe, pipe_v
        log(f"   arm {arm:16s} {time.time() - ta:6.0f}s | test AUC {roc_auc_score(yte, p_te):.4f} | tuned threshold {thr:.2f}")
    stage["arms_s"] = time.time() - t
    if base_pipe is None:
        raise RuntimeError("baseline arm failed")

    t = time.time()
    shap_share = None
    if name in SHAP_MODELS:
        shap_share = shap_field_share(base_pipe, Xtr, name, seed)
    perm = permutation_field_importance(base_val_pipe, Xtr.iloc[val_i].reset_index(drop=True), yval, seed)
    stage["importance_s"] = time.time() - t
    if shap_share is not None:
        score, method = shap_share, "SHAP"
    else:
        score, method = {f: perm[f] for f in FIELDS}, "permutation"
    ranking = sorted(FIELDS, key=lambda f: (-score[f], f))
    imp_rows = [{"split": split, "classifier": name, "field": f, "shap_share_pct": None if shap_share is None else shap_share[f],
                 "perm_auc_drop": perm[f], "rank_used": ranking.index(f) + 1, "rank_method": method} for f in FIELDS]
    imp_rows.append({"split": split, "classifier": name, "field": "SPATIAL(WARD+BOROUGH)", "shap_share_pct": None,
                     "perm_auc_drop": perm["SPATIAL"], "rank_used": None, "rank_method": method})
    log(f"   importance ({stage['importance_s']:.0f}s, {method}): ranking {ranking}")

    t = time.time()
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    sweep_rows = []
    for k in range(1, len(FIELDS) + 1):
        fk = ranking[:k]
        cvf, cva = [], []
        for a, b in skf.split(Xtr, ytr):
            pk = make_pipe(name, params, fk, seed, "Baseline", spw).fit(Xtr.iloc[a], ytr[a])
            p = pk.predict_proba(Xtr.iloc[b])[:, 1]
            cvf.append(macro_f1_at(ytr[b], p, 0.5))
            cva.append(roc_auc_score(ytr[b], p))
        if k == len(FIELDS):
            p_te = preds["p_Baseline"]
        else:
            p_te = make_pipe(name, params, fk, seed, "Baseline", spw).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            preds[f"p_k{k}"] = p_te
        row = {"split": split, "classifier": name, "rank_method": method, "k": k, "fields": "+".join(fk),
               "cv_macro_f1": float(np.mean(cvf)), "cv_auc": float(np.mean(cva))}
        row.update(test_metrics(yte, p_te))
        sweep_rows.append(row)
    best_cv = max(r["cv_macro_f1"] for r in sweep_rows)
    k_star = min(r["k"] for r in sweep_rows if r["cv_macro_f1"] >= best_cv - TIE_TOLERANCE)
    for r in sweep_rows:
        r["selected"] = int(r["k"] == k_star)
    stage["sweep_s"] = time.time() - t
    log(f"   sweep ({stage['sweep_s']:.0f}s): chosen k={k_star} -> {sweep_rows[k_star - 1]['fields']} "
        f"(CV F1 {sweep_rows[k_star - 1]['cv_macro_f1']:.4f} vs full {sweep_rows[-1]['cv_macro_f1']:.4f})")

    key = f"{split}__{name}"
    out = pd.DataFrame({"ROW_ID": df["ROW_ID"].iloc[te].to_numpy()})
    for c, v in preds.items():
        out[c] = np.asarray(v, dtype="float32")
    out.to_csv(os.path.join(OUT, "preds", key + ".csv.gz"), index=False, compression="gzip", float_format="%.5f")
    pd.DataFrame(imp_rows).to_csv(os.path.join(OUT, "importance", key + ".csv"), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(OUT, "sweep", key + ".csv"), index=False)
    meta = {"split": split, "classifier": name, "seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)),
            "test_dengue_prevalence": float(yte.mean()), "params": params, "tuning": tune_rows, "thresholds": thresholds,
            "val_macro_f1_at_threshold": val_f1, "ranking": ranking, "rank_method": method, "selected_k": int(k_star),
            "scale_pos_weight": spw, "timing_s": {**stage, "total_s": time.time() - t0}}
    with open(os.path.join(OUT, "meta", key + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return time.time() - t0


def main():
    for d in ("preds", "importance", "sweep", "meta"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    df = pd.read_csv(DATA)
    if SUBSAMPLE < 1.0:
        df = df.sample(frac=SUBSAMPLE, random_state=0).sort_values("ROW_ID")
    df = df.reset_index(drop=True)
    log(f"data: {len(df)} rows | dengue {int(df.DENGUE.sum())} | output {OUT} | threads {THREADS} | quick={QUICK}")
    splits = {}
    for i, s in enumerate(SEEDS):
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=s, stratify=df["DENGUE"])
        splits[f"seed{s}"] = (tr, te, s)
        if TEMPORAL and i == 0:   # the temporal split runs second, so a complete picture exists early
            splits["temporal"] = (np.where(df.YEAR <= 2022)[0], np.where(df.YEAR >= 2023)[0], 0)
    if TEMPORAL and "temporal" not in splits:
        splits["temporal"] = (np.where(df.YEAR <= 2022)[0], np.where(df.YEAR >= 2023)[0], 0)
    np.savez_compressed(os.path.join(OUT, "splits.npz"), **{f"{k}_{part}": v[i] for k, v in splits.items() for i, part in ((0, "train"), (1, "test"))})
    with open(os.path.join(OUT, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"python": sys.version.split()[0], "platform": platform.platform(), "threads": THREADS, "quick": QUICK,
                   "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "shap": shap.__version__,
                   "xgboost": sys.modules["xgboost"].__version__, "seeds": SEEDS, "temporal": TEMPORAL, "classifiers": CLASSIFIERS,
                   "rows": int(len(df)), "arms": ARMS, "tie_tolerance": TIE_TOLERANCE}, f, indent=1)
    units = [(sp, c) for sp in splits for c in CLASSIFIERS]
    done = 0
    for i, (sp, c) in enumerate(units, 1):
        if os.path.exists(os.path.join(OUT, "meta", f"{sp}__{c}.json")):
            done += 1
            continue
        if os.path.exists(os.path.join(OUT, "STOP")):
            log("STOP file found - stopping cleanly. Delete it to continue later.")
            return
        tr, te, seed = splits[sp]
        log(f"== unit {i}/{len(units)}: split {sp}, {c} (train {len(tr)}, test {len(te)})")
        try:
            secs = run_unit(sp, tr, te, seed, c, df)
        except Exception as e:
            log(f"!! unit failed: {type(e).__name__}: {str(e)[:300]} (it will be retried on the next run)")
            continue
        done += 1
        log(f"== finished {sp} {c} in {secs / 60:.1f} min | {done}/{len(units)} units done")
        with open(os.path.join(OUT, "progress.csv"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{sp},{c},{secs:.0f}\n")
    log(f"ALL DONE: {done}/{len(units)} units. Results are in {OUT}. Next: python analyze_results.py")


if __name__ == "__main__":
    main()
