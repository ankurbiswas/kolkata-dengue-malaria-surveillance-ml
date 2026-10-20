"""
03_feature_engineering.py

Builds the encoded feature matrix from cleaned_dataset.csv, mirroring the
encoding architecture described in Ankur's prior progress presentation:

  - Cyclic Temporal Encoding (Month): sin/cos trig mapping onto a circle,
    so December and January sit next to each other instead of far apart.
  - High-Cardinality Target Encoding (Ward): each of the 144 wards is
    mapped to a continuous risk score = its historical Dengue rate, using
    5-fold cross-validated encoding so no row leaks its own label into its
    own encoded value.
  - Spatial Indicator Recoding (Borough): 16 one-hot flags (Borough I-XVI).
  - Continuous Scaling (Age): robust (median/IQR) Z-score, resistant to the
    small number of remaining age outliers.
  - Treatment_Type -> binary Treatment_Hospital flag (1=Hospital, 0=Home).

Target: Disease_Label (1 = Dengue, 0 = Malaria) - matches the minority/
majority framing used throughout the presentation (Dengue is the minority,
harder-to-catch class).

NOTE: rows with missing Age/Ward/Borough/Sex/Treatment propagate as NaN in
the engineered columns - this is intentional. Imputation / row-exclusion
policy is a modeling-stage decision, not a feature-engineering one, and is
left for the next checkpoint.

Run: python3 03_feature_engineering.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(HERE, "..", "data")   # not included in the repository (see DATA.md)
IN_PATH = os.path.join(DATA_DIR, "cleaned_dataset.csv")
OUT_PATH = os.path.join(DATA_DIR, "engineered_dataset.csv")
REPORT_PATH = os.path.join(DATA_DIR, "feature_engineering_report.txt")

MONTH_TO_NUM = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "MAY": 5, "JUNE": 6,
    "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
}
BOROUGHS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII",
            "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]

RANDOM_SEED = 42
N_FOLDS = 5

log_lines = []


def log(msg):
    log_lines.append(str(msg))


def kfold_target_encode(df, group_col, target_col, n_splits=N_FOLDS, seed=RANDOM_SEED, smoothing=10):
    """
    Leakage-free target encoding: for each fold, the encoding for that fold's
    rows is computed only from the OTHER folds. Rows with a missing group_col
    value get NaN. A global (all-data) mapping is also returned for reference
    / for encoding any future out-of-sample ward.
    """
    encoded = pd.Series(np.nan, index=df.index, dtype=float)
    global_mean = df[target_col].mean()

    valid_mask = df[group_col].notna()
    valid_idx = df.index[valid_mask]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx_pos, holdout_idx_pos in kf.split(valid_idx):
        train_idx = valid_idx[train_idx_pos]
        holdout_idx = valid_idx[holdout_idx_pos]

        stats = df.loc[train_idx].groupby(group_col)[target_col].agg(["mean", "count"])
        # smoothing toward the global mean for wards with few training-fold observations
        smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)

        mapped = df.loc[holdout_idx, group_col].map(smoothed)
        encoded.loc[holdout_idx] = mapped.fillna(global_mean)

    # full-data mapping (for documentation / future inference on new rows)
    full_stats = df.loc[valid_idx].groupby(group_col)[target_col].agg(["mean", "count"])
    full_smoothed = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (full_stats["count"] + smoothing)

    return encoded, full_smoothed, global_mean


def main():
    df = pd.read_csv(IN_PATH, low_memory=False)
    n0 = len(df)
    log(f"Loaded cleaned_dataset.csv: {n0} rows")

    # --- Target ---
    df["Disease_Label"] = (df["Disease"] == "Dengue").astype(int)
    log(f"\nDisease_Label distribution (1=Dengue, 0=Malaria):")
    log(df["Disease_Label"].value_counts().to_string())
    log(f"Class ratio Malaria:Dengue = {(df['Disease_Label']==0).sum() / max((df['Disease_Label']==1).sum(),1):.2f} : 1")

    # --- Cyclic month encoding ---
    month_num = df["Month"].map(MONTH_TO_NUM)
    df["Month_sin"] = np.sin(2 * np.pi * month_num / 12)
    df["Month_cos"] = np.cos(2 * np.pi * month_num / 12)
    log(f"\nMonth cyclic encoding: {month_num.isna().sum()} rows have no resolvable month (NaN sin/cos)")

    # --- Ward target encoding (5-fold CV, target = Disease_Label) ---
    encoded, full_mapping, global_mean = kfold_target_encode(df, "Ward", "Disease_Label")
    df["Ward_TargetEnc"] = encoded
    log(f"\nWard target encoding: global Dengue rate = {global_mean:.4f}")
    log(f"Wards encoded: {full_mapping.shape[0]} (expect <= 144)")
    log("Top 10 highest-risk wards (full-data smoothed Dengue rate):")
    log(full_mapping.sort_values(ascending=False).head(10).to_string())
    log("Bottom 10 lowest-risk wards:")
    log(full_mapping.sort_values(ascending=True).head(10).to_string())
    full_mapping.rename("Dengue_Rate_Smoothed").to_csv(
        os.path.join(DATA_DIR, "ward_target_encoding_map.csv"))

    # --- Borough one-hot ---
    borough_dummies = pd.get_dummies(df["Borough"], prefix="Borough")
    for b in BOROUGHS:
        col = f"Borough_{b}"
        if col not in borough_dummies.columns:
            borough_dummies[col] = 0
    borough_dummies = borough_dummies[[f"Borough_{b}" for b in BOROUGHS]]
    df = pd.concat([df, borough_dummies], axis=1)
    log(f"\nBorough one-hot: {borough_dummies.shape[1]} flags created; "
        f"{df['Borough'].isna().sum()} rows have all-zero flags (missing borough)")

    # --- Age robust scaling ---
    scaler = RobustScaler()
    age_vals = df[["Age"]].values
    mask = ~np.isnan(age_vals).ravel()
    scaled = np.full(age_vals.shape[0], np.nan)
    scaled[mask] = scaler.fit_transform(age_vals[mask].reshape(-1, 1)).ravel()
    df["Age_Scaled"] = scaled
    log(f"\nAge robust scaling: center(median)={scaler.center_[0]:.2f}, "
        f"scale(IQR)={scaler.scale_[0]:.2f}; {(~mask).sum()} rows left NaN (missing age)")

    # --- Treatment binary flag ---
    df["Treatment_Hospital"] = df["Treatment_Type"].map({"Home": 0, "Hospital": 1})
    log(f"\nTreatment_Hospital: {df['Treatment_Hospital'].isna().sum()} rows NaN (missing treatment info)")

    feature_cols = [
        "Date", "Year", "Disease", "Disease_Label",
        "Month", "Month_sin", "Month_cos",
        "Age", "Age_Scaled",
        "Sex",
        "Ward", "Ward_TargetEnc",
        "Borough"] + [f"Borough_{b}" for b in BOROUGHS] + [
        "Treatment_Type", "Treatment_Hospital",
    ]
    out_df = df[feature_cols].copy()
    out_df.to_csv(OUT_PATH, index=False)

    log(f"\nFinal engineered dataset: {out_df.shape[0]} rows x {out_df.shape[1]} columns -> {OUT_PATH}")
    log("\nComplete-case count (no NaN in any of Month/Age/Sex/Ward/Borough/Treatment):")
    core_cols = ["Month_sin", "Age_Scaled", "Sex", "Ward_TargetEnc", "Borough", "Treatment_Hospital"]
    complete = out_df[core_cols].notna().all(axis=1)
    log(f"  {complete.sum()} / {len(out_df)} rows ({complete.mean()*100:.1f}%) are complete across all 6 target features")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(log_lines))
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
