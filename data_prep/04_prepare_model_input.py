"""
04_prepare_model_input.py

Maps our Phase 1 cleaned_dataset.csv onto the exact schema Ankur's original
"gold-standard" leakage-safe notebook expects: MONTH, AGE, SEX, WARD,
BOROUGH, PLACE OF TREATMENT, CLASS. Complete-case rows only (all 6 feature
fields present) -- the classifiers used downstream (KNN, LR, MLP,
GaussianNB) can't handle NaN natively.

CLASS convention matches the original notebooks exactly: 0 = Dengue,
1 = Malaria (NOTE: this is the OPPOSITE of Disease_Label in
engineered_dataset.csv from Phase 1, which used 1 = Dengue -- kept that way
here deliberately for direct comparability with the original presentation
and threshold_selection_results.csv numbers).

Run: python3 04_prepare_model_input.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(HERE, "..", "data")   # not included in the repository (see DATA.md)
IN_PATH = os.path.join(DATA_DIR, "cleaned_dataset.csv")
OUT_PATH = os.path.join(DATA_DIR, "model_input.csv")


def main():
    df = pd.read_csv(IN_PATH, low_memory=False)
    n0 = len(df)

    core = ["Month", "Age", "Sex", "Ward", "Borough", "Treatment_Type"]
    complete = df[core].notna().all(axis=1)
    df = df.loc[complete].copy()

    out = pd.DataFrame({
        "MONTH": df["Month"],
        "AGE": df["Age"],
        "SEX": df["Sex"],  # 'M' / 'F'
        "WARD": df["Ward"].astype(int),
        "BOROUGH": df["Borough"],  # Roman numerals I-XVI
        "PLACE OF TREATMENT": df["Treatment_Type"].map({
            "Home": "HOME TREATMENT", "Hospital": "HOSPITAL TREATMENT"
        }),
        "CLASS": (df["Disease"] == "Malaria").astype(int),  # 0=Dengue, 1=Malaria
    })

    out.to_csv(OUT_PATH, index=False)
    print(f"Loaded {n0} rows, kept {len(out)} complete-case rows -> {OUT_PATH}")
    print(f"\nClass distribution (0=Dengue, 1=Malaria):")
    print(out["CLASS"].value_counts().to_string())
    print(f"\nColumn dtypes:")
    print(out.dtypes.to_string())
    print(f"\nSample:")
    print(out.head())


if __name__ == "__main__":
    main()
