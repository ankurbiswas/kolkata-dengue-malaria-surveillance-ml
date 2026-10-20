"""Builds excluded_place_missing.csv: the notified cases that were left out of the modelling cohort ONLY because place of treatment is missing
(month, age, sex, ward and borough are all present).  Used by extra_analysis.py (part H) to check how the five-field model (without place of
treatment) performs on cases the main analysis never saw.

Provenance only: needs cleaned_dataset.csv from the Phase 1 pipeline, produced by 02_clean_data.py (data not included).
Same ethics scope as model_input_v2.csv (de-identified fields only).
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(HERE, "..", "data")   # not included in the repository (see DATA.md)
SRC = os.path.join(DATA_DIR, "cleaned_dataset.csv")
OUT = os.path.join(HERE, "..", "pipeline", "excluded_place_missing.csv")
df = pd.read_csv(SRC, low_memory=False)
five = ["Month", "Age", "Sex", "Ward", "Borough"]
sub = df.loc[df[five].notna().all(axis=1) & df["Treatment_Type"].isna()].copy()
out = pd.DataFrame({
    "ROW_ID": range(len(sub)),
    "YEAR": sub["Year"].astype(int).to_numpy(),
    "MONTH": sub["Month"].to_numpy(),
    "AGE": sub["Age"].to_numpy(),
    "SEX": sub["Sex"].to_numpy(),
    "WARD": sub["Ward"].astype(int).to_numpy(),
    "BOROUGH": sub["Borough"].to_numpy(),
    "DENGUE": (sub["Disease"] == "Dengue").astype(int).to_numpy(),
})
out.to_csv(OUT, index=False)
print("rows", len(out), "| dengue", int(out.DENGUE.sum()), "| malaria", int((1 - out.DENGUE).sum()))
print(out.groupby("YEAR").DENGUE.agg(["size", "sum"]).T.to_string())
