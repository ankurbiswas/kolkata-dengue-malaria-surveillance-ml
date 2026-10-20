"""Builds model_input_v2.csv: the same complete-case rows as model_input.csv, plus ROW_ID and YEAR,
with the target recoded so that DENGUE = 1 (Dengue) and 0 (Malaria).

Provenance only: needs cleaned_dataset.csv from the Phase 1 pipeline, produced by 02_clean_data.py (data not included).
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(HERE, "..", "data")   # not included in the repository (see DATA.md)
SRC = os.path.join(DATA_DIR, "cleaned_dataset.csv")
OLD = os.path.join(DATA_DIR, "model_input.csv")
OUT = os.path.join(HERE, "..", "pipeline", "model_input_v2.csv")

df = pd.read_csv(SRC, low_memory=False)
core = ["Month", "Age", "Sex", "Ward", "Borough", "Treatment_Type"]
df = df.loc[df[core].notna().all(axis=1)].copy()
out = pd.DataFrame({
    "ROW_ID": range(len(df)),
    "YEAR": df["Year"].astype(int).to_numpy(),
    "MONTH": df["Month"].to_numpy(),
    "AGE": df["Age"].to_numpy(),
    "SEX": df["Sex"].to_numpy(),
    "WARD": df["Ward"].astype(int).to_numpy(),
    "BOROUGH": df["Borough"].to_numpy(),
    "PLACE OF TREATMENT": df["Treatment_Type"].map({"Home": "HOME TREATMENT", "Hospital": "HOSPITAL TREATMENT"}).to_numpy(),
    "DENGUE": (df["Disease"] == "Dengue").astype(int).to_numpy(),
})
old = pd.read_csv(OLD)
assert len(old) == len(out), (len(old), len(out))
for c in ["MONTH", "AGE", "SEX", "WARD", "BOROUGH", "PLACE OF TREATMENT"]:
    assert (old[c].to_numpy() == out[c].to_numpy()).all(), c
assert ((1 - old["CLASS"]).to_numpy() == out["DENGUE"].to_numpy()).all()
out.to_csv(OUT, index=False)
print("rows", len(out), "| dengue", int(out.DENGUE.sum()), "| malaria", int((1 - out.DENGUE).sum()))
print("rows identical to model_input.csv (all 6 fields and the target): True")
print(out.groupby("YEAR").DENGUE.agg(["size", "sum"]).T.to_string())
