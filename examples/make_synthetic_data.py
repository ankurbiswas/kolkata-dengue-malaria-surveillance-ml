"""Generate a synthetic file with the same columns and value types as pipeline/model_input_v2.csv.

The values are random and contain NO real case data; they only let you check that the pipeline runs.
Dengue probability depends on borough, month and age so that the models have some signal to learn.

Usage:  python make_synthetic_data.py [n_rows]     (default 6000)  ->  examples/synthetic_model_input.csv
"""
import os
import sys

import numpy as np
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
rng = np.random.default_rng(2026)
HERE = os.path.dirname(os.path.abspath(__file__))

MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]
# 144 wards split into 16 boroughs of 9 consecutive wards (a simplification of the real KMC ward list)
ward = rng.integers(1, 145, N)
borough_idx = (ward - 1) // 9
month_idx = rng.integers(0, 12, N)
age = np.clip(np.round(rng.gamma(3.0, 11.0, N), 1), 0, 100)
sex = rng.choice(["M", "F"], N, p=[0.64, 0.36])
year = rng.integers(2014, 2026, N)

logit = (-1.0 + 2.2 * (borough_idx >= 9)            # southern boroughs X-XVI: more dengue
         + 1.2 * np.isin(month_idx, [7, 8, 9, 10])  # August-November peak
         - 0.02 * (age - 30) + 0.4 * (sex == "F") + rng.normal(0, 0.8, N))
dengue = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
place = np.where(rng.random(N) < 0.10 + 0.25 * dengue, "HOSPITAL TREATMENT", "HOME TREATMENT")

df = pd.DataFrame({
    "ROW_ID": np.arange(N), "YEAR": year, "MONTH": np.array(MONTHS)[month_idx], "AGE": age, "SEX": sex,
    "WARD": ward, "BOROUGH": np.array(ROMAN)[borough_idx], "PLACE OF TREATMENT": place, "DENGUE": dengue,
})
out = os.path.join(HERE, "synthetic_model_input.csv")
df.to_csv(out, index=False)
print(f"wrote {out}: {N} synthetic rows, dengue share {dengue.mean():.2f}")
