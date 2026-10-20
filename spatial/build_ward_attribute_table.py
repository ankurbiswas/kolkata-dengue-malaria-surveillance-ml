"""Build the ward-level attribute table (one row per KMC ward) for joining to the ward map in QGIS.

Inputs : engineered_dataset.csv.gz (2014-2025 line-listing), Kolkata_MC_WardMap.geojson (population fields only)
Outputs: ward_attribute_table.csv (+ .csvt column types for QGIS), ward_attribute_table_dictionary.csv
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CASES = Path(os.environ.get("CASES") or (HERE / ".." / "data" / "engineered_dataset.csv.gz"))
WARD_MAP_DIR = Path(os.environ.get("WARD_MAP_DIR") or (HERE / ".." / "data" / "Kolkata_MC_WardMap"))   # licensed (GISMAP IN), not included
GEO = WARD_MAP_DIR / "Geojson" / "Kolkata_MC_WardMap.geojson"
OUT = Path(os.environ.get("SPATIAL_OUT") or (HERE / "output"))
OUT.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2014, 2026))
WINDOWS = {"1419": range(2014, 2020), "2022": range(2020, 2023), "2325": range(2023, 2026), "1422": range(2014, 2023)}
PEAK_MONTHS = {"AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER"}
ROMAN = {r: i + 1 for i, r in enumerate(["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"])}
MIN_N = 10   # minimum cases for proportion/median fields
MIN_SEX = 5  # minimum cases per sex for the male:female rate ratio
SMOOTH_K = 10.0  # same shrinkage strength as WardTargetEncoder(smoothing=10) in script 05

# ---- population per ward (census 2011 + 2023 estimate) from the ward map
feats = json.load(open(GEO, encoding="utf-8"))["features"]
pop = pd.DataFrame([f["properties"] for f in feats])
pop = pop[pop["sourceward"].notna()].copy()
pop["ward_no"] = pop["sourceward"].astype(int)
pop = pop.set_index("ward_no").sort_index()[["sourceward", "totalpop11", "malepop11", "femalepop1", "totpop23"]]
wards = pop.index

def pop_year(y):
    return pop["totalpop11"] + (pop["totpop23"] - pop["totalpop11"]) * (y - 2011) / 12.0  # NaN for wards 142-144

def person_years(years):
    return sum(pop_year(y) for y in years)

# ---- cases
df = pd.read_csv(CASES, usecols=["Year", "Disease", "Month", "Age", "Sex", "Ward", "Borough", "Treatment_Type"])
n_all = len(df)
df = df.dropna(subset=["Ward"]).copy()
df["ward_no"] = df["Ward"].astype(int)
df = df[df["Year"].between(2014, 2025)]
print(f"rows: {n_all} total, {len(df)} used (dropped {n_all - len(df)} without a valid ward)")

def cnt(mask):
    return df[mask].groupby("ward_no").size().reindex(wards, fill_value=0)

D, M = df["Disease"].eq("Dengue"), df["Disease"].eq("Malaria")
out = pd.DataFrame(index=wards)
out.index.name = "ward_no"
out["sourceward"] = pop["sourceward"]
out["has_pop"] = pop["totalpop11"].notna().astype(int)

# borough (modal value per ward)
bor = df.groupby("ward_no")["Borough"].agg(lambda s: s.value_counts().index[0]).reindex(wards)
out["borough"] = bor
out["borough_no"] = bor.map(ROMAN)

# case counts and Dengue share
out["cases_dengue"] = cnt(D)
out["cases_malaria"] = cnt(M)
out["cases_total"] = out["cases_dengue"] + out["cases_malaria"]
g = out["cases_dengue"].sum() / out["cases_total"].sum()
out["dengue_share"] = (out["cases_dengue"] / out["cases_total"]).where(out["cases_total"] > 0)
out["dengue_share_sm10"] = (out["cases_dengue"] + SMOOTH_K * g) / (out["cases_total"] + SMOOTH_K)

# incidence per 100,000 person-years (annual average) using year-specific interpolated population
py_all = person_years(YEARS)
out["person_years"] = py_all.round(0)
for name, col in [("dengue", "cases_dengue"), ("malaria", "cases_malaria"), ("total", "cases_total")]:
    out[f"inc_{name}_100k"] = out[col] / py_all * 1e5

def eb_smooth(cases, py):
    """Global empirical-Bayes (Marshall) smoothing of rates; returns per-100k rates."""
    ok = py.notna() & (py > 0)
    O, P = cases[ok], py[ok]
    m = O.sum() / P.sum()
    r = O / P
    s2 = max((P * (r - m) ** 2).sum() / P.sum() - m / P.mean(), 0.0)
    w = s2 / (s2 + m / P)
    res = pd.Series(np.nan, index=cases.index)
    res[ok] = (w * r + (1 - w) * m) * 1e5
    return res

out["inc_dengue_eb_100k"] = eb_smooth(out["cases_dengue"], py_all)
out["inc_malaria_eb_100k"] = eb_smooth(out["cases_malaria"], py_all)

# period incidence (per 100k per year) and change 2023-25 vs 2014-22
for tag, yrs in WINDOWS.items():
    py = person_years(yrs)
    inrange = df["Year"].isin(list(yrs))
    out[f"dinc_{tag}"] = cnt(D & inrange) / py * 1e5
    out[f"minc_{tag}"] = cnt(M & inrange) / py * 1e5
out["dinc_chg"] = out["dinc_2325"] - out["dinc_1422"]
out["minc_chg"] = out["minc_2325"] - out["minc_1422"]

# sex: counts and male:female incidence rate ratio (census-2011 sex composition)
for name, mask in [("dengue", D), ("malaria", M)]:
    m_, f_ = cnt(mask & df["Sex"].eq("M")), cnt(mask & df["Sex"].eq("F"))
    out[f"{name}_male"], out[f"{name}_female"] = m_, f_
    irr = (m_ / f_.replace(0, np.nan)) * (pop["femalepop1"] / pop["malepop11"])
    out[f"irr_mf_{name}"] = irr.where((m_ >= MIN_SEX) & (f_ >= MIN_SEX))

# age and treatment among cases, per disease
def by_ward(mask, func):
    return df[mask].groupby("ward_no").apply(func, include_groups=False).reindex(wards)

for name, mask in [("dengue", D), ("malaria", M)]:
    a = df[mask & df["Age"].notna()]
    grp = a.groupby("ward_no")["Age"]
    n = grp.size().reindex(wards)
    out[f"median_age_{name}"] = grp.median().reindex(wards).where(n >= MIN_N)
    out[f"pct_child_{name}"] = (a.assign(x=a["Age"] < 15).groupby("ward_no")["x"].mean() * 100).reindex(wards).where(n >= MIN_N)
    out[f"pct_elder_{name}"] = (a.assign(x=a["Age"] >= 60).groupby("ward_no")["x"].mean() * 100).reindex(wards).where(n >= MIN_N)
    t = df[mask & df["Treatment_Type"].notna()]
    nt = t.groupby("ward_no").size().reindex(wards)
    out[f"pct_hosp_{name}"] = (t.assign(x=t["Treatment_Type"].eq("Hospital")).groupby("ward_no")["x"].mean() * 100).reindex(wards).where(nt >= MIN_N)
    s = df[mask & df["Month"].notna()]
    ns = s.groupby("ward_no").size().reindex(wards)
    out[f"pct_peak_{name}"] = (s.assign(x=s["Month"].isin(PEAK_MONTHS)).groupby("ward_no")["x"].mean() * 100).reindex(wards).where(ns >= MIN_N)

out = out.reset_index()
for c in out.select_dtypes("float").columns:
    out[c] = out[c].round(4)
out.to_csv(OUT / "ward_attribute_table.csv", index=False, na_rep="")

# .csvt so QGIS reads column types correctly (ids as text/integer, measures as real)
types = []
for c in out.columns:
    if c in ("sourceward", "borough"):
        types.append('"String"')
    elif str(out[c].dtype).startswith("int"):
        types.append('"Integer"')
    else:
        types.append('"Real"')
(OUT / "ward_attribute_table.csvt").write_text(",".join(types), encoding="utf-8")

DICT = [
    ("ward_no", "Ward number 1-144 (integer)", "join key (integer)"),
    ("sourceward", "Ward number as text; identical to the ward map's sourceward field", "JOIN KEY - use this one"),
    ("has_pop", "1 if the ward map has population for this ward; 0 for wards 142-144", "use to hatch/grey 'no data' wards on rate maps"),
    ("borough", "Borough (Roman numeral I-XVI), most frequent value among the ward's cases", "Figure 1, borough dissolve"),
    ("borough_no", "Borough as integer 1-16", "dissolve wards into boroughs"),
    ("cases_dengue / cases_malaria / cases_total", "Case counts 2014-2025", "counts (not for choropleth without normalising)"),
    ("dengue_share", "Dengue cases / all cases in the ward (0-1). A share among REPORTED cases, not incidence", "Figure 3 (raw)"),
    ("dengue_share_sm10", "dengue_share shrunk toward the city-wide share with strength 10 (same smoothing as the model's ward encoder)", "Figure 3 (recommended)"),
    ("person_years", "Sum of yearly ward population 2014-2025 (linear interpolation 2011 census -> 2023 estimate, extrapolated to 2025)", "denominator"),
    ("inc_dengue_100k / inc_malaria_100k / inc_total_100k", "Raw annual incidence per 100,000 person-years, 2014-2025", "Figure 2 (raw)"),
    ("inc_dengue_eb_100k / inc_malaria_eb_100k", "Empirical-Bayes smoothed incidence per 100,000 person-years (global Marshall estimator)", "Figure 2 (recommended)"),
    ("dinc_1419 / dinc_2022 / dinc_2325 / dinc_1422", "Dengue incidence per 100,000 per year in 2014-19 / 2020-22 / 2023-25 / 2014-22", "Figure 6"),
    ("minc_1419 / minc_2022 / minc_2325 / minc_1422", "Malaria incidence, same windows", "Figure 6"),
    ("dinc_chg / minc_chg", "Incidence 2023-25 minus 2014-22 (per 100,000 per year)", "Figure 6 difference map (diverging palette)"),
    ("dengue_male, dengue_female, malaria_male, malaria_female", "Cases by sex", "counts"),
    ("irr_mf_dengue / irr_mf_malaria", "Male:female incidence rate ratio using 2011 census sex composition; blank if fewer than 5 cases in either sex", "Supplementary sex map"),
    ("median_age_dengue / _malaria", "Median age of cases (years); blank if <10 cases with known age", "Age of cases - NOT an age-specific rate"),
    ("pct_child_dengue / _malaria", "% of cases aged <15; blank if <10", "Supplementary age map (proportion)"),
    ("pct_elder_dengue / _malaria", "% of cases aged >=60; blank if <10", "Supplementary age map (proportion)"),
    ("pct_hosp_dengue / _malaria", "% of cases treated in hospital/nursing home among known treatment place; blank if <10 known", "Supplementary care map"),
    ("pct_peak_dengue / _malaria", "% of cases in Aug-Nov (the four peak months in this dataset); blank if <10", "Supplementary seasonal map"),
]
pd.DataFrame(DICT, columns=["field", "meaning", "use"]).to_csv(OUT / "ward_attribute_table_dictionary.csv", index=False)

# ---- checks
print("table:", out.shape, "| wards:", out["ward_no"].nunique())
print("cases_total sum:", int(out["cases_total"].sum()), "| dengue:", int(out["cases_dengue"].sum()), "| malaria:", int(out["cases_malaria"].sum()))
print("wards without population:", out.loc[out["has_pop"] == 0, "ward_no"].tolist())
print("wards with <30 total cases:", out.loc[out["cases_total"] < 30, "ward_no"].tolist())
print("global dengue share:", round(g, 4))
for c in ["inc_dengue_100k", "inc_dengue_eb_100k", "inc_malaria_100k", "inc_malaria_eb_100k", "dengue_share", "dengue_share_sm10"]:
    print(f"{c}: min {out[c].min():.3f}  median {out[c].median():.3f}  max {out[c].max():.3f}")
print("EB shrinkage corr (dengue raw vs EB):", round(out["inc_dengue_100k"].corr(out["inc_dengue_eb_100k"]), 3))
print("rows with blanks per column (top):")
print(out.isna().sum().sort_values(ascending=False).head(8).to_string())
