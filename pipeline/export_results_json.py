"""
export_results_json.py -- collects every analysis table (plus a few spatial summaries) into one JSON file, so that the text and tables of the
Results section can be generated from the data and never typed by hand.

Usage:   python export_results_json.py [output_folder] [json_path]      (default: outputs_v2, <output_folder>/analysis/results_data.json)
"""
import os
import re
import sys
import glob
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs_v2")
AN = os.path.join(OUT, "analysis")
DEST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(AN, "results_data.json")
GIS = os.environ.get("GIS_TABLES") or os.path.join(HERE, "..", "spatial", "output")


def records(df):
    return json.loads(df.to_json(orient="records"))


data = {}
for f in sorted(glob.glob(os.path.join(AN, "*.csv"))):
    data[os.path.basename(f)[:-4]] = records(pd.read_csv(f))

with open(os.path.join(OUT, "run_info.json"), encoding="utf-8") as f:
    data["run_info"] = json.load(f)

# hyperparameters, thresholds and timing per unit
units = []
for mp in sorted(glob.glob(os.path.join(OUT, "meta", "*.json"))):
    with open(mp, encoding="utf-8") as f:
        m = json.load(f)
    units.append({"split": m["split"], "classifier": m["classifier"], "n_train": m["n_train"], "n_test": m["n_test"], "test_dengue_prevalence": m["test_dengue_prevalence"],
                  "params": m["params"], "thresholds": m["thresholds"], "ranking": m["ranking"], "rank_method": m["rank_method"], "selected_k": m["selected_k"],
                  "total_min": m["timing_s"]["total_s"] / 60})
data["units"] = units

# fields chosen on training data in each split (canonical order), with the number of splits that chose each subset
ORDER = ["WARD", "AGE", "MONTH", "SEX", "PLACE", "BOROUGH"]
sw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "sweep", "*.csv")))], ignore_index=True)
sel = sw[sw["selected"] == 1].copy()
sel["set"] = sel["fields"].map(lambda s: "+".join(sorted(s.split("+"), key=ORDER.index)))
sel["group"] = np.where(sel["split"].str.startswith("seed"), "random", "temporal")
cnt = sel.groupby(["group", "classifier", "set"]).agg(n_splits=("split", "nunique"), k=("k", "first")).reset_index().sort_values(["group", "classifier", "n_splits"], ascending=[True, True, False])
data["selected_sets"] = records(cnt)

# spatial summaries (ward attribute table, Moran / LISA)
try:
    wt = pd.read_csv(os.path.join(GIS, "ward_attribute_table.csv"))
    ss = pd.read_csv(os.path.join(GIS, "ward_spatial_stats.csv"), dtype={"sourceward": str})
    mg = pd.read_csv(os.path.join(GIS, "moran_global_summary.csv"))
    pop = wt[wt["has_pop"] == 1]
    sp = {"n_wards": int(len(wt)), "n_wards_with_population": int(len(pop)),
          "cases_dengue": int(wt["cases_dengue"].sum()), "cases_malaria": int(wt["cases_malaria"].sum()),
          "dengue_share_citywide": float(wt["cases_dengue"].sum() / wt["cases_total"].sum())}
    for name, col in (("dengue", "inc_dengue_100k"), ("malaria", "inc_malaria_100k")):
        s = pop[col]
        sp[f"inc_{name}"] = {"median": float(s.median()), "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75)), "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean())}
    s = wt["dengue_share"].dropna()
    sp["dengue_share_ward"] = {"median": float(s.median()), "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75)), "min": float(s.min()), "max": float(s.max())}
    for tag, col in (("1419", "dinc_1419"), ("2022", "dinc_2022"), ("2325", "dinc_2325"), ("1422", "dinc_1422")):
        sp[f"dinc_mean_{tag}"] = float(pop[col].mean())
    sp["dinc_increase_wards"] = int((pop["dinc_chg"] > 0).sum())
    sp["dinc_wards_compared"] = int(pop["dinc_chg"].notna().sum())
    lisa = {}
    for key, name in (("d", "dengue_incidence"), ("m", "malaria_incidence"), ("s", "dengue_share")):
        lisa[name] = {k: int(v) for k, v in ss[f"lisa_{key}"].value_counts().items()}
    sp["lisa"] = lisa
    sp["moran"] = records(mg)
    data["spatial"] = sp
except Exception as e:   # the model tables must still be exported when the GIS tables are missing
    print("spatial summaries skipped:", type(e).__name__, e)

# ward-level validation summaries written by make_figure5_table.py
ward = {}
for f in glob.glob(os.path.join(GIS, "ward_validation_*_summary.txt")):
    clf = re.match(r"ward_validation_(.+)_summary\.txt", os.path.basename(f)).group(1)
    txt = open(f, encoding="utf-8").read()
    g = lambda pat: float(re.search(pat, txt).group(1))
    ward[clf] = {"wards_used": int(re.search(r"predictions: (\d+) of (\d+)", txt).group(1)), "pearson_r": g(r"Pearson r .*= ([\-0-9.]+)"),
                 "weighted_mae": g(r"Weighted mean absolute error = ([\-0-9.]+)"), "morans_I": g(r"Moran's I of residuals = ([\-0-9.]+)"),
                 "morans_p": g(r"permutation p = ([\-0-9.]+)"), "clusters": txt.split("clusters:")[-1].strip()}
data["ward_validation"] = ward

with open(DEST, "w", encoding="utf-8") as f:
    json.dump(data, f)
print("wrote", DEST, "| tables:", len(data), "| size %.1f MB" % (os.path.getsize(DEST) / 1e6))
