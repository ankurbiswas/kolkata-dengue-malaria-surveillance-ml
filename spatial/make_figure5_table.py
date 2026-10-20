"""Figure 5 table: observed vs predicted dengue share per ward, plus Moran's I of the residuals.

Reads  pipeline\\outputs_v2\\analysis\\T8_ward_validation.csv  (from analyze_results.py)
Reads  the licensed ward boundaries locally (never copied or exported)
Writes ward_validation_<classifier>.csv  -> join to the ward map in QGIS on `sourceward`
       ward_validation_<classifier>_summary.txt

Usage:  python make_figure5_table.py [classifier]        (default XGBoost; env MIN_N = minimum test predictions per ward, default 30)
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen

HERE = os.path.dirname(os.path.abspath(__file__))
WARD_MAP_DIR = os.environ.get("WARD_MAP_DIR") or os.path.join(HERE, "..", "data", "Kolkata_MC_WardMap")   # licensed (GISMAP IN), not included
SHP = os.path.join(WARD_MAP_DIR, "Shp file", "Kolkata_MC_WardMap.shp")
T8 = os.environ.get("T8") or os.path.join(HERE, "..", "pipeline", "outputs_v2", "analysis", "T8_ward_validation.csv")
OUT = os.environ.get("SPATIAL_OUT") or os.path.join(HERE, "output")
CLF = sys.argv[1] if len(sys.argv) > 1 else "XGBoost"
MIN_N = int(os.environ.get("MIN_N", "30"))
QUAD = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}

w = pd.read_csv(T8)
w = w[w["classifier"] == CLF].copy()
if w.empty:
    sys.exit(f"classifier {CLF} not found in {T8}")
w["sourceward"] = w["WARD"].astype(int).astype(str)
gdf = gpd.read_file(SHP)
gdf = gdf[gdf["sourceward"].notna()].copy().to_crs(32645)
gdf["sourceward"] = gdf["sourceward"].astype(str)
g = gdf[["sourceward", "geometry"]].merge(w.drop(columns=["WARD"]), on="sourceward", how="left")
ok = g["n_test_predictions"].fillna(0) >= MIN_N
sub = g[ok].reset_index(drop=True)

wq = Queen.from_dataframe(sub, use_index=False)
wq.transform = "r"
resid = sub["residual_pred_minus_obs"].to_numpy(float)
np.random.seed(12345)   # the permutation p-value is reproducible
mi = Moran(resid, wq, permutations=999)
lm = Moran_Local(resid, wq, permutations=999, seed=12345)
cat = np.where(lm.p_sim < 0.05, pd.Series(lm.q).map(QUAD), "Not significant")
sub["lisa_residual"] = cat
sub["lisa_residual_p"] = lm.p_sim.round(4)
r = float(np.corrcoef(sub["observed_dengue_share"], sub["mean_predicted_p_dengue"])[0, 1])
wmae = float(np.average(np.abs(resid), weights=sub["n_test_predictions"]))
out = g[["sourceward"]].merge(sub.drop(columns=["geometry"]), on="sourceward", how="left")
out["observed_pct"] = out["observed_dengue_share"] * 100          # percentages and percentage points are easier to use in map legends
out["predicted_pct"] = out["mean_predicted_p_dengue"] * 100
out["residual_pp"] = out["residual_pred_minus_obs"] * 100
out.round(5).to_csv(os.path.join(OUT, f"ward_validation_{CLF}.csv"), index=False, na_rep="")
# .csvt so that QGIS reads sourceward as text (it must match the ward map's text field) and the measures as numbers
types = ['"String"' if c in ("sourceward", "classifier", "lisa_residual") else '"Real"' for c in out.columns]
with open(os.path.join(OUT, f"ward_validation_{CLF}.csvt"), "w", encoding="utf-8") as f:
    f.write(",".join(types))
sum_txt = (f"Ward-level validation, {CLF} (Baseline, default threshold), wards with >= {MIN_N} test predictions: {len(sub)} of {len(g)}\n"
           f"Pearson r (observed vs predicted dengue share) = {r:.3f}\n"
           f"Weighted mean absolute error = {wmae:.4f}\n"
           f"Moran's I of residuals = {mi.I:.3f}, permutation p = {mi.p_sim:.4f}  (0 = no spatial pattern left in the errors)\n"
           f"Significant residual clusters: {pd.Series(cat).value_counts().to_dict()}\n")
with open(os.path.join(OUT, f"ward_validation_{CLF}_summary.txt"), "w", encoding="utf-8") as f:
    f.write(sum_txt)
print(sum_txt)
