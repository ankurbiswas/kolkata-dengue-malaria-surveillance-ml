"""Global Moran's I and Local Moran's I (LISA) for ward-level Dengue incidence, Malaria incidence and Dengue share.

Reads the licensed ward boundaries locally (never copied or exported) and writes only attribute tables:
  ward_spatial_stats.csv (+ .csvt)   one row per ward, join to the ward map on `sourceward`
  moran_global_summary.csv           global Moran's I results for the paper
"""
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen

HERE = Path(__file__).resolve().parent
WARD_MAP_DIR = Path(os.environ.get("WARD_MAP_DIR") or (HERE / ".." / "data" / "Kolkata_MC_WardMap"))   # licensed (GISMAP IN), not included
SHP = WARD_MAP_DIR / "Shp file" / "Kolkata_MC_WardMap.shp"
OUT = Path(os.environ.get("SPATIAL_OUT") or (HERE / "output"))
TABLE = OUT / "ward_attribute_table.csv"
PERMS, SEED, ALPHA = 999, 12345, 0.05
QUAD = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}

gdf = gpd.read_file(SHP)
gdf = gdf[gdf["sourceward"].notna()].copy().to_crs(32645)
gdf["sourceward"] = gdf["sourceward"].astype(str)
tab = pd.read_csv(TABLE, dtype={"sourceward": str})
gdf = gdf[["sourceward", "geometry"]].merge(tab[["sourceward", "has_pop", "inc_dengue_100k", "inc_malaria_100k", "dengue_share_sm10"]], on="sourceward", how="left")
gdf["dengue_share_pct"] = gdf["dengue_share_sm10"] * 100
assert len(gdf) == 144

ANALYSES = [
    ("d", "Dengue incidence per 100,000", "inc_dengue_100k", gdf["has_pop"] == 1),
    ("m", "Malaria incidence per 100,000", "inc_malaria_100k", gdf["has_pop"] == 1),
    ("s", "Dengue share of reported cases (%)", "dengue_share_pct", gdf["has_pop"].notna()),
]

result = pd.DataFrame({"sourceward": gdf["sourceward"]})
summary = []

for key, label, col, mask in ANALYSES:
    g = gdf[mask].reset_index(drop=True)
    w = Queen.from_dataframe(g, use_index=False)
    islands = list(w.islands)
    w.transform = "r"
    y = g[col].to_numpy(dtype=float)

    gm = Moran(y, w, permutations=PERMS)
    summary.append(dict(variable=label, wards=len(g), islands=len(islands), transform="none", morans_I=gm.I, expected_I=gm.EI_sim,
                        z_sim=gm.z_sim, p_sim=gm.p_sim, permutations=PERMS))
    if col != "dengue_share_pct":  # skewed rates: repeat on log10 as a robustness check
        gl = Moran(np.log10(y), w, permutations=PERMS)
        summary.append(dict(variable=label, wards=len(g), islands=len(islands), transform="log10", morans_I=gl.I, expected_I=gl.EI_sim,
                            z_sim=gl.z_sim, p_sim=gl.p_sim, permutations=PERMS))

    lm = Moran_Local(y, w, permutations=PERMS, seed=SEED)
    sig = lm.p_sim < ALPHA
    code = np.where(sig, lm.q, 0)
    cat = pd.Series(code).map({0: "Not significant", **QUAD}).to_numpy()
    part = pd.DataFrame({"sourceward": g["sourceward"], f"lisa_{key}": cat, f"lisa_{key}_code": code,
                         f"lisa_{key}_p": lm.p_sim, f"lisa_{key}_I": lm.Is})
    result = result.merge(part, on="sourceward", how="left")
    print(f"\n{label}: n={len(g)}, islands={islands}")
    print(f"  Global Moran's I = {gm.I:.3f}, z = {gm.z_sim:.2f}, p = {gm.p_sim:.4f}")
    print(pd.Series(cat).value_counts().to_string())

for c in result.columns:
    if c.endswith("_code"):
        result[c] = result[c].astype("Int64")
    elif c.endswith(("_p", "_I")):
        result[c] = result[c].round(4)
result.to_csv(OUT / "ward_spatial_stats.csv", index=False, na_rep="")
types = ['"String"' if (c == "sourceward" or (c.startswith("lisa_") and not c.endswith(("_code", "_p", "_I")))) else
         '"Integer"' if c.endswith("_code") else '"Real"' for c in result.columns]
(OUT / "ward_spatial_stats.csvt").write_text(",".join(types), encoding="utf-8")
pd.DataFrame(summary).round(4).to_csv(OUT / "moran_global_summary.csv", index=False)
print("\nwritten:", OUT / "ward_spatial_stats.csv")
