# Ward-level result tables

Derived tables behind the spatial results and Fig. 12 of the article. They contain **no patient information and no case counts**:
the KMC line-lists and their ward-level aggregates cannot be shared (see `DATA.md`). Rows are keyed by `sourceward`, the KMC ward
number (1-144); the ward boundaries themselves are licensed from GISMAP IN and are not included.

| File | Contents |
|---|---|
| `global_morans_i.csv` | Global Moran's I of ward-level dengue incidence, malaria incidence (raw and log10) and the dengue share of notified cases: I, expected I, pseudo-z, pseudo-p (999 permutations, Queen contiguity, row-standardised weights). |
| `ward_lisa_clusters.csv` | Local Moran's I (LISA) per ward for dengue incidence (`lisa_d*`), malaria incidence (`lisa_m*`) and the dengue share (`lisa_s*`): cluster class (High-High, Low-Low, High-Low, Low-High, Not significant), numeric code, pseudo-p and local I. |
| `ward_validation_xgboost.csv` | Ward-level calibration of the XGBoost model (baseline arm, threshold 0.5, five random splits): observed dengue share, mean predicted probability of dengue, residual (predicted minus observed, as a proportion and in percentage points) and the LISA class of the residuals. |

Produced by `spatial/compute_spatial_stats.py` and `spatial/make_figure5_table.py`; the count column of the validation table was removed before publication.
