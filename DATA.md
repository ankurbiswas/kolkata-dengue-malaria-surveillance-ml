# Data

## Not included in this repository

- **KMC dengue and malaria line-lists, 2014–2025** (24 annual workbooks). These are patient-level surveillance records of the Kolkata Municipal Corporation. Names, addresses, contact numbers and religion are removed in `data_prep/01_merge_raw_data.py`, before any analysis, but the remaining case-level records cannot be shared under the terms of data access. Requests should be addressed to the Kolkata Municipal Corporation.
- **Derived case-level files** (`model_input_v2.csv`, `excluded_place_missing.csv`, cleaned and engineered datasets) and the **per-case test-set predictions**, for the same reason.
- **Ward boundaries** (Kolkata_MC_WardMap, GISMAP IN). Licensed; they cannot be redistributed.

## Included

`results/ward_tables/` holds derived ward-level tables (global and local Moran's I, and the ward-level validation of the XGBoost model). They contain no patient information and no case counts; ward-level aggregates of the line-lists themselves are not shared.

## Structure of the modelling file

`pipeline/model_input_v2.csv` has one row per notified case:

| Column | Type | Values |
|---|---|---|
| `ROW_ID` | integer | 0 … n−1 |
| `YEAR` | integer | 2014–2025 |
| `MONTH` | text | `JANUARY` … `DECEMBER` (reporting or blood-test month) |
| `AGE` | number | years (decimal for children recorded in months) |
| `SEX` | text | `M`, `F` |
| `WARD` | integer | 1–144 (KMC ward of the notification) |
| `BOROUGH` | text | `I` … `XVI` |
| `PLACE OF TREATMENT` | text | `HOME TREATMENT`, `HOSPITAL TREATMENT` (derived from free text) |
| `DENGUE` | integer | 1 = dengue, 0 = malaria (the line-list in which the case was notified) |

`examples/synthetic_model_input.csv` has exactly this structure with random values.
