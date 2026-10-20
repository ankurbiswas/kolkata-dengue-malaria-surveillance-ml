# Differentiating notified dengue from malaria in Kolkata from routine surveillance fields

Analysis code for the study *"Differentiating notified dengue from malaria in Kolkata from routine surveillance fields: a leakage-audited machine-learning study, 2014–2025"* (Biswas A, Das A, Mandal B, Das A, Chatterjee S; manuscript in preparation).

The study classifies notified cases from the Kolkata Municipal Corporation (KMC) line-lists as dengue or malaria using only six routine fields (ward, borough, month, age, sex and place of treatment). It compares seven classifiers under five class-imbalance strategies in a pipeline where every modelling choice is made on training data only. It also audits two leaks of an earlier workflow and analyses the ward-level spatial pattern of both diseases.

**This repository contains code only. The KMC line-lists and the licensed ward boundaries are not included; see [DATA.md](DATA.md).** A synthetic example file lets you check that the code runs.

## Repository layout

| Folder | Contents |
|---|---|
| `data_prep/` | Building the modelling file from the 24 annual KMC workbooks: `01_merge_raw_data.py` (map columns to one schema, drop identifiers), `02_clean_data.py` (dates, ages, wards, boroughs, place of treatment), `03_feature_engineering.py`, `04_prepare_model_input.py`, `05_make_model_input_v2.py` (final modelling file, 129,983 complete cases) and `06_make_excluded_input.py` (cases excluded only because place of treatment is missing). |
| `pipeline/` | The leakage-safe benchmark. `run_revised.py` (7 classifiers × 5 arms × 5 random splits + 1 temporal split; tuning, thresholds, SHAP/permutation importance and field selection on training data only), `analyze_results.py` (tables, paired bootstrap, McNemar, matched-recall check), `leakage_audit.py`, `extra_analysis.py` (field ablations, calibration slope, decision curves, operating points, subgroups, excluded records), `make_figures.py`, `make_workflow_figure.py`, `export_results_json.py`, `fastboot.py` (fast paired bootstrap). |
| `spatial/` | Ward-level incidence, empirical-Bayes smoothing, global and local Moran's I (LISA) and the ward-level validation table. Needs the licensed GISMAP ward map. |
| `examples/` | `make_synthetic_data.py` and `synthetic_model_input.csv`: random data with the same columns as the modelling file. |

## Requirements

Python 3.12 and the packages in `pipeline/requirements.txt` (pandas, numpy, scipy, scikit-learn ≥ 1.4 and < 1.11, imbalanced-learn, xgboost, shap, matplotlib). The spatial scripts also need geopandas, libpysal and esda. The published results used scikit-learn 1.9.0, imbalanced-learn 0.14.2, XGBoost 3.4.1, SHAP 0.52.0, pandas 3.0.5 and NumPy 2.5.2.

```bash
pip install -r pipeline/requirements.txt
```

## Quick test with synthetic data

```bash
python examples/make_synthetic_data.py
DATA=examples/synthetic_model_input.csv OUT=outputs_synthetic SEEDS=1 CLASSIFIERS=XGBoost,LogisticRegression python pipeline/run_revised.py
DATA=examples/synthetic_model_input.csv python pipeline/analyze_results.py outputs_synthetic
```

On Windows PowerShell, set the variables first, for example `$env:DATA="examples\synthetic_model_input.csv"`.

## Full analysis (with the original data)

1. Data preparation: place the KMC workbooks in `data/raw/` (or set `RAW_DIR`) and run `data_prep/01` to `06` in order. Intermediate files go to `data/` (or `DATA_DIR`); the modelling file is written to `pipeline/model_input_v2.csv`.
2. Model run: `python pipeline/run_revised.py` (about 2.7 hours on a 4-core laptop; resumable, finished units are skipped), or double-click `pipeline/run.bat` on Windows.
3. Analysis: `analyze_results.py`, `leakage_audit.py`, `extra_analysis.py`, `make_figures.py`, `make_workflow_figure.py`, then `export_results_json.py`.
4. Spatial: put the ward map in `data/Kolkata_MC_WardMap/` (or set `WARD_MAP_DIR`), then run `spatial/build_ward_attribute_table.py`, `compute_spatial_stats.py` and `make_figure5_table.py XGBoost`.

Main settings are environment variables: `DATA`, `OUT`, `SEEDS` (default `1,2,3,4,5`), `TEMPORAL` (default `1`), `CLASSIFIERS`, `THREADS`, `QUICK=1` (small test run) and `BOOT` (bootstrap resamples, default 1000). Every random operation is seeded, and the test-set predictions of every model are saved, so tables and figures can be regenerated without retraining.

## Note on `03_feature_engineering.py`

This script belongs to the earlier workflow and computes a ward risk score on all labels. **The final pipeline does not use that score**: ward is target-encoded inside the training folds only (`pipeline/run_revised.py`). `03` is kept because `spatial/build_ward_attribute_table.py` reads the cleaned fields from its output, and because it documents the earlier workflow whose leaks `pipeline/leakage_audit.py` measures.

## Licence and citation

Code released under the [MIT licence](LICENSE). If you use it, please cite the article (citation to be added on publication).
