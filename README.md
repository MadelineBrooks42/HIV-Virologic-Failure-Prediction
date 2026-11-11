# Cox and DeepSurv survival models with three feature-selection strategies

Python scripts that fit Cox proportional hazards and DeepSurv (PyTorch) models to time-to-event data, compare three feature-selection strategies, and evaluate the models with cross-validated concordance index (overall, 1-year and 5-year) and integrated Brier score, with Nadeau-Bengio corrected 95% confidence intervals across folds, and SHAP values. Two analysis notebooks then summarise and plot the results.

## Contents

| File | Model | Feature selection / purpose |
|---|---|---|
| `01_cox_feature_subset.py` | Cox PH | Domain-informed feature subset |
| `02_cox_powershap.py` | Cox PH | Feature subset, then PowerSHAP |
| `03_cox_shap_filter.py` | Cox PH | Feature subset, then SHAP-based filter |
| `04_deepsurv_feature_subset.py` | DeepSurv | Domain-informed feature subset |
| `05_deepsurv_powershap.py` | DeepSurv | PowerSHAP |
| `06_deepsurv_shap_filter.py` | DeepSurv | SHAP-based filter |
| `07_analyze_coefficients.ipynb` | - | Summarises and plots feature coefficients (hazard ratios) from scripts 01-06 |
| `08_analyze_log_risk_scores.ipynb` | - | Plots log risk score distributions from scripts 01-06 |

**Run order.** Scripts 01-06 are independent of each other: each reads the input data and writes its own results. Notebooks 07 and 08 read the result files written by 01-06, so run those first.

Each file opens with a header describing its purpose, input, outputs, configuration and package requirements.

The scripts are divided into cells with `# %%` markers, so they can be run as ordinary Python scripts or cell by cell in VS Code, Spyder, PyCharm or Jupyter (for example with Jupytext). Notebooks 07 and 08 are Jupyter notebooks because they display tables and figures inline.

## Setup

1. Install the packages used by the scripts and notebooks:

   ```
   pip install joblib lifelines matplotlib numpy openpyxl pandas pycox scikit-learn scikit-survival scipy shap statsmodels torch torchtuples tqdm notebook
   ```

2. Create these folders in the same folder as the scripts (they are listed in `.gitignore` so data and results are not committed):
   - `data/` - put your input CSV here as `data/input_data.csv`, or change `file_path` in STEP 2 of each script
   - `output/` - result files from scripts 01-06 are written here, and notebooks 07-08 read them from here
   - `figures/` - figures from notebooks 07-08 are written here

3. Run a script from this folder, because the paths are relative:

   ```
   python 01_cox_feature_subset.py
   ```

   Then open notebooks 07 and 08 in Jupyter from the same folder.

## Input data

Scripts 01-06 expect a CSV in start-stop (person-period) format, with one row per subject-interval. Columns used by the code include `naid` (subject ID), `tstart`, `tstop`, `failure` (event indicator) and the covariates. The data are not distributed with this repository.

## Outputs

Every file exported by scripts 01-06 is named `<filename>_<result>.<ext>`, where `filename` is the script's prefix (for example `cox_powershap`). The files written by each script are listed in its header.

## Analysis notebooks

- **07** reads `*_HazardRatios_ByFold.csv` (Cox) and `*_FeatureCoefficients.csv` (DeepSurv) for the three feature-selection methods and saves five figures to `figures/`.
- **08** reads `*_LogRiskScores_ByFold.csv` (Cox) and `*_LogRiskScores.csv` (DeepSurv) and saves three figures to `figures/`.
- **Simulated-data sections of 08.** The later part of notebook 08 also reads results from runs on simulated data, expected in `output/simulated/`, `output/simulated/time_fixed/` and `output/simulated/time_varying/`. Producing these requires running the scripts on a simulated dataset (script 02 has a `DUMMY_DATA` switch for this). Scripts 01-06 use time-fixed covariates, so the `time_varying` files cannot be produced by them. Skip those cells if you do not have these files.
