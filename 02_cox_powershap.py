# %% [markdown]
# # Cox model with feature subset, then PowerSHAP feature selection
#
# **Script:** `02_cox_powershap.py`  
# **Model:** Cox proportional hazards (`lifelines.CoxPHFitter`)  
# **Output file prefix:** `cox_powershap`
#
# ## Purpose
#
# - Cox model with feature subset; then PowerSHAP
# - Implements K-fold cross-validation
#
# ## Input
#
# A CSV in start-stop (person-period) format, read in STEP 2 from `file_path`. Columns used by the code include `naid` (subject ID), `tstart`, `tstop`, `failure` (event indicator) and the covariates. Set `file_path` to your own dataset.
#
# ## Outputs
#
# Written to `output_dir` as `cox_powershap` + the suffix below.
#
# - `cox_powershap_PowerSHAP_FeatureSelection_Results.csv`
# - `cox_powershap_HazardRatios_ByFold.csv`
# - `cox_powershap_LogRiskScores_ByFold.csv`
# - `cox_powershap_HazardRatios_FullDataset.csv`
# - `cox_powershap_SHAPValues.csv`
# - `cox_powershap_MeanAbsSHAPValues.csv`
# - `cox_powershap_FeatureCoefficients.csv`
# - `cox_powershap_CIndex.xlsx`
# - `cox_powershap_Brier.xlsx`
# - `cox_powershap_PopAvgSurvival.csv`
# - `cox_powershap_SurvPlot.png`
#
# ## Configuration
#
# Values as set in this script. `SEED`, `output_dir` and `filename` are set in STEP 1 and `file_path` in STEP 2; the remaining values are set in STEP 1 or where cross-validation is configured.
#
# | Variable | Value in this script | Meaning |
# |---|---|---|
# | `SEED` | `12345` | Random seed for reproducibility |
# | `DUMMY_DATA` | `False` | If `True`, generate a simulated dataset instead of reading `file_path` |
# | `output_dir` | `'output/'` | Folder for exported files (must end with `/` and must already exist) |
# | `filename` | `'cox_powershap'` | Prefix added to every exported file name |
# | `file_path` | `'data/input_data.csv'` | Input CSV (STEP 2) |
# | `n_folds` | `10` | Number of cross-validation folds |
# | `n_splits` | `10` | Number of cross-validation splits passed to the fold generator |
#
# ## Requirements
#
# Python packages imported by this script: `lifelines`, `matplotlib`, `numpy`, `openpyxl`, `pandas`, `pycox`, `scikit-learn`, `scikit-survival`, `scipy`, `shap`, `statsmodels`, `torch`, `torchtuples`, `tqdm`.

# %%
# Library imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lifelines.utils
import contextlib
import os
import itertools
import io
import random
import warnings

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines import CoxTimeVaryingFitter
from lifelines.statistics import logrank_test
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import brier_score
from pycox.evaluation import EvalSurv
from pycox.evaluation.concordance import concordance_td
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.nonparametric import kaplan_meier_estimator
from datetime import datetime
from scipy.stats import wilcoxon
from scipy.interpolate import interp1d
from scipy.stats import t as t_dist
from scipy.stats import rankdata
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests
from itertools import groupby
from tqdm import tqdm
from itertools import combinations

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
from torchtuples.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

# %%
# STEP 1
# Set random seed for reproducibility

SEED = 12345
np.random.seed(SEED)

DUMMY_DATA = False

# Set output folder and file name prefix for exports
# NOTE: output_dir must end with "/" and must exist before running
output_dir = "output/"
filename = "cox_powershap"

# %%
# STEP 2
# Load data

# ----------------------------------------------
# Input data: start-stop (person-period) format, one row per subject-interval
# NOTE: update file_path to point to your own dataset

file_path = "data/input_data.csv"

if not DUMMY_DATA:
    data = pd.read_csv(file_path)

    # Preview the dataset
    data_sorted = data.sort_values(by="naid")
    print(data_sorted.head(10))
    print(data_sorted.shape)

# -----------------------------------------------
# Dummy dataset with time-varying covariates

# Set seed for reproducibility

else:
    np.random.seed(SEED)

    # Number of individuals
    n = 1000

    # Time-fixed variables
    genders = np.random.randint(0, 2, size=n)  # Gender: 0 or 1
    races = np.random.randint(1, 5, size=n)  # Race: 1, 2, 3, or 4

    # Generate follow-up times (tstart and tstop) and number of time intervals (rows of person-time)
    rows = []
    for i in range(n):
        num_rows = np.random.randint(2, 7)  # Random number of rows between 2 and 6
        tstart = np.random.randint(0, 365 * 5)  # Random start time (up to 5 years)

        person_rows = []
        for j in range(num_rows):
            if j == 0:
                tstart_j = tstart
            else:
                tstart_j = person_rows[j - 1]["tstop"] + 1
            tstop_j = tstart_j + np.random.randint(1, 365 * 2)

            # Randomly generate hemoglobin and livfibrosis without affecting failure probability
            hemoglobin_j = np.round(np.random.uniform(6, 20), 1)
            livfibrosis_j = np.round(np.random.uniform(0.3, 3.0), 2)

            # Failure probability (now independent of hemoglobin and livfibrosis)
            p_failure = 0.3  # For example, fixed probability of 0.3
            failure_j = np.random.binomial(1, p_failure)

            # Randomized association with failure
            subst_use_j = np.random.binomial(1, 0.9 if failure_j == 1 else 0.1)
            depression_j = np.random.binomial(1, 0.7 if failure_j == 1 else 0.3)

            person_rows.append(
                {
                    "naid": i + 1,
                    "tstart": tstart_j,
                    "tstop": tstop_j,
                    "failure": failure_j,
                    "gender": genders[i],
                    "race": races[i],
                    "depression": depression_j,
                    "subst_use": subst_use_j,
                    "hemoglobin": hemoglobin_j,
                    "livfibrosis": livfibrosis_j,
                }
            )

        rows.extend(person_rows)

    # Convert to DataFrame
    data = pd.DataFrame(rows)

    # Remove rows after the first failure for each individual
    def remove_rows_after_failure(group):
        first_failure_idx = group[group["failure"] == 1].index
        if len(first_failure_idx) > 0:
            first_failure_idx = first_failure_idx[0]
            group = group.loc[:first_failure_idx]
        return group

    data = (
        data.groupby("naid", group_keys=False)
        .apply(remove_rows_after_failure)
        .reset_index(drop=True)
    )

    # Preview the dataset
    data_sorted = data.sort_values(by="naid")
    print(data_sorted.head(10))
    print(data_sorted.shape)

# %%
# STEP 3
# Data cleaning

# Check feature types; make sure categorical features are coded as such (necessary for one hot encoding to work properly)
continuous_cols = [
    "naid",
    "tstart",
    "tstop",
    "failure",
    "hemoglobin",
    "livfibrosis",
]  # covariates that should NOT be one-hot encoded
categorical_cols = [col for col in data.columns if col not in continuous_cols]

data[continuous_cols] = data[continuous_cols].astype("float")
data[categorical_cols] = data[categorical_cols].astype("category")
# print(data.dtypes)

# One hot encode categorical features as integers
Xt = pd.get_dummies(data, drop_first=True).astype(int)
# print(Xt[:5])

# df_baseline: full dataset with only baseline covariates (no scaling)
df_sorted = Xt.sort_values(["naid", "tstart"])

baseline_rows = (
    df_sorted.groupby("naid", as_index=False).first()  # take first observed covariates
)

survival_info = (
    df_sorted.groupby("naid")
    .agg(duration=("tstop", "max"), event=("failure", "max"))
    .reset_index()
)

df_baseline = baseline_rows.drop(columns=["tstart", "tstop", "failure"]).merge(
    survival_info, on="naid"
)

# %%
# STEP 4
# Create dataset with final variables for Cox model with cross-validation
# Manually ensure that all levels of selected covariates are included in final_vars
# Create Xts_1fold - entire scaled dataset for generating coefficients

final_vars = [
    "naid",
    "duration",
    "event",
    "race_2",
    "race_3",
    "race_4",
    "age_cat_btwn30to39",
    "age_cat_btwn40to49",
    "age_cat_btwn50to59",
    "age_cat_btwn60to69",
    "age_cat_gte70",
    "riskgroup_IDU",
    "riskgroup_MSM",
    "riskgroup_Other",
    "riskgroup_Unknown",
    "art_ini_2",
    "art_ini_3",
    "art_ini_4",
    "cd4_ini_1",
    "cd4_ini_2",
    "cd4_ini_3",
    "cd4_1",
    "cd4_2",
    "cd4_3",
    "ritonavir_1",
    "cobicistat_1",
    "bictegravir_1",
    "darunavir_1",
    "dolutegravir_1",
    "raltegravir_1",
    "nevirapine_1",
    "tenofovir_1",
    "tenofovir_taf_1",
    "rilpivirine_1",
    "efavirenz_1",
    "hemoglobin",
    "livfibrosis",
    "hbv_1",
    "hcv_1",
    "smoking_1",
    "tc_lipidmed_1",
    "tc_lipidmed_2",
    "htmed_1",
    "anxiety_1",
    "bipolar_1",
    "depression_1",
    "psychosis_1",
    "schizophrenia_1",
    "coca_1",
    "mari_1",
    "other_1",
]

if not DUMMY_DATA:
    Xt = df_baseline[final_vars]

else:
    Xt = df_baseline

# separate continuous and categorical features so scaling is only applied to continuous features
continuous_cols = ["hemoglobin", "livfibrosis"]  # continuous covariates to scale
categorical_cols = [col for col in Xt.columns if col not in continuous_cols]

scaler = StandardScaler()

# 0. Isolate true continuous predictors from structural metadata
true_continuous = ["hemoglobin", "livfibrosis"]
meta_cols = ["naid", "duration", "event"]

# 1. Dynamically identify the one-hot encoded categorical columns remaining in df_baseline
encoded_categorical_cols = [
    col
    for col in df_baseline.columns
    if col not in true_continuous and col not in meta_cols
]

# 2. Scale strictly the continuous predictors
Xts_num = scaler.fit_transform(df_baseline[true_continuous])
Xts_num = Xts_num.astype("float32")
Xts_num = pd.DataFrame(Xts_num, columns=true_continuous, index=df_baseline.index)

# 3. Concatenate structural columns, scaled continuous features, and categorical features
Xts_1fold = pd.concat(
    [df_baseline[meta_cols], Xts_num, df_baseline[encoded_categorical_cols]], axis=1
)

# %%
# STEP 5
# Set up folds for k-fold cross-validation

n_folds = 10
n_splits = 10

# %%
# STEP 6
# Prepare data across each fold
# 1) split data into train and test sets (keep naids together); ensure similar distribution of censoring and follow-up times
# 2) standardize features
# 3) save each fold's train and test data for use with CV wrappers
# 4) print log-rank test and censoring rates of train/test data in each fold

# -----------------------------------------------
# Stratified CV at subject level

Xts_train_folds = []
Xts_test_folds = []
naid_train_folds = []
naid_test_folds = []
cv_results = []

# Define covariate types using baseline dataset
continuous_cols = ["hemoglobin", "livfibrosis"]

categorical_cols = [
    col
    for col in Xt.columns
    if col not in continuous_cols + ["naid", "duration", "event"]
]

y_strat = Xt["event"].values
naids = Xt["naid"].values

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(naids, y_strat)):
    print(f"\n=== Fold {fold_idx + 1} ===")

    train_df = Xt.iloc[train_idx].copy()
    test_df = Xt.iloc[test_idx].copy()

    # -----------------------------------------------
    # Standardize (fit on train only)
    scaler = StandardScaler()

    x_train_num = pd.DataFrame(
        scaler.fit_transform(train_df[continuous_cols]),
        columns=continuous_cols,
        index=train_df.index,
    )

    x_test_num = pd.DataFrame(
        scaler.transform(test_df[continuous_cols]),
        columns=continuous_cols,
        index=test_df.index,
    )

    Xts_train = pd.concat(
        [
            x_train_num,
            train_df[categorical_cols],
            train_df[["duration", "event", "naid"]],
        ],
        axis=1,
    )

    Xts_test = pd.concat(
        [x_test_num, test_df[categorical_cols], test_df[["duration", "event", "naid"]]],
        axis=1,
    )

    Xts_train_folds.append(Xts_train.copy())
    Xts_test_folds.append(Xts_test.copy())
    naid_train_folds.append(train_df[["naid"]].copy())
    naid_test_folds.append(test_df[["naid"]].copy())

    # -----------------------------------------------
    # Log-rank check
    results = logrank_test(
        train_df["duration"],
        test_df["duration"],
        event_observed_A=train_df["event"],
        event_observed_B=test_df["event"],
    )

    print(f"Log-rank test p-value: {results.p_value:.4f}")
    print(f"Training set event rate: {train_df['event'].mean():.2%}")
    print(f"Test set event rate: {test_df['event'].mean():.2%}")

# -----------------------------------------------
# Remove identifiers/outcomes before modeling
# Xts_train_folds_filtered = [
#     df.drop(columns=['naid', 'duration', 'event']) for df in Xts_train_folds
# ]

# Xts_test_folds_filtered = [
#     df.drop(columns=['naid', 'duration', 'event']) for df in Xts_test_folds
# ]

# # Feature names
# Xts_names = Xts_train_folds_filtered[0].columns.tolist()

# %%
# STEP 7
# Fit the initial Cox model

# Storage for each fold's model
cph_models = []

non_feature_cols = ["naid", "duration", "event"]  # columns not used as features

for fold in range(n_folds):
    print(f"\n=== Fitting Cox Model: Fold {fold + 1} ===")

    # Get training and test data for this fold
    Xts_train = Xts_train_folds[fold].copy()
    Xts_test = Xts_test_folds[fold].copy()  # test set for this fold

    # Drop non-feature columns before fitting Cox
    feature_cols = [col for col in Xts_train.columns if col not in non_feature_cols]
    Xts_train_features = Xts_train[feature_cols]

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(
        Xts_train_features.join(Xts_train[["duration", "event"]]),
        duration_col="duration",
        event_col="event",
    )

    # Save model and coefficients
    cph_models.append(cph)

    # Print diagnostics
    print(f"Partial log-likelihood: {cph.log_likelihood_:.4f}")
    print(f"Partial AIC: {cph.AIC_partial_:.4f}")

# %%
# STEP 8
# Feature selection using PowerSHAP manual implementation
# Optimize n_iterations
# Fix n_samples; increase n_iterations until null SHAP distribution variance stabilizes

tqdm.disable = True

# ----------------------------
# Parameters
iteration_grid = [15, 30, 50, 75, 100]
max_iterations = max(iteration_grid)
alpha = 0.05
background_size = 100
n_samples = 300
tol = 0.02
patience = 2
SEED = SEED

# ----------------------------
# Global feature list
feature_names_global = [
    col
    for col in Xts_train_folds[0].columns
    if col not in ["naid", "duration", "event"]
]

# Storage
all_shap_values = []
all_noise_values = []

# ----------------------------
# Loop over folds
for fold in range(len(Xts_train_folds)):
    print(f"\n===== Fold {fold + 1} =====")

    X_train_df = Xts_train_folds[fold].copy()
    X_test_df = Xts_test_folds[fold].copy()

    prev_checkpoint_std = None
    stable_count = 0
    stop_iteration = max_iterations

    fold_shap = []
    fold_noise = []

    # ------------------------
    # Iterations
    for i in range(max_iterations):
        iter_seed = SEED + i
        np.random.seed(iter_seed)
        random.seed(iter_seed)

        # ----- Prepare feature matrices only -----
        X_train_feat = X_train_df[feature_names_global].copy()
        X_test_feat = X_test_df[feature_names_global].copy()

        # Add random noise
        X_train_feat["noise"] = np.random.uniform(0, 1, len(X_train_feat))
        X_test_feat["noise"] = np.random.uniform(0, 1, len(X_test_feat))
        augmented_features = feature_names_global + ["noise"]

        # ----- Fit Cox model -----
        cph = CoxPHFitter()
        # Only join once at fit, not needed in predict
        X_train_fit = X_train_feat.join(X_train_df[["duration", "event"]])
        cph.fit(
            X_train_fit, duration_col="duration", event_col="event", show_progress=False
        )

        # ----- Background sample -----
        background_idx = np.random.choice(
            len(X_train_feat), min(background_size, len(X_train_feat)), replace=False
        )
        background = X_train_feat.iloc[background_idx]

        # ----- predict_fn -----
        def predict_fn(X):
            # X is feature matrix only (augmented_features)
            X_df = pd.DataFrame(X, columns=augmented_features)
            return cph.predict_log_partial_hazard(X_df).values

        # ----- Compute SHAP -----
        explainer = shap.Explainer(
            predict_fn, background, algorithm="permutation", verbose=False
        )
        shap_values = explainer(X_test_feat, max_evals=n_samples).values
        shap_abs = np.abs(shap_values)
        mean_per_feature = shap_abs.mean(axis=0)

        fold_shap.append(mean_per_feature[:-1])  # real features
        fold_noise.append(mean_per_feature[-1])  # noise

        # ----- Stabilization -----
        current_iter = i + 1
        if current_iter in iteration_grid:
            null_std = np.std(fold_noise)
            print(
                f"Checkpoint {current_iter}: null_std = {null_std:.6f}"
            )  # <-- print null_std

            if prev_checkpoint_std is not None:
                rel_change = abs(null_std - prev_checkpoint_std) / prev_checkpoint_std
                stable_count = stable_count + 1 if rel_change < tol else 0
                if stable_count >= patience:
                    stop_iteration = current_iter
                    print(f"✓ Early stopping triggered at iteration {current_iter}")
                    break
            prev_checkpoint_std = null_std

    print(
        f"Fold {fold + 1} finished. Total iterations used: {stop_iteration}"
    )  # <-- print final iterations

    # ------------------------
    # Store fold results
    all_shap_values.append(np.array(fold_shap))
    all_noise_values.append(np.array(fold_noise))

# ----------------------------
# Combine folds
all_shap_values = np.concatenate(all_shap_values, axis=0)
all_noise_values = np.concatenate(all_noise_values, axis=0)

# ----------------------------
# Vectorized percentile-based p-values
noise_sorted = np.sort(all_noise_values)
n_noise = len(noise_sorted)

global_pvals = np.array(
    [
        1
        - (
            np.searchsorted(noise_sorted, all_shap_values[:, i], side="right") / n_noise
        ).mean()
        for i in range(all_shap_values.shape[1])
    ]
)

# Multiple testing correction
reject, pvals_corrected, _, _ = multipletests(
    global_pvals, alpha=alpha, method="fdr_bh"
)

# Final global selected features
global_selected = [f for f, r in zip(feature_names_global, reject) if r]

print("\n===== Global Selected Features =====")
print(sorted(global_selected))

# Export
df_export = pd.DataFrame(
    {
        "feature": feature_names_global,
        "pval": global_pvals,
        "pval_corrected": pvals_corrected,
        "selected": reject,
    }
)
df_export.to_csv(
    f"{output_dir}{filename}_PowerSHAP_FeatureSelection_Results.csv", index=False
)

# %%
# STEP 9
# Expand feature selection such that all levels of a selected feature are included

final_vars_raw = global_selected

# Define groups of categorical variables
categorical_groups = [
    ["race_2", "race_3", "race_4"],
    [
        "age_cat_btwn30to39",
        "age_cat_btwn40to49",
        "age_cat_btwn50to59",
        "age_cat_btwn60to69",
        "age_cat_gte70",
    ],
    ["riskgroup_IDU", "riskgroup_MSM", "riskgroup_Other", "riskgroup_Unknown"],
    ["art_ini_2", "art_ini_3", "art_ini_4"],
    ["cd4_ini_1", "cd4_ini_2", "cd4_ini_3"],
    ["cd4_1", "cd4_2", "cd4_3"],
    ["tc_lipidmed_1", "tc_lipidmed_2"],
]


def expand_categorical(selected_features, groups, available_columns):
    expanded = list(selected_features)
    for group in groups:
        # If any level selected → include entire group
        if any(f in selected_features for f in group):
            for f in group:
                if f in available_columns and f not in expanded:
                    expanded.append(f)
    return expanded


final_vars_expanded = expand_categorical(final_vars_raw, categorical_groups, Xt.columns)

# Final feature set
selected_features = sorted(final_vars_expanded)

print(f"Selected before expansion: {len(final_vars_raw)}")
print(f"Selected after expansion:  {len(final_vars_expanded)}")
print("Expanded final variables:", selected_features)

# %%
# STEP 10
# Fit the final Cox model; export fold-specific coefficients and test set risk scores

# Storage for each fold's model, coefficients, and risk scores
cph_models = []
coef_long_list = []
all_test_risk_scores = []

for fold in range(n_folds):
    print(f"\n=== Fitting Cox Model (PowerSHAP-selected): Fold {fold + 1} ===")

    # ---------------------------------------------
    # Extract the corresponding training and test data
    Xts_train = Xts_train_folds[fold].copy()
    Xts_test = Xts_test_folds[fold].copy()

    train_df = Xts_train[selected_features + ["duration", "event"]]
    test_df = Xts_test[selected_features + ["duration", "event"]]

    # ---------------------------------------------
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col="duration", event_col="event")

    # ---------------------------------------------
    # Store model and coefficients
    cph_models.append(cph)

    coef_df = (
        cph.summary.reset_index()
        .loc[:, ["covariate", "coef", "coef lower 95%", "coef upper 95%"]]
        .rename(
            columns={
                "covariate": "feature",
                "coef": "log_coef",
                "coef lower 95%": "log_lower_95CI",
                "coef upper 95%": "log_upper_95CI",
            }
        )
    )

    # ---- Add exponentiated coefficients (Hazard Ratios) ----
    coef_df["HR"] = np.exp(coef_df["log_coef"])
    coef_df["HR_lower_95CI"] = np.exp(coef_df["log_lower_95CI"])
    coef_df["HR_upper_95CI"] = np.exp(coef_df["log_upper_95CI"])

    coef_df["fold"] = fold + 1
    coef_long_list.append(coef_df)

    # ---------------------------------------------
    # Compute risk scores for test data
    Xts_test_features = Xts_test[selected_features]
    log_risk = cph.predict_log_partial_hazard(Xts_test_features).values
    Xts_test = Xts_test.copy()
    Xts_test["log_risk"] = log_risk
    Xts_test["fold"] = fold + 1

    all_test_risk_scores.append(
        Xts_test[["naid", "duration", "event", "log_risk", "fold"]]
    )

    # ---------------------------------------------
    # Diagnostics
    print(f"Partial log-likelihood: {cph.log_likelihood_:.4f}")
    print(f"Partial AIC: {cph.AIC_partial_:.4f}")

# Combine and export
coef_long_df = pd.concat(coef_long_list, ignore_index=True)
coef_long_df.to_csv(f"{output_dir}{filename}_HazardRatios_ByFold.csv", index=False)

risk_scores_df = pd.concat(all_test_risk_scores, ignore_index=True)
risk_scores_df.to_csv(f"{output_dir}{filename}_LogRiskScores_ByFold.csv", index=False)

# %%
# STEP 11
# Report coefficients (95% CI) from the Cox model fit on entire dataset

# Subset the existing DataFrame instead of redefining it as a list
Xts_1fold_selected = Xts_1fold[selected_features + ["naid", "duration", "event"]].copy()

# Fit the model
cph_all = CoxPHFitter()
cph_all.fit(
    Xts_1fold_selected.drop(columns=["naid"], errors="ignore"),
    duration_col="duration",
    event_col="event",
)

coef_all = (
    cph_all.summary.reset_index()
    .loc[:, ["covariate", "coef", "coef lower 95%", "coef upper 95%"]]
    .rename(
        columns={
            "covariate": "feature",
            "coef": "log_coef",
            "coef lower 95%": "log_lower_95CI",
            "coef upper 95%": "log_upper_95CI",
        }
    )
)

# Add exponentiated coefficients (Hazard Ratios)
coef_all["HR"] = np.exp(coef_all["log_coef"])
coef_all["HR_lower_95CI"] = np.exp(coef_all["log_lower_95CI"])
coef_all["HR_upper_95CI"] = np.exp(coef_all["log_upper_95CI"])

# Print diagnostics
print(f"Partial log-likelihood: {cph_all.log_likelihood_:.4f}")
print(f"Partial AIC: {cph_all.AIC_partial_:.4f}")

# Export coefficients and 95% CI
coef_all.to_csv(f"{output_dir}{filename}_HazardRatios_FullDataset.csv", index=False)


# %%

import numpy as np
import pandas as pd

categorical_groups = [
    ["race_2", "race_3", "race_4"],
    [
        "age_cat_btwn30to39",
        "age_cat_btwn40to49",
        "age_cat_btwn50to59",
        "age_cat_btwn60to69",
        "age_cat_gte70",
    ],
    ["riskgroup_IDU", "riskgroup_MSM", "riskgroup_Other", "riskgroup_Unknown"],
    ["art_ini_2", "art_ini_3", "art_ini_4"],
    ["cd4_ini_1", "cd4_ini_2", "cd4_ini_3"],
    ["cd4_1", "cd4_2", "cd4_3"],
    ["tc_lipidmed_1", "tc_lipidmed_2"],
]


def extract_average_marginal_effects(
    model, pandas_data, index_to_name, categorical_groups=None, is_deepsurv=False
):
    """
    Calculates the Average Marginal Effect (AME) on the log-risk scale.
    Yields the exact coefficient for Cox models and the expected local gradient for DeepSurv.
    Safely handles one-hot encoded categorical groups to prevent out-of-distribution artifacts.

    Parameters:
    - model: The fitted survival model (lifelines.CoxPHFitter or pycox DeepSurv).
    - pandas_data: DataFrame containing the evaluation data.
    - index_to_name: Dictionary mapping column indices to feature names.
    - categorical_groups: List of lists containing sibling dummy variables.
    - is_deepsurv: Boolean flag to switch prediction APIs.
    """
    if categorical_groups is None:
        categorical_groups = []

    coefficients = {}
    n_samples, n_features = pandas_data.shape

    # 1. API Wrapper to unify lifelines and DeepSurv predictions
    def predict_risk(X_input):
        if is_deepsurv:
            X_tensor = torch.tensor(
                X_input.to_numpy().astype(np.float32), dtype=torch.float32
            )

            model.eval()

            with torch.no_grad():
                risk = model(X_tensor)

            return risk.cpu().numpy().flatten()
        else:
            return model.predict_log_partial_hazard(X_input).to_numpy().flatten()

    # 2. Baseline predictions required for the continuous finite difference
    baseline_preds = predict_risk(pandas_data)

    for i in range(n_features):
        feature_name = index_to_name[i]
        col_data = pandas_data.iloc[:, i]

        # Heuristic to identify if the variable is binary (e.g., one-hot encoded)
        unique_vals = col_data.dropna().unique()
        is_binary = set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset(
            {0.0, 1.0}
        )

        if is_binary:
            # Find sibling dummy variables (if any) in the categorical groups
            sibling_features = []
            for group in categorical_groups:
                if feature_name in group:
                    # Isolate siblings that are actually present in the dataframe
                    sibling_features = [
                        f
                        for f in group
                        if f != feature_name and f in pandas_data.columns
                    ]
                    break

            # State 1: Target category is ON, all siblings are OFF
            df_1 = pandas_data.copy()
            df_1.iloc[:, i] = 1.0
            for sibling in sibling_features:
                df_1[sibling] = 0.0

            # State 0: Target category is OFF, all siblings are OFF (Revert to Reference Category)
            df_0 = pandas_data.copy()
            df_0.iloc[:, i] = 0.0
            for sibling in sibling_features:
                df_0[sibling] = 0.0

            preds_1 = predict_risk(df_1)
            preds_0 = predict_risk(df_0)

            # The expected value of the individual differences
            ame = np.mean(preds_1 - preds_0)

        else:
            # Continuous Estimand: Expected Partial Derivative via Finite Difference
            # E[ (f(X_{-j}, X_j + epsilon) - f(X)) / epsilon ]
            epsilon = 1e-4

            df_perturbed = pandas_data.copy()
            df_perturbed.iloc[:, i] = col_data + epsilon

            preds_perturbed = predict_risk(df_perturbed)

            # Calculate the gradient per individual, then compute the global mean
            gradients = (preds_perturbed - baseline_preds) / epsilon
            ame = np.mean(gradients)

        coefficients[feature_name] = ame

    return coefficients


# %%
# STEP 12
# Produce SHAP values for each fold's testing data

# -----------------------------------------------
# Map feature indices to names for coefficient extraction
original_feature_names = Xts_test_folds[0].columns.tolist()
index_to_name = {i: name for i, name in enumerate(selected_features)}

# Containers
all_folds_shap_df = []
mean_abs_shap_list = []
feature_coefficients_df = []

for fold in range(n_folds):
    print(f"\n===== Computing SHAP values for fold {fold + 1}/{n_folds} =====")

    # ------------------------------------------------------------
    # Extract test data for this fold with only selected features
    Xts_test = Xts_test_folds[fold].copy()
    X_test_features = Xts_test.loc[:, selected_features].copy()

    # Keep identifiers separate
    naid_values = Xts_test["naid"].values

    # Corresponding fitted Cox model
    cph = cph_models[fold]

    # ------------------------------------------------------------
    # Define prediction function for SHAP
    def predict_risk(X_input, model=cph, features=selected_features):
        X_input_df = pd.DataFrame(X_input, columns=features)
        X_input_df = X_input_df.loc[:, features]
        return model.predict_log_partial_hazard(X_input_df).values.flatten()

    # ------------------------------------------------------------
    # Initialize SHAP Explainer (model-agnostic)
    X_train_background = Xts_train_folds[fold].loc[:, selected_features].copy()
    X_train_background = X_train_background[selected_features]

    explainer = shap.Explainer(predict_risk, X_train_background)

    # Compute SHAP values
    shap_values = explainer(X_test_features)

    # ------------------------------------------------------------
    # Store SHAP values in DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=selected_features)
    shap_df["expected_value"] = shap_values.base_values
    shap_df["naid"] = naid_values
    shap_df["fold"] = fold + 1

    all_folds_shap_df.append(shap_df)

    # ------------------------------------------------------------
    # Compute mean absolute SHAP per feature for this fold
    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=selected_features,
        name=f"Fold{fold + 1} Mean Abs SHAP",
    ).sort_values(ascending=False)

    mean_abs_shap_list.append(mean_abs_shap)

    print(f"Fold {fold + 1} top 10 mean absolute SHAP values:")
    print(mean_abs_shap.head(10))

    # ----- Extract feature coefficients using exact marginalization ---
    coefficients = extract_average_marginal_effects(cph, X_test_features, index_to_name)

    coefficients_df = pd.DataFrame(
        {
            "feature": list(coefficients.keys()),
            "coefficient": list(coefficients.values()),
            "fold": fold + 1,
        }
    )
    feature_coefficients_df.append(coefficients_df)

# ==========================================================
# Combine all folds’ per-sample SHAP values
combined_shap_df = pd.concat(all_folds_shap_df, ignore_index=True)

# Export combined per-sample SHAP values
combined_shap_df.to_csv(f"{output_dir}{filename}_SHAPValues.csv", index=False)

# ==========================================================
# Combine mean absolute SHAP values per fold
mean_abs_shap_df = pd.concat(mean_abs_shap_list, axis=1).fillna(0.0)
mean_abs_shap_df.index.name = "feature"

ranked_shap_df = mean_abs_shap_df.copy()

# Rank all features in each fold (highest SHAP = rank 1)
for col in ranked_shap_df.columns:
    ranked_shap_df[col + "_rank"] = ranked_shap_df[col].rank(
        ascending=False,  # highest SHAP = rank 1
        method="min",  # ties get same minimum rank
    )

# Export ranked mean absolute SHAP values
ranked_shap_df.to_csv(f"{output_dir}{filename}_MeanAbsSHAPValues.csv")

# ----------------------------------------------
# Combine feature coefficients from all folds
combined_coefficients_df = pd.concat(feature_coefficients_df, ignore_index=True)
combined_coefficients_df.to_csv(
    f"{output_dir}{filename}_FeatureCoefficients.csv", index=False
)

# %%
# STEP 13
# Concordance index (c-index) (0-1; higher is better)

# USING SKLEARN CONCORDANCE_INDEX_IPCW

# C-index: proportion of concordant pairs among all comparable pairs; concordant if model assigns
# higher risk to the observation with shorter survival time

################################
# Parameters and storage
n_folds = len(cph_models)
cindex_results = []

################################
# Estimate tau from the full training data
all_X_train = pd.concat(Xts_train_folds, ignore_index=True)
surv_train_all = Surv.from_arrays(
    all_X_train["event"].astype(bool), all_X_train["duration"]
)

cens_global = CensoringDistributionEstimator()
cens_global.fit(surv_train_all)

valid_times_global = [
    t for t in all_X_train["duration"] if cens_global.predict_proba([t])[0] > 0
]
tau_global = min(all_X_train["duration"].quantile(0.95), max(valid_times_global))

print(f"Global tau (IPCW): {tau_global:.1f} days")

################################
# Loop C-index through folds
for fold in range(n_folds):
    print(f"\n=== C-index Evaluation: Fold {fold + 1} ===")

    Xts_test = all_test_risk_scores[fold].copy()
    Xts_train = Xts_train_folds[fold].copy()

    surv_train = Surv.from_arrays(
        Xts_train["event"].astype(bool), Xts_train["duration"].values
    )

    surv_test_all = Surv.from_arrays(
        Xts_test["event"].astype(bool), Xts_test["duration"].values
    )

    partial_hazards_test = np.exp(Xts_test["log_risk"].values)

    ################################
    # Point Estimates
    t1 = 365.25
    t5 = 5 * 365.25

    overall_cindex = concordance_index_ipcw(
        surv_train, surv_test_all, partial_hazards_test, tau=tau_global
    )[0]

    cindex_1yr = concordance_index_ipcw(
        surv_train, surv_test_all, partial_hazards_test, tau=t1
    )[0]

    cindex_5yr = concordance_index_ipcw(
        surv_train, surv_test_all, partial_hazards_test, tau=t5
    )[0]

    print(f"Fold {fold + 1} Overall C-index: {overall_cindex:.4f}")
    print(f"Fold {fold + 1} 1-Year C-index: {cindex_1yr:.4f}")
    print(f"Fold {fold + 1} 5-Year C-index: {cindex_5yr:.4f}")

    ################################
    # Save per-fold results
    cindex_results.append(
        {
            "fold": fold + 1,
            "cindex_overall": overall_cindex,
            "cindex_1yr": cindex_1yr,
            "cindex_5yr": cindex_5yr,
        }
    )

################################
# Create dataframe
cindex_df = pd.DataFrame(cindex_results)


################################
# Across-fold summary with Nadeau-Bengio correction
def nadeau_bengio_ci(values, k, confidence=0.95):
    # mean across folds
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    k = len(arr)
    mean_val = np.mean(arr)

    if k < 2:
        return mean_val, np.nan, np.nan, np.nan

    # sample variance across folds
    s2 = np.var(arr, ddof=1)

    # corrected variance
    corrected_var = (1 / k + 1 / (k - 1)) * s2
    corrected_se = np.sqrt(corrected_var)

    # t critical value
    t_crit = t_dist.ppf((1 + confidence) / 2, df=k - 1)
    margin = t_crit * corrected_se

    lower = mean_val - margin
    upper = mean_val + margin

    return mean_val, lower, upper, corrected_se


overall_mean, overall_lower, overall_upper, overall_se = nadeau_bengio_ci(
    cindex_df["cindex_overall"], k=n_folds
)

one_yr_mean, one_yr_lower, one_yr_upper, one_yr_se = nadeau_bengio_ci(
    cindex_df["cindex_1yr"], k=n_folds
)

five_yr_mean, five_yr_lower, five_yr_upper, five_yr_se = nadeau_bengio_ci(
    cindex_df["cindex_5yr"], k=n_folds
)

summary_df = pd.DataFrame(
    {
        "metric": ["Overall C-index", "C-index at 1 Year", "C-index at 5 Years"],
        "mean": [overall_mean, one_yr_mean, five_yr_mean],
        "lower_95CI": [overall_lower, one_yr_lower, five_yr_lower],
        "upper_95CI": [overall_upper, one_yr_upper, five_yr_upper],
        "nb_corrected_se": [overall_se, one_yr_se, five_yr_se],
    }
)

print("\n=== C-Index Summary Across Folds ===")
print(summary_df)

################################
# Export everything to one Excel file
excel_path = f"{output_dir}{filename}_CIndex.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    cindex_df.to_excel(writer, sheet_name="PerFold_Summary", index=False)
    summary_df.to_excel(writer, sheet_name="AcrossFold_Summary", index=False)

# %%
# STEP 14
# Integrated Brier Score (IBS) (0-1; lower is better)

# USING SKSURV INTEGRATED_BRIER_SCORE

# IBS: average of mean squared error between predicted survival probability and actual outcome
# Produce survival probabilities across time grid

# --------------------------------------------------
# Containers
# --------------------------------------------------
brier_summary_results = []
brier_time_results = []

ibs_list = []
ibs_null_list = []
ibs_gain_list = []

# --------------------------------------------------
# Global evaluation grid (clip for folds)
# --------------------------------------------------
global_eval_times = np.linspace(0, 5000, 100)

# 5000 chosen for this dataset based on distribution of observed
# follow-up times - update accordingly for other applications

# --------------------------------------------------
# Cross-validation loop
# --------------------------------------------------
for fold in range(n_folds):
    print(f"\n=== Brier Score Evaluation: Fold {fold + 1} ===")

    X_train = Xts_train_folds[fold].copy()
    X_test = all_test_risk_scores[fold].copy()
    cph = cph_models[fold]

    surv_train = Surv.from_arrays(X_train["event"].astype(bool), X_train["duration"])
    surv_test = Surv.from_arrays(X_test["event"].astype(bool), X_test["duration"])

    fold_eval_times = global_eval_times[
        (global_eval_times >= X_test["duration"].min())
        & (global_eval_times <= X_test["duration"].max())
    ]

    base_chf = cph.baseline_cumulative_hazard_
    H0_fn = interp1d(
        base_chf.index.values,
        base_chf.values.flatten(),
        kind="previous",
        bounds_error=False,
        fill_value=(0, base_chf.values.flatten()[-1]),
    )
    H0_eval = H0_fn(fold_eval_times)

    risk = np.exp(X_test["log_risk"].values)
    surv_probs = np.exp(-np.outer(risk, H0_eval))

    km_time, km_surv = kaplan_meier_estimator(
        X_train["event"].astype(bool), X_train["duration"]
    )
    surv_null = np.tile(np.interp(fold_eval_times, km_time, km_surv), (len(X_test), 1))

    # -------------------------------
    # Point estimates
    # -------------------------------
    ibs = integrated_brier_score(surv_train, surv_test, surv_probs, fold_eval_times)
    ibs_null = integrated_brier_score(surv_train, surv_test, surv_null, fold_eval_times)
    ibs_gain = ibs_null - ibs

    times_bs, scores_bs = brier_score(
        surv_train, surv_test, surv_probs, fold_eval_times
    )

    idx_1yr = np.abs(times_bs - 365.25).argmin()
    idx_5yr = np.abs(times_bs - 365.25 * 5).argmin()

    brier_1yr = scores_bs[idx_1yr]
    brier_5yr = scores_bs[idx_5yr]

    print(f"Fold {fold + 1} - IBS: {ibs:.4f}")
    print(f"Fold {fold + 1} - 1-year: {brier_1yr:.4f}")
    print(f"Fold {fold + 1} - 5-year: {brier_5yr:.4f}")

    # Save fold summary
    brier_summary_results.append(
        {
            "fold": fold + 1,
            "ibs": ibs,
            "ibs_null": ibs_null,
            "ibs_gain": ibs_gain,
            "brier_1yr": brier_1yr,
            "brier_5yr": brier_5yr,
        }
    )

    ibs_list.append(ibs)
    ibs_null_list.append(ibs_null)
    ibs_gain_list.append(ibs_gain)

    # Per-time results
    for t, s in zip(times_bs, scores_bs):
        brier_time_results.append(
            {
                "fold": fold + 1,
                "time": round(t, 3),
                "time_years": round(t / 365.25, 3),
                "brier_score": s,
            }
        )

# --------------------------------------------------
# Create dataframes
# --------------------------------------------------
brier_summary_df = pd.DataFrame(brier_summary_results)
brier_time_df = pd.DataFrame(brier_time_results)

# --------------------------------------------------
# Across-fold summary (mean + t CI)
# --------------------------------------------------
k = n_folds

ibs_m, ibs_l, ibs_u, ibs_se = nadeau_bengio_ci(ibs_list, k=n_folds)
null_m, null_l, null_u, null_se = nadeau_bengio_ci(ibs_null_list, k=n_folds)
gain_m, gain_l, gain_u, gain_se = nadeau_bengio_ci(ibs_gain_list, k=n_folds)
b1_m, b1_l, b1_u, b1_se = nadeau_bengio_ci(brier_summary_df["brier_1yr"], k=n_folds)
b5_m, b5_l, b5_u, b5_se = nadeau_bengio_ci(brier_summary_df["brier_5yr"], k=n_folds)

summary_df = pd.DataFrame(
    {
        "metric": ["ibs", "ibs_null", "ibs_gain", "brier_1yr", "brier_5yr"],
        "mean": [ibs_m, null_m, gain_m, b1_m, b5_m],
        "lower_95CI": [ibs_l, null_l, gain_l, b1_l, b5_l],
        "upper_95CI": [ibs_u, null_u, gain_u, b1_u, b5_u],
    }
)

print("\n=== Brier Score Summary Across Folds ===")
print(summary_df)

# --------------------------------------------------
# Export everything to one Excel file
# --------------------------------------------------
excel_path = f"{output_dir}{filename}_Brier.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    brier_summary_df.to_excel(writer, sheet_name="PerFold_Summary", index=False)
    brier_time_df.to_excel(writer, sheet_name="Brier_byTime", index=False)
    summary_df.to_excel(writer, sheet_name="AcrossFold_Summary", index=False)

# %%
# STEP 15
# Population-average survival function for full dataset
# Marginal survival curve averaged over covariate distribution in full dataset
# 95% CI reflects uncertainty due to covariate heterogeneity, not model parameter uncertainty

# ----------------------------------------------
# Define the common time grid for interpolation
min_time = Xts_1fold["duration"].min()
max_time = Xts_1fold["duration"].max()

time_grid = np.linspace(min_time, max_time - 1e-3, 100)

# ----------------------------------------------
# Use single fitted Cox model
cph_all = CoxPHFitter()

# Explicitly include the survival time and outcome vectors in the matrix
cph_all.fit(
    Xts_1fold[selected_features + ["duration", "event"]].drop(
        columns=["naid"], errors="ignore"
    ),
    duration_col="duration",
    event_col="event",
)

baseline_cumhaz = cph_all.baseline_cumulative_hazard_

# Interpolate baseline cumulative hazard
cumhaz_interp = np.interp(
    time_grid, baseline_cumhaz.index.to_numpy(), baseline_cumhaz.iloc[:, 0].to_numpy()
)

# --------------------------------------------------
# Individual risk scores (time-fixed)
# --------------------------------------------------
eta = cph_all.predict_log_partial_hazard(
    Xts_1fold[selected_features].drop(columns=["naid"], errors="ignore")
).values.flatten()
ph = np.exp(eta)

# --------------------------------------------------
# Survival matrix: S(t) = exp(-H0(t) * exp(eta))
# --------------------------------------------------
surv_matrix = np.exp(-np.outer(cumhaz_interp, ph))

# Population-average survival and SE, 95% CI
mean_survival = surv_matrix.mean(axis=1)
se_survival = surv_matrix.std(axis=1, ddof=1) / np.sqrt(surv_matrix.shape[1])

z = 1.96
lower_ci = mean_survival - z * se_survival
upper_ci = mean_survival + z * se_survival

# enforce valid survival bounds
lower_ci = np.clip(lower_ci, 0, 1)
upper_ci = np.clip(upper_ci, 0, 1)

# ----------------------------------------------
# Store results

surv_df = pd.DataFrame(
    {
        "time": time_grid,
        "mean_survival": mean_survival,
        "lower_95CI": lower_ci,
        "upper_95CI": upper_ci,
    }
)

output_csv_path = f"{output_dir}{filename}_PopAvgSurvival.csv"
surv_df.to_csv(output_csv_path, index=False)

# ----------------------------------------------
# Plot survival curve

plt.figure(figsize=(8, 5))

plt.plot(time_grid / 365.25, mean_survival, label="Population-average survival")

plt.fill_between(
    time_grid / 365.25,
    lower_ci,
    upper_ci,
    alpha=0.25,
    label="±1 SE across covariate distribution",
)

plt.title("Cox (Powershap): Population-Average Survival Function")
plt.xlabel("Years from suppression after ART initiation")
plt.ylabel("Survival probability")
plt.grid(True)
plt.legend()

output_surv_path = f"{output_dir}{filename}_SurvPlot.png"
plt.savefig(output_surv_path, dpi=300, bbox_inches="tight")
plt.show()
