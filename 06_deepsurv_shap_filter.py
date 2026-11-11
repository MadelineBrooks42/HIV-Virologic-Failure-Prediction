# %% [markdown]
# # DeepSurv with SHAP-based feature filtering
#
# **Script:** `06_deepsurv_shap_filter.py`  
# **Model:** DeepSurv neural network (PyTorch / `torchtuples`)  
# **Output file prefix:** `deepsurv_shap_filter`
#
# ## Purpose
#
# - DeepSurv with SHAP filter for feature selection
# - PyTorch for DeepSurv (standard neural network; AdamW optimizer)
# - Optimization of hyperparameters and activation function
# - Implements K-fold cross-validation
# - Feature selection: features are ranked by SHAP value within each fold, and features whose median rank across folds is at or below the 25th percentile of median ranks are kept as one global feature set
#
# ## Input
#
# A CSV in start-stop (person-period) format, read in STEP 2 from `file_path`. Columns used by the code include `naid` (subject ID), `tstart`, `tstop`, `failure` (event indicator) and the covariates. Set `file_path` to your own dataset.
#
# ## Outputs
#
# Written to `output_dir` as `deepsurv_shap_filter` + the suffix below.
#
# - `deepsurv_shap_filter_NestedCV_InitialGlobalSelection.xlsx`
# - `deepsurv_shap_filter_SHAP_FeatureSelection_Results.xlsx`
# - `deepsurv_shap_filter_NestedCV_FinalGlobalSelection.xlsx`
# - `deepsurv_shap_filter_Fold{fold`
# - `deepsurv_shap_filter_LogRiskScores.csv`
# - `deepsurv_shap_filter_TrainingLoss.png`
# - `deepsurv_shap_filter_SHAPValues.csv`
# - `deepsurv_shap_filter_MeanAbsSHAPValues.csv`
# - `deepsurv_shap_filter_FeatureCoefficients.csv`
# - `deepsurv_shap_filter_CIndex.xlsx`
# - `deepsurv_shap_filter_Brier.xlsx`
# - `deepsurv_shap_filter_PopulationAverageSurvival.csv`
# - `deepsurv_shap_filter_PopulationAverageSurvival.png`
#
# ## Configuration
#
# Values as set in this script. `SEED`, `output_dir` and `filename` are set in STEP 1 and `file_path` in STEP 2; the remaining values are set in STEP 1 or where cross-validation is configured.
#
# | Variable | Value in this script | Meaning |
# |---|---|---|
# | `SEED` | `12345` | Random seed for reproducibility |
# | `EPOCHS` | `10000` | Upper bound on DeepSurv training epochs for the final per-fold models |
# | `EARLY_STOP_PATIENCE` | `200` | Epochs without improvement before training stops (set equal to `EPOCHS` to disable early stopping) |
# | `EARLY_STOP_MIN_DELTA` | `0.0001` | Minimum loss improvement that counts as progress for early stopping |
# | `GRID_SEARCH_EPOCHS` | `2000` | Epoch cap for the hyperparameter grid search only |
# | `GRID_SEARCH_FOLDS` | `3` | Number of outer folds the hyperparameter grid is scored on (capped at `n_folds`) |
# | `SHAP_BACKGROUND_SIZE` | `100` | Number of training rows used as the SHAP background |
# | `output_dir` | `'output/'` | Folder for exported files (must end with `/` and must already exist) |
# | `filename` | `'deepsurv_shap_filter'` | Prefix added to every exported file name |
# | `file_path` | `'data/input_data.csv'` | Input CSV (STEP 2) |
# | `n_folds` | `10` | Number of cross-validation folds |
# | `n_splits` | `10` | Number of cross-validation splits passed to the fold generator |
# | `INNER_VAL_FRACTION` | `0.2` | Fraction of each selection fold's training data held out for validation during hyperparameter selection |
#
# ## Requirements
#
# Python packages imported by this script: `lifelines`, `matplotlib`, `numpy`, `openpyxl`, `pandas`, `pycox`, `scikit-learn`, `scikit-survival`, `scipy`, `shap`, `statsmodels`, `torch`, `torchtuples`, `tqdm`.

# %%
# Library imports

import random
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import os
import shap
import contextlib
import io
import warnings
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt

from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator

from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from pycox.evaluation import EvalSurv
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import CoxTime

from scipy.stats import wilcoxon
from scipy.interpolate import interp1d
from scipy.stats import t as t_dist
from scipy.stats import norm

from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from torchtuples.optim import AdamW
from itertools import combinations

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

EPOCHS = 10000

# Early stopping: halt training once the loss has failed to improve by more than
# EARLY_STOP_MIN_DELTA for EARLY_STOP_PATIENCE consecutive epochs. EPOCHS stays
# the upper bound. Set EARLY_STOP_PATIENCE = EPOCHS to disable.
EARLY_STOP_PATIENCE = 200
EARLY_STOP_MIN_DELTA = 1e-4

# Epoch cap for the hyperparameter grid search only. The search just needs to
# rank combinations, not fully converge each one, so it runs a shorter budget
# than the EPOCHS cap used to fit the final per-fold models.
GRID_SEARCH_EPOCHS = 2000

# Number of outer folds the hyperparameter grid is scored on. Selection produces
# ONE global parameter set that every outer fold reuses, so scoring the grid on a
# subset of the folds gives essentially the same choice at a fraction of the cost
# and keeps selection time independent of n_folds. Capped at n_folds below; set
# equal to n_folds to score the grid on every fold.
GRID_SEARCH_FOLDS = 3

# Number of training rows used as the SHAP background (masker). The permutation
# explainer evaluates the model over the whole background for every masked
# evaluation, so subsampling keeps SHAP cost independent of training-fold size.
SHAP_BACKGROUND_SIZE = 100

# Set output folder and file name prefix for exports
# NOTE: output_dir must end with "/" and must exist before running
output_dir = "output/"
filename = "deepsurv_shap_filter"

# %%
# STEP 2
# Load data

# ----------------------------------------------
# Input data: start-stop (person-period) format, one row per subject-interval
# NOTE: update file_path to point to your own dataset

file_path = "data/input_data.csv"

data = pd.read_csv(file_path)

# # Preview the dataset
# data_sorted = data.sort_values(by='naid')
# print(data_sorted.head(10))
# print(data_sorted.shape)

# -----------------------------------------------
# Dummy dataset with time-varying covariates

# # Set seed for reproducibility
# np.random.seed(SEED)

# # Number of individuals
# n = 800

# # Time-fixed variables
# genders = np.random.randint(0, 2, size=n)  # Gender: 0 or 1
# races = np.random.randint(1, 5, size=n)  # Race: 1, 2, 3, or 4

# # Generate follow-up times (tstart and tstop) and number of time intervals (rows of person-time)
# rows = []
# for i in range(n):
#     num_rows = np.random.randint(2, 7)  # Random number of rows between 2 and 6
#     tstart = np.random.randint(0, 365 * 5)  # Random start time (up to 5 years)

#     person_rows = []
#     for j in range(num_rows):
#         if j == 0:
#             tstart_j = tstart
#         else:
#             tstart_j = person_rows[j-1]['tstop'] + 1
#         tstop_j = tstart_j + np.random.randint(1, 365 * 2)

#         # Randomly generate hemoglobin and livfibrosis without affecting failure probability
#         hemoglobin_j = np.round(np.random.uniform(6, 20), 1)
#         livfibrosis_j = np.round(np.random.uniform(0.3, 3.0), 2)

#         # Failure probability (now independent of hemoglobin and livfibrosis)
#         p_failure = 0.3  # For example, fixed probability of 0.3
#         failure_j = np.random.binomial(1, p_failure)

#         # Randomized association with failure
#         subst_use_j = np.random.binomial(1, 0.9 if failure_j == 1 else 0.1)
#         depression_j = np.random.binomial(1, 0.7 if failure_j == 1 else 0.3)

#         person_rows.append({
#             'naid': i + 1,
#             'tstart': tstart_j,
#             'tstop': tstop_j,
#             'failure': failure_j,
#             'gender': genders[i],
#             'race': races[i],
#             'depression': depression_j,
#             'subst_use': subst_use_j,
#             'hemoglobin': hemoglobin_j,
#             'livfibrosis': livfibrosis_j
#         })

#     rows.extend(person_rows)

# # Convert to DataFrame
# data = pd.DataFrame(rows)

# # Remove rows after the first failure for each individual
# def remove_rows_after_failure(group):
#     first_failure_idx = group[group['failure'] == 1].index
#     if len(first_failure_idx) > 0:
#         first_failure_idx = first_failure_idx[0]
#         group = group.loc[:first_failure_idx]
#     return group

# data = data.groupby('naid', group_keys=False).apply(remove_rows_after_failure).reset_index(drop=True)

# # Preview the dataset
# # data_sorted = data.sort_values(by='naid')
# # print(data_sorted.head(10))
# # print(data_sorted.shape)

# %%
# STEP 3
# Convert to time-fixed dataset; one-hot encode categorical features

# df_baseline: full single-fold dataset with only baseline covariates (no scaling)
df_sorted = data.sort_values(["naid", "tstart"])

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

# Check feature types
continuous_cols = [
    "naid",
    "duration",
    "event",
    "hemoglobin",
    "livfibrosis",
]  # covariates that should NOT be one-hot encoded
categorical_cols = [col for col in df_baseline.columns if col not in continuous_cols]

df_baseline[continuous_cols] = df_baseline[continuous_cols].astype("float")
df_baseline[categorical_cols] = df_baseline[categorical_cols].astype("category")
# print(data.dtypes)

# One hot encode categorical features as integers
Xt = pd.get_dummies(df_baseline, drop_first=True).astype(int)
# print(Xt[:5])

# %%
# STEP 4
# Create dataset with feature subset for DeepSurv with cross-validation

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

Xt = Xt[final_vars]

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
# 4) print log-rank test and event rates of train/test data in each fold

# -----------------------------------------------
# Stratified CV at subject level

Xts_train_folds = []
Xts_test_folds = []
naid_train_folds = []
naid_test_folds = []
cv_results = []
scalers_folds = []  # Initialize the list to hold the scalers

# Define covariate types using baseline dataset
continuous_cols = ["hemoglobin", "livfibrosis"]

categorical_cols = [
    col
    for col in Xt.columns
    if col not in continuous_cols + ["naid", "duration", "event"]
]

y_strat = Xt["event"].values
naids = Xt["naid"].values

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(naids, y_strat)):
    print(f"\n=== Fold {fold_idx + 1} ===")

    train_df = Xt.iloc[train_idx].copy()
    test_df = Xt.iloc[test_idx].copy()

    # -----------------------------------------------
    # Standardize (fit on train only)

    fold_scaler = (
        StandardScaler()
    )  # Create a unique scaler object for this specific fold

    x_train_num = pd.DataFrame(
        fold_scaler.fit_transform(train_df[continuous_cols]),  # Use fold_scaler here
        columns=continuous_cols,
        index=train_df.index,
    )

    x_test_num = pd.DataFrame(
        fold_scaler.transform(test_df[continuous_cols]),  # Use fold_scaler here
        columns=continuous_cols,
        index=test_df.index,
    )

    scalers_folds.append(fold_scaler)  # Store the fitted scaler for use in Step 14

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
Xts_train_folds_filtered = [
    df.drop(columns=["naid", "duration", "event"]) for df in Xts_train_folds
]

Xts_test_folds_filtered = [
    df.drop(columns=["naid", "duration", "event"]) for df in Xts_test_folds
]

# Feature names
Xts_names = Xts_train_folds_filtered[0].columns.tolist()

# %%
# STEP 7
# Extract time and event for each dataset across folds

# -----------------------------------------------
# Xts_train and Xts_test datasets

# Initialize lists to hold y and event for train and test folds
y_train_folds = []
event_train_folds = []
y_test_folds = []
event_test_folds = []

# Loop over folds
for i in range(n_folds):
    train_df = Xts_train_folds[i]
    test_df = Xts_test_folds[i]

    # Use duration and event (time-fixed dataset)
    y_train_folds.append(train_df["duration"].values.astype("float32"))
    event_train_folds.append(train_df["event"].values.astype("int32"))

    y_test_folds.append(test_df["duration"].values.astype("float32"))
    event_test_folds.append(test_df["event"].values.astype("int32"))

# Assign to global environment
for i in range(n_folds):
    globals()[f"y_train_{i + 1}"] = y_train_folds[i]
    globals()[f"event_train_{i + 1}"] = event_train_folds[i]

    globals()[f"y_test_{i + 1}"] = y_test_folds[i]
    globals()[f"event_test_{i + 1}"] = event_test_folds[i]

# -----------------------------------------------
# Xt dataset

y_Xt = Xt["duration"].values.astype("float32")
event_Xt = Xt["event"].values.astype("int32")

# %%
# STEP 8
# Initial hyperparameter optimization using nested CV

# ----------------------------------------------
# Reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------------------------
# Hyperparameter grid
param_grid = {
    "num_nodes": [32, 64, 128],
    "num_layers": [1, 2, 3],
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "dropout": [0.1, 0.2, 0.3],
    "l2_reg": [1e-2, 1e-4, 1e-5],
    "activation": ["relu", "tanh", "sigmoid"],
}

# smaller grid for testing purposes
# param_grid = {
#     'num_nodes': [32, 64],
#     'num_layers': [1, 2],
#     'learning_rate': [1e-3, 1e-4],
#     'dropout': [0.1, 0.2],
#     'l2_reg': [1e-2, 1e-4],
#     'activation': ['relu', 'tanh']
# }

param_names = list(param_grid.keys())
param_combinations = list(itertools.product(*param_grid.values()))

# ----------------------------------------------
# Activation mapping
activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


# ----------------------------------------------
# Cox PH Loss
class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, durations, events, hazards):
        order = torch.argsort(durations, descending=True)
        hazards_sorted = hazards[order]
        events_sorted = events[order]

        exp_hazards = torch.exp(hazards_sorted)
        cum_sum_exp = torch.cumsum(exp_hazards, dim=0)
        log_risk = torch.log(cum_sum_exp)

        log_likelihood = torch.sum(events_sorted * (hazards_sorted - log_risk))
        return -log_likelihood


# ----------------------------------------------
# Storage
outer_cv_scores = {params: [] for params in param_combinations}  # all combos
outer_fold_results = []  # per fold storage

# ----------------------------------------------
# Hyperparameter selection on inner validation splits
#
# The grid is evaluated on a train/validation split carved out of each selection
# fold's OWN training data, and the scores are averaged across those folds to
# choose one global hyperparameter set. No outer test fold is used for selection.
#
# Selection is hoisted out of the outer-fold loop: because a single global
# parameter set is chosen regardless, the grid runs on the first
# GRID_SEARCH_FOLDS folds only, and all n_folds outer models reuse the winner.
INNER_VAL_FRACTION = 0.2
n_grid_search_folds = min(GRID_SEARCH_FOLDS, n_folds)

for outer_fold in range(n_grid_search_folds):
    print(
        f"\n===== SELECTION: FOLD {outer_fold + 1}/{n_grid_search_folds} "
        f"(of {n_folds} outer folds) ====="
    )

    X_select = Xts_train_folds_filtered[outer_fold].to_numpy().astype(np.float32)
    y_select = y_train_folds[outer_fold].astype(np.float32)
    event_select = event_train_folds[outer_fold].astype(bool)

    # Inner split of this fold's training data
    (
        X_inner_train,
        X_inner_val,
        y_inner_train,
        y_inner_val,
        event_inner_train,
        event_inner_val,
    ) = train_test_split(
        X_select,
        y_select,
        event_select,
        test_size=INNER_VAL_FRACTION,
        random_state=SEED + outer_fold,
        stratify=event_select,
    )

    print(
        f"{X_inner_train.shape[0]} inner-train rows, "
        f"{X_inner_val.shape[0]} inner-validation rows, "
        f"{len(param_combinations)} combinations"
    )

    # These are the same for every combination in this fold, so build them once
    X_inner_train_t = torch.tensor(X_inner_train).to(device)
    y_inner_train_t = torch.tensor(y_inner_train).to(device)
    event_inner_train_t = torch.tensor(event_inner_train).to(device)
    X_inner_val_t = torch.tensor(X_inner_val).to(device)

    tau = np.quantile(y_inner_train, 0.95)
    survival_inner_train = np.array(
        list(zip(event_inner_train, y_inner_train)),
        dtype=[("event", bool), ("time", float)],
    )
    survival_inner_val = np.array(
        list(zip(event_inner_val, y_inner_val)),
        dtype=[("event", bool), ("time", float)],
    )

    for combo_idx, params in enumerate(param_combinations):
        num_nodes, num_layers, learning_rate, dropout, l2_reg, activation_name = params

        # Build model
        layers = []
        input_dim = X_inner_train.shape[1]

        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, num_nodes))
            layers.append(activation_map[activation_name]())
            layers.append(nn.Dropout(dropout))
            input_dim = num_nodes

        layers.append(nn.Linear(input_dim, 1))
        model = nn.Sequential(*layers).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=l2_reg
        )
        criterion = CoxPHLoss()

        # Train on the inner training split
        model.train()
        best_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(GRID_SEARCH_EPOCHS):
            optimizer.zero_grad()
            output = model(X_inner_train_t)
            loss = criterion(
                y_inner_train_t,
                event_inner_train_t,
                output.flatten(),
            )
            loss.backward()
            optimizer.step()

            # Stop early once the loss has plateaued
            current_loss = loss.item()
            if current_loss < best_loss - EARLY_STOP_MIN_DELTA:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    break

        # Evaluate on the held-out inner validation split
        model.eval()
        with torch.no_grad():
            preds = model(X_inner_val_t).flatten().cpu().numpy()

        c_index = concordance_index_ipcw(
            survival_train=survival_inner_train,
            survival_test=survival_inner_val,
            estimate=preds,
            tau=tau,
        )[0]

        # Store results
        outer_cv_scores[params].append(c_index)

        outer_fold_results.append(
            {
                "Outer_Fold": outer_fold + 1,
                "num_nodes": num_nodes,
                "num_layers": num_layers,
                "learning_rate": learning_rate,
                "dropout": dropout,
                "l2_reg": l2_reg,
                "activation": activation_name,
                "Validation_C_index": c_index,
            }
        )

        if (combo_idx + 1) % 100 == 0:
            print(f"  {combo_idx + 1}/{len(param_combinations)} combinations evaluated")

    print(f"Finished selection for fold {outer_fold + 1}")

# ----------------------------------------------
# Mean and spread of the inner-validation C-index across the selection folds
mean_outer_scores = {k: np.mean(v) for k, v in outer_cv_scores.items()}
std_outer_scores = {k: np.std(v) for k, v in outer_cv_scores.items()}

# Global best hyperparameters
global_best_params = max(mean_outer_scores, key=mean_outer_scores.get)
print("\n===== GLOBAL BEST HYPERPARAMETERS =====")
print("Best Params:", global_best_params)
print("Mean inner-validation C-index:", mean_outer_scores[global_best_params])
print("Std across selection folds:", std_outer_scores[global_best_params])

# ----------------------------------------------
# Prepare Excel Export

# All hyperparameters summary
results_df = (
    pd.DataFrame(
        [
            {
                "num_nodes": k[0],
                "num_layers": k[1],
                "learning_rate": k[2],
                "dropout": k[3],
                "l2_reg": k[4],
                "activation": k[5],
                "Mean_Validation_C_index": mean_outer_scores[k],
                "Std_Validation_C_index": std_outer_scores[k],
            }
            for k in mean_outer_scores.keys()
        ]
    )
    .sort_values(by="Mean_Validation_C_index", ascending=False)
    .reset_index(drop=True)
)

# Global best dataframe
best_df = pd.DataFrame(
    [
        {
            "num_nodes": global_best_params[0],
            "num_layers": global_best_params[1],
            "learning_rate": global_best_params[2],
            "dropout": global_best_params[3],
            "l2_reg": global_best_params[4],
            "activation": global_best_params[5],
            "Mean_Validation_C_index": mean_outer_scores[global_best_params],
            "Std_Validation_C_index": std_outer_scores[global_best_params],
        }
    ]
)

# Per-fold results dataframe
fold_results_df = pd.DataFrame(outer_fold_results)

# Write to Excel
excel_filename = f"{output_dir}{filename}_NestedCV_InitialGlobalSelection.xlsx"

with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="All_InitialHyperparameters", index=False)
    best_df.to_excel(writer, sheet_name="Best_InitialHyperparameters", index=False)
    fold_results_df.to_excel(writer, sheet_name="Selection_InitialResults", index=False)

# %%
# STEP 9
# Fit and train initial model per fold using one set of optimal hyperparameters,
# before passing to Shap filter

# ==============================================================
# TRAIN & EVALUATE DEEPSURV USING SINGLE BEST HYPERPARAMETERS
# ==============================================================

# ----------------------------------------------
# Set seeds for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------
# Define Cox Proportional Hazards Loss
class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, durations, events, hazards):
        order = torch.argsort(durations, descending=True)
        hazards_sorted = hazards[order]
        events_sorted = events[order]

        exp_hazards = torch.exp(hazards_sorted)
        cum_sum_exp = torch.cumsum(exp_hazards, dim=0)
        log_risk = torch.log(cum_sum_exp)

        log_likelihood = torch.sum(events_sorted * (hazards_sorted - log_risk))
        return -log_likelihood


# ----------------------------------------------
# Storage for risk scores
risk_scores_per_fold = []
trained_models_per_fold = {}

# ----------------------------------------------
# Use a single global set of hyperparameters
# For example, from nested CV
optimal_params_dict = {
    "num_nodes": global_best_params[0],
    "num_layers": global_best_params[1],
    "learning_rate": global_best_params[2],
    "dropout": global_best_params[3],
    "l2_reg": global_best_params[4],
    "activation": global_best_params[5],
}

# Then you can access by key
num_nodes = optimal_params_dict["num_nodes"]
num_layers = optimal_params_dict["num_layers"]
learning_rate = optimal_params_dict["learning_rate"]
dropout = optimal_params_dict["dropout"]
l2_reg = optimal_params_dict["l2_reg"]
activation_name = optimal_params_dict["activation"]

print("Using global hyperparameters:", optimal_params_dict)

# ----------------------------------------------
# Train + evaluate model for each fold
for fold in range(n_folds):
    print(f"\n===== Training fold {fold + 1}/{n_folds} =====")

    # Load fold data
    X_train = Xts_train_folds_filtered[fold].to_numpy().astype(np.float32)
    X_test = Xts_test_folds_filtered[fold].to_numpy().astype(np.float32)

    durations_train = y_train_folds[fold].astype(np.float32)
    durations_test = y_test_folds[fold].astype(np.float32)

    # Convert event indicators to 0/1
    events_train = event_train_folds[fold].astype(int)
    events_test = event_test_folds[fold].astype(int)

    NAID_test = Xts_test_folds[fold]["naid"].values

    # Build model
    layers = []
    input_dim = X_train.shape[1]

    for _ in range(num_layers):
        layers.append(nn.Linear(input_dim, num_nodes))
        layers.append(
            {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation_name]()
        )
        layers.append(nn.Dropout(dropout))
        input_dim = num_nodes

    layers.append(nn.Linear(input_dim, 1))
    model = nn.Sequential(*layers).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    criterion = CoxPHLoss()

    # Training
    # Build the tensors once instead of rebuilding them from numpy on every epoch
    X_train_t = torch.tensor(X_train).to(device)
    durations_train_t = torch.tensor(durations_train).to(device)
    events_train_t = torch.tensor(events_train).to(device)

    model.train()
    best_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(
            durations_train_t,
            events_train_t,
            output.flatten(),
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.4f}")

        # Stop early once the loss has plateaued
        current_loss = loss.item()
        if current_loss < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Fold {fold + 1}: early stop at epoch {epoch}")
                break

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).to(device)
        risk_scores = model(X_test_tensor).flatten().cpu().numpy()

    # Save model and risk scores
    trained_models_per_fold[fold] = {
        "model": model,
        "risk_scores": risk_scores,
        "durations_test": durations_test,
        "events_test": events_test,
    }

# %%
# STEP 10
# Global feature selection using SHAP threshold filtering

# Store fold-wise results
results_per_fold = []

# Store mean SHAP values per feature per fold
mean_shap_feature_folds = []

# Get feature index-to-name mapping
original_feature_names = Xts_test_folds_filtered[0].columns.tolist()
index_to_name = {i: name for i, name in enumerate(original_feature_names)}

# Loop through folds
for i in range(n_folds):
    print(f"\n===== SHAP feature selection for fold {i + 1}/{n_folds} =====")

    # Extract trained model
    model_entry = trained_models_per_fold[i]
    model = model_entry["model"]
    model.eval()

    # Extract data
    X_train_df = Xts_train_folds_filtered[i]
    X_test_df = Xts_test_folds_filtered[i]

    # Wrapper class
    class DeepSurvWrapper(BaseEstimator):
        def __init__(self, model, feature_names, device=None):
            self.device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model = model.to(self.device)
            self.feature_names = feature_names

        def __call__(self, X):
            if isinstance(X, pd.DataFrame):
                X_df = X.loc[:, self.feature_names]
            else:
                X_df = pd.DataFrame(X, columns=self.feature_names)

            X_tensor = torch.tensor(X_df.values.astype(np.float32)).to(self.device)

            with torch.no_grad():
                preds = self.model(X_tensor).flatten().cpu().numpy()

            return preds

    wrapped_model = DeepSurvWrapper(model, X_test_df.columns.tolist())

    # SHAP explainer
    # Subsample the background: the permutation explainer runs the model over
    # the entire background for every masked evaluation.
    shap_background = X_train_df.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(X_train_df)), random_state=SEED
    )
    explainer = shap.Explainer(
        wrapped_model, shap_background, algorithm="permutation", verbose=False
    )
    shap_values = explainer(X_test_df)

    shap_matrix = shap_values.values

    # ----------------------------------------
    # Mean absolute SHAP values (for ranking)
    # ----------------------------------------
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    mean_abs_shap = np.nan_to_num(mean_abs_shap, nan=0.0)

    # ----------------------------------------
    # Mean SHAP values (directional)
    # ----------------------------------------
    mean_shap = shap_matrix.mean(axis=0)
    mean_shap = np.nan_to_num(mean_shap, nan=0.0)

    # Store mean SHAP per feature per fold
    mean_shap_df = pd.DataFrame(
        {
            "feature_index": np.arange(len(mean_shap)),
            "variable_name": X_test_df.columns,
            "fold": i + 1,
            "mean_shap_value": mean_shap,
        }
    )

    mean_shap_feature_folds.append(mean_shap_df)

    # ----------------------------------------
    # Ranking dataframe
    # ----------------------------------------
    selector_df = pd.DataFrame(
        {"feature_index": np.arange(len(mean_abs_shap)), "mean_abs_shap": mean_abs_shap}
    )

    selector_df["variable_name"] = X_test_df.columns

    # Rank features (highest SHAP = rank 1)
    selector_df["rank"] = selector_df["mean_abs_shap"].rank(
        method="average", ascending=False
    )

    selector_df["fold"] = i + 1

    results_per_fold.append(selector_df)

# ------------------------------------------------
# Combine fold results
# ------------------------------------------------
all_results = pd.concat(results_per_fold)

# ------------------------------------------------
# Compute mean + median ranks across folds
# ------------------------------------------------
rank_summary = (
    all_results.groupby("variable_name")["rank"]
    .agg(mean_rank="mean", median_rank="median")
    .reset_index()
)

# ------------------------------------------------
# Feature selection based on MEDIAN rank
# ------------------------------------------------
rank_threshold = np.percentile(rank_summary["median_rank"], 25)

selected_features = rank_summary[
    rank_summary["median_rank"] <= rank_threshold
].sort_values("median_rank")

# ------------------------------------------------
# Combine mean SHAP per feature per fold
# ------------------------------------------------
mean_shap_feature_all = pd.concat(mean_shap_feature_folds)

# ------------------------------------------------
# Threshold dataframe (for export)
# ------------------------------------------------
threshold_df = pd.DataFrame(
    {"metric": ["median_rank_threshold"], "value": [rank_threshold]}
)

# ------------------------------------------------
# Print results
# ------------------------------------------------
print("\n===== Final Feature Selection =====")
print(f"25th percentile median-rank threshold: {rank_threshold:.3f}")
print(f"Selected {len(selected_features)} / {len(rank_summary)} features\n")

print("Selected features:")
print(selected_features["variable_name"].tolist())

# ------------------------------------------------
# Export Excel
# ------------------------------------------------
excel_file = f"{output_dir}{filename}_SHAP_FeatureSelection_Results.xlsx"

with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
    all_results.to_excel(writer, sheet_name="Fold_Feature_Ranks", index=False)

    rank_summary.to_excel(writer, sheet_name="Rank_Summary", index=False)

    selected_features.to_excel(writer, sheet_name="Selected_Features", index=False)

    mean_shap_feature_all.to_excel(
        writer, sheet_name="Mean_SHAP_PerFeature_PerFold", index=False
    )

    threshold_df.to_excel(writer, sheet_name="Selection_Threshold", index=False)

# %%
# STEP 11
# Expand feature selection such that all levels of a selected feature are included

final_vars_raw = selected_features["variable_name"].tolist()

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
# STEP 12
# Final hyperparameter optimization using nested CV and SHAP-selected features per fold

# ----------------------------------------------
# Reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------------------------
# Hyperparameter grid
param_grid = {
    "num_nodes": [32, 64, 128],
    "num_layers": [1, 2, 3],
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "dropout": [0.1, 0.2, 0.3],
    "l2_reg": [1e-2, 1e-4, 1e-5],
    "activation": ["relu", "tanh", "sigmoid"],
}

# smaller grid for testing purposes
# param_grid = {
#     'num_nodes': [32, 64],
#     'num_layers': [1, 2],
#     'learning_rate': [1e-3, 1e-4],
#     'dropout': [0.1, 0.2],
#     'l2_reg': [1e-2, 1e-4],
#     'activation': ['relu', 'tanh']
# }

param_names = list(param_grid.keys())
param_combinations = list(itertools.product(*param_grid.values()))

# ----------------------------------------------
# Activation mapping
activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


# ----------------------------------------------
# Cox PH Loss
class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, durations, events, hazards):
        order = torch.argsort(durations, descending=True)
        hazards_sorted = hazards[order]
        events_sorted = events[order]

        exp_hazards = torch.exp(hazards_sorted)
        cum_sum_exp = torch.cumsum(exp_hazards, dim=0)
        log_risk = torch.log(cum_sum_exp)

        log_likelihood = torch.sum(events_sorted * (hazards_sorted - log_risk))
        return -log_likelihood


# ----------------------------------------------
# Storage
outer_cv_scores = {params: [] for params in param_combinations}  # all combos
outer_fold_results = []  # per fold storage

# Hyperparameter selection on inner validation splits
#
# The grid is evaluated on a train/validation split carved out of each selection
# fold's OWN training data, and the scores are averaged across those folds to
# choose one global hyperparameter set. No outer test fold is used for selection.
#
# Selection is hoisted out of the outer-fold loop: because a single global
# parameter set is chosen regardless, the grid runs on the first
# GRID_SEARCH_FOLDS folds only, and all n_folds outer models reuse the winner.
INNER_VAL_FRACTION = 0.2
n_grid_search_folds = min(GRID_SEARCH_FOLDS, n_folds)

for outer_fold in range(n_grid_search_folds):
    print(
        f"\n===== SELECTION: FOLD {outer_fold + 1}/{n_grid_search_folds} "
        f"(of {n_folds} outer folds) ====="
    )

    X_select = (
        Xts_train_folds_filtered[outer_fold][selected_features]
        .to_numpy()
        .astype(np.float32)
    )
    y_select = y_train_folds[outer_fold].astype(np.float32)
    event_select = event_train_folds[outer_fold].astype(bool)

    # Inner split of this fold's training data
    (
        X_inner_train,
        X_inner_val,
        y_inner_train,
        y_inner_val,
        event_inner_train,
        event_inner_val,
    ) = train_test_split(
        X_select,
        y_select,
        event_select,
        test_size=INNER_VAL_FRACTION,
        random_state=SEED + outer_fold,
        stratify=event_select,
    )

    print(
        f"{X_inner_train.shape[0]} inner-train rows, "
        f"{X_inner_val.shape[0]} inner-validation rows, "
        f"{len(param_combinations)} combinations"
    )

    # These are the same for every combination in this fold, so build them once
    X_inner_train_t = torch.tensor(X_inner_train).to(device)
    y_inner_train_t = torch.tensor(y_inner_train).to(device)
    event_inner_train_t = torch.tensor(event_inner_train).to(device)
    X_inner_val_t = torch.tensor(X_inner_val).to(device)

    tau = np.quantile(y_inner_train, 0.95)
    survival_inner_train = np.array(
        list(zip(event_inner_train, y_inner_train)),
        dtype=[("event", bool), ("time", float)],
    )
    survival_inner_val = np.array(
        list(zip(event_inner_val, y_inner_val)),
        dtype=[("event", bool), ("time", float)],
    )

    for combo_idx, params in enumerate(param_combinations):
        num_nodes, num_layers, learning_rate, dropout, l2_reg, activation_name = params

        # Build model
        layers = []
        input_dim = X_inner_train.shape[1]

        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, num_nodes))
            layers.append(activation_map[activation_name]())
            layers.append(nn.Dropout(dropout))
            input_dim = num_nodes

        layers.append(nn.Linear(input_dim, 1))
        model = nn.Sequential(*layers).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=l2_reg
        )
        criterion = CoxPHLoss()

        # Train on the inner training split
        model.train()
        best_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(GRID_SEARCH_EPOCHS):
            optimizer.zero_grad()
            output = model(X_inner_train_t)
            loss = criterion(
                y_inner_train_t,
                event_inner_train_t,
                output.flatten(),
            )
            loss.backward()
            optimizer.step()

            # Stop early once the loss has plateaued
            current_loss = loss.item()
            if current_loss < best_loss - EARLY_STOP_MIN_DELTA:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    break

        # Evaluate on the held-out inner validation split
        model.eval()
        with torch.no_grad():
            preds = model(X_inner_val_t).flatten().cpu().numpy()

        c_index = concordance_index_ipcw(
            survival_train=survival_inner_train,
            survival_test=survival_inner_val,
            estimate=preds,
            tau=tau,
        )[0]

        # Store results
        outer_cv_scores[params].append(c_index)

        outer_fold_results.append(
            {
                "Outer_Fold": outer_fold + 1,
                "num_nodes": num_nodes,
                "num_layers": num_layers,
                "learning_rate": learning_rate,
                "dropout": dropout,
                "l2_reg": l2_reg,
                "activation": activation_name,
                "Validation_C_index": c_index,
            }
        )

        if (combo_idx + 1) % 100 == 0:
            print(f"  {combo_idx + 1}/{len(param_combinations)} combinations evaluated")

    print(f"Finished selection for fold {outer_fold + 1}")

# ----------------------------------------------
# Mean and spread of the inner-validation C-index across the selection folds
mean_outer_scores = {k: np.mean(v) for k, v in outer_cv_scores.items()}
std_outer_scores = {k: np.std(v) for k, v in outer_cv_scores.items()}

# Global best hyperparameters
global_best_params = max(mean_outer_scores, key=mean_outer_scores.get)
print("\n===== GLOBAL BEST HYPERPARAMETERS =====")
print("Best Params:", global_best_params)
print("Mean inner-validation C-index:", mean_outer_scores[global_best_params])
print("Std across selection folds:", std_outer_scores[global_best_params])

# ----------------------------------------------
# Prepare Excel Export

# All hyperparameters summary
results_df = (
    pd.DataFrame(
        [
            {
                "num_nodes": k[0],
                "num_layers": k[1],
                "learning_rate": k[2],
                "dropout": k[3],
                "l2_reg": k[4],
                "activation": k[5],
                "Mean_Validation_C_index": mean_outer_scores[k],
                "Std_Validation_C_index": std_outer_scores[k],
            }
            for k in mean_outer_scores.keys()
        ]
    )
    .sort_values(by="Mean_Validation_C_index", ascending=False)
    .reset_index(drop=True)
)

# Global best dataframe
best_df = pd.DataFrame(
    [
        {
            "num_nodes": global_best_params[0],
            "num_layers": global_best_params[1],
            "learning_rate": global_best_params[2],
            "dropout": global_best_params[3],
            "l2_reg": global_best_params[4],
            "activation": global_best_params[5],
            "Mean_Validation_C_index": mean_outer_scores[global_best_params],
            "Std_Validation_C_index": std_outer_scores[global_best_params],
        }
    ]
)

# Per-fold results dataframe
fold_results_df = pd.DataFrame(outer_fold_results)

# Write to Excel
excel_filename = f"{output_dir}{filename}_NestedCV_FinalGlobalSelection.xlsx"

with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="All_FinalHyperparameters", index=False)
    best_df.to_excel(writer, sheet_name="Best_FinalHyperparameters", index=False)
    fold_results_df.to_excel(writer, sheet_name="Selection_FinalResults", index=False)

# %%
# STEP 13
# Fit and train final model using optimal hyperparameters and SHAP-selected features
# Export test set NAIDs' risk scores from each fold-specific model

# ==============================================================
# TRAIN & EVALUATE DEEPSURV USING SINGLE BEST HYPERPARAMETERS
# ==============================================================

# ----------------------------------------------
# Set seeds for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------
# Define Cox Proportional Hazards Loss
class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, durations, events, hazards):
        order = torch.argsort(durations, descending=True)
        hazards_sorted = hazards[order]
        events_sorted = events[order]

        exp_hazards = torch.exp(hazards_sorted)
        cum_sum_exp = torch.cumsum(exp_hazards, dim=0)
        log_risk = torch.log(cum_sum_exp)

        log_likelihood = torch.sum(events_sorted * (hazards_sorted - log_risk))
        return -log_likelihood


# ----------------------------------------------
# Storage for risk scores
risk_scores_per_fold = []
trained_models_per_fold = {}

# ----------------------------------------------
# Use a single global set of hyperparameters
optimal_params_dict = {
    "num_nodes": global_best_params[0],
    "num_layers": global_best_params[1],
    "learning_rate": global_best_params[2],
    "dropout": global_best_params[3],
    "l2_reg": global_best_params[4],
    "activation": global_best_params[5],
}

# Then you can access by key
num_nodes = optimal_params_dict["num_nodes"]
num_layers = optimal_params_dict["num_layers"]
learning_rate = optimal_params_dict["learning_rate"]
dropout = optimal_params_dict["dropout"]
l2_reg = optimal_params_dict["l2_reg"]
activation_name = optimal_params_dict["activation"]

print("Using global hyperparameters:", optimal_params_dict)
print("Using global features:", selected_features)

# ----------------------------------------------
# Train + evaluate model for each fold
for fold in range(n_folds):
    print(f"\n===== Training fold {fold + 1}/{n_folds} =====")

    # Load fold data
    X_train = (
        Xts_train_folds_filtered[fold][selected_features].to_numpy().astype(np.float32)
    )
    X_test = (
        Xts_test_folds_filtered[fold][selected_features].to_numpy().astype(np.float32)
    )

    durations_train = y_train_folds[fold].astype(np.float32)
    durations_test = y_test_folds[fold].astype(np.float32)

    # Convert event indicators to 0/1
    events_train = event_train_folds[fold].astype(int)
    events_test = event_test_folds[fold].astype(int)

    NAID_test = Xts_test_folds[fold]["naid"].values

    # Build model
    layers = []
    input_dim = X_train.shape[1]

    for _ in range(num_layers):
        layers.append(nn.Linear(input_dim, num_nodes))
        layers.append(
            {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation_name]()
        )
        layers.append(nn.Dropout(dropout))
        input_dim = num_nodes

    layers.append(nn.Linear(input_dim, 1))
    model = nn.Sequential(*layers).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    criterion = CoxPHLoss()

    fold_loss_history = []

    # Training
    # Build the tensors once instead of rebuilding them from numpy on every epoch
    X_train_t = torch.tensor(X_train).to(device)
    durations_train_t = torch.tensor(durations_train).to(device)
    events_train_t = torch.tensor(events_train).to(device)

    model.train()
    best_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(
            durations_train_t,
            events_train_t,
            output.flatten(),
        )
        loss.backward()
        optimizer.step()

        fold_loss_history.append(loss.item())

        if epoch % 10 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.4f}")

        # Stop early once the loss has plateaued
        current_loss = loss.item()
        if current_loss < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Fold {fold + 1}: early stop at epoch {epoch}")
                break

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).to(device)
        risk_scores = model(X_test_tensor).flatten().cpu().numpy()

    # Save model and risk scores
    trained_models_per_fold[fold] = {
        "model": model,
        "risk_scores": risk_scores,
        "durations_test": durations_test,
        "events_test": events_test,
        "loss_history": fold_loss_history,
    }

    df_fold_preds = pd.DataFrame(
        {
            "naid": NAID_test,
            "duration": durations_test,
            "event": events_test,
            "log_risk": risk_scores,
            "fold": fold + 1,
        }
    )
    risk_scores_per_fold.append(df_fold_preds)

    # Save full model with Pickle
    model_file = f"{output_dir}{filename}_Fold{fold + 1}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

# ----------------------------------------------
# Export all folds' risk scores
all_folds_preds = pd.concat(risk_scores_per_fold, axis=0).reset_index(drop=True)
all_folds_preds.to_csv(f"{output_dir}{filename}_LogRiskScores.csv", index=False)

plt.figure(figsize=(10, 6))

for fold in range(n_folds):
    loss_history = trained_models_per_fold[fold]["loss_history"]
    plt.plot(loss_history, label=f"Fold {fold + 1}", alpha=0.7)

plt.title("DeepSurv Training Loss via Negative Log Partial Likelihood")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Save the plot
output_loss_plot = f"{output_dir}{filename}_TrainingLoss.png"
plt.savefig(output_loss_plot, dpi=300)

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
    model,
    data_input,
    index_to_name,
    categorical_groups=None,
    is_deepsurv=False,
    scaler=None,
    continuous_cols=None,
):
    """
    Calculates the Average Marginal Effect (AME) on the log-risk scale.
    """
    if categorical_groups is None:
        categorical_groups = []
    if continuous_cols is None:
        continuous_cols = []

    if isinstance(data_input, torch.Tensor):
        np_data = data_input.detach().cpu().numpy()
        num_cols = np_data.shape[1]
        col_names = [index_to_name.get(i, f"Feature_{i}") for i in range(num_cols)]
        pandas_data = pd.DataFrame(np_data, columns=col_names)
    elif isinstance(data_input, pd.DataFrame):
        pandas_data = data_input.copy()
    else:
        raise TypeError("data_input must be a pandas DataFrame or a PyTorch Tensor")

    coefficients = {}
    n_samples, n_features = pandas_data.shape

    def predict_risk(X_input):
        if is_deepsurv:
            X_tensor = torch.tensor(
                X_input.to_numpy().astype(np.float32), dtype=torch.float32
            )
            model.eval()
            with torch.no_grad():
                risk = model(X_tensor.to(device))
            return risk.cpu().numpy().flatten()
        else:
            return model.predict_log_partial_hazard(X_input).to_numpy().flatten()

    baseline_preds = predict_risk(pandas_data)

    for i in range(n_features):
        feature_name = index_to_name[i]
        col_data = pandas_data.iloc[:, i]

        unique_vals = col_data.dropna().unique()
        is_binary = set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset(
            {0.0, 1.0}
        )

        if is_binary:
            sibling_features = []
            for group in categorical_groups:
                if feature_name in group:
                    sibling_features = [
                        f
                        for f in group
                        if f != feature_name and f in pandas_data.columns
                    ]
                    break

            df_1 = pandas_data.copy()
            df_1.iloc[:, i] = 1.0
            for sibling in sibling_features:
                df_1[sibling] = 0.0

            df_0 = pandas_data.copy()
            df_0.iloc[:, i] = 0.0
            for sibling in sibling_features:
                df_0[sibling] = 0.0

            preds_1 = predict_risk(df_1)
            preds_0 = predict_risk(df_0)
            ame = np.mean(preds_1 - preds_0)

        else:
            # Continuous Estimand: Expected Partial Derivative via Finite Difference
            # If the feature is in continuous_cols, we must account for the scaler's variance

            epsilon = 1e-4
            df_perturbed = pandas_data.copy()

            if scaler is not None and feature_name in continuous_cols:
                # Find the index of the feature in the scaler to retrieve the correct scale factor (sigma)
                scaler_feature_idx = continuous_cols.index(feature_name)
                scale_factor = scaler.scale_[scaler_feature_idx]

                # A perturbation of 'epsilon' in the original scale translates to 'epsilon / sigma' in the standard scale
                scaled_epsilon = epsilon / scale_factor
                df_perturbed.iloc[:, i] = col_data + scaled_epsilon
            else:
                df_perturbed.iloc[:, i] = col_data + epsilon

            preds_perturbed = predict_risk(df_perturbed)

            # Gradients must be computed relative to the original epsilon to reflect original units
            gradients = (preds_perturbed - baseline_preds) / epsilon
            ame = np.mean(gradients)

        coefficients[feature_name] = ame

    return coefficients


# %%
# STEP 14
# Produce SHAP values for each fold's testing data using SHAP-selected features

# ----------------------------------------------
# Ensure model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------
# Map feature indices to names for coefficient extraction
original_feature_names = selected_features
index_to_name = {i: name for i, name in enumerate(original_feature_names)}

# ----------------------------------------------
# Loop over folds
all_folds_shap_df = []
mean_abs_shap_list = []
feature_coefficients_df = []

for fold in range(n_folds):
    print(f"\n===== Computing SHAP values for fold {fold + 1}/{n_folds} =====")

    # Test set data for SHAP calculation
    X_test_fold_df = Xts_test_folds_filtered[fold].loc[:, selected_features]
    X_test_fold_array = X_test_fold_df.to_numpy(dtype=np.float32)

    X_train_fold_df = Xts_train_folds_filtered[fold].loc[:, selected_features]
    X_train_fold_array = X_train_fold_df.to_numpy(dtype=np.float32)

    naid_test = Xts_test_folds[fold]["naid"].values

    # ----- Define predict function compatible with PyTorch model -----
    def predict_risk(X_input):
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = (
                trained_models_per_fold[fold]["model"](X_tensor).flatten().cpu().numpy()
            )
        return preds

    # ----- Subsample the SHAP background -----
    # The permutation explainer runs the model over the entire background for
    # every masked evaluation, so using the full training fold makes SHAP cost
    # grow with training-fold size.
    background_idx = np.random.choice(
        X_train_fold_array.shape[0],
        min(SHAP_BACKGROUND_SIZE, X_train_fold_array.shape[0]),
        replace=False,
    )
    shap_background = X_train_fold_array[background_idx]

    # ----- Initialize SHAP Explainer -----
    explainer = shap.Explainer(
        predict_risk, shap_background, algorithm="permutation", verbose=False
    )

    # ----- Calculate SHAP values -----
    shap_values = explainer(X_test_fold_array)

    # ----- Create DataFrame for SHAP values per sample -----
    shap_df = pd.DataFrame(shap_values.values, columns=X_test_fold_df.columns)
    shap_df["expected_value"] = shap_values.base_values
    shap_df["naid"] = naid_test
    shap_df["fold"] = fold + 1

    # Store per-sample SHAP values
    all_folds_shap_df.append(shap_df)

    # ----- Compute mean absolute SHAP values for this fold -----
    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=selected_features,
        name=f"Fold{fold + 1} Mean Abs SHAP",
    ).sort_values(ascending=False)  # rank descending

    mean_abs_shap_list.append(mean_abs_shap)

    print(f"Fold {fold + 1} mean absolute SHAP values (top features):")
    print(mean_abs_shap.head(10))

    # ----- Extract feature coefficients using exact marginalization ---
    tensor_data = torch.tensor(X_train_fold_array, dtype=torch.float32).to(device)

    # Extract coefficients applying the preprocessing adjustment
    coefficients = extract_average_marginal_effects(
        trained_models_per_fold[fold]["model"],
        tensor_data,
        index_to_name,
        categorical_groups=categorical_groups,
        is_deepsurv=True,
        scaler=scalers_folds[fold],  # <--- Pass the saved scaler for this fold
        continuous_cols=continuous_cols,  # <--- Pass the continuous column names
    )

    coefficients_df = pd.DataFrame(
        {
            "feature": list(coefficients.keys()),
            "coefficient": list(coefficients.values()),
            "fold": fold + 1,
        }
    )
    feature_coefficients_df.append(coefficients_df)

# ----------------------------------------------
# Combine all folds per-sample SHAP values
combined_shap_df = pd.concat(all_folds_shap_df, ignore_index=True)

# ----------------------------------------------
#  Export combined per-sample SHAP values
combined_shap_df.to_csv(f"{output_dir}{filename}_SHAPValues.csv", index=False)

# ----------------------------------------------
# Combine mean absolute SHAP values per fold
mean_abs_shap_df = pd.concat(mean_abs_shap_list, axis=1).fillna(0.0)
mean_abs_shap_df.index.name = "feature"

mean_abs_shap_df = mean_abs_shap_df.fillna(0.0)

# ----------------------------------------------
# Add rank per fold
ranked_shap_df = mean_abs_shap_df.copy()
for col in ranked_shap_df.columns:
    # Rank all features, highest mean SHAP = rank 1
    ranked_shap_df[col + "_rank"] = ranked_shap_df[col].rank(
        ascending=False,
        method="min",  # ties get the same minimum rank
    )

# ----------------------------------------------
# Export ranked mean absolute SHAP values
ranked_shap_df.to_csv(f"{output_dir}{filename}_MeanAbsSHAPValues.csv")

# Export feature coefficients
feature_coefficients_all = pd.concat(feature_coefficients_df, ignore_index=True)
feature_coefficients_all.to_csv(
    f"{output_dir}{filename}_FeatureCoefficients.csv", index=False
)

# %%
# STEP 15
# Concordance index (c-index) (0-1; higher is better)

# USING SKLEARN CONCORDANCE_INDEX_IPCW

# C-index: proportion of concordant pairs among all comparable pairs; concordant if model assigns
# higher risk to the observation with shorter survival time

################################
# Parameters and storage
n_folds = len(trained_models_per_fold)  # number of folds
cindex_results = []

################################
# Estimate global tau from all training data
all_X_train = pd.concat(Xts_train_folds, ignore_index=True)
surv_train_all = Surv.from_arrays(
    all_X_train["event"].astype(bool), all_X_train["duration"].values
)

cens_global = CensoringDistributionEstimator()
cens_global.fit(surv_train_all)

valid_times_global = [
    t for t in all_X_train["duration"] if cens_global.predict_proba([t])[0] > 0
]
tau_global = min(all_X_train["duration"].quantile(0.95), max(valid_times_global))

print(f"Global tau (IPCW): {tau_global:.1f} days")

################################
# Loop over folds
for fold in range(n_folds):
    print(f"\n=== C-index Evaluation: Fold {fold + 1} ===")

    # Test and train sets
    Xts_test = risk_scores_per_fold[fold].copy()  # includes 'log_risk'
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
    # Save per-fold summary
    cindex_results.append(
        {
            "fold": fold + 1,
            "cindex_overall": overall_cindex,
            "cindex_1yr": cindex_1yr,
            "cindex_5yr": cindex_5yr,
        }
    )

################################
# Create dataframes
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
# STEP 16
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
# Breslow baseline
# --------------------------------------------------
def breslow_baseline(durations, events, exp_risk):
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    exp_risk = exp_risk[order]

    event_times = np.unique(durations[events])
    H0 = np.zeros(len(event_times))

    for i, t in enumerate(event_times):
        d_i = np.sum((durations == t) & events)
        risk_set = np.sum(exp_risk[durations >= t])
        H0[i] = d_i / risk_set

    return event_times, np.cumsum(H0)


# --------------------------------------------------
# Cross-validation loop
# --------------------------------------------------
for fold in range(n_folds):
    print(f"\n=== Brier Score Evaluation: Fold {fold + 1} ===")

    X_train = Xts_train_folds[fold].copy()
    X_test = risk_scores_per_fold[fold].copy()
    model = trained_models_per_fold[fold]["model"]
    device = next(model.parameters()).device
    model.eval()

    # -------------------------------
    # Compute log-risk for training set
    # -------------------------------
    import torch

    with torch.no_grad():
        X_train_tensor = torch.tensor(
            X_train[Xts_train_folds_filtered[fold][selected_features].columns]
            .to_numpy()
            .astype(np.float32),
            dtype=torch.float32,
        ).to(device)
        log_risk_train = model(X_train_tensor).flatten().cpu().numpy()

    exp_risk_train = np.exp(log_risk_train)

    # -------------------------------
    # Structured survival arrays
    # -------------------------------
    surv_train = Surv.from_arrays(
        X_train["event"].astype(bool), X_train["duration"].values
    )
    surv_test = Surv.from_arrays(
        X_test["event"].astype(bool), X_test["duration"].values
    )

    # -------------------------------
    # Breslow baseline
    # -------------------------------
    t0, H0 = breslow_baseline(
        X_train["duration"].values, X_train["event"].astype(bool).values, exp_risk_train
    )
    from scipy.interpolate import interp1d

    fold_eval_times = global_eval_times[
        (global_eval_times >= X_test["duration"].min())
        & (global_eval_times <= X_test["duration"].max())
    ]

    H0_fn = interp1d(
        t0,
        H0,
        kind="previous",
        bounds_error=False,
        fill_value=(0, H0[-1]),
    )

    H0_eval = H0_fn(fold_eval_times)

    # -------------------------------
    # Test set survival predictions
    # -------------------------------
    risk_test = np.exp(X_test["log_risk"].values)
    surv_probs = np.exp(-np.outer(risk_test, H0_eval))

    # -------------------------------
    # Null-model (Kaplan-Meier)
    # -------------------------------
    from sksurv.nonparametric import kaplan_meier_estimator

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

    # -------------------------------
    # Save fold summary
    # -------------------------------
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
# Convert to DataFrames
# --------------------------------------------------
brier_summary_df = pd.DataFrame(brier_summary_results)
brier_time_df = pd.DataFrame(brier_time_results)

# --------------------------------------------------
# Across-fold summary
# --------------------------------------------------
ibs_m, ibs_l, ibs_u, ibs_se = nadeau_bengio_ci(ibs_list, k=n_folds)

null_m, null_l, null_u, null_se = nadeau_bengio_ci(ibs_null_list, k=n_folds)

gain_m, gain_l, gain_u, gain_se = nadeau_bengio_ci(ibs_gain_list, k=n_folds)

b1_m, b1_l, b1_u, b1_se = nadeau_bengio_ci(brier_summary_df["brier_1yr"], k=n_folds)

b5_m, b5_l, b5_u, b5_se = nadeau_bengio_ci(brier_summary_df["brier_5yr"], k=n_folds)

summary_df = pd.DataFrame(
    {
        "metric": [
            "ibs",
            "ibs_null",
            "ibs_gain",
            "brier_1yr",
            "brier_5yr",
        ],
        "mean": [
            ibs_m,
            null_m,
            gain_m,
            b1_m,
            b5_m,
        ],
        "SE_NB": [
            ibs_se,
            null_se,
            gain_se,
            b1_se,
            b5_se,
        ],
        "lower_95CI": [
            ibs_l,
            null_l,
            gain_l,
            b1_l,
            b5_l,
        ],
        "upper_95CI": [
            ibs_u,
            null_u,
            gain_u,
            b1_u,
            b5_u,
        ],
    }
)

print("\n=== DeepSurv Brier Score Summary Across Folds ===")
print(summary_df)

# --------------------------------------------------
# Export to Excel
# --------------------------------------------------
excel_path = f"{output_dir}{filename}_Brier.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    brier_summary_df.to_excel(writer, sheet_name="PerFold_Summary", index=False)
    brier_time_df.to_excel(writer, sheet_name="Brier_byTime", index=False)
    summary_df.to_excel(writer, sheet_name="AcrossFold_Summary", index=False)

# %%
# STEP 17
# Population-average survival function
# Averaged over covariate distribution in Xts_1fold

# --------------------------------------------------
# Select model
# --------------------------------------------------
# Use first fold model

trained_model = trained_models_per_fold[0]["model"]
trained_model.eval()

device = next(trained_model.parameters()).device

# --------------------------------------------------
# Data
# --------------------------------------------------
# Combine the train and test portions transformed for fold 0. This ensures the
# population data use the same scaler and one-hot-encoded features as the
# selected fold-0 model.
X_all = pd.concat(
    [Xts_train_folds[0], Xts_test_folds[0]],
    ignore_index=True,
)

# Preserve the exact selected-feature order used during model training, and
# explicitly convert the resulting matrix to a PyTorch-compatible dtype.
X_features = X_all[selected_features].astype(np.float32)

# --------------------------------------------------
# Time grid
# --------------------------------------------------
time_grid = np.linspace(
    X_all["duration"].min(),
    X_all["duration"].max() - 1e-3,
    100,
)

# --------------------------------------------------
# Predict log-risk for all observations
# --------------------------------------------------
with torch.no_grad():
    X_tensor = torch.as_tensor(
        X_features.to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    log_risk = trained_model(X_tensor).flatten().cpu().numpy()

risk_scores = np.exp(log_risk)

# --------------------------------------------------
# Breslow baseline cumulative hazard
# --------------------------------------------------
df_all = pd.DataFrame(
    {
        "duration": X_all["duration"].values,
        "event": X_all["event"].values,
        "risk": risk_scores,
    }
).sort_values("duration")

cumulative_hazard = 0.0
baseline_hazard = {}

event_times = np.sort(df_all.loc[df_all["event"] == 1, "duration"].unique())

for t in event_times:
    at_risk = df_all["duration"] >= t

    n_events = ((df_all["duration"] == t) & (df_all["event"] == 1)).sum()

    risk_sum = df_all.loc[at_risk, "risk"].sum()

    h_t = n_events / risk_sum

    cumulative_hazard += h_t

    baseline_hazard[t] = cumulative_hazard

# --------------------------------------------------
# Interpolate cumulative hazard
# --------------------------------------------------
baseline_times = np.array(list(baseline_hazard.keys()))
cumhaz_values = np.array(list(baseline_hazard.values()))

cumhaz_interp = np.interp(
    time_grid,
    baseline_times,
    cumhaz_values,
    left=0,
    right=cumhaz_values[-1],
)

# --------------------------------------------------
# Individual survival curves
# --------------------------------------------------
surv_matrix = np.exp(-np.outer(cumhaz_interp, risk_scores))

# --------------------------------------------------
# Population-average survival
# --------------------------------------------------
mean_survival = surv_matrix.mean(axis=1)

# --------------------------------------------------
# Save survival function
# --------------------------------------------------
surv_df = pd.DataFrame(
    {
        "time_days": time_grid,
        "time_years": time_grid / 365.25,
        "mean_survival": mean_survival,
    }
)

output_csv_path = f"{output_dir}{filename}_PopulationAverageSurvival.csv"

surv_df.to_csv(output_csv_path, index=False)

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    time_grid / 365.25,
    mean_survival,
    linewidth=2,
)

plt.title("DeepSurv (SHAP Filter): Population-Average Survival Function")

plt.xlabel("Years from baseline")
plt.ylabel("Survival probability")

plt.grid(True)

output_plot_path = f"{output_dir}{filename}_PopulationAverageSurvival.png"

plt.savefig(
    output_plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
