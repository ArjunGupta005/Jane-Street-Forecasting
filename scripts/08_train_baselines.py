# ============================================================
# IMPORTS
# ============================================================
# This script trains and evaluates the first forecasting baselines:
#
# 1. Zero prediction
# 2. Previous-day responder baseline
# 3. Weighted linear regression
# 4. Ridge regression
# 5. Elastic Net
#
# Every model is evaluated on observations that occur strictly
# after the training period.


import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


# ============================================================
# ALLOW IMPORTS FROM PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# LOAD PROJECT MODULES
# ============================================================

from src.baseline_models import (
    create_elastic_net_model,
    create_ridge_model,
    create_weighted_linear_regression,
    predict_zero,
)

from src.config import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    METRICS_PATH,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_PATH,
    WEIGHT_COLUMN,
)

from src.metrics import (
    calculate_forecast_metrics,
)

from src.preprocessing import (
    apply_standardizer,
    build_model_feature_list,
    fit_standardizer,
    get_linear_model_config,
    preprocess_polars_dataframe,
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

METRICS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVELOPMENT EXPERIMENT DATES
# ============================================================
# Decision:
# Use fold 3 as the first model-development period:
#
# training:   date_id 0 -> 1298
# validation: date_id 1299 -> 1498
#
# Why:
# This preserves a large historical training period while keeping
# an untouched future 200-date validation window.
#
# Fold 4 remains untouched for now and provides an additional
# later validation period once we begin serious model selection.

TRAIN_START_DATE = 0
TRAIN_END_DATE = 1298

VALIDATION_START_DATE = 1299
VALIDATION_END_DATE = 1498


# ============================================================
# CPU-FRIENDLY SYSTEMATIC SUBSAMPLING
# ============================================================
# Decision:
# Keep only observations whose time_id is divisible by a fixed
# interval during early model development.
#
# Why:
# Training linear models on all 32+ million fold-3 training rows
# would greatly slow iteration on a CPU-only machine.
#
# Sampling by time_id is preferable to taking the first N rows
# because it preserves:
#
# - the entire chronological range
# - many observations from every date
# - the cross-section of symbols
# - multiple intraday locations
#
# Tradeoff:
# These development scores are estimates of full-data performance.
# Promising models will later be retrained on substantially more
# or all available observations.

TRAIN_TIME_STRIDE = 40

VALIDATION_TIME_STRIDE = 20


# ============================================================
# LOAD TRAINING DATA LAZILY
# ============================================================

training_scan = pl.scan_parquet(
    str(
        TRAIN_PATH
        / "**"
        / "*.parquet"
    )
)


# ============================================================
# DEFINE REQUIRED MODEL COLUMNS
# ============================================================

linear_config = get_linear_model_config()

model_features = build_model_feature_list(
    FEATURE_COLUMNS,
    linear_config,
)

required_columns = (
    [
        DATE_COLUMN,
        TIME_COLUMN,
        SYMBOL_COLUMN,
        WEIGHT_COLUMN,
        TARGET_COLUMN,
    ]
    + FEATURE_COLUMNS
)


# ============================================================
# BUILD CHRONOLOGICAL TRAINING SAMPLE
# ============================================================
# No random split is performed.
#
# We first restrict observations by date and then apply the
# deterministic time_id stride.

print("\n" + "=" * 80)
print("LOADING DEVELOPMENT TRAINING SAMPLE")
print("=" * 80)

start_time = time.time()

training_sample = (
    training_scan
    .filter(
        (pl.col(DATE_COLUMN) >= TRAIN_START_DATE)
        &
        (pl.col(DATE_COLUMN) <= TRAIN_END_DATE)
        &
        (
            pl.col(TIME_COLUMN)
            % TRAIN_TIME_STRIDE
            == 0
        )
    )
    .select(required_columns)
    .collect()
)

print(
    f"Training rows collected: "
    f"{training_sample.height:,}"
)

print(
    f"Training dates: "
    f"{training_sample[DATE_COLUMN].min()} -> "
    f"{training_sample[DATE_COLUMN].max()}"
)

print(
    f"Load time: "
    f"{time.time() - start_time:.2f} seconds"
)


# ============================================================
# BUILD CHRONOLOGICAL VALIDATION SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("LOADING DEVELOPMENT VALIDATION SAMPLE")
print("=" * 80)

start_time = time.time()

validation_sample = (
    training_scan
    .filter(
        (pl.col(DATE_COLUMN) >= VALIDATION_START_DATE)
        &
        (pl.col(DATE_COLUMN) <= VALIDATION_END_DATE)
        &
        (
            pl.col(TIME_COLUMN)
            % VALIDATION_TIME_STRIDE
            == 0
        )
    )
    .select(required_columns)
    .collect()
)

print(
    f"Validation rows collected: "
    f"{validation_sample.height:,}"
)

print(
    f"Validation dates: "
    f"{validation_sample[DATE_COLUMN].min()} -> "
    f"{validation_sample[DATE_COLUMN].max()}"
)

print(
    f"Load time: "
    f"{time.time() - start_time:.2f} seconds"
)


# ============================================================
# TEMPORAL INTEGRITY CHECK
# ============================================================
# This assertion prevents accidental overlap between model fitting
# and evaluation.

assert (
    training_sample[DATE_COLUMN].max()
    <
    validation_sample[DATE_COLUMN].min()
)


# ============================================================
# PREPROCESS TRAINING DATA
# ============================================================
# Linear models require finite numerical inputs.
#
# The verified preprocessing pipeline therefore:
#
# 1. creates missingness indicators
# 2. zero-imputes the 79 raw features
#
# Scaling is fitted separately below using training observations
# only.

print("\n" + "=" * 80)
print("PREPROCESSING TRAINING AND VALIDATION DATA")
print("=" * 80)

training_processed = preprocess_polars_dataframe(
    training_sample,
    FEATURE_COLUMNS,
    linear_config,
)

validation_processed = preprocess_polars_dataframe(
    validation_sample,
    FEATURE_COLUMNS,
    linear_config,
)


# ============================================================
# CONVERT MODEL MATRICES TO PANDAS
# ============================================================
# scikit-learn accepts NumPy/pandas inputs.
#
# Only the systematic development samples are converted, not the
# full 47-million-row dataset.

X_train = (
    training_processed
    .select(model_features)
    .to_pandas()
)

X_validation = (
    validation_processed
    .select(model_features)
    .to_pandas()
)

y_train = (
    training_processed[TARGET_COLUMN]
    .to_numpy()
)

y_validation = (
    validation_processed[TARGET_COLUMN]
    .to_numpy()
)

training_weights = (
    training_processed[WEIGHT_COLUMN]
    .to_numpy()
)

validation_weights = (
    validation_processed[WEIGHT_COLUMN]
    .to_numpy()
)


# ============================================================
# STANDARDIZE PREDICTORS USING TRAINING DATA ONLY
# ============================================================
# Decision:
# Standardize every predictor for the initial regularized-linear
# baseline.
#
# Why:
# Ridge and Elastic Net penalties operate on coefficient magnitude.
# Without scaling, features with different numerical ranges would
# receive unequal effective regularization.
#
# Leakage prevention:
# Means and standard deviations are estimated ONLY from X_train.
# The validation data is transformed using those frozen training
# statistics.

print("\nFitting training-only standardizer...")

standardization_state = fit_standardizer(
    X_train,
    model_features,
)

X_train_scaled = apply_standardizer(
    X_train,
    standardization_state,
)

X_validation_scaled = apply_standardizer(
    X_validation,
    standardization_state,
)


# ============================================================
# FREE UNNECESSARY INTERMEDIATE DATA
# ============================================================
# Mechanical memory-management step.
#
# The raw Polars frames are no longer needed once the model arrays
# and targets have been extracted.

del training_sample
del validation_sample
del training_processed
del validation_processed
del X_train
del X_validation


# ============================================================
# RESULTS CONTAINER
# ============================================================

model_results = []


# ============================================================
# MODEL EVALUATION HELPER
# ============================================================

def record_model_results(
    model_name,
    predictions,
    training_seconds=None,
):

    metrics = calculate_forecast_metrics(
        y_validation,
        predictions,
        validation_weights,
    )

    result = {
        "model": model_name,
        "training_seconds": training_seconds,
        **metrics,
    }

    model_results.append(result)

    print("\n" + "-" * 80)
    print(model_name)
    print("-" * 80)

    print(
        f"Weighted zero-mean R²: "
        f"{metrics['weighted_zero_mean_r2']:.8f}"
    )

    print(
        f"Weighted RMSE:         "
        f"{metrics['weighted_rmse']:.6f}"
    )

    print(
        f"Weighted MAE:          "
        f"{metrics['weighted_mae']:.6f}"
    )

    print(
        f"Information coefficient: "
        f"{metrics['information_coefficient']:.6f}"
    )

    print(
        f"Directional accuracy:  "
        f"{metrics['directional_accuracy']:.6f}"
    )


# ============================================================
# MODEL 1: ZERO BASELINE
# ============================================================
# This should produce weighted zero-mean R² = 0 by definition.

zero_predictions = predict_zero(
    len(y_validation)
)

record_model_results(
    "zero_baseline",
    zero_predictions,
    training_seconds=0.0,
)


# ============================================================
# MODEL 2: PREVIOUS-DAY RESPONDER BASELINE
# ============================================================
# Decision:
# Predict today's responder_6 using responder_6 from the same
# symbol and time_id on the immediately previous date.
#
# Why:
# The competition's inference design exposes previous-day responder
# information, making this a realistic leakage-safe baseline.
#
# This tests whether responder_6 itself contains useful one-day
# persistence.
#
# IMPORTANT:
# We require EXACTLY date_id - 1.
#
# A simple shift over symbol/time would be dangerous because if a
# symbol were absent on one date it could accidentally use an older
# observation as though it were yesterday's value.

print("\nBuilding previous-day responder baseline...")

lag_source = (
    training_scan
    .filter(
        (pl.col(DATE_COLUMN) >= VALIDATION_START_DATE - 1)
        &
        (pl.col(DATE_COLUMN) <= VALIDATION_END_DATE - 1)
        &
        (
            pl.col(TIME_COLUMN)
            % VALIDATION_TIME_STRIDE
            == 0
        )
    )
    .select(
        (
            pl.col(DATE_COLUMN)
            + 1
        ).alias(DATE_COLUMN),

        pl.col(TIME_COLUMN),

        pl.col(SYMBOL_COLUMN),

        pl.col(TARGET_COLUMN)
        .alias("previous_day_responder_6"),
    )
)

validation_keys = (
    training_scan
    .filter(
        (pl.col(DATE_COLUMN) >= VALIDATION_START_DATE)
        &
        (pl.col(DATE_COLUMN) <= VALIDATION_END_DATE)
        &
        (
            pl.col(TIME_COLUMN)
            % VALIDATION_TIME_STRIDE
            == 0
        )
    )
    .select(
        [
            DATE_COLUMN,
            TIME_COLUMN,
            SYMBOL_COLUMN,
        ]
    )
)

lag_predictions_frame = (
    validation_keys
    .join(
        lag_source,
        on=[
            DATE_COLUMN,
            TIME_COLUMN,
            SYMBOL_COLUMN,
        ],
        how="left",
    )
    .select(
        pl.col(
            "previous_day_responder_6"
        )
        .fill_null(0.0)
        .alias("prediction")
    )
    .collect()
)

lag_predictions = (
    lag_predictions_frame["prediction"]
    .to_numpy()
)

assert (
    len(lag_predictions)
    == len(y_validation)
)

record_model_results(
    "previous_day_responder_baseline",
    lag_predictions,
    training_seconds=0.0,
)


# ============================================================
# MODEL 3: WEIGHTED LINEAR REGRESSION
# ============================================================
# Decision:
# Use observation weights during fitting.
#
# Why:
# Rows with larger Jane Street weights have more influence on the
# competition objective, so the training loss should reflect the
# same relative importance.

print("\nTraining weighted linear regression...")

linear_model = (
    create_weighted_linear_regression()
)

start_time = time.time()

linear_model.fit(
    X_train_scaled,
    y_train,
    sample_weight=training_weights,
)

linear_training_seconds = (
    time.time()
    - start_time
)

linear_predictions = linear_model.predict(
    X_validation_scaled
)

record_model_results(
    "weighted_linear_regression",
    linear_predictions,
    linear_training_seconds,
)


# ============================================================
# MODEL 4: RIDGE REGRESSION
# ============================================================
# Initial alpha = 1.0.
#
# This is not yet hyperparameter tuning. We first establish whether
# coefficient shrinkage improves temporal generalization at all.

print("\nTraining Ridge regression...")

ridge_model = create_ridge_model(
    alpha=1.0,
)

start_time = time.time()

ridge_model.fit(
    X_train_scaled,
    y_train,
    sample_weight=training_weights,
)

ridge_training_seconds = (
    time.time()
    - start_time
)

ridge_predictions = ridge_model.predict(
    X_validation_scaled
)

record_model_results(
    "ridge_alpha_1",
    ridge_predictions,
    ridge_training_seconds,
)


# ============================================================
# MODEL 5: ELASTIC NET
# ============================================================
# Elastic Net is more computationally expensive than Ridge, but
# tests whether sparse coefficient selection helps this weak-signal
# problem.

print("\nTraining Elastic Net...")

elastic_net_model = create_elastic_net_model(
    alpha=0.001,
    l1_ratio=0.5,
)

start_time = time.time()

elastic_net_model.fit(
    X_train_scaled,
    y_train,
    sample_weight=training_weights,
)

elastic_training_seconds = (
    time.time()
    - start_time
)

elastic_predictions = (
    elastic_net_model.predict(
        X_validation_scaled
    )
)

record_model_results(
    "elastic_net_alpha_0.001_l1_0.5",
    elastic_predictions,
    elastic_training_seconds,
)


# ============================================================
# COMBINE MODEL RESULTS
# ============================================================

results_frame = pd.DataFrame(
    model_results
)

results_frame = (
    results_frame
    .sort_values(
        "weighted_zero_mean_r2",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE BASELINE RESULTS
# ============================================================

results_path = (
    METRICS_PATH
    / "baseline_model_results.csv"
)

results_frame.to_csv(
    results_path,
    index=False,
)


# ============================================================
# PRINT MODEL RANKING
# ============================================================

print("\n" + "=" * 100)
print("BASELINE MODEL RANKING")
print("=" * 100)

display_columns = [
    "model",
    "weighted_zero_mean_r2",
    "weighted_rmse",
    "weighted_mae",
    "information_coefficient",
    "directional_accuracy",
    "training_seconds",
]

print(
    results_frame[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved baseline results to:")
print(results_path)

print("\nBaseline experiment complete.")
