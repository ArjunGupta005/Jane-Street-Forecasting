# ============================================================
# IMPORTS
# ============================================================
# This script trains the first nonlinear forecasting model using
# the same chronological development split used for the linear
# baselines.

import sys
import time
from pathlib import Path

import lightgbm as lgb
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

from src.config import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    METRICS_PATH,
    RANDOM_SEED,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_PATH,
    WEIGHT_COLUMN,
)

from src.metrics import (
    calculate_forecast_metrics,
)

from src.tree_models import (
    create_lightgbm_baseline,
)


# ============================================================
# DEVELOPMENT DATE RANGES
# ============================================================
# Decision:
# Use exactly the same temporal period as the first linear-model
# experiment.
#
# Why:
# Holding the train/validation dates constant makes the comparison
# between linear and nonlinear models meaningful.

TRAIN_START_DATE = 0
TRAIN_END_DATE = 1298

VALIDATION_START_DATE = 1299
VALIDATION_END_DATE = 1498


# ============================================================
# SYSTEMATIC CPU-FRIENDLY SUBSAMPLING
# ============================================================
# Decision:
# Use the same systematic time_id sampling as the linear baseline.
#
# Why:
# If LightGBM used substantially more observations, we would not
# know whether improvement came from nonlinear modeling or simply
# additional data.

TRAIN_TIME_STRIDE = 40

VALIDATION_TIME_STRIDE = 20


# ============================================================
# MODEL FEATURES
# ============================================================
# Decision:
# Start with raw features plus time_id and symbol_id.
#
# We do NOT add date_id initially.
#
# Why:
# date_id may encourage memorization of historical drift rather
# than relationships that generalize to future periods.
#
# We also initially let LightGBM handle missing values directly,
# rather than zero-imputing them.

MODEL_FEATURES = (
    FEATURE_COLUMNS
    + [
        TIME_COLUMN,
        SYMBOL_COLUMN,
    ]
)


# ============================================================
# LAZY-SCAN TRAINING DATA
# ============================================================

training_scan = pl.scan_parquet(
    str(
        TRAIN_PATH
        / "**"
        / "*.parquet"
    )
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

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
# LOAD TRAINING SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("LOADING LIGHTGBM TRAINING SAMPLE")
print("=" * 80)

load_start = time.time()

training_data = (
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
    f"Training rows: "
    f"{training_data.height:,}"
)

print(
    f"Training dates: "
    f"{training_data[DATE_COLUMN].min()} -> "
    f"{training_data[DATE_COLUMN].max()}"
)

print(
    f"Load time: "
    f"{time.time() - load_start:.2f} seconds"
)


# ============================================================
# LOAD VALIDATION SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("LOADING LIGHTGBM VALIDATION SAMPLE")
print("=" * 80)

load_start = time.time()

validation_data = (
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
    f"Validation rows: "
    f"{validation_data.height:,}"
)

print(
    f"Validation dates: "
    f"{validation_data[DATE_COLUMN].min()} -> "
    f"{validation_data[DATE_COLUMN].max()}"
)

print(
    f"Load time: "
    f"{time.time() - load_start:.2f} seconds"
)


# ============================================================
# TEMPORAL INTEGRITY CHECK
# ============================================================

assert (
    training_data[DATE_COLUMN].max()
    <
    validation_data[DATE_COLUMN].min()
)


# ============================================================
# BUILD MODEL MATRICES
# ============================================================
# Decision:
# Leave native nulls untouched.
#
# Why:
# LightGBM learns the preferred branch for missing observations
# directly during tree construction.
#
# Unlike Ridge/Elastic Net:
#
# - no imputation is required
# - no standardization is required

X_train = (
    training_data
    .select(MODEL_FEATURES)
    .to_pandas()
)

X_validation = (
    validation_data
    .select(MODEL_FEATURES)
    .to_pandas()
)

y_train = (
    training_data[TARGET_COLUMN]
    .to_numpy()
)

y_validation = (
    validation_data[TARGET_COLUMN]
    .to_numpy()
)

training_weights = (
    training_data[WEIGHT_COLUMN]
    .to_numpy()
)

validation_weights = (
    validation_data[WEIGHT_COLUMN]
    .to_numpy()
)


# ============================================================
# SYMBOL ID CATEGORICAL TREATMENT
# ============================================================
# Decision:
# Treat symbol_id as categorical rather than numerical.
#
# Why:
# symbol_id is an encrypted identifier.
#
# Symbol 30 is not economically "three times larger" than symbol
# 10, so imposing numerical distance would be artificial.
#
# LightGBM can directly work with pandas categorical predictors.
#
# Tradeoff:
# New symbols may appear in future periods. We therefore still
# cannot allow the model to depend entirely on memorized symbol
# identities.

X_train[SYMBOL_COLUMN] = (
    X_train[SYMBOL_COLUMN]
    .astype("category")
)

X_validation[SYMBOL_COLUMN] = (
    pd.Categorical(
        X_validation[SYMBOL_COLUMN],
        categories=(
            X_train[SYMBOL_COLUMN]
            .cat
            .categories
        ),
    )
)


# ============================================================
# CREATE LIGHTGBM MODEL
# ============================================================

model = create_lightgbm_baseline(
    random_seed=RANDOM_SEED,
)


# ============================================================
# TRAIN LIGHTGBM
# ============================================================
# Decision:
# Use validation loss for early stopping.
#
# Why:
# Additional trees eventually begin fitting noise.
#
# Early stopping terminates training if the validation objective
# does not improve for 100 consecutive boosting rounds.
#
# This is still model-development validation; the later fold 4
# remains available for stronger confirmation.

print("\n" + "=" * 80)
print("TRAINING LIGHTGBM")
print("=" * 80)

training_start = time.time()

model.fit(
    X_train,
    y_train,
    sample_weight=training_weights,
    eval_set=[
        (
            X_validation,
            y_validation,
        )
    ],
    eval_sample_weight=[
        validation_weights
    ],
    eval_metric="l2",
    categorical_feature=[
        SYMBOL_COLUMN
    ],
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=100,
            verbose=True,
        ),
        lgb.log_evaluation(
            period=50,
        ),
    ],
)

training_seconds = (
    time.time()
    - training_start
)


# ============================================================
# REPORT BEST ITERATION
# ============================================================

print(
    f"\nBest boosting iteration: "
    f"{model.best_iteration_}"
)

print(
    f"Training time: "
    f"{training_seconds:.2f} seconds"
)


# ============================================================
# GENERATE VALIDATION PREDICTIONS
# ============================================================

predictions = model.predict(
    X_validation,
    num_iteration=model.best_iteration_,
)


# ============================================================
# CALCULATE FORECAST METRICS
# ============================================================

metrics = calculate_forecast_metrics(
    y_validation,
    predictions,
    validation_weights,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("LIGHTGBM BASELINE RESULTS")
print("=" * 80)

for metric_name, metric_value in metrics.items():

    print(
        f"{metric_name:<30} "
        f"{metric_value:.8f}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
# Decision:
# Save gain-based feature importance.
#
# Why:
# This gives us an early view of which predictors LightGBM relies
# on most.
#
# Tradeoff:
# Feature importance is NOT equivalent to causal importance or
# even guaranteed out-of-sample usefulness.
#
# Correlated features can substitute for one another, and unstable
# features may appear important during one regime.
#
# We will later evaluate importance stability across temporal
# folds.

feature_importance = pd.DataFrame(
    {
        "feature": MODEL_FEATURES,
        "importance_gain":
            model.booster_
            .feature_importance(
                importance_type="gain"
            ),
        "importance_split":
            model.booster_
            .feature_importance(
                importance_type="split"
            ),
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        "importance_gain",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

result_row = {
    "model":
        "lightgbm_raw_baseline",
    "train_start_date":
        TRAIN_START_DATE,
    "train_end_date":
        TRAIN_END_DATE,
    "validation_start_date":
        VALIDATION_START_DATE,
    "validation_end_date":
        VALIDATION_END_DATE,
    "train_time_stride":
        TRAIN_TIME_STRIDE,
    "validation_time_stride":
        VALIDATION_TIME_STRIDE,
    "best_iteration":
        model.best_iteration_,
    "training_seconds":
        training_seconds,
    **metrics,
}

result_frame = pd.DataFrame(
    [result_row]
)

result_frame.to_csv(
    METRICS_PATH
    / "lightgbm_baseline_results.csv",
    index=False,
)

feature_importance.to_csv(
    METRICS_PATH
    / "lightgbm_feature_importance.csv",
    index=False,
)


# ============================================================
# PRINT TOP FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("TOP 20 LIGHTGBM FEATURES BY GAIN")
print("=" * 80)

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved results to:")
print(
    METRICS_PATH
    / "lightgbm_baseline_results.csv"
)

print("\nSaved feature importance to:")
print(
    METRICS_PATH
    / "lightgbm_feature_importance.csv"
)

print("\nLightGBM baseline complete.")
