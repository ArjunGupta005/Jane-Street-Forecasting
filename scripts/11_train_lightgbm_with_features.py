# ============================================================
# IMPORTS
# ============================================================
# This script measures the incremental value of two engineered
# feature families on top of the raw-feature LightGBM baseline
# (weighted zero-mean R² = 0.01353):
#
#   1. raw                       (control point)
#   2. raw + cross-sectional     (market mean/deviation)
#   3. raw + rolling             (per-symbol lag/rolling stats)
#   4. raw + both
#
# Every model uses the exact same train/validation dates, the same
# time_id subsampling stride, the same Jane Street weights, and the
# same LightGBM hyperparameters as scripts/09_train_lightgbm_baseline.py,
# so any score change can be attributed to the feature family added
# rather than to a confounded experimental setup.

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

from src.feature_engineering import (
    add_cross_sectional_features,
    build_symbol_rolling_feature_table,
    cross_sectional_feature_names,
    merge_symbol_rolling_features,
    rolling_feature_names,
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
# Identical to the raw LightGBM baseline experiment.
#
# Why:
# Holding dates, stride, weights, and model hyperparameters fixed
# isolates the effect of the engineered feature families being
# tested.

TRAIN_START_DATE = 0
TRAIN_END_DATE = 1298

VALIDATION_START_DATE = 1299
VALIDATION_END_DATE = 1498

TRAIN_TIME_STRIDE = 40
VALIDATION_TIME_STRIDE = 20


# ============================================================
# RAW MODEL FEATURES
# ============================================================

RAW_MODEL_FEATURES = (
    FEATURE_COLUMNS
    + [
        TIME_COLUMN,
        SYMBOL_COLUMN,
    ]
)

CROSS_SECTIONAL_FEATURES = cross_sectional_feature_names()
ROLLING_FEATURES = rolling_feature_names()


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
# LOAD RAW TRAINING AND VALIDATION SAMPLES
# ============================================================
# Identical row selection to the raw baseline: date-range filter
# plus a systematic time_id stride, not a random sample.

print("\n" + "=" * 80)
print("LOADING TRAINING SAMPLE")
print("=" * 80)

load_start = time.time()

training_data = (
    training_scan
    .filter(
        (pl.col(DATE_COLUMN) >= TRAIN_START_DATE)
        & (pl.col(DATE_COLUMN) <= TRAIN_END_DATE)
        & (pl.col(TIME_COLUMN) % TRAIN_TIME_STRIDE == 0)
    )
    .select(required_columns)
    .collect()
)

print(f"Training rows: {training_data.height:,}")
print(f"Load time: {time.time() - load_start:.2f} seconds")

print("\n" + "=" * 80)
print("LOADING VALIDATION SAMPLE")
print("=" * 80)

load_start = time.time()

validation_data = (
    training_scan
    .filter(
        (pl.col(DATE_COLUMN) >= VALIDATION_START_DATE)
        & (pl.col(DATE_COLUMN) <= VALIDATION_END_DATE)
        & (pl.col(TIME_COLUMN) % VALIDATION_TIME_STRIDE == 0)
    )
    .select(required_columns)
    .collect()
)

print(f"Validation rows: {validation_data.height:,}")
print(f"Load time: {time.time() - load_start:.2f} seconds")

assert (
    training_data[DATE_COLUMN].max()
    < validation_data[DATE_COLUMN].min()
)


# ============================================================
# BUILD THE LEAKAGE-SAFE SYMBOL ROLLING FEATURE TABLE
# ============================================================
# Decision:
# Build ONE rolling/lag table spanning the full combined date range
# (train start -> validation end), rather than building it
# separately per split.
#
# Why this is still leakage-safe:
# Every column in this table is derived from a shift(1) series, so
# a row's engineered value for date d is built exclusively from
# that symbol's own strictly-earlier trading days - regardless of
# whether date d itself falls in the training or validation window.
# Building one continuous table simply lets validation dates near
# VALIDATION_START_DATE correctly see their most recent prior
# trading day, which lives in the training period, without any
# special-casing at the boundary.

print("\n" + "=" * 80)
print("BUILDING SYMBOL ROLLING/LAG FEATURE TABLE")
print("=" * 80)

rolling_build_start = time.time()

symbol_rolling_table = build_symbol_rolling_feature_table(
    training_scan,
    start_date=TRAIN_START_DATE,
    end_date=VALIDATION_END_DATE,
)

print(
    f"Symbol-day rows: {symbol_rolling_table.height:,}"
)
print(
    f"Build time: {time.time() - rolling_build_start:.2f} seconds"
)


# ============================================================
# ADD ENGINEERED FEATURES TO TRAINING AND VALIDATION DATA
# ============================================================

print("\n" + "=" * 80)
print("ADDING ENGINEERED FEATURES")
print("=" * 80)

engineering_start = time.time()

training_processed = add_cross_sectional_features(
    training_data
)
training_processed = merge_symbol_rolling_features(
    training_processed,
    symbol_rolling_table,
)

validation_processed = add_cross_sectional_features(
    validation_data
)
validation_processed = merge_symbol_rolling_features(
    validation_processed,
    symbol_rolling_table,
)

assert training_processed.height == training_data.height, (
    "Feature engineering changed the training row count."
)
assert validation_processed.height == validation_data.height, (
    "Feature engineering changed the validation row count."
)

print(
    f"Engineering time: "
    f"{time.time() - engineering_start:.2f} seconds"
)

print(
    f"Training columns after engineering: "
    f"{len(training_processed.columns)}"
)


# ============================================================
# CONVERT TO PANDAS ONCE, THEN SUBSET PER EXPERIMENT
# ============================================================
# Decision:
# Convert the fully-engineered frames to pandas a single time and
# select different column subsets for each of the four experiments,
# rather than repeating the polars->pandas conversion four times.

all_candidate_columns = (
    RAW_MODEL_FEATURES
    + CROSS_SECTIONAL_FEATURES
    + ROLLING_FEATURES
)

X_train_full = (
    training_processed
    .select(all_candidate_columns)
    .to_pandas()
)

X_validation_full = (
    validation_processed
    .select(all_candidate_columns)
    .to_pandas()
)

y_train = training_processed[TARGET_COLUMN].to_numpy()
y_validation = validation_processed[TARGET_COLUMN].to_numpy()

training_weights = training_processed[WEIGHT_COLUMN].to_numpy()
validation_weights = validation_processed[WEIGHT_COLUMN].to_numpy()


# ============================================================
# SYMBOL ID CATEGORICAL TREATMENT
# ============================================================
# Identical to the raw baseline: symbol_id is an encrypted
# identifier, not a meaningful numerical scale.

X_train_full[SYMBOL_COLUMN] = (
    X_train_full[SYMBOL_COLUMN]
    .astype("category")
)

X_validation_full[SYMBOL_COLUMN] = (
    pd.Categorical(
        X_validation_full[SYMBOL_COLUMN],
        categories=(
            X_train_full[SYMBOL_COLUMN]
            .cat
            .categories
        ),
    )
)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

experiments = [
    (
        "raw",
        RAW_MODEL_FEATURES,
    ),
    (
        "raw_plus_cross_sectional",
        RAW_MODEL_FEATURES + CROSS_SECTIONAL_FEATURES,
    ),
    (
        "raw_plus_rolling",
        RAW_MODEL_FEATURES + ROLLING_FEATURES,
    ),
    (
        "raw_plus_both",
        RAW_MODEL_FEATURES
        + CROSS_SECTIONAL_FEATURES
        + ROLLING_FEATURES,
    ),
]


# ============================================================
# TRAIN EACH EXPERIMENT
# ============================================================

experiment_results = []
last_feature_importance = None
last_model_features = None

for experiment_name, experiment_features in experiments:

    print("\n" + "=" * 80)
    print(f"TRAINING: {experiment_name}")
    print(f"Feature count: {len(experiment_features)}")
    print("=" * 80)

    X_train = X_train_full[experiment_features]
    X_validation = X_validation_full[experiment_features]

    model = create_lightgbm_baseline(
        random_seed=RANDOM_SEED,
    )

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
                verbose=False,
            ),
            lgb.log_evaluation(
                period=0,
            ),
        ],
    )

    training_seconds = time.time() - training_start

    predictions = model.predict(
        X_validation,
        num_iteration=model.best_iteration_,
    )

    metrics = calculate_forecast_metrics(
        y_validation,
        predictions,
        validation_weights,
    )

    result_row = {
        "model": experiment_name,
        "feature_count": len(experiment_features),
        "best_iteration": model.best_iteration_,
        "training_seconds": training_seconds,
        **metrics,
    }

    experiment_results.append(result_row)

    print(
        f"Best iteration: {model.best_iteration_}   "
        f"Training time: {training_seconds:.2f}s"
    )
    print(
        f"Weighted zero-mean R²: "
        f"{metrics['weighted_zero_mean_r2']:.8f}"
    )
    print(
        f"Information coefficient: "
        f"{metrics['information_coefficient']:.6f}"
    )

    if experiment_name == "raw_plus_both":

        last_model_features = experiment_features

        last_feature_importance = pd.DataFrame(
            {
                "feature": experiment_features,
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
        ).sort_values(
            "importance_gain",
            ascending=False,
        ).reset_index(drop=True)


# ============================================================
# COMBINE AND SAVE RESULTS
# ============================================================

results_frame = (
    pd.DataFrame(experiment_results)
    .sort_values(
        "weighted_zero_mean_r2",
        ascending=False,
    )
    .reset_index(drop=True)
)

results_path = (
    METRICS_PATH
    / "lightgbm_feature_engineering_results.csv"
)

results_frame.to_csv(
    results_path,
    index=False,
)

if last_feature_importance is not None:

    importance_path = (
        METRICS_PATH
        / "lightgbm_raw_plus_both_feature_importance.csv"
    )

    last_feature_importance.to_csv(
        importance_path,
        index=False,
    )


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 100)
print("FEATURE ENGINEERING COMPARISON")
print("=" * 100)

display_columns = [
    "model",
    "feature_count",
    "weighted_zero_mean_r2",
    "weighted_rmse",
    "information_coefficient",
    "directional_accuracy",
    "best_iteration",
    "training_seconds",
]

print(
    results_frame[display_columns].to_string(index=False)
)

if last_feature_importance is not None:

    print("\n" + "=" * 80)
    print("TOP 25 FEATURES BY GAIN: raw_plus_both")
    print("=" * 80)

    print(
        last_feature_importance
        .head(25)
        .to_string(index=False)
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved comparison results to:")
print(results_path)

print("\nFeature engineering comparison complete.")
