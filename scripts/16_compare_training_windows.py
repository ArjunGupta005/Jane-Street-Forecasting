# ============================================================
# IMPORTS
# ============================================================
# This script tests whether removing stale historical data improves
# performance on the Fold 4 validation window, where the winning
# model showed meaningful degradation (R² 0.016244 -> 0.009040).
#
# Validation period is held FIXED at [1499, 1698] across all three
# runs. Only the training window changes:
#
#   Expanding   : [0,   1498]  (same as Fold 4 in script 15)
#   Rolling 700 : [700, 1498]
#   Rolling 900 : [900, 1498]
#
# The feature set (raw + cross-sectional - symbol_id) and the
# LightGBM configuration are IDENTICAL across all three runs and
# identical to every prior experiment on this specification. The
# only variable under test is how much training history is used.
#
# Question:
# Does removing stale historical data improve performance in the
# later regime, or does the larger expanding sample still win
# despite nonstationarity?

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
# PROJECT IMPORTS
# ============================================================

from src.config import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    METRICS_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_PATH,
    WEIGHT_COLUMN,
)

from src.feature_engineering import (
    CORE_ENGINEERING_FEATURES,
    add_cross_sectional_features,
    cross_sectional_feature_names,
)

from src.metrics import (
    calculate_forecast_metrics,
)

from src.tree_models import (
    create_lightgbm_baseline,
)


# ============================================================
# WINNING FEATURE SET
# ============================================================
# Identical to scripts 12/15: raw features + cross-sectional
# features, symbol_id excluded.

CROSS_SECTIONAL_BASE_FEATURES = CORE_ENGINEERING_FEATURES

CROSS_SECTIONAL_FEATURES = cross_sectional_feature_names(
    CROSS_SECTIONAL_BASE_FEATURES
)

MODEL_FEATURES = (
    FEATURE_COLUMNS
    + [TIME_COLUMN]
    + CROSS_SECTIONAL_FEATURES
)


# ============================================================
# FIXED VALIDATION PERIOD
# ============================================================
# Held constant across every training-window variant so the
# comparison isolates exactly one variable: how much training
# history is used.

VALIDATION_START_DATE = 1499
VALIDATION_END_DATE = 1698


# ============================================================
# TRAINING WINDOW VARIANTS
# ============================================================

TRAINING_WINDOWS = [
    ("expanding", 0, 1498),
    ("rolling_700", 700, 1498),
    ("rolling_900", 900, 1498),
]


# ============================================================
# CPU-FRIENDLY SYSTEMATIC SAMPLING
# ============================================================
# Identical stride policy to every prior LightGBM experiment.

TRAIN_TIME_STRIDE = 40
VALIDATION_TIME_STRIDE = 20


# ============================================================
# LAZY-SCAN + CROSS-SECTIONALLY ENRICH FULL DATASET
# ============================================================

training_scan = pl.scan_parquet(
    str(
        TRAIN_PATH
        / "**"
        / "*.parquet"
    )
)

enriched_scan = add_cross_sectional_features(
    training_scan,
    features=CROSS_SECTIONAL_BASE_FEATURES,
)

required_select_columns = (
    MODEL_FEATURES
    + [
        DATE_COLUMN,
        TARGET_COLUMN,
        WEIGHT_COLUMN,
    ]
)


# ============================================================
# LOAD THE FIXED VALIDATION SET ONCE
# ============================================================
# Decision:
# Load and hold the validation set once rather than reloading it
# per training window.
#
# Why:
# The validation period never changes across these three runs, so
# repeating the load/filter would be wasted work and, more
# importantly, would risk the three runs silently using
# non-identical validation data if anything upstream changed
# between loads.

print("\n" + "=" * 90)
print("LOADING FIXED VALIDATION SET")
print("=" * 90)

load_start = time.time()

validation_data = (
    enriched_scan
    .filter(
        (pl.col(DATE_COLUMN) >= VALIDATION_START_DATE)
        & (pl.col(DATE_COLUMN) <= VALIDATION_END_DATE)
        & (pl.col(TIME_COLUMN) % VALIDATION_TIME_STRIDE == 0)
    )
    .select(required_select_columns)
    .collect()
)

print(f"Validation rows: {validation_data.height:,}")
print(f"Load time: {time.time() - load_start:.2f} seconds")

X_validation = (
    validation_data
    .select(MODEL_FEATURES)
    .to_pandas()
)
y_validation = validation_data[TARGET_COLUMN].to_numpy()
validation_weights = validation_data[WEIGHT_COLUMN].to_numpy()


# ============================================================
# RESULTS CONTAINER
# ============================================================

window_results = []


# ============================================================
# TRAINING WINDOW LOOP
# ============================================================
# Each window retrains LightGBM from scratch with the exact same
# create_lightgbm_baseline() configuration used in every prior
# experiment. No hyperparameters are changed here - the only
# variable under test is the training date range.

for window_name, train_start_date, train_end_date in TRAINING_WINDOWS:

    print("\n" + "=" * 90)
    print(f"TRAINING WINDOW: {window_name}")
    print(f"Train dates: {train_start_date} -> {train_end_date}")
    print("=" * 90)

    load_start = time.time()

    training_data = (
        enriched_scan
        .filter(
            (pl.col(DATE_COLUMN) >= train_start_date)
            & (pl.col(DATE_COLUMN) <= train_end_date)
            & (pl.col(TIME_COLUMN) % TRAIN_TIME_STRIDE == 0)
        )
        .select(required_select_columns)
        .collect()
    )

    print(f"Training rows: {training_data.height:,}")
    print(f"Load time: {time.time() - load_start:.2f} seconds")

    assert (
        training_data[DATE_COLUMN].max()
        < VALIDATION_START_DATE
    )

    X_train = (
        training_data
        .select(MODEL_FEATURES)
        .to_pandas()
    )
    y_train = training_data[TARGET_COLUMN].to_numpy()
    training_weights = training_data[WEIGHT_COLUMN].to_numpy()

    training_row_count = training_data.height

    del training_data

    model = create_lightgbm_baseline(
        random_seed=RANDOM_SEED,
    )

    print("\nTraining LightGBM...")

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
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=True,
            ),
            lgb.log_evaluation(
                period=100,
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
        "window": window_name,
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "train_days": train_end_date - train_start_date + 1,
        "training_rows": training_row_count,
        "best_iteration": model.best_iteration_,
        "training_seconds": training_seconds,
        **metrics,
    }

    window_results.append(result_row)

    print("\n" + "-" * 90)
    print(f"{window_name.upper()} RESULT")
    print("-" * 90)
    print(
        f"Weighted zero-mean R²: "
        f"{metrics['weighted_zero_mean_r2']:.8f}"
    )
    print(
        f"Information coefficient: "
        f"{metrics['information_coefficient']:.8f}"
    )
    print(
        f"Directional accuracy: "
        f"{metrics['directional_accuracy']:.8f}"
    )
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Training seconds: {training_seconds:.2f}")

    del X_train
    del y_train
    del training_weights
    del predictions
    del model


# ============================================================
# COMBINE AND SAVE RESULTS
# ============================================================

results_frame = (
    pd.DataFrame(window_results)
    .sort_values(
        "weighted_zero_mean_r2",
        ascending=False,
    )
    .reset_index(drop=True)
)

results_path = (
    METRICS_PATH
    / "training_window_comparison_results.csv"
)

results_frame.to_csv(
    results_path,
    index=False,
)


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 110)
print("TRAINING WINDOW COMPARISON (fixed validation 1499-1698)")
print("=" * 110)

display_columns = [
    "window",
    "train_start_date",
    "train_end_date",
    "train_days",
    "training_rows",
    "weighted_zero_mean_r2",
    "information_coefficient",
    "directional_accuracy",
    "best_iteration",
    "training_seconds",
]

print(
    results_frame[display_columns].to_string(index=False)
)


# ============================================================
# INTERPRETATION
# ============================================================

best_window = results_frame.iloc[0]

expanding_row = results_frame[
    results_frame["window"] == "expanding"
].iloc[0]

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(
    f"Best window: {best_window['window']} "
    f"(R² = {best_window['weighted_zero_mean_r2']:.6f})"
)

if best_window["window"] == "expanding":
    print(
        "The full expanding history remains best. Despite regime "
        "shifts, the larger sample still provides useful signal - "
        "stale data is not clearly hurting this model."
    )
else:
    improvement = (
        best_window["weighted_zero_mean_r2"]
        - expanding_row["weighted_zero_mean_r2"]
    )
    print(
        f"A shorter training window beat the expanding baseline by "
        f"{improvement:+.6f} R². This is evidence that old data is "
        f"hurting generalization into this later regime, and "
        f"adaptive/rolling retraining deserves further "
        f"investigation."
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved comparison results to:")
print(results_path)

print("\nTraining window comparison complete.")
