# ============================================================
# IMPORTS
# ============================================================
# This script tests how much the winning model's performance
# depends on training-data RECENCY specifically, as opposed to
# training-data VOLUME (already tested in script 16).
#
# Two scenarios, same fixed validation period [1499, 1698], same
# ~800-day training-history length in both cases:
#
#   No gap     : train [700, 1498]  (0-day gap to validation)
#   Gap stress : train [500, 1298]  (200-day gap to validation)
#
# The "no gap" scenario is exactly rolling_700 from script 16,
# rerun here in the same script as the gap scenario so both numbers
# come from one identical implementation - the same reasoning used
# for the fold comparison (script 15) and the window comparison
# (script 16).
#
# Why this isolates staleness rather than volume:
# Both training sets span roughly 800 days and are built with the
# same TRAIN_TIME_STRIDE, so row counts are similar. The only
# structural difference is a 200-day gap between the end of
# training and the start of validation in the gap-stress scenario -
# simulating a model that was trained once and then deployed for
# 200 days without being refreshed.
#
# If the gap model drops sharply relative to "no gap," recency of
# information is crucial, which strengthens the case for
# walk-forward retraining / online updating later. If it barely
# changes, the later regime is simply harder regardless of model
# freshness.

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
# Identical to scripts 12/15/16: raw features + cross-sectional
# features, symbol_id excluded. No hyperparameter changes.

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

VALIDATION_START_DATE = 1499
VALIDATION_END_DATE = 1698


# ============================================================
# GAP STRESS SCENARIOS
# ============================================================
# Both training windows span ~800 days so row counts stay
# comparable; only the gap to validation differs.

SCENARIOS = [
    (
        "no_gap",
        700,
        1498,
    ),
    (
        "gap_stress",
        500,
        1298,
    ),
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
# per scenario, so both scenarios are scored against byte-identical
# validation data.

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

scenario_results = []


# ============================================================
# SCENARIO LOOP
# ============================================================
# Each scenario retrains LightGBM from scratch with the exact same
# create_lightgbm_baseline() configuration. No hyperparameters are
# changed - the only variable under test is the gap between the end
# of training and the start of validation.

for scenario_name, train_start_date, train_end_date in SCENARIOS:

    gap_days = (
        VALIDATION_START_DATE
        - train_end_date
        - 1
    )

    print("\n" + "=" * 90)
    print(f"SCENARIO: {scenario_name}")
    print(
        f"Train dates: {train_start_date} -> {train_end_date}   "
        f"Gap: {gap_days} days"
    )
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
        "scenario": scenario_name,
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "gap_days": gap_days,
        "training_rows": training_row_count,
        "best_iteration": model.best_iteration_,
        "training_seconds": training_seconds,
        **metrics,
    }

    scenario_results.append(result_row)

    print("\n" + "-" * 90)
    print(f"{scenario_name.upper()} RESULT")
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

results_frame = pd.DataFrame(scenario_results)

results_path = (
    METRICS_PATH
    / "gap_stress_test_results.csv"
)

results_frame.to_csv(
    results_path,
    index=False,
)


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 110)
print("GAP STRESS TEST (fixed validation 1499-1698)")
print("=" * 110)

display_columns = [
    "scenario",
    "train_start_date",
    "train_end_date",
    "gap_days",
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

no_gap_row = results_frame[
    results_frame["scenario"] == "no_gap"
].iloc[0]

gap_stress_row = results_frame[
    results_frame["scenario"] == "gap_stress"
].iloc[0]

r2_drop = (
    no_gap_row["weighted_zero_mean_r2"]
    - gap_stress_row["weighted_zero_mean_r2"]
)

relative_drop = (
    r2_drop / no_gap_row["weighted_zero_mean_r2"]
    if no_gap_row["weighted_zero_mean_r2"] > 0
    else float("nan")
)

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(
    f"No-gap R²:     {no_gap_row['weighted_zero_mean_r2']:.6f}"
)
print(
    f"Gap-stress R²: {gap_stress_row['weighted_zero_mean_r2']:.6f}"
)
print(
    f"Absolute drop: {r2_drop:+.6f}   "
    f"Relative drop: {relative_drop:+.2%}"
)

if r2_drop > 0 and relative_drop > 0.15:
    print(
        "\nThe 200-day gap caused a substantial drop. Recency of "
        "information appears to matter meaningfully - this "
        "strengthens the case for walk-forward retraining / "
        "online updating rather than a single static model."
    )
elif r2_drop > 0:
    print(
        "\nThe 200-day gap caused a mild drop. Recency helps "
        "somewhat, but the later regime appears difficult largely "
        "independent of model freshness."
    )
else:
    print(
        "\nThe gap scenario did not underperform the no-gap "
        "scenario. The later regime appears simply harder "
        "regardless of how recently the model was trained."
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved gap stress test results to:")
print(results_path)

print("\nGap stress test complete.")
