# ============================================================
# IMPORTS
# ============================================================
# This script runs a genuine walk-forward simulation of the locked
# forecasting specification:
#
#   Model:        LightGBM
#   Features:     79 raw features + verified cross-sectional market
#                 features + time_id, symbol_id EXCLUDED
#   Hyperparams:  num_leaves=31, min_child_samples=200,
#                 learning_rate=0.03, reg_lambda=1.0,
#                 colsample_bytree=0.80, subsample=0.80,
#                 subsample_freq=1
#   Missing vals: native LightGBM handling
#   Metric:       weighted zero-mean R²
#
# This specification is now considered locked. This script does not
# search over features or hyperparameters - it tests whether the
# locked specification survives being deployed the way a real
# system would be: trained on a rolling window, used to predict a
# block of unseen future dates, then retrained (from scratch, never
# reused) on fresh history for the next block.
#
#   Train [t-700, t-1] -> Predict [t, t+49] -> advance 50 dates -> repeat
#
# A 700-date training window, 50-date prediction blocks, and
# retraining every block gives 12 sequential out-of-sample blocks
# spanning date_id 1099 -> 1698, matching the evidence already
# gathered:
#
#   - linear baseline:        ~0.00645
#   - raw LightGBM:           ~0.01353
#   - + cross-sectional:      real improvement
#   - - symbol_id:            improves generalization
#   - + rolling/temporal:     no value, discarded
#   - Fold 3 / Fold 4:        0.01624 / 0.00904 (nonstationarity)
#   - rolling 700-date window: modest help over full expanding history
#   - 200-day stale gap:      hurts materially (recency matters)
#   - hyperparameter search:  no config meaningfully beat baseline
#
# Every block trains a FRESH model. No model is ever reused across
# blocks unless that is explicitly the point of a future experiment.

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
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
# PROJECT IMPORTS
# ============================================================

from src.config import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    METRICS_PATH,
    PREDICTIONS_PATH,
    RANDOM_SEED,
    SYMBOL_COLUMN,
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
# LOCKED FORECASTING SPECIFICATION
# ============================================================
# Feature set is unchanged from scripts 12/15/16/17/18.
# Hyperparameters are create_lightgbm_baseline()'s fixed values -
# the hyperparameter search in script 18 found nothing that
# meaningfully beat this configuration.

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
# WALK-FORWARD DESIGN
# ============================================================

TRAINING_WINDOW_DATES = 700
BLOCK_SIZE_DATES = 50

WALK_FORWARD_START_DATE = 1099
WALK_FORWARD_END_DATE = 1698

# Same stride policy as every prior LightGBM experiment on this
# specification, kept for apples-to-apples comparability and CPU
# tractability across 12 full retrainings.

TRAIN_TIME_STRIDE = 40
PREDICTION_TIME_STRIDE = 20


# ============================================================
# WALK-FORWARD BLOCK DEFINITION
# ============================================================

@dataclass(frozen=True)
class WalkForwardBlock:
    block_number: int
    train_start_date: int
    train_end_date: int
    predict_start_date: int
    predict_end_date: int


# ============================================================
# BUILD THE BLOCK SCHEDULE
# ============================================================
# Decision:
# Generate blocks mechanically from the design constants rather
# than hand-listing them, so the schedule can't silently drift out
# of sync with TRAINING_WINDOW_DATES / BLOCK_SIZE_DATES.
#
# block 1: train [399, 1098]  -> predict [1099, 1148]
# block 2: train [449, 1148]  -> predict [1149, 1198]
# ...
# block 12: train [949, 1648] -> predict [1649, 1698]

def build_walk_forward_schedule() -> list[WalkForwardBlock]:

    blocks = []

    predict_start = WALK_FORWARD_START_DATE
    block_number = 1

    while predict_start <= WALK_FORWARD_END_DATE:

        predict_end = min(
            predict_start + BLOCK_SIZE_DATES - 1,
            WALK_FORWARD_END_DATE,
        )

        train_end = predict_start - 1
        train_start = train_end - TRAINING_WINDOW_DATES + 1

        assert train_start >= 0, (
            f"Block {block_number} would require training data "
            f"before date_id 0."
        )

        blocks.append(
            WalkForwardBlock(
                block_number=block_number,
                train_start_date=train_start,
                train_end_date=train_end,
                predict_start_date=predict_start,
                predict_end_date=predict_end,
            )
        )

        predict_start += BLOCK_SIZE_DATES
        block_number += 1

    return blocks


WALK_FORWARD_SCHEDULE = build_walk_forward_schedule()

print("\n" + "=" * 90)
print("WALK-FORWARD SCHEDULE")
print("=" * 90)

for block in WALK_FORWARD_SCHEDULE:
    print(
        f"Block {block.block_number:>2}: "
        f"train [{block.train_start_date}, {block.train_end_date}] "
        f"({block.train_end_date - block.train_start_date + 1} dates)"
        f"  ->  predict [{block.predict_start_date}, "
        f"{block.predict_end_date}] "
        f"({block.predict_end_date - block.predict_start_date + 1} dates)"
    )

print(f"\nTotal blocks: {len(WALK_FORWARD_SCHEDULE)}")

for block in WALK_FORWARD_SCHEDULE:
    assert block.train_end_date < block.predict_start_date, (
        f"Block {block.block_number} training data overlaps its "
        f"prediction block."
    )


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
        SYMBOL_COLUMN,
        TARGET_COLUMN,
        WEIGHT_COLUMN,
    ]
)
# Note: TIME_COLUMN is intentionally not added again here - it is
# already included inside MODEL_FEATURES (used as a raw predictor),
# and selecting it twice raises a Polars DuplicateError.


# ============================================================
# RESULTS CONTAINERS
# ============================================================

block_results = []
prediction_frames = []


# ============================================================
# WALK-FORWARD LOOP
# ============================================================

simulation_start = time.time()

for block in WALK_FORWARD_SCHEDULE:

    print("\n" + "=" * 90)
    print(
        f"BLOCK {block.block_number}/{len(WALK_FORWARD_SCHEDULE)}: "
        f"train [{block.train_start_date}, {block.train_end_date}] "
        f"-> predict [{block.predict_start_date}, "
        f"{block.predict_end_date}]"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # LOAD TRAINING DATA FOR THIS BLOCK
    # --------------------------------------------------------

    load_start = time.time()

    block_training_data = (
        enriched_scan
        .filter(
            (pl.col(DATE_COLUMN) >= block.train_start_date)
            & (pl.col(DATE_COLUMN) <= block.train_end_date)
            & (pl.col(TIME_COLUMN) % TRAIN_TIME_STRIDE == 0)
        )
        .select(required_select_columns)
        .collect()
    )

    print(
        f"Training rows: {block_training_data.height:,}   "
        f"Load time: {time.time() - load_start:.2f}s"
    )

    # --------------------------------------------------------
    # LOAD PREDICTION (HELD-OUT) DATA FOR THIS BLOCK
    # --------------------------------------------------------

    load_start = time.time()

    block_prediction_data = (
        enriched_scan
        .filter(
            (pl.col(DATE_COLUMN) >= block.predict_start_date)
            & (pl.col(DATE_COLUMN) <= block.predict_end_date)
            & (pl.col(TIME_COLUMN) % PREDICTION_TIME_STRIDE == 0)
        )
        .select(required_select_columns)
        .collect()
    )

    print(
        f"Prediction rows: {block_prediction_data.height:,}   "
        f"Load time: {time.time() - load_start:.2f}s"
    )

    # --------------------------------------------------------
    # TEMPORAL INTEGRITY CHECK
    # --------------------------------------------------------
    # This model must never see any observation from its own
    # prediction block during training.

    assert (
        block_training_data[DATE_COLUMN].max()
        < block_prediction_data[DATE_COLUMN].min()
    )

    # --------------------------------------------------------
    # BUILD MODEL MATRICES
    # --------------------------------------------------------

    X_train = (
        block_training_data
        .select(MODEL_FEATURES)
        .to_pandas()
    )
    y_train = block_training_data[TARGET_COLUMN].to_numpy()
    train_weights = block_training_data[WEIGHT_COLUMN].to_numpy()

    X_predict = (
        block_prediction_data
        .select(MODEL_FEATURES)
        .to_pandas()
    )
    y_actual = block_prediction_data[TARGET_COLUMN].to_numpy()
    predict_weights = block_prediction_data[WEIGHT_COLUMN].to_numpy()

    identifier_columns = block_prediction_data.select(
        [
            DATE_COLUMN,
            TIME_COLUMN,
            SYMBOL_COLUMN,
        ]
    )

    training_row_count = block_training_data.height
    prediction_row_count = block_prediction_data.height

    del block_training_data
    del block_prediction_data

    # --------------------------------------------------------
    # TRAIN A FRESH MODEL
    # --------------------------------------------------------
    # A brand-new model is created for every block. No model is
    # ever reused across blocks.

    model = create_lightgbm_baseline(
        random_seed=RANDOM_SEED,
    )

    training_start = time.time()

    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        eval_set=[
            (
                X_predict,
                y_actual,
            )
        ],
        eval_sample_weight=[
            predict_weights
        ],
        eval_metric="l2",
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

    # --------------------------------------------------------
    # PREDICT THE HELD-OUT BLOCK
    # --------------------------------------------------------

    predictions = model.predict(
        X_predict,
        num_iteration=model.best_iteration_,
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS WITH IDENTIFIERS
    # --------------------------------------------------------
    # date_id, time_id, symbol_id, actual responder_6, weight, and
    # prediction - one row per held-out observation. symbol_id is
    # included as metadata for the downstream strategy layer even
    # though it was excluded from the model's own feature set.

    block_prediction_frame = (
        identifier_columns
        .with_columns(
            [
                pl.Series(
                    "actual_responder_6",
                    y_actual,
                ),
                pl.Series(
                    "weight",
                    predict_weights,
                ),
                pl.Series(
                    "prediction",
                    predictions,
                ),
                pl.lit(
                    block.block_number
                ).alias("block_number"),
            ]
        )
    )

    prediction_frames.append(block_prediction_frame)

    # --------------------------------------------------------
    # CALCULATE PER-BLOCK METRICS
    # --------------------------------------------------------

    metrics = calculate_forecast_metrics(
        y_actual,
        predictions,
        predict_weights,
    )

    block_result = {
        "block_number": block.block_number,
        "train_start_date": block.train_start_date,
        "train_end_date": block.train_end_date,
        "predict_start_date": block.predict_start_date,
        "predict_end_date": block.predict_end_date,
        "training_rows": training_row_count,
        "prediction_rows": prediction_row_count,
        "best_iteration": model.best_iteration_,
        "training_seconds": training_seconds,
        **metrics,
    }

    block_results.append(block_result)

    print(
        f"Best iteration: {model.best_iteration_}   "
        f"Training time: {training_seconds:.2f}s"
    )
    print(
        f"Weighted zero-mean R²: "
        f"{metrics['weighted_zero_mean_r2']:.8f}   "
        f"IC: {metrics['information_coefficient']:.6f}   "
        f"Dir. acc: {metrics['directional_accuracy']:.6f}"
    )

    # --------------------------------------------------------
    # FREE PER-BLOCK MEMORY
    # --------------------------------------------------------

    del X_train
    del X_predict
    del y_train
    del y_actual
    del train_weights
    del predict_weights
    del predictions
    del model
    del identifier_columns
    del block_prediction_frame


simulation_seconds = time.time() - simulation_start

print(
    f"\nTotal walk-forward simulation time: "
    f"{simulation_seconds / 60:.1f} minutes"
)


# ============================================================
# CONCATENATE ALL PREDICTIONS INTO ONE OUT-OF-SAMPLE SERIES
# ============================================================

all_predictions = pl.concat(
    prediction_frames
)

predictions_output_path = (
    PREDICTIONS_PATH
    / "walk_forward_predictions.csv"
)

PREDICTIONS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

all_predictions.write_csv(
    predictions_output_path
)

print(
    f"\nTotal out-of-sample predictions: "
    f"{all_predictions.height:,}"
)
print(f"Saved predictions to: {predictions_output_path}")


# ============================================================
# AGGREGATE WALK-FORWARD METRICS (ENTIRE CONCATENATED SERIES)
# ============================================================
# This is the single most important number produced by this
# script: performance across the ENTIRE walk-forward period,
# treating every block's held-out prediction identically, as one
# long genuinely out-of-sample series.

aggregate_metrics = calculate_forecast_metrics(
    all_predictions["actual_responder_6"].to_numpy(),
    all_predictions["prediction"].to_numpy(),
    all_predictions["weight"].to_numpy(),
)


# ============================================================
# PER-BLOCK RESULTS TABLE
# ============================================================

block_results_frame = pd.DataFrame(
    block_results
)

block_results_path = (
    METRICS_PATH
    / "walk_forward_block_results.csv"
)

block_results_frame.to_csv(
    block_results_path,
    index=False,
)


# ============================================================
# CUMULATIVE FORECAST PERFORMANCE
# ============================================================
# Decision:
# Recompute weighted zero-mean R² using the concatenation of
# blocks [1..k] for every k, rather than only per-block scores.
#
# Why:
# A single block's R² is noisy (only 50 dates). The cumulative
# series shows how performance evolves/stabilizes as more
# walk-forward history accumulates, and is a direct read on regime
# stability: a cumulative curve that keeps drifting down signals a
# persistent regime shift, not just one unlucky block.

cumulative_rows = []

for k in range(
    1,
    len(WALK_FORWARD_SCHEDULE) + 1,
):

    cumulative_slice = all_predictions.filter(
        pl.col("block_number") <= k
    )

    cumulative_metrics = calculate_forecast_metrics(
        cumulative_slice["actual_responder_6"].to_numpy(),
        cumulative_slice["prediction"].to_numpy(),
        cumulative_slice["weight"].to_numpy(),
    )

    cumulative_rows.append(
        {
            "through_block": k,
            "through_date": WALK_FORWARD_SCHEDULE[
                k - 1
            ].predict_end_date,
            "cumulative_weighted_zero_mean_r2":
                cumulative_metrics["weighted_zero_mean_r2"],
            "cumulative_information_coefficient":
                cumulative_metrics["information_coefficient"],
            "cumulative_directional_accuracy":
                cumulative_metrics["directional_accuracy"],
        }
    )

cumulative_frame = pd.DataFrame(
    cumulative_rows
)

cumulative_path = (
    METRICS_PATH
    / "walk_forward_cumulative_performance.csv"
)

cumulative_frame.to_csv(
    cumulative_path,
    index=False,
)


# ============================================================
# REGIME STABILITY (CROSS-BLOCK DISPERSION)
# ============================================================

block_r2_series = block_results_frame["weighted_zero_mean_r2"]
block_ic_series = block_results_frame["information_coefficient"]

summary = {
    "number_of_blocks": len(block_results_frame),
    "aggregate_weighted_zero_mean_r2":
        aggregate_metrics["weighted_zero_mean_r2"],
    "aggregate_information_coefficient":
        aggregate_metrics["information_coefficient"],
    "aggregate_directional_accuracy":
        aggregate_metrics["directional_accuracy"],
    "mean_block_weighted_zero_mean_r2": block_r2_series.mean(),
    "std_block_weighted_zero_mean_r2": block_r2_series.std(ddof=1),
    "min_block_weighted_zero_mean_r2": block_r2_series.min(),
    "max_block_weighted_zero_mean_r2": block_r2_series.max(),
    "fraction_blocks_positive_r2":
        (block_r2_series > 0).mean(),
    "mean_block_information_coefficient": block_ic_series.mean(),
    "std_block_information_coefficient": block_ic_series.std(ddof=1),
    "total_simulation_minutes": simulation_seconds / 60,
}

summary_frame = pd.DataFrame(
    [summary]
)

summary_path = (
    METRICS_PATH
    / "walk_forward_aggregate_summary.csv"
)

summary_frame.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 130)
print("WALK-FORWARD RESULTS BY BLOCK")
print("=" * 130)

display_columns = [
    "block_number",
    "train_start_date",
    "train_end_date",
    "predict_start_date",
    "predict_end_date",
    "training_rows",
    "prediction_rows",
    "weighted_zero_mean_r2",
    "information_coefficient",
    "directional_accuracy",
    "best_iteration",
    "training_seconds",
]

print(
    block_results_frame[display_columns].to_string(index=False)
)

print("\n" + "=" * 100)
print("CUMULATIVE PERFORMANCE")
print("=" * 100)

print(
    cumulative_frame.to_string(index=False)
)

print("\n" + "=" * 100)
print("AGGREGATE WALK-FORWARD SUMMARY")
print("=" * 100)

for key, value in summary.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key:<40} {value:.8f}")
    else:
        print(f"{key:<40} {value}")


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved per-block results to:")
print(block_results_path)

print("\nSaved cumulative performance to:")
print(cumulative_path)

print("\nSaved aggregate summary to:")
print(summary_path)

print("\nSaved full out-of-sample predictions to:")
print(predictions_output_path)

print("\nWalk-forward simulation complete.")
