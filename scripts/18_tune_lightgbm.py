# ============================================================
# IMPORTS
# ============================================================
# This script runs a small, curated LightGBM hyperparameter search
# around the locked-in structural specification:
#
#   raw features + cross-sectional features - symbol_id
#   training window: 700 -> 1498
#   validation window: 1499 -> 1698
#
# Why a curated set instead of a grid:
# The full Cartesian product of the five candidate parameters would
# be 3 x 3 x 3 x 2 x 3 = 162 models. On CPU that wastes time and,
# more importantly, risks overfitting hyperparameters to this one
# validation window. Instead we test ~10 configurations, each
# varying ONE dimension away from the current baseline, so any
# improvement can be attributed to a specific, interpretable change.
#
# Acceptance rule (set by the user, not a data-driven threshold):
# A tuned configuration is only considered meaningfully better than
# the baseline if it beats it by at least +0.0003 to +0.0005
# weighted zero-mean R² on this window. Smaller differences are
# treated as noise, not evidence of a better model.

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
    create_lightgbm_model,
)


# ============================================================
# LOCKED STRUCTURAL SPECIFICATION
# ============================================================
# Decision:
# raw features + cross-sectional features, symbol_id excluded,
# 700 -> 1498 training window, 1499 -> 1698 validation.
#
# Why:
# Established across scripts 12 (symbol ablation), 15 (temporal
# CV), 16 (training window comparison), and 17 (gap stress test).
# This script tunes hyperparameters ONLY - the feature set and
# date ranges are fixed.

CROSS_SECTIONAL_BASE_FEATURES = CORE_ENGINEERING_FEATURES

CROSS_SECTIONAL_FEATURES = cross_sectional_feature_names(
    CROSS_SECTIONAL_BASE_FEATURES
)

MODEL_FEATURES = (
    FEATURE_COLUMNS
    + [TIME_COLUMN]
    + CROSS_SECTIONAL_FEATURES
)

TRAIN_START_DATE = 700
TRAIN_END_DATE = 1498

VALIDATION_START_DATE = 1499
VALIDATION_END_DATE = 1698

TRAIN_TIME_STRIDE = 40
VALIDATION_TIME_STRIDE = 20

MEANINGFUL_IMPROVEMENT_THRESHOLD = 0.0003


# ============================================================
# CURATED HYPERPARAMETER CONFIGURATIONS
# ============================================================
# Each configuration (after #1, the baseline) changes exactly one
# dimension away from the baseline, so any score change can be
# attributed to that specific change rather than a confounded
# combination of several simultaneous changes.

HYPERPARAMETER_CONFIGS = [
    {
        "name": "1_baseline",
        "num_leaves": 31,
        "min_child_samples": 200,
        "learning_rate": 0.03,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.80,
    },
    {
        "name": "2_smaller_trees",
        "num_leaves": 15,
        "min_child_samples": 200,
        "learning_rate": 0.03,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.80,
    },
    {
        "name": "3_larger_trees",
        "num_leaves": 63,
        "min_child_samples": 200,
        "learning_rate": 0.03,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.80,
    },
    {
        "name": "4_stronger_leaf_regularization",
        "num_leaves": 31,
        "min_child_samples": 500,
        "learning_rate": 0.03,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.80,
    },
    {
        "name": "5_weaker_leaf_regularization",
        "num_leaves": 31,
        "min_child_samples": 100,
        "learning_rate": 0.03,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.80,
    },
    {
        "name": "6_slower_learning",
        "num_leaves": 31,
        "min_child_samples": 200,
        "learning_rate": 0.02,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.80,
    },
    {
        "name": "7_faster_learning",
        "num_leaves": 31,
        "min_child_samples": 200,
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.80,
    },
    {
        "name": "8_stronger_l2",
        "num_leaves": 31,
        "min_child_samples": 200,
        "learning_rate": 0.03,
        "reg_lambda": 5.0,
        "colsample_bytree": 0.80,
    },
    {
        "name": "9_less_feature_sampling",
        "num_leaves": 31,
        "min_child_samples": 200,
        "learning_rate": 0.03,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.70,
    },
    {
        "name": "10_all_features",
        "num_leaves": 31,
        "min_child_samples": 200,
        "learning_rate": 0.03,
        "reg_lambda": 1.0,
        "colsample_bytree": 1.00,
    },
]


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
# LOAD TRAINING AND VALIDATION DATA ONCE
# ============================================================
# Decision:
# Load the 700->1498 / 1499->1698 split a single time and reuse it
# across all 10 configurations.
#
# Why:
# Only the model hyperparameters change across configurations - the
# underlying data never does. Reloading per configuration would
# waste time and risk the ten runs silently using non-identical
# data if anything upstream changed between loads.

print("\n" + "=" * 90)
print("LOADING TRAINING DATA (700 -> 1498)")
print("=" * 90)

load_start = time.time()

training_data = (
    enriched_scan
    .filter(
        (pl.col(DATE_COLUMN) >= TRAIN_START_DATE)
        & (pl.col(DATE_COLUMN) <= TRAIN_END_DATE)
        & (pl.col(TIME_COLUMN) % TRAIN_TIME_STRIDE == 0)
    )
    .select(required_select_columns)
    .collect()
)

print(f"Training rows: {training_data.height:,}")
print(f"Load time: {time.time() - load_start:.2f} seconds")

print("\n" + "=" * 90)
print("LOADING VALIDATION DATA (1499 -> 1698)")
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

assert (
    training_data[DATE_COLUMN].max()
    < validation_data[DATE_COLUMN].min()
)

X_train = (
    training_data
    .select(MODEL_FEATURES)
    .to_pandas()
)
y_train = training_data[TARGET_COLUMN].to_numpy()
training_weights = training_data[WEIGHT_COLUMN].to_numpy()

X_validation = (
    validation_data
    .select(MODEL_FEATURES)
    .to_pandas()
)
y_validation = validation_data[TARGET_COLUMN].to_numpy()
validation_weights = validation_data[WEIGHT_COLUMN].to_numpy()

del training_data
del validation_data


# ============================================================
# RESULTS CONTAINER
# ============================================================

search_results = []


# ============================================================
# HYPERPARAMETER SEARCH LOOP
# ============================================================

for config in HYPERPARAMETER_CONFIGS:

    print("\n" + "=" * 90)
    print(f"CONFIG: {config['name']}")
    print(
        f"num_leaves={config['num_leaves']}  "
        f"min_child_samples={config['min_child_samples']}  "
        f"learning_rate={config['learning_rate']}  "
        f"reg_lambda={config['reg_lambda']}  "
        f"colsample_bytree={config['colsample_bytree']}"
    )
    print("=" * 90)

    model = create_lightgbm_model(
        num_leaves=config["num_leaves"],
        min_child_samples=config["min_child_samples"],
        learning_rate=config["learning_rate"],
        reg_lambda=config["reg_lambda"],
        colsample_bytree=config["colsample_bytree"],
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
        **config,
        "best_iteration": model.best_iteration_,
        "training_seconds": training_seconds,
        **metrics,
    }

    search_results.append(result_row)

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

    del model
    del predictions


# ============================================================
# COMBINE AND SAVE RESULTS
# ============================================================

results_frame = (
    pd.DataFrame(search_results)
    .sort_values(
        "weighted_zero_mean_r2",
        ascending=False,
    )
    .reset_index(drop=True)
)

results_path = (
    METRICS_PATH
    / "lightgbm_hyperparameter_search_results.csv"
)

results_frame.to_csv(
    results_path,
    index=False,
)


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 130)
print("LIGHTGBM HYPERPARAMETER SEARCH RESULTS")
print("=" * 130)

display_columns = [
    "name",
    "num_leaves",
    "min_child_samples",
    "learning_rate",
    "reg_lambda",
    "colsample_bytree",
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
# INTERPRETATION: MEANINGFUL IMPROVEMENT CHECK
# ============================================================

baseline_row = results_frame[
    results_frame["name"] == "1_baseline"
].iloc[0]

baseline_r2 = baseline_row["weighted_zero_mean_r2"]

results_frame["r2_delta_vs_baseline"] = (
    results_frame["weighted_zero_mean_r2"] - baseline_r2
)

meaningfully_better = results_frame[
    results_frame["r2_delta_vs_baseline"]
    >= MEANINGFUL_IMPROVEMENT_THRESHOLD
].sort_values(
    "weighted_zero_mean_r2",
    ascending=False,
)

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(f"Baseline R² (1_baseline): {baseline_r2:.6f}")
print(
    f"Meaningful improvement threshold: "
    f"+{MEANINGFUL_IMPROVEMENT_THRESHOLD:.4f} R²"
)

if meaningfully_better.empty:
    print(
        "\nNo configuration beat the baseline by the required "
        "margin. The baseline configuration remains the best-"
        "supported choice; differences among the other "
        "configurations should be treated as noise on this single "
        "validation window."
    )
else:
    print(
        f"\n{len(meaningfully_better)} configuration(s) beat the "
        f"baseline by at least "
        f"+{MEANINGFUL_IMPROVEMENT_THRESHOLD:.4f} R²:"
    )
    print(
        meaningfully_better[
            [
                "name",
                "weighted_zero_mean_r2",
                "r2_delta_vs_baseline",
                "information_coefficient",
            ]
        ].to_string(index=False)
    )
    print(
        "\nThese candidates should be re-checked on Fold 3 as well "
        "before being trusted - a win on this single validation "
        "window alone is not sufficient to lock in a tuned "
        "configuration."
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved hyperparameter search results to:")
print(results_path)

print("\nHyperparameter search complete.")
