# ============================================================
# IMPORTS
# ============================================================
# This script tests whether the raw LightGBM/cross-sectional edge
# is transferable market structure or largely symbol-identity
# memorization.
#
# symbol_id dominated feature-gain in every prior experiment
# (~229,867 gain vs ~50,314 for the next feature, an ~4.6x gap).
# Dominant importance is not automatically overfitting - this
# ablation is what actually tells us whether it generalizes.
#
# Four variants, same fold, same strides, same LightGBM settings:
#
#   1. raw_with_symbol                  (control)
#   2. raw_without_symbol
#   3. raw_cross_sectional_without_symbol
#   4. raw_cross_sectional_with_symbol  (current best from script 11)
#
# Interpretation:
# - (2) barely worse than (1)          -> edge is mostly transferable
# - (2) much worse, but (3) recovers   -> market context replaces
#   most of it                            identity memorization
# - (3) still far below (1)/(4)        -> symbol-specific structure
#                                          is genuinely central and
#                                          must be treated explicitly

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
    cross_sectional_feature_names,
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
# Identical to scripts 09 and 11, so results remain comparable.

TRAIN_START_DATE = 0
TRAIN_END_DATE = 1298

VALIDATION_START_DATE = 1299
VALIDATION_END_DATE = 1498

TRAIN_TIME_STRIDE = 40
VALIDATION_TIME_STRIDE = 20


# ============================================================
# CANDIDATE FEATURE GROUPS
# ============================================================

RAW_WITHOUT_SYMBOL = (
    FEATURE_COLUMNS
    + [
        TIME_COLUMN,
    ]
)

RAW_WITH_SYMBOL = (
    RAW_WITHOUT_SYMBOL
    + [
        SYMBOL_COLUMN,
    ]
)

CROSS_SECTIONAL_FEATURES = cross_sectional_feature_names()


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
# ADD CROSS-SECTIONAL FEATURES
# ============================================================
# No rolling/lag features this round - script 11 showed they gave
# essentially no benefit while more than doubling training time.

print("\n" + "=" * 80)
print("ADDING CROSS-SECTIONAL FEATURES")
print("=" * 80)

engineering_start = time.time()

training_processed = add_cross_sectional_features(
    training_data
)

validation_processed = add_cross_sectional_features(
    validation_data
)

assert training_processed.height == training_data.height
assert validation_processed.height == validation_data.height

print(
    f"Engineering time: "
    f"{time.time() - engineering_start:.2f} seconds"
)


# ============================================================
# CONVERT TO PANDAS ONCE, THEN SUBSET PER EXPERIMENT
# ============================================================

all_candidate_columns = (
    RAW_WITH_SYMBOL
    + CROSS_SECTIONAL_FEATURES
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
# Cast once on the full frame. Experiments that omit symbol_id
# simply never select this column, so the cast is harmless for
# those runs.

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
        "raw_with_symbol",
        RAW_WITH_SYMBOL,
    ),
    (
        "raw_without_symbol",
        RAW_WITHOUT_SYMBOL,
    ),
    (
        "raw_cross_sectional_without_symbol",
        RAW_WITHOUT_SYMBOL + CROSS_SECTIONAL_FEATURES,
    ),
    (
        "raw_cross_sectional_with_symbol",
        RAW_WITH_SYMBOL + CROSS_SECTIONAL_FEATURES,
    ),
]


# ============================================================
# TRAIN EACH EXPERIMENT
# ============================================================

experiment_results = []

for experiment_name, experiment_features in experiments:

    print("\n" + "=" * 80)
    print(f"TRAINING: {experiment_name}")
    print(f"Feature count: {len(experiment_features)}")
    print(
        f"symbol_id included: "
        f"{SYMBOL_COLUMN in experiment_features}"
    )
    print("=" * 80)

    X_train = X_train_full[experiment_features]
    X_validation = X_validation_full[experiment_features]

    categorical_feature = (
        [SYMBOL_COLUMN]
        if SYMBOL_COLUMN in experiment_features
        else "auto"
    )

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
        categorical_feature=categorical_feature,
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
        "symbol_id_included": SYMBOL_COLUMN in experiment_features,
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
    / "lightgbm_symbol_ablation_results.csv"
)

results_frame.to_csv(
    results_path,
    index=False,
)


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 100)
print("SYMBOL ABLATION COMPARISON")
print("=" * 100)

display_columns = [
    "model",
    "symbol_id_included",
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


# ============================================================
# INTERPRETATION HELPERS
# ============================================================
# Mechanical calculation of the two deltas that answer the research
# question, printed explicitly so the conclusion doesn't require
# manually comparing rows.

results_by_name = {
    row["model"]: row
    for row in experiment_results
}

symbol_removal_cost = (
    results_by_name["raw_with_symbol"]["weighted_zero_mean_r2"]
    - results_by_name["raw_without_symbol"]["weighted_zero_mean_r2"]
)

cross_sectional_recovery = (
    results_by_name["raw_cross_sectional_without_symbol"][
        "weighted_zero_mean_r2"
    ]
    - results_by_name["raw_without_symbol"]["weighted_zero_mean_r2"]
)

recovery_fraction = (
    cross_sectional_recovery / symbol_removal_cost
    if symbol_removal_cost > 0
    else float("nan")
)

remaining_gap_to_best = (
    results_by_name["raw_cross_sectional_with_symbol"][
        "weighted_zero_mean_r2"
    ]
    - results_by_name["raw_cross_sectional_without_symbol"][
        "weighted_zero_mean_r2"
    ]
)

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(
    f"R² cost of removing symbol_id (raw_with - raw_without): "
    f"{symbol_removal_cost:.8f}"
)

print(
    f"R² recovered by adding cross-sectional features without "
    f"symbol_id: {cross_sectional_recovery:.8f}"
)

print(
    f"Fraction of symbol_id's value recovered by cross-sectional "
    f"features: {recovery_fraction:.2%}"
)

print(
    f"Remaining gap: cross_sectional_with_symbol - "
    f"cross_sectional_without_symbol: {remaining_gap_to_best:.8f}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved comparison results to:")
print(results_path)

print("\nSymbol ablation experiment complete.")
