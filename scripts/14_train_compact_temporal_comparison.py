# ============================================================
# IMPORTS
# ============================================================
# This script tests the compact temporal feature set against the
# current best-known control:
#
#     raw + cross-sectional - symbol_id
#     weighted zero-mean R² = 0.016244
#
# Only ONE new model is trained (control + compact temporal),
# not four. The control is not retrained here - its saved score
# from scripts/12_symbol_ablation.py is the fixed benchmark.
#
# H0: compact temporal context adds no value
# H1: R² > 0.016244
#
# Interpretation bands (weighted zero-mean R²):
#   < 0.0160          temporal block hurts; discard
#   0.0160 - 0.0164    essentially noise / no clear benefit
#   0.0165 - 0.0170    small useful improvement
#   0.0170 - 0.0180    meaningful temporal edge
#   > 0.0180           very strong result; temporal context becomes core
#
# IC is tracked alongside R²: a rising R² with collapsing IC would
# call for inspecting calibration rather than accepting the result
# outright.

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
    add_compact_temporal_features,
    add_cross_sectional_features,
    build_compact_temporal_daily_table,
    cross_sectional_feature_names,
    get_compact_temporal_feature_names,
)

from src.metrics import (
    calculate_forecast_metrics,
)

from src.tree_models import (
    create_lightgbm_baseline,
)


# ============================================================
# SAVED BENCHMARK
# ============================================================
# From scripts/12_symbol_ablation.py: raw_cross_sectional_without_symbol.
# We do not retrain the control here; we compare against its saved
# score to avoid burning compute re-deriving an already-established
# number.

BENCHMARK_MODEL_NAME = "raw_cross_sectional_without_symbol"
BENCHMARK_WEIGHTED_ZERO_MEAN_R2 = 0.016244
BENCHMARK_INFORMATION_COEFFICIENT = 0.119612


# ============================================================
# DEVELOPMENT DATE RANGES
# ============================================================
# Identical to every prior LightGBM experiment on this fold.

TRAIN_START_DATE = 0
TRAIN_END_DATE = 1298

VALIDATION_START_DATE = 1299
VALIDATION_END_DATE = 1498

TRAIN_TIME_STRIDE = 40
VALIDATION_TIME_STRIDE = 20


# ============================================================
# FEATURE GROUPS
# ============================================================

RAW_WITHOUT_SYMBOL = (
    FEATURE_COLUMNS
    + [
        TIME_COLUMN,
    ]
)

CROSS_SECTIONAL_FEATURES = cross_sectional_feature_names()
TEMPORAL_FEATURES = get_compact_temporal_feature_names()

CONTROL_FEATURES = (
    RAW_WITHOUT_SYMBOL
    + CROSS_SECTIONAL_FEATURES
)

TEMPORAL_MODEL_FEATURES = (
    CONTROL_FEATURES
    + TEMPORAL_FEATURES
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
# BUILD THE COMPACT TEMPORAL DAILY TABLE
# ============================================================
# Decision:
# Build one table spanning the full combined date range
# (TRAIN_START_DATE -> VALIDATION_END_DATE), same reasoning as the
# earlier rolling-feature experiment: every column here is derived
# from a shifted/previous-day series, so it is leakage-safe
# regardless of whether the "current" observation falls in the
# training or validation window. Building one continuous table lets
# validation dates near VALIDATION_START_DATE correctly see their
# most recent prior trading day without special-casing the
# boundary.

print("\n" + "=" * 80)
print("BUILDING COMPACT TEMPORAL DAILY TABLE")
print("=" * 80)

temporal_build_start = time.time()

history_scan = (
    training_scan
    .filter(
        (pl.col(DATE_COLUMN) >= TRAIN_START_DATE)
        & (pl.col(DATE_COLUMN) <= VALIDATION_END_DATE)
    )
)

daily_temporal_table = build_compact_temporal_daily_table(
    history_scan
)

print(
    f"Build time: "
    f"{time.time() - temporal_build_start:.2f} seconds"
)


# ============================================================
# LOAD + ENGINEER TRAINING AND VALIDATION DATA
# ============================================================
# Decision:
# Chain cross-sectional and compact temporal engineering entirely
# in lazy mode, then collect once per split.

def load_and_engineer(
    start_date: int,
    end_date: int,
    time_stride: int,
) -> pl.DataFrame:

    observation_scan = (
        training_scan
        .filter(
            (pl.col(DATE_COLUMN) >= start_date)
            & (pl.col(DATE_COLUMN) <= end_date)
            & (pl.col(TIME_COLUMN) % time_stride == 0)
        )
        .select(required_columns)
    )

    with_cross_sectional = add_cross_sectional_features(
        observation_scan
    )

    with_temporal = add_compact_temporal_features(
        with_cross_sectional,
        daily_temporal_table,
    )

    full_columns = (
        required_columns
        + CROSS_SECTIONAL_FEATURES
        + TEMPORAL_FEATURES
    )

    return (
        with_temporal
        .select(full_columns)
        .collect()
    )


print("\n" + "=" * 80)
print("LOADING AND ENGINEERING TRAINING DATA")
print("=" * 80)

load_start = time.time()

training_processed = load_and_engineer(
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TRAIN_TIME_STRIDE,
)

print(f"Training rows: {training_processed.height:,}")
print(f"Load + engineer time: {time.time() - load_start:.2f} seconds")

print("\n" + "=" * 80)
print("LOADING AND ENGINEERING VALIDATION DATA")
print("=" * 80)

load_start = time.time()

validation_processed = load_and_engineer(
    VALIDATION_START_DATE,
    VALIDATION_END_DATE,
    VALIDATION_TIME_STRIDE,
)

print(f"Validation rows: {validation_processed.height:,}")
print(f"Load + engineer time: {time.time() - load_start:.2f} seconds")

assert (
    training_processed[DATE_COLUMN].max()
    < validation_processed[DATE_COLUMN].min()
)


# ============================================================
# CONVERT TO PANDAS ONCE
# ============================================================

all_candidate_columns = TEMPORAL_MODEL_FEATURES

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
# TRAIN: CONTROL + COMPACT TEMPORAL FEATURES
# ============================================================
# No symbol_id anywhere in this experiment - the symbol ablation
# result showed it hurts generalization - so no categorical
# handling is required.

print("\n" + "=" * 80)
print("TRAINING: raw_cross_sectional_no_symbol_plus_compact_temporal")
print(f"Feature count: {len(TEMPORAL_MODEL_FEATURES)}")
print("=" * 80)

X_train = X_train_full[TEMPORAL_MODEL_FEATURES]
X_validation = X_validation_full[TEMPORAL_MODEL_FEATURES]

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

print(
    f"Best iteration: {model.best_iteration_}   "
    f"Training time: {training_seconds:.2f}s"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(
    {
        "feature": TEMPORAL_MODEL_FEATURES,
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
# SAVE RESULTS
# ============================================================

result_row = {
    "model": "raw_cross_sectional_no_symbol_plus_compact_temporal",
    "feature_count": len(TEMPORAL_MODEL_FEATURES),
    "best_iteration": model.best_iteration_,
    "training_seconds": training_seconds,
    **metrics,
}

comparison_rows = [
    {
        "model": BENCHMARK_MODEL_NAME,
        "feature_count": len(CONTROL_FEATURES),
        "best_iteration": None,
        "training_seconds": None,
        "weighted_zero_mean_r2": BENCHMARK_WEIGHTED_ZERO_MEAN_R2,
        "information_coefficient": BENCHMARK_INFORMATION_COEFFICIENT,
    },
    result_row,
]

results_frame = pd.DataFrame(comparison_rows)

results_path = (
    METRICS_PATH
    / "lightgbm_compact_temporal_results.csv"
)

results_frame.to_csv(
    results_path,
    index=False,
)

importance_path = (
    METRICS_PATH
    / "lightgbm_compact_temporal_feature_importance.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False,
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n" + "=" * 100)
print("COMPACT TEMPORAL FEATURE COMPARISON")
print("=" * 100)

print(
    f"Benchmark ({BENCHMARK_MODEL_NAME}):"
)
print(
    f"  weighted_zero_mean_r2   = "
    f"{BENCHMARK_WEIGHTED_ZERO_MEAN_R2:.6f}"
)
print(
    f"  information_coefficient = "
    f"{BENCHMARK_INFORMATION_COEFFICIENT:.6f}"
)

print(
    f"\nNew model "
    f"(raw_cross_sectional_no_symbol_plus_compact_temporal):"
)
for metric_name, metric_value in metrics.items():
    print(
        f"  {metric_name:<26} = {metric_value:.8f}"
    )

r2_delta = (
    metrics["weighted_zero_mean_r2"]
    - BENCHMARK_WEIGHTED_ZERO_MEAN_R2
)

ic_delta = (
    metrics["information_coefficient"]
    - BENCHMARK_INFORMATION_COEFFICIENT
)

print(
    f"\nDelta R²: {r2_delta:+.8f}"
)
print(
    f"Delta IC: {ic_delta:+.6f}"
)


# ============================================================
# INTERPRETATION BAND
# ============================================================

new_r2 = metrics["weighted_zero_mean_r2"]

if new_r2 < 0.0160:
    interpretation = (
        "Temporal block HURTS performance relative to the "
        "benchmark; discard."
    )
elif new_r2 < 0.0165:
    interpretation = (
        "Essentially noise / no clear benefit over the benchmark."
    )
elif new_r2 < 0.0170:
    interpretation = (
        "Small useful improvement."
    )
elif new_r2 < 0.0180:
    interpretation = (
        "Meaningful temporal edge."
    )
else:
    interpretation = (
        "Very strong result; temporal context becomes core."
    )

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print(f"New weighted zero-mean R²: {new_r2:.6f}")
print(interpretation)

if ic_delta < 0 and r2_delta > 0:
    print(
        "\nCAUTION: R² improved but IC declined - inspect "
        "calibration rather than accepting this result outright."
    )


# ============================================================
# TOP FEATURES BY GAIN
# ============================================================

print("\n" + "=" * 80)
print("TOP 25 FEATURES BY GAIN")
print("=" * 80)

print(
    feature_importance
    .head(25)
    .to_string(index=False)
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved comparison results to:")
print(results_path)

print("\nSaved feature importance to:")
print(importance_path)

print("\nCompact temporal feature experiment complete.")
