# ============================================================
# IMPORTS
# ============================================================
# This script validates the current winning LightGBM specification
# across multiple chronological validation periods.
#
# Current winner:
#
# raw Jane Street features
# + cross-sectional market features
# - raw symbol_id
#
# Why:
# A model should not be accepted based on one favorable validation
# window. Financial relationships are nonstationary, so the same
# specification must survive multiple future periods.
#
# Note:
# Cross-sectional feature construction is imported from
# src/feature_engineering.py (add_cross_sectional_features,
# cross_sectional_feature_names, CORE_ENGINEERING_FEATURES) rather
# than reimplemented here. Those functions were already verified in
# scripts/10_test_feature_engineering.py, and this script validates
# the winning MODEL, not the feature-engineering logic - duplicating
# that logic here would risk two implementations silently drifting
# apart.

import sys
import time
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

from src.temporal_cv import (
    DEVELOPMENT_TEMPORAL_FOLDS,
)

from src.tree_models import (
    create_lightgbm_baseline,
)


# ============================================================
# CROSS-SECTIONAL FEATURE BASE COLUMNS
# ============================================================
# Decision:
# Use the exact same 16-feature cross-sectional base set that
# produced the best development-fold result.
#
# Why:
# We are validating the existing winner, not introducing another
# modeling change at the same time. This keeps the experiment
# interpretable: only the historical train/validation period
# changes.
#
# CORE_ENGINEERING_FEATURES in src/feature_engineering.py already
# is this exact 16-feature set (verified in script 10), so we reuse
# it directly rather than retyping the list here.

CROSS_SECTIONAL_BASE_FEATURES = CORE_ENGINEERING_FEATURES

CROSS_SECTIONAL_FEATURES = cross_sectional_feature_names(
    CROSS_SECTIONAL_BASE_FEATURES
)


# ============================================================
# FINAL MODEL FEATURES
# ============================================================
# symbol_id is deliberately excluded.
#
# Ablation testing (scripts/12_symbol_ablation.py) showed that
# removing symbol_id IMPROVED forward-period weighted R².
#
# This is a major modeling decision:
#
# feature importance showed symbol_id was heavily used in-sample,
# but temporal validation showed that dependence hurt
# generalization.

MODEL_FEATURES = (
    FEATURE_COLUMNS
    + [TIME_COLUMN]
    + CROSS_SECTIONAL_FEATURES
)


# ============================================================
# CPU-FRIENDLY SYSTEMATIC SAMPLING
# ============================================================
# Keep exactly the same development sampling policy used in the
# previous LightGBM experiments.
#
# This ensures our new temporal comparison is apples-to-apples.

TRAIN_TIME_STRIDE = 40
VALIDATION_TIME_STRIDE = 20


# ============================================================
# LAZY-SCAN FULL TRAINING DATASET
# ============================================================

training_scan = pl.scan_parquet(
    str(
        TRAIN_PATH
        / "**"
        / "*.parquet"
    )
)


# ============================================================
# CREATE CROSS-SECTIONALLY ENRICHED DATASET
# ============================================================
# Decision:
# Compute current-timestamp market statistics before applying the
# time_id sampling.
#
# Why:
# Market means should represent the actual cross-section available
# at that timestamp. If we first sampled time/symbol observations
# and then calculated market means, we could distort the
# cross-sectional statistic.
#
# add_cross_sectional_features() operates purely with .with_columns()
# window expressions, so it works identically whether given an
# eager DataFrame or, as here, a LazyFrame - the transformation
# stays lazy until .collect() is called per fold below.

enriched_scan = add_cross_sectional_features(
    training_scan,
    features=CROSS_SECTIONAL_BASE_FEATURES,
)


# ============================================================
# RESULTS CONTAINER
# ============================================================

fold_results = []
feature_importance_results = []


# ============================================================
# TEMPORAL CV LOOP
# ============================================================
# DEVELOPMENT_TEMPORAL_FOLDS (src/temporal_cv.py) currently
# contains:
#
# Fold 3:
# Train 0-1298
# Validate 1299-1498
#
# Fold 4:
# Train 0-1498
# Validate 1499-1698
#
# Each fold retrains the model from scratch.
#
# This is critical:
# Fold 4 cannot reuse the Fold 3 model because its training
# information set is different.

for fold in DEVELOPMENT_TEMPORAL_FOLDS:

    print("\n" + "=" * 90)
    print(f"STARTING {fold.name.upper()}")
    print("=" * 90)

    print(
        f"Training dates: "
        f"{fold.train_start_date} -> "
        f"{fold.train_end_date}"
    )
    print(
        f"Validation dates: "
        f"{fold.validation_start_date} -> "
        f"{fold.validation_end_date}"
    )

    # ========================================================
    # LOAD TRAINING FOLD
    # ========================================================

    print("\nLoading training fold...")

    load_start = time.time()

    fold_training_data = (
        enriched_scan
        .filter(
            (pl.col(DATE_COLUMN) >= fold.train_start_date)
            &
            (pl.col(DATE_COLUMN) <= fold.train_end_date)
            &
            (
                pl.col(TIME_COLUMN)
                % TRAIN_TIME_STRIDE
                == 0
            )
        )
        .select(
            MODEL_FEATURES
            + [
                DATE_COLUMN,
                TARGET_COLUMN,
                WEIGHT_COLUMN,
            ]
        )
        .collect()
    )

    print(
        f"Training rows: "
        f"{fold_training_data.height:,}"
    )
    print(
        f"Training load time: "
        f"{time.time() - load_start:.2f} seconds"
    )

    # ========================================================
    # LOAD VALIDATION FOLD
    # ========================================================

    print("\nLoading validation fold...")

    load_start = time.time()

    fold_validation_data = (
        enriched_scan
        .filter(
            (pl.col(DATE_COLUMN) >= fold.validation_start_date)
            &
            (pl.col(DATE_COLUMN) <= fold.validation_end_date)
            &
            (
                pl.col(TIME_COLUMN)
                % VALIDATION_TIME_STRIDE
                == 0
            )
        )
        .select(
            MODEL_FEATURES
            + [
                DATE_COLUMN,
                TARGET_COLUMN,
                WEIGHT_COLUMN,
            ]
        )
        .collect()
    )

    print(
        f"Validation rows: "
        f"{fold_validation_data.height:,}"
    )
    print(
        f"Validation load time: "
        f"{time.time() - load_start:.2f} seconds"
    )

    # ========================================================
    # TEMPORAL INTEGRITY CHECK
    # ========================================================

    assert (
        fold_training_data[
            DATE_COLUMN
        ].max()
        <
        fold_validation_data[
            DATE_COLUMN
        ].min()
    )

    # ========================================================
    # BUILD NUMPY / PANDAS MATRICES
    # ========================================================
    # LightGBM does not require standardization.
    #
    # Native missing values are preserved.

    X_train = (
        fold_training_data
        .select(MODEL_FEATURES)
        .to_pandas()
    )
    y_train = (
        fold_training_data[
            TARGET_COLUMN
        ]
        .to_numpy()
    )
    train_weights = (
        fold_training_data[
            WEIGHT_COLUMN
        ]
        .to_numpy()
    )

    X_validation = (
        fold_validation_data
        .select(MODEL_FEATURES)
        .to_pandas()
    )
    y_validation = (
        fold_validation_data[
            TARGET_COLUMN
        ]
        .to_numpy()
    )
    validation_weights = (
        fold_validation_data[
            WEIGHT_COLUMN
        ]
        .to_numpy()
    )

    # ========================================================
    # FREE POLARS FRAMES
    # ========================================================
    # Mechanical memory-management step.

    del fold_training_data
    del fold_validation_data

    # ========================================================
    # CREATE FRESH LIGHTGBM MODEL
    # ========================================================

    model = create_lightgbm_baseline(
        random_seed=RANDOM_SEED,
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print("\nTraining LightGBM...")

    model_start = time.time()

    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
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

    training_seconds = (
        time.time()
        - model_start
    )

    # ========================================================
    # GENERATE PREDICTIONS
    # ========================================================

    predictions = model.predict(
        X_validation,
        num_iteration=model.best_iteration_,
    )

    # ========================================================
    # CALCULATE STANDARD FORECAST METRICS
    # ========================================================

    metrics = calculate_forecast_metrics(
        y_validation,
        predictions,
        validation_weights,
    )

    # ========================================================
    # STORE FOLD RESULTS
    # ========================================================

    fold_result = {
        "fold":
            fold.name,
        "train_start_date":
            fold.train_start_date,
        "train_end_date":
            fold.train_end_date,
        "validation_start_date":
            fold.validation_start_date,
        "validation_end_date":
            fold.validation_end_date,
        "training_rows":
            len(y_train),
        "validation_rows":
            len(y_validation),
        "best_iteration":
            model.best_iteration_,
        "training_seconds":
            training_seconds,
        **metrics,
    }

    fold_results.append(
        fold_result
    )

    # ========================================================
    # SAVE FOLD FEATURE IMPORTANCE
    # ========================================================
    # We store importance separately for each fold.
    #
    # Why:
    # A feature that appears important in only one historical
    # regime may not be reliable.

    fold_importance = pd.DataFrame(
        {
            "fold":
                fold.name,
            "feature":
                MODEL_FEATURES,
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

    feature_importance_results.append(
        fold_importance
    )

    # ========================================================
    # PRINT FOLD RESULT
    # ========================================================

    print("\n" + "-" * 90)
    print(f"{fold.name.upper()} RESULT")
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
    print(
        f"Best iteration: "
        f"{model.best_iteration_}"
    )
    print(
        f"Training seconds: "
        f"{training_seconds:.2f}"
    )

    # ========================================================
    # FREE MODEL MATRICES
    # ========================================================

    del X_train
    del X_validation
    del y_train
    del y_validation
    del train_weights
    del validation_weights
    del predictions
    del model


# ============================================================
# COMBINE TEMPORAL CV RESULTS
# ============================================================

results_frame = pd.DataFrame(
    fold_results
)


# ============================================================
# CALCULATE CROSS-FOLD SUMMARY
# ============================================================
# Decision:
# Report both mean performance and dispersion across folds.
#
# Why:
# A strong mean with extreme fold instability is less convincing
# than a slightly lower but stable score.

summary = {
    "number_of_folds":
        len(results_frame),
    "mean_weighted_zero_mean_r2":
        results_frame[
            "weighted_zero_mean_r2"
        ].mean(),
    "std_weighted_zero_mean_r2":
        results_frame[
            "weighted_zero_mean_r2"
        ].std(ddof=1),
    "minimum_weighted_zero_mean_r2":
        results_frame[
            "weighted_zero_mean_r2"
        ].min(),
    "maximum_weighted_zero_mean_r2":
        results_frame[
            "weighted_zero_mean_r2"
        ].max(),
    "mean_information_coefficient":
        results_frame[
            "information_coefficient"
        ].mean(),
    "std_information_coefficient":
        results_frame[
            "information_coefficient"
        ].std(ddof=1),
}

summary_frame = pd.DataFrame(
    [summary]
)


# ============================================================
# COMBINE FEATURE IMPORTANCE
# ============================================================

importance_frame = pd.concat(
    feature_importance_results,
    ignore_index=True,
)


# ============================================================
# FEATURE IMPORTANCE STABILITY
# ============================================================
# Aggregate gain importance across folds.
#
# We calculate mean and standard deviation to see whether a
# predictor remains important through different market regimes.

importance_stability = (
    importance_frame
    .groupby(
        "feature",
        as_index=False,
    )
    .agg(
        mean_gain=(
            "importance_gain",
            "mean",
        ),
        std_gain=(
            "importance_gain",
            "std",
        ),
        mean_splits=(
            "importance_split",
            "mean",
        ),
    )
    .sort_values(
        "mean_gain",
        ascending=False,
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_frame.to_csv(
    METRICS_PATH
    / "winner_temporal_cv_results.csv",
    index=False,
)

summary_frame.to_csv(
    METRICS_PATH
    / "winner_temporal_cv_summary.csv",
    index=False,
)

importance_frame.to_csv(
    METRICS_PATH
    / "winner_temporal_cv_feature_importance_by_fold.csv",
    index=False,
)

importance_stability.to_csv(
    METRICS_PATH
    / "winner_feature_importance_stability.csv",
    index=False,
)


# ============================================================
# PRINT FINAL TEMPORAL CV TABLE
# ============================================================

print("\n" + "=" * 100)
print("WINNING MODEL TEMPORAL CROSS-VALIDATION")
print("=" * 100)

print(
    results_frame[
        [
            "fold",
            "train_start_date",
            "train_end_date",
            "validation_start_date",
            "validation_end_date",
            "weighted_zero_mean_r2",
            "information_coefficient",
            "directional_accuracy",
            "best_iteration",
            "training_seconds",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("CROSS-FOLD SUMMARY")
print("=" * 100)

for key, value in summary.items():
    if isinstance(
        value,
        (float, np.floating),
    ):
        print(
            f"{key:<40} "
            f"{value:.8f}"
        )
    else:
        print(
            f"{key:<40} "
            f"{value}"
        )


# ============================================================
# PRINT STABLE FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 100)
print("TOP FEATURES ACROSS TEMPORAL FOLDS")
print("=" * 100)

print(
    importance_stability
    .head(25)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved temporal CV results to:")
print(
    METRICS_PATH
    / "winner_temporal_cv_results.csv"
)

print("\nTemporal validation complete.")
