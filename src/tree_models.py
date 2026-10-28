# ============================================================
# TREE-BASED MODEL MODULE
# ============================================================
# This module defines nonlinear tree-based forecasting models.
#
# Why:
# Linear models established that the raw Jane Street predictors
# contain measurable out-of-sample signal.
#
# Tree boosting now tests whether nonlinear relationships,
# thresholds, and feature interactions provide additional
# predictive value.

from lightgbm import LGBMRegressor


# ============================================================
# LIGHTGBM BASELINE
# ============================================================
# Decision:
# LightGBM is our first nonlinear model.
#
# Why:
# - strong performance on large tabular datasets
# - handles missing values natively
# - does not require standardization
# - captures nonlinear relationships
# - captures feature interactions automatically
# - relatively efficient on CPU
#
# Tradeoffs:
# - less interpretable than linear regression
# - more hyperparameters
# - easier to overfit
# - performance can deteriorate under regime change
#
# We intentionally begin with a conservative model rather than
# heavily tuning before establishing a baseline.

def create_lightgbm_baseline(
    random_seed: int = 42,
) -> LGBMRegressor:

    return LGBMRegressor(

        # ----------------------------------------------------
        # OBJECTIVE
        # ----------------------------------------------------
        # Weighted squared error aligns with maximizing Jane
        # Street's weighted zero-mean R² because the metric
        # denominator is constant with respect to predictions.
        objective="regression",

        # ----------------------------------------------------
        # BOOSTING ITERATIONS
        # ----------------------------------------------------
        # We set a relatively large maximum number of trees and
        # rely on early stopping to determine when validation
        # performance stops improving.
        n_estimators=1500,

        # ----------------------------------------------------
        # LEARNING RATE
        # ----------------------------------------------------
        # A modest learning rate usually produces more stable
        # generalization than fitting a small number of aggressive
        # trees.
        learning_rate=0.03,

        # ----------------------------------------------------
        # TREE COMPLEXITY
        # ----------------------------------------------------
        # num_leaves controls the complexity of each tree.
        #
        # 31 is deliberately conservative for the first baseline.
        # We will tune this only after establishing whether raw
        # nonlinear structure is useful.
        num_leaves=31,

        # ----------------------------------------------------
        # MINIMUM LEAF SIZE
        # ----------------------------------------------------
        # Larger leaves reduce the risk of learning extremely
        # specific patterns from noisy financial observations.
        min_child_samples=200,

        # ----------------------------------------------------
        # FEATURE SUBSAMPLING
        # ----------------------------------------------------
        # Each tree sees only 80% of predictors.
        #
        # This can reduce variance and prevent a small group of
        # features from dominating every tree.
        colsample_bytree=0.80,

        # ----------------------------------------------------
        # ROW SUBSAMPLING
        # ----------------------------------------------------
        # Each boosting iteration can use a subset of training
        # observations.
        #
        # This reduces training cost and adds regularization.
        subsample=0.80,
        subsample_freq=1,

        # ----------------------------------------------------
        # REGULARIZATION
        # ----------------------------------------------------
        reg_alpha=0.0,
        reg_lambda=1.0,

        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------
        random_state=random_seed,

        # ----------------------------------------------------
        # CPU EXECUTION
        # ----------------------------------------------------
        # Use the available CPU cores.
        #
        # No GPU is required for this baseline.
        n_jobs=-1,
        verbosity=-1,
    )


# ============================================================
# CONFIGURABLE LIGHTGBM MODEL (FOR HYPERPARAMETER SEARCH)
# ============================================================
# Decision:
# create_lightgbm_baseline() above stays a fixed reference
# configuration - every experiment prior to tuning (raw baseline,
# feature engineering, symbol ablation, temporal CV, training
# window, gap stress) used it unchanged, and other scripts still
# depend on that being a stable, unparameterized function.
#
# This function exposes the five hyperparameters most likely to
# matter (num_leaves, min_child_samples, learning_rate, reg_lambda,
# colsample_bytree) for a small curated search, while every other
# setting (objective, n_estimators, subsample, subsample_freq,
# reg_alpha, n_jobs, verbosity) stays identical to the baseline so
# the search isn't confounded by unrelated changes.

def create_lightgbm_model(
    num_leaves: int = 31,
    min_child_samples: int = 200,
    learning_rate: float = 0.03,
    reg_lambda: float = 1.0,
    colsample_bytree: float = 0.80,
    subsample: float = 0.80,
    n_estimators: int = 1500,
    random_seed: int = 42,
) -> LGBMRegressor:

    return LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        subsample_freq=1,
        reg_alpha=0.0,
        reg_lambda=reg_lambda,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=-1,
    )
