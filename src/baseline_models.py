# ============================================================
# BASELINE MODEL MODULE
# ============================================================
# This module defines the first forecasting models in the project.
#
# Philosophy:
# Complex models should only be accepted if they outperform simple,
# interpretable alternatives under the same temporal validation.
#
# Baselines answer:
#
# "How much value are we actually getting from model complexity?"


import numpy as np

from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    Ridge,
)


# ============================================================
# ZERO PREDICTION BASELINE
# ============================================================
# Decision:
# Predict zero for every observation.
#
# Why:
# Jane Street's primary metric is weighted zero-mean R².
# By construction, this predictor receives a score of exactly 0.
#
# Any useful forecasting model must therefore beat this baseline.
#
# Tradeoff:
# This model contains no information and cannot exploit any
# structure in the predictors.

def predict_zero(
    number_of_rows: int,
) -> np.ndarray:

    return np.zeros(
        number_of_rows,
        dtype=np.float64,
    )


# ============================================================
# WEIGHTED LINEAR REGRESSION
# ============================================================
# Decision:
# Fit ordinary linear regression while incorporating Jane Street's
# observation weights.
#
# Why:
# This establishes whether the 79 anonymized features contain
# useful approximately linear predictive information.
#
# The model is deliberately simple and highly interpretable.
#
# Tradeoffs:
# - assumes additive linear relationships
# - cannot naturally capture interactions or thresholds
# - can be unstable when predictors are correlated
# - does not protect coefficients from becoming unnecessarily large
#
# Scaling is performed outside this class because preprocessing
# must be fitted exclusively on the training period.

def create_weighted_linear_regression():
    return LinearRegression()


# ============================================================
# RIDGE REGRESSION
# ============================================================
# Decision:
# Add an L2 penalty to linear regression.
#
# Why:
# The Jane Street feature set contains correlated predictors and
# weak individual signals. Ridge shrinks coefficients rather than
# allowing noisy predictors to produce extreme estimates.
#
# Tradeoff:
# Ridge generally retains all features rather than performing
# feature selection.
#
# alpha controls regularization strength:
#
# small alpha -> closer to ordinary linear regression
# large alpha -> stronger coefficient shrinkage

def create_ridge_model(
    alpha: float = 1.0,
):
    return Ridge(
        alpha=alpha,
        random_state=None,
    )


# ============================================================
# ELASTIC NET
# ============================================================
# Decision:
# Combine L1 and L2 regularization.
#
# Why:
# Elastic Net can both shrink coefficients and push some noisy
# coefficients exactly toward zero.
#
# This is useful when many predictors contain only weak signal.
#
# Tradeoffs:
# - slower than Ridge
# - requires tuning both alpha and l1_ratio
# - aggressive regularization may remove weak signals that become
#   useful when combined with others
#
# l1_ratio:
#
# 0 -> behaves like Ridge
# 1 -> behaves like Lasso
#
# We begin with a balanced value of 0.5 and treat it as a baseline,
# not an optimized setting.

def create_elastic_net_model(
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
):
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=3000,
        tol=1e-4,
        selection="cyclic",
    )
