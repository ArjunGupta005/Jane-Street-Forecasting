# ============================================================
# METRICS MODULE
# ============================================================
# This module centralizes forecasting and strategy evaluation.
#
# Why:
# Every model must be evaluated with exactly the same metric
# definitions. Keeping metrics in one module prevents subtle
# inconsistencies between baseline, boosting, neural-network,
# and walk-forward experiments.


import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# INPUT VALIDATION
# ============================================================
# Mechanical safety check:
# Targets, predictions, and weights must have equal length.

def validate_metric_inputs(
    y_true,
    y_pred,
    weights=None,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)

    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have equal length."
        )

    if weights is not None:

        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        if weights.ndim != 1:
            weights = weights.reshape(-1)

        if len(weights) != len(y_true):
            raise ValueError(
                "weights must have the same length as y_true."
            )

        if np.any(weights < 0):
            raise ValueError(
                "weights cannot contain negative values."
            )

    if not np.all(np.isfinite(y_true)):
        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "y_pred contains NaN or infinite values."
        )

    if weights is not None:

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "weights contain NaN or infinite values."
            )

    return y_true, y_pred, weights


# ============================================================
# WEIGHTED ZERO-MEAN R-SQUARED
# ============================================================
# This is the primary Jane Street forecasting metric.
#
# IMPORTANT:
# This differs from ordinary R².
#
# Ordinary R² compares the model against predicting the mean of
# the validation target.
#
# Jane Street's zero-mean R² compares the model against predicting
# zero for every observation.
#
# Formula:
#
#                 Σ w_i (y_i - yhat_i)^2
# R²_zero = 1 - ---------------------------
#                 Σ w_i y_i^2
#
# Interpretation:
#
# > 0 : model beats predicting zero
# = 0 : model performs exactly like zero prediction
# < 0 : model is worse than predicting zero
#
# In this low-signal financial problem, even apparently small
# positive values can represent meaningful predictive information.

def weighted_zero_mean_r2(
    y_true,
    y_pred,
    weights,
) -> float:

    y_true, y_pred, weights = validate_metric_inputs(
        y_true,
        y_pred,
        weights,
    )

    numerator = np.sum(
        weights
        * np.square(
            y_true - y_pred
        )
    )

    denominator = np.sum(
        weights
        * np.square(y_true)
    )

    if denominator <= 0:
        raise ValueError(
            "Weighted zero-mean R² denominator must be positive."
        )

    return float(
        1.0
        - numerator / denominator
    )


# ============================================================
# WEIGHTED MEAN SQUARED ERROR
# ============================================================
# Decision:
# Report weighted MSE in addition to Jane Street's primary metric.
#
# Why:
# R² is useful for relative forecasting performance, while MSE
# provides the raw magnitude of squared prediction error.

def weighted_mse(
    y_true,
    y_pred,
    weights,
) -> float:

    y_true, y_pred, weights = validate_metric_inputs(
        y_true,
        y_pred,
        weights,
    )

    weight_sum = np.sum(weights)

    if weight_sum <= 0:
        raise ValueError(
            "Sum of weights must be positive."
        )

    return float(
        np.sum(
            weights
            * np.square(
                y_true - y_pred
            )
        )
        / weight_sum
    )


# ============================================================
# WEIGHTED ROOT MEAN SQUARED ERROR
# ============================================================

def weighted_rmse(
    y_true,
    y_pred,
    weights,
) -> float:

    return float(
        np.sqrt(
            weighted_mse(
                y_true,
                y_pred,
                weights,
            )
        )
    )


# ============================================================
# WEIGHTED MEAN ABSOLUTE ERROR
# ============================================================
# MAE is less sensitive to extreme prediction errors than MSE.
#
# This is useful because responder_6 is heavy-tailed and clipped
# at -5 and +5.

def weighted_mae(
    y_true,
    y_pred,
    weights,
) -> float:

    y_true, y_pred, weights = validate_metric_inputs(
        y_true,
        y_pred,
        weights,
    )

    weight_sum = np.sum(weights)

    if weight_sum <= 0:
        raise ValueError(
            "Sum of weights must be positive."
        )

    return float(
        np.sum(
            weights
            * np.abs(
                y_true - y_pred
            )
        )
        / weight_sum
    )


# ============================================================
# INFORMATION COEFFICIENT
# ============================================================
# Decision:
# Track Pearson correlation between predictions and realized
# responder_6.
#
# In quantitative finance this is commonly called an information
# coefficient, or IC.
#
# Why:
# IC answers a slightly different question than R²:
#
# "Do larger predictions tend to correspond to larger realized
# outcomes?"
#
# A model can have useful ranking ability even if its prediction
# magnitudes are poorly calibrated.
#
# Tradeoff:
# Pearson IC measures linear association and can be influenced by
# outliers. Later we can optionally add rank IC (Spearman).

def information_coefficient(
    y_true,
    y_pred,
) -> float:

    y_true, y_pred, _ = validate_metric_inputs(
        y_true,
        y_pred,
    )

    if np.std(y_true) == 0:
        return np.nan

    if np.std(y_pred) == 0:
        return np.nan

    return float(
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )


# ============================================================
# DIRECTIONAL ACCURACY
# ============================================================
# Decision:
# Measure how often the prediction has the same sign as the
# realized target.
#
# Why:
# This is intuitive for trading interpretation.
#
# Tradeoff:
# Directional accuracy ignores magnitude entirely. A tiny correct
# prediction and a huge correct prediction count equally.
#
# Therefore it is a diagnostic, not a model-selection objective.

def directional_accuracy(
    y_true,
    y_pred,
) -> float:

    y_true, y_pred, _ = validate_metric_inputs(
        y_true,
        y_pred,
    )

    correct_direction = (
        np.sign(y_true)
        == np.sign(y_pred)
    )

    return float(
        np.mean(correct_direction)
    )


# ============================================================
# COMPLETE FORECAST METRIC REPORT
# ============================================================
# This function gives every model the same standard evaluation
# report.

def calculate_forecast_metrics(
    y_true,
    y_pred,
    weights,
) -> dict[str, float]:

    y_true, y_pred, weights = validate_metric_inputs(
        y_true,
        y_pred,
        weights,
    )

    metrics = {
        "weighted_zero_mean_r2":
            weighted_zero_mean_r2(
                y_true,
                y_pred,
                weights,
            ),

        "weighted_mse":
            weighted_mse(
                y_true,
                y_pred,
                weights,
            ),

        "weighted_rmse":
            weighted_rmse(
                y_true,
                y_pred,
                weights,
            ),

        "weighted_mae":
            weighted_mae(
                y_true,
                y_pred,
                weights,
            ),

        "ordinary_r2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),

        "ordinary_mse":
            float(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            ),

        "ordinary_rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),

        "ordinary_mae":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "information_coefficient":
            information_coefficient(
                y_true,
                y_pred,
            ),

        "directional_accuracy":
            directional_accuracy(
                y_true,
                y_pred,
            ),
    }

    return metrics


# ============================================================
# DAILY STRATEGY RETURN AGGREGATION
# ============================================================
# This begins the strategy/backtesting side of the project.
#
# IMPORTANT:
# A forecast does not itself have a Sharpe ratio.
#
# We first need:
#
# prediction -> position -> realized strategy return
#
# This function assumes that a position has already been created
# for each observation and aggregates observation-level P&L into
# one return per date.
#
# Initial P&L definition:
#
#     weighted position * realized responder
#
# We normalize by total absolute weighted exposure for that date.
#
# Why:
# Normalization reduces changes in P&L caused merely by there being
# more symbols/rows on later dates.
#
# Tradeoff:
# responder_6 is an anonymized response, not a literal tradable
# percentage return. Therefore these strategy returns should be
# interpreted as a backtest proxy rather than realized dollar P&L.

def calculate_daily_strategy_returns(
    dataframe: pd.DataFrame,
    date_column: str = "date_id",
    target_column: str = "responder_6",
    position_column: str = "position",
    weight_column: str = "weight",
) -> pd.DataFrame:

    required_columns = {
        date_column,
        target_column,
        position_column,
        weight_column,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing strategy columns: {missing_columns}"
        )

    strategy_data = dataframe[
        [
            date_column,
            target_column,
            position_column,
            weight_column,
        ]
    ].copy()

    # --------------------------------------------------------
    # OBSERVATION-LEVEL P&L
    # --------------------------------------------------------

    strategy_data["weighted_pnl"] = (
        strategy_data[position_column]
        * strategy_data[target_column]
        * strategy_data[weight_column]
    )

    strategy_data["absolute_weighted_exposure"] = (
        np.abs(
            strategy_data[position_column]
        )
        * strategy_data[weight_column]
    )

    # --------------------------------------------------------
    # DAILY AGGREGATION
    # --------------------------------------------------------

    daily_data = (
        strategy_data
        .groupby(
            date_column,
            as_index=False,
        )
        .agg(
            total_pnl=(
                "weighted_pnl",
                "sum",
            ),
            total_exposure=(
                "absolute_weighted_exposure",
                "sum",
            ),
        )
    )

    daily_data["strategy_return"] = np.where(
        daily_data["total_exposure"] > 0,
        daily_data["total_pnl"]
        / daily_data["total_exposure"],
        0.0,
    )

    return daily_data


# ============================================================
# ANNUALIZED SHARPE RATIO
# ============================================================
# Decision:
# Calculate Sharpe using date-level strategy returns and an
# annualization factor of sqrt(252).
#
# Why:
# time_id intervals are not guaranteed to represent equal elapsed
# time. Using an intraday annualization assumption would therefore
# be difficult to defend.
#
# Aggregating to date_id first gives us a cleaner daily unit.
#
# Assumption:
# We interpret each date_id as a trading day for annualization.
#
# No risk-free rate is subtracted because the responder is
# anonymized and represents a short-horizon forecasting target
# rather than a conventional investment return.

def annualized_sharpe_ratio(
    daily_returns,
    periods_per_year: int = 252,
) -> float:

    daily_returns = np.asarray(
        daily_returns,
        dtype=np.float64,
    )

    daily_returns = daily_returns[
        np.isfinite(daily_returns)
    ]

    if len(daily_returns) < 2:
        return np.nan

    volatility = np.std(
        daily_returns,
        ddof=1,
    )

    if volatility == 0:
        return np.nan

    return float(
        (
            np.mean(daily_returns)
            / volatility
        )
        * np.sqrt(periods_per_year)
    )


# ============================================================
# ANNUALIZED VOLATILITY
# ============================================================

def annualized_volatility(
    daily_returns,
    periods_per_year: int = 252,
) -> float:

    daily_returns = np.asarray(
        daily_returns,
        dtype=np.float64,
    )

    daily_returns = daily_returns[
        np.isfinite(daily_returns)
    ]

    if len(daily_returns) < 2:
        return np.nan

    return float(
        np.std(
            daily_returns,
            ddof=1,
        )
        * np.sqrt(periods_per_year)
    )


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================
# Decision:
# Measure drawdown from an additive cumulative P&L curve.
#
# Why:
# responder_6 is not necessarily a conventional percentage return.
# Compounding via (1 + return) may therefore create an economic
# interpretation that the anonymized target does not justify.
#
# We use cumulative strategy-return units instead.

def maximum_drawdown(
    daily_returns,
) -> float:

    daily_returns = np.asarray(
        daily_returns,
        dtype=np.float64,
    )

    daily_returns = daily_returns[
        np.isfinite(daily_returns)
    ]

    if len(daily_returns) == 0:
        return np.nan

    cumulative_pnl = np.cumsum(
        daily_returns
    )

    running_peak = np.maximum.accumulate(
        cumulative_pnl
    )

    drawdown = (
        cumulative_pnl
        - running_peak
    )

    return float(
        np.min(drawdown)
    )


# ============================================================
# STRATEGY HIT RATE
# ============================================================
# Percentage of dates with positive strategy returns.

def strategy_hit_rate(
    daily_returns,
) -> float:

    daily_returns = np.asarray(
        daily_returns,
        dtype=np.float64,
    )

    daily_returns = daily_returns[
        np.isfinite(daily_returns)
    ]

    if len(daily_returns) == 0:
        return np.nan

    return float(
        np.mean(
            daily_returns > 0
        )
    )


# ============================================================
# COMPLETE STRATEGY METRIC REPORT
# ============================================================

def calculate_strategy_metrics(
    daily_returns,
    periods_per_year: int = 252,
) -> dict[str, float]:

    daily_returns = np.asarray(
        daily_returns,
        dtype=np.float64,
    )

    daily_returns = daily_returns[
        np.isfinite(daily_returns)
    ]

    if len(daily_returns) == 0:
        raise ValueError(
            "Strategy return series is empty."
        )

    return {
        "number_of_days":
            int(len(daily_returns)),

        "mean_daily_return":
            float(
                np.mean(daily_returns)
            ),

        "daily_volatility":
            float(
                np.std(
                    daily_returns,
                    ddof=1,
                )
            )
            if len(daily_returns) > 1
            else np.nan,

        "annualized_sharpe":
            annualized_sharpe_ratio(
                daily_returns,
                periods_per_year,
            ),

        "annualized_volatility":
            annualized_volatility(
                daily_returns,
                periods_per_year,
            ),

        "maximum_drawdown":
            maximum_drawdown(
                daily_returns,
            ),

        "hit_rate":
            strategy_hit_rate(
                daily_returns,
            ),

        "cumulative_pnl":
            float(
                np.sum(daily_returns)
            ),
    }
