# ============================================================
# IMPORTS
# ============================================================
# This script validates the metric implementations using small
# deterministic examples before they are used on model output.

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ALLOW IMPORTS FROM PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOAD METRICS
# ============================================================

from src.metrics import (
    annualized_sharpe_ratio,
    calculate_daily_strategy_returns,
    calculate_forecast_metrics,
    calculate_strategy_metrics,
    maximum_drawdown,
    weighted_zero_mean_r2,
)


# ============================================================
# TEST DATA
# ============================================================

y_true = np.array(
    [
        1.0,
        -1.0,
        2.0,
        -2.0,
    ]
)

weights = np.array(
    [
        1.0,
        1.0,
        2.0,
        2.0,
    ]
)


# ============================================================
# TEST 1: ZERO PREDICTION MUST HAVE ZERO SCORE
# ============================================================
# This is the defining baseline for zero-mean R².

zero_predictions = np.zeros_like(
    y_true
)

zero_score = weighted_zero_mean_r2(
    y_true,
    zero_predictions,
    weights,
)

assert np.isclose(
    zero_score,
    0.0,
)

print(
    "PASS: zero prediction produces "
    "weighted zero-mean R² = 0."
)


# ============================================================
# TEST 2: PERFECT PREDICTION MUST HAVE SCORE 1
# ============================================================

perfect_score = weighted_zero_mean_r2(
    y_true,
    y_true,
    weights,
)

assert np.isclose(
    perfect_score,
    1.0,
)

print(
    "PASS: perfect prediction produces "
    "weighted zero-mean R² = 1."
)


# ============================================================
# TEST 3: BAD PREDICTION CAN BE NEGATIVE
# ============================================================
# Predicting the exact opposite direction should perform much
# worse than the zero baseline.

opposite_predictions = (
    -y_true
)

opposite_score = weighted_zero_mean_r2(
    y_true,
    opposite_predictions,
    weights,
)

assert opposite_score < 0

print(
    "PASS: poor predictions can produce "
    "negative weighted zero-mean R²."
)


# ============================================================
# TEST 4: COMPLETE FORECAST REPORT
# ============================================================

sample_predictions = np.array(
    [
        0.8,
        -0.7,
        1.5,
        -1.8,
    ]
)

forecast_metrics = calculate_forecast_metrics(
    y_true,
    sample_predictions,
    weights,
)

required_forecast_metrics = {
    "weighted_zero_mean_r2",
    "weighted_mse",
    "weighted_rmse",
    "weighted_mae",
    "ordinary_r2",
    "ordinary_mse",
    "ordinary_rmse",
    "ordinary_mae",
    "information_coefficient",
    "directional_accuracy",
}

assert (
    set(forecast_metrics.keys())
    == required_forecast_metrics
)

print(
    "PASS: complete forecast metric report created."
)


# ============================================================
# TEST 5: DAILY STRATEGY AGGREGATION
# ============================================================

strategy_data = pd.DataFrame(
    {
        "date_id": [
            1,
            1,
            2,
            2,
        ],

        "responder_6": [
            1.0,
            -0.5,
            0.8,
            -1.2,
        ],

        "position": [
            1.0,
            -1.0,
            1.0,
            -1.0,
        ],

        "weight": [
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    }
)

daily_returns = calculate_daily_strategy_returns(
    strategy_data
)

assert len(daily_returns) == 2

assert np.all(
    np.isfinite(
        daily_returns["strategy_return"]
    )
)

print(
    "PASS: observation-level P&L aggregates "
    "to date-level strategy returns."
)


# ============================================================
# TEST 6: SHARPE AND DRAWDOWN
# ============================================================

synthetic_daily_returns = np.array(
    [
        0.01,
        0.02,
        -0.01,
        0.015,
        -0.005,
    ]
)

sharpe = annualized_sharpe_ratio(
    synthetic_daily_returns
)

drawdown = maximum_drawdown(
    synthetic_daily_returns
)

assert np.isfinite(sharpe)
assert drawdown <= 0

print(
    "PASS: Sharpe and maximum drawdown "
    "produce valid values."
)


# ============================================================
# TEST 7: COMPLETE STRATEGY REPORT
# ============================================================

strategy_metrics = calculate_strategy_metrics(
    synthetic_daily_returns
)

required_strategy_metrics = {
    "number_of_days",
    "mean_daily_return",
    "daily_volatility",
    "annualized_sharpe",
    "annualized_volatility",
    "maximum_drawdown",
    "hit_rate",
    "cumulative_pnl",
}

assert (
    set(strategy_metrics.keys())
    == required_strategy_metrics
)

print(
    "PASS: complete strategy metric report created."
)


# ============================================================
# PRINT EXAMPLE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE FORECAST METRICS")
print("=" * 70)

for metric_name, metric_value in forecast_metrics.items():

    print(
        f"{metric_name:<30} "
        f"{metric_value:.6f}"
    )


print("\n" + "=" * 70)
print("EXAMPLE STRATEGY METRICS")
print("=" * 70)

for metric_name, metric_value in strategy_metrics.items():

    if isinstance(
        metric_value,
        (float, np.floating),
    ):
        print(
            f"{metric_name:<30} "
            f"{metric_value:.6f}"
        )
    else:
        print(
            f"{metric_name:<30} "
            f"{metric_value}"
        )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("ALL METRIC TESTS PASSED")
print("=" * 70)
