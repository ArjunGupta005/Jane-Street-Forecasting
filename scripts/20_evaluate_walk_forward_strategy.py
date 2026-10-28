# ============================================================
# IMPORTS
# ============================================================
# This script converts the walk-forward out-of-sample predictions
# (scripts/19_walk_forward_simulation.py) into a simple long/short
# strategy proxy and evaluates three position-sizing rules:
#
#   1. sign                  : position = sign(prediction)
#   2. proportional          : position = prediction
#   3. clipped_proportional  : position = clip(prediction, -c, c)
#
# IMPORTANT CAUTION:
# responder_6 is an anonymized Jane Street target, not a literal
# tradable percentage return. The Sharpe, drawdown, and turnover
# figures below should be described as:
#
#     "Converted walk-forward predictions into a weighted
#      long/short strategy proxy and evaluated annualized Sharpe,
#      drawdown, and turnover."
#
# NOT as a literal deployable portfolio Sharpe. This wording
# distinction matters and is repeated in the final printout.
#
# We deliberately do not tune the clip threshold against many
# candidate values. A single, data-driven choice - the 99th
# percentile of |prediction| across the full walk-forward series -
# is used instead, to avoid quietly overfitting a strategy
# hyperparameter to this one out-of-sample series.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    FIGURES_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
)

from src.strategy import (
    PositionSizingRule,
    evaluate_strategy,
)


# ============================================================
# LOAD WALK-FORWARD PREDICTIONS
# ============================================================

predictions_path = (
    PREDICTIONS_PATH
    / "walk_forward_predictions.csv"
)

print("\n" + "=" * 80)
print("LOADING WALK-FORWARD PREDICTIONS")
print("=" * 80)

predictions = pl.read_csv(
    predictions_path
)

print(f"Rows loaded: {predictions.height:,}")
print(
    f"Date range: {predictions['date_id'].min()} -> "
    f"{predictions['date_id'].max()}"
)
print(
    f"Blocks: {predictions['block_number'].min()} -> "
    f"{predictions['block_number'].max()}"
)


# ============================================================
# DATA-DRIVEN CLIP THRESHOLD
# ============================================================
# Decision:
# Clip at the 99th percentile of |prediction| across the entire
# walk-forward series.
#
# Why:
# This removes only the most extreme tail of predictions rather
# than an arbitrary round number, and is a single deterministic
# choice rather than a swept hyperparameter.

absolute_predictions = (
    predictions["prediction"]
    .abs()
)

clip_value = float(
    absolute_predictions
    .quantile(0.99)
)

print(
    f"\nPrediction magnitude summary:"
)
print(
    f"  mean |prediction|:   "
    f"{absolute_predictions.mean():.6f}"
)
print(
    f"  median |prediction|: "
    f"{absolute_predictions.median():.6f}"
)
print(
    f"  99th pct |prediction| (clip value): "
    f"{clip_value:.6f}"
)
print(
    f"  max |prediction|:    "
    f"{absolute_predictions.max():.6f}"
)


# ============================================================
# POSITION SIZING RULES
# ============================================================

SIZING_RULES = [
    PositionSizingRule(
        name="sign",
        method="sign",
    ),
    PositionSizingRule(
        name="proportional",
        method="proportional",
    ),
    PositionSizingRule(
        name="clipped_proportional",
        method="clipped_proportional",
        clip_value=clip_value,
    ),
]


# ============================================================
# EVALUATE EACH STRATEGY
# ============================================================

strategy_results = []
daily_returns_by_strategy = {}

for rule in SIZING_RULES:

    print("\n" + "=" * 80)
    print(f"STRATEGY: {rule.name}")
    print("=" * 80)

    result, daily_returns_frame = evaluate_strategy(
        predictions,
        rule,
    )

    strategy_results.append(result)
    daily_returns_by_strategy[rule.name] = daily_returns_frame

    print(
        f"Average |position|:      "
        f"{result['average_absolute_position']:.6f}"
    )
    print(
        f"Annualized Sharpe:       "
        f"{result['annualized_sharpe']:.6f}"
    )
    print(
        f"Annualized volatility:   "
        f"{result['annualized_volatility']:.6f}"
    )
    print(
        f"Maximum drawdown:        "
        f"{result['maximum_drawdown']:.6f}"
    )
    print(
        f"Cumulative P&L (proxy):  "
        f"{result['cumulative_pnl']:.6f}"
    )
    print(
        f"Hit rate:                "
        f"{result['hit_rate']:.6f}"
    )
    print(
        f"Mean daily turnover/symbol: "
        f"{result['mean_daily_turnover_per_symbol']:.6f}"
    )


# ============================================================
# COMBINE AND SAVE RESULTS
# ============================================================

results_frame = pd.DataFrame(
    strategy_results
)

results_path = (
    METRICS_PATH
    / "strategy_comparison_results.csv"
)

results_frame.to_csv(
    results_path,
    index=False,
)


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 130)
print("STRATEGY COMPARISON (weighted long/short proxy)")
print("=" * 130)

display_columns = [
    "strategy",
    "annualized_sharpe",
    "maximum_drawdown",
    "cumulative_pnl",
    "hit_rate",
    "mean_daily_turnover_per_symbol",
    "average_absolute_position",
    "annualized_volatility",
    "number_of_days",
]

print(
    results_frame[display_columns].to_string(index=False)
)


# ============================================================
# SHARPE-PER-UNIT-TURNOVER
# ============================================================
# Decision:
# Report a simple efficiency ratio (Sharpe / turnover) alongside
# the raw comparison.
#
# Why:
# A strategy with the highest raw Sharpe but far higher turnover
# than the alternatives may be less attractive once realistic
# trading costs are considered. This ratio makes that tradeoff
# visible without requiring us to assume a specific cost model.

results_frame["sharpe_per_unit_turnover"] = (
    results_frame["annualized_sharpe"]
    / results_frame["mean_daily_turnover_per_symbol"]
)

print("\n" + "=" * 80)
print("SHARPE PER UNIT OF DAILY TURNOVER")
print("=" * 80)

print(
    results_frame[
        [
            "strategy",
            "annualized_sharpe",
            "mean_daily_turnover_per_symbol",
            "sharpe_per_unit_turnover",
        ]
    ].to_string(index=False)
)


# ============================================================
# PLOT CUMULATIVE P&L PROXY BY STRATEGY
# ============================================================

FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

plt.figure(figsize=(12, 6))

for rule in SIZING_RULES:

    daily_returns_frame = daily_returns_by_strategy[rule.name]

    sorted_returns = (
        daily_returns_frame
        .sort_values("date_id")
    )

    cumulative_pnl = (
        sorted_returns["strategy_return"]
        .cumsum()
    )

    plt.plot(
        sorted_returns["date_id"],
        cumulative_pnl,
        label=rule.name,
    )

plt.axhline(
    0,
    linestyle="--",
    color="gray",
)

plt.title(
    "Walk-Forward Strategy Proxy: Cumulative P&L by Position Sizing Rule\n"
    "(proxy units - responder_6 is anonymized, not a literal return)"
)
plt.xlabel("date_id")
plt.ylabel("Cumulative P&L (proxy units)")
plt.legend()
plt.tight_layout()

figure_path = (
    FIGURES_PATH
    / "strategy_cumulative_pnl.png"
)

plt.savefig(
    figure_path,
    dpi=150,
)

plt.close()


# ============================================================
# FINAL STATUS AND CAUTION
# ============================================================

print("\n" + "=" * 90)
print("IMPORTANT CAUTION ON INTERPRETATION")
print("=" * 90)

print(
    "responder_6 is an anonymized Jane Street target, not a "
    "literal tradable percentage return. These results should be "
    "described as:\n\n"
    "    'Converted walk-forward predictions into a weighted "
    "long/short strategy proxy and evaluated annualized Sharpe, "
    "drawdown, and turnover.'\n\n"
    "They should NOT be described as a literal deployable "
    "portfolio Sharpe unless a defensible economic mapping from "
    "responder units to tradable returns is separately "
    "established."
)

print("\nSaved strategy comparison to:")
print(results_path)

print("\nSaved cumulative P&L figure to:")
print(figure_path)

print("\nStrategy evaluation complete.")
