# ============================================================
# IMPORTS
# ============================================================
# This script tests how sensitive the daily-symbol portfolio
# strategy proxy (scripts/21) is to trading friction.
#
# We do NOT pick one arbitrary transaction cost and announce the
# resulting Sharpe. responder_6 is anonymized and its units are not
# known to correspond to a specific real percentage return, so there
# is no defensible way to claim "Jane Street would charge N basis
# points." Instead we sweep several abstract cost levels and report
# how the result degrades, to answer:
#
#     How sensitive is the apparent strategy edge to trading
#     friction, given daily turnover around ~1.44?
#
# Cost model:
#
#     cost_d = c * turnover_d
#     net_return_d = strategy_return_d - cost_d
#
# where turnover_d is the sparse-symbol-aware daily turnover series
# from src/strategy.py (calculate_symbol_day_turnover_series),
# already used to report the ~1.44 mean daily turnover for
# clipped_proportional in script 21.

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

from src.metrics import (
    calculate_daily_strategy_returns,
    calculate_strategy_metrics,
)

from src.strategy import (
    PositionSizingRule,
    aggregate_symbol_day,
    build_positions,
    calculate_symbol_day_turnover_series,
)


# ============================================================
# COST LEVELS
# ============================================================
# Deliberately abstract, not calibrated to any claimed real-world
# basis-point figure. Chosen to span from "no friction" to a level
# large enough to visibly stress the strategy given turnover ~1.44.
#
# IMPORTANT:
# The first version of this sweep (0 to 0.001) turned out to be far
# too narrow to be informative - it left Sharpe essentially flat
# (11.17 -> 10.75, a 3.8% decline) because the script's own
# approximate-breakeven diagnostic put the point where mean net
# return reaches zero around c ~= 0.0275, roughly 27x above the
# highest level originally tested. A sensitivity sweep that never
# gets near the breakeven point doesn't actually answer "how
# sensitive is this to cost" - it just confirms the strategy
# survives negligible friction, which was never in doubt. The range
# below extends well past that breakeven estimate so the sweep
# actually traces out where the edge degrades and disappears.

COST_LEVELS = [
    (
        "none",
        0.0,
    ),
    (
        "low",
        0.0001,
    ),
    (
        "medium",
        0.00025,
    ),
    (
        "medium_high",
        0.0005,
    ),
    (
        "high",
        0.001,
    ),
    (
        "very_high",
        0.0025,
    ),
    (
        "extreme",
        0.005,
    ),
    (
        "near_breakeven",
        0.01,
    ),
    (
        "at_breakeven_estimate",
        0.0275,
    ),
    (
        "beyond_breakeven",
        0.05,
    ),
]


# ============================================================
# LOAD PREDICTIONS AND REBUILD THE WINNING PORTFOLIO STRATEGY
# ============================================================
# Identical construction to script 21/22: symbol-day aggregation,
# clip threshold from the symbol-day prediction distribution,
# clipped_proportional sizing.

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

symbol_day_data = aggregate_symbol_day(
    predictions
)

clip_value = float(
    symbol_day_data["prediction"]
    .abs()
    .quantile(0.99)
)

sizing_rule = PositionSizingRule(
    name="clipped_proportional",
    method="clipped_proportional",
    clip_value=clip_value,
)

positioned = build_positions(
    symbol_day_data,
    method=sizing_rule.method,
    prediction_column="prediction",
    clip_value=sizing_rule.clip_value,
)

print(f"Symbol-day rows: {positioned.height:,}")
print(f"Clip threshold used: {clip_value:.6f}")


# ============================================================
# GROSS (NO-COST) DAILY RETURNS
# ============================================================

positioned_pd = positioned.to_pandas()

gross_daily_returns = calculate_daily_strategy_returns(
    positioned_pd,
    date_column="date_id",
    target_column="actual_responder_6",
    position_column="position",
    weight_column="weight",
)

gross_metrics = calculate_strategy_metrics(
    gross_daily_returns["strategy_return"].to_numpy()
)

print(
    f"\nGross (no-cost) annualized Sharpe: "
    f"{gross_metrics['annualized_sharpe']:.6f}"
)
print(
    f"(should match script 21/22's clipped_proportional result, "
    f"as a self-consistency check)"
)


# ============================================================
# PER-DATE TURNOVER SERIES
# ============================================================

daily_turnover = calculate_symbol_day_turnover_series(
    positioned,
    position_column="position",
    symbol_column="symbol_id",
    date_column="date_id",
)

daily_turnover_pd = daily_turnover.to_pandas()

print(
    f"\nMean daily turnover: "
    f"{daily_turnover_pd['daily_turnover'].mean():.6f}   "
    f"(over {len(daily_turnover_pd)} dates)"
)


# ============================================================
# MERGE RETURNS AND TURNOVER
# ============================================================
# Decision:
# Left-join so every return-bearing date is kept even if it has no
# defined turnover (only the very first date, which has no prior
# position to compare against). Missing turnover is filled with 0 -
# i.e. the initial deployment day is assumed cost-free, consistent
# with how turnover is already defined (it only exists once a prior
# position exists).

combined = gross_daily_returns.merge(
    daily_turnover_pd,
    on="date_id",
    how="left",
)

combined["daily_turnover"] = (
    combined["daily_turnover"]
    .fillna(0.0)
)

mean_gross_return = combined["strategy_return"].mean()
mean_turnover = combined["daily_turnover"].mean()


# ============================================================
# APPROXIMATE BREAKEVEN COST
# ============================================================
# Decision:
# A simple diagnostic - the cost-per-unit-turnover level at which
# MEAN daily net return would reach exactly zero, assuming mean
# return and mean turnover as fixed averages.
#
# Tradeoff:
# This is illustrative only. It ignores the fact that turnover and
# return vary together day to day rather than at their averages, so
# treat it as an order-of-magnitude signpost, not a precise
# threshold.

approximate_breakeven_cost = (
    mean_gross_return / mean_turnover
    if mean_turnover > 0
    else float("nan")
)

print(
    f"\nApproximate breakeven cost level (mean return / mean "
    f"turnover): {approximate_breakeven_cost:.8f}"
)
print(
    "(illustrative only - uses averages, not the actual "
    "day-by-day joint distribution)"
)


# ============================================================
# SWEEP COST LEVELS
# ============================================================

cost_results = []
net_returns_by_level = {}

print("\n" + "=" * 90)
print("TRANSACTION COST SENSITIVITY SWEEP")
print("=" * 90)

for level_name, cost_per_unit_turnover in COST_LEVELS:

    net_return = (
        combined["strategy_return"]
        - cost_per_unit_turnover * combined["daily_turnover"]
    )

    net_returns_by_level[level_name] = net_return

    net_metrics = calculate_strategy_metrics(
        net_return.to_numpy()
    )

    average_daily_cost = (
        cost_per_unit_turnover
        * combined["daily_turnover"]
    ).mean()

    cost_as_fraction_of_gross_return = (
        average_daily_cost / mean_gross_return
        if mean_gross_return != 0
        else float("nan")
    )

    result_row = {
        "cost_level": level_name,
        "cost_per_unit_turnover": cost_per_unit_turnover,
        "average_daily_cost": average_daily_cost,
        "cost_as_fraction_of_gross_return":
            cost_as_fraction_of_gross_return,
        **net_metrics,
    }

    cost_results.append(result_row)

    print(
        f"\n--- {level_name} (c={cost_per_unit_turnover}) ---"
    )
    print(
        f"Average daily cost:        {average_daily_cost:.8f}"
    )
    print(
        f"Cost / gross daily return: "
        f"{cost_as_fraction_of_gross_return:.4%}"
    )
    print(
        f"Annualized Sharpe:         "
        f"{net_metrics['annualized_sharpe']:.6f}"
    )
    print(
        f"Maximum drawdown:          "
        f"{net_metrics['maximum_drawdown']:.6f}"
    )
    print(
        f"Cumulative P&L (proxy):    "
        f"{net_metrics['cumulative_pnl']:.6f}"
    )
    print(
        f"Hit rate:                  "
        f"{net_metrics['hit_rate']:.6f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

cost_results_frame = pd.DataFrame(
    cost_results
)

cost_results_path = (
    METRICS_PATH
    / "transaction_cost_sensitivity_results.csv"
)

cost_results_frame.to_csv(
    cost_results_path,
    index=False,
)


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 130)
print("TRANSACTION COST SENSITIVITY (clipped_proportional, daily-symbol portfolio)")
print("=" * 130)

display_columns = [
    "cost_level",
    "cost_per_unit_turnover",
    "average_daily_cost",
    "cost_as_fraction_of_gross_return",
    "annualized_sharpe",
    "maximum_drawdown",
    "cumulative_pnl",
    "hit_rate",
]

print(
    cost_results_frame[display_columns].to_string(index=False)
)


# ============================================================
# PLOT NET CUMULATIVE P&L ACROSS COST LEVELS
# ============================================================

FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

plt.figure(figsize=(12, 6))

sorted_dates = combined["date_id"]

for level_name, _ in COST_LEVELS:

    cumulative_net_pnl = (
        net_returns_by_level[level_name]
        .cumsum()
    )

    plt.plot(
        sorted_dates,
        cumulative_net_pnl,
        label=level_name,
    )

plt.axhline(
    0,
    linestyle="--",
    color="gray",
)

plt.title(
    "Transaction Cost Sensitivity: Cumulative Net P&L Proxy\n"
    "(clipped_proportional, daily-symbol portfolio - abstract cost levels)"
)
plt.xlabel("date_id")
plt.ylabel("Cumulative net P&L (proxy units)")
plt.legend()
plt.tight_layout()

figure_path = (
    FIGURES_PATH
    / "transaction_cost_sensitivity.png"
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
    "These cost levels are abstract sensitivity bands, not a claim "
    "about actual Jane Street trading economics - responder_6 is "
    "anonymized, so no specific basis-point figure is defensible. "
    "The correct conclusion to draw is about SENSITIVITY: how much "
    "the apparent edge compresses as an abstract friction "
    "assumption increases, given daily turnover around "
    f"{mean_turnover:.2f}. It is NOT a claim about what this "
    "strategy would actually earn net of real trading costs."
)

print("\nSaved cost sensitivity results to:")
print(cost_results_path)

print("\nSaved net cumulative P&L figure to:")
print(figure_path)

print("\nTransaction cost sensitivity analysis complete.")
