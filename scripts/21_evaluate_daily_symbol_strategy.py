# ============================================================
# IMPORTS
# ============================================================
# This script replaces scripts/20_evaluate_walk_forward_strategy.py
# as the source of any reported Sharpe/drawdown/turnover numbers.
#
# Script 20 averaged ~39 symbols x ~20 intraday time_id snapshots
# into a single date_id observation, which collapsed daily variance
# via the law of large numbers and produced an unannualized Sharpe
# of 20-24 with a 97% daily hit rate - both far outside any
# credible range for a real strategy (typical rough scale:
# 0-1 weak, 1-2 good, 2-3 excellent, 3-4 exceptional/scrutinize,
# 4-5+ very unusual, 20+ almost certainly a construction artifact),
# especially given the forecasting model itself only has ~53%
# observation-level directional accuracy.
#
# This script fixes that by collapsing intraday observations to ONE
# position per (date_id, symbol_id) BEFORE any position or P&L is
# computed, so each symbol contributes exactly one bet per day
# instead of ~20:
#
#   walk-forward predictions
#         |
#   group by (date_id, symbol_id)
#         |
#   aggregate prediction within symbol-day (mean)
#         |
#   aggregate realized responder within symbol-day (mean)
#         |
#   aggregate weight within symbol-day (mean, NOT sum)
#         |
#   construct ONE position for that symbol-day
#         |
#   cross-sectional daily portfolio (normalized by |weighted exposure|)
#         |
#   daily return proxy
#         |
#   Sharpe / drawdown / turnover (turnover uses each symbol's most
#   recent AVAILABLE prior date, preserving sparse-symbol semantics)
#
# A separate (date_id, time_id) snapshot-level robustness check is
# also produced, WITHOUT collapsing intraday observations first, and
# is deliberately left unannualized because Jane Street's own
# documentation states time_id spacing does not represent guaranteed
# equal real elapsed time.
#
# CAUTION ON WORDING:
# responder_6 is anonymized. Even after this fix, results must be
# described as:
#
#     "Annualized Sharpe of X on a responder-based long/short
#      strategy proxy in walk-forward simulation."
#
# NOT:
#
#     "This trading strategy would have made a Sharpe of X."

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
    aggregate_symbol_day,
    build_positions,
    calculate_snapshot_strategy_returns,
    evaluate_daily_symbol_strategy,
    summarize_snapshot_returns,
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

print(f"Raw intraday rows loaded: {predictions.height:,}")
print(
    f"Date range: {predictions['date_id'].min()} -> "
    f"{predictions['date_id'].max()}"
)


# ============================================================
# DATA-DRIVEN CLIP THRESHOLD
# ============================================================
# Decision:
# A single, deterministic, data-driven clip value (99th percentile
# of |prediction|), not a swept hyperparameter.
#
# CRITICAL FIX:
# This must be computed on the symbol-day AVERAGED prediction
# distribution, not the raw intraday one. An earlier version of
# this script computed the threshold from raw intraday predictions
# (99th pct ~= 0.399) and applied it to symbol-day means, whose
# distribution is much narrower (averaging ~20 intraday predictions
# per symbol-day shrinks the range, the same mean-reversion-of-
# variance effect responsible for the original inflated Sharpe
# bug). Since 0.399 sat far outside the symbol-day mean prediction
# range, the clip never activated - clipped_proportional silently
# degenerated into plain proportional (confirmed: both produced
# byte-identical Sharpe, average |position|, and every other
# metric). Computing the threshold on the actual aggregated
# distribution that sizing is applied to fixes this.

symbol_day_predictions = aggregate_symbol_day(
    predictions
)

symbol_day_clip_value = float(
    symbol_day_predictions["prediction"]
    .abs()
    .quantile(0.99)
)

raw_intraday_clip_value = float(
    predictions["prediction"]
    .abs()
    .quantile(0.99)
)

print(
    f"\nClip threshold (99th pct |prediction|, symbol-day means, "
    f"used in the PRIMARY backtest): "
    f"{symbol_day_clip_value:.6f}"
)
print(
    f"Clip threshold (99th pct |prediction|, raw intraday, used "
    f"ONLY in the snapshot robustness check below): "
    f"{raw_intraday_clip_value:.6f}"
)
print(
    "(these must stay separate - applying the raw-intraday "
    "threshold to symbol-day means, or vice versa, silently makes "
    "clipped_proportional degenerate into plain proportional or "
    "over-clip everything; that was the earlier bug)"
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
        clip_value=symbol_day_clip_value,
    ),
]

# Separate rule set for the snapshot robustness check, which uses
# RAW intraday predictions and therefore needs the raw-intraday
# clip threshold, not the symbol-day one.

SNAPSHOT_SIZING_RULES = [
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
        clip_value=raw_intraday_clip_value,
    ),
]


# ============================================================
# PRIMARY BACKTEST: ONE POSITION PER (date_id, symbol_id)
# ============================================================

print("\n" + "=" * 80)
print("PRIMARY BACKTEST: ONE POSITION PER (date_id, symbol_id)")
print("=" * 80)

primary_results = []
primary_daily_returns_by_strategy = {}

for rule in SIZING_RULES:

    print(f"\n--- {rule.name} ---")

    result, daily_returns_frame, positioned = (
        evaluate_daily_symbol_strategy(
            predictions,
            rule,
        )
    )

    primary_results.append(result)
    primary_daily_returns_by_strategy[rule.name] = daily_returns_frame

    print(
        f"Symbol-days:              {result['symbol_days']:,}"
    )
    print(
        f"Average |position|:       "
        f"{result['average_absolute_position']:.6f}"
    )
    print(
        f"Annualized Sharpe:        "
        f"{result['annualized_sharpe']:.6f}"
    )
    print(
        f"Annualized volatility:    "
        f"{result['annualized_volatility']:.6f}"
    )
    print(
        f"Maximum drawdown:         "
        f"{result['maximum_drawdown']:.6f}"
    )
    print(
        f"Cumulative P&L (proxy):   "
        f"{result['cumulative_pnl']:.6f}"
    )
    print(
        f"Hit rate:                 "
        f"{result['hit_rate']:.6f}"
    )
    print(
        f"Mean daily turnover:      "
        f"{result['mean_daily_turnover']:.6f}"
        f"  (over {result['turnover_dates']} dates)"
    )


# ============================================================
# SAVE PRIMARY RESULTS
# ============================================================

primary_results_frame = pd.DataFrame(
    primary_results
)

primary_results_path = (
    METRICS_PATH
    / "daily_symbol_strategy_results.csv"
)

primary_results_frame.to_csv(
    primary_results_path,
    index=False,
)


# ============================================================
# PRINT PRIMARY COMPARISON TABLE
# ============================================================

print("\n" + "=" * 130)
print(
    "PRIMARY STRATEGY COMPARISON "
    "(one position per date_id x symbol_id)"
)
print("=" * 130)

display_columns = [
    "strategy",
    "annualized_sharpe",
    "maximum_drawdown",
    "cumulative_pnl",
    "hit_rate",
    "mean_daily_turnover",
    "average_absolute_position",
    "annualized_volatility",
    "number_of_days",
]

print(
    primary_results_frame[display_columns].to_string(index=False)
)

primary_results_frame["sharpe_per_unit_turnover"] = (
    primary_results_frame["annualized_sharpe"]
    / primary_results_frame["mean_daily_turnover"]
)

print("\n" + "=" * 80)
print("SHARPE PER UNIT OF DAILY TURNOVER")
print("=" * 80)

print(
    primary_results_frame[
        [
            "strategy",
            "annualized_sharpe",
            "mean_daily_turnover",
            "sharpe_per_unit_turnover",
        ]
    ].to_string(index=False)
)


# ============================================================
# SANITY CHECK AGAINST THE INFLATED SCRIPT-20 RESULT
# ============================================================

print("\n" + "=" * 80)
print("SANITY CHECK: WAS THE ARTIFICIAL SMOOTHING REDUCED?")
print("=" * 80)

for _, row in primary_results_frame.iterrows():
    flag = (
        "STILL SUSPICIOUSLY HIGH"
        if row["annualized_sharpe"] > 5
        else "within a defensible range"
    )
    print(
        f"{row['strategy']:<22} Sharpe = "
        f"{row['annualized_sharpe']:>8.4f}   ({flag})"
    )


# ============================================================
# PLOT CUMULATIVE P&L PROXY BY STRATEGY (PRIMARY BACKTEST)
# ============================================================

FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

plt.figure(figsize=(12, 6))

for rule in SIZING_RULES:

    daily_returns_frame = (
        primary_daily_returns_by_strategy[rule.name]
    )

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
    "Daily-Symbol Strategy Proxy: Cumulative P&L by Sizing Rule\n"
    "(one position per date_id x symbol_id - proxy units only)"
)
plt.xlabel("date_id")
plt.ylabel("Cumulative P&L (proxy units)")
plt.legend()
plt.tight_layout()

primary_figure_path = (
    FIGURES_PATH
    / "daily_symbol_strategy_cumulative_pnl.png"
)

plt.savefig(
    primary_figure_path,
    dpi=150,
)

plt.close()


# ============================================================
# ROBUSTNESS CHECK: (date_id, time_id) SNAPSHOT CROSS-SECTION
# ============================================================
# This uses the RAW intraday predictions directly - no symbol-day
# aggregation - as the opposite-extreme aggregation choice, purely
# to check whether the daily-symbol result above is sensitive to
# how aggregation was done.
#
# NOT annualized: time_id spacing is not guaranteed to represent
# equal real elapsed time.

print("\n" + "=" * 80)
print("ROBUSTNESS CHECK: (date_id, time_id) SNAPSHOT CROSS-SECTION")
print("(deliberately NOT annualized)")
print("=" * 80)

snapshot_results = []

for rule in SNAPSHOT_SIZING_RULES:

    print(f"\n--- {rule.name} ---")

    positioned_raw = build_positions(
        predictions,
        method=rule.method,
        prediction_column="prediction",
        clip_value=rule.clip_value,
    )

    snapshot_returns_frame = calculate_snapshot_strategy_returns(
        positioned_raw
    )

    snapshot_summary = summarize_snapshot_returns(
        snapshot_returns_frame["snapshot_return"].to_numpy()
    )

    snapshot_summary["strategy"] = rule.name

    snapshot_results.append(snapshot_summary)

    print(
        f"Snapshots:                  "
        f"{snapshot_summary['number_of_snapshots']:,}"
    )
    print(
        f"Mean snapshot return:       "
        f"{snapshot_summary['mean_snapshot_return']:.8f}"
    )
    print(
        f"Std snapshot return:        "
        f"{snapshot_summary['std_snapshot_return']:.8f}"
    )
    print(
        f"Unannualized mean/std:      "
        f"{snapshot_summary['unannualized_mean_std_ratio']:.6f}"
    )
    print(
        f"Hit rate:                   "
        f"{snapshot_summary['hit_rate']:.6f}"
    )
    print(
        f"Cumulative P&L (proxy):     "
        f"{snapshot_summary['cumulative_pnl']:.6f}"
    )


snapshot_results_frame = pd.DataFrame(
    snapshot_results
)

snapshot_results_path = (
    METRICS_PATH
    / "snapshot_strategy_results.csv"
)

snapshot_results_frame.to_csv(
    snapshot_results_path,
    index=False,
)

print("\n" + "=" * 110)
print("SNAPSHOT-LEVEL COMPARISON (unannualized)")
print("=" * 110)

snapshot_display_columns = [
    "strategy",
    "number_of_snapshots",
    "mean_snapshot_return",
    "std_snapshot_return",
    "unannualized_mean_std_ratio",
    "hit_rate",
    "cumulative_pnl",
]

print(
    snapshot_results_frame[
        snapshot_display_columns
    ].to_string(index=False)
)


# ============================================================
# FINAL STATUS AND CAUTION
# ============================================================

print("\n" + "=" * 90)
print("IMPORTANT CAUTION ON INTERPRETATION")
print("=" * 90)

print(
    "responder_6 is an anonymized Jane Street target - even with "
    "this corrected aggregation, these numbers are NOT a literal "
    "tradable-return backtest. The strongest defensible wording "
    "is:\n\n"
    "    'Annualized Sharpe of X on a responder-based long/short "
    "strategy proxy in walk-forward simulation.'\n\n"
    "NOT:\n\n"
    "    'This trading strategy would have made a Sharpe of X.'\n\n"
    "The primary (date_id x symbol_id) backtest is the number to "
    "report for annualized Sharpe/drawdown/turnover. The snapshot-"
    "level check above is a robustness check only and is "
    "deliberately left unannualized."
)

print("\nSaved primary strategy results to:")
print(primary_results_path)

print("\nSaved snapshot-level results to:")
print(snapshot_results_path)

print("\nSaved cumulative P&L figure to:")
print(primary_figure_path)

print("\nDaily-symbol strategy evaluation complete.")
