# ============================================================
# IMPORTS
# ============================================================
# This script answers the question raised by the ~11.17 portfolio
# Sharpe from scripts/21_evaluate_daily_symbol_strategy.py:
#
#     How much of that Sharpe comes from genuine predictive edge
#     at the instrument level, versus cross-sectional
#     diversification across ~39 symbols?
#
# For every symbol_id, we build that symbol's OWN daily strategy-
# return proxy across the walk-forward period, using the exact same
# clipped_proportional sizing rule as the winning portfolio
# strategy, then compute its own Sharpe in isolation.
#
# Because a single-symbol "portfolio" has only one constituent, the
# cross-sectional P&L formula collapses to:
#
#     R_{s,d} = sign(position_{s,d}) * ybar_{s,d}
#
# (weight cancels out of the ratio when there is only one symbol),
# so this is genuinely the return of trading that symbol alone with
# no diversification benefit whatsoever.
#
# Interpretation guide:
#
#     median/mean single-symbol Sharpe ~1-2, portfolio Sharpe ~11
#         -> coherent story: modest instrument-level edge,
#            dramatic variance reduction from diversification
#
#     individual symbols routinely showing Sharpe 6-10
#         -> still a methodological issue somewhere, keep debugging
#
# We do NOT target a "believable" number here. We report whatever
# this decomposition actually shows.

import sys
from pathlib import Path

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
    METRICS_PATH,
    PREDICTIONS_PATH,
)

from src.metrics import (
    calculate_daily_strategy_returns,
    calculate_strategy_metrics,
    information_coefficient,
)

from src.strategy import (
    PositionSizingRule,
    aggregate_symbol_day,
    build_positions,
)


# ============================================================
# CONSTANTS
# ============================================================
# Decision:
# A symbol needs a minimum number of days present before its Sharpe
# is trusted enough to include in the summary statistics (median,
# mean, min/max, % positive).
#
# Why:
# Sharpe estimated from a handful of days is extremely noisy. This
# does not discard those symbols from the output - they are still
# saved with their full detail - it only excludes them from the
# aggregate summary so a few thinly-observed symbols can't dominate
# the median/mean.

MIN_DAYS_FOR_SHARPE_SUMMARY = 60

WINNING_SIZING_RULE = PositionSizingRule(
    name="clipped_proportional",
    method="clipped_proportional",
    clip_value=None,
)


# ============================================================
# LOAD WALK-FORWARD PREDICTIONS AND AGGREGATE TO SYMBOL-DAY
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

symbol_day_data = aggregate_symbol_day(
    predictions
)

print(f"Symbol-day rows: {symbol_day_data.height:,}")
print(
    f"Unique symbols: "
    f"{symbol_day_data['symbol_id'].n_unique()}"
)


# ============================================================
# APPLY THE WINNING SIZING RULE (clipped_proportional)
# ============================================================
# Same clip-threshold discipline as script 21: computed from the
# symbol-day AVERAGED prediction distribution, not the raw intraday
# one.

clip_value = float(
    symbol_day_data["prediction"]
    .abs()
    .quantile(0.99)
)

print(
    f"\nClip threshold (99th pct |prediction|, symbol-day means): "
    f"{clip_value:.6f}"
)

sizing_rule = PositionSizingRule(
    name=WINNING_SIZING_RULE.name,
    method=WINNING_SIZING_RULE.method,
    clip_value=clip_value,
)

positioned = build_positions(
    symbol_day_data,
    method=sizing_rule.method,
    prediction_column="prediction",
    clip_value=sizing_rule.clip_value,
)


# ============================================================
# PORTFOLIO-LEVEL RESULT (RECOMPUTED HERE FOR A CONSISTENCY CHECK)
# ============================================================
# This should reproduce script 21's ~11.17 Sharpe for
# clipped_proportional. Recomputing it in this script, rather than
# hardcoding the earlier number, means this comparison is
# self-verifying rather than trusting a copy-pasted figure.

positioned_pd = positioned.to_pandas()

portfolio_daily_returns = calculate_daily_strategy_returns(
    positioned_pd,
    date_column="date_id",
    target_column="actual_responder_6",
    position_column="position",
    weight_column="weight",
)

portfolio_metrics = calculate_strategy_metrics(
    portfolio_daily_returns["strategy_return"].to_numpy()
)

print("\n" + "=" * 80)
print("PORTFOLIO-LEVEL RESULT (all symbols, cross-sectional)")
print("=" * 80)

print(
    f"Portfolio annualized Sharpe: "
    f"{portfolio_metrics['annualized_sharpe']:.6f}"
)
print(
    f"(should match script 21's clipped_proportional result, "
    f"~11.17, as a self-consistency check)"
)


# ============================================================
# PER-SYMBOL DECOMPOSITION
# ============================================================
# For each symbol_id in isolation:
#
#   R_{s,d} = sign(position_{s,d}) * ybar_{s,d}
#
# computed via the SAME calculate_daily_strategy_returns() function
# used for the full portfolio, just applied to a single-symbol
# subset, so the P&L definition is guaranteed identical.

print("\n" + "=" * 80)
print("COMPUTING PER-SYMBOL STRATEGY RETURNS")
print("=" * 80)

unique_symbols = sorted(
    positioned["symbol_id"].unique().to_list()
)

symbol_rows = []
symbol_daily_return_frames = []

for symbol in unique_symbols:

    symbol_data = (
        positioned
        .filter(
            pl.col("symbol_id") == symbol
        )
        .sort("date_id")
    )

    days_available = symbol_data.height

    symbol_data_pd = symbol_data.to_pandas()

    symbol_daily_returns = calculate_daily_strategy_returns(
        symbol_data_pd,
        date_column="date_id",
        target_column="actual_responder_6",
        position_column="position",
        weight_column="weight",
    )

    symbol_daily_returns["symbol_id"] = symbol

    symbol_daily_return_frames.append(
        symbol_daily_returns
    )

    if days_available >= 2:

        symbol_metrics = calculate_strategy_metrics(
            symbol_daily_returns["strategy_return"].to_numpy()
        )

    else:

        symbol_metrics = {
            "annualized_sharpe": float("nan"),
            "hit_rate": float("nan"),
            "cumulative_pnl": float("nan"),
            "maximum_drawdown": float("nan"),
        }

    symbol_ic = information_coefficient(
        symbol_data["actual_responder_6"].to_numpy(),
        symbol_data["prediction"].to_numpy(),
    )

    symbol_rows.append(
        {
            "symbol_id": symbol,
            "days_available": days_available,
            "meets_min_days": (
                days_available
                >= MIN_DAYS_FOR_SHARPE_SUMMARY
            ),
            "annualized_sharpe": symbol_metrics[
                "annualized_sharpe"
            ],
            "information_coefficient": symbol_ic,
            "hit_rate": symbol_metrics["hit_rate"],
            "cumulative_pnl": symbol_metrics["cumulative_pnl"],
            "maximum_drawdown": symbol_metrics[
                "maximum_drawdown"
            ],
        }
    )


symbol_summary_frame = pd.DataFrame(
    symbol_rows
).sort_values(
    "annualized_sharpe",
    ascending=False,
)

symbol_summary_path = (
    METRICS_PATH
    / "single_symbol_sharpe_decomposition.csv"
)

symbol_summary_frame.to_csv(
    symbol_summary_path,
    index=False,
)

print(
    f"\nSymbols evaluated: {len(symbol_summary_frame)}"
)
print(
    f"Symbols meeting >= {MIN_DAYS_FOR_SHARPE_SUMMARY} days "
    f"threshold: "
    f"{symbol_summary_frame['meets_min_days'].sum()}"
)


# ============================================================
# AGGREGATE SINGLE-SYMBOL STATISTICS
# ============================================================

eligible_symbols = symbol_summary_frame[
    symbol_summary_frame["meets_min_days"]
]

eligible_sharpe = (
    eligible_symbols["annualized_sharpe"]
    .dropna()
)

single_symbol_summary = {
    "symbols_evaluated": len(symbol_summary_frame),
    "symbols_meeting_min_days": len(eligible_symbols),
    "median_single_symbol_sharpe": eligible_sharpe.median(),
    "mean_single_symbol_sharpe": eligible_sharpe.mean(),
    "std_single_symbol_sharpe": eligible_sharpe.std(ddof=1),
    "min_single_symbol_sharpe": eligible_sharpe.min(),
    "max_single_symbol_sharpe": eligible_sharpe.max(),
    "fraction_positive_sharpe": (
        eligible_sharpe > 0
    ).mean(),
    "portfolio_sharpe": portfolio_metrics["annualized_sharpe"],
}

print("\n" + "=" * 90)
print("SINGLE-SYMBOL VS PORTFOLIO SHARPE DECOMPOSITION")
print("=" * 90)

for key, value in single_symbol_summary.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key:<35} {value:.6f}")
    else:
        print(f"{key:<35} {value}")

diversification_multiple = (
    single_symbol_summary["portfolio_sharpe"]
    / single_symbol_summary["median_single_symbol_sharpe"]
    if single_symbol_summary["median_single_symbol_sharpe"]
    and single_symbol_summary["median_single_symbol_sharpe"] > 0
    else float("nan")
)

print(
    f"\nPortfolio Sharpe / median single-symbol Sharpe: "
    f"{diversification_multiple:.2f}x"
)

if eligible_sharpe.median() > 5:
    print(
        "\nWARNING: median single-symbol Sharpe itself exceeds 5. "
        "This suggests a methodological issue persists even at the "
        "single-instrument level, independent of cross-sectional "
        "diversification - continue debugging rather than trusting "
        "the portfolio number."
    )
else:
    print(
        "\nMedian single-symbol Sharpe is modest. This is "
        "consistent with the portfolio Sharpe being substantially "
        "explained by cross-sectional diversification rather than "
        "an inflated per-instrument edge."
    )


# ============================================================
# RECONCILIATION CHECK: EQUAL-WEIGHT AVERAGE VS PORTFOLIO
# ============================================================
# Decision:
# Build an independent "equal-weight average of individual-symbol
# returns" series by averaging each symbol's OWN daily return
# (computed above, in isolation) across all symbols present on that
# date - completely independent of the weighted cross-sectional
# portfolio construction used elsewhere.
#
# Why:
# If these two series (equal-weight average vs. the actual
# weight-and-position-normalized portfolio) don't roughly reconcile
# - similar shape, meaningfully correlated - that would indicate a
# bug in one of the two constructions, not just a difference in
# weighting philosophy.

all_symbol_daily_returns = pd.concat(
    symbol_daily_return_frames,
    ignore_index=True,
)

equal_weight_daily = (
    all_symbol_daily_returns
    .groupby(
        "date_id",
        as_index=False,
    )
    .agg(
        equal_weight_return=(
            "strategy_return",
            "mean",
        ),
        symbols_present=(
            "symbol_id",
            "nunique",
        ),
    )
)

reconciliation = portfolio_daily_returns.merge(
    equal_weight_daily,
    on="date_id",
    how="inner",
)

reconciliation_correlation = (
    reconciliation["strategy_return"]
    .corr(
        reconciliation["equal_weight_return"]
    )
)

equal_weight_metrics = calculate_strategy_metrics(
    reconciliation["equal_weight_return"].to_numpy()
)

print("\n" + "=" * 90)
print("RECONCILIATION: EQUAL-WEIGHT AVERAGE VS ACTUAL PORTFOLIO")
print("=" * 90)

print(
    f"Dates compared: {len(reconciliation)}"
)
print(
    f"Correlation (equal-weight vs weighted portfolio daily "
    f"returns): {reconciliation_correlation:.6f}"
)
print(
    f"Equal-weight average Sharpe: "
    f"{equal_weight_metrics['annualized_sharpe']:.6f}"
)
print(
    f"Weighted portfolio Sharpe:   "
    f"{portfolio_metrics['annualized_sharpe']:.6f}"
)

if reconciliation_correlation > 0.7:
    print(
        "\nHigh correlation - the two independent constructions "
        "broadly agree, which is a good sign neither contains a "
        "silent bug specific to its own aggregation path."
    )
else:
    print(
        "\nLOW correlation between the two constructions - this "
        "needs investigation before trusting either result."
    )


# ============================================================
# SAVE RECONCILIATION AND SUMMARY
# ============================================================

reconciliation_path = (
    METRICS_PATH
    / "single_symbol_portfolio_reconciliation.csv"
)

reconciliation.to_csv(
    reconciliation_path,
    index=False,
)

summary_frame = pd.DataFrame(
    [single_symbol_summary]
)

summary_path = (
    METRICS_PATH
    / "single_symbol_vs_portfolio_summary.csv"
)

summary_frame.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# PRINT TOP AND BOTTOM SYMBOLS
# ============================================================

display_columns = [
    "symbol_id",
    "days_available",
    "annualized_sharpe",
    "information_coefficient",
    "hit_rate",
    "cumulative_pnl",
]

print("\n" + "=" * 100)
print(
    f"TOP 10 SYMBOLS BY SHARPE "
    f"(>= {MIN_DAYS_FOR_SHARPE_SUMMARY} days)"
)
print("=" * 100)

print(
    eligible_symbols
    .sort_values("annualized_sharpe", ascending=False)
    [display_columns]
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 100)
print(
    f"BOTTOM 10 SYMBOLS BY SHARPE "
    f"(>= {MIN_DAYS_FOR_SHARPE_SUMMARY} days)"
)
print("=" * 100)

print(
    eligible_symbols
    .sort_values("annualized_sharpe", ascending=True)
    [display_columns]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 90)
print("REMINDER")
print("=" * 90)
print(
    "These remain responder-based strategy-proxy metrics on "
    "anonymized instruments with unknown transaction economics. "
    "This script's purpose is decomposition (edge vs. "
    "diversification), not producing a single headline number."
)

print("\nSaved per-symbol decomposition to:")
print(symbol_summary_path)

print("\nSaved single-symbol vs portfolio summary to:")
print(summary_path)

print("\nSaved reconciliation series to:")
print(reconciliation_path)

print("\nSingle-symbol Sharpe decomposition complete.")
