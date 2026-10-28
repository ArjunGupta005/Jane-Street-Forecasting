# ============================================================
# STRATEGY MODULE
# ============================================================
# This module converts raw forecasts into a simple trading rule
# and evaluates that rule using risk metrics.
#
# IMPORTANT CAUTION:
# responder_6 is an anonymized Jane Street target, not a literal
# tradable percentage return. Every Sharpe/drawdown/turnover number
# produced by this module is therefore a STRATEGY-PROXY metric, not
# a literal deployable portfolio Sharpe, unless a defensible mapping
# from responder units to real tradable returns is established.
#
# The correct way to describe results from this module is:
#
#     "Converted walk-forward predictions into a weighted
#      long/short strategy proxy and evaluated annualized Sharpe,
#      drawdown, and turnover."
#
# NOT:
#
#     "This strategy would achieve a Sharpe of X in production."
#
# This module builds on src/metrics.py, which already implements
# leakage-safe daily P&L aggregation (calculate_daily_strategy_returns)
# and the risk-metric report (calculate_strategy_metrics). This
# module adds position sizing and turnover, which metrics.py does
# not cover.

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from src.metrics import (
    calculate_daily_strategy_returns,
    calculate_strategy_metrics,
)


# ============================================================
# POSITION SIZING RULES
# ============================================================
# Decision:
# Convert a raw prediction into a position using one of three
# rules:
#
# sign:
#     position = sign(prediction)
#
#     Every observation gets a full-size long or short position.
#     Simple and robust to prediction miscalibration, but discards
#     all magnitude information.
#
# proportional:
#     position = prediction
#
#     Position size scales directly with predicted magnitude.
#     Uses more information than sign-only, but is sensitive to
#     outlier predictions and to whether prediction magnitude is
#     well calibrated.
#
# clipped_proportional:
#     position = clip(prediction, -c, c)
#
#     A middle ground: retains magnitude information for typical
#     predictions while preventing a small number of extreme
#     predictions from dominating exposure and turnover.
#
# Tradeoff:
# We deliberately do not tune the clip threshold c against multiple
# candidate values here. A single, data-driven choice (e.g. the 99th
# percentile of |prediction|) is used instead, to avoid quietly
# overfitting a strategy hyperparameter to this one out-of-sample
# series.

def build_positions(
    dataframe: pl.DataFrame,
    method: str,
    prediction_column: str = "prediction",
    clip_value: float | None = None,
    position_column: str = "position",
) -> pl.DataFrame:

    if method == "sign":

        position_expression = (
            pl.col(prediction_column)
            .sign()
        )

    elif method == "proportional":

        position_expression = (
            pl.col(prediction_column)
        )

    elif method == "clipped_proportional":

        if clip_value is None:
            raise ValueError(
                "clipped_proportional requires clip_value."
            )

        position_expression = (
            pl.col(prediction_column)
            .clip(
                -clip_value,
                clip_value,
            )
        )

    else:
        raise ValueError(
            "method must be one of: "
            "'sign', 'proportional', 'clipped_proportional'."
        )

    return dataframe.with_columns(
        position_expression
        .alias(position_column)
    )


# ============================================================
# TURNOVER
# ============================================================
# Decision:
# Measure turnover as the absolute change in position between
# consecutive observations for the SAME symbol, sorted
# chronologically.
#
# Why:
# A strategy with an attractive Sharpe ratio can still be
# economically meaningless if it requires flipping position on
# every observation. Turnover makes that visible before we accept
# a sizing rule based on Sharpe alone.
#
# Caveat:
# This is computed on the systematically time_id-strided
# walk-forward prediction series, not a tick-by-tick production
# feed, so it should be read as a directional comparison BETWEEN
# sizing rules on identical data, not an absolute production
# turnover figure.

def calculate_turnover(
    dataframe: pl.DataFrame,
    position_column: str = "position",
    symbol_column: str = "symbol_id",
    date_column: str = "date_id",
    time_column: str = "time_id",
) -> dict[str, float]:

    sorted_data = dataframe.sort(
        [
            symbol_column,
            date_column,
            time_column,
        ]
    )

    with_change = sorted_data.with_columns(
        (
            pl.col(position_column)
            - pl.col(position_column)
            .shift(1)
            .over(symbol_column)
        )
        .abs()
        .alias("absolute_position_change")
    )

    valid_changes = with_change.filter(
        pl.col("absolute_position_change")
        .is_not_null()
    )

    mean_absolute_position_change = (
        valid_changes["absolute_position_change"]
        .mean()
    )

    # --------------------------------------------------------
    # DAILY TURNOVER
    # --------------------------------------------------------
    # Total absolute position change per date_id, averaged across
    # the symbols active that date. This normalizes away the fact
    # that later dates in the dataset have more active symbols than
    # earlier dates.

    daily_turnover = (
        valid_changes
        .group_by(date_column)
        .agg(
            pl.col("absolute_position_change")
            .sum()
            .alias("total_position_change"),
            pl.col(symbol_column)
            .n_unique()
            .alias("active_symbols"),
        )
        .with_columns(
            (
                pl.col("total_position_change")
                / pl.col("active_symbols")
            )
            .alias("mean_daily_turnover_per_symbol")
        )
    )

    mean_daily_turnover_per_symbol = (
        daily_turnover["mean_daily_turnover_per_symbol"]
        .mean()
    )

    return {
        "mean_absolute_position_change":
            float(mean_absolute_position_change),
        "mean_daily_turnover_per_symbol":
            float(mean_daily_turnover_per_symbol),
    }


# ============================================================
# POSITION SIZING RULE DEFINITION
# ============================================================

@dataclass(frozen=True)
class PositionSizingRule:
    name: str
    method: str
    clip_value: float | None = None


# ============================================================
# EVALUATE ONE STRATEGY
# ============================================================
# This function chains together everything needed to go from raw
# predictions to a full risk-metric report for ONE position-sizing
# rule:
#
#     prediction -> position -> daily P&L proxy -> risk metrics
#                             -> turnover
#
# Decision:
# Daily P&L aggregation and the risk-metric report are delegated to
# src/metrics.py (calculate_daily_strategy_returns,
# calculate_strategy_metrics), which already implement:
#
# - prediction x responder_6 x weight per observation
# - normalization by total absolute weighted exposure per date
# - additive (not compounded) cumulative P&L, appropriate because
#   responder_6 is not a conventional percentage return
#
# This module does not reimplement that logic - it only adds
# position sizing and turnover, which metrics.py does not cover.

def evaluate_strategy(
    dataframe: pl.DataFrame,
    rule: PositionSizingRule,
    prediction_column: str = "prediction",
    target_column: str = "actual_responder_6",
    weight_column: str = "weight",
    date_column: str = "date_id",
    time_column: str = "time_id",
    symbol_column: str = "symbol_id",
    periods_per_year: int = 252,
) -> tuple[dict, pd.DataFrame]:

    positioned = build_positions(
        dataframe,
        method=rule.method,
        prediction_column=prediction_column,
        clip_value=rule.clip_value,
    )

    positioned_pd = positioned.to_pandas()

    daily_returns_frame = calculate_daily_strategy_returns(
        positioned_pd,
        date_column=date_column,
        target_column=target_column,
        position_column="position",
        weight_column=weight_column,
    )

    strategy_metrics = calculate_strategy_metrics(
        daily_returns_frame["strategy_return"].to_numpy(),
        periods_per_year=periods_per_year,
    )

    turnover_metrics = calculate_turnover(
        positioned,
        position_column="position",
        symbol_column=symbol_column,
        date_column=date_column,
        time_column=time_column,
    )

    average_absolute_position = (
        positioned["position"]
        .abs()
        .mean()
    )

    result = {
        "strategy": rule.name,
        "method": rule.method,
        "clip_value": rule.clip_value,
        "average_absolute_position": float(average_absolute_position),
        **strategy_metrics,
        **turnover_metrics,
    }

    return result, daily_returns_frame


# ============================================================
# WHY THE FUNCTIONS ABOVE PRODUCE AN INFLATED SHARPE
# ============================================================
# evaluate_strategy() above averages roughly 39 symbols x ~20
# intraday time_id snapshots into a single date_id observation.
# Averaging hundreds of quasi-independent, slightly-positive-
# expectation bets collapses daily variance via the law of large
# numbers (sigma_average ~ sigma / sqrt(N)) while the mean survives,
# which is exactly the mechanism that produced an unannualized
# Sharpe of 20-24 and a 97% daily hit rate - both far outside any
# credible range for a real trading strategy, despite the
# underlying forecasting model only having ~53% directional
# accuracy at the observation level.
#
# We deliberately do NOT delete evaluate_strategy() or
# scripts/20_evaluate_walk_forward_strategy.py. They stay in the
# project as a documented example of exactly this pitfall - the
# functions below are the corrected replacement used for any
# reported Sharpe/drawdown/turnover figures.


# ============================================================
# PRIMARY BACKTEST: ONE POSITION PER (date_id, symbol_id)
# ============================================================
# Decision:
# Collapse intraday observations to a single symbol-day value
# BEFORE any position or P&L is computed, so each symbol
# contributes exactly one bet per day rather than ~20.
#
#     pbar_{s,d} = mean_t( prediction_{s,d,t} )
#     ybar_{s,d} = mean_t( actual_responder_6_{s,d,t} )
#     wbar_{s,d} = mean_t( weight_{s,d,t} )      <- MEAN, not sum
#
# Why mean and not sum for weight:
# Summing intraday weights would make a symbol observed at more
# time_id snapshots that day mechanically more influential in the
# cross-sectional portfolio, which conflates "how important is this
# symbol" with "how many times did we happen to sample it." The
# mean keeps weight interpretable as a per-observation importance
# figure, consistent with how weight is used everywhere else in
# this project.

def aggregate_symbol_day(
    dataframe: pl.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "actual_responder_6",
    weight_column: str = "weight",
    date_column: str = "date_id",
    symbol_column: str = "symbol_id",
) -> pl.DataFrame:

    return (
        dataframe
        .group_by(
            [
                date_column,
                symbol_column,
            ]
        )
        .agg(
            pl.col(prediction_column)
            .mean()
            .alias(prediction_column),

            pl.col(target_column)
            .mean()
            .alias(target_column),

            pl.col(weight_column)
            .mean()
            .alias(weight_column),

            pl.len()
            .alias("intraday_observation_count"),
        )
        .sort(
            [
                date_column,
                symbol_column,
            ]
        )
    )


# ============================================================
# SYMBOL-DAY TURNOVER
# ============================================================
# Decision:
# Compare each symbol's position to that SAME symbol's position on
# its most recent PRIOR AVAILABLE date, not a literal date_id - 1,
# and only include a symbol in a given date's turnover once both
# the current and previous positions actually exist.
#
# Turnover_d = sum_s |p_{s,d} - p_{s,d-1}| / sum_s |p_{s,d-1}|
#
# then averaged across all dates that had at least one symbol with
# a valid prior position.
#
# Why "most recent available" rather than date_id - 1:
# EDA and feature-engineering testing both confirmed individual
# symbols have genuine day-level gaps in the dataset. Treating a
# gap as "the position went from p to 0 and back" would fabricate
# turnover that never happened; treating it as "compare to the last
# time we actually had a position for this symbol" is the leakage-
# safe, gap-aware convention already used for lag features
# elsewhere in this project (src/feature_engineering.py).

def calculate_symbol_day_turnover_series(
    dataframe: pl.DataFrame,
    position_column: str = "position",
    symbol_column: str = "symbol_id",
    date_column: str = "date_id",
) -> pl.DataFrame:
    """
    Per-date turnover series, one row per date_id that has at least
    one symbol with a valid prior position. Extracted as its own
    function (rather than only an aggregate dict) so downstream
    consumers - such as a transaction-cost sensitivity analysis -
    can subtract a per-date cost from that date's return, not just
    read a single summary number.
    """

    sorted_data = dataframe.sort(
        [
            symbol_column,
            date_column,
        ]
    )

    with_previous = sorted_data.with_columns(
        pl.col(position_column)
        .shift(1)
        .over(
            symbol_column,
            order_by=date_column,
        )
        .alias("previous_position")
    )

    valid_pairs = with_previous.filter(
        pl.col("previous_position")
        .is_not_null()
    )

    daily_turnover = (
        valid_pairs
        .with_columns(
            (
                pl.col(position_column)
                - pl.col("previous_position")
            )
            .abs()
            .alias("absolute_position_change"),

            pl.col("previous_position")
            .abs()
            .alias("absolute_previous_position"),
        )
        .group_by(date_column)
        .agg(
            pl.col("absolute_position_change")
            .sum()
            .alias("total_position_change"),

            pl.col("absolute_previous_position")
            .sum()
            .alias("total_previous_exposure"),
        )
        .with_columns(
            pl.when(
                pl.col("total_previous_exposure") > 0
            )
            .then(
                pl.col("total_position_change")
                / pl.col("total_previous_exposure")
            )
            .otherwise(None)
            .alias("daily_turnover")
        )
        .drop_nulls("daily_turnover")
        .sort(date_column)
    )

    return daily_turnover


def calculate_symbol_day_turnover(
    dataframe: pl.DataFrame,
    position_column: str = "position",
    symbol_column: str = "symbol_id",
    date_column: str = "date_id",
) -> dict[str, float]:

    daily_turnover = calculate_symbol_day_turnover_series(
        dataframe,
        position_column=position_column,
        symbol_column=symbol_column,
        date_column=date_column,
    )

    if daily_turnover.height == 0:
        return {
            "mean_daily_turnover": float("nan"),
            "turnover_dates": 0,
        }

    return {
        "mean_daily_turnover":
            float(
                daily_turnover["daily_turnover"].mean()
            ),
        "turnover_dates":
            int(daily_turnover.height),
    }


# ============================================================
# EVALUATE THE PRIMARY SYMBOL-DAY STRATEGY
# ============================================================
# Full pipeline:
#
#   walk-forward predictions
#         |
#   aggregate_symbol_day()          <- one row per (date_id, symbol_id)
#         |
#   build_positions()               <- sign / proportional / clipped
#         |
#   calculate_daily_strategy_returns()   <- reused from src/metrics.py
#         |                             (now grouping symbol-days, not
#         |                              intraday rows, into each date)
#   calculate_strategy_metrics()    <- annualized Sharpe/drawdown/etc.
#         |
#   calculate_symbol_day_turnover()

def evaluate_daily_symbol_strategy(
    dataframe: pl.DataFrame,
    rule: PositionSizingRule,
    prediction_column: str = "prediction",
    target_column: str = "actual_responder_6",
    weight_column: str = "weight",
    date_column: str = "date_id",
    symbol_column: str = "symbol_id",
    periods_per_year: int = 252,
) -> tuple[dict, pd.DataFrame, pl.DataFrame]:

    symbol_day_data = aggregate_symbol_day(
        dataframe,
        prediction_column=prediction_column,
        target_column=target_column,
        weight_column=weight_column,
        date_column=date_column,
        symbol_column=symbol_column,
    )

    positioned = build_positions(
        symbol_day_data,
        method=rule.method,
        prediction_column=prediction_column,
        clip_value=rule.clip_value,
    )

    positioned_pd = positioned.to_pandas()

    daily_returns_frame = calculate_daily_strategy_returns(
        positioned_pd,
        date_column=date_column,
        target_column=target_column,
        position_column="position",
        weight_column=weight_column,
    )

    strategy_metrics = calculate_strategy_metrics(
        daily_returns_frame["strategy_return"].to_numpy(),
        periods_per_year=periods_per_year,
    )

    turnover_metrics = calculate_symbol_day_turnover(
        positioned,
        position_column="position",
        symbol_column=symbol_column,
        date_column=date_column,
    )

    average_absolute_position = (
        positioned["position"]
        .abs()
        .mean()
    )

    result = {
        "strategy": rule.name,
        "method": rule.method,
        "clip_value": rule.clip_value,
        "symbol_days": positioned.height,
        "average_absolute_position": float(average_absolute_position),
        **strategy_metrics,
        **turnover_metrics,
    }

    return result, daily_returns_frame, positioned


# ============================================================
# ROBUSTNESS CHECK: (date_id, time_id) SNAPSHOT CROSS-SECTION
# ============================================================
# Decision:
# Compute a cross-sectional strategy return at every observed
# (date_id, time_id) snapshot, WITHOUT first collapsing intraday
# observations. This is the direct opposite aggregation choice from
# the primary backtest above, used purely as a robustness check on
# whether the daily-symbol result is an artifact of that specific
# aggregation.
#
#     R_t = sum_s( w_{s,t} * position_{s,t} * y_{s,t} )
#           / sum_s( w_{s,t} * |position_{s,t}| )
#
# CRITICAL:
# This series is intentionally NOT annualized. Jane Street's own
# documentation states time_id spacing is not guaranteed to
# represent equal real elapsed time, so multiplying by
# sqrt(252 * snapshots_per_day) would fabricate a false
# equal-spacing assumption on top of an already-approximate proxy.
# We report only unannualized descriptive statistics instead.

def calculate_snapshot_strategy_returns(
    dataframe: pl.DataFrame,
    position_column: str = "position",
    target_column: str = "actual_responder_6",
    weight_column: str = "weight",
    date_column: str = "date_id",
    time_column: str = "time_id",
) -> pl.DataFrame:

    snapshot_data = dataframe.with_columns(
        (
            pl.col(position_column)
            * pl.col(target_column)
            * pl.col(weight_column)
        )
        .alias("weighted_pnl"),

        (
            pl.col(position_column)
            .abs()
            * pl.col(weight_column)
        )
        .alias("absolute_weighted_exposure"),
    )

    snapshot_returns = (
        snapshot_data
        .group_by(
            [
                date_column,
                time_column,
            ]
        )
        .agg(
            pl.col("weighted_pnl")
            .sum()
            .alias("total_pnl"),

            pl.col("absolute_weighted_exposure")
            .sum()
            .alias("total_exposure"),
        )
        .sort(
            [
                date_column,
                time_column,
            ]
        )
        .with_columns(
            pl.when(
                pl.col("total_exposure") > 0
            )
            .then(
                pl.col("total_pnl")
                / pl.col("total_exposure")
            )
            .otherwise(0.0)
            .alias("snapshot_return")
        )
    )

    return snapshot_returns


# ============================================================
# UNANNUALIZED SNAPSHOT SUMMARY
# ============================================================
# Deliberately parallel to, but distinct from,
# calculate_strategy_metrics(): no annualization factor is applied
# anywhere in this function.

def summarize_snapshot_returns(
    snapshot_returns: np.ndarray,
) -> dict[str, float]:

    snapshot_returns = np.asarray(
        snapshot_returns,
        dtype=np.float64,
    )

    snapshot_returns = snapshot_returns[
        np.isfinite(snapshot_returns)
    ]

    if len(snapshot_returns) == 0:
        raise ValueError(
            "Snapshot return series is empty."
        )

    mean_return = float(
        np.mean(snapshot_returns)
    )

    std_return = (
        float(
            np.std(
                snapshot_returns,
                ddof=1,
            )
        )
        if len(snapshot_returns) > 1
        else float("nan")
    )

    mean_std_ratio = (
        mean_return / std_return
        if std_return
        and std_return > 0
        else float("nan")
    )

    return {
        "number_of_snapshots": int(len(snapshot_returns)),
        "mean_snapshot_return": mean_return,
        "std_snapshot_return": std_return,
        "unannualized_mean_std_ratio": mean_std_ratio,
        "hit_rate": float(
            np.mean(snapshot_returns > 0)
        ),
        "cumulative_pnl": float(
            np.sum(snapshot_returns)
        ),
    }
