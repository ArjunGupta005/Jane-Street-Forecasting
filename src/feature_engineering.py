# ============================================================
# FEATURE ENGINEERING MODULE
# ============================================================
# This module builds engineered feature families on top of the
# raw Jane Street predictors.
#
# The raw-feature LightGBM baseline (weighted zero-mean R² =
# 0.01353) is the control point every engineered feature group
# must be measured against.
#
# We build feature families as separate, composable functions so
# scripts/11_train_lightgbm_with_features.py can test them
# independently (raw vs +cross-sectional vs +rolling vs +both) and
# attribute any score improvement to a specific idea rather than a
# tangle of simultaneous changes.
#
# CRITICAL LEAKAGE RULE (per-symbol rolling/lag features):
#
#     Wrong:
#         rolling_mean = current_and_previous_values.mean()
#
#     Correct:
#         historical_values = feature.shift(1)
#         rolling_mean = historical_values.rolling(...)
#
# Every rolling/lag feature in this module is built from a
# shift(1) series so that a given date_id's engineered features
# never include that date_id's own observations.


import polars as pl


# ============================================================
# CORE FEATURE SET FOR ENGINEERING
# ============================================================
# Decision:
# Restrict cross-sectional and rolling feature engineering to a
# compact set of 16 raw features rather than all 79.
#
# Why:
# These features topped both the linear-correlation ranking and
# the LightGBM gain ranking from the raw baseline experiments.
# Engineering all 79 predictors would multiply RAM and runtime
# cost for features already shown to carry little individual
# signal.
#
# Tradeoff:
# A feature ranked low individually might still contribute through
# engineered cross-sectional or rolling structure. We can expand
# this list later if this compact experiment shows engineered
# features are valuable in general.

CORE_ENGINEERING_FEATURES = [
    "feature_06",
    "feature_04",
    "feature_07",
    "feature_59",
    "feature_36",
    "feature_49",
    "feature_60",
    "feature_38",
    "feature_01",
    "feature_52",
    "feature_58",
    "feature_05",
    "feature_68",
    "feature_02",
    "feature_47",
    "feature_08",
]


# ============================================================
# ROLLING WINDOW SIZES (IN TRADING DAYS)
# ============================================================
# Decision:
# Use two window lengths, 5 and 10 days.
#
# Why:
# date_id is the fundamental chronological unit used throughout
# this project (EDA, temporal CV). Short windows are appropriate
# here because early-history symbols have limited prior days
# available, and because this is a first, compact experiment
# rather than a tuned final feature set.

ROLLING_WINDOWS = [5, 10]


# ============================================================
# GROUP 1: CROSS-SECTIONAL MARKET FEATURES
# ============================================================
# Decision:
# For each (date_id, time_id) snapshot, compute the mean of each
# core feature across every symbol present at that exact moment,
# and each symbol's deviation from that snapshot mean.
#
# Why this is NOT look-ahead leakage:
# Jane Street's real evaluation API delivers test observations in
# batches keyed by (date_id, time_id) across every symbol
# simultaneously. So at inference time, every symbol's raw feature
# values for the CURRENT snapshot are already available before any
# prediction for that snapshot is required. Aggregating across
# symbols within the same snapshot uses no future date_id/time_id
# and no responder values, so it is contemporaneous cross-sectional
# information rather than temporal look-ahead.
#
# Tradeoff:
# If the evaluation API ever changed to deliver data one symbol at
# a time instead of a full cross-sectional batch, this feature
# family would need to be revisited.

def add_cross_sectional_features(
    dataframe: pl.DataFrame,
    features: list[str] = CORE_ENGINEERING_FEATURES,
    date_column: str = "date_id",
    time_column: str = "time_id",
) -> pl.DataFrame:

    market_mean_expressions = [
        pl.col(feature)
        .mean()
        .over([date_column, time_column])
        .alias(f"{feature}_market_mean")
        for feature in features
    ]

    processed = dataframe.with_columns(
        market_mean_expressions
    )

    deviation_expressions = [
        (
            pl.col(feature)
            - pl.col(f"{feature}_market_mean")
        )
        .alias(f"{feature}_market_dev")
        for feature in features
    ]

    processed = processed.with_columns(
        deviation_expressions
    )

    return processed


def cross_sectional_feature_names(
    features: list[str] = CORE_ENGINEERING_FEATURES,
) -> list[str]:

    names = []

    for feature in features:
        names.append(f"{feature}_market_mean")
        names.append(f"{feature}_market_dev")

    return names


# ============================================================
# GROUP 2: PER-SYMBOL ROLLING FEATURES
# ============================================================
# Decision:
# Collapse raw intraday rows to one row per (symbol_id, date_id)
# using the mean of each core feature across that day's time_id
# observations, BEFORE computing any rolling/lag statistic.
#
# Why:
# date_id is the chronological unit used everywhere else in this
# project. Building rolling windows over days per symbol keeps the
# leakage logic easy to reason about and cheap to compute, since
# the resulting table is only (symbols x dates) in size rather than
# tens of millions of intraday rows.

def build_symbol_daily_features(
    training_scan: pl.LazyFrame,
    features: list[str] = CORE_ENGINEERING_FEATURES,
    date_column: str = "date_id",
    symbol_column: str = "symbol_id",
) -> pl.DataFrame:
    """
    Collapse a (possibly lazy, possibly date-filtered) row-level
    scan to one row per (symbol_id, date_id), using the mean of
    each core feature across that day's intraday observations.

    This intentionally uses the FULL set of intraday rows for the
    requested date range (not a time_id-strided sample), so the
    resulting daily mean is an accurate summary of each symbol's
    day rather than an artifact of whatever row stride a particular
    model-training experiment happens to use.
    """

    daily_expressions = [
        pl.col(feature).mean().alias(feature)
        for feature in features
    ]

    symbol_daily = (
        training_scan
        .group_by([symbol_column, date_column])
        .agg(daily_expressions)
        .sort([symbol_column, date_column])
        .collect()
    )

    return symbol_daily


def add_symbol_rolling_features(
    symbol_daily: pl.DataFrame,
    features: list[str] = CORE_ENGINEERING_FEATURES,
    windows: list[int] = ROLLING_WINDOWS,
    date_column: str = "date_id",
    symbol_column: str = "symbol_id",
) -> pl.DataFrame:
    """
    Given a per-(symbol_id, date_id) daily feature table, build
    leakage-safe historical features: lag_1, lag_2, a 1-day delta,
    and rolling mean/std/z-score over each requested window.

    CRITICAL:
    Every rolling/lag column is derived from a shift(1) series (via
    lag_1), so a given date_id's engineered features are built
    exclusively from STRICTLY earlier days for that symbol. The
    current day's own observation never contributes to its own
    engineered feature.
    """

    processed = symbol_daily

    for feature in features:

        lag_1_name = f"{feature}_lag_1"
        lag_2_name = f"{feature}_lag_2"
        delta_name = f"{feature}_delta_1"

        processed = processed.with_columns(
            pl.col(feature)
            .shift(1)
            .over(
                symbol_column,
                order_by=date_column,
            )
            .alias(lag_1_name)
        )

        processed = processed.with_columns(
            pl.col(feature)
            .shift(2)
            .over(
                symbol_column,
                order_by=date_column,
            )
            .alias(lag_2_name)
        )

        processed = processed.with_columns(
            (
                pl.col(lag_1_name)
                - pl.col(lag_2_name)
            ).alias(delta_name)
        )

        for window in windows:

            rolling_mean_name = (
                f"{feature}_rolling_mean_{window}"
            )
            rolling_std_name = (
                f"{feature}_rolling_std_{window}"
            )
            rolling_z_name = (
                f"{feature}_rolling_z_{window}"
            )

            # ------------------------------------------------
            # Rolling statistics are computed on the ALREADY
            # shifted lag_1 series, not on the raw feature.
            #
            # This is the mechanical enforcement of the shift-
            # before-rolling rule: rolling_mean(lag_1) at date d
            # only ever sees days strictly before d.
            # ------------------------------------------------

            processed = processed.with_columns(
                pl.col(lag_1_name)
                .rolling_mean(
                    window_size=window,
                    min_samples=window,
                )
                .over(
                    symbol_column,
                    order_by=date_column,
                )
                .alias(rolling_mean_name)
            )

            processed = processed.with_columns(
                pl.col(lag_1_name)
                .rolling_std(
                    window_size=window,
                    min_samples=window,
                )
                .over(
                    symbol_column,
                    order_by=date_column,
                )
                .alias(rolling_std_name)
            )

            processed = processed.with_columns(
                pl.when(
                    pl.col(rolling_std_name) > 0
                )
                .then(
                    (
                        pl.col(lag_1_name)
                        - pl.col(rolling_mean_name)
                    )
                    / pl.col(rolling_std_name)
                )
                .otherwise(None)
                .alias(rolling_z_name)
            )

    return processed


def merge_symbol_rolling_features(
    raw_dataframe: pl.DataFrame,
    symbol_daily_with_rolling: pl.DataFrame,
    date_column: str = "date_id",
    symbol_column: str = "symbol_id",
    base_features: list[str] = CORE_ENGINEERING_FEATURES,
) -> pl.DataFrame:
    """
    Left-join the leakage-safe per-(symbol_id, date_id) rolling/lag
    features onto a raw row-level dataframe.

    Because every joined column was built from a shift(1) series,
    broadcasting one (symbol, date) row's engineered values across
    every intraday time_id row of that same day is safe: those
    values reflect only strictly earlier days for that symbol, not
    anything from the current day.

    The raw daily-mean columns used internally to build the rolling
    features (i.e. base_features) are deliberately excluded from
    the join, since raw_dataframe already has intraday versions of
    those same columns and joining would create a name collision.
    """

    engineered_columns = [
        column
        for column in symbol_daily_with_rolling.columns
        if column not in (symbol_column, date_column)
        and column not in base_features
    ]

    join_frame = symbol_daily_with_rolling.select(
        [symbol_column, date_column] + engineered_columns
    )

    merged = raw_dataframe.join(
        join_frame,
        on=[symbol_column, date_column],
        how="left",
    )

    return merged


def build_symbol_rolling_feature_table(
    training_scan: pl.LazyFrame,
    start_date: int,
    end_date: int,
    features: list[str] = CORE_ENGINEERING_FEATURES,
    windows: list[int] = ROLLING_WINDOWS,
    date_column: str = "date_id",
    symbol_column: str = "symbol_id",
) -> pl.DataFrame:
    """
    One-shot convenience helper.

    Scans [start_date, end_date] in full (not row-strided) to
    compute accurate per-symbol daily means, then builds leakage-
    safe lag/rolling features on that daily series.

    Returns a small (symbols x dates) table ready to be joined via
    merge_symbol_rolling_features() onto any row-level sample that
    covers the same date range.
    """

    date_filtered_scan = training_scan.filter(
        (pl.col(date_column) >= start_date)
        & (pl.col(date_column) <= end_date)
    )

    symbol_daily = build_symbol_daily_features(
        date_filtered_scan,
        features=features,
        date_column=date_column,
        symbol_column=symbol_column,
    )

    symbol_daily_with_rolling = add_symbol_rolling_features(
        symbol_daily,
        features=features,
        windows=windows,
        date_column=date_column,
        symbol_column=symbol_column,
    )

    return symbol_daily_with_rolling


def rolling_feature_names(
    features: list[str] = CORE_ENGINEERING_FEATURES,
    windows: list[int] = ROLLING_WINDOWS,
) -> list[str]:

    names = []

    for feature in features:

        names.append(f"{feature}_lag_1")
        names.append(f"{feature}_lag_2")
        names.append(f"{feature}_delta_1")

        for window in windows:
            names.append(f"{feature}_rolling_mean_{window}")
            names.append(f"{feature}_rolling_std_{window}")
            names.append(f"{feature}_rolling_z_{window}")

    return names


# ============================================================
# COMPACT TEMPORAL FEATURE SET
# ============================================================
# Earlier experiments added a large block of rolling/lag features
# and reduced out-of-sample performance.
#
# Decision:
# Reintroduce temporal context using only a small set of strong
# raw features and three deliberately distinct transformations.
#
# Why:
# This tests whether temporal information itself is useful without
# overwhelming LightGBM with a large number of redundant columns.

COMPACT_TEMPORAL_BASE_FEATURES = [
    "feature_04",
    "feature_06",
    "feature_07",
    "feature_59",
    "feature_36",
    "feature_49",
    "feature_60",
    "feature_38",
]


# ============================================================
# BUILD DAILY SYMBOL FEATURE HISTORY
# ============================================================
# For every symbol and date, calculate the daily mean of each
# selected feature.
#
# IMPORTANT:
# These daily means are NOT directly joined back onto observations
# from the same date.
#
# Doing so would cause look-ahead bias because an observation at an
# early time_id would gain information from later time_ids in the
# same day.
#
# Instead, the daily summaries become available only to subsequent
# days.

def build_compact_temporal_daily_table(
    training_data: pl.LazyFrame,
    feature_columns: list[str] = COMPACT_TEMPORAL_BASE_FEATURES,
    rolling_window: int = 20,
) -> pl.LazyFrame:

    # --------------------------------------------------------
    # DAILY SYMBOL AGGREGATION
    # --------------------------------------------------------
    # Each row represents one completed symbol-day.

    daily_symbol_features = (
        training_data
        .select(
            [
                "date_id",
                "symbol_id",
            ]
            + feature_columns
        )
        .group_by(
            [
                "symbol_id",
                "date_id",
            ]
        )
        .agg(
            [
                pl.col(feature)
                .mean()
                .alias(
                    f"{feature}_daily_mean"
                )
                for feature in feature_columns
            ]
        )
        .sort(
            [
                "symbol_id",
                "date_id",
            ]
        )
    )

    # ========================================================
    # PREVIOUS AVAILABLE SYMBOL-DAY
    # ========================================================
    # Decision:
    # Use the most recent actually observed prior day for a symbol.
    #
    # Why:
    # EDA/testing showed that symbols do not necessarily appear on
    # every date_id.
    #
    # A positional shift therefore means:
    #
    # "most recent completed observation for this symbol"
    #
    # rather than incorrectly assuming that date_id - 1 always
    # exists.
    #
    # This remains leakage-safe because every shifted value comes
    # strictly from an earlier observed date.

    lag_expressions = []

    for feature in feature_columns:

        daily_column = (
            f"{feature}_daily_mean"
        )

        lag_expressions.append(
            pl.col(daily_column)
            .shift(1)
            .over(
                "symbol_id",
                order_by="date_id",
            )
            .alias(
                f"{feature}_previous_day_mean"
            )
        )

    daily_symbol_features = (
        daily_symbol_features
        .with_columns(
            lag_expressions
        )
    )

    # ========================================================
    # ROLLING 20-DAY HISTORICAL MEAN
    # ========================================================
    # Decision:
    # Calculate a wider 20-observation rolling average.
    #
    # Why:
    # The earlier 5/10-day feature block may have been too noisy
    # for a nonstationary financial target.
    #
    # A 20-day window provides slower-moving regime context while
    # remaining short enough to adapt.
    #
    # CRITICAL LEAKAGE RULE:
    #
    # We roll the PREVIOUS-DAY column, not the current day's daily
    # mean.
    #
    # Therefore the value available on date d contains information
    # only from completed symbol-days before date d.

    rolling_expressions = []

    for feature in feature_columns:

        previous_day_column = (
            f"{feature}_previous_day_mean"
        )

        rolling_expressions.append(
            pl.col(previous_day_column)
            .rolling_mean(
                window_size=rolling_window,
                min_samples=1,
            )
            .over(
                "symbol_id",
                order_by="date_id",
            )
            .alias(
                f"{feature}_rolling_{rolling_window}_prior_mean"
            )
        )

    daily_symbol_features = (
        daily_symbol_features
        .with_columns(
            rolling_expressions
        )
    )

    # ========================================================
    # KEEP ONLY FEATURES AVAILABLE TO FUTURE OBSERVATIONS
    # ========================================================

    output_columns = [
        "date_id",
        "symbol_id",
    ]

    for feature in feature_columns:
        output_columns.extend(
            [
                f"{feature}_previous_day_mean",
                f"{feature}_rolling_{rolling_window}_prior_mean",
            ]
        )

    return (
        daily_symbol_features
        .select(output_columns)
    )


# ============================================================
# ADD COMPACT TEMPORAL FEATURES TO OBSERVATION DATA
# ============================================================
# This function joins historical symbol state onto the current
# row and creates a current-vs-history delta.
#
# Example:
#
# feature_06_delta_previous_day
#
# =
#
# current feature_06
# -
# previous completed day's mean feature_06
#
# This gives the model a simple "surprise" or deviation measure
# without using future observations.

def add_compact_temporal_features(
    observation_data: pl.LazyFrame,
    daily_temporal_table: pl.LazyFrame,
    feature_columns: list[str] = COMPACT_TEMPORAL_BASE_FEATURES,
) -> pl.LazyFrame:

    enriched_data = (
        observation_data
        .join(
            daily_temporal_table,
            on=[
                "date_id",
                "symbol_id",
            ],
            how="left",
        )
    )

    # ========================================================
    # CURRENT VALUE MINUS PREVIOUS-DAY STATE
    # ========================================================

    delta_expressions = []

    for feature in feature_columns:

        previous_day_column = (
            f"{feature}_previous_day_mean"
        )

        delta_expressions.append(
            (
                pl.col(feature)
                - pl.col(previous_day_column)
            )
            .alias(
                f"{feature}_delta_previous_day"
            )
        )

    enriched_data = (
        enriched_data
        .with_columns(
            delta_expressions
        )
    )

    return enriched_data


# ============================================================
# COMPACT TEMPORAL FEATURE NAMES
# ============================================================

def get_compact_temporal_feature_names(
    feature_columns: list[str] = COMPACT_TEMPORAL_BASE_FEATURES,
    rolling_window: int = 20,
) -> list[str]:

    output_features = []

    for feature in feature_columns:
        output_features.extend(
            [
                f"{feature}_previous_day_mean",
                f"{feature}_delta_previous_day",
                f"{feature}_rolling_{rolling_window}_prior_mean",
            ]
        )

    return output_features
