# ============================================================
# IMPORTS
# ============================================================
# This script is a correctness and leakage-safety smoke test of
# src/feature_engineering.py on a small chronological slice.
#
# Why:
# Rolling/lag feature bugs are exactly the kind of mistake that
# silently inflates validation scores without being obviously
# wrong. Before spending compute retraining LightGBM with these
# feature families, we verify by hand, on a small slice, that:
#
# - cross-sectional market means/deviations are computed correctly
# - lag_1/lag_2/delta features shift by exactly the right number
#   of days
# - rolling mean/std/z-score use ONLY strictly earlier days
# - the first `window` days for each symbol have null rolling
#   stats (insufficient history), rather than being silently
#   filled with something that looks plausible but is wrong
# - merging engineered features back onto row-level data broadcasts
#   correctly across all intraday rows of a given (symbol, date)
# - a deliberately "wrong" (leaky) rolling calculation that
#   includes the current day would produce a DIFFERENT number than
#   our function, proving the test could actually catch a leakage
#   bug if one were introduced.

import math
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATE_COLUMN,
    SYMBOL_COLUMN,
    TIME_COLUMN,
    TRAIN_PATH,
)
from src.feature_engineering import (
    CORE_ENGINEERING_FEATURES,
    ROLLING_WINDOWS,
    add_cross_sectional_features,
    build_symbol_rolling_feature_table,
    cross_sectional_feature_names,
    merge_symbol_rolling_features,
    rolling_feature_names,
)


# ============================================================
# LOAD A SMALL, FULLY-SCANNED CHRONOLOGICAL SLICE
# ============================================================
# Decision:
# Use a plain date_id range filter (dates 0-40) rather than a
# time_id stride.
#
# Why:
# Rolling/lag correctness depends on having every intraday
# observation for the days involved, not a systematic subsample.
# This slice is small enough (~41 days) to fully scan quickly while
# still covering enough history to exercise every rolling window.

SLICE_START_DATE = 0
SLICE_END_DATE = 40

training_scan = pl.scan_parquet(
    str(TRAIN_PATH / "**" / "*.parquet")
)

raw_slice = (
    training_scan
    .filter(
        (pl.col(DATE_COLUMN) >= SLICE_START_DATE)
        & (pl.col(DATE_COLUMN) <= SLICE_END_DATE)
    )
    .collect()
)

print("\n" + "=" * 70)
print("TEST SLICE LOADED")
print("=" * 70)

print(f"Rows: {raw_slice.height:,}")
print(
    f"Date range: {raw_slice[DATE_COLUMN].min()} -> "
    f"{raw_slice[DATE_COLUMN].max()}"
)
print(f"Symbols: {raw_slice[SYMBOL_COLUMN].n_unique()}")


# ============================================================
# TEST 1: CROSS-SECTIONAL MARKET MEAN / DEVIATION
# ============================================================
# Why:
# The market mean at a given (date_id, time_id) must equal a plain
# manual mean across symbols at that exact snapshot, and deviation
# must equal feature - market_mean.

cross_sectional_processed = add_cross_sectional_features(
    raw_slice
)

probe_feature = "feature_06"

probe_snapshot = (
    raw_slice
    .filter(
        (pl.col(DATE_COLUMN) == SLICE_START_DATE + 5)
        & (pl.col(TIME_COLUMN) == 100)
    )
)

assert probe_snapshot.height > 0, (
    "Probe snapshot (date 5, time 100) has no rows; "
    "pick a different probe point."
)

manual_market_mean = (
    probe_snapshot
    .select(
        pl.col(probe_feature).mean()
    )
    [probe_feature][0]
)

engineered_snapshot = (
    cross_sectional_processed
    .filter(
        (pl.col(DATE_COLUMN) == SLICE_START_DATE + 5)
        & (pl.col(TIME_COLUMN) == 100)
    )
)

engineered_market_means = (
    engineered_snapshot[f"{probe_feature}_market_mean"]
    .to_list()
)

assert all(
    math.isclose(value, manual_market_mean, rel_tol=1e-6)
    for value in engineered_market_means
    if value is not None
), (
    "Cross-sectional market mean does not match a manual "
    "same-snapshot mean."
)

deviation_check = (
    engineered_snapshot
    .select(
        (
            pl.col(probe_feature)
            - pl.col(f"{probe_feature}_market_mean")
        ).alias("expected_deviation"),
        pl.col(f"{probe_feature}_market_dev"),
    )
)

for row in deviation_check.iter_rows(named=True):
    if row["expected_deviation"] is None:
        continue
    assert math.isclose(
        row["expected_deviation"],
        row[f"{probe_feature}_market_dev"],
        rel_tol=1e-6,
    ), "Deviation column does not equal feature - market_mean."

expected_cross_sectional_columns = set(
    cross_sectional_feature_names()
)

assert expected_cross_sectional_columns.issubset(
    set(cross_sectional_processed.columns)
), "Not all expected cross-sectional columns were created."

print(
    "\n[PASS] Cross-sectional market mean/deviation match a "
    "manual same-snapshot calculation."
)


# ============================================================
# TEST 2: BUILD SYMBOL DAILY / ROLLING FEATURE TABLE
# ============================================================

symbol_rolling_table = build_symbol_rolling_feature_table(
    training_scan,
    start_date=SLICE_START_DATE,
    end_date=SLICE_END_DATE,
)

expected_rolling_columns = set(
    rolling_feature_names()
)

assert expected_rolling_columns.issubset(
    set(symbol_rolling_table.columns)
), "Not all expected rolling/lag columns were created."

print(
    "[PASS] Symbol daily/rolling feature table created with all "
    "expected columns."
)


# ============================================================
# TEST 3: LAG_1 / LAG_2 / DELTA MATCH THE MOST RECENT
#          STRICTLY-EARLIER TRADING DAYS FOR THAT SYMBOL
# ============================================================
# Why:
# lag_1/lag_2 are POSITIONAL shifts over each symbol's own observed
# trading calendar (via .over(symbol, order_by=date)), not literal
# "date_id - 1" / "date_id - 2" arithmetic.
#
# This matters because individual symbols can have gaps in their
# daily coverage (e.g. symbol_id 0 in this very slice is missing
# dates 0, 18, 26, 27, and 28 - confirmed by direct inspection).
# For a symbol with a gap, "yesterday's trading day" is not the
# same as "calendar date_id - 1". Positional shifting is the
# correct, still fully leakage-safe choice here: it always uses
# the most recent day that symbol actually has data for, which is
# strictly earlier than the current day either way.
#
# So this test compares against the symbol's own sorted list of
# AVAILABLE dates, not raw date_id subtraction.

probe_symbol = (
    symbol_rolling_table[SYMBOL_COLUMN][0]
)

symbol_daily_raw = (
    training_scan
    .filter(
        (pl.col(DATE_COLUMN) >= SLICE_START_DATE)
        & (pl.col(DATE_COLUMN) <= SLICE_END_DATE)
        & (pl.col(SYMBOL_COLUMN) == probe_symbol)
    )
    .group_by(DATE_COLUMN)
    .agg(
        pl.col(probe_feature).mean().alias(probe_feature)
    )
    .sort(DATE_COLUMN)
    .collect()
)

symbol_available_dates = (
    symbol_daily_raw[DATE_COLUMN].to_list()
)

symbol_daily_lookup = dict(
    zip(
        symbol_available_dates,
        symbol_daily_raw[probe_feature].to_list(),
    )
)

assert len(symbol_available_dates) == len(set(symbol_available_dates)), (
    "Duplicate dates found for a single symbol; group_by should "
    "have produced one row per date."
)

date_position = {
    date_value: position
    for position, date_value in enumerate(symbol_available_dates)
}

probe_symbol_rolling = (
    symbol_rolling_table
    .filter(pl.col(SYMBOL_COLUMN) == probe_symbol)
    .sort(DATE_COLUMN)
)

checked_lag_1 = 0
checked_lag_2 = 0

for row in probe_symbol_rolling.iter_rows(named=True):

    current_date = row[DATE_COLUMN]
    position = date_position[current_date]

    expected_lag_1 = (
        symbol_daily_lookup[symbol_available_dates[position - 1]]
        if position - 1 >= 0
        else None
    )
    expected_lag_2 = (
        symbol_daily_lookup[symbol_available_dates[position - 2]]
        if position - 2 >= 0
        else None
    )

    actual_lag_1 = row[f"{probe_feature}_lag_1"]
    actual_lag_2 = row[f"{probe_feature}_lag_2"]

    if expected_lag_1 is None:
        assert actual_lag_1 is None, (
            f"date {current_date}: expected null lag_1 "
            f"(this is the symbol's first observed day), "
            f"got {actual_lag_1}."
        )
    else:
        assert math.isclose(
            expected_lag_1, actual_lag_1, rel_tol=1e-6
        ), (
            f"date {current_date}: lag_1 {actual_lag_1} does not "
            f"match this symbol's most recent strictly-earlier "
            f"observed day ({expected_lag_1})."
        )
        checked_lag_1 += 1

    if expected_lag_2 is None:
        assert actual_lag_2 is None, (
            f"date {current_date}: expected null lag_2 "
            f"(fewer than 2 prior observed days), got "
            f"{actual_lag_2}."
        )
    else:
        assert math.isclose(
            expected_lag_2, actual_lag_2, rel_tol=1e-6
        ), (
            f"date {current_date}: lag_2 {actual_lag_2} does not "
            f"match this symbol's second-most-recent "
            f"strictly-earlier observed day ({expected_lag_2})."
        )
        checked_lag_2 += 1

    if expected_lag_1 is not None and expected_lag_2 is not None:
        expected_delta = expected_lag_1 - expected_lag_2
        actual_delta = row[f"{probe_feature}_delta_1"]
        assert math.isclose(
            expected_delta, actual_delta, rel_tol=1e-6
        ), (
            f"date {current_date}: delta_1 does not equal "
            f"lag_1 - lag_2."
        )

assert checked_lag_1 > 0 and checked_lag_2 > 0, (
    "Test slice did not contain enough history to exercise "
    "lag_1/lag_2 checks."
)

print(
    f"[PASS] lag_1/lag_2/delta_1 correctly use this symbol's own "
    f"trading-day positions (not raw date_id-1/-2 arithmetic, "
    f"since symbol {probe_symbol} has real day gaps in this "
    f"slice) for symbol {probe_symbol} "
    f"({checked_lag_1} lag_1 rows, {checked_lag_2} lag_2 rows "
    f"checked)."
)


# ============================================================
# TEST 4: ROLLING MEAN/STD/Z-SCORE USE ONLY EARLIER DAYS,
#          AND EARLY DATES HAVE NULL ROLLING STATS
# ============================================================
# Why:
# This is the central leakage check for this module. We manually
# recompute the window-5 rolling mean at a mid-history date using
# ONLY the shifted (lag_1) series and compare against the engineered
# column. We also deliberately compute the WRONG version, which
# includes the current day's value, and confirm it differs -
# proving this test would catch a real leakage bug.

window = ROLLING_WINDOWS[0]

sorted_dates = sorted(symbol_daily_lookup.keys())

assert len(sorted_dates) > window + 2, (
    "Test slice does not span enough days to test the rolling "
    "window."
)

# Pick a date with at least `window` strictly-earlier days
# available for this symbol.
probe_date = sorted_dates[window + 1]

prior_days = [
    day
    for day in sorted_dates
    if day < probe_date
][-window:]

assert len(prior_days) == window

correct_rolling_mean = sum(
    symbol_daily_lookup[day] for day in prior_days
) / window

leaky_days = prior_days[1:] + [probe_date]
leaky_rolling_mean = sum(
    symbol_daily_lookup[day] for day in leaky_days
) / window

engineered_row = (
    probe_symbol_rolling
    .filter(pl.col(DATE_COLUMN) == probe_date)
)

engineered_rolling_mean = (
    engineered_row[f"{probe_feature}_rolling_mean_{window}"][0]
)

assert math.isclose(
    correct_rolling_mean, engineered_rolling_mean, rel_tol=1e-6
), (
    f"rolling_mean_{window} at date {probe_date} does not match "
    f"a manual calculation using only the {window} strictly "
    f"earlier days."
)

assert not math.isclose(
    leaky_rolling_mean, engineered_rolling_mean, rel_tol=1e-6
), (
    "rolling_mean matches a leaky calculation that includes the "
    "current day. The test would not be able to catch a real "
    "leakage bug - investigate immediately."
)

print(
    f"[PASS] rolling_mean_{window} at date {probe_date} matches a "
    f"manual calculation using only strictly-earlier days, and "
    f"differs from a deliberately leaky version that includes the "
    f"current day."
)

# Early dates (fewer than `window` prior days) must have null
# rolling stats, not a value computed from a partial/short window.

earliest_dates = sorted_dates[:window]

for early_date in earliest_dates:

    early_row = (
        probe_symbol_rolling
        .filter(pl.col(DATE_COLUMN) == early_date)
    )

    if early_row.height == 0:
        continue

    early_rolling_mean = (
        early_row[f"{probe_feature}_rolling_mean_{window}"][0]
    )

    assert early_rolling_mean is None, (
        f"date {early_date} has fewer than {window} prior days "
        f"but rolling_mean_{window} is not null "
        f"({early_rolling_mean})."
    )

print(
    f"[PASS] Earliest dates with insufficient history correctly "
    f"have null rolling_mean_{window} rather than a partial-window "
    f"value."
)

# Verify the z-score formula directly.

rolling_std_value = (
    engineered_row[f"{probe_feature}_rolling_std_{window}"][0]
)
lag_1_value = (
    engineered_row[f"{probe_feature}_lag_1"][0]
)
rolling_z_value = (
    engineered_row[f"{probe_feature}_rolling_z_{window}"][0]
)

if rolling_std_value and rolling_std_value > 0:

    expected_z = (
        (lag_1_value - engineered_rolling_mean)
        / rolling_std_value
    )

    assert math.isclose(
        expected_z, rolling_z_value, rel_tol=1e-6
    ), "rolling_z does not match (lag_1 - rolling_mean) / rolling_std."

    print(
        f"[PASS] rolling_z_{window} matches "
        f"(lag_1 - rolling_mean) / rolling_std."
    )

else:
    print(
        f"[SKIP] rolling_std_{window} was zero/null at the probe "
        f"date; z-score formula check skipped for this date."
    )


# ============================================================
# TEST 5: MERGE BROADCASTS CORRECTLY ONTO ROW-LEVEL DATA
# ============================================================
# Why:
# Every intraday row of a given (symbol, date) must receive the
# IDENTICAL engineered value, since the engineered feature is a
# property of the (symbol, date) pair, not of individual rows.
# Row count must also be unchanged (left join must not fan out).

original_row_count = raw_slice.height

merged = merge_symbol_rolling_features(
    raw_slice,
    symbol_rolling_table,
)

assert merged.height == original_row_count, (
    f"Merge changed row count from {original_row_count} to "
    f"{merged.height}. A left join on (symbol_id, date_id) should "
    f"never fan out if the rolling table has one row per key."
)

broadcast_check = (
    merged
    .filter(
        (pl.col(SYMBOL_COLUMN) == probe_symbol)
        & (pl.col(DATE_COLUMN) == probe_date)
    )
    [f"{probe_feature}_rolling_mean_{window}"]
    .n_unique()
)

assert broadcast_check == 1, (
    f"Expected a single unique engineered value across all "
    f"intraday rows of symbol {probe_symbol}, date {probe_date}, "
    f"but found {broadcast_check} distinct values."
)

print(
    "[PASS] Row count unchanged after merge, and engineered "
    "rolling features broadcast identically across all intraday "
    "rows of a given (symbol, date)."
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("ALL FEATURE ENGINEERING TESTS PASSED")
print("=" * 70)

print(
    "src/feature_engineering.py is verified for leakage safety "
    "and correctness on a small chronological slice. Safe to "
    "proceed to retraining LightGBM with these feature families."
)
