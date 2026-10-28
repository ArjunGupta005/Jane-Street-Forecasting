# ============================================================
# IMPORTS
# ============================================================
# This script verifies the leakage safety and mathematical
# correctness of the redesigned compact temporal features.

import sys
from pathlib import Path

import numpy as np
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
    TRAIN_PATH,
)

from src.feature_engineering import (
    COMPACT_TEMPORAL_BASE_FEATURES,
    add_compact_temporal_features,
    build_compact_temporal_daily_table,
    get_compact_temporal_feature_names,
)


# ============================================================
# SMALL TEST PERIOD
# ============================================================
# We deliberately use a small but multi-day interval so manual
# verification remains practical.

TEST_START_DATE = 500
TEST_END_DATE = 550


# ============================================================
# READ SOURCE DATA
# ============================================================

training_scan = pl.scan_parquet(
    str(
        TRAIN_PATH
        / "**"
        / "*.parquet"
    )
)


# ============================================================
# INCLUDE EARLIER HISTORY
# ============================================================
# Decision:
# The history table starts before the observation test period.
#
# Why:
# Date 500 should have access to prior symbol history.
#
# If we started the history table exactly at date 500, its lagged
# values would incorrectly appear missing.

HISTORY_START_DATE = 450

history_scan = (
    training_scan
    .filter(
        (pl.col("date_id") >= HISTORY_START_DATE)
        &
        (pl.col("date_id") <= TEST_END_DATE)
    )
)

observation_scan = (
    training_scan
    .filter(
        (pl.col("date_id") >= TEST_START_DATE)
        &
        (pl.col("date_id") <= TEST_END_DATE)
    )
)


# ============================================================
# BUILD TEMPORAL TABLE
# ============================================================

daily_temporal_table = (
    build_compact_temporal_daily_table(
        history_scan,
    )
)


# ============================================================
# ADD TEMPORAL FEATURES
# ============================================================

enriched_data = (
    add_compact_temporal_features(
        observation_scan,
        daily_temporal_table,
    )
    .select(
        [
            "date_id",
            "time_id",
            "symbol_id",
        ]
        + COMPACT_TEMPORAL_BASE_FEATURES
        + get_compact_temporal_feature_names()
    )
    .collect()
)

print(
    f"\nRows tested: "
    f"{enriched_data.height:,}"
)


# ============================================================
# TEST 1: EXPECTED FEATURE COUNT
# ============================================================
# Eight source features × three transformations = 24 features.

temporal_feature_names = (
    get_compact_temporal_feature_names()
)

assert len(
    temporal_feature_names
) == (
    len(COMPACT_TEMPORAL_BASE_FEATURES)
    * 3
)

print(
    "PASS: compact temporal feature count is correct."
)


# ============================================================
# BUILD DAILY DATA FOR MANUAL VERIFICATION
# ============================================================

daily_manual = (
    history_scan
    .select(
        [
            "date_id",
            "symbol_id",
        ]
        + COMPACT_TEMPORAL_BASE_FEATURES
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
            for feature
            in COMPACT_TEMPORAL_BASE_FEATURES
        ]
    )
    .sort(
        [
            "symbol_id",
            "date_id",
        ]
    )
    .collect()
)


# ============================================================
# SELECT ONE SYMBOL WITH MULTIPLE DAYS
# ============================================================

symbol_counts = (
    daily_manual
    .group_by("symbol_id")
    .len()
    .sort(
        "len",
        descending=True,
    )
)

test_symbol = (
    symbol_counts["symbol_id"][0]
)

symbol_history = (
    daily_manual
    .filter(
        pl.col("symbol_id")
        == test_symbol
    )
    .sort("date_id")
)

assert symbol_history.height >= 25


# ============================================================
# TEST 2: PREVIOUS-DAY FEATURE IS STRICTLY HISTORICAL
# ============================================================
# Select an observed date with a predecessor in the actual symbol
# history.
#
# Because symbols can skip calendar dates, we verify positional
# historical semantics rather than assuming date_id - 1 exists.
#
# IMPORTANT:
# symbol_history spans the full history window (from
# HISTORY_START_DATE), but enriched_data only contains rows from
# the observation window (TEST_START_DATE onward). test_index must
# therefore be chosen so current_date actually falls inside the
# observation window - otherwise every per-feature lookup below
# would silently find zero matching rows, hit the `continue` guard
# every time, and this test would report PASS without checking
# anything at all.

test_index = next(
    index
    for index, date_value in enumerate(
        symbol_history["date_id"].to_list()
    )
    if date_value >= TEST_START_DATE
)

assert test_index >= 1, (
    "test_index has no prior day available; pick a later probe "
    "point or an earlier HISTORY_START_DATE."
)

current_date = (
    symbol_history["date_id"][test_index]
)

previous_date = (
    symbol_history["date_id"][test_index - 1]
)

assert current_date >= TEST_START_DATE, (
    "current_date falls outside the observation window; "
    "per-feature checks below would silently no-op."
)

checked_lag_features = 0

for feature in COMPACT_TEMPORAL_BASE_FEATURES:

    expected_previous_value = (
        symbol_history[
            f"{feature}_daily_mean"
        ][test_index - 1]
    )

    actual_values = (
        enriched_data
        .filter(
            (pl.col("symbol_id") == test_symbol)
            &
            (pl.col("date_id") == current_date)
        )
        [
            f"{feature}_previous_day_mean"
        ]
    )

    assert actual_values.len() > 0, (
        f"No enriched row found for symbol {test_symbol}, date "
        f"{current_date}, feature {feature}. This test would "
        f"otherwise silently skip verification."
    )

    actual_value = actual_values[0]

    if (
        expected_previous_value is None
        and actual_value is None
    ):
        checked_lag_features += 1
        continue

    assert np.isclose(
        actual_value,
        expected_previous_value,
        equal_nan=True,
    )

    checked_lag_features += 1

assert checked_lag_features == len(COMPACT_TEMPORAL_BASE_FEATURES)
assert previous_date < current_date

print(
    f"PASS: previous-day means come strictly from earlier "
    f"available symbol history "
    f"({checked_lag_features}/{len(COMPACT_TEMPORAL_BASE_FEATURES)} "
    f"features actually checked, symbol {test_symbol}, "
    f"date {current_date})."
)


# ============================================================
# TEST 3: CURRENT-DAY INFORMATION IS NOT USED IN LAG FEATURE
# ============================================================
# If the previous-day feature accidentally included the current
# day's daily mean, it would often equal or move toward that value.
#
# We explicitly verify it against the independently calculated
# previous historical row.

checked_test_3_features = 0

for feature in COMPACT_TEMPORAL_BASE_FEATURES:

    expected_value = (
        symbol_history[
            f"{feature}_daily_mean"
        ][test_index - 1]
    )

    rows = (
        enriched_data
        .filter(
            (pl.col("symbol_id") == test_symbol)
            &
            (pl.col("date_id") == current_date)
        )
    )

    assert rows.height > 0, (
        f"No enriched row found for symbol {test_symbol}, date "
        f"{current_date}. This test would otherwise silently "
        f"skip verification."
    )

    actual_value = (
        rows[
            f"{feature}_previous_day_mean"
        ][0]
    )

    if (
        expected_value is None
        and actual_value is None
    ):
        checked_test_3_features += 1
        continue

    assert np.isclose(
        actual_value,
        expected_value,
        equal_nan=True,
    )

    checked_test_3_features += 1

assert checked_test_3_features == len(COMPACT_TEMPORAL_BASE_FEATURES)

print(
    f"PASS: current-day observations do not enter "
    f"previous-day features "
    f"({checked_test_3_features}/{len(COMPACT_TEMPORAL_BASE_FEATURES)} "
    f"features actually checked)."
)


# ============================================================
# TEST 4: DELTA CALCULATION
# ============================================================
# Verify:
#
# current raw feature - previous-day daily mean

sample_row = (
    enriched_data
    .filter(
        (pl.col("symbol_id") == test_symbol)
        &
        (pl.col("date_id") == current_date)
    )
    .head(1)
)

assert sample_row.height > 0, (
    f"No enriched row found for symbol {test_symbol}, date "
    f"{current_date}. This test would otherwise silently skip "
    f"verification."
)

checked_delta_features = 0

for feature in COMPACT_TEMPORAL_BASE_FEATURES:

    current_value = (
        sample_row[feature][0]
    )

    previous_value = (
        sample_row[
            f"{feature}_previous_day_mean"
        ][0]
    )

    actual_delta = (
        sample_row[
            f"{feature}_delta_previous_day"
        ][0]
    )

    if (
        current_value is None
        or previous_value is None
    ):
        continue

    expected_delta = (
        current_value
        - previous_value
    )

    assert np.isclose(
        actual_delta,
        expected_delta,
        equal_nan=True,
    )

    checked_delta_features += 1

assert checked_delta_features > 0, (
    "No delta comparisons were actually performed; every feature "
    "had a null current or previous value."
)

print(
    f"PASS: current-vs-previous-day deltas are correct "
    f"({checked_delta_features}/{len(COMPACT_TEMPORAL_BASE_FEATURES)} "
    f"features checked)."
)


# ============================================================
# TEST 5: ROLLING FEATURE USES ONLY PRIOR HISTORY
# ============================================================
# Manually calculate the mean of up to the previous 20 completed
# symbol-days.
#
# Current date's daily mean is deliberately excluded.

ROLLING_WINDOW = 20

historical_start = max(
    0,
    test_index - ROLLING_WINDOW,
)

historical_end = test_index

checked_rolling_features = 0

for feature in COMPACT_TEMPORAL_BASE_FEATURES:

    historical_values = (
        symbol_history[
            f"{feature}_daily_mean"
        ][
            historical_start:
            historical_end
        ]
        .drop_nulls()
        .to_numpy()
    )

    assert len(historical_values) > 0, (
        f"No historical values available to build an expected "
        f"rolling mean for {feature}; widen the test window."
    )

    expected_rolling_mean = (
        historical_values.mean()
    )

    actual_values = (
        enriched_data
        .filter(
            (pl.col("symbol_id") == test_symbol)
            &
            (pl.col("date_id") == current_date)
        )
        [
            f"{feature}_rolling_20_prior_mean"
        ]
    )

    assert actual_values.len() > 0, (
        f"No enriched row found for symbol {test_symbol}, date "
        f"{current_date}, feature {feature}. This test would "
        f"otherwise silently skip verification."
    )

    actual_rolling_mean = (
        actual_values[0]
    )

    assert np.isclose(
        actual_rolling_mean,
        expected_rolling_mean,
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
    )

    checked_rolling_features += 1

assert checked_rolling_features == len(COMPACT_TEMPORAL_BASE_FEATURES)

print(
    f"PASS: rolling means contain only prior completed "
    f"symbol-days "
    f"({checked_rolling_features}/{len(COMPACT_TEMPORAL_BASE_FEATURES)} "
    f"features checked)."
)


# ============================================================
# TEST 6: ROW COUNT DOES NOT CHANGE
# ============================================================
# Feature engineering should enrich observations, never duplicate
# or delete them.

original_count = (
    observation_scan
    .select(
        pl.len()
    )
    .collect()
    .item()
)

assert (
    enriched_data.height
    == original_count
)

print(
    "PASS: temporal join preserves observation row count."
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("ALL COMPACT TEMPORAL FEATURE TESTS PASSED")
print("=" * 80)

print(
    f"Test symbol: {test_symbol}"
)

print(
    f"Verification date: {current_date}"
)

print(
    f"Previous available date: {previous_date}"
)
