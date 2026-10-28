# ============================================================
# IMPORTS
# ============================================================
# This script is a fast smoke test of src/preprocessing.py on a
# small chronological slice of the training data.
#
# Why:
# We deliberately do not run preprocessing over all 47 million
# rows yet. If something is wrong with the logic (missingness
# indicators, imputation, leakage in fitted statistics), it should
# fail loudly here in seconds rather than silently during a much
# more expensive full-dataset run or, worse, inside temporal CV.

import math
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_PATH,
    WEIGHT_COLUMN,
)
from src.preprocessing import (
    STRUCTURAL_MISSING_FEATURES,
    build_model_feature_list,
    calculate_feature_medians,
    get_linear_model_config,
    get_tree_model_config,
    median_impute_polars,
    preprocess_polars_dataframe,
)


# ============================================================
# LOAD A SMALL CHRONOLOGICAL SLICE
# ============================================================
# Decision:
# Sort by date_id, time_id, symbol_id and take the first 100,000
# rows rather than an arbitrary sample.
#
# Why:
# This keeps the slice genuinely chronological (rather than an
# unordered mix of partitions) and, since it lands in the earliest
# dates, it actually contains the feature_21/26/27/31 hard
# availability break identified during EDA. A random sample could
# easily miss that regime entirely and give us false confidence.

TEST_ROWS = 100_000

raw_slice = (
    pl.scan_parquet(
        str(TRAIN_PATH / "**" / "*.parquet")
    )
    .sort([DATE_COLUMN, TIME_COLUMN, SYMBOL_COLUMN])
    .head(TEST_ROWS)
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

assert raw_slice.height == TEST_ROWS, (
    "Expected exactly the requested number of rows."
)


# ============================================================
# TEST 1: TREE CONFIG PRESERVES NATIVE MISSING VALUES
# ============================================================
# Why:
# LightGBM/XGBoost are meant to see real NaNs so they can learn
# their own missingness splits. If imputation_method="none" ever
# silently filled values, that design decision would be broken.

tree_config = get_tree_model_config()

tree_processed = preprocess_polars_dataframe(
    raw_slice,
    FEATURE_COLUMNS,
    tree_config,
)

for feature in STRUCTURAL_MISSING_FEATURES:

    original_nulls = raw_slice[feature].null_count()
    processed_nulls = tree_processed[feature].null_count()

    assert original_nulls == processed_nulls, (
        f"{feature}: tree preprocessing must not fill nulls, but "
        f"null count changed from {original_nulls} to "
        f"{processed_nulls}."
    )

print("\n[PASS] Tree config: native NaNs preserved in feature columns.")


# ============================================================
# TEST 2: MISSING INDICATORS MATCH ORIGINAL NULLS
# ============================================================
# Why:
# The indicator columns are only useful if they exactly mirror
# where the original data was null, both for the tree config and
# for the linear config (added before imputation).

for feature in STRUCTURAL_MISSING_FEATURES:

    indicator_column = f"{feature}_missing"

    assert indicator_column in tree_processed.columns, (
        f"Expected missing indicator column {indicator_column} "
        f"was not created."
    )

    expected_flags = (
        raw_slice[feature]
        .is_null()
        .cast(pl.Int8)
    )

    actual_flags = tree_processed[indicator_column]

    assert expected_flags.equals(actual_flags), (
        f"{indicator_column} does not match is_null() on the "
        f"original feature."
    )

print("[PASS] Missing indicators exactly match original null positions.")


# ============================================================
# TEST 3: LINEAR CONFIG APPLIES ZERO IMPUTATION
# ============================================================
# Why:
# Linear models cannot accept NaNs. Zero imputation must remove
# every null from the feature columns while the indicator columns
# still remember where those nulls originally were.

linear_config = get_linear_model_config()

linear_processed = preprocess_polars_dataframe(
    raw_slice,
    FEATURE_COLUMNS,
    linear_config,
)

remaining_nulls = {
    feature: linear_processed[feature].null_count()
    for feature in FEATURE_COLUMNS
    if linear_processed[feature].null_count() > 0
}

assert not remaining_nulls, (
    f"Zero imputation left nulls behind: {remaining_nulls}"
)

for feature in STRUCTURAL_MISSING_FEATURES:

    indicator_column = f"{feature}_missing"

    expected_flags = (
        raw_slice[feature]
        .is_null()
        .cast(pl.Int8)
    )

    actual_flags = linear_processed[indicator_column]

    assert expected_flags.equals(actual_flags), (
        f"{indicator_column} should still reflect original "
        f"missingness after zero imputation."
    )

print("[PASS] Linear config: zero imputation removed all nulls,")
print("       and missing indicators still reflect the original nulls.")


# ============================================================
# TEST 4: ROW COUNT, TARGET, AND IDENTIFIERS ARE UNCHANGED
# ============================================================
# Why:
# Preprocessing should never drop/reorder rows or silently mutate
# the target or identifier columns. If it did, every downstream
# joint between predictions, targets, and dates/symbols/weights
# would be corrupted.

identifier_columns = [
    DATE_COLUMN,
    TIME_COLUMN,
    SYMBOL_COLUMN,
    WEIGHT_COLUMN,
]

for processed_name, processed_frame in [
    ("tree", tree_processed),
    ("linear", linear_processed),
]:

    assert processed_frame.height == raw_slice.height, (
        f"{processed_name} preprocessing changed row count."
    )

    assert processed_frame[TARGET_COLUMN].equals(
        raw_slice[TARGET_COLUMN]
    ), (
        f"{processed_name} preprocessing altered {TARGET_COLUMN}."
    )

    for identifier in identifier_columns:

        assert processed_frame[identifier].equals(
            raw_slice[identifier]
        ), (
            f"{processed_name} preprocessing altered "
            f"{identifier}."
        )

print("[PASS] Row count, target, and identifier columns are unchanged")
print("       for both tree and linear preprocessing.")


# ============================================================
# TEST 5: FEATURE COUNT MATCHES EXPECTATION
# ============================================================
# Why:
# build_model_feature_list() is what model scripts will actually
# use to select columns for training. Its output must line up with
# what preprocess_polars_dataframe() actually produced.

tree_feature_list = build_model_feature_list(
    FEATURE_COLUMNS,
    tree_config,
)

linear_feature_list = build_model_feature_list(
    FEATURE_COLUMNS,
    linear_config,
)

expected_indicator_count = len(
    [
        feature
        for feature in STRUCTURAL_MISSING_FEATURES
        if feature in FEATURE_COLUMNS
    ]
)

expected_extra_columns = expected_indicator_count + 2
# +2 accounts for time_id and symbol_id, both True by default.
# date_id defaults to False and is intentionally excluded.

expected_total_features = (
    len(FEATURE_COLUMNS) + expected_extra_columns
)

assert len(tree_feature_list) == expected_total_features, (
    f"Tree feature list has {len(tree_feature_list)} columns, "
    f"expected {expected_total_features}."
)

assert len(linear_feature_list) == expected_total_features, (
    f"Linear feature list has {len(linear_feature_list)} columns, "
    f"expected {expected_total_features}."
)

assert "date_id" not in tree_feature_list, (
    "date_id should not be included by default."
)

missing_from_frame = [
    column
    for column in tree_feature_list
    if column not in tree_processed.columns
]

assert not missing_from_frame, (
    f"build_model_feature_list() references columns that do not "
    f"exist in the processed frame: {missing_from_frame}"
)

print(
    f"[PASS] Feature count matches expectation: "
    f"{expected_total_features} columns "
    f"({len(FEATURE_COLUMNS)} base features + "
    f"{expected_indicator_count} missing indicators + "
    f"time_id + symbol_id)."
)


# ============================================================
# TEST 6: NO LOOK-AHEAD IN TRAINING-FOLD MEDIAN CALCULATION
# ============================================================
# Why:
# This is the single most important check in this script. Median
# imputation must be fit ONLY on a training period and reused
# unchanged on validation/future data. We verify this two ways:
#
# 1. Medians calculated from an early-only slice must match a
#    manual median computed on that same early-only slice.
# 2. Those same early-only medians must differ from medians
#    calculated using the entire test slice, proving that the
#    function is sensitive to which rows it is given rather than
#    silently pulling in the full dataset.

split_date = int(
    raw_slice[DATE_COLUMN].median()
)

train_portion = raw_slice.filter(
    pl.col(DATE_COLUMN) <= split_date
)

validation_portion = raw_slice.filter(
    pl.col(DATE_COLUMN) > split_date
)

assert train_portion.height > 0 and validation_portion.height > 0, (
    "Train/validation split produced an empty partition."
)

train_only_medians = calculate_feature_medians(
    train_portion,
    FEATURE_COLUMNS,
)

full_slice_medians = calculate_feature_medians(
    raw_slice,
    FEATURE_COLUMNS,
)

# Check 1: train-only medians match a manual calculation on the
# exact same rows.

probe_feature = "feature_39"

manual_train_median = (
    train_portion
    .select(
        pl.col(probe_feature).median()
    )
    [probe_feature][0]
)

assert math.isclose(
    train_only_medians[probe_feature],
    manual_train_median,
    rel_tol=1e-9,
), (
    "calculate_feature_medians() does not match a manual median "
    "computed on the same training-only rows."
)

# Check 2: train-only medians differ from full-slice medians for
# at least one feature with material missingness/drift, confirming
# the function is not accidentally using validation-period rows.

differing_features = [
    feature
    for feature in STRUCTURAL_MISSING_FEATURES
    if not math.isclose(
        train_only_medians[feature],
        full_slice_medians[feature],
        rel_tol=1e-9,
    )
]

assert differing_features, (
    "Expected training-only medians to differ from full-slice "
    "medians for at least one structurally missing feature. "
    "If they are identical, this may indicate the training split "
    "is accidentally including validation rows."
)

# Apply the training-only medians to the validation portion and
# confirm no nulls remain and no error occurs.

validation_imputed = median_impute_polars(
    validation_portion,
    train_only_medians,
)

validation_remaining_nulls = {
    feature: validation_imputed[feature].null_count()
    for feature in FEATURE_COLUMNS
    if validation_imputed[feature].null_count() > 0
}

assert not validation_remaining_nulls, (
    f"Median imputation left nulls in validation data: "
    f"{validation_remaining_nulls}"
)

print(
    "[PASS] No look-ahead: training-fold medians match a manual "
    "calculation on training-only rows, differ from full-slice "
    "medians, and impute validation data cleanly."
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("ALL PREPROCESSING TESTS PASSED")
print("=" * 70)

print(
    "src/preprocessing.py is verified on a "
    f"{TEST_ROWS:,}-row chronological slice. "
    "Safe to proceed to temporal cross-validation."
)
