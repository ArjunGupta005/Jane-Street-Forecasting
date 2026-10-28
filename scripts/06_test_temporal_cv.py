# ============================================================
# IMPORTS
# ============================================================
# This script verifies the temporal cross-validation design
# against the real Jane Street training dataset.

import sys
from pathlib import Path

import polars as pl


# ============================================================
# ALLOW IMPORTS FROM PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOAD PROJECT MODULES
# ============================================================

from src.config import (
    DATE_COLUMN,
    TRAIN_PATH,
)

from src.temporal_cv import (
    DEFAULT_TEMPORAL_FOLDS,
    DEVELOPMENT_TEMPORAL_FOLDS,
    GAP_STRESS_FOLD,
    print_fold_summary,
    validate_temporal_fold,
)


# ============================================================
# LAZY-SCAN TRAINING DATA
# ============================================================
# Decision:
# Filter by date_id directly instead of globally sorting the
# 47-million-row dataset.
#
# Why:
# Physical parquet partitions already follow chronological order,
# and date_id explicitly defines the ordering.
#
# This is significantly cheaper than repeatedly sorting the full
# dataset before selecting a fold.

training_data = pl.scan_parquet(
    str(TRAIN_PATH / "**" / "*.parquet")
)


# ============================================================
# PRINT FULL TEMPORAL CV DESIGN
# ============================================================

print_fold_summary(
    DEFAULT_TEMPORAL_FOLDS
)


# ============================================================
# VERIFY EACH FOLD AGAINST THE REAL DATA
# ============================================================
# For each fold we check:
#
# - training row count
# - validation row count
# - observed min/max training date
# - observed min/max validation date
# - zero overlap
#
# We only aggregate metadata, so the full observations are never
# loaded into memory.

print("\n" + "=" * 80)
print("VERIFYING FOLDS AGAINST DATASET")
print("=" * 80)


for fold in DEFAULT_TEMPORAL_FOLDS:

    validate_temporal_fold(fold)

    print(f"\nChecking {fold.name}...")

    # --------------------------------------------------------
    # TRAINING RANGE
    # --------------------------------------------------------

    training_fold = (
        training_data
        .filter(
            (pl.col(DATE_COLUMN) >= fold.train_start_date)
            &
            (pl.col(DATE_COLUMN) <= fold.train_end_date)
        )
    )

    training_summary = (
        training_fold
        .select(
            pl.len().alias("rows"),
            pl.col(DATE_COLUMN).min().alias("minimum_date"),
            pl.col(DATE_COLUMN).max().alias("maximum_date"),
            pl.col(DATE_COLUMN).n_unique().alias("unique_dates"),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    # --------------------------------------------------------
    # VALIDATION RANGE
    # --------------------------------------------------------

    validation_fold = (
        training_data
        .filter(
            (pl.col(DATE_COLUMN) >= fold.validation_start_date)
            &
            (pl.col(DATE_COLUMN) <= fold.validation_end_date)
        )
    )

    validation_summary = (
        validation_fold
        .select(
            pl.len().alias("rows"),
            pl.col(DATE_COLUMN).min().alias("minimum_date"),
            pl.col(DATE_COLUMN).max().alias("maximum_date"),
            pl.col(DATE_COLUMN).n_unique().alias("unique_dates"),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Training rows:       "
        f"{training_summary['rows']:,}"
    )

    print(
        f"Training dates:      "
        f"{training_summary['minimum_date']} -> "
        f"{training_summary['maximum_date']}"
    )

    print(
        f"Training date count: "
        f"{training_summary['unique_dates']}"
    )

    print(
        f"Validation rows:     "
        f"{validation_summary['rows']:,}"
    )

    print(
        f"Validation dates:    "
        f"{validation_summary['minimum_date']} -> "
        f"{validation_summary['maximum_date']}"
    )

    print(
        f"Validation date count: "
        f"{validation_summary['unique_dates']}"
    )


    # ========================================================
    # ASSERT EXPECTED DATE BOUNDARIES
    # ========================================================
    # These assertions fail immediately if the filtered data does
    # not match the intended chronological fold.

    assert (
        training_summary["minimum_date"]
        == fold.train_start_date
    )

    assert (
        training_summary["maximum_date"]
        == fold.train_end_date
    )

    assert (
        validation_summary["minimum_date"]
        == fold.validation_start_date
    )

    assert (
        validation_summary["maximum_date"]
        == fold.validation_end_date
    )

    assert (
        training_summary["maximum_date"]
        < validation_summary["minimum_date"]
    )


# ============================================================
# DEVELOPMENT FOLD SUMMARY
# ============================================================
# These are the folds we will usually use while iterating on CPU.

print("\n" + "=" * 80)
print("CPU DEVELOPMENT FOLDS")
print("=" * 80)

print_fold_summary(
    DEVELOPMENT_TEMPORAL_FOLDS
)


# ============================================================
# GAP STRESS TEST SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("GAP STRESS TEST")
print("=" * 80)

print_fold_summary(
    [GAP_STRESS_FOLD]
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("TEMPORAL CROSS-VALIDATION TEST PASSED")
print("=" * 80)

print(
    "All training periods occur strictly before their "
    "corresponding validation periods."
)
