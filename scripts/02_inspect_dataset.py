# ============================================================
# IMPORTS
# ============================================================
# This script inspects the Jane Street dataset structure using
# Polars lazy scanning.
#
# Why:
# The full dataset is too large to load into memory casually.
# Lazy scanning allows us to inspect parquet files efficiently
# and only compute the summaries we actually request.

import sys
from pathlib import Path

import polars as pl
import psutil


# ============================================================
# ALLOW IMPORTS FROM THE PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOAD PROJECT CONFIGURATION
# ============================================================

from src.config import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_PATH,
    WEIGHT_COLUMN,
)


# ============================================================
# SYSTEM MEMORY CHECK
# ============================================================
# Decision:
# We inspect available system memory before attempting large
# operations.
#
# Why:
# The parquet dataset is compressed on disk. Its in-memory size
# can be much larger than its file size.
#
# Tradeoff:
# Available RAM can change while programs are running, so this is
# only a snapshot rather than a strict safety guarantee.

memory = psutil.virtual_memory()

print("\n" + "=" * 70)
print("SYSTEM MEMORY")
print("=" * 70)

print(f"Total RAM:     {memory.total / (1024 ** 3):.2f} GB")
print(f"Available RAM: {memory.available / (1024 ** 3):.2f} GB")
print(f"RAM used:      {memory.percent:.1f}%")


# ============================================================
# LOCATE TRAINING PARQUET FILES
# ============================================================
# The Jane Street training set is partitioned into multiple
# directories. We recursively locate every parquet file instead
# of assuming a specific nested filename structure.

train_files = sorted(TRAIN_PATH.rglob("*.parquet"))

print("\n" + "=" * 70)
print("TRAINING FILE DISCOVERY")
print("=" * 70)

print(f"Number of parquet files found: {len(train_files)}")

for file_path in train_files:
    print(file_path.relative_to(PROJECT_ROOT))


# ============================================================
# CHECK TOTAL DISK SIZE
# ============================================================
# This tells us the compressed parquet size on disk.
#
# Important:
# Disk size is not the same as memory usage after decompression.

total_disk_bytes = sum(
    file_path.stat().st_size
    for file_path in train_files
)

print(
    f"\nCompressed training data size: "
    f"{total_disk_bytes / (1024 ** 3):.2f} GB"
)


# ============================================================
# INSPECT PARQUET SCHEMA
# ============================================================
# Decision:
# Read the schema from the first training file before performing
# broader computations.
#
# Why:
# This verifies column names and dtypes without loading the data.

first_file = train_files[0]

schema = pl.scan_parquet(first_file).collect_schema()

print("\n" + "=" * 70)
print("DATASET SCHEMA")
print("=" * 70)

for column_name, column_type in schema.items():
    print(f"{column_name:<20} {column_type}")


# ============================================================
# VERIFY EXPECTED COLUMNS
# ============================================================
# We check that the expected identifiers, features, target, and
# weight columns are actually present.

required_columns = [
    DATE_COLUMN,
    TIME_COLUMN,
    SYMBOL_COLUMN,
    WEIGHT_COLUMN,
    TARGET_COLUMN,
] + FEATURE_COLUMNS

missing_columns = [
    column
    for column in required_columns
    if column not in schema
]

print("\n" + "=" * 70)
print("COLUMN VALIDATION")
print("=" * 70)

if missing_columns:
    print("Missing expected columns:")
    for column in missing_columns:
        print(column)
else:
    print("All expected identifier, feature, weight, and target columns found.")


# ============================================================
# PARTITION-BY-PARTITION SUMMARY
# ============================================================
# Decision:
# Summarize each parquet file independently.
#
# Why:
# This preserves memory and lets us understand how chronology is
# distributed across the physical partitions.
#
# We collect:
# - row count
# - minimum and maximum date_id
# - minimum and maximum time_id
# - number of unique symbols
#
# Tradeoff:
# This requires scanning every parquet file once, so it may take
# several minutes on CPU.

partition_summaries = []

print("\n" + "=" * 70)
print("PARTITION SUMMARIES")
print("=" * 70)

for index, file_path in enumerate(train_files):

    print(
        f"\nScanning file {index + 1}/{len(train_files)}: "
        f"{file_path.relative_to(PROJECT_ROOT)}"
    )

    lazy_frame = pl.scan_parquet(file_path)

    summary = (
        lazy_frame
        .select(
            pl.len().alias("rows"),
            pl.col(DATE_COLUMN).min().alias("min_date"),
            pl.col(DATE_COLUMN).max().alias("max_date"),
            pl.col(TIME_COLUMN).min().alias("min_time"),
            pl.col(TIME_COLUMN).max().alias("max_time"),
            pl.col(SYMBOL_COLUMN).n_unique().alias("unique_symbols"),
        )
        .collect()
        .row(0, named=True)
    )

    partition_summaries.append(
        {
            "file": str(file_path.relative_to(PROJECT_ROOT)),
            **summary,
        }
    )

    print(f"Rows:           {summary['rows']:,}")
    print(f"Date range:     {summary['min_date']} -> {summary['max_date']}")
    print(f"Time range:     {summary['min_time']} -> {summary['max_time']}")
    print(f"Unique symbols: {summary['unique_symbols']}")


# ============================================================
# COMBINE PARTITION SUMMARY
# ============================================================

partition_summary_frame = pl.DataFrame(partition_summaries)

print("\n" + "=" * 70)
print("COMBINED PARTITION SUMMARY")
print("=" * 70)

print(partition_summary_frame)


# ============================================================
# TOTAL ROW COUNT
# ============================================================

total_rows = partition_summary_frame["rows"].sum()

print(f"\nTotal training rows: {total_rows:,}")


# ============================================================
# CHECK CHRONOLOGICAL PARTITION ORDER
# ============================================================
# Decision:
# Verify whether each parquet partition follows the previous one
# chronologically.
#
# Why:
# Temporal cross-validation and walk-forward testing depend on
# preserving time order.
#
# A partition that overlaps or moves backward in time would need
# special handling.

print("\n" + "=" * 70)
print("CHRONOLOGICAL PARTITION CHECK")
print("=" * 70)

chronological = True

for index in range(1, len(partition_summaries)):

    previous_max_date = partition_summaries[index - 1]["max_date"]
    current_min_date = partition_summaries[index]["min_date"]

    if current_min_date < previous_max_date:
        chronological = False

        print(
            f"Potential overlap: file {index} ends at "
            f"{previous_max_date}, but file {index + 1} starts at "
            f"{current_min_date}."
        )

if chronological:
    print("Training partitions appear to follow chronological order.")


# ============================================================
# WHOLE-DATASET DATE AND SYMBOL SUMMARY
# ============================================================
# We now scan all training files as one lazy dataset.
#
# This still does not load the complete dataset into memory.
# Polars processes the parquet files as required by the query.

all_training_data = pl.scan_parquet(
    str(TRAIN_PATH / "**" / "*.parquet")
)

overall_summary = (
    all_training_data
    .select(
        pl.col(DATE_COLUMN).min().alias("minimum_date"),
        pl.col(DATE_COLUMN).max().alias("maximum_date"),
        pl.col(DATE_COLUMN).n_unique().alias("unique_dates"),
        pl.col(TIME_COLUMN).n_unique().alias("unique_time_ids"),
        pl.col(SYMBOL_COLUMN).n_unique().alias("unique_symbols"),
    )
    .collect()
)

print("\n" + "=" * 70)
print("OVERALL TEMPORAL STRUCTURE")
print("=" * 70)

print(overall_summary)


# ============================================================
# TARGET SUMMARY
# ============================================================
# We inspect responder_6 before modeling.
#
# Because financial targets are often heavy-tailed and centered
# near zero, we want the mean, standard deviation, extrema, and
# quantiles before selecting modeling assumptions.

target_summary = (
    all_training_data
    .select(
        pl.col(TARGET_COLUMN).count().alias("non_null_count"),
        pl.col(TARGET_COLUMN).null_count().alias("null_count"),
        pl.col(TARGET_COLUMN).mean().alias("mean"),
        pl.col(TARGET_COLUMN).std().alias("std"),
        pl.col(TARGET_COLUMN).min().alias("minimum"),
        pl.col(TARGET_COLUMN).quantile(0.01).alias("q01"),
        pl.col(TARGET_COLUMN).quantile(0.25).alias("q25"),
        pl.col(TARGET_COLUMN).median().alias("median"),
        pl.col(TARGET_COLUMN).quantile(0.75).alias("q75"),
        pl.col(TARGET_COLUMN).quantile(0.99).alias("q99"),
        pl.col(TARGET_COLUMN).max().alias("maximum"),
    )
    .collect()
)

print("\n" + "=" * 70)
print("TARGET SUMMARY: responder_6")
print("=" * 70)

print(target_summary)


# ============================================================
# FEATURE MISSINGNESS
# ============================================================
# Decision:
# Measure missingness before choosing an imputation method.
#
# Why:
# We should not automatically fill NaNs with zero just because a
# leaderboard solution used that approach. The proper workflow is:
#
# observe missingness -> form hypotheses -> compare preprocessing
# choices using temporal validation.
#
# Tradeoff:
# Computing missing counts over all 79 features requires another
# pass through the full dataset and may take some time.

print("\n" + "=" * 70)
print("FEATURE MISSINGNESS")
print("=" * 70)

missing_counts = (
    all_training_data
    .select(
        [
            pl.col(column).null_count().alias(column)
            for column in FEATURE_COLUMNS
        ]
    )
    .collect()
)

missing_rows = []

for feature in FEATURE_COLUMNS:

    missing_count = missing_counts[feature][0]

    missing_percentage = (
        missing_count / total_rows * 100
        if total_rows > 0
        else 0
    )

    missing_rows.append(
        {
            "feature": feature,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        }
    )

missing_summary = (
    pl.DataFrame(missing_rows)
    .sort("missing_percentage", descending=True)
)

print(missing_summary)


# ============================================================
# SAVE DATASET AUDIT RESULTS
# ============================================================
# Mechanical step:
# Save the partition and missingness summaries so we can reference
# them later without rescanning the full dataset.

audit_output_path = PROJECT_ROOT / "outputs" / "metrics"

audit_output_path.mkdir(
    parents=True,
    exist_ok=True,
)

partition_summary_frame.write_csv(
    audit_output_path / "partition_summary.csv"
)

missing_summary.write_csv(
    audit_output_path / "feature_missingness.csv"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)

print(
    "Saved partition summary to "
    "outputs/metrics/partition_summary.csv"
)

print(
    "Saved feature missingness to "
    "outputs/metrics/feature_missingness.csv"
)
