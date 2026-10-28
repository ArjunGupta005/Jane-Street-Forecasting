# ============================================================
# IMPORTS
# ============================================================
# This script isolates partition 9 so we can verify that the
# parquet file is readable before scanning the full dataset.

from pathlib import Path

import polars as pl


# ============================================================
# LOCATE PARTITION 9
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARTITION_9_PATH = (
    PROJECT_ROOT
    / "train.parquet"
    / "partition_id=9"
    / "part-0.parquet"
)


# ============================================================
# CHECK FILE EXISTENCE
# ============================================================

print(f"Checking: {PARTITION_9_PATH}")

if not PARTITION_9_PATH.exists():
    raise FileNotFoundError(
        "Partition 9 was not found at the expected location."
    )


# ============================================================
# READ PARQUET METADATA AND SAMPLE
# ============================================================
# Decision:
# We read only the schema and first few rows.
#
# Why:
# If the parquet footer or internal blocks are corrupted, this
# should expose the problem without unnecessarily scanning the
# entire partition.

lazy_frame = pl.scan_parquet(PARTITION_9_PATH)

print("\nSchema:")
print(lazy_frame.collect_schema())

print("\nFirst 5 rows:")
print(
    lazy_frame
    .head(5)
    .collect()
)


# ============================================================
# CHECK ROW COUNT
# ============================================================
# This forces Polars to scan the partition and confirms that the
# whole parquet file can be read successfully.

row_count = (
    lazy_frame
    .select(
        pl.len().alias("rows")
    )
    .collect()
    ["rows"][0]
)

print(f"\nPartition 9 row count: {row_count:,}")


# ============================================================
# FINAL STATUS
# ============================================================

print("\nPartition 9 is readable.")
