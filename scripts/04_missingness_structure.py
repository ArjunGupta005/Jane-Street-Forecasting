# ============================================================
# IMPORTS
# ============================================================
# This script investigates whether feature missingness is
# systematically associated with time or symbol identity.
#
# Why:
# Missing values in financial datasets are often not random.
# They may occur because a feature was unavailable for certain
# symbols, introduced later in time, or generated only under
# specific market conditions.
#
# If missingness is structured, the missing indicator itself may
# contain information and should not automatically be discarded.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl


# ============================================================
# ALLOW IMPORTS FROM PROJECT ROOT
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
    FIGURES_PATH,
    SYMBOL_COLUMN,
    TRAIN_PATH,
)


# ============================================================
# OUTPUT PATHS
# ============================================================

METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics"

FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LAZY-SCAN TRAINING DATA
# ============================================================
# We continue to use lazy parquet scanning because the full
# 47-million-row dataset should not be materialized in memory.

training_data = pl.scan_parquet(
    str(TRAIN_PATH / "**" / "*.parquet")
)


# ============================================================
# SELECT FEATURES WITH MATERIAL MISSINGNESS
# ============================================================
# Decision:
# Focus this analysis on features with at least 1% missingness.
#
# Why:
# Features with tiny amounts of missing data are less likely to
# justify complex missingness modeling.
#
# Tradeoff:
# A low-missingness feature could still have informative missing
# values, but we prioritize the strongest structural patterns
# first.

missingness_file = (
    METRICS_PATH
    / "feature_missingness.csv"
)

missingness_summary = pd.read_csv(
    missingness_file
)

features_with_missingness = (
    missingness_summary.loc[
        missingness_summary["missing_percentage"] >= 1.0,
        "feature",
    ]
    .tolist()
)

print("\n" + "=" * 70)
print("FEATURES SELECTED FOR MISSINGNESS ANALYSIS")
print("=" * 70)

for feature in features_with_missingness:
    print(feature)


# ============================================================
# MISSINGNESS BY DATE
# ============================================================
# Decision:
# Compute the fraction missing for each selected feature on each
# date_id.
#
# Why:
# This reveals whether features appear/disappear over time or
# whether missingness increases during particular regimes.

date_missingness = (
    training_data
    .group_by(DATE_COLUMN)
    .agg(
        [
            pl.col(feature)
            .is_null()
            .mean()
            .alias(feature)
            for feature in features_with_missingness
        ]
    )
    .sort(DATE_COLUMN)
    .collect()
)

date_missingness_pd = (
    date_missingness
    .to_pandas()
)


# ============================================================
# SAVE DATE-LEVEL MISSINGNESS
# ============================================================

date_missingness_pd.to_csv(
    METRICS_PATH / "missingness_by_date.csv",
    index=False,
)


# ============================================================
# PLOT MISSINGNESS THROUGH TIME
# ============================================================
# One plot is generated per feature.
#
# Why:
# Separate plots make structural breaks much easier to inspect
# than putting many features onto one crowded chart.

for feature in features_with_missingness:

    plt.figure(figsize=(12, 4))

    plt.plot(
        date_missingness_pd[DATE_COLUMN],
        date_missingness_pd[feature],
    )

    plt.title(
        f"Missingness Through Time: {feature}"
    )

    plt.xlabel("date_id")
    plt.ylabel("Fraction missing")

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH
        / f"missingness_time_{feature}.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# MISSINGNESS BY SYMBOL
# ============================================================
# Decision:
# Calculate the fraction missing for each selected feature for
# every symbol_id.
#
# Why:
# Some features may only exist for specific instruments.
# If so, missingness may partially encode symbol identity.

symbol_missingness = (
    training_data
    .group_by(SYMBOL_COLUMN)
    .agg(
        [
            pl.col(feature)
            .is_null()
            .mean()
            .alias(feature)
            for feature in features_with_missingness
        ]
    )
    .sort(SYMBOL_COLUMN)
    .collect()
)

symbol_missingness_pd = (
    symbol_missingness
    .to_pandas()
)


# ============================================================
# SAVE SYMBOL-LEVEL MISSINGNESS
# ============================================================

symbol_missingness_pd.to_csv(
    METRICS_PATH / "missingness_by_symbol.csv",
    index=False,
)


# ============================================================
# PLOT MISSINGNESS BY SYMBOL
# ============================================================

for feature in features_with_missingness:

    plt.figure(figsize=(12, 4))

    plt.bar(
        symbol_missingness_pd[SYMBOL_COLUMN].astype(str),
        symbol_missingness_pd[feature],
    )

    plt.title(
        f"Missingness by Symbol: {feature}"
    )

    plt.xlabel("symbol_id")
    plt.ylabel("Fraction missing")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH
        / f"missingness_symbol_{feature}.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# SUMMARY STATISTICS FOR DATE-LEVEL VARIATION
# ============================================================
# Decision:
# Measure how much each feature's missing rate changes through
# time.
#
# Features with very large ranges or standard deviations are
# likely to have non-random missingness patterns.

date_variation_rows = []

for feature in features_with_missingness:

    series = date_missingness_pd[feature]

    date_variation_rows.append(
        {
            "feature": feature,
            "mean_missing": series.mean(),
            "std_missing": series.std(),
            "min_missing": series.min(),
            "max_missing": series.max(),
            "range_missing": (
                series.max()
                - series.min()
            ),
        }
    )


date_variation_summary = (
    pd.DataFrame(date_variation_rows)
    .sort_values(
        "range_missing",
        ascending=False,
    )
)

date_variation_summary.to_csv(
    METRICS_PATH
    / "missingness_date_variation_summary.csv",
    index=False,
)


# ============================================================
# SUMMARY STATISTICS FOR SYMBOL-LEVEL VARIATION
# ============================================================
# Why:
# Large symbol-to-symbol variation suggests that missingness is
# related to instrument coverage rather than random noise.

symbol_variation_rows = []

for feature in features_with_missingness:

    series = symbol_missingness_pd[feature]

    symbol_variation_rows.append(
        {
            "feature": feature,
            "mean_missing": series.mean(),
            "std_missing": series.std(),
            "min_missing": series.min(),
            "max_missing": series.max(),
            "range_missing": (
                series.max()
                - series.min()
            ),
        }
    )


symbol_variation_summary = (
    pd.DataFrame(symbol_variation_rows)
    .sort_values(
        "range_missing",
        ascending=False,
    )
)

symbol_variation_summary.to_csv(
    METRICS_PATH
    / "missingness_symbol_variation_summary.csv",
    index=False,
)


# ============================================================
# PRINT MOST STRUCTURED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("LARGEST TIME-BASED MISSINGNESS VARIATION")
print("=" * 70)

print(
    date_variation_summary.head(15)
)


print("\n" + "=" * 70)
print("LARGEST SYMBOL-BASED MISSINGNESS VARIATION")
print("=" * 70)

print(
    symbol_variation_summary.head(15)
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("MISSINGNESS STRUCTURE ANALYSIS COMPLETE")
print("=" * 70)

print(
    "Saved date-level missingness to "
    "outputs/metrics/missingness_by_date.csv"
)

print(
    "Saved symbol-level missingness to "
    "outputs/metrics/missingness_by_symbol.csv"
)

print(
    "Saved missingness plots to "
    "outputs/figures/"
)
