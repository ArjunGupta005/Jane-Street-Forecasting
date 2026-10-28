# ============================================================
# IMPORTS
# ============================================================
# This script performs the first serious exploratory analysis of
# the Jane Street training data.
#
# The emphasis is on time-series structure, nonstationarity,
# cross-sectional behavior, and leakage-safe diagnostics.
#
# Because the dataset contains more than 47 million rows, we avoid
# collecting the full training set into memory.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.graphics.tsaplots import plot_acf


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
    FIGURES_PATH,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_PATH,
    WEIGHT_COLUMN,
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LAZY-SCAN THE FULL TRAINING DATA
# ============================================================
# Decision:
# Use Polars lazy scanning across all parquet partitions.
#
# Why:
# We need whole-dataset summaries without loading 47+ million rows
# into memory.
#
# Tradeoff:
# Each aggregate query still scans data from disk, so this script
# may take several minutes on CPU.

training_data = pl.scan_parquet(
    str(TRAIN_PATH / "**" / "*.parquet")
)


# ============================================================
# DAILY TARGET SUMMARY
# ============================================================
# Decision:
# Aggregate responder_6 to the date_id level.
#
# Why:
# date_id gives us a stable chronological unit, unlike time_id,
# whose real-world spacing is not guaranteed to be uniform.
#
# This aggregation will also be useful later when calculating
# daily strategy returns and annualized Sharpe.

daily_target = (
    training_data
    .group_by(DATE_COLUMN)
    .agg(
        pl.col(TARGET_COLUMN).mean().alias("target_mean"),
        pl.col(TARGET_COLUMN).std().alias("target_std"),
        pl.col(TARGET_COLUMN).median().alias("target_median"),
        pl.col(TARGET_COLUMN).quantile(0.25).alias("target_q25"),
        pl.col(TARGET_COLUMN).quantile(0.75).alias("target_q75"),
        pl.len().alias("row_count"),
    )
    .sort(DATE_COLUMN)
    .collect()
)

daily_target_pd = daily_target.to_pandas()


# ============================================================
# PLOT DAILY TARGET MEAN
# ============================================================
# Why:
# A drifting daily mean may indicate nonstationarity or regime
# changes in the conditional target distribution.

plt.figure(figsize=(12, 5))

plt.plot(
    daily_target_pd[DATE_COLUMN],
    daily_target_pd["target_mean"],
)

plt.axhline(
    0,
    linestyle="--",
)

plt.title("Daily Mean of responder_6")
plt.xlabel("date_id")
plt.ylabel("Mean responder_6")

plt.tight_layout()

plt.savefig(
    FIGURES_PATH / "daily_target_mean.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT DAILY TARGET VOLATILITY
# ============================================================
# Decision:
# Use daily cross-sectional standard deviation as a simple measure
# of changing target dispersion.
#
# Why:
# Financial series commonly exhibit heteroskedasticity and regime-
# dependent volatility.
#
# This does not prove a GARCH process. It simply tells us whether
# conditional variance is visibly changing through time.

plt.figure(figsize=(12, 5))

plt.plot(
    daily_target_pd[DATE_COLUMN],
    daily_target_pd["target_std"],
)

plt.title("Daily Cross-Sectional Standard Deviation of responder_6")
plt.xlabel("date_id")
plt.ylabel("Standard deviation")

plt.tight_layout()

plt.savefig(
    FIGURES_PATH / "daily_target_volatility.png",
    dpi=150,
)

plt.close()


# ============================================================
# ROLLING TARGET STATISTICS
# ============================================================
# Decision:
# Calculate rolling 20-day and 60-day target summaries.
#
# Why:
# Rolling statistics help expose slow-moving regime shifts and
# changes in the target distribution.
#
# Tradeoff:
# These are descriptive EDA statistics only. We will not directly
# use future-inclusive rolling values as model features.

daily_target_pd["rolling_mean_20"] = (
    daily_target_pd["target_mean"]
    .rolling(window=20)
    .mean()
)

daily_target_pd["rolling_mean_60"] = (
    daily_target_pd["target_mean"]
    .rolling(window=60)
    .mean()
)

daily_target_pd["rolling_std_20"] = (
    daily_target_pd["target_mean"]
    .rolling(window=20)
    .std()
)

daily_target_pd["rolling_std_60"] = (
    daily_target_pd["target_mean"]
    .rolling(window=60)
    .std()
)


# ============================================================
# PLOT ROLLING TARGET MEAN
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    daily_target_pd[DATE_COLUMN],
    daily_target_pd["rolling_mean_20"],
    label="20-day rolling mean",
)

plt.plot(
    daily_target_pd[DATE_COLUMN],
    daily_target_pd["rolling_mean_60"],
    label="60-day rolling mean",
)

plt.axhline(
    0,
    linestyle="--",
)

plt.title("Rolling Mean of Daily responder_6")
plt.xlabel("date_id")
plt.ylabel("Rolling mean")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_PATH / "rolling_target_mean.png",
    dpi=150,
)

plt.close()


# ============================================================
# AUTOCORRELATION OF DAILY TARGET
# ============================================================
# Decision:
# Examine autocorrelation of the daily mean target.
#
# Why:
# Autocorrelation tells us whether aggregate target behavior has
# persistence over time.
#
# Important:
# This is not enough to justify ARIMA by itself. The main dataset
# is a panel of many symbols and features rather than a single
# scalar time series.

daily_mean_series = (
    daily_target_pd["target_mean"]
    .dropna()
)

plt.figure(figsize=(10, 5))

plot_acf(
    daily_mean_series,
    lags=60,
    zero=False,
)

plt.title("Autocorrelation of Daily Mean responder_6")
plt.tight_layout()

plt.savefig(
    FIGURES_PATH / "daily_target_acf.png",
    dpi=150,
)

plt.close()


# ============================================================
# SYMBOL-LEVEL TARGET SUMMARY
# ============================================================
# Decision:
# Compare responder_6 distributions across symbols.
#
# Why:
# If target behavior differs materially by symbol, symbol identity
# or symbol-specific rolling statistics may contain useful
# information.
#
# Tradeoff:
# symbol_id is anonymized and new symbols may appear in future
# data. We should therefore avoid building a strategy that depends
# entirely on memorizing individual symbols.

symbol_summary = (
    training_data
    .group_by(SYMBOL_COLUMN)
    .agg(
        pl.len().alias("rows"),
        pl.col(TARGET_COLUMN).mean().alias("target_mean"),
        pl.col(TARGET_COLUMN).std().alias("target_std"),
        pl.col(TARGET_COLUMN).median().alias("target_median"),
    )
    .sort(SYMBOL_COLUMN)
    .collect()
)

symbol_summary_pd = symbol_summary.to_pandas()


# ============================================================
# PLOT TARGET MEAN BY SYMBOL
# ============================================================

plt.figure(figsize=(12, 5))

plt.bar(
    symbol_summary_pd[SYMBOL_COLUMN].astype(str),
    symbol_summary_pd["target_mean"],
)

plt.title("Mean responder_6 by Symbol")
plt.xlabel("symbol_id")
plt.ylabel("Mean responder_6")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    FIGURES_PATH / "target_mean_by_symbol.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT TARGET VOLATILITY BY SYMBOL
# ============================================================

plt.figure(figsize=(12, 5))

plt.bar(
    symbol_summary_pd[SYMBOL_COLUMN].astype(str),
    symbol_summary_pd["target_std"],
)

plt.title("Responder_6 Standard Deviation by Symbol")
plt.xlabel("symbol_id")
plt.ylabel("Standard deviation")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    FIGURES_PATH / "target_std_by_symbol.png",
    dpi=150,
)

plt.close()


# ============================================================
# FEATURE-TARGET CORRELATION SAMPLE
# ============================================================
# Decision:
# Estimate feature correlations using a sample rather than the
# full 47-million-row dataset.
#
# Why:
# Correlation is a useful screening diagnostic, but calculating
# the full 79x79 structure over the entire dataset would be
# expensive and unnecessary at this stage.
#
# Tradeoff:
# A sample introduces estimation noise. More importantly, simple
# Pearson correlation only measures linear association and can
# miss nonlinear predictive relationships.
#
# We therefore treat this only as exploratory evidence.
#
# Note:
# Sample sizes are kept conservative (250k / 150k) given this
# machine's 16 GB RAM ceiling, rather than the larger 1M / 500k
# defaults, to avoid an out-of-memory crash during pandas
# conversion.

CORRELATION_SAMPLE_ROWS = 250_000

sample_for_correlation = (
    training_data
    .select(
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )
    .filter(
        pl.col(TARGET_COLUMN).is_not_null()
    )
    .head(CORRELATION_SAMPLE_ROWS)
    .collect()
)

correlation_pd = sample_for_correlation.to_pandas()


# ============================================================
# CALCULATE FEATURE CORRELATIONS WITH TARGET
# ============================================================

feature_correlations = (
    correlation_pd
    .corr(numeric_only=True)[TARGET_COLUMN]
    .drop(TARGET_COLUMN)
    .sort_values(
        key=lambda series: series.abs(),
        ascending=False,
    )
)

feature_correlation_frame = (
    feature_correlations
    .rename("correlation")
    .reset_index()
    .rename(columns={"index": "feature"})
)

print("\n" + "=" * 70)
print("TOP 20 ABSOLUTE FEATURE-TARGET CORRELATIONS")
print("=" * 70)

print(
    feature_correlation_frame.head(20)
)


# ============================================================
# CORRELATION STABILITY THROUGH TIME
# ============================================================
# Decision:
# Compare feature-target correlation in an early and late period.
#
# Why:
# Financial relationships often decay or reverse over time.
# A feature that looks strong in aggregate may be unstable.
#
# This is an important nonstationarity diagnostic.

EARLY_START_DATE = 0
EARLY_END_DATE = 300

LATE_START_DATE = 1399
LATE_END_DATE = 1698


def calculate_period_correlations(
    start_date,
    end_date,
    label,
    sample_rows=150_000,
):
    """
    Calculate feature-target correlations for a defined
    chronological period.
    """

    period_data = (
        training_data
        .filter(
            (pl.col(DATE_COLUMN) >= start_date)
            & (pl.col(DATE_COLUMN) <= end_date)
        )
        .select(
            FEATURE_COLUMNS
            + [TARGET_COLUMN]
        )
        .head(sample_rows)
        .collect()
        .to_pandas()
    )

    correlations = (
        period_data
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
    )

    return correlations.rename(label)


early_correlations = calculate_period_correlations(
    EARLY_START_DATE,
    EARLY_END_DATE,
    "early_correlation",
)

late_correlations = calculate_period_correlations(
    LATE_START_DATE,
    LATE_END_DATE,
    "late_correlation",
)

correlation_stability = pd.concat(
    [
        early_correlations,
        late_correlations,
    ],
    axis=1,
)

correlation_stability["absolute_change"] = (
    correlation_stability["late_correlation"]
    - correlation_stability["early_correlation"]
).abs()

correlation_stability = (
    correlation_stability
    .sort_values(
        "absolute_change",
        ascending=False,
    )
)


print("\n" + "=" * 70)
print("FEATURE CORRELATION INSTABILITY")
print("=" * 70)

print(
    correlation_stability.head(20)
)


# ============================================================
# DAILY WEIGHT SUMMARY
# ============================================================
# Why:
# Sample weights affect the competition objective and may also
# matter when constructing economically meaningful strategy
# returns later.

daily_weights = (
    training_data
    .group_by(DATE_COLUMN)
    .agg(
        pl.col(WEIGHT_COLUMN).mean().alias("mean_weight"),
        pl.col(WEIGHT_COLUMN).sum().alias("total_weight"),
    )
    .sort(DATE_COLUMN)
    .collect()
    .to_pandas()
)


# ============================================================
# PLOT TOTAL WEIGHT THROUGH TIME
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    daily_weights[DATE_COLUMN],
    daily_weights["total_weight"],
)

plt.title("Total Sample Weight by Date")
plt.xlabel("date_id")
plt.ylabel("Total weight")

plt.tight_layout()

plt.savefig(
    FIGURES_PATH / "daily_total_weight.png",
    dpi=150,
)

plt.close()


# ============================================================
# SAVE EDA TABLES
# ============================================================

metrics_path = PROJECT_ROOT / "outputs" / "metrics"

metrics_path.mkdir(
    parents=True,
    exist_ok=True,
)

daily_target.write_csv(
    metrics_path / "daily_target_summary.csv"
)

symbol_summary.write_csv(
    metrics_path / "symbol_target_summary.csv"
)

feature_correlation_frame.to_csv(
    metrics_path / "feature_target_correlations.csv",
    index=False,
)

correlation_stability.to_csv(
    metrics_path / "feature_correlation_stability.csv"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(
    f"Daily target observations: "
    f"{len(daily_target_pd):,}"
)

print(
    f"Symbols analyzed: "
    f"{len(symbol_summary_pd):,}"
)

print("\nSaved figures to:")
print(FIGURES_PATH)

print("\nSaved EDA tables to:")
print(metrics_path)
