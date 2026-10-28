# ============================================================
# IMPORTS
# ============================================================
# This script generates the final consolidated diagnostic figures
# for the project README/writeup, built entirely from the CSV
# artifacts already saved by earlier scripts (no retraining, no new
# data loading beyond what those scripts already produced).
#
# Figures produced (outputs/figures/final_*.png):
#
#   1. Walk-forward weighted R² and IC by block
#   2. Cumulative walk-forward performance
#   3. Fold 3 vs Fold 4 degradation
#   4. Training-window comparison (expanding vs rolling)
#   5. Gap-stress comparison
#   6. Single-symbol Sharpe distribution vs portfolio Sharpe
#   7. Transaction-cost sensitivity curve
#   8. Top stable feature importances across temporal folds
#
# Styling decisions:
# - A single fixed categorical color per recurring entity (e.g.
#   "fold_3" is always the same color everywhere it appears; each
#   sizing rule keeps one color across every strategy chart) rather
#   than re-cycling colors per chart.
# - Sequential/magnitude-only bars (one series, no categorical
#   distinction) use a single hue, not a rainbow.
# - No dual-axis charts. Two subplots are used instead wherever two
#   different-scale measures needed to be shown together.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    FIGURES_PATH,
    METRICS_PATH,
)


FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FIXED COLOR PALETTE
# ============================================================
# A small, consistent qualitative palette reused by entity across
# every figure, plus one sequential blue for single-series
# magnitude bars.

COLOR_BLUE = "#4C72B0"
COLOR_ORANGE = "#DD8452"
COLOR_GREEN = "#55A868"
COLOR_RED = "#C44E52"
COLOR_PURPLE = "#8172B2"
COLOR_GRAY = "#7F7F7F"

STRATEGY_COLORS = {
    "sign": COLOR_BLUE,
    "proportional": COLOR_ORANGE,
    "clipped_proportional": COLOR_GREEN,
}

FOLD_COLORS = {
    "fold_3": COLOR_BLUE,
    "fold_4": COLOR_ORANGE,
}

WINDOW_COLORS = {
    "expanding": COLOR_BLUE,
    "rolling_700": COLOR_GREEN,
    "rolling_900": COLOR_ORANGE,
}

GAP_COLORS = {
    "no_gap": COLOR_BLUE,
    "gap_stress": COLOR_RED,
}

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["font.size"] = 11


# ============================================================
# FIGURE 1: WALK-FORWARD R² AND IC BY BLOCK
# ============================================================

block_results = pd.read_csv(
    METRICS_PATH / "walk_forward_block_results.csv"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5),
)

axes[0].bar(
    block_results["block_number"],
    block_results["weighted_zero_mean_r2"],
    color=COLOR_BLUE,
)
axes[0].axhline(0, color=COLOR_GRAY, linewidth=0.8)
axes[0].set_title("Walk-Forward Weighted Zero-Mean R² by Block")
axes[0].set_xlabel("Block")
axes[0].set_ylabel("Weighted zero-mean R²")
axes[0].set_xticks(block_results["block_number"])

axes[1].bar(
    block_results["block_number"],
    block_results["information_coefficient"],
    color=COLOR_GREEN,
)
axes[1].axhline(0, color=COLOR_GRAY, linewidth=0.8)
axes[1].set_title("Walk-Forward Information Coefficient by Block")
axes[1].set_xlabel("Block")
axes[1].set_ylabel("Information coefficient")
axes[1].set_xticks(block_results["block_number"])

fig.suptitle(
    "12 sequential out-of-sample walk-forward blocks "
    "(date_id 1099 → 1698), fresh model retrained per block",
    y=1.02,
    fontsize=10,
    color=COLOR_GRAY,
)

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "final_walk_forward_r2_and_ic_by_block.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# FIGURE 2: CUMULATIVE WALK-FORWARD PERFORMANCE
# ============================================================

cumulative = pd.read_csv(
    METRICS_PATH / "walk_forward_cumulative_performance.csv"
)

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    cumulative["through_block"],
    cumulative["cumulative_weighted_zero_mean_r2"],
    color=COLOR_BLUE,
    marker="o",
    markersize=5,
    linewidth=2,
)
ax.axhline(0, color=COLOR_GRAY, linewidth=0.8, linestyle="--")

ax.set_title(
    "Cumulative Walk-Forward Weighted Zero-Mean R²\n"
    "(concatenation of blocks 1..k, k = 1 → 12)"
)
ax.set_xlabel("Through block k")
ax.set_ylabel("Cumulative weighted zero-mean R²")
ax.set_xticks(cumulative["through_block"])

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "final_cumulative_walk_forward_performance.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# FIGURE 3: FOLD 3 VS FOLD 4 DEGRADATION
# ============================================================

fold_results = pd.read_csv(
    METRICS_PATH / "winner_temporal_cv_results.csv"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 5),
)

fold_colors = [
    FOLD_COLORS[fold]
    for fold in fold_results["fold"]
]

axes[0].bar(
    fold_results["fold"],
    fold_results["weighted_zero_mean_r2"],
    color=fold_colors,
)
axes[0].set_title("Weighted Zero-Mean R²")
axes[0].set_ylabel("Weighted zero-mean R²")

axes[1].bar(
    fold_results["fold"],
    fold_results["information_coefficient"],
    color=fold_colors,
)
axes[1].set_title("Information Coefficient")
axes[1].set_ylabel("Information coefficient")

fig.suptitle(
    "Fold 3 (val 1299–1498) vs Fold 4 (val 1499–1698): "
    "nonstationarity degradation",
    y=1.03,
    fontsize=11,
)

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "final_fold3_vs_fold4_degradation.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# FIGURE 4: TRAINING-WINDOW COMPARISON
# ============================================================

window_results = pd.read_csv(
    METRICS_PATH / "training_window_comparison_results.csv"
)

window_order = [
    "expanding",
    "rolling_700",
    "rolling_900",
]

window_results["window"] = pd.Categorical(
    window_results["window"],
    categories=window_order,
    ordered=True,
)
window_results = window_results.sort_values("window")

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    window_results["window"],
    window_results["weighted_zero_mean_r2"],
    color=[
        WINDOW_COLORS[window]
        for window in window_results["window"]
    ],
)

ax.set_title(
    "Training-Window Comparison\n"
    "(fixed validation 1499–1698)"
)
ax.set_ylabel("Weighted zero-mean R²")
ax.set_xlabel("Training window")

for index, row in window_results.reset_index().iterrows():
    ax.text(
        index,
        row["weighted_zero_mean_r2"] + 0.0001,
        f"{row['weighted_zero_mean_r2']:.5f}",
        ha="center",
        fontsize=9,
        color=COLOR_GRAY,
    )

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "final_training_window_comparison.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# FIGURE 5: GAP-STRESS COMPARISON
# ============================================================

gap_results = pd.read_csv(
    METRICS_PATH / "gap_stress_test_results.csv"
)

fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    gap_results["scenario"],
    gap_results["weighted_zero_mean_r2"],
    color=[
        GAP_COLORS[scenario]
        for scenario in gap_results["scenario"]
    ],
)

ax.set_title(
    "Gap-Stress Test\n"
    "(same ~800-day training volume, fixed validation 1499–1698)"
)
ax.set_ylabel("Weighted zero-mean R²")
ax.set_xlabel("Scenario")

for index, row in gap_results.reset_index().iterrows():
    ax.text(
        index,
        row["weighted_zero_mean_r2"] + 0.0001,
        f"{row['weighted_zero_mean_r2']:.5f}\n(gap={row['gap_days']}d)",
        ha="center",
        fontsize=9,
        color=COLOR_GRAY,
    )

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "final_gap_stress_comparison.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# FIGURE 6: SINGLE-SYMBOL SHARPE DISTRIBUTION VS PORTFOLIO
# ============================================================

symbol_sharpe = pd.read_csv(
    METRICS_PATH / "single_symbol_sharpe_decomposition.csv"
)

portfolio_summary = pd.read_csv(
    METRICS_PATH / "single_symbol_vs_portfolio_summary.csv"
).iloc[0]

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    symbol_sharpe["annualized_sharpe"],
    bins=15,
    color=COLOR_BLUE,
    edgecolor="white",
)

ax.axvline(
    portfolio_summary["median_single_symbol_sharpe"],
    color=COLOR_GREEN,
    linestyle="--",
    linewidth=2,
    label=(
        f"Median single-symbol Sharpe = "
        f"{portfolio_summary['median_single_symbol_sharpe']:.2f}"
    ),
)

ax.axvline(
    portfolio_summary["portfolio_sharpe"],
    color=COLOR_RED,
    linestyle="--",
    linewidth=2,
    label=(
        f"39-symbol portfolio Sharpe = "
        f"{portfolio_summary['portfolio_sharpe']:.2f}"
    ),
)

ax.set_title(
    "Single-Symbol Sharpe Distribution vs. Cross-Sectional "
    "Portfolio Sharpe\n(clipped_proportional sizing)"
)
ax.set_xlabel("Annualized Sharpe (single symbol, in isolation)")
ax.set_ylabel("Number of symbols")
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "final_single_symbol_sharpe_distribution.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# FIGURE 7: TRANSACTION-COST SENSITIVITY CURVE
# ============================================================

cost_results = pd.read_csv(
    METRICS_PATH / "transaction_cost_sensitivity_results.csv"
)

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    cost_results["cost_per_unit_turnover"],
    cost_results["annualized_sharpe"],
    color=COLOR_BLUE,
    marker="o",
    markersize=6,
    linewidth=2,
)

ax.axhline(
    0,
    color=COLOR_RED,
    linestyle="--",
    linewidth=1.2,
    label="Sharpe = 0 (breakeven)",
)

# symlog mirrors around zero by default, which would draw a
# negative-axis region with no data (every cost level is >= 0).
# Clipping xlim to the non-negative range removes that misleading
# mirrored half while keeping the log-like spacing for the
# positive values.
ax.set_xscale("symlog", linthresh=0.0001)
ax.set_xlim(
    left=0,
    right=cost_results["cost_per_unit_turnover"].max() * 1.6,
)

ax.set_title(
    "Transaction Cost Sensitivity\n"
    "(clipped_proportional, daily-symbol portfolio, "
    "abstract cost units)"
)
ax.set_xlabel(
    "Cost per unit of daily turnover (log-like scale, "
    "abstract units)"
)
ax.set_ylabel("Annualized Sharpe")
ax.legend(fontsize=9, loc="lower left")

y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min - 1.5, y_max + 3.0)

for _, row in cost_results.iterrows():
    ax.annotate(
        row["cost_level"],
        (
            row["cost_per_unit_turnover"],
            row["annualized_sharpe"],
        ),
        textcoords="offset points",
        xytext=(0, 10),
        ha="left",
        fontsize=7.5,
        color=COLOR_GRAY,
        rotation=40,
    )

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "final_transaction_cost_sensitivity.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# FIGURE 8: TOP STABLE FEATURE IMPORTANCES ACROSS TEMPORAL FOLDS
# ============================================================

importance_stability = pd.read_csv(
    METRICS_PATH / "winner_feature_importance_stability.csv"
)

top_features = (
    importance_stability
    .sort_values("mean_gain", ascending=False)
    .head(15)
    .sort_values("mean_gain", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    top_features["feature"],
    top_features["mean_gain"],
    xerr=top_features["std_gain"],
    color=COLOR_PURPLE,
    ecolor=COLOR_GRAY,
    capsize=3,
)

ax.set_title(
    "Top 15 Features by Mean Gain, Fold 3 + Fold 4\n"
    "(error bars = std across the two folds)"
)
ax.set_xlabel("Mean gain importance")

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "final_top_stable_feature_importance.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved 8 final diagnostic figures to:")
print(FIGURES_PATH)

for figure_name in sorted(FIGURES_PATH.glob("final_*.png")):
    print(f"  {figure_name.name}")

print("\nFinal diagnostics generation complete.")
