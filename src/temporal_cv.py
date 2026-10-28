# ============================================================
# TEMPORAL CROSS-VALIDATION MODULE
# ============================================================
# This module defines leakage-safe expanding-window validation
# splits for the Jane Street time-series forecasting problem.
#
# Why:
# Random K-fold validation is inappropriate because it would mix
# past and future observations. Financial relationships are
# nonstationary, so validation must preserve chronology.
#
# We use expanding windows:
#
#   Fold 1: train on earlier dates -> validate on later dates
#   Fold 2: train on more history -> validate on later dates
#   ...
#
# This approximates how a real forecasting system would be used.


from dataclasses import dataclass


# ============================================================
# TEMPORAL FOLD DEFINITION
# ============================================================
# Each fold contains:
#
# - training start/end date
# - optional gap between training and validation
# - validation start/end date
#
# Why include a gap?
# A gap can test whether the model survives a period of temporal
# separation rather than benefiting only from immediately adjacent
# observations.
#
# Tradeoff:
# Larger gaps are more realistic for some deployment scenarios but
# reduce the amount of data available for fitting.

@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_start_date: int
    train_end_date: int
    validation_start_date: int
    validation_end_date: int

    @property
    def gap_days(self) -> int:
        return (
            self.validation_start_date
            - self.train_end_date
            - 1
        )

    @property
    def training_days(self) -> int:
        return (
            self.train_end_date
            - self.train_start_date
            + 1
        )

    @property
    def validation_days(self) -> int:
        return (
            self.validation_end_date
            - self.validation_start_date
            + 1
        )


# ============================================================
# DEFAULT EXPANDING-WINDOW FOLDS
# ============================================================
# Decision:
# Use later historical periods for validation because they more
# closely resemble the future deployment problem.
#
# We deliberately do not devote large validation windows to the
# earliest part of the dataset, where several features did not
# yet exist.
#
# The folds still preserve those early regimes inside the growing
# training history, allowing the model to experience the dataset's
# real evolution.
#
# Fold design:
#
# Fold 1:
# train 0-898
# validate 899-1098
#
# Fold 2:
# train 0-1098
# validate 1099-1298
#
# Fold 3:
# train 0-1298
# validate 1299-1498
#
# Fold 4:
# train 0-1498
# validate 1499-1698
#
# Each validation period is 200 dates, similar to the validation
# horizon used by strong competition solutions.
#
# Tradeoff:
# Four folds increase runtime. During early experimentation we may
# use only the last two folds, then run all four for final model
# comparisons.

DEFAULT_TEMPORAL_FOLDS = [
    TemporalFold(
        name="fold_1",
        train_start_date=0,
        train_end_date=898,
        validation_start_date=899,
        validation_end_date=1098,
    ),
    TemporalFold(
        name="fold_2",
        train_start_date=0,
        train_end_date=1098,
        validation_start_date=1099,
        validation_end_date=1298,
    ),
    TemporalFold(
        name="fold_3",
        train_start_date=0,
        train_end_date=1298,
        validation_start_date=1299,
        validation_end_date=1498,
    ),
    TemporalFold(
        name="fold_4",
        train_start_date=0,
        train_end_date=1498,
        validation_start_date=1499,
        validation_end_date=1698,
    ),
]


# ============================================================
# DEVELOPMENT FOLDS
# ============================================================
# Decision:
# Provide a smaller fold set for fast CPU experimentation.
#
# Why:
# Running every model on all four folds over tens of millions of
# rows would make iteration unnecessarily slow.
#
# We first use recent folds to reject weak ideas cheaply, then
# promote promising models to full validation.

DEVELOPMENT_TEMPORAL_FOLDS = [
    DEFAULT_TEMPORAL_FOLDS[2],
    DEFAULT_TEMPORAL_FOLDS[3],
]


# ============================================================
# GAP-STRESS FOLD
# ============================================================
# Decision:
# Include one explicit temporal-gap test.
#
# Why:
# The EDA showed regime shifts and unstable feature correlations.
# A model may perform well on an immediately adjacent validation
# period but deteriorate after a large distribution shift.
#
# This stress test imitates a scenario where the model is trained,
# then deployed substantially later without retraining.

GAP_STRESS_FOLD = TemporalFold(
    name="gap_stress",
    train_start_date=0,
    train_end_date=1298,
    validation_start_date=1499,
    validation_end_date=1698,
)


# ============================================================
# FOLD VALIDATION
# ============================================================
# Mechanical integrity checks:
# - dates must be ordered
# - validation must occur after training
# - date ranges must be non-empty

def validate_temporal_fold(
    fold: TemporalFold,
) -> None:

    if fold.train_start_date > fold.train_end_date:
        raise ValueError(
            f"{fold.name}: training start occurs after training end."
        )

    if fold.validation_start_date > fold.validation_end_date:
        raise ValueError(
            f"{fold.name}: validation start occurs after validation end."
        )

    if fold.validation_start_date <= fold.train_end_date:
        raise ValueError(
            f"{fold.name}: validation overlaps training data."
        )


# ============================================================
# VALIDATE FOLD COLLECTION
# ============================================================

def validate_fold_collection(
    folds: list[TemporalFold],
) -> None:

    if not folds:
        raise ValueError(
            "At least one temporal fold is required."
        )

    for fold in folds:
        validate_temporal_fold(fold)


# ============================================================
# PRINT FOLD SUMMARY
# ============================================================
# This helper makes experiment logs easier to interpret.

def print_fold_summary(
    folds: list[TemporalFold],
) -> None:

    validate_fold_collection(folds)

    print("\n" + "=" * 80)
    print("TEMPORAL CROSS-VALIDATION DESIGN")
    print("=" * 80)

    for fold in folds:

        print(f"\n{fold.name}")

        print(
            f"Training:   "
            f"{fold.train_start_date} -> {fold.train_end_date} "
            f"({fold.training_days} dates)"
        )

        print(
            f"Gap:        "
            f"{fold.gap_days} dates"
        )

        print(
            f"Validation: "
            f"{fold.validation_start_date} -> "
            f"{fold.validation_end_date} "
            f"({fold.validation_days} dates)"
        )
