# ============================================================
# INFERENCE MODULE
# ============================================================
# This module implements the production-style rolling inference
# pipeline for the locked forecasting specification:
#
#   Model:      LightGBM (locked hyperparameters)
#   Features:   79 raw features + cross-sectional market features
#               + time_id, symbol_id EXCLUDED
#
# Why this is simpler than it could have been:
# The compact temporal feature block (multi-day lag/rolling history)
# was tested and rejected (script 14) for adding no value on top of
# the cross-sectional feature set. Because the locked model
# therefore depends on no multi-day history store, a single
# incoming (date_id, time_id) cross-sectional batch already
# contains everything the model needs: that batch's raw features
# and that batch's own cross-section. No state needs to be carried
# between batches.
#
# Information flow reproduced by this module:
#
#   new (date_id, time_id) batch arrives
#         |
#   construct contemporaneous cross-sectional features
#   (mean/deviation across symbols WITHIN this exact batch only -
#    the same verified function used throughout training, so
#    training and inference can never silently drift apart)
#         |
#   select the locked MODEL_FEATURES, in the locked order
#         |
#   run the locked LightGBM model
#         |
#   return one prediction per input row, with identifiers preserved


from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.config import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    SYMBOL_COLUMN,
    TIME_COLUMN,
)
from src.feature_engineering import (
    CORE_ENGINEERING_FEATURES,
    add_cross_sectional_features,
    cross_sectional_feature_names,
)
from src.tree_models import create_lightgbm_model


# ============================================================
# LOCKED MODEL FEATURE SCHEMA
# ============================================================
# This is the single source of truth for the locked feature list.
# Every training/evaluation script since script 12 (symbol
# ablation) has used this exact feature family; this module reuses
# the same construction rather than retyping it, so training and
# inference cannot silently drift apart.

CROSS_SECTIONAL_BASE_FEATURES = CORE_ENGINEERING_FEATURES

CROSS_SECTIONAL_FEATURES = cross_sectional_feature_names(
    CROSS_SECTIONAL_BASE_FEATURES
)

MODEL_FEATURES = (
    FEATURE_COLUMNS
    + [TIME_COLUMN]
    + CROSS_SECTIONAL_FEATURES
)

assert len(set(MODEL_FEATURES)) == len(MODEL_FEATURES), (
    "MODEL_FEATURES contains a duplicate feature name."
)

assert SYMBOL_COLUMN not in MODEL_FEATURES, (
    "symbol_id must never be a model feature - the symbol "
    "ablation experiment (script 12) showed it hurts "
    "generalization despite dominating in-sample gain."
)


# ============================================================
# TRAIN THE LOCKED MODEL FOR PRODUCTION-STYLE INFERENCE
# ============================================================
# Decision:
# Unlike every development/evaluation script, a model trained for
# actual deployment cannot use early stopping against a held-out
# validation window, because a deployed model should be trained on
# ALL available recent history up to the moment it goes live - there
# is no future slice left to hold out.
#
# Instead we fix n_estimators at a value informed by the best
# iterations observed across scripts 16-19 (typically in the
# 250-700 range for a ~700-date training window), rather than the
# 1500-tree ceiling used during development. Every other
# hyperparameter matches the locked specification exactly.

PRODUCTION_N_ESTIMATORS = 500


def train_locked_model(
    training_scan: pl.LazyFrame,
    train_start_date: int,
    train_end_date: int,
    date_column: str = DATE_COLUMN,
    time_column: str = TIME_COLUMN,
    target_column: str = "responder_6",
    weight_column: str = "weight",
    time_stride: int = 40,
    random_seed: int = 42,
    n_estimators: int = PRODUCTION_N_ESTIMATORS,
):
    """
    Train the locked LightGBM specification on a date range, using
    the same time_id stride convention as every prior CPU-friendly
    experiment. Returns the fitted model.
    """

    enriched_scan = add_cross_sectional_features(
        training_scan,
        features=CROSS_SECTIONAL_BASE_FEATURES,
    )

    training_data = (
        enriched_scan
        .filter(
            (pl.col(date_column) >= train_start_date)
            & (pl.col(date_column) <= train_end_date)
            & (pl.col(time_column) % time_stride == 0)
        )
        .select(
            MODEL_FEATURES
            + [
                target_column,
                weight_column,
            ]
        )
        .collect()
    )

    X_train = (
        training_data
        .select(MODEL_FEATURES)
        .to_pandas()
    )
    y_train = training_data[target_column].to_numpy()
    train_weights = training_data[weight_column].to_numpy()

    model = create_lightgbm_model(
        n_estimators=n_estimators,
        random_seed=random_seed,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
    )

    return model


def save_locked_model(
    model,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "model_features": MODEL_FEATURES,
        },
        output_path,
    )


def load_locked_model(
    input_path: Path,
):

    payload = joblib.load(
        input_path
    )

    return payload["model"], payload["model_features"]


# ============================================================
# ROLLING INFERENCE ENGINE
# ============================================================
# Decision:
# predict_batch() deliberately REJECTS any input containing more
# than one (date_id, time_id) snapshot.
#
# Why:
# This mirrors how Jane Street's real evaluation API delivers data:
# one cross-sectional batch at a time. Enforcing it here means the
# cross-sectional feature construction inside predict_batch is
# structurally incapable of seeing any other timestamp, past or
# future - there is nothing else in the frame for it to see. This
# is a stronger isolation guarantee than a runtime check on dates
# actually used; it is a guarantee about what data physically
# reaches the function at all.

@dataclass
class RollingInferenceEngine:

    model: object
    model_features: list[str] = field(
        default_factory=lambda: list(MODEL_FEATURES)
    )

    def __post_init__(self):
        self._validate_schema()

    def _validate_schema(self):

        if len(set(self.model_features)) != len(self.model_features):
            raise ValueError(
                "model_features contains a duplicate feature name."
            )

        if SYMBOL_COLUMN in self.model_features:
            raise ValueError(
                f"{SYMBOL_COLUMN} must not be a model feature."
            )

    def predict_batch(
        self,
        batch: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Predict on exactly one (date_id, time_id) cross-sectional
        batch. Raises if the batch spans more than one snapshot, or
        if a required raw feature column is missing.

        Returns a DataFrame with date_id, time_id, symbol_id (kept
        as identifier metadata, never fed to the model) and
        prediction, in the same row order as the input batch.
        """

        unique_snapshots = (
            batch
            .select(
                [
                    DATE_COLUMN,
                    TIME_COLUMN,
                ]
            )
            .unique()
            .height
        )

        if unique_snapshots != 1:
            raise ValueError(
                f"predict_batch() requires exactly one "
                f"(date_id, time_id) snapshot, found "
                f"{unique_snapshots}. This engine never accepts "
                f"multi-timestamp input, by design."
            )

        missing_raw_columns = [
            column
            for column in FEATURE_COLUMNS
            + [
                DATE_COLUMN,
                TIME_COLUMN,
                SYMBOL_COLUMN,
            ]
            if column not in batch.columns
        ]

        if missing_raw_columns:
            raise ValueError(
                f"Batch is missing required raw columns: "
                f"{missing_raw_columns}"
            )

        enriched = add_cross_sectional_features(
            batch,
            features=CROSS_SECTIONAL_BASE_FEATURES,
        )

        missing_model_features = [
            feature
            for feature in self.model_features
            if feature not in enriched.columns
        ]

        if missing_model_features:
            raise ValueError(
                f"Missing required model features after "
                f"cross-sectional enrichment: "
                f"{missing_model_features}"
            )

        feature_matrix = (
            enriched
            .select(self.model_features)
            .to_pandas()
        )

        predictions = self.model.predict(
            feature_matrix
        )

        return (
            enriched
            .select(
                [
                    DATE_COLUMN,
                    TIME_COLUMN,
                    SYMBOL_COLUMN,
                ]
            )
            .with_columns(
                pl.Series(
                    "prediction",
                    predictions,
                )
            )
        )

    def predict_sequential_batches(
        self,
        dataframe: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Convenience driver: splits `dataframe` into its constituent
        (date_id, time_id) snapshots, in chronological order, and
        calls predict_batch() on each one independently -
        mirroring "advance to next timestamp" in the real deployment
        loop. Never passes more than one snapshot to any single
        predict_batch() call.
        """

        snapshot_keys = (
            dataframe
            .select(
                [
                    DATE_COLUMN,
                    TIME_COLUMN,
                ]
            )
            .unique()
            .sort(
                [
                    DATE_COLUMN,
                    TIME_COLUMN,
                ]
            )
        )

        batch_results = []

        for key_row in snapshot_keys.iter_rows(named=True):

            batch = dataframe.filter(
                (pl.col(DATE_COLUMN) == key_row[DATE_COLUMN])
                & (pl.col(TIME_COLUMN) == key_row[TIME_COLUMN])
            )

            batch_results.append(
                self.predict_batch(batch)
            )

        return pl.concat(
            batch_results
        )
