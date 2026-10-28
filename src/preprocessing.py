# ============================================================
# PREPROCESSING MODULE
# ============================================================
# This module contains leakage-safe preprocessing functions for
# the Jane Street forecasting project.
#
# IMPORTANT DESIGN PRINCIPLE:
#
# We do not use one preprocessing pipeline for every model.
#
# Tree models such as LightGBM and XGBoost can natively handle
# missing values and may learn useful missingness structure.
#
# Linear models, KNN, and neural networks generally require
# explicit imputation and may also benefit from standardization.
#
# Therefore, preprocessing behavior is controlled by model type.

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURES WITH STRUCTURAL MISSINGNESS
# ============================================================
# EDA showed that these features contain strong availability
# regime breaks rather than ordinary scattered missing values.
#
# feature_00-04 and feature_21/26/27/31 are completely unavailable
# during portions of the historical sample.
#
# Missingness indicators are therefore potentially useful because
# "feature unavailable" is different from "feature observed at 0".
#
# Tradeoff:
# Availability indicators partially encode historical regime/time.
# This is not look-ahead leakage because availability is observable
# at prediction time, but the relationship may not remain stable
# in future periods. Temporal CV must determine whether the flags
# actually generalize.

STRUCTURAL_MISSING_FEATURES = [
    "feature_00",
    "feature_01",
    "feature_02",
    "feature_03",
    "feature_04",
    "feature_21",
    "feature_26",
    "feature_27",
    "feature_31",
    "feature_39",
    "feature_42",
    "feature_50",
    "feature_53",
    "feature_41",
    "feature_44",
]


# ============================================================
# STRONG AVAILABILITY-BREAK FEATURES
# ============================================================
# These are separated from the softer missingness features because
# they contain nearly discrete historical availability regimes.
#
# This distinction will later let us test whether hard availability
# indicators help without automatically adding indicators for every
# feature containing a null.

HARD_AVAILABILITY_FEATURES = [
    "feature_00",
    "feature_01",
    "feature_02",
    "feature_03",
    "feature_04",
    "feature_21",
    "feature_26",
    "feature_27",
    "feature_31",
]


# ============================================================
# PREPROCESSING CONFIGURATION
# ============================================================
# A dataclass keeps experiment choices explicit and reproducible.
#
# Rather than editing preprocessing logic whenever we want to test
# a new idea, model scripts can create a configuration describing
# the desired preprocessing policy.

@dataclass
class PreprocessingConfig:

    # Add binary columns identifying structural missingness.
    add_missing_indicators: bool = True

    # Fill missing numerical values.
    # Valid options:
    # "zero"
    # "median"
    # "none"
    imputation_method: str = "zero"

    # Standardize numerical predictors.
    standardize: bool = False

    # Include symbol_id as a predictor.
    include_symbol_id: bool = True

    # Include time_id as a predictor.
    include_time_id: bool = True

    # Include date_id as a predictor.
    #
    # We leave this False initially.
    #
    # Why:
    # date_id can allow a model to memorize historical drift rather
    # than learn relationships that generalize.
    #
    # We will explicitly test date_id later under temporal CV.
    include_date_id: bool = False


# ============================================================
# STANDARDIZATION STATE
# ============================================================
# Standardization parameters must be learned ONLY from training
# data.
#
# This object stores the fitted means and standard deviations so
# exactly the same transformation can be applied to validation
# and future observations.
#
# This prevents leakage caused by calculating scaling parameters
# using the entire dataset.

@dataclass
class StandardizationState:
    scaler: StandardScaler
    columns: list[str]


# ============================================================
# ADD MISSINGNESS INDICATORS
# ============================================================
# Decision:
# Create binary flags only for features whose missingness showed
# meaningful structural behavior during EDA.
#
# Why:
# A missing indicator allows models that require imputation to
# distinguish:
#
#     feature value really equals 0
#
# from:
#
#     feature was not available
#
# Tradeoff:
# Adding indicators increases dimensionality and could allow models
# to overfit historical availability regimes.
#
# Their usefulness must therefore be validated temporally.

def add_missing_indicators_polars(
    dataframe: pl.DataFrame,
    features: list[str] = STRUCTURAL_MISSING_FEATURES,
) -> pl.DataFrame:

    available_features = [
        feature
        for feature in features
        if feature in dataframe.columns
    ]

    indicator_expressions = [
        pl.col(feature)
        .is_null()
        .cast(pl.Int8)
        .alias(f"{feature}_missing")
        for feature in available_features
    ]

    return dataframe.with_columns(
        indicator_expressions
    )


# ============================================================
# ZERO IMPUTATION
# ============================================================
# Decision:
# Zero is our initial explicit-imputation baseline.
#
# Why:
# - computationally very cheap
# - does not require fitting statistics
# - works well with standardized or approximately centered features
# - was competitive in strong Jane Street solutions
#
# Tradeoff:
# Zero does not necessarily represent the conditional expectation
# of a missing value.
#
# Missing indicators are therefore retained for structurally
# missing features so the model can distinguish an actual zero
# from an imputed zero.

def zero_impute_polars(
    dataframe: pl.DataFrame,
    feature_columns: list[str],
) -> pl.DataFrame:

    available_features = [
        feature
        for feature in feature_columns
        if feature in dataframe.columns
    ]

    expressions = [
        pl.col(feature)
        .fill_null(0.0)
        .fill_nan(0.0)
        .alias(feature)
        for feature in available_features
    ]

    return dataframe.with_columns(
        expressions
    )


# ============================================================
# CALCULATE TRAINING MEDIANS
# ============================================================
# Decision:
# Median imputation is provided as an alternative baseline.
#
# Why:
# Financial variables can be heavy-tailed. The median is less
# sensitive to extreme observations than the mean.
#
# CRITICAL:
# These medians must be calculated ONLY from the training portion
# of a temporal fold.
#
# Calculating medians over training + validation would leak future
# information into preprocessing.

def calculate_feature_medians(
    training_data: pl.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:

    available_features = [
        feature
        for feature in feature_columns
        if feature in training_data.columns
    ]

    median_frame = (
        training_data
        .select(
            [
                pl.col(feature)
                .median()
                .alias(feature)
                for feature in available_features
            ]
        )
    )

    medians = {}

    for feature in available_features:

        median_value = median_frame[feature][0]

        if median_value is None or np.isnan(median_value):
            median_value = 0.0

        medians[feature] = float(median_value)

    return medians


# ============================================================
# APPLY MEDIAN IMPUTATION
# ============================================================
# The medians supplied here should come from the corresponding
# training fold only.

def median_impute_polars(
    dataframe: pl.DataFrame,
    feature_medians: dict[str, float],
) -> pl.DataFrame:

    expressions = []

    for feature, median_value in feature_medians.items():

        if feature not in dataframe.columns:
            continue

        expressions.append(
            pl.col(feature)
            .fill_null(median_value)
            .fill_nan(median_value)
            .alias(feature)
        )

    return dataframe.with_columns(
        expressions
    )


# ============================================================
# SELECT MODEL FEATURES
# ============================================================
# This function creates the final predictor list used by a model.
#
# We deliberately separate predictor selection from the physical
# transformation of data so model scripts can clearly report which
# variables were included.

def build_model_feature_list(
    base_feature_columns: list[str],
    config: PreprocessingConfig,
) -> list[str]:

    model_features = list(base_feature_columns)

    # --------------------------------------------------------
    # Missing indicator features
    # --------------------------------------------------------

    if config.add_missing_indicators:

        missing_indicator_columns = [
            f"{feature}_missing"
            for feature in STRUCTURAL_MISSING_FEATURES
            if feature in base_feature_columns
        ]

        model_features.extend(
            missing_indicator_columns
        )

    # --------------------------------------------------------
    # Time identifier
    # --------------------------------------------------------
    # time_id may capture stable intraday effects.
    #
    # Unlike an actual timestamp, Jane Street notes that the
    # spacing between time_id values is not guaranteed to be
    # uniform, so we treat it as an ordinal predictor rather than
    # real elapsed time.

    if config.include_time_id:
        model_features.append("time_id")

    # --------------------------------------------------------
    # Symbol identifier
    # --------------------------------------------------------
    # symbol_id may capture persistent differences between
    # instruments.
    #
    # Tradeoff:
    # New symbols can appear in future data, so we should not
    # rely entirely on symbol memorization.

    if config.include_symbol_id:
        model_features.append("symbol_id")

    # --------------------------------------------------------
    # Date identifier
    # --------------------------------------------------------

    if config.include_date_id:
        model_features.append("date_id")

    return model_features


# ============================================================
# APPLY BASIC PREPROCESSING
# ============================================================
# This is the main preprocessing function used before model-
# specific scaling.
#
# It intentionally does NOT calculate any statistics using
# validation/future observations.
#
# If median imputation is requested, precomputed training medians
# must be supplied by the caller.

def preprocess_polars_dataframe(
    dataframe: pl.DataFrame,
    feature_columns: list[str],
    config: PreprocessingConfig,
    feature_medians: dict[str, float] | None = None,
) -> pl.DataFrame:

    processed_data = dataframe

    # --------------------------------------------------------
    # Add structural missingness indicators BEFORE imputation
    # --------------------------------------------------------
    # This must occur before filling nulls because otherwise the
    # original missingness information would be lost.

    if config.add_missing_indicators:
        processed_data = add_missing_indicators_polars(
            processed_data,
        )

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    if config.imputation_method == "zero":

        processed_data = zero_impute_polars(
            processed_data,
            feature_columns,
        )

    elif config.imputation_method == "median":

        if feature_medians is None:
            raise ValueError(
                "Median imputation requires feature_medians "
                "calculated from training data."
            )

        processed_data = median_impute_polars(
            processed_data,
            feature_medians,
        )

    elif config.imputation_method == "none":
        # No transformation is required.
        #
        # This option is particularly useful for LightGBM and
        # XGBoost, which can natively handle missing values.
        pass

    else:
        raise ValueError(
            "Unknown imputation_method. "
            "Use 'zero', 'median', or 'none'."
        )

    return processed_data


# ============================================================
# FIT STANDARDIZATION
# ============================================================
# Decision:
# Scaling is fitted only when required by the model.
#
# Useful for:
# - Ridge
# - Lasso
# - Elastic Net
# - KNN
# - neural networks
#
# Usually unnecessary for:
# - LightGBM
# - XGBoost
#
# Why:
# Distance-based and regularized models are sensitive to feature
# scale, whereas tree splits are not.
#
# Tradeoff:
# Standardization requires another fitted preprocessing state and
# increases computational cost.

def fit_standardizer(
    training_data: pd.DataFrame,
    columns: list[str],
) -> StandardizationState:

    scaler = StandardScaler()

    scaler.fit(
        training_data[columns]
    )

    return StandardizationState(
        scaler=scaler,
        columns=columns,
    )


# ============================================================
# APPLY STANDARDIZATION
# ============================================================
# The same scaler fitted on the training period must be reused
# unchanged for validation and walk-forward observations.

def apply_standardizer(
    dataframe: pd.DataFrame,
    state: StandardizationState,
) -> pd.DataFrame:

    processed_data = dataframe.copy()

    processed_data[state.columns] = (
        state.scaler.transform(
            processed_data[state.columns]
        )
    )

    return processed_data


# ============================================================
# MODEL-SPECIFIC DEFAULT CONFIGURATIONS
# ============================================================
# These configurations give us sensible baselines while keeping
# every assumption explicit.
#
# We will still test alternatives during temporal CV.

def get_tree_model_config() -> PreprocessingConfig:

    # LightGBM/XGBoost can handle missing values directly.
    #
    # We initially retain native missing values instead of forcing
    # an arbitrary replacement value.

    return PreprocessingConfig(
        add_missing_indicators=True,
        imputation_method="none",
        standardize=False,
        include_symbol_id=True,
        include_time_id=True,
        include_date_id=False,
    )


def get_linear_model_config() -> PreprocessingConfig:

    # Linear models cannot operate with NaNs.
    #
    # Zero imputation plus missing indicators gives us a simple,
    # computationally efficient baseline.

    return PreprocessingConfig(
        add_missing_indicators=True,
        imputation_method="zero",
        standardize=True,
        include_symbol_id=True,
        include_time_id=True,
        include_date_id=False,
    )


def get_knn_model_config() -> PreprocessingConfig:

    # KNN depends directly on distances between observations.
    #
    # Missing values therefore require imputation and predictors
    # must be standardized so large-scale features do not dominate
    # the distance calculation.
    #
    # NOTE:
    # KNN is unlikely to scale well to 47 million rows.
    # We retain it as an educational/small-sample benchmark rather
    # than expecting it to become the production model.

    return PreprocessingConfig(
        add_missing_indicators=True,
        imputation_method="zero",
        standardize=True,
        include_symbol_id=True,
        include_time_id=True,
        include_date_id=False,
    )


def get_neural_network_config() -> PreprocessingConfig:

    # Neural networks generally train more reliably on finite,
    # standardized numerical inputs.

    return PreprocessingConfig(
        add_missing_indicators=True,
        imputation_method="zero",
        standardize=True,
        include_symbol_id=True,
        include_time_id=True,
        include_date_id=False,
    )
