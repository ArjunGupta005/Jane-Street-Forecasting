# ============================================================
# PROJECT CONFIGURATION
# ============================================================
# This file stores shared paths and core dataset definitions.
#
# Why:
# Centralizing configuration prevents file paths, feature names,
# and experiment settings from being duplicated across scripts.
# If the project structure changes later, we only need to update
# the relevant value here rather than editing many files.


from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================
# Path(__file__) points to this config.py file.
# parent gives the src folder.
# parent.parent gives the JaneStreet2 project root.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "train.parquet"
TEST_PATH = PROJECT_ROOT / "test.parquet"
LAGS_PATH = PROJECT_ROOT / "lags.parquet"

FEATURES_METADATA_PATH = PROJECT_ROOT / "features.csv"
RESPONDERS_METADATA_PATH = PROJECT_ROOT / "responders.csv"
SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "sample_submission.csv"

MODELS_PATH = PROJECT_ROOT / "models"
CACHE_PATH = PROJECT_ROOT / "cache"

OUTPUTS_PATH = PROJECT_ROOT / "outputs"
FIGURES_PATH = OUTPUTS_PATH / "figures"
METRICS_PATH = OUTPUTS_PATH / "metrics"
PREDICTIONS_PATH = OUTPUTS_PATH / "predictions"


# ============================================================
# TARGET AND IDENTIFIER COLUMNS
# ============================================================
# responder_6 is the competition forecasting target.
#
# date_id and time_id jointly define chronological ordering.
# symbol_id identifies the anonymized financial instrument.
# weight is required for Jane Street's weighted zero-mean R2.

TARGET_COLUMN = "responder_6"

DATE_COLUMN = "date_id"
TIME_COLUMN = "time_id"
SYMBOL_COLUMN = "symbol_id"
WEIGHT_COLUMN = "weight"


# ============================================================
# FEATURE COLUMNS
# ============================================================
# The competition contains 79 anonymized predictors numbered
# feature_00 through feature_78.

FEATURE_COLUMNS = [
    f"feature_{number:02d}"
    for number in range(79)
]


# ============================================================
# RESPONDER COLUMNS
# ============================================================
# responder_6 is the primary target, but the remaining responders
# may later be useful for diagnostics or auxiliary-target models.

RESPONDER_COLUMNS = [
    f"responder_{number}"
    for number in range(9)
]


# ============================================================
# REPRODUCIBILITY
# ============================================================
# A fixed seed allows comparable experiments when algorithms use
# randomness. This does not make every GPU operation deterministic,
# but it gives us a consistent baseline for model comparisons.

RANDOM_SEED = 42
