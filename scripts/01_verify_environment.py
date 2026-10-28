# ============================================================
# IMPORTS
# ============================================================
# This script performs a lightweight environment check before we
# begin reading the multi-gigabyte training dataset.

import sys
from pathlib import Path

import lightgbm
import numpy
import pandas
import polars
import pyarrow
import sklearn
import torch
import xgboost


# ============================================================
# ALLOW IMPORTS FROM THE PROJECT ROOT
# ============================================================
# Scripts live inside the scripts folder while reusable modules
# live inside src. Adding the project root to Python's import path
# lets this script import src.config when executed directly.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOAD PROJECT CONFIGURATION
# ============================================================

from src.config import (
    FEATURES_METADATA_PATH,
    LAGS_PATH,
    RESPONDERS_METADATA_PATH,
    TEST_PATH,
    TRAIN_PATH,
)


# ============================================================
# CHECK PYTHON AND PACKAGE VERSIONS
# ============================================================
# Recording package versions is useful for reproducibility.
# Machine-learning libraries can change behavior between releases.

print("\nPython version:")
print(sys.version)

print("\nPackage versions:")
print(f"NumPy:       {numpy.__version__}")
print(f"Pandas:      {pandas.__version__}")
print(f"Polars:      {polars.__version__}")
print(f"PyArrow:     {pyarrow.__version__}")
print(f"Scikit-learn:{sklearn.__version__}")
print(f"XGBoost:     {xgboost.__version__}")
print(f"LightGBM:    {lightgbm.__version__}")
print(f"PyTorch:     {torch.__version__}")


# ============================================================
# CHECK DATASET PATHS
# ============================================================
# Before building the data loader, verify that our configuration
# resolves to the expected Jane Street files and directories.

dataset_paths = {
    "Training data": TRAIN_PATH,
    "Test data": TEST_PATH,
    "Lag data": LAGS_PATH,
    "Feature metadata": FEATURES_METADATA_PATH,
    "Responder metadata": RESPONDERS_METADATA_PATH,
}

print("\nDataset paths:")

for name, path in dataset_paths.items():
    status = "FOUND" if path.exists() else "MISSING"
    print(f"{name:<20} {status:<8} {path}")


# ============================================================
# CHECK GPU AVAILABILITY
# ============================================================
# GPU acceleration is not required for the early data-analysis
# stages. Later, it may substantially reduce training time for
# neural networks and some boosting configurations.

print("\nGPU information:")

if torch.cuda.is_available():
    print("CUDA GPU available")
    print(torch.cuda.get_device_name(0))
else:
    print("No CUDA GPU detected. CPU execution will be used.")


# ============================================================
# FINAL ENVIRONMENT STATUS
# ============================================================

required_paths = [
    TRAIN_PATH,
    TEST_PATH,
    LAGS_PATH,
    FEATURES_METADATA_PATH,
    RESPONDERS_METADATA_PATH,
]

if all(path.exists() for path in required_paths):
    print("\nEnvironment and dataset structure are ready.")
else:
    print("\nEnvironment loaded, but one or more dataset paths are missing.")
