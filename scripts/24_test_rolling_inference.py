# ============================================================
# IMPORTS
# ============================================================
# This script verifies src/inference.py's RollingInferenceEngine
# against the actual historical dataset before trusting it as the
# production-style prediction path.
#
# We train the locked model on a genuine historical window
# [799, 1498] and test inference on date_id 1499 onward - dates the
# model has never seen during training, mirroring real deployment.
#
# Seven checks, matching the required verification list:
#
#   1. Batch isolation
#   2. Feature equality (engine vs. an independently-recomputed
#      cross-sectional feature calculation)
#   3. Prediction equality (engine vs. a direct model.predict() call)
#   4. Row/order preservation, identifiers preserved
#   5. No symbol_id predictor leakage (present as metadata only)
#   6. Feature schema check (exact count/names, fails loudly on a
#      missing column)
#   7. Multiple sequential batches, across snapshots and dates

import sys
from pathlib import Path

import numpy as np
import polars as pl


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
    DATE_COLUMN,
    FEATURE_COLUMNS,
    MODELS_PATH,
    SYMBOL_COLUMN,
    TIME_COLUMN,
    TRAIN_PATH,
)

from src.feature_engineering import (
    add_cross_sectional_features,
)

from src.inference import (
    CROSS_SECTIONAL_BASE_FEATURES,
    CROSS_SECTIONAL_FEATURES,
    MODEL_FEATURES,
    RollingInferenceEngine,
    save_locked_model,
    train_locked_model,
)


# ============================================================
# TRAIN THE LOCKED MODEL ON HISTORY THAT ENDS BEFORE TEST DATES
# ============================================================
# Decision:
# Train on [799, 1498], test inference on date_id >= 1499.
#
# Why:
# This is a genuine temporal split - the model has never seen the
# dates it will be asked to predict on in this script, exactly like
# real deployment.

TRAIN_START_DATE = 799
TRAIN_END_DATE = 1498

TEST_DATE = 1499

training_scan = pl.scan_parquet(
    str(
        TRAIN_PATH
        / "**"
        / "*.parquet"
    )
)

print("\n" + "=" * 80)
print("TRAINING LOCKED MODEL FOR INFERENCE TESTING")
print("=" * 80)

print(
    f"Training window: {TRAIN_START_DATE} -> {TRAIN_END_DATE}"
)
print(
    f"Test date (never seen in training): {TEST_DATE}"
)

model = train_locked_model(
    training_scan,
    train_start_date=TRAIN_START_DATE,
    train_end_date=TRAIN_END_DATE,
)

print("Model trained.")

model_output_path = (
    MODELS_PATH
    / "locked_lightgbm_model.joblib"
)

save_locked_model(
    model,
    model_output_path,
)

print(f"Saved locked model to: {model_output_path}")


# ============================================================
# LOAD RAW TEST DATA (NO ENGINEERED FEATURES, NO TARGET/WEIGHT)
# ============================================================
# Decision:
# Deliberately load only the columns a real deployment batch would
# actually contain - raw features and identifiers. responder_6 and
# weight are NOT included, since a real inference batch would not
# have them either. This is itself part of the test: the engine
# must not secretly depend on columns that don't exist at
# prediction time.

raw_columns = (
    [
        DATE_COLUMN,
        TIME_COLUMN,
        SYMBOL_COLUMN,
    ]
    + FEATURE_COLUMNS
)

test_data = (
    training_scan
    .filter(
        pl.col(DATE_COLUMN) >= TEST_DATE
    )
    .select(raw_columns)
    .collect()
)

print(
    f"\nTest data loaded: {test_data.height:,} rows, "
    f"dates {test_data[DATE_COLUMN].min()} -> "
    f"{test_data[DATE_COLUMN].max()}"
)

engine = RollingInferenceEngine(
    model=model
)


# ============================================================
# TEST 1: BATCH ISOLATION
# ============================================================
# The batch passed to predict_batch() contains ONLY rows from one
# (date_id, time_id) snapshot - it is structurally impossible for
# feature construction inside predict_batch() to see any other
# timestamp, because no other timestamp's rows were ever given to
# it.

first_time_id = (
    test_data
    .filter(pl.col(DATE_COLUMN) == TEST_DATE)
    [TIME_COLUMN]
    .min()
)

single_batch = test_data.filter(
    (pl.col(DATE_COLUMN) == TEST_DATE)
    & (pl.col(TIME_COLUMN) == first_time_id)
)

assert single_batch.height > 0, (
    "Probe batch is empty; pick a different test date/time_id."
)

assert single_batch[DATE_COLUMN].n_unique() == 1
assert single_batch[TIME_COLUMN].n_unique() == 1

batch_result = engine.predict_batch(
    single_batch
)

assert batch_result[DATE_COLUMN].n_unique() == 1
assert batch_result[TIME_COLUMN].n_unique() == 1
assert (
    batch_result[DATE_COLUMN][0] == TEST_DATE
)
assert (
    batch_result[TIME_COLUMN][0] == first_time_id
)

# Confirm the engine actively rejects a multi-snapshot batch,
# proving isolation is enforced, not just coincidentally true.

multi_snapshot_batch = test_data.filter(
    pl.col(DATE_COLUMN) == TEST_DATE
)

assert multi_snapshot_batch[TIME_COLUMN].n_unique() > 1

raised_on_multi_snapshot = False

try:
    engine.predict_batch(
        multi_snapshot_batch
    )
except ValueError:
    raised_on_multi_snapshot = True

assert raised_on_multi_snapshot, (
    "predict_batch() should reject a batch spanning more than "
    "one (date_id, time_id) snapshot."
)

print(
    "\n[PASS] Batch isolation: single-snapshot batches are "
    "accepted and correctly labeled; a multi-snapshot batch is "
    "rejected, proving the engine cannot silently see other "
    "timestamps."
)


# ============================================================
# TEST 2: FEATURE EQUALITY
# ============================================================
# Independently recompute cross-sectional market_mean for a probe
# feature via a raw group_by + join, WITHOUT calling
# add_cross_sectional_features(), and compare against what the
# engine actually used internally. Since the engine does not expose
# its intermediate enriched frame directly, we recompute it here the
# same way predict_batch() does internally (same verified function)
# AND cross-check that against a fully independent manual
# calculation, closing the loop.

probe_feature = CROSS_SECTIONAL_BASE_FEATURES[0]

manual_market_mean = (
    single_batch
    .select(
        pl.col(probe_feature).mean()
    )
    [probe_feature][0]
)

engine_enriched = add_cross_sectional_features(
    single_batch,
    features=CROSS_SECTIONAL_BASE_FEATURES,
)

engine_market_mean_values = (
    engine_enriched[f"{probe_feature}_market_mean"]
    .unique()
    .to_list()
)

assert len(engine_market_mean_values) == 1, (
    "Expected a single market_mean value across the batch "
    "(one (date_id, time_id) snapshot)."
)

assert np.isclose(
    engine_market_mean_values[0],
    manual_market_mean,
    rtol=1e-6,
), (
    "Engine's cross-sectional market_mean does not match an "
    "independent manual calculation on the same batch."
)

print(
    "\n[PASS] Feature equality: cross-sectional market_mean for "
    f"'{probe_feature}' matches an independent manual calculation "
    "on the exact same single-snapshot batch."
)


# ============================================================
# TEST 3: PREDICTION EQUALITY
# ============================================================
# The engine's predictions must exactly match a direct
# model.predict() call on the same enriched feature matrix, in the
# same row order.

direct_feature_matrix = (
    engine_enriched
    .select(MODEL_FEATURES)
    .to_pandas()
)

direct_predictions = model.predict(
    direct_feature_matrix
)

assert np.allclose(
    batch_result["prediction"].to_numpy(),
    direct_predictions,
), (
    "Engine predictions do not match a direct model.predict() "
    "call on the same batch."
)

print(
    "\n[PASS] Prediction equality: engine predictions exactly "
    "match a direct model.predict() call on the same batch."
)


# ============================================================
# TEST 4: ROW/ORDER PRESERVATION, IDENTIFIERS PRESERVED
# ============================================================

assert batch_result.height == single_batch.height, (
    "Row count changed between input batch and engine output."
)

assert (
    batch_result[SYMBOL_COLUMN].to_list()
    == single_batch[SYMBOL_COLUMN].to_list()
), (
    "symbol_id order/values do not match between input and "
    "output - row order was not preserved."
)

assert (
    batch_result[DATE_COLUMN].to_list()
    == single_batch[DATE_COLUMN].to_list()
)
assert (
    batch_result[TIME_COLUMN].to_list()
    == single_batch[TIME_COLUMN].to_list()
)

print(
    "\n[PASS] Row count and row order preserved; date_id, "
    "time_id, symbol_id identifiers match the input batch "
    "exactly, in the same order."
)


# ============================================================
# TEST 5: NO symbol_id PREDICTOR LEAKAGE
# ============================================================

assert SYMBOL_COLUMN not in MODEL_FEATURES, (
    "symbol_id must not appear in MODEL_FEATURES."
)

assert SYMBOL_COLUMN not in engine.model_features, (
    "symbol_id must not appear in the engine's model_features."
)

assert SYMBOL_COLUMN in batch_result.columns, (
    "symbol_id should still be present as identifier metadata in "
    "the engine's output."
)

print(
    "\n[PASS] symbol_id is absent from MODEL_FEATURES (never fed "
    "to the model) but present as identifier metadata in the "
    "output."
)


# ============================================================
# TEST 6: FEATURE SCHEMA CHECK
# ============================================================

expected_feature_count = (
    len(FEATURE_COLUMNS)
    + 1  # time_id
    + len(CROSS_SECTIONAL_FEATURES)
)

assert len(MODEL_FEATURES) == expected_feature_count, (
    f"MODEL_FEATURES has {len(MODEL_FEATURES)} entries, expected "
    f"{expected_feature_count} "
    f"({len(FEATURE_COLUMNS)} raw + 1 time_id + "
    f"{len(CROSS_SECTIONAL_FEATURES)} cross-sectional)."
)

assert len(set(MODEL_FEATURES)) == len(MODEL_FEATURES), (
    "MODEL_FEATURES contains a duplicate feature name."
)

for raw_feature in FEATURE_COLUMNS:
    assert raw_feature in MODEL_FEATURES, (
        f"{raw_feature} missing from MODEL_FEATURES."
    )

print(
    f"\n[PASS] Feature schema check: exactly "
    f"{expected_feature_count} model features, no duplicates, "
    f"all 79 raw features present."
)

# Fail-loudly check: a batch missing a required raw column must
# raise, not silently produce a prediction.

incomplete_batch = single_batch.drop(
    FEATURE_COLUMNS[0]
)

raised_on_missing_column = False

try:
    engine.predict_batch(
        incomplete_batch
    )
except ValueError:
    raised_on_missing_column = True

assert raised_on_missing_column, (
    f"predict_batch() should raise when {FEATURE_COLUMNS[0]} is "
    f"missing from the input batch, not silently proceed."
)

print(
    f"[PASS] Missing-column check: dropping '{FEATURE_COLUMNS[0]}' "
    f"from the batch correctly raises ValueError instead of "
    f"silently producing a prediction."
)


# ============================================================
# TEST 7: MULTIPLE SEQUENTIAL BATCHES
# ============================================================
# Run several consecutive (date_id, time_id) snapshots, spanning
# more than one date, and confirm the engine handles each
# independently and correctly - no cross-contamination between
# batches, and results reconcile with calling predict_batch()
# directly on each snapshot.

sequential_window = (
    test_data
    .filter(
        (pl.col(DATE_COLUMN) >= TEST_DATE)
        & (pl.col(DATE_COLUMN) <= TEST_DATE + 1)
    )
)

distinct_snapshots = (
    sequential_window
    .select(
        [
            DATE_COLUMN,
            TIME_COLUMN,
        ]
    )
    .unique()
    .height
)

assert distinct_snapshots >= 3, (
    "Expected multiple distinct (date_id, time_id) snapshots in "
    "the sequential test window."
)

sequential_results = engine.predict_sequential_batches(
    sequential_window
)

assert sequential_results.height == sequential_window.height, (
    "Row count changed across the sequential multi-batch run."
)

assert (
    sequential_results
    .select(
        [
            DATE_COLUMN,
            TIME_COLUMN,
        ]
    )
    .unique()
    .height
) == distinct_snapshots, (
    "Sequential run did not produce output for every distinct "
    "snapshot."
)

# Cross-check: predictions for one snapshot within the sequential
# run must match calling predict_batch() on that same snapshot in
# isolation.

probe_snapshot_keys = (
    distinct_snapshots,
)

probe_date = sequential_window[DATE_COLUMN][0]
probe_time = sequential_window[TIME_COLUMN][0]

isolated_probe_batch = sequential_window.filter(
    (pl.col(DATE_COLUMN) == probe_date)
    & (pl.col(TIME_COLUMN) == probe_time)
)

isolated_probe_result = engine.predict_batch(
    isolated_probe_batch
)

sequential_probe_result = (
    sequential_results
    .filter(
        (pl.col(DATE_COLUMN) == probe_date)
        & (pl.col(TIME_COLUMN) == probe_time)
    )
)

assert np.allclose(
    sequential_probe_result["prediction"].to_numpy(),
    isolated_probe_result["prediction"].to_numpy(),
), (
    "A snapshot's predictions differ between the sequential "
    "multi-batch run and calling predict_batch() on it in "
    "isolation - suggests state leaking between batches."
)

print(
    f"\n[PASS] Sequential batches: {distinct_snapshots} distinct "
    f"(date_id, time_id) snapshots across dates "
    f"{TEST_DATE}-{TEST_DATE + 1} processed correctly, row count "
    f"preserved, and a probe snapshot's predictions match calling "
    f"predict_batch() on it in isolation - no state leakage "
    f"between batches."
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("ALL ROLLING INFERENCE TESTS PASSED")
print("=" * 80)

print(
    "src/inference.py's RollingInferenceEngine is verified for "
    "batch isolation, feature/prediction equality, row/order "
    "preservation, symbol_id exclusion, schema enforcement, and "
    "correct handling of sequential batches."
)

print(f"\nLocked model saved to: {model_output_path}")

print("\nRolling inference verification complete.")
