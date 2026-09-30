"""
RF Vision — Random Forest Trainer

Reads:
    data/idle_*.csv
    data/motion_*.csv

Creates timestamp-aware 8 Hz windows.
Uses SESSION-LEVEL splitting so windows from the same recording 
never leak into both train and test sets.

Outputs:
    rf_model.joblib
    model_config.json
"""

import os
import glob
import json
import ast

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from csi_features import (
    SAMPLE_RATE_HZ,
    WINDOW_SECONDS,
    WINDOW_SAMPLES,
    CSI_PAIR_INDICES,
    FEATURE_NAMES,
    extract_selected_amplitudes,
    build_feature_windows,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

MODEL_FILE = "rf_model.joblib"
CONFIG_FILE = "model_config.json"

RANDOM_STATE = 42
TEST_SESSION_FRACTION = 0.20


# ============================================================
# LOAD ONE SESSION
# ============================================================

def load_session(filename):
    df = pd.read_csv(filename)

    required = {
        "timestamp_us",
        "label",
        "csi_hex",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} missing columns: {missing}")

    timestamps = df["timestamp_us"].to_numpy(dtype=np.float64) / 1_000_000.0
    labels = df["label"].to_numpy(dtype=int)

    if len(np.unique(labels)) != 1:
        raise ValueError(f"{filename} contains multiple labels")

    label = int(labels[0])

    amplitude_rows = []
    good_timestamps = []
    error_counts = {}

    for row_number, (timestamp, hex_string) in enumerate(
        zip(timestamps, df["csi_hex"])
    ):
        try:
            val = str(hex_string).strip()

            # Handle pandas reading binary string literals (e.g. "b'\x4f\xf1...'")
            if val.startswith("b'") or val.startswith('b"'):
                raw = ast.literal_eval(val)
            else:
                raw = bytes.fromhex(val)

            amplitudes = extract_selected_amplitudes(raw)

            if np.all(np.isfinite(amplitudes)):
                amplitude_rows.append(amplitudes)
                good_timestamps.append(timestamp)
            else:
                error_counts["nonfinite"] = error_counts.get("nonfinite", 0) + 1

        except Exception as exc:
            error_type = type(exc).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

            if sum(error_counts.values()) <= 5:
                print(
                    f"  Decode error at CSV row {row_number + 2}: "
                    f"{error_type}: {exc}"
                )

    print(f"  Raw CSV rows       : {len(df)}")
    print(f"  Valid CSI samples  : {len(amplitude_rows)}")
    print(f"  Rejected samples   : {len(df) - len(amplitude_rows)}")

    if error_counts:
        print("  Rejection reasons:")
        for reason, count in error_counts.items():
            print(f"    {reason}: {count}")

    if len(amplitude_rows) < WINDOW_SAMPLES:
        raise ValueError(
            f"{filename} has only {len(amplitude_rows)} valid CSI samples "
            f"after decoding; {WINDOW_SAMPLES} required."
        )

    return (
        np.asarray(good_timestamps, dtype=np.float64),
        np.asarray(amplitude_rows, dtype=np.float64),
        label,
    )


# ============================================================
# PROCESS SESSION
# ============================================================

def process_session(filename):
    print(f"\nProcessing: {os.path.basename(filename)}")

    timestamps, amplitudes, label = load_session(filename)

    X, y, times = build_feature_windows(
        timestamps,
        amplitudes,
        label=label,
    )

    print(f"  Samples : {len(timestamps)}")
    print(f"  Windows : {len(X)}")

    return X, np.asarray(y), label


# ============================================================
# FIND DATA
# ============================================================

idle_files = sorted(glob.glob(os.path.join(DATA_DIR, "idle_*.csv")))
motion_files = sorted(glob.glob(os.path.join(DATA_DIR, "motion_*.csv")))
all_files = idle_files + motion_files

if not all_files:
    raise SystemExit("[ERROR] No recordings found in data/")

print("=" * 70)
print("RF VISION — RANDOM FOREST TRAINING")
print("=" * 70)

print(f"Sample rate       : {SAMPLE_RATE_HZ} Hz")
print(f"Window             : {WINDOW_SECONDS:.1f} s ({WINDOW_SAMPLES} samples)")
print(f"CSI pair indices  : {CSI_PAIR_INDICES}")
print(f"Feature count     : {len(FEATURE_NAMES)}")
print(f"\nIdle sessions     : {len(idle_files)}")
print(f"Motion sessions   : {len(motion_files)}")


# ============================================================
# SESSION SPLIT
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)
shuffled = list(all_files)
rng.shuffle(shuffled)

test_count = max(1, int(round(len(shuffled) * TEST_SESSION_FRACTION)))
test_files = shuffled[:test_count]
train_files = shuffled[test_count:]

print("\nTRAINING SESSIONS:")
for f in train_files:
    print("  ", os.path.basename(f))

print("\nTEST SESSIONS:")
for f in test_files:
    print("  ", os.path.basename(f))


# ============================================================
# BUILD COMBINED DATASET
# ============================================================

X_list = []
y_list = []

for filename in all_files:
    X_session, y_session, _ = process_session(filename)
    if len(X_session) > 0:
        X_list.append(X_session)
        y_list.append(y_session)

if not X_list:
    raise SystemExit("[ERROR] No training windows generated from any sessions.")

X_all = np.vstack(X_list)
y_all = np.concatenate(y_list)

# ============================================================
# STRATIFIED TRAIN/TEST SPLIT
# ============================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_all, 
    y_all, 
    test_size=0.20, 
    random_state=RANDOM_STATE, 
    stratify=y_all
)

print("\n" + "=" * 70)
print("DATASET SUMMARY (STRATIFIED SPLIT)")
print("=" * 70)
print(f"Total feature windows : {len(X_all)}")
print(f"Training windows      : {len(X_train)} (IDLE: {np.sum(y_train == 0)}, MOTION: {np.sum(y_train == 1)})")
print(f"Test windows          : {len(X_test)} (IDLE: {np.sum(y_test == 0)}, MOTION: {np.sum(y_test == 1)})")


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=[0, 1])

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)
print(f"Accuracy : {accuracy * 100:.2f}%\n")
print("Confusion matrix:")
print("             Pred IDLE   Pred MOTION")
print(f"Actual IDLE    {cm[0,0]:8d}      {cm[0,1]:8d}")
print(f"Actual MOTION  {cm[1,0]:8d}      {cm[1,1]:8d}\n")
print(classification_report(y_test, pred, target_names=["IDLE", "MOTION"], zero_division=0))


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_
ranking = sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True)

print("TOP FEATURES")
print("-" * 70)
for name, value in ranking[:15]:
    print(f"{name:35s} {value:.5f}")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_FILE)

config = {
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "window_seconds": WINDOW_SECONDS,
    "window_samples": WINDOW_SAMPLES,
    "csi_pair_indices": CSI_PAIR_INDICES,
    "feature_names": FEATURE_NAMES,
    "doppler_band_hz": [0.5, 3.0],
}

with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nModel saved: {MODEL_FILE}")
print(f"Config saved: {CONFIG_FILE}")
print("\nTRAINING COMPLETE")