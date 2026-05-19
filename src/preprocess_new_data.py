"""
preprocess_new_data.py - Preprocessing pipeline for UCI Obesity dataset
MLOps Lead: Rafiujjaman Ratul (103484584)

Reads:  train/train.csv, test/test.csv, data/new_data.csv (optional)
Writes: artifacts/data/X_train.npy, y_train.npy, X_test.npy, y_test.npy
        artifacts/preprocessing/scaler.pkl, encoder.pkl, feature_columns.json
        artifacts/metadata/data_version.txt
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_CSV   = os.path.join("train", "train.csv")
TEST_CSV    = os.path.join("test", "test.csv")
NEW_DATA    = os.path.join("data", "new_data.csv")

ARTIFACTS   = "artifacts"
DATA_DIR    = os.path.join(ARTIFACTS, "data")
PREP_DIR    = os.path.join(ARTIFACTS, "preprocessing")
META_DIR    = os.path.join(ARTIFACTS, "metadata")

for d in [DATA_DIR, PREP_DIR, META_DIR, "logs"]:
    os.makedirs(d, exist_ok=True)

# ── Target column ─────────────────────────────────────────────────────────────
TARGET = "NObeyesdad"

# ── Categorical columns to one-hot encode ────────────────────────────────────
CATEGORICAL_COLS = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]

# ── Numerical columns to scale ────────────────────────────────────────────────
NUMERICAL_COLS = [
    "Age", "Height", "Weight", "FCVC", "NCP",
    "CH2O", "FAF", "TUE",
]


def load_csv(path: str) -> pd.DataFrame:
    log.info("Loading: %s", path)
    df = pd.read_csv(path)
    log.info("  Shape: %s", df.shape)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BMI and activity score features."""
    df = df.copy()
    # BMI
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    # Activity score: physical activity frequency minus screen time
    df["activity_score"] = df["FAF"] - df["TUE"]
    log.info("Engineered features: BMI, activity_score")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    log.info("After one-hot encoding: %d columns", df.shape[1])
    return df


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """Full preprocessing pipeline. Returns X_train, y_train, X_test, y_test."""

    log.info("Starting preprocessing pipeline")

    # ── Feature engineering ───────────────────────────────────────────────────
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    # ── Separate features and target ──────────────────────────────────────────
    y_train_raw = train_df.pop(TARGET)
    y_test_raw  = test_df.pop(TARGET)

    # ── One-hot encode ────────────────────────────────────────────────────────
    train_df = encode_categoricals(train_df)
    test_df  = encode_categoricals(test_df)

    # ── Align columns (test may have fewer after encoding) ────────────────────
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    feature_cols = train_df.columns.tolist()
    log.info("Total features after encoding: %d", len(feature_cols))

    # ── Scale numerical features ──────────────────────────────────────────────
    scaler = StandardScaler()
    num_cols_present = [c for c in NUMERICAL_COLS + ["BMI", "activity_score"]
                        if c in train_df.columns]

    train_df[num_cols_present] = scaler.fit_transform(
        train_df[num_cols_present]
    )
    test_df[num_cols_present] = scaler.transform(
        test_df[num_cols_present]
    )
    log.info("Scaled %d numerical columns", len(num_cols_present))

    # ── Encode labels ─────────────────────────────────────────────────────────
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test  = encoder.transform(y_test_raw)
    log.info("Classes: %s", list(encoder.classes_))

    X_train = train_df.values.astype(np.float32)
    X_test  = test_df.values.astype(np.float32)

    return X_train, y_train, X_test, y_test, scaler, encoder, feature_cols


def merge_new_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Append new_data.csv to training set if it exists."""
    if not os.path.exists(NEW_DATA):
        log.info("No new_data.csv found — skipping merge")
        return train_df

    log.info("Found new_data.csv — merging into training set")
    new_df = load_csv(NEW_DATA)

    # Basic schema check
    missing = set(train_df.columns) - set(new_df.columns)
    if missing:
        log.warning("New data missing columns: %s — filling with 0", missing)
        for col in missing:
            new_df[col] = 0

    combined = pd.concat([train_df, new_df], ignore_index=True)
    log.info("Training set after merge: %s", combined.shape)
    return combined


def run():
    log.info("=" * 60)
    log.info("STARTING PREPROCESSING")
    log.info("=" * 60)

    train_df = load_csv(TRAIN_CSV)
    test_df  = load_csv(TEST_CSV)

    # Merge new data if available
    train_df = merge_new_data(train_df)

    X_train, y_train, X_test, y_test, scaler, encoder, feature_cols = \
        preprocess(train_df, test_df)

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)
    log.info("Saved numpy arrays -> %s", DATA_DIR)

    # ── Save scaler & encoder ─────────────────────────────────────────────────
    joblib.dump(scaler,  os.path.join(PREP_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(PREP_DIR, "encoder.pkl"))
    log.info("Saved scaler and encoder -> %s", PREP_DIR)

    # ── Save feature columns ──────────────────────────────────────────────────
    with open(os.path.join(PREP_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info("Saved feature_columns.json")

    # ── Save data version ─────────────────────────────────────────────────────
    version = datetime.now().strftime("data_%Y%m%d_%H%M%S")
    with open(os.path.join(META_DIR, "data_version.txt"), "w") as f:
        f.write(version)
    log.info("Data version: %s", version)

    log.info("=" * 60)
    log.info("PREPROCESSING COMPLETE")
    log.info("  X_train: %s  y_train: %s", X_train.shape, y_train.shape)
    log.info("  X_test:  %s  y_test:  %s", X_test.shape,  y_test.shape)
    log.info("=" * 60)


if __name__ == "__main__":
    run()
