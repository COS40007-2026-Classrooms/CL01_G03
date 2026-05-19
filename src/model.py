"""
model.py - Training script for UCI Obesity Classification
MLOps Lead: Rafiujjaman Ratul (103484584)
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS   = "artifacts"
MODEL_PATH  = os.path.join(ARTIFACTS, "models", "model.keras")
HISTORY_PATH = os.path.join(ARTIFACTS, "metrics", "training_history.json")
META_DIR    = os.path.join(ARTIFACTS, "metadata")

for d in [
    os.path.join(ARTIFACTS, "models"),
    os.path.join(ARTIFACTS, "metrics"),
    META_DIR,
    "logs",
]:
    os.makedirs(d, exist_ok=True)


def load_data():
    """Load preprocessed numpy arrays from artifacts/data/."""
    data_dir = os.path.join(ARTIFACTS, "data")
    log.info("Loading training data from %s", data_dir)

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    log.info("Train: %s  Test: %s", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    """Build a simple feed-forward classification model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="obesity_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    log.info("=" * 60)
    log.info("STARTING MODEL TRAINING")
    log.info("=" * 60)

    X_train, y_train, X_test, y_test = load_data()

    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    input_dim   = X_train.shape[1]
    log.info("Classes: %d  Features: %d", num_classes, input_dim)

    model = build_model(input_dim, num_classes)
    model.summary(print_fn=log.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    log.info("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    log.info("Model saved -> %s", MODEL_PATH)

    # ── Save training history ─────────────────────────────────────────────────
    history_data = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_data, f, indent=2)
    log.info("Training history saved -> %s", HISTORY_PATH)

    # ── Save metadata ─────────────────────────────────────────────────────────
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    with open(os.path.join(META_DIR, "model_version.txt"), "w") as f:
        f.write(version)
    with open(os.path.join(META_DIR, "last_retrain.txt"), "w") as f:
        f.write(datetime.now().isoformat())

    log.info("Model version: %s", version)

    # ── Quick eval on test set ────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    log.info("=" * 60)
    log.info("TEST  ->  Loss: %.4f  |  Accuracy: %.4f", loss, acc)
    log.info("=" * 60)
    log.info("TRAINING COMPLETE")

    return model, history


if __name__ == "__main__":
    # Suppress TF info messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    train()
