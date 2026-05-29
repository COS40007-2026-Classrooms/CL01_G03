"""
evaluate.py - Model evaluation script for UCI Obesity Classification
MLOps Lead: Rafiujjaman Ratul (103484584) (Neural Network)
Ishank Malhotra (104210599) (Random Forest)

Reads:  artifacts/models/model.keras
        artifacts/models/rf_model.pkl
        artifacts/data/X_test.npy, y_test.npy
        artifacts/preprocessing/encoder.pkl
Writes: artifacts/metrics/evaluation_metrics.json
        reports/performance_report.html
"""

import os
import json
import logging
import pickle
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS    = "artifacts"
MODEL_PATH   = os.path.join(ARTIFACTS, "models", "model.keras")
RF_PATH      = os.path.join(ARTIFACTS, "models", "rf_model.pkl")
DATA_DIR     = os.path.join(ARTIFACTS, "data")
PREP_DIR     = os.path.join(ARTIFACTS, "preprocessing")
METRICS_PATH = os.path.join(ARTIFACTS, "metrics", "evaluation_metrics.json")
REPORT_DIR   = "reports"

for d in [os.path.join(ARTIFACTS, "metrics"), REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Thresholds (from Group Task 1 MLOps design) ───────────────────────────────
MIN_ACCURACY = 0.80
MIN_F1       = 0.75


def load_artifacts():
    log.info("Loading model from %s", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    encoder = joblib.load(os.path.join(PREP_DIR, "encoder.pkl"))

    log.info("Test set: %s", X_test.shape)
    return model, X_test, y_test, encoder


def evaluate_model(y_test, y_pred, y_proba, class_names, model_name):
    """Compute and log metrics for a given model."""
    accuracy     = float(accuracy_score(y_test, y_pred))
    macro_f1     = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1  = float(f1_score(y_test, y_pred, average="weighted"))
    per_class_f1 = f1_score(y_test, y_pred, average=None,
                             labels=range(len(class_names)))
    clf_report   = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    conf_matrix  = confusion_matrix(y_test, y_pred).tolist()

    log.info("=" * 60)
    log.info("%s EVALUATION", model_name.upper())
    log.info("=" * 60)
    log.info("Accuracy:    %.4f  (threshold: %.2f)", accuracy, MIN_ACCURACY)
    log.info("Macro F1:    %.4f  (threshold: %.2f)", macro_f1, MIN_F1)
    log.info("Weighted F1: %.4f", weighted_f1)
    for i, cls in enumerate(class_names):
        log.info("  F1 [%s]: %.4f", cls, per_class_f1[i])

    passed = accuracy >= MIN_ACCURACY and all(f >= MIN_F1 for f in per_class_f1)
    log.info("Evaluation gate: %s", "PASSED" if passed else "FAILED")

    return {
        "model_name":            model_name,
        "timestamp":             datetime.now().isoformat(),
        "accuracy":              accuracy,
        "macro_f1":              macro_f1,
        "weighted_f1":           weighted_f1,
        "per_class_f1":          {cls: float(per_class_f1[i])
                                  for i, cls in enumerate(class_names)},
        "classification_report": clf_report,
        "confusion_matrix":      conf_matrix,
        "gate_passed":           passed,
        "thresholds": {
            "min_accuracy": MIN_ACCURACY,
            "min_f1":       MIN_F1,
        },
    }


def evaluate():
    log.info("=" * 60)
    log.info("STARTING MODEL EVALUATION")
    log.info("=" * 60)

    model, X_test, y_test, encoder = load_artifacts()
    class_names = list(encoder.classes_)

    # ── Neural Network ────────────────────────────────────────────────────────
    y_proba_nn = model.predict(X_test, verbose=0)
    y_pred_nn  = np.argmax(y_proba_nn, axis=1)
    nn_metrics = evaluate_model(y_test, y_pred_nn, y_proba_nn,
                                class_names, "Neural Network")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_metrics = None
    if os.path.exists(RF_PATH):
        with open(RF_PATH, "rb") as f:
            rf = pickle.load(f)
        y_pred_rf  = rf.predict(X_test)
        y_proba_rf = rf.predict_proba(X_test)
        rf_metrics = evaluate_model(y_test, y_pred_rf, y_proba_rf,
                                    class_names, "Random Forest")
    else:
        log.warning("rf_model.pkl not found, skipping RF evaluation")

    # ── Save combined metrics JSON ────────────────────────────────────────────
    combined = {
        "neural_network": nn_metrics,
        "random_forest":  rf_metrics,
        "timestamp":   nn_metrics["timestamp"],
        "accuracy":    nn_metrics["accuracy"],
        "macro_f1":    nn_metrics["macro_f1"],
        "weighted_f1": nn_metrics["weighted_f1"],
        "gate_passed": nn_metrics["gate_passed"],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    log.info("Metrics saved -> %s", METRICS_PATH)

    # ── HTML report ───────────────────────────────────────────────────────────
    _write_html_report(nn_metrics, rf_metrics, class_names)

    log.info("=" * 60)
    log.info("EVALUATION COMPLETE  |  NN Gate: %s  |  RF Gate: %s",
             "PASSED" if nn_metrics["gate_passed"] else "FAILED",
             "PASSED" if rf_metrics and rf_metrics["gate_passed"] else "N/A")
    log.info("=" * 60)

    return combined


def _write_html_report(nn_metrics: dict, rf_metrics: dict, class_names: list):
    """Write HTML report with both Neural Network and Random Forest results."""

    def gate_badge(passed):
        color = "#28a745" if passed else "#dc3545"
        label = "PASSED ✔" if passed else "FAILED ✘"
        return f"<span style='display:inline-block;padding:6px 16px;border-radius:4px;color:white;font-weight:bold;background:{color}'>{label}</span>"

    def f1_rows(metrics):
        rows = ""
        for cls, f1 in metrics["per_class_f1"].items():
            color = "#28a745" if f1 >= MIN_F1 else "#dc3545"
            rows += f"<tr><td>{cls}</td><td style='color:{color}'>{f1:.4f}</td></tr>\n"
        return rows

    def cm_rows(metrics):
        rows = ""
        conf = metrics["confusion_matrix"]
        for i, row in enumerate(conf):
            cells = "".join(f"<td>{v}</td>" for v in row)
            rows += f"<tr><td><b>{class_names[i]}</b></td>{cells}</tr>\n"
        return rows

    cm_headers = "".join(f"<th>{c}</th>" for c in class_names)

    comparison_section = ""
    if rf_metrics:
        best = "Random Forest" if rf_metrics["accuracy"] >= nn_metrics["accuracy"] else "Neural Network"
        comparison_section = f"""
<h2>Model Comparison</h2>
<table>
  <tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Weighted F1</th><th>Gate</th></tr>
  <tr><td>Neural Network</td><td>{nn_metrics['accuracy']:.4f}</td><td>{nn_metrics['macro_f1']:.4f}</td><td>{nn_metrics['weighted_f1']:.4f}</td><td>{gate_badge(nn_metrics['gate_passed'])}</td></tr>
  <tr><td><b>Random Forest</b></td><td><b>{rf_metrics['accuracy']:.4f}</b></td><td><b>{rf_metrics['macro_f1']:.4f}</b></td><td><b>{rf_metrics['weighted_f1']:.4f}</b></td><td>{gate_badge(rf_metrics['gate_passed'])}</td></tr>
</table>
<p><b>Best model: {best}</b></p>
"""

    rf_section = ""
    if rf_metrics:
        rf_section = f"""
<h2>Random Forest — Core Metrics</h2>
<p><b>Gate:</b> {gate_badge(rf_metrics['gate_passed'])}</p>
<table>
  <tr><th>Metric</th><th>Value</th><th>Threshold</th></tr>
  <tr><td>Accuracy</td><td>{rf_metrics['accuracy']:.4f}</td><td>≥ {MIN_ACCURACY}</td></tr>
  <tr><td>Macro F1</td><td>{rf_metrics['macro_f1']:.4f}</td><td>≥ {MIN_F1}</td></tr>
  <tr><td>Weighted F1</td><td>{rf_metrics['weighted_f1']:.4f}</td><td>—</td></tr>
</table>

<h2>Random Forest — Per-Class F1 Scores</h2>
<table>
  <tr><th>Class</th><th>F1 Score</th></tr>
  {f1_rows(rf_metrics)}
</table>

<h2>Random Forest — Confusion Matrix</h2>
<table>
  <tr><th>Actual \\ Predicted</th>{cm_headers}</tr>
  {cm_rows(rf_metrics)}
</table>
"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>Model Evaluation Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
  h1 {{ color: #2E75B6; }} h2 {{ color: #444; border-bottom: 1px solid #ddd; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #2E75B6; color: white; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
</style>
</head><body>
<h1>Model Evaluation Report</h1>
<p><b>Generated:</b> {nn_metrics['timestamp']}</p>

{comparison_section}

<h2>Neural Network — Core Metrics</h2>
<p><b>Gate:</b> {gate_badge(nn_metrics['gate_passed'])}</p>
<table>
  <tr><th>Metric</th><th>Value</th><th>Threshold</th></tr>
  <tr><td>Accuracy</td><td>{nn_metrics['accuracy']:.4f}</td><td>≥ {MIN_ACCURACY}</td></tr>
  <tr><td>Macro F1</td><td>{nn_metrics['macro_f1']:.4f}</td><td>≥ {MIN_F1}</td></tr>
  <tr><td>Weighted F1</td><td>{nn_metrics['weighted_f1']:.4f}</td><td>—</td></tr>
</table>

<h2>Neural Network — Per-Class F1 Scores</h2>
<table>
  <tr><th>Class</th><th>F1 Score</th></tr>
  {f1_rows(nn_metrics)}
</table>

<h2>Neural Network — Confusion Matrix</h2>
<table>
  <tr><th>Actual \\ Predicted</th>{cm_headers}</tr>
  {cm_rows(nn_metrics)}
</table>

{rf_section}
</body></html>"""

    report_path = os.path.join(REPORT_DIR, "performance_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("HTML report saved -> %s", report_path)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    evaluate()