"""
evaluate.py - Model evaluation script for UCI Obesity Classification
MLOps Lead: Rafiujjaman Ratul (103484584)

Reads:  artifacts/models/model.keras
        artifacts/data/X_test.npy, y_test.npy
        artifacts/preprocessing/encoder.pkl
Writes: artifacts/metrics/evaluation_metrics.json
        reports/performance_report.html
"""

import os
import json
import logging
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


def evaluate():
    log.info("=" * 60)
    log.info("STARTING MODEL EVALUATION")
    log.info("=" * 60)

    model, X_test, y_test, encoder = load_artifacts()
    class_names = list(encoder.classes_)

    # ── Predictions ───────────────────────────────────────────────────────────
    y_proba = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    # ── Core metrics ──────────────────────────────────────────────────────────
    accuracy    = float(accuracy_score(y_test, y_pred))
    macro_f1    = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=range(len(class_names)))

    clf_report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
    )
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    log.info("Accuracy:    %.4f  (threshold: %.2f)", accuracy, MIN_ACCURACY)
    log.info("Macro F1:    %.4f  (threshold: %.2f)", macro_f1, MIN_F1)
    log.info("Weighted F1: %.4f", weighted_f1)

    for i, cls in enumerate(class_names):
        log.info("  F1 [%s]: %.4f", cls, per_class_f1[i])

    # ── Gate check ────────────────────────────────────────────────────────────
    passed = accuracy >= MIN_ACCURACY and all(
        f >= MIN_F1 for f in per_class_f1
    )
    status = "PASSED" if passed else "FAILED"
    log.info("Evaluation gate: %s", status)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics = {
        "timestamp":          datetime.now().isoformat(),
        "accuracy":           accuracy,
        "macro_f1":           macro_f1,
        "weighted_f1":        weighted_f1,
        "per_class_f1":       {cls: float(per_class_f1[i])
                               for i, cls in enumerate(class_names)},
        "classification_report": clf_report,
        "confusion_matrix":   conf_matrix,
        "gate_passed":        passed,
        "thresholds": {
            "min_accuracy": MIN_ACCURACY,
            "min_f1":       MIN_F1,
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved -> %s", METRICS_PATH)

    # ── HTML performance report ───────────────────────────────────────────────
    _write_html_report(metrics, class_names, conf_matrix)

    log.info("=" * 60)
    log.info("EVALUATION COMPLETE  |  Gate: %s", status)
    log.info("=" * 60)

    return metrics


def _write_html_report(metrics: dict, class_names: list, conf_matrix: list):
    """Write a simple HTML performance report."""
    gate_color = "#28a745" if metrics["gate_passed"] else "#dc3545"
    gate_label = "PASSED ✔" if metrics["gate_passed"] else "FAILED ✘"

    rows = ""
    for cls, f1 in metrics["per_class_f1"].items():
        color = "#28a745" if f1 >= MIN_F1 else "#dc3545"
        rows += f"<tr><td>{cls}</td><td style='color:{color}'>{f1:.4f}</td></tr>\n"

    cm_rows = ""
    for i, row in enumerate(conf_matrix):
        cells = "".join(f"<td>{v}</td>" for v in row)
        cm_rows += f"<tr><td><b>{class_names[i]}</b></td>{cells}</tr>\n"

    cm_headers = "".join(f"<th>{c}</th>" for c in class_names)

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
  .badge {{ display: inline-block; padding: 6px 16px; border-radius: 4px;
            color: white; font-weight: bold; background: {gate_color}; }}
</style>
</head><body>
<h1>Model Evaluation Report</h1>
<p><b>Generated:</b> {metrics['timestamp']}</p>
<p><b>Gate:</b> <span class='badge'>{gate_label}</span></p>

<h2>Core Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th><th>Threshold</th></tr>
  <tr><td>Accuracy</td><td>{metrics['accuracy']:.4f}</td><td>≥ {MIN_ACCURACY}</td></tr>
  <tr><td>Macro F1</td><td>{metrics['macro_f1']:.4f}</td><td>≥ {MIN_F1}</td></tr>
  <tr><td>Weighted F1</td><td>{metrics['weighted_f1']:.4f}</td><td>—</td></tr>
</table>

<h2>Per-Class F1 Scores</h2>
<table>
  <tr><th>Class</th><th>F1 Score</th></tr>
  {rows}
</table>

<h2>Confusion Matrix</h2>
<table>
  <tr><th>Actual \\ Predicted</th>{cm_headers}</tr>
  {cm_rows}
</table>
</body></html>"""

    report_path = os.path.join(REPORT_DIR, "performance_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("HTML report saved -> %s", report_path)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    evaluate()
