"""
monitor.py - Drift detection & performance monitoring for UCI Obesity Classification
MLOps Lead: Rafiujjaman Ratul (103484584)

Detects:  Data drift (KS test for numerical, chi-squared for categorical)
          Concept drift (sliding window F1 vs baseline)
          Performance degradation vs thresholds
Writes:   artifacts/metrics/monitoring_metrics.json
          reports/drift_report.json
          reports/monitoring_dashboard.html
          monitoring/logs/
          monitoring/alerts/
"""

import os
import json
import logging
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

# ── Logging ───────────────────────────────────────────────────────────────────
for d in ["logs", "monitoring/logs", "monitoring/alerts", "monitoring/reports"]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/monitoring.log"),
        logging.FileHandler("monitoring/logs/monitor.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

drift_log = logging.getLogger("drift")
drift_handler = logging.FileHandler("logs/drift_detection.log")
drift_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
drift_log.addHandler(drift_handler)
drift_log.setLevel(logging.INFO)

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS    = "artifacts"
MODEL_PATH   = os.path.join(ARTIFACTS, "models", "model.keras")
DATA_DIR     = os.path.join(ARTIFACTS, "data")
PREP_DIR     = os.path.join(ARTIFACTS, "preprocessing")
METRICS_DIR  = os.path.join(ARTIFACTS, "metrics")
REPORT_DIR   = "reports"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_ACCURACY   = 0.80
MIN_F1         = 0.75
KS_ALPHA       = 0.05
CHI2_ALPHA     = 0.05
WINDOW_SIZE    = 50


# ── Numpy JSON Encoder ────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_artifacts():
    model   = tf.keras.models.load_model(MODEL_PATH)
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    encoder = joblib.load(os.path.join(PREP_DIR, "encoder.pkl"))

    with open(os.path.join(PREP_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    return model, X_train, X_test, y_test, encoder, feature_cols


# ── Data Drift Detection ──────────────────────────────────────────────────────

def detect_data_drift(X_train, X_test, feature_cols):
    drift_log.info("Running data drift detection (KS test per feature)")

    results = {}
    drifted = []

    for i, col in enumerate(feature_cols):
        train_vals = X_train[:, i]
        test_vals  = X_test[:, i]

        ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)
        drifted_flag = bool(p_value < KS_ALPHA)

        results[col] = {
            "ks_statistic": float(ks_stat),
            "p_value":       float(p_value),
            "drifted":       drifted_flag,
        }

        if drifted_flag:
            drifted.append(col)
            drift_log.warning(
                "DRIFT detected -- %s  KS=%.4f  p=%.4f", col, ks_stat, p_value
            )

    drift_log.info(
        "Data drift summary: %d/%d features drifted", len(drifted), len(feature_cols)
    )

    return {
        "feature_results": results,
        "drifted_features": drifted,
        "drift_detected":   bool(len(drifted) > 0),
        "drift_rate":       float(len(drifted) / max(len(feature_cols), 1)),
    }


# ── Concept Drift Detection ───────────────────────────────────────────────────

def detect_concept_drift(model, X_test, y_test, baseline_f1):
    drift_log.info("Running concept drift detection (sliding window F1)")

    n = len(X_test)
    window = min(WINDOW_SIZE, n)

    X_recent = X_test[-window:]
    y_recent = y_test[-window:]

    y_proba   = model.predict(X_recent, verbose=0)
    y_pred    = np.argmax(y_proba, axis=1)
    window_f1 = float(f1_score(y_recent, y_pred, average="macro", zero_division=0))

    drop          = baseline_f1 - window_f1
    concept_drift = bool(drop > 0.05)

    drift_log.info(
        "Concept drift -- baseline F1: %.4f  window F1: %.4f  drop: %.4f  detected: %s",
        baseline_f1, window_f1, drop, concept_drift,
    )

    return {
        "baseline_f1":   baseline_f1,
        "window_f1":     window_f1,
        "window_size":   window,
        "f1_drop":       float(drop),
        "concept_drift": concept_drift,
    }


# ── Performance Monitoring ────────────────────────────────────────────────────

def monitor_performance(model, X_test, y_test, encoder):
    log.info("Monitoring model performance")

    y_proba    = model.predict(X_test, verbose=0)
    y_pred     = np.argmax(y_proba, axis=1)
    confidence = np.max(y_proba, axis=1)

    accuracy     = float(accuracy_score(y_test, y_pred))
    macro_f1     = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    per_class_f1 = f1_score(y_test, y_pred, average=None,
                             labels=range(len(encoder.classes_)),
                             zero_division=0)

    unique, counts = np.unique(y_pred, return_counts=True)
    label_dist = {
        encoder.classes_[int(u)]: int(c)
        for u, c in zip(unique, counts)
    }

    perf = {
        "accuracy":           accuracy,
        "macro_f1":           macro_f1,
        "per_class_f1":       {cls: float(per_class_f1[i])
                               for i, cls in enumerate(encoder.classes_)},
        "confidence_mean":    float(np.mean(confidence)),
        "confidence_std":     float(np.std(confidence)),
        "low_confidence_pct": float(np.mean(confidence < 0.6) * 100),
        "label_distribution": label_dist,
        "accuracy_ok":        bool(accuracy >= MIN_ACCURACY),
        "f1_ok":              bool(all(f >= MIN_F1 for f in per_class_f1)),
    }

    log.info("Accuracy: %.4f  Macro F1: %.4f  Low-conf%%: %.1f%%",
             accuracy, macro_f1, perf["low_confidence_pct"])

    return perf


# ── Alert Writer ──────────────────────────────────────────────────────────────

def write_alert(alert_type, message):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("monitoring", "alerts", f"{alert_type}_{timestamp}.json")
    alert = {
        "type":      alert_type,
        "message":   message,
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(alert, f, indent=2)
    log.warning("ALERT [%s]: %s", alert_type, message)


# ── Dashboard HTML ────────────────────────────────────────────────────────────

def write_dashboard(perf, data_drift, concept_drift, rf_metrics=None):
    def badge(ok, yes="OK", no="WARNING"):
        color = "#28a745" if ok else "#ffc107"
        label = yes if ok else no
        return f"<span style='background:{color};color:white;padding:3px 10px;border-radius:3px'>{label}</span>"

    f1_rows = "".join(
        f"<tr><td>{cls}</td><td>{f1:.4f}</td>"
        f"<td>{badge(f1 >= MIN_F1)}</td></tr>"
        for cls, f1 in perf["per_class_f1"].items()
    )

    dist_rows = "".join(
        f"<tr><td>{cls}</td><td>{cnt}</td></tr>"
        for cls, cnt in perf["label_distribution"].items()
    )

    drifted_list = ", ".join(data_drift["drifted_features"]) or "None"

    if rf_metrics:
        rf_acc = rf_metrics.get("accuracy", 0)
        rf_f1  = rf_metrics.get("macro_f1", 0)
        comparison_table = f"""
<h2>Model Comparison</h2>
<table>
  <tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Role</th></tr>
  <tr><td>Neural Network (monitored)</td><td>{perf['accuracy']:.4f}</td><td>{perf['macro_f1']:.4f}</td><td>Production</td></tr>
  <tr><td><b>Random Forest (baseline)</b></td><td><b>{rf_acc:.4f}</b></td><td><b>{rf_f1:.4f}</b></td><td>Best model</td></tr>
</table>
<p><i>Monitoring tracks the Neural Network in production. Random Forest scores from training evaluation.</i></p>
"""
    else:
        comparison_table = ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>Monitoring Dashboard -- CL01_G03</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 20px; }}
  h1 {{ color: #2E75B6; }} h2 {{ color: #444; border-bottom: 1px solid #ddd; padding-bottom:4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #2E75B6; color: white; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .metric {{ display:inline-block; margin:8px; padding:12px 20px;
             border-radius:6px; background:#f0f4ff; text-align:center; }}
  .metric b {{ display:block; font-size:1.5em; color:#2E75B6; }}
</style>
</head><body>
<h1>Monitoring Dashboard</h1>
<p><b>Generated:</b> {datetime.now().isoformat()}</p>
<p><b>Group:</b> CL01_G03 | UCI Obesity Classification</p>

{comparison_table}

<h2>Performance Summary</h2>
<div>
  <div class='metric'><b>{perf['accuracy']:.4f}</b>Accuracy</div>
  <div class='metric'><b>{perf['macro_f1']:.4f}</b>Macro F1</div>
  <div class='metric'><b>{perf['confidence_mean']:.4f}</b>Avg Confidence</div>
  <div class='metric'><b>{perf['low_confidence_pct']:.1f}%</b>Low Confidence</div>
</div>

<h2>Per-Class F1 Scores</h2>
<table>
  <tr><th>Class</th><th>F1 Score</th><th>Status</th></tr>
  {f1_rows}
</table>

<h2>Data Drift (KS Test)</h2>
<p>Drift detected: {badge(not data_drift['drift_detected'], 'No Drift', 'Drift Detected')}</p>
<p>Drifted features ({len(data_drift['drifted_features'])}): {drifted_list}</p>
<p>Drift rate: {data_drift['drift_rate']:.1%}</p>

<h2>Concept Drift (Sliding Window F1)</h2>
<p>Status: {badge(not concept_drift['concept_drift'], 'Stable', 'Concept Drift')}</p>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Baseline F1</td><td>{concept_drift['baseline_f1']:.4f}</td></tr>
  <tr><td>Window F1 (last {concept_drift['window_size']} samples)</td>
      <td>{concept_drift['window_f1']:.4f}</td></tr>
  <tr><td>F1 Drop</td><td>{concept_drift['f1_drop']:.4f}</td></tr>
</table>

<h2>Prediction Label Distribution</h2>
<table>
  <tr><th>Class</th><th>Count</th></tr>
  {dist_rows}
</table>
</body></html>"""

    path = os.path.join(REPORT_DIR, "monitoring_dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Dashboard saved -> %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("STARTING MONITORING")
    log.info("=" * 60)

    model, X_train, X_test, y_test, encoder, feature_cols = load_artifacts()

    eval_path = os.path.join(METRICS_DIR, "evaluation_metrics.json")
    baseline_f1 = 0.80
    rf_metrics  = None
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_metrics = json.load(f)
        baseline_f1 = eval_metrics.get("macro_f1", 0.80)
        rf_metrics  = eval_metrics.get("random_forest")
        log.info("Loaded baseline F1 from evaluation: %.4f", baseline_f1)

    perf          = monitor_performance(model, X_test, y_test, encoder)
    data_drift    = detect_data_drift(X_train, X_test, feature_cols)
    concept_drift = detect_concept_drift(model, X_test, y_test, baseline_f1)

    if not perf["accuracy_ok"]:
        write_alert("performance",
                    f"Accuracy {perf['accuracy']:.4f} below threshold {MIN_ACCURACY}")
    if not perf["f1_ok"]:
        write_alert("performance",
                    f"F1 score below threshold {MIN_F1} for one or more classes")
    if data_drift["drift_detected"]:
        write_alert("data_drift",
                    f"Data drift detected in: {data_drift['drifted_features']}")
    if concept_drift["concept_drift"]:
        write_alert("concept_drift",
                    f"Concept drift: F1 dropped {concept_drift['f1_drop']:.4f}")

    monitoring_metrics = {
        "timestamp":    datetime.now().isoformat(),
        "performance":  perf,
        "data_drift":   data_drift,
        "concept_drift": concept_drift,
        "retraining_recommended": bool(
            not perf["accuracy_ok"]
            or not perf["f1_ok"]
            or data_drift["drift_detected"]
            or concept_drift["concept_drift"]
        ),
    }

    with open(os.path.join(METRICS_DIR, "monitoring_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(monitoring_metrics, f, indent=2, cls=NumpyEncoder)

    with open(os.path.join(REPORT_DIR, "drift_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "data_drift":    data_drift,
            "concept_drift": concept_drift,
            "timestamp":     datetime.now().isoformat(),
        }, f, indent=2, cls=NumpyEncoder)

    write_dashboard(perf, data_drift, concept_drift, rf_metrics)

    log.info("=" * 60)
    log.info("MONITORING COMPLETE")
    log.info("  Retrain recommended: %s",
             monitoring_metrics["retraining_recommended"])
    log.info("=" * 60)

    return monitoring_metrics


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    run()