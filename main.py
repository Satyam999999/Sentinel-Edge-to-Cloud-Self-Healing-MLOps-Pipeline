import os
import argparse
import yaml

from detector import AnomalyDetector
from metrics import ModelMetrics
from logger import get_logger
from mlflow_tracking import init_mlflow, log_run
from s3_uploader import S3Uploader

logger = get_logger("Main")


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # ---------------------------
    # CLI Argument Parsing
    # ---------------------------
    parser = argparse.ArgumentParser(description="Sentinel Inference Service")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    args = parser.parse_args()

    image_path = args.image

    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return

    # ---------------------------
    # Load Configuration
    # ---------------------------
    config = load_config("config.yaml")

    # ---------------------------
    # Initialize MLflow
    # ---------------------------
    init_mlflow("Sentinel-Anomaly-Detection")

    # ---------------------------
    # Load Model
    # ---------------------------
    detector = AnomalyDetector(
        weights_path=config["model"]["weights_path"],
        device=config["system"]["device"]
    )

    # ---------------------------
    # Run Inference
    # ---------------------------
    results, confidences, latency = detector.predict(
        image_path=image_path,
        conf_threshold=config["model"]["confidence_threshold"]
    )

    low_conf_threshold = config["active_learning"]["low_confidence_threshold"]

    # ---------------------------
    # Improved Confidence Logic
    # ---------------------------
    if len(confidences) == 0:
        logger.info("No detections found — storing for review.")
        trigger_upload = True
        avg_conf = 0.0
    else:
        avg_conf = sum(confidences) / len(confidences)
        trigger_upload = avg_conf < low_conf_threshold

    # ---------------------------
    # Active Learning Upload
    # ---------------------------
    if trigger_upload:
        logger.info("Active learning triggered — uploading to S3.")
        uploader = S3Uploader(config["active_learning"]["s3_bucket"])
        uploader.upload_file(image_path, avg_conf)
    else:
        logger.info("Confidence acceptable — no upload required.")

    # ---------------------------
    # Create Metrics Object
    # ---------------------------
    metrics_obj = ModelMetrics(
        mAP=0.78,  # Placeholder (real eval later)
        precision=0.81,
        recall=0.74,
        avg_confidence=avg_conf,
        latency_ms=latency
    )

    drift_detected = metrics_obj.is_drift_detected(
        config["monitoring"]["baseline_map"],
        config["monitoring"]["drift_threshold"]
    )

    latency_violation = metrics_obj.is_latency_violation(
        config["monitoring"]["latency_threshold_ms"]
    )

    # ---------------------------
    # Log to MLflow
    # ---------------------------
    params = {
        "confidence_threshold": config["model"]["confidence_threshold"],
        "device": config["system"]["device"]
    }

    metrics = {
        "mAP": metrics_obj.mAP,
        "precision": metrics_obj.precision,
        "recall": metrics_obj.recall,
        "avg_confidence": metrics_obj.avg_confidence,
        "latency_ms": metrics_obj.latency_ms,
        "drift_detected": int(drift_detected),
        "latency_violation": int(latency_violation)
    }

    log_run(params, metrics)

    # ---------------------------
    # Final Logs
    # ---------------------------
    logger.info("Inference completed successfully.")
    logger.info(f"Drift detected: {drift_detected}")
    logger.info(f"Latency violation: {latency_violation}")


if __name__ == "__main__":
    main()
