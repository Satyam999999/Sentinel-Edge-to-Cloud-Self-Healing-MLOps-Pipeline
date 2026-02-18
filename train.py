import mlflow
import mlflow.pyfunc
from logger import get_logger
from datetime import datetime
import os

logger = get_logger("Training")

MODEL_NAME = "Sentinel-Anomaly-Model"


def train():
    mlflow.set_experiment("Sentinel-Retraining")

    with mlflow.start_run() as run:

        # ------------------------
        # Mock Training Metrics
        # ------------------------
        mock_map = 0.82
        mock_precision = 0.79

        mlflow.log_param("epochs", 5)
        mlflow.log_metric("mAP", mock_map)
        mlflow.log_metric("precision", mock_precision)

        # ------------------------
        # Save Dummy Model Artifact
        # ------------------------
        model_path = "model.txt"

        with open(model_path, "w") as f:
            f.write(f"Mock Sentinel model - {datetime.utcnow()}")

        mlflow.log_artifact(model_path)

        # ------------------------
        # Register Model
        # ------------------------
        model_uri = f"runs:/{run.info.run_id}/{model_path}"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        logger.info(f"Model registered as version {result.version}")

        # ------------------------
        # Promote To Production
        # ------------------------
        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )

        logger.info("Model promoted to Production stage.")


if __name__ == "__main__":
    train()
