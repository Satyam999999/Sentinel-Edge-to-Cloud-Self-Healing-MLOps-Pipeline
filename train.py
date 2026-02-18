import mlflow
import mlflow.pyfunc
from logger import get_logger
from datetime import datetime

logger = get_logger("Training")

MODEL_NAME = "Sentinel-Anomaly-Model"


class DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return ["mock_prediction"] * len(model_input)


def train():
    mlflow.set_experiment("Sentinel-Retraining")

    with mlflow.start_run() as run:

        # ------------------------
        # Mock Metrics
        # ------------------------
        mlflow.log_param("epochs", 5)
        mlflow.log_metric("mAP", 0.82)
        mlflow.log_metric("precision", 0.79)

        # ------------------------
        # Log Proper MLflow Model
        # ------------------------
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DummyModel()
        )

        model_uri = f"runs:/{run.info.run_id}/model"

        # ------------------------
        # Register Model
        # ------------------------
        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        logger.info(f"Model registered as version {result.version}")

        # ------------------------
        # Promote to Production
        # ------------------------
        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )

        logger.info("Model promoted to Production.")


if __name__ == "__main__":
    train()
