import mlflow
import mlflow.pytorch
from logger import get_logger

logger = get_logger("MLflow")

def init_mlflow(experiment_name="Sentinel-Anomaly-Detection"):
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set: {experiment_name}")

def log_run(params: dict, metrics: dict, model=None):
    with mlflow.start_run():

        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model (optional)
        if model:
            mlflow.pytorch.log_model(model, "model")

        logger.info("MLflow run logged successfully")
