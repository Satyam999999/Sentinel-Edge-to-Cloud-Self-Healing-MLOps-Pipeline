import mlflow
from logger import get_logger

logger = get_logger("Training")

def train():
    mlflow.set_experiment("Sentinel-Retraining")

    with mlflow.start_run():
        mlflow.log_param("epochs", 5)
        mlflow.log_metric("mock_mAP", 0.82)
        mlflow.log_metric("mock_precision", 0.79)

        logger.info("Mock retraining complete (CI mode).")

if __name__ == "__main__":
    train()
