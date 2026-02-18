import mlflow
from ultralytics import YOLO
from logger import get_logger

logger = get_logger("Training")

def train():
    mlflow.set_experiment("Sentinel-Retraining")

    with mlflow.start_run():

        model = YOLO("yolov8n.pt")

        results = model.train(
            data="data.yaml",   # your dataset config
            epochs=5,
            imgsz=640
        )

        mlflow.log_param("epochs", 5)
        mlflow.log_metric("final_map", results.results_dict.get("metrics/mAP50", 0))

        mlflow.pytorch.log_model(model.model, "model")

        logger.info("Retraining complete and logged.")

if __name__ == "__main__":
    train()
