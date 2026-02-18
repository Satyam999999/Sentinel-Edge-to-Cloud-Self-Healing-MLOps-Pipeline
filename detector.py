import time
from ultralytics import YOLO
from logger import get_logger

logger = get_logger("Detector")

class AnomalyDetector:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.model = YOLO(weights_path)
        self.device = device
        logger.info("YOLOv8 model loaded successfully")

    def predict(self, image_path: str, conf_threshold: float):
        start_time = time.time()

        results = self.model(
            image_path,
            conf=conf_threshold,
            device=self.device
        )

        latency_ms = (time.time() - start_time) * 1000

        confidences = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    confidences.append(float(box.conf))

        logger.info(f"Inference latency: {latency_ms:.2f} ms")

        return results, confidences, latency_ms
