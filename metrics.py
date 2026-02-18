from dataclasses import dataclass

@dataclass
class ModelMetrics:
    mAP: float
    precision: float
    recall: float
    avg_confidence: float
    latency_ms: float

    def is_drift_detected(self, baseline_map: float, threshold: float):
        return abs(self.mAP - baseline_map) > threshold

    def is_latency_violation(self, latency_threshold: float):
        return self.latency_ms > latency_threshold
