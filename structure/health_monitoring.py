import numpy as np
from typing import Dict, Any, List, Optional

class StructuralHealthMonitor:
    """
    Monitors structural health using sensor data (strain, vibration, temperature, etc.).
    Detects anomalies, damage, and degradation in CNT-reinforced structures.
    """
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        # thresholds: e.g., {'strain': 0.01, 'temp': 80, 'vibration': 5.0}
        self.thresholds = thresholds or {'strain': 0.01, 'temp': 80, 'vibration': 5.0}
        self.history: List[Dict[str, Any]] = []
        self.anomalies: List[Dict[str, Any]] = []

    def ingest(self, sensor_data: Dict[str, Any]):
        self.history.append(sensor_data)
        anomaly = self.detect_anomaly(sensor_data)
        if anomaly:
            self.anomalies.append(anomaly)
        return anomaly

    def detect_anomaly(self, sensor_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Check for threshold exceedances
        anomaly = {}
        for k, v in self.thresholds.items():
            if k in sensor_data and abs(sensor_data[k]) > v:
                anomaly[k] = sensor_data[k]
        return anomaly if anomaly else None

    def get_anomalies(self) -> List[Dict[str, Any]]:
        return self.anomalies

    def get_health_status(self) -> str:
        if self.anomalies:
            return f"ANOMALY DETECTED: {self.anomalies[-1]}"
        return "HEALTHY"

    def summarize(self) -> Dict[str, Any]:
        return {
            'history_len': len(self.history),
            'anomaly_count': len(self.anomalies),
            'last_status': self.get_health_status()
        }
