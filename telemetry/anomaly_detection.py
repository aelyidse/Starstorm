from typing import Dict, Any, List, Optional
import numpy as np

class TelemetryAnomalyDetector:
    """
    Detects anomalies in telemetry data for validation and alerting.
    Supports threshold checks, statistical outlier detection, and rule-based validation.
    """
    def __init__(self, thresholds: Optional[Dict[str, Dict[str, float]]] = None):
        # thresholds: {'field': {'min': x, 'max': y}}
        self.thresholds = thresholds or {}
        self.history: List[Dict[str, Any]] = []
        self.anomalies: List[Dict[str, Any]] = []

    def validate(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        detected = self.check_thresholds(telemetry)
        detected += self.detect_outliers(telemetry)
        self.history.append({'telemetry': telemetry, 'anomalies': detected})
        if detected:
            self.anomalies.extend(detected)
        return detected

    def check_thresholds(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for field, lims in self.thresholds.items():
            if field in telemetry:
                val = telemetry[field]
                if 'min' in lims and val < lims['min']:
                    results.append({'field': field, 'type': 'below_min', 'value': val, 'min': lims['min']})
                if 'max' in lims and val > lims['max']:
                    results.append({'field': field, 'type': 'above_max', 'value': val, 'max': lims['max']})
        return results

    def detect_outliers(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Simple statistical outlier detection (z-score > 3)
        outliers = []
        if len(self.history) < 10:
            return outliers
        for field in telemetry:
            vals = [h['telemetry'][field] for h in self.history[-50:] if field in h['telemetry']]
            if len(vals) < 10:
                continue
            mean = np.mean(vals)
            std = np.std(vals)
            if std == 0:
                continue
            z = abs((telemetry[field] - mean) / std)
            if z > 3:
                outliers.append({'field': field, 'type': 'outlier', 'value': telemetry[field], 'mean': mean, 'std': std, 'z': z})
        return outliers

    def get_anomalies(self) -> List[Dict[str, Any]]:
        return self.anomalies

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
