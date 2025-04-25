from typing import Dict, Any, List, Optional, Callable
import time

class RealTimeStatusMonitor:
    """
    Monitors real-time system status and generates alerts for critical events or anomalies.
    Supports threshold-based, event-driven, and custom alerting logic.
    """
    def __init__(self, alert_callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None, thresholds: Optional[Dict[str, float]] = None):
        self.status_history: List[Dict[str, Any]] = []
        self.alert_log: List[Dict[str, Any]] = []
        self.alert_callbacks = alert_callbacks or []
        self.thresholds = thresholds or {}

    def ingest_status(self, status: Dict[str, Any]):
        self.status_history.append(status)
        alerts = self.detect_alerts(status)
        for alert in alerts:
            self.alert_log.append(alert)
            for cb in self.alert_callbacks:
                cb(alert)
        return alerts

    def detect_alerts(self, status: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts = []
        for k, v in self.thresholds.items():
            if k in status and status[k] > v:
                alerts.append({'type': 'threshold', 'key': k, 'value': status[k], 'threshold': v, 'time': time.time()})
        if 'anomaly' in status and status['anomaly']:
            alerts.append({'type': 'anomaly', 'details': status['anomaly'], 'time': time.time()})
        return alerts

    def get_status_history(self) -> List[Dict[str, Any]]:
        return self.status_history

    def get_alert_log(self) -> List[Dict[str, Any]]:
        return self.alert_log
