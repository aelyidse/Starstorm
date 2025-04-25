import numpy as np
from typing import Dict, Any, List, Optional

class UnifiedSensorDataCollector:
    """
    Collects and synchronizes data from heterogeneous sensors (IMU, GPS, star tracker, cameras, radar, etc.).
    Supports time alignment, health checks, and data buffering.
    """
    def __init__(self, sensor_list: List[str]):
        self.sensor_list = sensor_list
        self.data_buffers: Dict[str, List[Dict[str, Any]]] = {s: [] for s in sensor_list}
        self.timestamps: Dict[str, List[float]] = {s: [] for s in sensor_list}
        self.health: Dict[str, bool] = {s: True for s in sensor_list}

    def add_data(self, sensor: str, data: Dict[str, Any], timestamp: float):
        assert sensor in self.sensor_list
        self.data_buffers[sensor].append(data)
        self.timestamps[sensor].append(timestamp)

    def get_latest(self, sensor: str) -> Optional[Dict[str, Any]]:
        if not self.data_buffers[sensor]:
            return None
        return self.data_buffers[sensor][-1]

    def get_synced_snapshot(self, sync_time: float, window: float = 0.01) -> Dict[str, Any]:
        # Returns closest data for each sensor within window of sync_time
        snapshot = {}
        for s in self.sensor_list:
            times = np.array(self.timestamps[s])
            if len(times) == 0:
                snapshot[s] = None
                continue
            idx = np.argmin(np.abs(times - sync_time))
            if np.abs(times[idx] - sync_time) <= window:
                snapshot[s] = self.data_buffers[s][idx]
            else:
                snapshot[s] = None
        return snapshot

    def set_health(self, sensor: str, status: bool):
        self.health[sensor] = status

    def get_health(self, sensor: str) -> bool:
        return self.health.get(sensor, False)

class SensorPreprocessor:
    """
    Preprocesses raw sensor data for downstream fusion and analysis.
    Supports filtering, normalization, and outlier rejection.
    """
    def __init__(self):
        pass

    def preprocess(self, sensor: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # Example: apply basic filtering (extend for each sensor type)
        out = data.copy()
        if sensor == 'imu' and 'accel' in data:
            # Simple mean filter for accelerometer
            out['accel'] = np.clip(np.array(data['accel']), -50, 50)
        if sensor == 'gps' and 'pos' in data:
            # Remove outliers
            pos = np.array(data['pos'])
            if np.any(np.abs(pos) > 1e7):
                out['pos'] = np.zeros_like(pos)
        # Add more sensor-specific preprocessing as needed
        return out
