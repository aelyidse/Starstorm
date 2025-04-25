from typing import Dict, Any, Optional
import time

class VehicleState:
    """
    Comprehensive vehicle state representation for telemetry processing and system monitoring.
    Tracks position, velocity, attitude, system health, payloads, environment, and mission status.
    """
    def __init__(self):
        self.timestamp: float = time.time()
        self.position: Dict[str, float] = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0}
        self.velocity: Dict[str, float] = {'vx': 0.0, 'vy': 0.0, 'vz': 0.0}
        self.attitude: Dict[str, float] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.system_health: Dict[str, str] = {}  # subsystem: status
        self.payload_status: Dict[str, str] = {}  # payload: status
        self.environment: Dict[str, Any] = {}  # e.g., temperature, radiation
        self.mission_status: Dict[str, Any] = {}  # e.g., phase, objectives
        self.extra: Dict[str, Any] = {}  # extensible for custom fields

    def update(self, data: Dict[str, Any]):
        self.timestamp = data.get('timestamp', time.time())
        if 'position' in data:
            self.position.update(data['position'])
        if 'velocity' in data:
            self.velocity.update(data['velocity'])
        if 'attitude' in data:
            self.attitude.update(data['attitude'])
        if 'system_health' in data:
            self.system_health.update(data['system_health'])
        if 'payload_status' in data:
            self.payload_status.update(data['payload_status'])
        if 'environment' in data:
            self.environment.update(data['environment'])
        if 'mission_status' in data:
            self.mission_status.update(data['mission_status'])
        if 'extra' in data:
            self.extra.update(data['extra'])

    def as_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'position': self.position,
            'velocity': self.velocity,
            'attitude': self.attitude,
            'system_health': self.system_health,
            'payload_status': self.payload_status,
            'environment': self.environment,
            'mission_status': self.mission_status,
            'extra': self.extra
        }

    def __repr__(self):
        return f"<VehicleState {self.as_dict()}>"
