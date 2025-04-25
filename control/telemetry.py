import json
import zlib
from typing import Dict, Any, List, Optional

class TelemetryAggregator:
    """
    Aggregates telemetry data from multiple sources for efficient transmission.
    Supports snapshot, delta, and prioritized aggregation.
    """
    def __init__(self, keys_priority: Optional[List[str]] = None):
        self.last_snapshot: Dict[str, Any] = {}
        self.keys_priority = keys_priority or []

    def aggregate(self, telemetry: Dict[str, Any], mode: str = 'delta') -> Dict[str, Any]:
        if mode == 'snapshot':
            self.last_snapshot = telemetry.copy()
            return telemetry.copy()
        elif mode == 'delta':
            delta = {k: v for k, v in telemetry.items() if self.last_snapshot.get(k) != v}
            self.last_snapshot.update(delta)
            return delta
        elif mode == 'priority':
            prioritized = {k: telemetry[k] for k in self.keys_priority if k in telemetry}
            self.last_snapshot.update(prioritized)
            return prioritized
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

class TelemetryCompressor:
    """
    Compresses telemetry payloads for bandwidth-constrained communication.
    Supports zlib (deflate), with optional JSON serialization.
    """
    def __init__(self, level: int = 6):
        self.level = level

    def compress(self, payload: Dict[str, Any]) -> bytes:
        json_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        return zlib.compress(json_bytes, self.level)

    def decompress(self, compressed: bytes) -> Dict[str, Any]:
        json_bytes = zlib.decompress(compressed)
        return json.loads(json_bytes.decode('utf-8'))
