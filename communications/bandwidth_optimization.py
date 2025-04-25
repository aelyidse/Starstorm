import numpy as np
from typing import Dict, Any, List, Optional

class TelemetryBandwidthOptimizer:
    """
    Optimizes telemetry and command data transmission for bandwidth-limited channels.
    Supports prioritization, compression, and adaptive rate control.
    """
    def __init__(self, max_bandwidth_bps: float):
        self.max_bandwidth_bps = max_bandwidth_bps
        self.priorities = {}
        self.compression_level = 0  # 0 = none, 1 = low, 2 = high

    def set_priorities(self, priorities: Dict[str, int]):
        # priorities: dict of data_type -> priority (lower = higher priority)
        self.priorities = priorities

    def set_compression_level(self, level: int):
        assert level in [0, 1, 2]
        self.compression_level = level

    def compress(self, data: bytes) -> bytes:
        import zlib
        if self.compression_level == 0:
            return data
        elif self.compression_level == 1:
            return zlib.compress(data, level=3)
        else:
            return zlib.compress(data, level=9)

    def optimize(self, data_packets: List[Dict[str, Any]], time_window_s: float = 1.0) -> List[Dict[str, Any]]:
        # data_packets: [{'type': str, 'data': bytes, 'size': int}]
        # Sort by priority
        sorted_packets = sorted(
            data_packets,
            key=lambda pkt: self.priorities.get(pkt['type'], 100)
        )
        # Compress and select packets to fit bandwidth
        total_bits = 0
        selected = []
        for pkt in sorted_packets:
            compressed_data = self.compress(pkt['data'])
            pkt_size_bits = len(compressed_data) * 8
            if total_bits + pkt_size_bits > self.max_bandwidth_bps * time_window_s:
                continue  # Skip lower priority
            selected.append({**pkt, 'data': compressed_data, 'size': pkt_size_bits // 8})
            total_bits += pkt_size_bits
        return selected

    def get_bandwidth_utilization(self, selected_packets: List[Dict[str, Any]], time_window_s: float = 1.0) -> float:
        total_bits = sum(pkt['size'] * 8 for pkt in selected_packets)
        return total_bits / (self.max_bandwidth_bps * time_window_s)
