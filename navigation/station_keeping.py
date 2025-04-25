import numpy as np
from typing import Dict, Any

class StationKeepingController:
    """
    Maintains orbital position within a specified box or tolerance.
    Computes required maneuvers for longitude, inclination, and eccentricity control.
    """
    def __init__(self, mu: float = 3.986004418e14):
        self.mu = mu  # Earth's gravitational parameter

    def compute_station_keeping(self, state: Dict[str, Any], target: Dict[str, Any], tolerance_m: float = 1000.0) -> Dict[str, Any]:
        # state: {'pos': np.array, 'vel': np.array}
        # target: {'pos': np.array, 'vel': np.array}
        pos_err = target['pos'] - state['pos']
        vel_err = target['vel'] - state['vel']
        distance = np.linalg.norm(pos_err)
        if distance < tolerance_m:
            return {'maneuver': 'none', 'distance_m': distance}
        # Compute required delta-v direction (Hill's frame, simplified)
        delta_v = pos_err * 0.01 + vel_err * 0.5  # Tunable gains
        max_delta_v = 2.0  # m/s, limit for gentle station-keeping
        if np.linalg.norm(delta_v) > max_delta_v:
            delta_v = delta_v / np.linalg.norm(delta_v) * max_delta_v
        return {'maneuver': 'delta_v', 'delta_v_mps': delta_v, 'distance_m': distance}
