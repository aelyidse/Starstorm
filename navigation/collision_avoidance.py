import numpy as np
from typing import List, Dict, Any, Tuple

class CollisionAvoidanceSystem:
    """
    Detects and avoids collisions with air and space objects.
    Supports nearest approach prediction and maneuver recommendation.
    """
    def __init__(self, safety_radius_m: float = 100.0):
        self.safety_radius_m = safety_radius_m

    def detect_conflicts(self, own_pos: np.ndarray, own_vel: np.ndarray, objects: List[Dict[str, Any]], lookahead_s: float = 300.0) -> List[Dict[str, Any]]:
        # objects: [{'pos': np.array, 'vel': np.array, 'type': 'air'|'space'}]
        conflicts = []
        for obj in objects:
            tca, dca = self._predict_closest_approach(own_pos, own_vel, obj['pos'], obj['vel'])
            if 0 <= tca <= lookahead_s and dca < self.safety_radius_m:
                conflicts.append({'type': obj['type'], 'tca_s': tca, 'dca_m': dca, 'object': obj})
        return conflicts

    def _predict_closest_approach(self, p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> Tuple[float, float]:
        # Returns time of closest approach (tca) and distance (dca)
        dp = p2 - p1
        dv = v2 - v1
        dv2 = np.dot(dv, dv)
        if dv2 == 0:
            tca = 0.0
        else:
            tca = -np.dot(dp, dv) / dv2
        closest_p1 = p1 + v1 * tca
        closest_p2 = p2 + v2 * tca
        dca = np.linalg.norm(closest_p1 - closest_p2)
        return tca, dca

    def recommend_maneuver(self, own_pos: np.ndarray, own_vel: np.ndarray, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Simple avoidance: recommend velocity change perpendicular to conflict
        if not conflicts:
            return {'maneuver': 'none'}
        # Take the most urgent conflict
        conflict = min(conflicts, key=lambda c: c['tca_s'])
        obj = conflict['object']
        rel_pos = obj['pos'] - own_pos
        rel_vel = obj['vel'] - own_vel
        # Perpendicular direction
        perp = np.cross(rel_pos, rel_vel)
        if np.linalg.norm(perp) < 1e-6:
            perp = np.random.randn(3)
        perp /= np.linalg.norm(perp)
        delta_v = perp * 5.0  # Recommend 5 m/s perpendicular change
        return {'maneuver': 'delta_v', 'delta_v_mps': delta_v, 'conflict': conflict}
