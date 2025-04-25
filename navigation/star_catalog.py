import numpy as np
from typing import List, Tuple, Dict, Any
import bisect

class StellarPatternDatabase:
    """
    Stores a catalog of star patterns for star tracker recognition.
    Supports efficient angular distance search and k-vector pattern matching.
    """
    def __init__(self, star_list: List[Tuple[str, np.ndarray]]):
        self.star_list = star_list  # [(star_id, unit_vector)]
        # Precompute pairwise angular distances for fast pattern matching
        self.patterns = self._build_patterns(star_list)
        self.sorted_angles = sorted(self.patterns.keys())

    def _build_patterns(self, star_list):
        patterns = {}
        for i, (id1, v1) in enumerate(star_list):
            for j, (id2, v2) in enumerate(star_list):
                if i < j:
                    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
                    patterns[(id1, id2)] = angle
        return patterns

    def find_pattern(self, observed_angle: float, tolerance_deg: float = 0.1) -> List[Tuple[str, str]]:
        # Efficiently search for all star pairs matching observed_angle within tolerance
        idx = bisect.bisect_left(self.sorted_angles, observed_angle - tolerance_deg)
        matches = []
        while idx < len(self.sorted_angles) and self.sorted_angles[idx] <= observed_angle + tolerance_deg:
            angle = self.sorted_angles[idx]
            for pair, a in self.patterns.items():
                if abs(a - angle) < 1e-6 and abs(a - observed_angle) <= tolerance_deg:
                    matches.append(pair)
            idx += 1
        return matches

    def get_star_vector(self, star_id: str) -> np.ndarray:
        for sid, vec in self.star_list:
            if sid == star_id:
                return vec
        raise KeyError(f"Star ID {star_id} not found in catalog.")

    def add_star(self, star_id: str, vector: np.ndarray):
        self.star_list.append((star_id, vector))
        self.patterns = self._build_patterns(self.star_list)
        self.sorted_angles = sorted(self.patterns.keys())
