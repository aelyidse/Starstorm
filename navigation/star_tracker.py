import numpy as np
from typing import List, Tuple, Dict, Any

class StarTracker:
    """
    Simulates star tracker pattern recognition and attitude determination.
    Performs centroiding, pattern matching, and attitude solution (QUEST algorithm).
    """
    def __init__(self, star_catalog: List[Tuple[str, np.ndarray]]):
        """
        star_catalog: List of (star_id, inertial unit vector)
        """
        self.star_catalog = star_catalog

    def centroid_stars(self, image: np.ndarray, threshold: float = 0.5) -> List[np.ndarray]:
        # Simple centroiding: returns list of [x, y] centroids above threshold
        stars = np.argwhere(image > threshold)
        return [np.array([float(y), float(x)]) for x, y in stars]

    def match_patterns(self, observed_vectors: List[np.ndarray], tolerance_deg: float = 1.0) -> List[Tuple[str, np.ndarray]]:
        # Naive nearest-neighbor matching to catalog
        matches = []
        for obs in observed_vectors:
            best_id = None
            best_angle = float('inf')
            for star_id, cat_vec in self.star_catalog:
                angle = np.degrees(np.arccos(np.clip(np.dot(obs, cat_vec), -1.0, 1.0)))
                if angle < best_angle and angle < tolerance_deg:
                    best_angle = angle
                    best_id = star_id
            if best_id is not None:
                matches.append((best_id, obs))
        return matches

    def attitude_determination(self, body_vectors: List[np.ndarray], inertial_vectors: List[np.ndarray]) -> np.ndarray:
        # QUEST algorithm for optimal quaternion
        B = np.zeros((3, 3))
        for b, r in zip(body_vectors, inertial_vectors):
            B += np.outer(b, r)
        S = B + B.T
        sigma = np.trace(B)
        Z = np.array([
            B[1, 2] - B[2, 1],
            B[2, 0] - B[0, 2],
            B[0, 1] - B[1, 0]
        ])
        K = np.zeros((4, 4))
        K[0, 0] = sigma
        K[0, 1:4] = Z
        K[1:4, 0] = Z
        K[1:4, 1:4] = S - sigma * np.eye(3)
        # Eigenvector with max eigenvalue gives quaternion
        eigvals, eigvecs = np.linalg.eigh(K)
        q = eigvecs[:, np.argmax(eigvals)]
        if q[0] < 0:
            q = -q
        return q  # [w, x, y, z]
