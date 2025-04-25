import numpy as np
from typing import List, Dict, Any

class OpticalAlignmentCalibrator:
    """
    Calibrates and compensates for optical system misalignment in star trackers and cameras.
    Supports boresight correction, distortion estimation, and iterative refinement.
    """
    def __init__(self):
        self.boresight_offset = np.zeros(3)  # [dx, dy, dz] in radians
        self.distortion_coeffs = np.zeros(2)  # [k1, k2] radial distortion

    def calibrate_boresight(self, measured_vecs: List[np.ndarray], true_vecs: List[np.ndarray]) -> np.ndarray:
        # Estimate average rotation (small angle) between measured and true vectors
        delta = np.zeros(3)
        for m, t in zip(measured_vecs, true_vecs):
            cross = np.cross(m, t)
            delta += cross / (1 + np.dot(m, t))
        delta /= len(measured_vecs)
        self.boresight_offset = delta
        return delta

    def apply_boresight(self, vec: np.ndarray) -> np.ndarray:
        # Applies boresight correction (small angle approximation)
        return vec + np.cross(self.boresight_offset, vec)

    def calibrate_distortion(self, measured_points: List[np.ndarray], true_points: List[np.ndarray], cx: float, cy: float) -> np.ndarray:
        # Fit simple radial distortion model: r' = r(1 + k1*r^2 + k2*r^4)
        r_meas = [np.linalg.norm(p - [cx, cy]) for p in measured_points]
        r_true = [np.linalg.norm(p - [cx, cy]) for p in true_points]
        A = np.vstack([np.power(r, 2) for r in r_meas] + [np.power(r, 4) for r in r_meas]).T
        b = (np.array(r_true) - np.array(r_meas)) / np.array(r_meas)
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.distortion_coeffs = coeffs[:2]
        return self.distortion_coeffs

    def undistort(self, point: np.ndarray, cx: float, cy: float) -> np.ndarray:
        # Remove radial distortion
        dx, dy = point[0] - cx, point[1] - cy
        r = np.sqrt(dx**2 + dy**2)
        k1, k2 = self.distortion_coeffs
        scale = 1 + k1 * r**2 + k2 * r**4
        return np.array([cx + dx * scale, cy + dy * scale])
