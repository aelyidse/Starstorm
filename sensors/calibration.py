import numpy as np
from typing import Dict, Any, List, Optional

class SensorAutoCalibrator:
    """
    Performs automatic calibration routines for sensor alignment (extrinsic/intrinsic).
    Supports offset, scale, and orientation correction between heterogeneous sensors.
    """
    def __init__(self):
        self.calibration_params: Dict[str, Dict[str, Any]] = {}

    def calibrate_offset(self, ref_data: np.ndarray, sensor_data: np.ndarray) -> float:
        # Estimate offset between reference and sensor
        return float(np.mean(ref_data - sensor_data))

    def calibrate_scale(self, ref_data: np.ndarray, sensor_data: np.ndarray) -> float:
        # Estimate scaling factor
        return float(np.std(ref_data) / (np.std(sensor_data) + 1e-8))

    def calibrate_orientation(self, ref_vectors: np.ndarray, sensor_vectors: np.ndarray) -> np.ndarray:
        # Find optimal rotation matrix using SVD (Kabsch algorithm)
        H = sensor_vectors.T @ ref_vectors
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        return R

    def run_full_calibration(self, ref: Dict[str, Any], sensors: Dict[str, Any], align_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        ref: reference sensor data dict
        sensors: dict of sensor_name -> data dict
        align_keys: list of keys to align/calibrate (e.g., ['pos', 'accel'])
        """
        results = {}
        for s, data in sensors.items():
            params = {}
            for k in align_keys:
                if k in ref and k in data:
                    ref_val = np.array(ref[k])
                    sensor_val = np.array(data[k])
                    params[k+'_offset'] = self.calibrate_offset(ref_val, sensor_val)
                    params[k+'_scale'] = self.calibrate_scale(ref_val, sensor_val)
                    if ref_val.ndim == 2 and sensor_val.ndim == 2 and ref_val.shape == sensor_val.shape:
                        params[k+'_orientation'] = self.calibrate_orientation(ref_val, sensor_val)
            results[s] = params
            self.calibration_params[s] = params
        return results

    def get_calibration(self, sensor: str) -> Optional[Dict[str, Any]]:
        return self.calibration_params.get(sensor, None)
