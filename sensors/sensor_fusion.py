import numpy as np
from typing import Dict, Any, Optional

class MultiSpectralSensorFusion:
    """
    Fuses multi-spectral and heterogeneous sensor data (e.g., visual, IR, UV, radar, GPS, IMU).
    Supports weighted averaging, Kalman filtering, and confidence-based integration.
    """
    def __init__(self):
        self.last_state: Optional[Dict[str, Any]] = None

    def weighted_average_fusion(self, data: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
        fused = {}
        keys = set().union(*(d.keys() for d in data.values()))
        for k in keys:
            vals = [data[s][k] for s in data if k in data[s] and data[s][k] is not None]
            ws = [weights[s] for s in data if k in data[s] and data[s][k] is not None]
            if vals:
                fused[k] = np.average(np.stack(vals), axis=0, weights=ws)
            else:
                fused[k] = None
        return fused

    def kalman_fusion(self, pred: np.ndarray, meas: np.ndarray, pred_cov: np.ndarray, meas_cov: np.ndarray) -> np.ndarray:
        # Simple Kalman update for a single variable
        K = pred_cov @ np.linalg.inv(pred_cov + meas_cov)
        fused = pred + K @ (meas - pred)
        return fused

    def confidence_fusion(self, data: Dict[str, Any], confidences: Dict[str, float]) -> Dict[str, Any]:
        # Like weighted average, but weights are dynamic/confidence-based
        return self.weighted_average_fusion(data, confidences)

    def fuse(self, multispectral_data: Dict[str, Any], mode: str = 'confidence', meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        multispectral_data: {'visual': {...}, 'ir': {...}, 'uv': {...}, 'radar': {...}, ...}
        mode: 'confidence', 'weighted', or 'kalman'
        meta: Optional dict for weights/confidences/covariances
        """
        if mode == 'confidence' and meta and 'confidences' in meta:
            fused = self.confidence_fusion(multispectral_data, meta['confidences'])
        elif mode == 'weighted' and meta and 'weights' in meta:
            fused = self.weighted_average_fusion(multispectral_data, meta['weights'])
        elif mode == 'kalman' and meta:
            fused = {}
            for k in multispectral_data:
                fused[k] = self.kalman_fusion(
                    meta['pred'][k], meta['meas'][k], meta['pred_cov'][k], meta['meas_cov'][k]
                )
        else:
            # Default: simple average
            fused = self.weighted_average_fusion(multispectral_data, {k: 1.0 for k in multispectral_data})
        self.last_state = fused
        return fused
