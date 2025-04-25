import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from typing import Dict, Any, Tuple

class RealTimeBackgroundMatcher:
    """
    Real-time image processing for background matching in active camouflage.
    Performs denoising, color normalization, and spatial resampling for seamless blending.
    """
    def __init__(self, panel_shape: Tuple[int, int], blur_sigma: float = 1.0):
        self.panel_shape = panel_shape
        self.blur_sigma = blur_sigma

    def process(self, image: np.ndarray) -> np.ndarray:
        # 1. Denoise background image
        filtered = gaussian_filter(image, self.blur_sigma)
        # 2. Normalize intensity (scale to [0, 1])
        norm = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered) + 1e-8)
        # 3. Resample to panel shape for actuator reproduction
        if norm.shape != self.panel_shape:
            zoom_factors = (
                self.panel_shape[0] / norm.shape[0],
                self.panel_shape[1] / norm.shape[1]
            )
            norm = zoom(norm, zoom_factors, order=1)
        return norm

    def match_multispectral(self, env_maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Process each spectral band for background matching
        matched = {}
        for band, img in env_maps.items():
            matched[band] = self.process(img)
        return matched
