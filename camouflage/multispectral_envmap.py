import numpy as np
from typing import Dict, Any, Optional

class MultiSpectralEnvironmentalMapper:
    """
    Captures and stores environmental data across multiple spectral bands (visible, IR, UV, etc.).
    Provides spatially resolved maps for active camouflage reproduction.
    """
    def __init__(self, shape: tuple, bands: Optional[list] = None):
        self.shape = shape  # (height, width)
        self.bands = bands if bands is not None else ['visible', 'nir', 'mir', 'uv']
        self.env_maps = {band: np.zeros(shape) for band in self.bands}

    def update_band(self, band: str, data: np.ndarray):
        assert band in self.env_maps
        assert data.shape == self.shape
        self.env_maps[band] = data.copy()

    def get_map(self, band: str) -> np.ndarray:
        return self.env_maps[band].copy()

    def get_all_maps(self) -> Dict[str, np.ndarray]:
        return {b: m.copy() for b, m in self.env_maps.items()}

class MultiSpectralReproducer:
    """
    Reproduces environmental appearance across multiple spectral bands for active camouflage.
    Maps environmental data onto actuator arrays (e.g., display panels, emissive surfaces).
    """
    def __init__(self, panel_shape: tuple, bands: Optional[list] = None):
        self.panel_shape = panel_shape
        self.bands = bands if bands is not None else ['visible', 'nir', 'mir', 'uv']
        self.current_pattern = {band: np.zeros(panel_shape) for band in self.bands}

    def set_target_pattern(self, env_maps: Dict[str, np.ndarray]):
        for band, data in env_maps.items():
            if band in self.current_pattern:
                # Resample if needed
                from scipy.ndimage import zoom
                zoom_factors = (
                    self.panel_shape[0] / data.shape[0],
                    self.panel_shape[1] / data.shape[1]
                )
                self.current_pattern[band] = zoom(data, zoom_factors, order=1)

    def get_current_pattern(self) -> Dict[str, np.ndarray]:
        return {b: p.copy() for b, p in self.current_pattern.items()}
