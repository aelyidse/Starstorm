import numpy as np
from typing import Dict, Any, Optional

class ThermalSignatureModel:
    """
    Models thermal emission and apparent temperature for ADAPTIV-style stealth.
    Supports panel-level temperature control and environmental blending.
    """
    def __init__(self, panel_layout: np.ndarray, emissivity: float = 0.95):
        self.panel_layout = panel_layout  # 2D array (panel grid)
        self.emissivity = emissivity
        self.sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        self.panel_temps = np.full(panel_layout.shape, 293.15)  # Default 20°C

    def set_panel_temps(self, temps: np.ndarray):
        assert temps.shape == self.panel_layout.shape
        self.panel_temps = temps

    def compute_radiance(self) -> np.ndarray:
        # Radiance (W/m^2) for each panel
        return self.emissivity * self.sigma * np.power(self.panel_temps, 4)

    def apparent_temperature(self) -> np.ndarray:
        # Apparent temperature (K) for each panel
        return self.panel_temps.copy()

class EnvironmentalMatcher:
    """
    Matches thermal signature to environment for adaptive stealth.
    Computes optimal panel temperatures to blend with background.
    """
    def __init__(self, panel_layout: np.ndarray):
        self.panel_layout = panel_layout

    def match_environment(self, env_temp_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # env_temp_map: 2D array of background temps (K)
        # mask: 2D boolean array (where to match)
        assert env_temp_map.shape == self.panel_layout.shape
        if mask is not None:
            assert mask.shape == self.panel_layout.shape
            matched = np.where(mask, env_temp_map, 293.15)  # Default to 20°C if not matched
        else:
            matched = env_temp_map.copy()
        return matched

class ThermalSignatureTransformer:
    """
    Transforms and maps the thermal signature of a target vehicle onto the current panel grid for signature mimicry.
    Supports resizing, interpolation, and pattern fitting.
    """
    def __init__(self, panel_layout: np.ndarray):
        self.panel_layout = panel_layout

    def mimic_signature(self, target_signature: np.ndarray, method: str = 'bilinear') -> np.ndarray:
        # Resample target_signature to match panel_layout shape
        from scipy.ndimage import zoom
        target_shape = target_signature.shape
        panel_shape = self.panel_layout.shape
        zoom_factors = (panel_shape[0] / target_shape[0], panel_shape[1] / target_shape[1])
        if method == 'bilinear':
            transformed = zoom(target_signature, zoom_factors, order=1)
        elif method == 'nearest':
            transformed = zoom(target_signature, zoom_factors, order=0)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        return transformed
