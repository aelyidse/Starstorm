import numpy as np
from typing import Dict, Any, Optional

class SpectrumManager:
    """
    Manages emission and reflectance across visual, IR, and UV bands for adaptive camouflage.
    Supports independent or coordinated control of each spectral channel.
    """
    def __init__(self, panel_shape: tuple, bands: Optional[list] = None):
        self.panel_shape = panel_shape
        self.bands = bands if bands is not None else ['visible', 'ir', 'uv']
        self.target_pattern = {band: np.zeros(panel_shape) for band in self.bands}
        self.current_pattern = {band: np.zeros(panel_shape) for band in self.bands}
        self.response_rate = 1.0  # Fractional per step (can be tuned per band)

    def set_target_pattern(self, band: str, pattern: np.ndarray):
        assert band in self.target_pattern
        assert pattern.shape == self.panel_shape
        self.target_pattern[band] = pattern.copy()

    def step(self, dt: float = 1.0):
        # Move current pattern toward target for each band
        for band in self.bands:
            delta = self.target_pattern[band] - self.current_pattern[band]
            max_delta = self.response_rate * dt
            delta = np.clip(delta, -max_delta, max_delta)
            self.current_pattern[band] += delta
            self.current_pattern[band] = np.clip(self.current_pattern[band], 0.0, 1.0)
        return {b: p.copy() for b, p in self.current_pattern.items()}

    def get_current_pattern(self) -> Dict[str, np.ndarray]:
        return {b: p.copy() for b, p in self.current_pattern.items()}

    def blend_patterns(self, weights: Dict[str, float]) -> np.ndarray:
        # Blend all spectral bands into a composite for sensor fusion or visualization
        composite = np.zeros(self.panel_shape)
        for band, w in weights.items():
            composite += self.current_pattern[band] * w
        return composite / max(1e-8, sum(weights.values()))
