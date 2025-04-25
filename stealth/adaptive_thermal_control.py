import numpy as np
from typing import Any, Dict, Optional

class AdaptiveThermalEmissionController:
    """
    Adaptive controller for dynamic thermal emission patterns (ADAPTIV panel control).
    Adjusts panel temperatures in real time to blend with environment or evade detection.
    """
    def __init__(self, panel_shape: tuple, min_temp_K: float = 273.15, max_temp_K: float = 373.15):
        self.panel_shape = panel_shape
        self.min_temp_K = min_temp_K
        self.max_temp_K = max_temp_K
        self.target_pattern = np.full(panel_shape, 293.15)
        self.current_pattern = np.full(panel_shape, 293.15)
        self.response_rate = 1.0  # K/s, can be tuned

    def set_target_pattern(self, pattern: np.ndarray):
        assert pattern.shape == self.panel_shape
        self.target_pattern = np.clip(pattern, self.min_temp_K, self.max_temp_K)

    def step(self, dt: float, env_temp_map: Optional[np.ndarray] = None, blend_weight: float = 1.0):
        # Optionally blend with environment for stealth
        if env_temp_map is not None:
            assert env_temp_map.shape == self.panel_shape
            blend = blend_weight * env_temp_map + (1 - blend_weight) * self.target_pattern
            self.target_pattern = np.clip(blend, self.min_temp_K, self.max_temp_K)
        # Move current pattern toward target
        delta = self.target_pattern - self.current_pattern
        max_delta = self.response_rate * dt
        delta = np.clip(delta, -max_delta, max_delta)
        self.current_pattern += delta
        self.current_pattern = np.clip(self.current_pattern, self.min_temp_K, self.max_temp_K)
        return self.current_pattern.copy()

    def get_current_pattern(self) -> np.ndarray:
        return self.current_pattern.copy()
