import numpy as np
from typing import Any, Dict, Optional

class AdaptiveMetamaterialController:
    """
    Dynamically adapts metamaterial electromagnetic properties (epsilon, mu, conductivity) for real-time signature management.
    Supports closed-loop control, pattern switching, and frequency-specific adaptation.
    """
    def __init__(self, panel_shape: tuple, epsilon_range: tuple = (1.0, 10.0), mu_range: tuple = (1.0, 10.0), conductivity_range: tuple = (0.0, 1e4)):
        self.panel_shape = panel_shape
        self.epsilon = np.full(panel_shape, epsilon_range[0])
        self.mu = np.full(panel_shape, mu_range[0])
        self.conductivity = np.full(panel_shape, conductivity_range[0])
        self.epsilon_range = epsilon_range
        self.mu_range = mu_range
        self.conductivity_range = conductivity_range
        self.target_pattern: Optional[Dict[str, np.ndarray]] = None
        self.response_rate = 0.1  # Fractional per step

    def set_target_pattern(self, epsilon: np.ndarray, mu: np.ndarray, conductivity: np.ndarray):
        assert epsilon.shape == self.panel_shape
        assert mu.shape == self.panel_shape
        assert conductivity.shape == self.panel_shape
        self.target_pattern = {
            'epsilon': np.clip(epsilon, *self.epsilon_range),
            'mu': np.clip(mu, *self.mu_range),
            'conductivity': np.clip(conductivity, *self.conductivity_range)
        }

    def step(self, dt: float = 1.0):
        if self.target_pattern is not None:
            for prop in ['epsilon', 'mu', 'conductivity']:
                current = getattr(self, prop)
                target = self.target_pattern[prop]
                delta = (target - current) * self.response_rate * dt
                setattr(self, prop, np.clip(current + delta, getattr(self, f'{prop}_range')[0], getattr(self, f'{prop}_range')[1]))
        return {
            'epsilon': self.epsilon.copy(),
            'mu': self.mu.copy(),
            'conductivity': self.conductivity.copy()
        }

    def get_current_pattern(self) -> Dict[str, np.ndarray]:
        return {
            'epsilon': self.epsilon.copy(),
            'mu': self.mu.copy(),
            'conductivity': self.conductivity.copy()
        }
