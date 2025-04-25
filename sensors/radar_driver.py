import numpy as np
from typing import Dict, Any

class RadarDriver:
    """
    Simulates or interfaces with a radar sensor.
    Provides range, angle, and Doppler data.
    """
    def __init__(self, n_targets: int = 3):
        self.n_targets = n_targets

    def read(self) -> Dict[str, Any]:
        # Simulate radar detections
        ranges = np.random.uniform(100, 10000, self.n_targets)
        angles = np.random.uniform(-np.pi, np.pi, self.n_targets)
        doppler = np.random.normal(0, 10, self.n_targets)
        return {'ranges': ranges, 'angles': angles, 'doppler': doppler}
