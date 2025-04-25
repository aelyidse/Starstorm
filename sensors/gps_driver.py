import numpy as np
from typing import Dict, Any

class GPSDriver:
    """
    Simulates or interfaces with a GPS receiver.
    Provides position and velocity data.
    """
    def __init__(self, noise_std: float = 1.0):
        self.noise_std = noise_std

    def read(self) -> Dict[str, Any]:
        # Simulate GPS data
        pos = np.random.normal(0, self.noise_std, 3)
        vel = np.random.normal(0, self.noise_std, 3)
        return {'pos': pos, 'vel': vel}
