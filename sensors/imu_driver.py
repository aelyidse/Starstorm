import numpy as np
from typing import Dict, Any

class IMUDriver:
    """
    Simulates or interfaces with an inertial measurement unit (IMU).
    Provides accelerometer and gyroscope data.
    """
    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    def read(self) -> Dict[str, Any]:
        # Simulate IMU data
        accel = np.random.normal(0, self.noise_std, 3)
        gyro = np.random.normal(0, self.noise_std, 3)
        return {'accel': accel, 'gyro': gyro}
