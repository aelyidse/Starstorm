import numpy as np
from typing import Dict, Any

class IMUProcessor:
    """
    Processes inertial measurement unit (IMU) data with drift compensation.
    Supports bias estimation, zero-velocity updates, and optional fusion with aiding sensors.
    """
    def __init__(self, gyro_bias: np.ndarray = None, accel_bias: np.ndarray = None, drift_rate: float = 1e-4):
        self.gyro_bias = gyro_bias if gyro_bias is not None else np.zeros(3)
        self.accel_bias = accel_bias if accel_bias is not None else np.zeros(3)
        self.drift_rate = drift_rate  # rad/s or m/s^2 per second
        self.estimated_gyro_bias = self.gyro_bias.copy()
        self.estimated_accel_bias = self.accel_bias.copy()
        self.last_update_time = None

    def process(self, gyro: np.ndarray, accel: np.ndarray, dt: float, aiding: Dict[str, Any] = None) -> Dict[str, Any]:
        # Remove estimated bias
        gyro_unbiased = gyro - self.estimated_gyro_bias
        accel_unbiased = accel - self.estimated_accel_bias
        # Bias drift compensation (simple integrator, can be replaced with Kalman filter)
        if aiding and aiding.get('zero_velocity', False):
            # Zero-velocity update (ZUPT) for bias correction
            self.estimated_accel_bias += accel_unbiased * 0.01  # small correction
            self.estimated_gyro_bias += gyro_unbiased * 0.01
        else:
            # Simulate slow random walk (drift)
            self.estimated_gyro_bias += np.random.normal(0, self.drift_rate * dt, 3)
            self.estimated_accel_bias += np.random.normal(0, self.drift_rate * dt, 3)
        return {
            'gyro_unbiased': gyro_unbiased,
            'accel_unbiased': accel_unbiased,
            'estimated_gyro_bias': self.estimated_gyro_bias.copy(),
            'estimated_accel_bias': self.estimated_accel_bias.copy()
        }
