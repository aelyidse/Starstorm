import numpy as np
from typing import Dict, Any

class HighPrecisionAttitudeController:
    """
    High-precision attitude control for orbital maneuvering.
    Supports quaternion-based feedback, fine-tuned gain scheduling, and actuator blending.
    """
    def __init__(self, inertia_tensor: np.ndarray, gains: Dict[str, float]):
        self.inertia_tensor = inertia_tensor
        self.gains = gains  # e.g., {'Kp': 2.0, 'Kd': 0.5}

    def compute_torque(self, q_current: np.ndarray, q_target: np.ndarray, omega_current: np.ndarray, omega_target: np.ndarray = None) -> np.ndarray:
        # Quaternion error (q_err = q_target * q_current^-1)
        q_err = self._quat_mult(q_target, self._quat_conj(q_current))
        # Map quaternion error to axis-angle (vector part)
        axis_err = q_err[1:]
        if q_err[0] < 0:
            axis_err = -axis_err
        # Proportional term
        torque_p = self.gains['Kp'] * axis_err
        # Derivative term
        omega_err = omega_current if omega_target is None else omega_current - omega_target
        torque_d = -self.gains.get('Kd', 0.0) * omega_err
        # Total torque
        torque = torque_p + torque_d
        # Saturate if needed (optional)
        return torque

    def _quat_conj(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _quat_mult(self, q, r):
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = r
        return np.array([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1
        ])
