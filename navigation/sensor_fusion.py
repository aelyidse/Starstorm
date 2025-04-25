import numpy as np
from typing import Dict, Any, Optional

class NavigationSensorFusion:
    """
    Fuses GPS, IMU, and star tracker data for robust state estimation.
    Implements an extended Kalman filter (EKF) for position, velocity, and attitude.
    """
    def __init__(self, initial_state: Dict[str, Any], initial_cov: Optional[np.ndarray] = None):
        # State: [x, y, z, vx, vy, vz, q0, q1, q2, q3, bgx, bgy, bgz]
        self.state = initial_state.copy()
        self.P = initial_cov if initial_cov is not None else np.eye(13)
        self.Q = np.eye(13) * 1e-4  # Process noise
        self.R_gps = np.eye(6) * 10.0  # GPS measurement noise
        self.R_star = np.eye(4) * 0.01  # Star tracker attitude noise
        self.R_imu = np.eye(6) * 0.1  # IMU pseudo-measurement noise

    def predict(self, imu: Dict[str, Any], dt: float):
        # Simple state propagation using IMU (no gravity or earth rotation for brevity)
        x = self.state
        acc = imu['accel_unbiased']
        gyro = imu['gyro_unbiased']
        # Position and velocity update
        for i in range(3):
            x['vel'][i] += acc[i] * dt
            x['pos'][i] += x['vel'][i] * dt
        # Attitude update (quaternion integration)
        omega = gyro
        q = x['quat']
        dq = 0.5 * self._quat_mult(q, np.concatenate([[0], omega]))
        q_new = q + dq * dt
        q_new /= np.linalg.norm(q_new)
        x['quat'] = q_new
        # Bias update (random walk)
        for i, axis in enumerate(['bgx', 'bgy', 'bgz']):
            x[axis] += np.random.normal(0, 1e-6)
        # Covariance update (linearized)
        self.P += self.Q * dt

    def update_gps(self, gps: Dict[str, Any]):
        # GPS provides position and velocity
        z = np.concatenate([gps['pos'], gps['vel']])
        x = np.concatenate([self.state['pos'], self.state['vel']])
        y = z - x
        H = np.zeros((6, 13))
        H[:3, :3] = np.eye(3)  # pos
        H[3:6, 3:6] = np.eye(3)  # vel
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self._apply_state_update(dx)
        self.P = (np.eye(13) - K @ H) @ self.P

    def update_star(self, star: Dict[str, Any]):
        # Star tracker provides quaternion
        z = star['quat']
        x = self.state['quat']
        y = z - x
        H = np.zeros((4, 13))
        H[:, 6:10] = np.eye(4)
        S = H @ self.P @ H.T + self.R_star
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self._apply_state_update(dx)
        self.P = (np.eye(13) - K @ H) @ self.P

    def _apply_state_update(self, dx):
        # Applies state correction dx (length 13)
        self.state['pos'] += dx[:3]
        self.state['vel'] += dx[3:6]
        self.state['quat'] += dx[6:10]
        self.state['quat'] /= np.linalg.norm(self.state['quat'])
        self.state['bgx'] += dx[10]
        self.state['bgy'] += dx[11]
        self.state['bgz'] += dx[12]

    def _quat_mult(self, q, r):
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = r
        return np.array([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1
        ])

    def get_state(self) -> Dict[str, Any]:
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.state.items()}
