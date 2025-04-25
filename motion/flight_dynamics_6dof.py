import numpy as np
from typing import Dict, Any

class FlightDynamics6DOF:
    """
    6-DOF flight dynamics model for an octagonal airframe.
    Models translational and rotational motion, including unique inertia properties.
    """
    def __init__(self,
                 mass_kg: float,
                 inertia_tensor: np.ndarray,  # 3x3 matrix
                 airframe_dims_m: Dict[str, float],  # {'width': ..., 'height': ..., ...}
                 Cd: float = 0.6,
                 Cl: float = 0.0,
                 area_m2: float = 1.0):
        self.mass_kg = mass_kg
        self.inertia_tensor = inertia_tensor
        self.airframe_dims_m = airframe_dims_m
        self.Cd = Cd
        self.Cl = Cl
        self.area_m2 = area_m2
        # State: position, velocity, orientation (quaternion), angular velocity
        self.state = {
            'pos_m': np.zeros(3),
            'vel_mps': np.zeros(3),
            'quat': np.array([1.0, 0.0, 0.0, 0.0]),
            'omega_radps': np.zeros(3)
        }

    def set_state(self, pos_m, vel_mps, quat, omega_radps):
        self.state['pos_m'] = np.array(pos_m)
        self.state['vel_mps'] = np.array(vel_mps)
        self.state['quat'] = np.array(quat)
        self.state['omega_radps'] = np.array(omega_radps)

    def get_state(self) -> Dict[str, Any]:
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.state.items()}

    def step(self, dt: float, force_N: np.ndarray, moment_Nm: np.ndarray, wind_mps: np.ndarray = None) -> Dict[str, Any]:
        # Unpack state
        pos = self.state['pos_m']
        vel = self.state['vel_mps']
        quat = self.state['quat']
        omega = self.state['omega_radps']
        # Aerodynamic drag
        rel_vel = vel - (wind_mps if wind_mps is not None else 0.0)
        v_mag = np.linalg.norm(rel_vel)
        drag = -0.5 * self.Cd * self.area_m2 * v_mag * rel_vel
        # Net force
        net_force = force_N + drag
        # Translational motion
        acc = net_force / self.mass_kg
        vel_new = vel + acc * dt
        pos_new = pos + vel * dt + 0.5 * acc * dt ** 2
        # Rotational motion
        I = self.inertia_tensor
        I_inv = np.linalg.inv(I)
        domega = I_inv @ (moment_Nm - np.cross(omega, I @ omega))
        omega_new = omega + domega * dt
        # Quaternion integration (simple Euler)
        dq = 0.5 * self._quat_mult(quat, np.concatenate([[0], omega]))
        quat_new = quat + dq * dt
        quat_new /= np.linalg.norm(quat_new)
        # Update state
        self.state['pos_m'] = pos_new
        self.state['vel_mps'] = vel_new
        self.state['quat'] = quat_new
        self.state['omega_radps'] = omega_new
        return self.get_state()

    def _quat_mult(self, q, r):
        # Hamilton product
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = r
        return np.array([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1
        ])
