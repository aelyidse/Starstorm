import numpy as np
from typing import List, Tuple, Dict, Any

class ThrustAllocator:
    """
    Allocates thrust and gimbal angles for a rotatable engine configuration (e.g., octagonal airframe).
    Supports 2D/3D allocation, actuator limits, and optimal distribution for control commands.
    """
    def __init__(self, engine_positions: List[np.ndarray], max_thrusts: List[float], max_gimbal_deg: float = 10.0):
        self.engine_positions = engine_positions  # List of np.array([x, y, z])
        self.max_thrusts = max_thrusts
        self.max_gimbal_deg = max_gimbal_deg
        self.num_engines = len(engine_positions)

    def allocate(self, desired_force: np.ndarray, desired_moment: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts [{thrust_N, gimbal_deg, vector_N}], one per engine.
        Uses least-squares allocation with actuator constraints.
        """
        # Build allocation matrix
        A = []  # Rows: [Fx, Fy, Fz, Mx, My, Mz]
        for i, pos in enumerate(self.engine_positions):
            # Assume thrust vector starts along +X, will be rotated by gimbal
            # For allocation, use unit vectors for now
            thrust_dir = np.array([1.0, 0.0, 0.0])
            moment_arm = np.cross(pos, thrust_dir)
            A.append(np.concatenate([thrust_dir, moment_arm]))
        A = np.array(A).T  # shape (6, num_engines)
        # Desired wrench
        wrench = np.concatenate([desired_force, desired_moment])
        # Least-squares solution
        thrusts, residuals, rank, s = np.linalg.lstsq(A, wrench, rcond=None)
        # Apply thrust and gimbal limits
        allocation = []
        for i in range(self.num_engines):
            thrust = float(np.clip(thrusts[i], 0.0, self.max_thrusts[i]))
            # Compute required gimbal (for now, assume only pitch/yaw)
            if np.linalg.norm(desired_force) > 1e-6:
                gimbal_pitch = np.degrees(np.arctan2(desired_force[2], desired_force[0]))
                gimbal_yaw = np.degrees(np.arctan2(desired_force[1], desired_force[0]))
                gimbal_pitch = np.clip(gimbal_pitch, -self.max_gimbal_deg, self.max_gimbal_deg)
                gimbal_yaw = np.clip(gimbal_yaw, -self.max_gimbal_deg, self.max_gimbal_deg)
            else:
                gimbal_pitch = 0.0
                gimbal_yaw = 0.0
            # Compute thrust vector
            pitch_rad = np.radians(gimbal_pitch)
            yaw_rad = np.radians(gimbal_yaw)
            x = thrust * np.cos(pitch_rad) * np.cos(yaw_rad)
            y = thrust * np.sin(yaw_rad)
            z = -thrust * np.sin(pitch_rad) * np.cos(yaw_rad)
            allocation.append({
                'thrust_N': thrust,
                'gimbal_deg': (gimbal_pitch, gimbal_yaw),
                'vector_N': (x, y, z)
            })
        return allocation
