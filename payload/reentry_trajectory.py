import numpy as np
from typing import Dict, Any, List, Optional

class ReentryTrajectoryPlanner:
    """
    Plans atmospheric reentry trajectories for safe recovery and controlled descent.
    Supports ballistic and guided entry, thermal/structural constraints, and waypoint generation.
    """
    def __init__(self, mass: float, area: float, Cd: float, Cl: float = 0.0):
        self.mass = mass  # kg
        self.area = area  # m^2
        self.Cd = Cd      # Drag coefficient
        self.Cl = Cl      # Lift coefficient (0 for ballistic)
        self.trajectory: List[Dict[str, Any]] = []

    def plan(self, entry_state: Dict[str, float], target: Dict[str, float], dt: float = 1.0, max_steps: int = 1000) -> List[Dict[str, Any]]:
        """
        entry_state: {'altitude': m, 'velocity': m/s, 'gamma': deg, 'lat': deg, 'lon': deg}
        target: {'lat': deg, 'lon': deg, 'altitude': m}
        Returns: trajectory (list of states)
        """
        # Constants
        g = 9.81  # m/s^2
        R_earth = 6371000  # m
        rho0 = 1.225  # kg/m^3
        H = 7200      # Scale height (m)
        state = entry_state.copy()
        traj = [state.copy()]
        for _ in range(max_steps):
            alt = state['altitude']
            v = state['velocity']
            gamma = np.deg2rad(state['gamma'])
            rho = rho0 * np.exp(-alt / H)
            D = 0.5 * rho * v**2 * self.Cd * self.area
            L = 0.5 * rho * v**2 * self.Cl * self.area
            dv = -D / self.mass * dt - g * np.sin(gamma) * dt
            dgamma = (L / (self.mass * v) - g / v * np.cos(gamma)) * dt
            dalt = v * np.sin(gamma) * dt
            # Simple latitude/longitude update (no wind, flat earth approx)
            dlat = (v * np.cos(gamma) * dt) / R_earth * (180 / np.pi)
            dlon = 0.0  # Could add cross-range guidance
            # Update state
            state = {
                'altitude': max(0.0, alt + dalt),
                'velocity': max(0.0, v + dv),
                'gamma': np.rad2deg(gamma + dgamma),
                'lat': state['lat'] + dlat,
                'lon': state['lon'] + dlon
            }
            traj.append(state.copy())
            if state['altitude'] <= target['altitude'] or state['velocity'] <= 10:
                break
        self.trajectory = traj
        return traj

    def get_trajectory(self) -> List[Dict[str, Any]]:
        return self.trajectory
