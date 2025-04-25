import math
from typing import Dict, Any

class IonThruster:
    """
    High-fidelity ion thruster physics model with realistic power/thrust curves.
    Models xenon propellant consumption, power draw, and variable efficiency.
    """
    def __init__(self,
                 max_power_W: float,
                 min_power_W: float,
                 efficiency_curve: Dict[float, float],  # {power_W: efficiency (0-1)}
                 isp_s: float,
                 max_thrust_N: float):
        self.max_power_W = max_power_W
        self.min_power_W = min_power_W
        self.efficiency_curve = efficiency_curve
        self.isp_s = isp_s
        self.max_thrust_N = max_thrust_N
        self.current_power_W = 0.0
        self.xenon_mass_kg = 0.0
        self.running = False

    def load_propellant(self, xenon_mass_kg: float):
        self.xenon_mass_kg = xenon_mass_kg

    def set_power(self, power_W: float):
        self.current_power_W = max(self.min_power_W, min(self.max_power_W, power_W))

    def start(self):
        self.running = True

    def shutdown(self):
        self.running = False
        self.current_power_W = 0.0

    def get_efficiency(self) -> float:
        # Interpolate efficiency curve
        keys = sorted(self.efficiency_curve.keys())
        if self.current_power_W <= keys[0]:
            return self.efficiency_curve[keys[0]]
        if self.current_power_W >= keys[-1]:
            return self.efficiency_curve[keys[-1]]
        for i in range(1, len(keys)):
            if self.current_power_W < keys[i]:
                k0, k1 = keys[i-1], keys[i]
                e0, e1 = self.efficiency_curve[k0], self.efficiency_curve[k1]
                return e0 + (e1 - e0) * (self.current_power_W - k0) / (k1 - k0)
        return 0.0

    def compute_thrust(self) -> float:
        if not self.running or self.xenon_mass_kg <= 0 or self.current_power_W <= 0:
            return 0.0
        eff = self.get_efficiency()
        thrust = self.max_thrust_N * (self.current_power_W / self.max_power_W) * eff
        return max(0.0, min(self.max_thrust_N, thrust))

    def xenon_consumption(self, dt: float) -> float:
        if not self.running or self.current_power_W <= 0:
            return 0.0
        g0 = 9.80665
        thrust = self.compute_thrust()
        mdot = thrust / (self.isp_s * g0)
        used = mdot * dt
        self.xenon_mass_kg = max(0.0, self.xenon_mass_kg - used)
        return used

    def update(self, dt: float) -> Dict[str, Any]:
        thrust = self.compute_thrust()
        xenon_used = self.xenon_consumption(dt)
        return {
            'thrust_N': thrust,
            'xenon_mass_kg': self.xenon_mass_kg,
            'xenon_used_kg': xenon_used,
            'power_W': self.current_power_W,
            'efficiency': self.get_efficiency()
        }
