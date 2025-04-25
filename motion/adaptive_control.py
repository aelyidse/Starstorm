from typing import Dict, Any, Callable

class AdaptiveController:
    """
    Adaptive control law for flight in varying atmospheric densities.
    Adjusts gains and control allocation based on real-time density.
    """
    def __init__(self, base_gains: Dict[str, float], gain_schedule: Callable[[float], Dict[str, float]]):
        """
        base_gains: e.g., {'Kp': ..., 'Kd': ...}
        gain_schedule: function mapping density_kg_m3 -> gains dict
        """
        self.base_gains = base_gains
        self.gain_schedule = gain_schedule
        self.current_gains = base_gains.copy()

    def update_gains(self, density_kg_m3: float):
        self.current_gains = self.gain_schedule(density_kg_m3)

    def compute_control(self, error: Dict[str, float], error_dot: Dict[str, float]) -> Dict[str, float]:
        # Example: simple PD control for each axis
        u = {}
        for axis in error:
            Kp = self.current_gains.get(f'Kp_{axis}', self.current_gains.get('Kp', 1.0))
            Kd = self.current_gains.get(f'Kd_{axis}', self.current_gains.get('Kd', 0.0))
            u[axis] = Kp * error[axis] + Kd * error_dot.get(axis, 0.0)
        return u

# Example gain schedule

def example_gain_schedule(density_kg_m3: float) -> Dict[str, float]:
    # Gains decrease at lower density (less aerodynamic authority)
    base_Kp = 1.0
    base_Kd = 0.2
    scale = min(1.0, density_kg_m3 / 1.225)  # 1.225 kg/m^3 = sea level
    return {'Kp': base_Kp * scale, 'Kd': base_Kd * scale}
