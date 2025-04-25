from typing import Dict, Any, List, Optional
import numpy as np

class PropellantUsagePredictor:
    """
    Predicts future propellant usage and recommends optimal burn profiles.
    Supports basic linear prediction, moving average, and polynomial fit.
    """
    def __init__(self, history_depth: int = 50):
        self.history_depth = history_depth
        self.usage_history: List[float] = []  # e.g., fuel usage rate (kg/s)
        self.time_history: List[float] = []

    def record(self, usage_rate: float, timestamp: float):
        self.usage_history.append(usage_rate)
        self.time_history.append(timestamp)
        if len(self.usage_history) > self.history_depth:
            self.usage_history.pop(0)
            self.time_history.pop(0)

    def predict(self, future_t: float = 10.0, method: str = 'linear') -> float:
        # Predict total usage over future_t seconds
        if not self.usage_history or len(self.usage_history) < 2:
            return 0.0
        times = np.array(self.time_history)
        usages = np.array(self.usage_history)
        dt = times[-1] - times[0] if times[-1] != times[0] else 1.0
        if method == 'linear':
            # Linear extrapolation
            avg_rate = np.mean(usages)
            return avg_rate * future_t
        elif method == 'moving_avg':
            avg_rate = np.mean(usages[-min(10, len(usages)):])
            return avg_rate * future_t
        elif method == 'polyfit':
            # Fit a 2nd degree polynomial
            coeffs = np.polyfit(times - times[0], usages, 2)
            poly = np.poly1d(coeffs)
            t0 = times[-1] - times[0]
            t1 = t0 + future_t
            # Integrate polynomial over [t0, t1]
            integral = poly.integ()
            return float(integral(t1) - integral(t0))
        else:
            raise ValueError(f"Unknown prediction method: {method}")

class PropellantOptimizer:
    """
    Recommends optimal burn profiles for propellant usage minimization.
    Supports constant, step, and custom burn strategies.
    """
    def __init__(self):
        pass

    def optimize(self, mission_time: float, propellant_mass_kg: float, min_thrust_N: float, max_thrust_N: float, isp_s: float, profile: str = 'constant') -> Dict[str, Any]:
        g0 = 9.80665
        if profile == 'constant':
            # Evenly spread usage
            avg_thrust = (min_thrust_N + max_thrust_N) / 2
            mdot = avg_thrust / (isp_s * g0)
            total_used = mdot * mission_time
            return {
                'recommended_thrust_N': avg_thrust,
                'expected_propellant_used_kg': min(total_used, propellant_mass_kg),
                'profile': 'constant'
            }
        elif profile == 'step':
            # Step profile: high thrust at start, then low
            t1 = mission_time * 0.3
            t2 = mission_time - t1
            mdot1 = max_thrust_N / (isp_s * g0)
            mdot2 = min_thrust_N / (isp_s * g0)
            used1 = mdot1 * t1
            used2 = mdot2 * t2
            total_used = used1 + used2
            return {
                'recommended_thrust_profile': [(max_thrust_N, t1), (min_thrust_N, t2)],
                'expected_propellant_used_kg': min(total_used, propellant_mass_kg),
                'profile': 'step'
            }
        else:
            raise ValueError(f"Unknown optimization profile: {profile}")
