import math
from typing import Dict, Any, Optional

class ThermosphericModel:
    """
    Models the physical properties of the thermosphere (80km - 600km altitude).
    Provides atmospheric density, temperature, and pressure as a function of altitude and solar activity.
    """
    def __init__(self, F10_7: float = 150.0, Ap: float = 15.0):
        # F10.7: Solar radio flux, Ap: geomagnetic index (space weather)
        self.F10_7 = F10_7
        self.Ap = Ap

    def temperature(self, altitude_m: float) -> float:
        # Empirical model: temperature increases with altitude in thermosphere
        if altitude_m < 80000:
            return 200.0  # K, lower boundary
        # Simple model: T = T0 + a * (h - h0)
        T0 = 200.0
        a = 0.004
        T = T0 + a * (altitude_m - 80000)
        # Solar activity effect
        T += 0.05 * (self.F10_7 - 70)
        return min(T, 2000.0)

    def density(self, altitude_m: float) -> float:
        # Exponential decrease, modulated by temperature
        T = self.temperature(altitude_m)
        h = altitude_m / 1000.0
        # U.S. Standard Atmosphere 1976 approximation
        if h < 86:
            return 0.0
        rho0 = 5.297e-7  # kg/m^3 @ 86km
        scale_height = 27.0  # km
        rho = rho0 * math.exp(-(h - 86) / scale_height) * (200.0 / T)
        return rho

    def pressure(self, altitude_m: float) -> float:
        # Ideal gas law: p = rho * R * T
        rho = self.density(altitude_m)
        T = self.temperature(altitude_m)
        R = 287.05  # J/(kg*K)
        return rho * R * T

class PhysicsEngine:
    """
    High-precision physics simulation engine for vehicle dynamics in the thermosphere.
    Models gravity, drag, thrust, and orbital mechanics.
    """
    def __init__(self, model: Optional[ThermosphericModel] = None):
        self.model = model or ThermosphericModel()
        self.G = 6.67430e-11  # m^3/kg/s^2
        self.M_earth = 5.972e24  # kg
        self.R_earth = 6371000.0  # m

    def gravity(self, altitude_m: float) -> float:
        r = self.R_earth + altitude_m
        return self.G * self.M_earth / (r ** 2)

    def drag_force(self, altitude_m: float, velocity_mps: float, area_m2: float, Cd: float = 2.2) -> float:
        rho = self.model.density(altitude_m)
        return 0.5 * rho * velocity_mps ** 2 * Cd * area_m2

    def update_state(self, state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Update vehicle state over dt seconds.
        state: {
            'altitude_m', 'velocity_mps', 'mass_kg', 'thrust_N', 'area_m2', 'Cd', 'orientation', 'external_forces'
        }
        """
        altitude = state['altitude_m']
        velocity = state['velocity_mps']
        mass = state['mass_kg']
        thrust = state['thrust_N']
        area = state['area_m2']
        Cd = state.get('Cd', 2.2)
        external_forces = state.get('external_forces', 0.0)

        g = self.gravity(altitude)
        drag = self.drag_force(altitude, velocity, area, Cd)
        net_force = thrust - drag - mass * g + external_forces
        acceleration = net_force / mass
        new_velocity = velocity + acceleration * dt
        new_altitude = altitude + new_velocity * dt
        return {
            **state,
            'altitude_m': new_altitude,
            'velocity_mps': new_velocity,
            'acceleration_mps2': acceleration,
            'drag_N': drag,
            'gravity_mps2': g
        }

    def orbital_velocity(self, altitude_m: float) -> float:
        r = self.R_earth + altitude_m
        return math.sqrt(self.G * self.M_earth / r)
