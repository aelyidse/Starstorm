from typing import Dict, Any
import math

class XenonStorageTank:
    """
    Models a high-pressure xenon storage tank.
    Supports ideal gas law and pressure/temperature effects.
    """
    def __init__(self, volume_L: float, max_pressure_bar: float, temperature_K: float = 293.15):
        self.volume_L = volume_L
        self.max_pressure_bar = max_pressure_bar
        self.temperature_K = temperature_K
        self.xenon_mass_kg = 0.0
        self.R = 8.314462618  # J/(mol·K)
        self.M_xe = 131.293e-3  # kg/mol

    def load(self, mass_kg: float):
        self.xenon_mass_kg = min(mass_kg, self.get_max_mass())

    def get_pressure_bar(self) -> float:
        # Ideal gas law: P = nRT/V
        n = self.xenon_mass_kg / self.M_xe
        V_m3 = self.volume_L / 1000.0
        P_Pa = n * self.R * self.temperature_K / V_m3
        return P_Pa / 1e5  # Pa to bar

    def get_max_mass(self) -> float:
        # Max mass at max pressure
        V_m3 = self.volume_L / 1000.0
        n_max = self.max_pressure_bar * 1e5 * V_m3 / (self.R * self.temperature_K)
        return n_max * self.M_xe

    def withdraw(self, mass_kg: float) -> float:
        withdrawn = min(self.xenon_mass_kg, mass_kg)
        self.xenon_mass_kg -= withdrawn
        return withdrawn

class PressureRegulator:
    """
    Simulates a pressure regulator for xenon feed.
    Maintains set output pressure within regulator limits.
    """
    def __init__(self, set_pressure_bar: float, min_pressure_bar: float = 1.0, max_pressure_bar: float = 20.0):
        self.set_pressure_bar = set_pressure_bar
        self.min_pressure_bar = min_pressure_bar
        self.max_pressure_bar = max_pressure_bar

    def regulate(self, input_pressure_bar: float) -> float:
        if input_pressure_bar < self.set_pressure_bar:
            return max(input_pressure_bar, self.min_pressure_bar)
        return min(self.set_pressure_bar, self.max_pressure_bar)

class XenonFlowController:
    """
    Simulates mass flow control (MFC) for xenon delivery to thruster.
    Supports setpoint tracking and flow rate limits.
    """
    def __init__(self, max_flow_mg_s: float):
        self.max_flow_mg_s = max_flow_mg_s
        self.setpoint_mg_s = 0.0

    def set_flow(self, setpoint_mg_s: float):
        self.setpoint_mg_s = max(0.0, min(self.max_flow_mg_s, setpoint_mg_s))

    def deliver(self, dt: float) -> float:
        # Returns xenon mass delivered in kg
        mg = self.setpoint_mg_s * dt
        delivered_kg = mg / 1e6
        return delivered_kg
