import math
from typing import Dict, Any

class AtmosphericLayer:
    def __init__(self, name: str, base_altitude: float, top_altitude: float, base_temp: float, lapse_rate: float, base_pressure: float, base_density: float):
        self.name = name
        self.base_altitude = base_altitude  # meters
        self.top_altitude = top_altitude    # meters
        self.base_temp = base_temp          # Kelvin
        self.lapse_rate = lapse_rate        # K/m
        self.base_pressure = base_pressure  # Pascals
        self.base_density = base_density    # kg/m^3

class AtmosphericModel:
    """
    Implements a multi-layer atmospheric model from ground level to the thermosphere (~600km),
    based on the U.S. Standard Atmosphere 1976 and empirical thermospheric extensions.
    """
    def __init__(self):
        # Define atmospheric layers: [name, base_altitude(m), top_altitude(m), base_temp(K), lapse_rate(K/m), base_pressure(Pa), base_density(kg/m^3)]
        self.layers = [
            AtmosphericLayer("Troposphere", 0, 11000, 288.15, -0.0065, 101325, 1.225),
            AtmosphericLayer("Stratosphere1", 11000, 20000, 216.65, 0.0, 22632.1, 0.36391),
            AtmosphericLayer("Stratosphere2", 20000, 32000, 216.65, 0.001, 5474.89, 0.08803),
            AtmosphericLayer("Stratosphere3", 32000, 47000, 228.65, 0.0028, 868.019, 0.01322),
            AtmosphericLayer("Mesosphere1", 47000, 51000, 270.65, 0.0, 110.906, 0.00143),
            AtmosphericLayer("Mesosphere2", 51000, 71000, 270.65, -0.0028, 66.9389, 0.00086),
            AtmosphericLayer("Mesosphere3", 71000, 84852, 214.65, -0.002, 3.95642, 0.000064),
            AtmosphericLayer("Thermosphere1", 84852, 120000, 186.87, 0.004, 0.3734, 1e-6),
            AtmosphericLayer("Thermosphere2", 120000, 600000, 300.0, 0.002, 1e-4, 1e-10),
        ]
        self.R = 287.05  # J/(kg*K)
        self.g0 = 9.80665  # m/s^2

    def get_layer(self, altitude_m: float) -> AtmosphericLayer:
        for layer in self.layers:
            if layer.base_altitude <= altitude_m < layer.top_altitude:
                return layer
        return self.layers[-1]  # Thermosphere2 for >600km

    def temperature(self, altitude_m: float) -> float:
        layer = self.get_layer(altitude_m)
        delta_h = altitude_m - layer.base_altitude
        T = layer.base_temp + layer.lapse_rate * delta_h
        return T

    def pressure(self, altitude_m: float) -> float:
        layer = self.get_layer(altitude_m)
        T = self.temperature(altitude_m)
        if layer.lapse_rate == 0:
            # Isothermal layer
            p = layer.base_pressure * math.exp(-self.g0 * (altitude_m - layer.base_altitude) / (self.R * layer.base_temp))
        else:
            # Gradient layer
            p = layer.base_pressure * (T / layer.base_temp) ** (-self.g0 / (self.R * layer.lapse_rate))
        return p

    def density(self, altitude_m: float) -> float:
        p = self.pressure(altitude_m)
        T = self.temperature(altitude_m)
        rho = p / (self.R * T)
        return rho

    def describe(self, altitude_m: float) -> Dict[str, Any]:
        layer = self.get_layer(altitude_m)
        return {
            'layer': layer.name,
            'altitude_m': altitude_m,
            'temperature_K': self.temperature(altitude_m),
            'pressure_Pa': self.pressure(altitude_m),
            'density_kgm3': self.density(altitude_m),
        }
