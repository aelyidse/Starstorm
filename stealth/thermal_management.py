import numpy as np
from typing import Dict, Any, List

class HeatSource:
    """
    Models heat generation from vehicle subsystems (e.g., propulsion, avionics).
    """
    def __init__(self, name: str, power_W: float):
        self.name = name
        self.power_W = power_W

    def get_heat(self, dt: float) -> float:
        # Returns heat generated over dt seconds (Joules)
        return self.power_W * dt

class ThermalDissipation:
    """
    Models heat dissipation via conduction, convection, and radiation.
    """
    def __init__(self, area_m2: float, emissivity: float, h_conv: float = 0.0):
        self.area_m2 = area_m2
        self.emissivity = emissivity
        self.h_conv = h_conv  # Convective heat transfer coefficient (W/m^2/K)
        self.sigma = 5.670374419e-8  # Stefan-Boltzmann constant

    def radiative_loss(self, temp_K: float, env_temp_K: float = 293.15) -> float:
        # Net radiative loss (W)
        return self.emissivity * self.sigma * self.area_m2 * (temp_K**4 - env_temp_K**4)

    def convective_loss(self, temp_K: float, env_temp_K: float = 293.15) -> float:
        # Convective loss (W)
        return self.h_conv * self.area_m2 * (temp_K - env_temp_K)

    def total_loss(self, temp_K: float, env_temp_K: float = 293.15) -> float:
        return self.radiative_loss(temp_K, env_temp_K) + self.convective_loss(temp_K, env_temp_K)

class ThermalSystemSimulator:
    """
    Simulates heat generation, accumulation, and dissipation for the vehicle.
    Supports multiple heat sources and dissipation paths.
    """
    def __init__(self, mass_kg: float, c_p_J_kgK: float, dissipation: ThermalDissipation, initial_temp_K: float = 293.15):
        self.mass_kg = mass_kg
        self.c_p_J_kgK = c_p_J_kgK
        self.dissipation = dissipation
        self.temp_K = initial_temp_K
        self.heat_sources: List[HeatSource] = []

    def add_heat_source(self, source: HeatSource):
        self.heat_sources.append(source)

    def step(self, dt: float, env_temp_K: float = 293.15) -> Dict[str, Any]:
        # 1. Accumulate heat from all sources
        total_heat_J = sum(src.get_heat(dt) for src in self.heat_sources)
        # 2. Compute heat loss
        loss_W = self.dissipation.total_loss(self.temp_K, env_temp_K)
        loss_J = loss_W * dt
        # 3. Net heat change
        delta_Q = total_heat_J - loss_J
        delta_T = delta_Q / (self.mass_kg * self.c_p_J_kgK)
        self.temp_K += delta_T
        return {
            'temp_K': self.temp_K,
            'heat_generated_J': total_heat_J,
            'heat_lost_J': loss_J,
            'net_heat_J': delta_Q
        }
