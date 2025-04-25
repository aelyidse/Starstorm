from typing import Dict, Any
import math

class RocketFuelTank:
    """
    Models a cryogenic or storable rocket fuel tank.
    Supports ullage, boiloff, and pressurization effects.
    """
    def __init__(self, volume_L: float, fuel_density_kg_m3: float, max_pressure_bar: float, boiloff_rate_kg_per_hr: float = 0.0, temperature_K: float = 90.0):
        self.volume_L = volume_L
        self.fuel_density_kg_m3 = fuel_density_kg_m3
        self.max_pressure_bar = max_pressure_bar
        self.boiloff_rate_kg_per_hr = boiloff_rate_kg_per_hr
        self.temperature_K = temperature_K
        self.fuel_mass_kg = 0.0
        self.pressure_bar = 1.0

    def load(self, mass_kg: float):
        max_mass = self.get_max_mass()
        self.fuel_mass_kg = min(mass_kg, max_mass)
        self.pressure_bar = min(self.max_pressure_bar, 1.0 + self.fuel_mass_kg / max_mass * (self.max_pressure_bar - 1.0))

    def get_max_mass(self) -> float:
        return self.volume_L / 1000.0 * self.fuel_density_kg_m3

    def withdraw(self, mass_kg: float) -> float:
        withdrawn = min(self.fuel_mass_kg, mass_kg)
        self.fuel_mass_kg -= withdrawn
        return withdrawn

    def update_thermal(self, dt: float) -> float:
        # dt in seconds, boiloff in kg
        boiloff_kg = self.boiloff_rate_kg_per_hr * (dt / 3600.0)
        self.fuel_mass_kg = max(0.0, self.fuel_mass_kg - boiloff_kg)
        return boiloff_kg

    def get_state(self) -> Dict[str, Any]:
        return {
            'fuel_mass_kg': self.fuel_mass_kg,
            'pressure_bar': self.pressure_bar,
            'temperature_K': self.temperature_K
        }

class RocketFuelManagementSystem:
    """
    Simulates consumption and thermal management for a rocket engine's fuel and oxidizer tanks.
    """
    def __init__(self, fuel_tank: RocketFuelTank, oxidizer_tank: RocketFuelTank):
        self.fuel_tank = fuel_tank
        self.oxidizer_tank = oxidizer_tank

    def consume(self, fuel_kg: float, oxidizer_kg: float) -> Dict[str, float]:
        actual_fuel = self.fuel_tank.withdraw(fuel_kg)
        actual_oxidizer = self.oxidizer_tank.withdraw(oxidizer_kg)
        return {'fuel_consumed_kg': actual_fuel, 'oxidizer_consumed_kg': actual_oxidizer}

    def update_thermal(self, dt: float) -> Dict[str, float]:
        boiloff_fuel = self.fuel_tank.update_thermal(dt)
        boiloff_oxidizer = self.oxidizer_tank.update_thermal(dt)
        return {'fuel_boiloff_kg': boiloff_fuel, 'oxidizer_boiloff_kg': boiloff_oxidizer}

    def get_state(self) -> Dict[str, Any]:
        return {
            'fuel_tank': self.fuel_tank.get_state(),
            'oxidizer_tank': self.oxidizer_tank.get_state()
        }
