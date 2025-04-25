import math
from typing import Dict, Any, Optional, Tuple
from enum import Enum

class AtmosphericLayer(Enum):
    """Atmospheric layers by altitude range"""
    TROPOSPHERE = "Troposphere"
    STRATOSPHERE = "Stratosphere"
    MESOSPHERE = "Mesosphere"
    THERMOSPHERE = "Thermosphere"
    EXOSPHERE = "Exosphere"
    SPACE = "Space"

class AtmosphericModel:
    """
    Models Earth's atmospheric properties at different altitudes.
    Provides temperature, pressure, and density calculations.
    """
    
    # Standard atmospheric constants
    SEA_LEVEL_PRESSURE = 101325.0  # Pa
    SEA_LEVEL_TEMPERATURE = 288.15  # K
    SEA_LEVEL_DENSITY = 1.225  # kg/m³
    EARTH_RADIUS = 6371000.0  # m
    
    # Layer boundaries (meters)
    LAYER_BOUNDARIES = {
        AtmosphericLayer.TROPOSPHERE: (0, 11000),
        AtmosphericLayer.STRATOSPHERE: (11000, 50000),
        AtmosphericLayer.MESOSPHERE: (50000, 85000),
        AtmosphericLayer.THERMOSPHERE: (85000, 500000),
        AtmosphericLayer.EXOSPHERE: (500000, 10000000),
        AtmosphericLayer.SPACE: (10000000, float('inf'))
    }
    
    def __init__(self, altitude_m: float):
        self.altitude_m = altitude_m
        
    def get_layer(self) -> AtmosphericLayer:
        """Determine the atmospheric layer at the current altitude"""
        for layer, (lower, upper) in self.LAYER_BOUNDARIES.items():
            if lower <= self.altitude_m < upper:
                return layer
        return AtmosphericLayer.SPACE
    
    def temperature_K(self) -> float:
        """Calculate atmospheric temperature in Kelvin"""
        layer = self.get_layer()
        
        if layer == AtmosphericLayer.TROPOSPHERE:
            # Temperature decreases linearly in troposphere
            return self.SEA_LEVEL_TEMPERATURE - 0.0065 * self.altitude_m
        elif layer == AtmosphericLayer.STRATOSPHERE:
            # Temperature increases in stratosphere
            return 216.65 + 0.001 * (self.altitude_m - 11000)
        elif layer == AtmosphericLayer.MESOSPHERE:
            # Temperature decreases in mesosphere
            return 282.65 - 0.0028 * (self.altitude_m - 50000)
        elif layer == AtmosphericLayer.THERMOSPHERE:
            # Temperature increases dramatically in thermosphere
            base_temp = 165.65
            temp_increase = min(1000, 0.025 * (self.altitude_m - 85000))
            return base_temp + temp_increase
        else:
            # Exosphere and beyond
            return 1000.0  # Approximate
    
    def pressure_Pa(self) -> float:
        """Calculate atmospheric pressure in Pascals"""
        layer = self.get_layer()
        
        if layer == AtmosphericLayer.TROPOSPHERE:
            # Barometric formula
            T0 = self.SEA_LEVEL_TEMPERATURE
            T = self.temperature_K()
            return self.SEA_LEVEL_PRESSURE * (T / T0) ** (-5.255877)
        elif layer == AtmosphericLayer.STRATOSPHERE:
            # Simplified model for stratosphere
            return 22632.1 * math.exp(-0.000157 * (self.altitude_m - 11000))
        elif layer == AtmosphericLayer.MESOSPHERE:
            # Simplified model for mesosphere
            return 868.0 * math.exp(-0.000116 * (self.altitude_m - 50000))
        elif layer == AtmosphericLayer.THERMOSPHERE:
            # Exponential decay
            return 0.373 * math.exp(-0.000025 * (self.altitude_m - 85000))
        else:
            # Exosphere and beyond
            return 1e-10 * math.exp(-(self.altitude_m - 500000) / 100000)
    
    def density_kg_m3(self) -> float:
        """Calculate atmospheric density in kg/m³"""
        # Ideal gas law: ρ = P/(R*T)
        R = 287.05  # Specific gas constant for dry air, J/(kg·K)
        return self.pressure_Pa() / (R * self.temperature_K())
    
    def describe(self) -> Dict[str, Any]:
        """Return a dictionary of atmospheric properties"""
        return {
            'altitude_m': self.altitude_m,
            'layer': self.get_layer().value,
            'temperature_K': self.temperature_K(),
            'pressure_Pa': self.pressure_Pa(),
            'density_kg_m3': self.density_kg_m3()
        }