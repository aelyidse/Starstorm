import numpy as np
from typing import Dict, List, Optional, Tuple

class MetamaterialProperties:
    """Class to store and manage metamaterial properties for electromagnetic simulation."""
    
    def __init__(self, permittivity: float, permeability: float, 
                 refractive_index: float, frequency_range: Tuple[float, float]):
        self.permittivity = permittivity
        self.permeability = permeability
        self.refractive_index = refractive_index
        self.frequency_range = frequency_range  # in GHz (min, max)

class MetamaterialLayer:
    """Represents a single layer of metamaterial with specific electromagnetic properties."""
    
    def __init__(self, name: str, thickness: float, properties: MetamaterialProperties):
        self.name = name
        self.thickness = thickness  # in mm
        self.properties = properties

class MetamaterialSimulator:
    """Simulates electromagnetic behavior of metamaterial structures for stealth applications."""
    
    def __init__(self):
        self.layers: List[MetamaterialLayer] = []
        self.frequency_response: Dict[float, float] = {}

    def add_layer(self, layer: MetamaterialLayer) -> None:
        """Add a metamaterial layer to the simulation stack."""
        self.layers.append(layer)
        self._update_frequency_response()

    def _update_frequency_response(self) -> None:
        """Update the frequency response based on current layer stack."""
        # Simulate frequency response across the operational spectrum
        frequencies = np.linspace(2.0, 300.0, 100)  # GHz range
        for freq in frequencies:
            absorption = self._calculate_absorption(freq)
            self.frequency_response[freq] = absorption

    def _calculate_absorption(self, frequency: float) -> float:
        """Calculate absorption coefficient for a given frequency."""
        total_absorption = 0.0
        for layer in self.layers:
            if layer.properties.frequency_range[0] <= frequency <= layer.properties.frequency_range[1]:
                # Simplified absorption model based on negative refractive index
                absorption = (abs(layer.properties.refractive_index) * layer.thickness * 
                            np.exp(-frequency / 100.0))
                total_absorption += absorption
        return min(total_absorption, 1.0)  # Cap at 100% absorption

    def calculate_radar_cross_section(self, frequency: float, 
                                    incident_angle: float = 0.0) -> float:
        """Calculate RCS reduction based on metamaterial absorption."""
        absorption = self.frequency_response.get(frequency, 0.0)
        # Adjust RCS based on incident angle (simplified model)
        angle_factor = np.cos(np.radians(incident_angle))
        rcs_reduction = absorption * angle_factor
        return rcs_reduction

    def simulate_cloaking_efficiency(self, frequency: float) -> float:
        """Simulate cloaking efficiency at a given frequency."""
        absorption = self.frequency_response.get(frequency, 0.0)
        # Efficiency is a function of absorption and negative refraction
        efficiency = absorption * 0.8  # Simplified model
        for layer in self.layers:
            if layer.properties.refractive_index < 0:
                efficiency += 0.1  # Bonus for negative refractive index
        return min(efficiency, 1.0)

    def configure_split_ring_resonator_layer(self, ring_diameter: float = 6.0, 
                                           gap_width: float = 0.2, 
                                           thickness: float = 0.1) -> MetamaterialLayer:
        """Configure a split-ring resonator layer as specified in Starstorm.rtf."""
        props = MetamaterialProperties(
            permittivity=-2.5,
            permeability=-1.8,
            refractive_index=-2.0,
            frequency_range=(2.0, 18.0)  # X, Ku, K bands
        )
        return MetamaterialLayer(
            name="Split-Ring Resonator",
            thickness=thickness,
            properties=props
        )

    def configure_frequency_selective_surface(self, cell_size: float = 3.5, 
                                            thickness: float = 0.2) -> MetamaterialLayer:
        """Configure a frequency-selective surface layer."""
        props = MetamaterialProperties(
            permittivity=-1.8,
            permeability=-1.2,
            refractive_index=-1.5,
            frequency_range=(2.0, 18.0)
        )
        return MetamaterialLayer(
            name="Frequency Selective Surface",
            thickness=thickness,
            properties=props
        )

    def configure_nanotube_composite(self, thickness: float = 0.3) -> MetamaterialLayer:
        """Configure a carbon nanotube composite layer for millimeter wave absorption."""
        props = MetamaterialProperties(
            permittivity=-1.2,
            permeability=-0.8,
            refractive_index=-1.0,
            frequency_range=(30.0, 300.0)  # Millimeter wave spectrum
        )
        return MetamaterialLayer(
            name="Nanotube Composite",
            thickness=thickness,
            properties=props
        )

# Example usage
if __name__ == "__main__":
    simulator = MetamaterialSimulator()
    
    # Add layers as specified in Starstorm.rtf
    simulator.add_layer(simulator.configure_split_ring_resonator_layer())
    simulator.add_layer(simulator.configure_frequency_selective_surface())
    simulator.add_layer(simulator.configure_nanotube_composite())
    
    # Test simulation at different frequencies
    test_frequencies = [2.0, 10.0, 35.0, 100.0]  # GHz
    for freq in test_frequencies:
        rcs = simulator.calculate_radar_cross_section(freq)
        cloak_eff = simulator.simulate_cloaking_efficiency(freq)
        print(f"Frequency: {freq} GHz")
        print(f"RCS Reduction: {rcs*100:.1f}%")
        print(f"Cloaking Efficiency: {cloak_eff*100:.1f}%")
        print("---")
