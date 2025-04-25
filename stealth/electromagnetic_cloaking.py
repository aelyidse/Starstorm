from typing import Dict, List, Tuple
import numpy as np

class ElectromagneticProperties:
    """Class to store electromagnetic properties for cloaking simulation."""
    
    def __init__(self, permittivity: float, permeability: float, conductivity: float):
        self.permittivity = permittivity  # Relative permittivity (epsilon_r)
        self.permeability = permeability  # Relative permeability (mu_r)
        self.conductivity = conductivity  # Conductivity in S/m

class CloakingLayer:
    """Represents a single layer in the electromagnetic cloaking structure."""
    
    def __init__(self, thickness: float, properties: ElectromagneticProperties):
        self.thickness = thickness  # in meters
        self.properties = properties

class ElectromagneticCloakingSimulator:
    """Simulates advanced electromagnetic cloaking using transformation optics."""
    
    def __init__(self, radius: float = 1.0, num_layers: int = 10):
        self.radius = radius  # Radius of the cloaked object in meters
        self.num_layers = num_layers
        self.layers: List[CloakingLayer] = []
        self.initialize_layers()
        self.background_permittivity = 1.0  # Vacuum permittivity
        self.background_permeability = 1.0  # Vacuum permeability
        self.is_active: bool = True
        self.power_consumption: float = 0.0  # Simulated power in watts

    def initialize_layers(self) -> None:
        """Initialize the cloaking layers using transformation optics principles."""
        self.layers = []
        for i in range(self.num_layers):
            r_inner = self.radius * (i / self.num_layers)
            r_outer = self.radius * ((i + 1) / self.num_layers)
            thickness = r_outer - r_inner
            # Transformation optics: radially dependent permittivity and permeability
            # Simplified from Pendry's transformation optics for a spherical cloak
            factor = (self.radius / (self.radius - r_inner)) ** 3 if r_inner < self.radius else 1.0
            permittivity = 1.0 * factor
            permeability = 1.0 * factor
            conductivity = 0.01 * (i + 1)  # Increasing conductivity outward for loss
            props = ElectromagneticProperties(permittivity, permeability, conductivity)
            self.layers.append(CloakingLayer(thickness, props))

    def set_active(self, active: bool) -> None:
        """Set the operational status of the cloaking system."""
        self.is_active = active
        self.power_consumption = 1000.0 if active else 0.0  # Simulated power draw

    def set_background_medium(self, permittivity: float, permeability: float) -> None:
        """Set the background electromagnetic properties."""
        self.background_permittivity = permittivity
        self.background_permeability = permeability

    def calculate_reflection_coefficient(self, frequency: float, angle: float = 0.0) -> complex:
        """Calculate reflection coefficient for a given frequency and incidence angle."""
        if not self.is_active:
            return 1.0 + 0.0j  # Full reflection if cloak is off
        
        wavelength = 3e8 / frequency  # Speed of light / frequency
        k0 = 2 * np.pi / wavelength  # Wave number in vacuum
        
        # Simplified multilayer reflection calculation
        # Using transfer matrix method for normal incidence (angle=0 approximation)
        eta_0 = 376.73  # Impedance of free space
        reflection = 0.0 + 0.0j
        transmission = 1.0 + 0.0j
        
        for layer in reversed(self.layers):
            # Impedance of the layer (approximate for simplicity)
            n = np.sqrt(layer.properties.permeability / layer.properties.permittivity)
            eta = eta_0 * n
            k = k0 * np.sqrt(layer.properties.permittivity * layer.properties.permeability)
            d = layer.thickness
            # Transfer matrix elements (simplified)
            cos_kd = np.cos(k * d)
            sin_kd = np.sin(k * d)
            t11 = cos_kd
            t12 = 1j * eta * sin_kd
            t21 = 1j * sin_kd / eta
            t22 = cos_kd
            # Update reflection and transmission (matrix multiplication)
            r_new = (t11 * reflection + t12) / (t21 * reflection + t22)
            reflection = r_new
        
        return reflection

    def calculate_radar_cross_section(self, frequency: float, angles: List[float] = None) -> Dict[float, float]:
        """Calculate radar cross-section (RCS) reduction across specified angles."""
        if angles is None:
            angles = [0.0, 45.0, 90.0]  # Default angles in degrees
        rcs_values = {}
        uncloaked_rcs = np.pi * self.radius ** 2  # Simplified geometric RCS for sphere
        
        for angle in angles:
            reflection = self.calculate_reflection_coefficient(frequency, np.radians(angle))
            reflection_magnitude = np.abs(reflection)
            # RCS reduction proportional to reflection coefficient magnitude squared
            rcs = uncloaked_rcs * (reflection_magnitude ** 2)
            rcs_values[angle] = max(1e-6, rcs)  # Avoid zero for log scale plotting
        return rcs_values

    def calculate_cloaking_efficiency(self, frequency_range: Tuple[float, float], 
                                    num_points: int = 10) -> Dict[float, float]:
        """Calculate cloaking efficiency across a frequency range."""
        frequencies = np.linspace(frequency_range[0], frequency_range[1], num_points)
        efficiencies = {}
        for freq in frequencies:
            rcs_dict = self.calculate_radar_cross_section(freq)
            avg_rcs_reduction = np.mean(list(rcs_dict.values()))
            uncloaked_rcs = np.pi * self.radius ** 2
            # Efficiency as percentage reduction in RCS
            efficiency = max(0.0, 100.0 * (1.0 - avg_rcs_reduction / uncloaked_rcs))
            efficiencies[freq] = efficiency
        return efficiencies

    def adapt_to_threat(self, frequency: float, power_limit: float = 1500.0) -> float:
        """Adapt cloaking parameters to detected threat frequency within power constraints."""
        if not self.is_active:
            return 0.0
        # Simplified adaptation: adjust layer properties based on frequency
        wavelength = 3e8 / frequency
        scaling_factor = min(1.5, max(0.5, wavelength / self.radius))
        for layer in self.layers:
            layer.properties.permittivity *= scaling_factor
            layer.properties.permeability *= scaling_factor
        # Power consumption increases with adaptation
        adaptation_power = min(power_limit, 500.0 * scaling_factor)
        self.power_consumption = adaptation_power
        return adaptation_power

    def get_system_status(self) -> Dict[str, any]:
        """Return status and performance metrics of the cloaking system."""
        return {
            "active": self.is_active,
            "power_consumption": self.power_consumption,
            "number_of_layers": len(self.layers),
            "total_thickness": sum(layer.thickness for layer in self.layers),
            "background_permittivity": self.background_permittivity,
            "background_permeability": self.background_permeability
        }

# Example usage
if __name__ == "__main__":
    # Create an electromagnetic cloaking simulator
    cloak = ElectromagneticCloakingSimulator(radius=2.0, num_layers=15)
    
    # Test RCS reduction at different angles for X-band radar (10 GHz)
    rcs_values = cloak.calculate_radar_cross_section(frequency=10e9, angles=[0.0, 30.0, 60.0, 90.0])
    print("Radar Cross-Section (RCS) at 10 GHz:")
    for angle, rcs in rcs_values.items():
        print(f"  Angle {angle}°: {rcs:.2e} m²")
    
    # Test cloaking efficiency across a frequency range
    efficiency = cloak.calculate_cloaking_efficiency(frequency_range=(1e9, 20e9), num_points=5)
    print("\nCloaking Efficiency Across Frequencies:")
    for freq, eff in efficiency.items():
        print(f"  Frequency {freq/1e9:.1f} GHz: {eff:.1f}%")
    
    # Adapt to a specific threat frequency
    power_used = cloak.adapt_to_threat(frequency=15e9)
    print(f"\nPower used for adaptation to 15 GHz threat: {power_used:.1f} W")
    
    # Check system status
    status = cloak.get_system_status()
    print("\nSystem Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test with cloaking system off
    cloak.set_active(False)
    rcs_values_off = cloak.calculate_radar_cross_section(frequency=10e9, angles=[0.0])
    print(f"\nRCS at 10 GHz (cloak off, 0°): {rcs_values_off[0.0]:.2e} m²")
