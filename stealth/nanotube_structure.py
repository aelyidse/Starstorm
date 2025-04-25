from typing import Dict, List, Tuple
import numpy as np

class CarbonNanotubeStructure:
    """Simulates structural properties of carbon nanotube composites for stealth applications."""
    
    def __init__(self, surface_area: float = 10.0, volume_fraction: float = 0.15):
        self.surface_area = surface_area  # in square meters
        self.volume_fraction = volume_fraction  # CNT volume fraction in composite
        self.youngs_modulus_cnt = 1000.0  # GPa (typical for CNT)
        self.youngs_modulus_matrix = 3.0  # GPa (typical for polymer matrix)
        self.tensile_strength_cnt = 60.0  # GPa (typical for CNT)
        self.density_cnt = 1.4  # g/cm^3
        self.density_matrix = 1.2  # g/cm^3
        self.thermal_conductivity_cnt = 3000.0  # W/mK (along axis)
        self.em_absorption_factor = 0.85  # High EM absorption for stealth
        self.current_stress: float = 0.0  # MPa
        self.current_strain: float = 0.0
        self.composite_properties: Dict[str, float] = self._calculate_composite_properties()

    def _calculate_composite_properties(self) -> Dict[str, float]:
        """Calculate effective properties of CNT composite using rule of mixtures."""
        vf = self.volume_fraction
        vm = 1.0 - vf
        
        # Effective Young's Modulus (Halpin-Tsai model approximation)
        modulus_longitudinal = (vf * self.youngs_modulus_cnt + 
                               vm * self.youngs_modulus_matrix)
        modulus_transverse = self.youngs_modulus_matrix * (
            (1 + 2 * vf * (self.youngs_modulus_cnt / self.youngs_modulus_matrix - 1)) /
            (1 - vf * (self.youngs_modulus_cnt / self.youngs_modulus_matrix - 1))
        )
        
        # Effective density
        density = vf * self.density_cnt + vm * self.density_matrix
        
        # Effective thermal conductivity (simplified)
        thermal_cond = vf * self.thermal_conductivity_cnt + vm * 0.2  # Matrix has low conductivity
        
        return {
            "youngs_modulus_long": modulus_longitudinal,
            "youngs_modulus_trans": modulus_transverse,
            "tensile_strength": vf * self.tensile_strength_cnt,
            "density": density,
            "thermal_conductivity": thermal_cond
        }

    def apply_stress(self, stress_magnitude: float, direction: str = "longitudinal") -> float:
        """Apply stress to the composite and calculate resulting strain."""
        self.current_stress = stress_magnitude  # in MPa
        modulus = (self.composite_properties["youngs_modulus_long"] * 1000 if direction == "longitudinal"
                  else self.composite_properties["youngs_modulus_trans"] * 1000)  # Convert to MPa
        self.current_strain = stress_magnitude / modulus if modulus > 0 else 0.0
        return self.current_strain

    def calculate_failure_point(self, safety_margin: float = 1.5) -> Dict[str, float]:
        """Calculate failure stress and strain with safety margin."""
        max_stress = self.composite_properties["tensile_strength"] * 1000 / safety_margin  # MPa
        max_strain_long = max_stress / (self.composite_properties["youngs_modulus_long"] * 1000)
        max_strain_trans = max_stress / (self.composite_properties["youngs_modulus_trans"] * 1000)
        return {
            "max_stress": max_stress,
            "max_strain_longitudinal": max_strain_long,
            "max_strain_transverse": max_strain_trans,
            "safety_factor": safety_margin
        }

    def thermal_stress_analysis(self, temp_change: float) -> float:
        """Calculate thermal stress from temperature change."""
        # CNT has very low thermal expansion coefficient (~1e-6 /K)
        thermal_expansion_coeff = 1e-6 * self.volume_fraction + 2e-5 * (1 - self.volume_fraction)
        modulus = self.composite_properties["youngs_modulus_long"] * 1000  # MPa
        thermal_stress = thermal_expansion_coeff * temp_change * modulus
        self.current_stress += thermal_stress
        return thermal_stress

    def em_absorption_efficiency(self, frequency: float) -> float:
        """Calculate electromagnetic absorption efficiency at given frequency."""
        # Enhanced absorption at higher frequencies (simplified model)
        freq_factor = min(1.0, frequency / 30.0)  # Peaks at 30GHz and above
        efficiency = self.em_absorption_factor * freq_factor
        return min(efficiency, 1.0)

    def reinforce_aerogel(self, aerogel_strength: float) -> Dict[str, float]:
        """Calculate enhanced strength when reinforcing aerogel insulation."""
        combined_compressive = aerogel_strength + (self.composite_properties["tensile_strength"] * 0.5)
        return {
            "enhanced_compressive_strength": combined_compressive,
            "cnt_contribution": self.composite_properties["tensile_strength"] * 0.5
        }

    def get_structural_status(self) -> Dict[str, float]:
        """Return current structural status of the CNT composite."""
        return {
            "current_stress": self.current_stress,
            "current_strain": self.current_strain,
            "youngs_modulus_long": self.composite_properties["youngs_modulus_long"],
            "youngs_modulus_trans": self.composite_properties["youngs_modulus_trans"],
            "tensile_strength": self.composite_properties["tensile_strength"],
            "density": self.composite_properties["density"]
        }

# Example usage
if __name__ == "__main__":
    cnt_composite = CarbonNanotubeStructure(surface_area=10.0, volume_fraction=0.15)
    
    # Display initial properties
    print("Initial Composite Properties:")
    for key, value in cnt_composite.composite_properties.items():
        print(f"  {key}: {value:.2f}")
    
    # Apply mechanical stress
    strain = cnt_composite.apply_stress(500.0, direction="longitudinal")  # 500 MPa
    print(f"\nApplied Stress: 500 MPa")
    print(f"Resulting Strain (longitudinal): {strain:.6f}")
    
    # Calculate failure point
    failure = cnt_composite.calculate_failure_point(safety_margin=1.5)
    print("\nFailure Point Analysis:")
    for key, value in failure.items():
        print(f"  {key}: {value:.2f}")
    
    # Calculate thermal stress
    thermal_stress = cnt_composite.thermal_stress_analysis(temp_change=100.0)  # 100K change
    print(f"\nThermal Stress from 100K change: {thermal_stress:.2f} MPa")
    
    # EM absorption
    freqs = [10.0, 30.0, 100.0]  # GHz
    print("\nEM Absorption Efficiency:")
    for freq in freqs:
        eff = cnt_composite.em_absorption_efficiency(freq)
        print(f"  {freq} GHz: {eff*100:.1f}%")
    
    # Reinforce aerogel
    aero_reinforce = cnt_composite.reinforce_aerogel(aerogel_strength=2.5)
    print("\nAerogel Reinforcement:")
    for key, value in aero_reinforce.items():
        print(f"  {key}: {value:.2f}")
    
    # Current status
    status = cnt_composite.get_structural_status()
    print("\nCurrent Structural Status:")
    for key, value in status.items():
        print(f"  {key}: {value:.2f}")
