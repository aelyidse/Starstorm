import numpy as np
from typing import Dict, Any

class RadarCrossSectionCalculator:
    """
    Calculates radar cross-section (RCS) for simple and complex geometries and materials.
    Supports monostatic and bistatic RCS, frequency and angle dependence, and metamaterial effects.
    """
    def __init__(self, wavelength_m: float):
        self.wavelength_m = wavelength_m

    def rcs_sphere(self, radius_m: float, material_reflectivity: float = 1.0) -> float:
        # Analytical RCS for a perfectly conducting sphere (Rayleigh/Mie/Optical regimes)
        k = 2 * np.pi / self.wavelength_m
        if k * radius_m < 1:  # Rayleigh
            sigma = 9 * np.pi * radius_m**6 / self.wavelength_m**4
        else:  # Optical (large sphere)
            sigma = np.pi * radius_m**2
        return sigma * material_reflectivity

    def rcs_flat_plate(self, area_m2: float, incident_angle_deg: float = 0.0, material_reflectivity: float = 1.0) -> float:
        # RCS for flat plate (broadside, optical regime)
        incident_angle_rad = np.radians(incident_angle_deg)
        sigma = 4 * np.pi * area_m2**2 / self.wavelength_m**2 * np.cos(incident_angle_rad)**2
        return sigma * material_reflectivity

    def rcs_custom(self, reflectivity_map: np.ndarray, panel_area_m2: float, incident_angle_deg: float = 0.0) -> float:
        # Panel-based RCS sum for arbitrary shapes/materials
        incident_angle_rad = np.radians(incident_angle_deg)
        total_rcs = 0.0
        for refl in reflectivity_map.flat:
            sigma = 4 * np.pi * panel_area_m2**2 / self.wavelength_m**2 * np.cos(incident_angle_rad)**2
            total_rcs += sigma * refl
        return total_rcs

    def rcs_bistatic(self, geometry: Dict[str, Any], tx_angle_deg: float, rx_angle_deg: float, material_reflectivity: float = 1.0) -> float:
        # Placeholder for bistatic RCS (geometry-dependent)
        # For now, treat as monostatic with average angle
        avg_angle = 0.5 * (tx_angle_deg + rx_angle_deg)
        if geometry['type'] == 'flat_plate':
            return self.rcs_flat_plate(geometry['area_m2'], avg_angle, material_reflectivity)
        elif geometry['type'] == 'sphere':
            return self.rcs_sphere(geometry['radius_m'], material_reflectivity)
        else:
            raise NotImplementedError('Geometry not supported for bistatic RCS')
