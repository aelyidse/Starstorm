import math
from typing import Dict, Any, Tuple

class EMWavePropagation:
    """
    Models electromagnetic wave propagation for stealth system validation.
    Supports free-space, layered media, and surface interaction (reflection, absorption, transmission).
    """
    def __init__(self, frequency_Hz: float, incident_angle_deg: float = 0.0):
        self.frequency_Hz = frequency_Hz
        self.incident_angle_deg = incident_angle_deg
        self.c = 299792458.0  # m/s
        self.wavelength_m = self.c / frequency_Hz

    def free_space_loss(self, distance_m: float) -> float:
        # Friis transmission equation (dB)
        if distance_m == 0:
            return 0.0
        return 20 * math.log10(distance_m) + 20 * math.log10(self.frequency_Hz) - 147.55

    def reflection_coefficient(self, n1: complex, n2: complex) -> complex:
        # Normal incidence reflection (Fresnel)
        return (n1 - n2) / (n1 + n2)

    def transmission_coefficient(self, n1: complex, n2: complex) -> complex:
        # Normal incidence transmission (Fresnel)
        return 2 * n1 / (n1 + n2)

    def multilayer_reflectance(self, layers: Tuple[Tuple[complex, float], ...]) -> float:
        """
        layers: sequence of (complex index of refraction, thickness in meters), from incident to exit
        Returns total power reflectance (0-1)
        """
        r_total = 0.0
        t_total = 1.0
        n_prev = 1.0
        for (n, d) in layers:
            r = abs(self.reflection_coefficient(n_prev, n)) ** 2
            t = abs(self.transmission_coefficient(n_prev, n)) ** 2
            # Attenuation in the layer
            alpha = 4 * math.pi * n.imag / self.wavelength_m
            att = math.exp(-alpha * d)
            t_total *= t * att
            r_total += r * (1 - att)
            n_prev = n
        return r_total

    def radar_cross_section(self, area_m2: float, reflectance: float) -> float:
        # Monostatic RCS for flat plate (simplified)
        return area_m2 * reflectance

    def validate_stealth(self, layers: Tuple[Tuple[complex, float], ...], area_m2: float) -> Dict[str, Any]:
        reflectance = self.multilayer_reflectance(layers)
        rcs = self.radar_cross_section(area_m2, reflectance)
        return {
            'frequency_Hz': self.frequency_Hz,
            'wavelength_m': self.wavelength_m,
            'incident_angle_deg': self.incident_angle_deg,
            'multilayer_reflectance': reflectance,
            'radar_cross_section_m2': rcs
        }
