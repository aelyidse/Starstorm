import numpy as np
from typing import Dict, Any, Optional

class MetamaterialSurface:
    """
    Models electromagnetic (EM) interaction for metamaterial surfaces.
    Supports calculation of reflection, absorption, and transmission coefficients for arbitrary frequencies and incident angles.
    """
    def __init__(self, epsilon_r: float, mu_r: float, thickness_m: float, conductivity_Sm: float = 0.0):
        self.epsilon_r = epsilon_r  # Relative permittivity
        self.mu_r = mu_r            # Relative permeability
        self.thickness_m = thickness_m
        self.conductivity_Sm = conductivity_Sm
        self.epsilon_0 = 8.854187817e-12
        self.mu_0 = 4 * np.pi * 1e-7

    def get_impedance(self, freq_Hz: float) -> float:
        # Surface impedance for given frequency
        omega = 2 * np.pi * freq_Hz
        epsilon = self.epsilon_0 * self.epsilon_r
        mu = self.mu_0 * self.mu_r
        sigma = self.conductivity_Sm
        return np.sqrt(1j * omega * mu / (sigma + 1j * omega * epsilon))

    def fresnel_coefficients(self, freq_Hz: float, incident_angle_deg: float = 0.0) -> Dict[str, Any]:
        # Calculate reflection and transmission using Fresnel equations
        theta_i = np.radians(incident_angle_deg)
        n1 = 1.0  # Assume air
        n2 = np.sqrt(self.epsilon_r * self.mu_r)
        # Snell's law
        sin_theta_t = n1 / n2 * np.sin(theta_i)
        if np.abs(sin_theta_t) > 1.0:
            # Total internal reflection
            return {'R_s': 1.0, 'R_p': 1.0, 'T_s': 0.0, 'T_p': 0.0}
        theta_t = np.arcsin(sin_theta_t)
        # s-polarization
        rs = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
        ts = 2 * n1 * np.cos(theta_i) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
        # p-polarization
        rp = (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))
        tp = 2 * n1 * np.cos(theta_i) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))
        return {'R_s': np.abs(rs)**2, 'R_p': np.abs(rp)**2, 'T_s': np.abs(ts)**2, 'T_p': np.abs(tp)**2}

    def absorption(self, freq_Hz: float, incident_angle_deg: float = 0.0) -> float:
        # Estimate absorption using 1 - (reflection + transmission)
        coeffs = self.fresnel_coefficients(freq_Hz, incident_angle_deg)
        R = 0.5 * (coeffs['R_s'] + coeffs['R_p'])
        T = 0.5 * (coeffs['T_s'] + coeffs['T_p'])
        A = 1.0 - R - T
        return max(0.0, min(1.0, A))
