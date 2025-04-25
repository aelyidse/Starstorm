import numpy as np
from typing import Dict, Any, Optional

class ThermosphericRadiationEnvironment:
    """
    Models radiation environment for thermospheric operation (100–600 km altitude).
    Supports trapped particles, solar events, cosmic rays, and shielding effects.
    """
    def __init__(self, altitude_km: float, solar_activity: str = 'average', shielding_gcm2: float = 0.0):
        self.altitude_km = altitude_km
        self.solar_activity = solar_activity  # 'low', 'average', 'high'
        self.shielding_gcm2 = shielding_gcm2
        self.env_profile: Optional[Dict[str, float]] = None

    def model_trapped_particles(self) -> float:
        # Simple empirical model for trapped proton/electron flux (arbitrary units)
        base_flux = 1e4 * np.exp(-self.altitude_km / 200)
        if self.solar_activity == 'high':
            base_flux *= 1.5
        elif self.solar_activity == 'low':
            base_flux *= 0.7
        return base_flux

    def model_solar_events(self) -> float:
        # Solar particle event contribution (arbitrary units)
        if self.solar_activity == 'high':
            return 1e4 * np.random.uniform(1, 3)
        elif self.solar_activity == 'low':
            return 1e2 * np.random.uniform(0.5, 1.5)
        return 1e3 * np.random.uniform(0.8, 1.2)

    def model_cosmic_rays(self) -> float:
        # Galactic cosmic ray background (arbitrary units)
        return 1e2 * np.exp(-self.shielding_gcm2 / 10)

    def compute_total_dose(self) -> Dict[str, float]:
        # Combine all sources and apply shielding
        trapped = self.model_trapped_particles() * np.exp(-self.shielding_gcm2 / 5)
        solar = self.model_solar_events() * np.exp(-self.shielding_gcm2 / 2)
        cosmic = self.model_cosmic_rays()
        total = trapped + solar + cosmic
        self.env_profile = {
            'trapped_flux': trapped,
            'solar_flux': solar,
            'cosmic_flux': cosmic,
            'total_dose': total
        }
        return self.env_profile

    def get_environment_profile(self) -> Optional[Dict[str, float]]:
        return self.env_profile
