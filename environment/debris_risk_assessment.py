import numpy as np
from typing import Dict, Any, Optional

class SpaceDebrisRiskAssessor:
    """
    Assesses risk of space debris impact for a vehicle in orbit or during atmospheric transit.
    Supports flux estimation, probability of impact, and lethality assessment.
    """
    def __init__(self, cross_section_m2: float, mission_duration_h: float, altitude_km: float):
        self.cross_section_m2 = cross_section_m2
        self.mission_duration_h = mission_duration_h
        self.altitude_km = altitude_km
        self.last_result: Optional[Dict[str, Any]] = None

    def debris_flux(self) -> float:
        # Empirical flux model (objects >1mm, per m^2 per year)
        # LEO: ~10^-6 to 10^-4, GEO: ~10^-7, scale with altitude
        if self.altitude_km < 2000:
            base_flux = 1e-5 * np.exp(-self.altitude_km / 1000)
        else:
            base_flux = 1e-7
        return base_flux

    def probability_of_impact(self) -> float:
        # P = 1 - exp(-flux * area * time)
        flux = self.debris_flux()  # per m^2 per year
        time_yr = self.mission_duration_h / 8760.0
        P = 1 - np.exp(-flux * self.cross_section_m2 * time_yr)
        return P

    def lethality_index(self, shield_threshold_j: float = 1e3) -> float:
        # Estimate lethality: fraction of impacts above shield threshold
        # Assume simple power-law for debris energy distribution
        # E > shield_threshold_j (arbitrary, e.g. 1kJ for Whipple shield)
        # For demo: lethality = fraction above threshold
        lethality = 0.05  # Placeholder: 5% of impacts above threshold
        return lethality

    def assess(self) -> Dict[str, Any]:
        flux = self.debris_flux()
        P = self.probability_of_impact()
        lethality = self.lethality_index()
        result = {
            'debris_flux_per_m2_per_year': flux,
            'probability_of_impact': P,
            'lethality_index': lethality,
            'expected_lethal_impacts': P * lethality
        }
        self.last_result = result
        return result

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        return self.last_result
