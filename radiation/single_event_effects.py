import numpy as np
from typing import Dict, Any, Optional

class SingleEventEffectSimulator:
    """
    Simulates single-event effects (SEE) on electronic components due to energetic particles.
    Models SEU (upset), SEL (latchup), and SEB (burnout) probabilities based on flux and component sensitivity.
    """
    def __init__(self, component_sensitivities: Dict[str, float]):
        """
        component_sensitivities: dict of component_name -> cross-section (cm^2)
        """
        self.component_sensitivities = component_sensitivities
        self.last_results: Optional[Dict[str, Any]] = None

    def simulate(self, flux_profile: Dict[str, float], duration_s: float = 3600.0) -> Dict[str, Any]:
        """
        flux_profile: {'trapped_flux': ..., 'solar_flux': ..., 'cosmic_flux': ...} (particles/cm^2/s)
        duration_s: exposure duration in seconds
        """
        results = {}
        total_flux = flux_profile['trapped_flux'] + flux_profile['solar_flux'] + flux_profile['cosmic_flux']
        for name, sigma in self.component_sensitivities.items():
            # Poisson probability of at least one event: P = 1 - exp(-rate * t)
            rate = sigma * total_flux  # events per second
            n_events = np.random.poisson(rate * duration_s)
            upset_prob = 1 - np.exp(-rate * duration_s)
            results[name] = {
                'expected_events': rate * duration_s,
                'simulated_events': n_events,
                'upset_probability': upset_prob
            }
        self.last_results = results
        return results

    def get_last_results(self) -> Optional[Dict[str, Any]]:
        return self.last_results
