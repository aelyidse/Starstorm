from typing import Dict, Any, List
import numpy as np

class EnvironmentalConditionMonitor:
    """
    Monitors environmental conditions and detects hazards for vehicle safety and survivability.
    Supports detection of extreme temperatures, radiation, pressure, debris, and chemical hazards.
    """
    def __init__(self, thresholds: Dict[str, float]):
        """
        thresholds: e.g., {'temp_high': 350, 'temp_low': -80, 'rad': 1e5, 'pressure_low': 100, ...}
        """
        self.thresholds = thresholds
        self.history: List[Dict[str, Any]] = []
        self.hazards: List[Dict[str, Any]] = []

    def ingest(self, env_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.history.append(env_data)
        detected = self.detect_hazards(env_data)
        if detected:
            self.hazards.extend(detected)
        return detected

    def detect_hazards(self, env_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        hazards = []
        if 'temperature' in env_data:
            if env_data['temperature'] > self.thresholds.get('temp_high', 1e6):
                hazards.append({'type': 'high_temp', 'value': env_data['temperature']})
            if env_data['temperature'] < self.thresholds.get('temp_low', -1e6):
                hazards.append({'type': 'low_temp', 'value': env_data['temperature']})
        if 'radiation' in env_data and env_data['radiation'] > self.thresholds.get('rad', 1e9):
            hazards.append({'type': 'high_radiation', 'value': env_data['radiation']})
        if 'pressure' in env_data and env_data['pressure'] < self.thresholds.get('pressure_low', 0):
            hazards.append({'type': 'low_pressure', 'value': env_data['pressure']})
        if 'debris_flux' in env_data and env_data['debris_flux'] > self.thresholds.get('debris_flux', 1e-3):
            hazards.append({'type': 'high_debris_flux', 'value': env_data['debris_flux']})
        if 'chem_hazard' in env_data and env_data['chem_hazard']:
            hazards.append({'type': 'chemical_hazard', 'value': env_data['chem_hazard']})
        return hazards

    def get_hazards(self) -> List[Dict[str, Any]]:
        return self.hazards

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
