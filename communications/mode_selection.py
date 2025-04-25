from typing import Dict, Any

class CommunicationModeSelector:
    """
    Selects the optimal communication mode based on mission parameters, threat environment, and system constraints.
    Supports modes such as secure, LPI, FHSS, satellite, and emergency fallback.
    """
    def __init__(self, available_modes: Dict[str, Dict[str, Any]]):
        """
        available_modes: dict of mode name to capability/constraint dict
        Example: {'secure': {...}, 'lpi': {...}, 'fhss': {...}, 'satcom': {...}, 'emergency': {...}}
        """
        self.available_modes = available_modes
        self.current_mode = None

    def select_mode(self, mission_params: Dict[str, Any], threat_env: Dict[str, Any], system_status: Dict[str, Any]) -> str:
        # Example logic (can be extended with scoring, rules, or ML):
        if mission_params.get('stealth_required', False) and 'lpi' in self.available_modes:
            self.current_mode = 'lpi'
        elif threat_env.get('jamming_detected', False) and 'fhss' in self.available_modes:
            self.current_mode = 'fhss'
        elif mission_params.get('satellite_link', False) and 'satcom' in self.available_modes:
            self.current_mode = 'satcom'
        elif system_status.get('critical', False) and 'emergency' in self.available_modes:
            self.current_mode = 'emergency'
        elif 'secure' in self.available_modes:
            self.current_mode = 'secure'
        else:
            self.current_mode = next(iter(self.available_modes.keys()))
        return self.current_mode

    def get_current_mode(self) -> str:
        return self.current_mode
