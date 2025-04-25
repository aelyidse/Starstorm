from typing import Dict, Any, Optional

class AdaptiveShieldingController:
    """
    Controls adaptive shielding systems for space vehicles.
    Supports dynamic adjustment of shielding thickness, orientation, or active elements in response to threats.
    """
    def __init__(self, base_thickness: float, max_thickness: float):
        self.base_thickness = base_thickness
        self.max_thickness = max_thickness
        self.current_thickness = base_thickness
        self.history = []
        self.active_mode = 'passive'  # or 'active'

    def assess_threat(self, env_profile: Dict[str, float], debris_risk: Dict[str, Any]) -> float:
        # Simple threat score: weighted sum of radiation dose and debris risk
        dose_score = env_profile.get('total_dose', 0.0) / 1e5
        debris_score = debris_risk.get('probability_of_impact', 0.0)
        return dose_score + debris_score

    def adjust_shielding(self, threat_score: float):
        # Increase thickness proportionally to threat, within limits
        target_thickness = min(self.max_thickness, self.base_thickness * (1 + threat_score))
        self.current_thickness = target_thickness
        self.active_mode = 'active' if threat_score > 0.5 else 'passive'
        self.history.append({'threat_score': threat_score, 'thickness': self.current_thickness, 'mode': self.active_mode})
        return self.current_thickness

    def get_status(self) -> Dict[str, Any]:
        return {
            'current_thickness': self.current_thickness,
            'mode': self.active_mode,
            'history': self.history[-10:]  # last 10 adjustments
        }
