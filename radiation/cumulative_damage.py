from typing import Dict, Any, Optional, List

class CumulativeRadiationDamageTracker:
    """
    Tracks cumulative radiation dose and damage for materials and electronic components.
    Supports dose accumulation, threshold checks, and degradation estimation.
    """
    def __init__(self, dose_limits: Dict[str, float]):
        """
        dose_limits: dict of item/component -> max allowable dose (rad or Gy)
        """
        self.dose_limits = dose_limits
        self.cumulative_dose: Dict[str, float] = {k: 0.0 for k in dose_limits}
        self.degradation: Dict[str, float] = {k: 0.0 for k in dose_limits}
        self.history: List[Dict[str, Any]] = []

    def add_dose(self, item: str, dose: float):
        if item in self.cumulative_dose:
            self.cumulative_dose[item] += dose
            self.history.append({'item': item, 'dose': dose, 'total': self.cumulative_dose[item]})
            self.degradation[item] = self.estimate_degradation(item)

    def estimate_degradation(self, item: str) -> float:
        # Linear degradation: 0 (none) to 1 (complete failure at limit)
        limit = self.dose_limits[item]
        dose = self.cumulative_dose[item]
        return min(1.0, dose / limit)

    def get_status(self, item: str) -> Dict[str, Any]:
        return {
            'cumulative_dose': self.cumulative_dose[item],
            'degradation': self.degradation[item],
            'dose_limit': self.dose_limits[item],
            'exceeded': self.cumulative_dose[item] >= self.dose_limits[item]
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        return {k: self.get_status(k) for k in self.cumulative_dose}

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
