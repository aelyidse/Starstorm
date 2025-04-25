from typing import Dict, Any, List, Optional

class DamageAssessmentAndMissionImpact:
    """
    Assesses structural damage and predicts mission impact based on health monitoring and stress analysis.
    Supports severity grading, functional loss estimation, and mission success prediction.
    """
    def __init__(self, thresholds: Dict[str, float], mission_criteria: Dict[str, Any]):
        """
        thresholds: e.g., {'max_stress': 1e8, 'max_anomaly_count': 3}
        mission_criteria: e.g., {'min_payloads': 2, 'max_damage': 0.2}
        """
        self.thresholds = thresholds
        self.mission_criteria = mission_criteria
        self.damage_report: Optional[Dict[str, Any]] = None
        self.impact_report: Optional[Dict[str, Any]] = None

    def assess_damage(self, health_summary: Dict[str, Any], stress_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Grade damage based on anomalies and stress exceedances
        anomaly_count = health_summary.get('anomaly_count', 0)
        max_stress = max((r['total_stress'] for r in stress_results), default=0.0)
        damage_severity = 'none'
        if anomaly_count == 0 and max_stress < self.thresholds['max_stress']:
            damage_severity = 'none'
        elif anomaly_count <= 2 and max_stress < 1.2 * self.thresholds['max_stress']:
            damage_severity = 'minor'
        elif anomaly_count <= 5 or max_stress < 1.5 * self.thresholds['max_stress']:
            damage_severity = 'moderate'
        else:
            damage_severity = 'severe'
        self.damage_report = {
            'anomaly_count': anomaly_count,
            'max_stress': max_stress,
            'damage_severity': damage_severity
        }
        return self.damage_report

    def predict_mission_impact(self, payload_status: Dict[str, str], damage_report: Dict[str, Any]) -> Dict[str, Any]:
        # Estimate functional loss and mission success probability
        n_operational = sum(1 for v in payload_status.values() if v == 'deployed')
        damage_level = damage_report['damage_severity']
        max_damage = {'none': 0.0, 'minor': 0.1, 'moderate': 0.3, 'severe': 1.0}[damage_level]
        can_complete = (
            n_operational >= self.mission_criteria.get('min_payloads', 1)
            and max_damage <= self.mission_criteria.get('max_damage', 1.0)
        )
        success_prob = max(0.0, 1.0 - max_damage)
        self.impact_report = {
            'n_operational_payloads': n_operational,
            'damage_level': damage_level,
            'mission_success_probability': success_prob,
            'can_complete_mission': can_complete
        }
        return self.impact_report

    def get_reports(self) -> Dict[str, Any]:
        return {
            'damage_report': self.damage_report,
            'impact_report': self.impact_report
        }
