from typing import Dict, Any, List, Optional
import numpy as np

class AutonomousThreatResponder:
    """
    Selects and initiates autonomous threat responses based on threat type, severity, and mission context.
    Supports rule-based, priority-based, and probabilistic selection strategies.
    """
    def __init__(self, response_catalog: List[Dict[str, Any]]):
        """
        response_catalog: List of {'type': threat_type, 'response': action, 'priority': int, ...}
        """
        self.response_catalog = response_catalog
        self.history: List[Dict[str, Any]] = []
        self.last_response: Optional[Dict[str, Any]] = None

    def select_response(self, threat: Dict[str, Any], context: Optional[Dict[str, Any]] = None, strategy: str = 'priority') -> Optional[Dict[str, Any]]:
        """
        threat: {'type': threat_type, 'severity': float, ...}
        context: Optional mission/system context
        strategy: 'priority', 'probabilistic', or 'rule'
        """
        candidates = [r for r in self.response_catalog if r['type'] == threat['type']]
        if not candidates:
            return None
        if strategy == 'priority':
            # Highest priority response
            selected = max(candidates, key=lambda r: r.get('priority', 0))
        elif strategy == 'probabilistic':
            # Weighted random by priority
            priorities = np.array([r.get('priority', 1) for r in candidates])
            idx = int(np.random.choice(len(candidates), p=priorities/priorities.sum()))
            selected = candidates[idx]
        elif strategy == 'rule':
            # Example: escalate for high severity
            for r in sorted(candidates, key=lambda r: -r.get('priority', 0)):
                if threat.get('severity', 0) > 0.8 and r.get('escalate', False):
                    selected = r
                    break
            else:
                selected = candidates[0]
        else:
            selected = candidates[0]
        self.last_response = selected
        self.history.append({'threat': threat, 'response': selected, 'context': context})
        return selected

    def get_last_response(self) -> Optional[Dict[str, Any]]:
        return self.last_response

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
