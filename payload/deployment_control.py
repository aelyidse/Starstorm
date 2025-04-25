from typing import Dict, Any, Optional
import time

class PayloadDeploymentController:
    """
    Controls deployment and retraction of payload systems (e.g., sensors, antennas, effectors).
    Supports state management, safety checks, and timed actuation.
    """
    def __init__(self, payloads: Dict[str, Dict[str, Any]]):
        """
        payloads: dict of payload_name -> {'state': 'retracted'|'deployed', 'deploy_time': float, 'retract_time': float, ...}
        """
        self.payloads = payloads
        self.last_action_time: Dict[str, float] = {p: 0.0 for p in payloads}

    def deploy(self, name: str) -> bool:
        if self.payloads[name]['state'] == 'deployed':
            return False
        # Simulate deployment time
        time.sleep(self.payloads[name].get('deploy_time', 1.0))
        self.payloads[name]['state'] = 'deployed'
        self.last_action_time[name] = time.time()
        return True

    def retract(self, name: str) -> bool:
        if self.payloads[name]['state'] == 'retracted':
            return False
        # Simulate retraction time
        time.sleep(self.payloads[name].get('retract_time', 1.0))
        self.payloads[name]['state'] = 'retracted'
        self.last_action_time[name] = time.time()
        return True

    def get_state(self, name: str) -> Optional[str]:
        return self.payloads.get(name, {}).get('state', None)

    def get_all_states(self) -> Dict[str, str]:
        return {k: v['state'] for k, v in self.payloads.items()}
