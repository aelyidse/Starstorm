from typing import Dict, Any, Optional
import time

class PayloadStatusMonitor:
    """
    Monitors payload system status and performs diagnostics for health and readiness.
    Supports state polling, fault detection, and diagnostic reporting.
    """
    def __init__(self, payloads: Dict[str, Dict[str, Any]]):
        """
        payloads: dict of payload_name -> {'state': 'retracted'|'deployed', ...}
        """
        self.payloads = payloads
        self.status: Dict[str, Dict[str, Any]] = {k: {'state': v['state'], 'last_check': time.time(), 'fault': False, 'diagnostic': 'OK'} for k, v in payloads.items()}

    def poll_status(self, name: str) -> Dict[str, Any]:
        # Simulate polling hardware for status
        state = self.payloads[name]['state']
        health = self.simulate_health_check(name)
        self.status[name].update({'state': state, 'last_check': time.time(), 'fault': not health, 'diagnostic': 'OK' if health else 'FAULT'})
        return self.status[name]

    def simulate_health_check(self, name: str) -> bool:
        # Placeholder for real health check logic
        # For demo: randomly fail with low probability
        import random
        return random.random() > 0.01

    def run_diagnostics(self, name: str) -> Dict[str, Any]:
        # Simulate running diagnostics
        status = self.poll_status(name)
        if status['fault']:
            status['diagnostic'] = 'Fault detected: check connections or power.'
        else:
            status['diagnostic'] = 'OK'
        return status

    def get_status(self, name: str) -> Optional[Dict[str, Any]]:
        return self.status.get(name, None)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        return self.status
