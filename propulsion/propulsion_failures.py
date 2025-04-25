import random
from typing import Dict, Any, Optional

class PropulsionFailureSimulator:
    """
    Simulates failure modes and contingencies for propulsion systems.
    Supports stochastic and deterministic fault injection, detection, and recovery.
    """
    def __init__(self, failure_rates: Optional[Dict[str, float]] = None):
        # failure_rates: {'engine_shutdown': 1e-5, 'fuel_leak': 2e-6, ...} per second
        self.failure_rates = failure_rates or {}
        self.active_failures = set()

    def inject_random_failures(self, dt: float):
        # For each failure mode, probabilistically inject based on rate and dt
        for mode, rate in self.failure_rates.items():
            if random.random() < 1.0 - pow(1.0 - rate, dt):
                self.active_failures.add(mode)

    def inject_failure(self, mode: str):
        self.active_failures.add(mode)

    def clear_failure(self, mode: str):
        self.active_failures.discard(mode)

    def clear_all(self):
        self.active_failures.clear()

    def get_failures(self) -> set:
        return set(self.active_failures)

    def apply_failures(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        # Modify system_state in-place or return a new dict reflecting active failures
        state = system_state.copy()
        for mode in self.active_failures:
            if mode == 'engine_shutdown':
                state['thrust_N'] = 0.0
                state['engine_running'] = False
            elif mode == 'fuel_leak':
                state['fuel_leak_kg'] = state.get('fuel_leak_kg', 0.0) + random.uniform(0.01, 0.1)
            elif mode == 'thrust_vector_failure':
                state['gimbal_deg'] = (0.0, 0.0)
                state['vectoring_active'] = False
            elif mode == 'power_loss':
                state['power_W'] = 0.0
            elif mode == 'xenon_valve_stuck':
                state['xenon_flow_stuck'] = True
            # Extend with additional failure modes as needed
        return state

    def detect_failures(self, system_state: Dict[str, Any]) -> Dict[str, bool]:
        # Example: detect based on abnormal states
        detections = {}
        if system_state.get('thrust_N', 1.0) == 0.0 and system_state.get('engine_running', True):
            detections['engine_shutdown'] = True
        if system_state.get('fuel_leak_kg', 0.0) > 0.0:
            detections['fuel_leak'] = True
        if not system_state.get('vectoring_active', True):
            detections['thrust_vector_failure'] = True
        if system_state.get('power_W', 1.0) == 0.0:
            detections['power_loss'] = True
        if system_state.get('xenon_flow_stuck', False):
            detections['xenon_valve_stuck'] = True
        return detections
