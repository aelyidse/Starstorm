from typing import Dict, Any, List, Optional

class TelemetryPrioritizer:
    """
    Prioritizes telemetry data for transmission based on operational context and mission phase.
    Supports dynamic priority assignment, context-aware filtering, and custom logic.
    """
    def __init__(self, default_priorities: Optional[Dict[str, int]] = None):
        # Example: {'system_health': 1, 'position': 2, 'payload_status': 3, ...}
        self.default_priorities = default_priorities or {
            'system_health': 1,
            'position': 2,
            'velocity': 3,
            'attitude': 4,
            'payload_status': 5,
            'environment': 6,
            'mission_status': 7,
            'extra': 8
        }
        self.context_overrides: Dict[str, Dict[str, int]] = {}
        self.last_priorities: Dict[str, int] = self.default_priorities.copy()

    def set_context_override(self, context: str, priorities: Dict[str, int]):
        self.context_overrides[context] = priorities

    def get_priorities(self, context: Optional[str] = None) -> Dict[str, int]:
        if context and context in self.context_overrides:
            self.last_priorities = self.context_overrides[context]
        else:
            self.last_priorities = self.default_priorities.copy()
        return self.last_priorities

    def prioritize(self, vehicle_state: Dict[str, Any], context: Optional[str] = None) -> List[str]:
        # Returns ordered list of telemetry fields to transmit
        priorities = self.get_priorities(context)
        present_fields = [k for k in vehicle_state if k in priorities]
        ordered = sorted(present_fields, key=lambda k: priorities[k])
        return ordered
