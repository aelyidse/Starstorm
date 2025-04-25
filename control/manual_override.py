from typing import Dict, Any, Optional
import time

class ManualOverrideManager:
    """
    Implements manual override capabilities for all automated systems.
    Supports override activation, command injection, status reporting, and logging.
    """
    def __init__(self, subsystems: Optional[list] = None):
        self.subsystems = subsystems or []
        self.override_active: Dict[str, bool] = {s: False for s in self.subsystems}
        self.override_log: list = []
        self.last_override: Optional[Dict[str, Any]] = None

    def activate_override(self, subsystem: str, command: Dict[str, Any]):
        self.override_active[subsystem] = True
        entry = {
            'subsystem': subsystem,
            'command': command,
            'action': 'activate',
            'time': time.time()
        }
        self.override_log.append(entry)
        self.last_override = entry
        # In a real system, inject the command to the subsystem controller here

    def deactivate_override(self, subsystem: str):
        self.override_active[subsystem] = False
        entry = {
            'subsystem': subsystem,
            'action': 'deactivate',
            'time': time.time()
        }
        self.override_log.append(entry)
        self.last_override = entry
        # In a real system, return control to automation here

    def is_override_active(self, subsystem: str) -> bool:
        return self.override_active.get(subsystem, False)

    def get_override_log(self):
        return self.override_log

    def get_last_override(self):
        return self.last_override
