"""
RotatableArm: Models a rotatable and extendable arm for engine or payload mounting.
"""
import numpy as np
from typing import Dict, Any

class RotatableArm:
    """
    Represents a rotatable and extendable arm for mounting propulsion or payload modules.
    Enhanced with status, power, damage, and safety checks.
    """
    def __init__(self, max_extension_m: float = 2.0, min_extension_m: float = 0.5, power_draw_watt: float = 60.0):
        self.angle_deg = 0.0
        self.extension_m = min_extension_m
        self.max_extension_m = max_extension_m
        self.min_extension_m = min_extension_m
        self.mounted_module = None
        self.status = 'idle'  # 'idle', 'moving', 'locked', 'error'
        self.damaged = False
        self.power_draw_watt = power_draw_watt
        self.locked = False
        self.position_report = {'angle_deg': self.angle_deg, 'extension_m': self.extension_m}

    def rotate(self, angle_delta_deg: float):
        if self.locked or self.damaged:
            self.status = 'locked' if self.locked else 'error'
            return self.angle_deg
        self.status = 'moving'
        self.angle_deg = (self.angle_deg + angle_delta_deg) % 360
        self.position_report['angle_deg'] = self.angle_deg
        self.status = 'idle'
        return self.angle_deg

    def extend(self, extension_delta_m: float):
        if self.locked or self.damaged:
            self.status = 'locked' if self.locked else 'error'
            return self.extension_m
        self.status = 'moving'
        self.extension_m = np.clip(self.extension_m + extension_delta_m, self.min_extension_m, self.max_extension_m)
        self.position_report['extension_m'] = self.extension_m
        self.status = 'idle'
        return self.extension_m

    def mount(self, module: Any):
        if self.damaged:
            return "Cannot mount: arm damaged."
        self.mounted_module = module
        return f"Module {module} mounted."

    def lock(self):
        self.locked = True
        self.status = 'locked'
        return 'Arm locked.'

    def unlock(self):
        self.locked = False
        self.status = 'idle'
        return 'Arm unlocked.'

    def take_damage(self):
        self.damaged = True
        self.status = 'error'
        return 'Arm damaged!'

    def repair(self):
        self.damaged = False
        self.status = 'idle'
        return 'Arm repaired.'

    def get_state(self) -> Dict[str, Any]:
        return {
            'angle_deg': self.angle_deg,
            'extension_m': self.extension_m,
            'mounted_module': str(self.mounted_module),
            'status': self.status,
            'locked': self.locked,
            'damaged': self.damaged,
            'power_draw_watt': self.power_draw_watt,
            'position_report': self.position_report,
        }
