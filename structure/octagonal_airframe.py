"""
OctagonalAirframe: Models the octagonal airframe structure and its mechanical features.
"""
import numpy as np
from typing import Dict, Any

class OctagonalAirframe:
    """
    Represents an octagonal airframe optimized for high-altitude and space operations.
    Includes mounting points for rotatable/extendable arms and payload bays.
    Enhanced with material, mass, and bay operation logic.
    """
    def __init__(self, arm_count: int = 4, radius_m: float = 1.5, material: str = 'carbon composite', wall_thickness_m: float = 0.02, base_mass_kg: float = 120.0):
        self.arm_count = arm_count
        self.radius_m = radius_m
        self.material = material
        self.wall_thickness_m = wall_thickness_m
        self.base_mass_kg = base_mass_kg
        self.shape = self._generate_octagon()
        self.arm_mounts = self._compute_arm_mounts()
        self.payload_bay = {'state': 'retracted', 'capacity_kg': 200.0, 'current_payload': None}
        self.health = 1.0  # 1.0 = pristine, 0.0 = destroyed

    def _generate_octagon(self):
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]
        return np.stack([self.radius_m * np.cos(angles), self.radius_m * np.sin(angles)], axis=-1)

    def _compute_arm_mounts(self):
        return [self.shape[i] for i in range(0, 8, 8 // self.arm_count)]

    def extend_arm(self, arm_idx: int):
        if 0 <= arm_idx < self.arm_count:
            return f"Arm {arm_idx} extended."
        return f"Invalid arm index."

    def retract_arm(self, arm_idx: int):
        if 0 <= arm_idx < self.arm_count:
            return f"Arm {arm_idx} retracted."
        return f"Invalid arm index."

    def open_payload_bay(self):
        self.payload_bay['state'] = 'open'
        return "Payload bay opened."

    def close_payload_bay(self):
        self.payload_bay['state'] = 'closed'
        return "Payload bay closed."

    def load_payload(self, payload: Any, mass_kg: float):
        if mass_kg > self.payload_bay['capacity_kg']:
            return "Payload too heavy."
        self.payload_bay['current_payload'] = payload
        return f"Payload loaded: {payload} ({mass_kg} kg)"

    def unload_payload(self):
        p = self.payload_bay['current_payload']
        self.payload_bay['current_payload'] = None
        return f"Payload unloaded: {p}"

    def take_damage(self, fraction: float):
        self.health = max(0.0, self.health - fraction)
        return f"Airframe health: {self.health:.2f}"

    def get_geometry(self) -> Dict[str, Any]:
        return {
            'vertices': self.shape.tolist(),
            'arm_mounts': [m.tolist() for m in self.arm_mounts],
            'payload_bay_state': self.payload_bay['state'],
            'material': self.material,
            'wall_thickness_m': self.wall_thickness_m,
            'base_mass_kg': self.base_mass_kg,
            'health': self.health
        }
