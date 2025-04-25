import numpy as np
from typing import Dict, Any, Optional

class NanotubeMaterialSimulator:
    """
    Simulates material properties of reinforced carbon nanotube (CNT) structures.
    Supports mechanical, thermal, and electrical property estimation for composites.
    """
    def __init__(self, matrix_props: Dict[str, float], cnt_props: Dict[str, float], cnt_fraction: float):
        """
        matrix_props: e.g., {'E': Young's modulus, 'rho': density, 'k': thermal conductivity, ...}
        cnt_props: e.g., {'E': Young's modulus, 'rho': density, 'k': thermal conductivity, ...}
        cnt_fraction: volume fraction of CNTs in composite [0, 1]
        """
        self.matrix_props = matrix_props
        self.cnt_props = cnt_props
        self.cnt_fraction = cnt_fraction
        self.composite_props: Optional[Dict[str, float]] = None

    def rule_of_mixtures(self, key: str) -> float:
        # Simple linear rule of mixtures for parallel load
        return (self.cnt_fraction * self.cnt_props[key] +
                (1 - self.cnt_fraction) * self.matrix_props[key])

    def inverse_rule_of_mixtures(self, key: str) -> float:
        # For series load (e.g., thermal/electrical resistances)
        return 1.0 / (self.cnt_fraction / self.cnt_props[key] + (1 - self.cnt_fraction) / self.matrix_props[key])

    def simulate(self) -> Dict[str, float]:
        # Compute key composite properties
        props = {}
        for key in self.matrix_props:
            if key in ['E', 'rho']:
                props[key] = self.rule_of_mixtures(key)
            elif key in ['k', 'sigma']:
                props[key] = self.inverse_rule_of_mixtures(key)
            else:
                props[key] = self.rule_of_mixtures(key)
        self.composite_props = props
        return props

    def get_composite_properties(self) -> Optional[Dict[str, float]]:
        return self.composite_props
