import numpy as np
from typing import Dict, Any, List

class DynamicStressAnalyzer:
    """
    Performs dynamic stress analysis of CNT-reinforced structures under varying flight conditions.
    Supports time-varying loads, acceleration, and environmental effects.
    """
    def __init__(self, material_props: Dict[str, float], geometry: Dict[str, Any]):
        """
        material_props: e.g., {'E': Young's modulus, 'rho': density, ...}
        geometry: e.g., {'length': L, 'area': A, 'I': moment of inertia, ...}
        """
        self.material_props = material_props
        self.geometry = geometry
        self.results: List[Dict[str, Any]] = []

    def compute_stress(self, force: float, area: float) -> float:
        return force / area

    def compute_bending_stress(self, moment: float, y: float, I: float) -> float:
        return moment * y / I

    def analyze(self, flight_profile: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        flight_profile: list of dicts with {'time': t, 'accel': a, 'load_factor': n, 'env': {...}}
        """
        E = self.material_props.get('E', 1.0)
        rho = self.material_props.get('rho', 1.0)
        A = self.geometry.get('area', 1.0)
        I = self.geometry.get('I', 1.0)
        L = self.geometry.get('length', 1.0)
        y = self.geometry.get('y_max', 1.0)
        results = []
        for step in flight_profile:
            a = step.get('accel', 0.0)  # m/s^2
            n = step.get('load_factor', 1.0)
            mass = rho * L * A
            F = mass * a * n  # Dynamic force
            stress = self.compute_stress(F, A)
            # Assume bending moment due to load at tip (cantilever)
            M = F * L
            bending_stress = self.compute_bending_stress(M, y, I)
            results.append({
                'time': step['time'],
                'axial_stress': stress,
                'bending_stress': bending_stress,
                'total_stress': stress + bending_stress,
                'accel': a,
                'load_factor': n,
                'env': step.get('env', {})
            })
        self.results = results
        return results

    def get_max_stress(self) -> float:
        if not self.results:
            return 0.0
        return max(r['total_stress'] for r in self.results)
