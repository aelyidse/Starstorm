import numpy as np
from typing import Dict, Any, List

class ThermalProtectionSystem:
    """
    Simulates and manages thermal protection for atmospheric reentry.
    Supports heat flux modeling, ablation prediction, and active/passive control.
    """
    def __init__(self, material_props: Dict[str, float], thickness: float):
        """
        material_props: {'k': conductivity, 'rho': density, 'c': specific heat, 'ablation_temp': K, ...}
        thickness: TPS thickness in meters
        """
        self.material_props = material_props
        self.thickness = thickness
        self.history: List[Dict[str, Any]] = []
        self.ablation_depth = 0.0

    def compute_heat_flux(self, v: float, rho_atm: float) -> float:
        # Sutton-Graves: q = k * sqrt(rho_atm / R_n) * v^3
        # For simplicity, assume blunt body R_n = 1m, k = 1.83e-4 (SI)
        k = 1.83e-4
        R_n = 1.0
        return k * np.sqrt(rho_atm / R_n) * v ** 3

    def step(self, v: float, rho_atm: float, dt: float, T_surface: float) -> Dict[str, Any]:
        # Compute heat flux and update TPS state
        q = self.compute_heat_flux(v, rho_atm)
        # Heat absorbed by TPS
        m = self.material_props['rho'] * self.thickness
        dT = q * dt / (m * self.material_props['c'])
        T_surface += dT
        ablation = 0.0
        if T_surface > self.material_props['ablation_temp']:
            # Estimate ablation rate (simple linear loss)
            ablation = 1e-5 * (T_surface - self.material_props['ablation_temp']) * dt
            self.ablation_depth += ablation
            T_surface = self.material_props['ablation_temp']
        state = {
            'heat_flux': q,
            'T_surface': T_surface,
            'ablation_depth': self.ablation_depth,
            'thickness_remaining': max(0.0, self.thickness - self.ablation_depth)
        }
        self.history.append(state)
        return state

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
