from typing import Dict, Any, Optional
from .rocket_engine import RocketEngine
from .ion_thruster import IonThruster

class DualPropulsionController:
    """
    Manages mode transitions between rocket and ion propulsion.
    Ensures safe, deterministic, and context-aware switching.
    """
    def __init__(self, rocket: RocketEngine, ion: IonThruster):
        self.rocket = rocket
        self.ion = ion
        self.mode = 'rocket'  # 'rocket', 'ion', or 'off'
        self.last_mode = None

    def transition(self, target_mode: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Attempts to transition propulsion mode. Returns True if successful.
        context: Optional dict with keys like 'altitude', 'velocity', 'power_available', etc.
        """
        if target_mode == self.mode:
            return True  # Already in desired mode
        # Safety/context checks
        if target_mode == 'rocket':
            if self.rocket.fuel_mass_kg <= 0 or self.rocket.oxidizer_mass_kg <= 0:
                return False  # Insufficient propellant
            self.ion.shutdown()
            self.rocket.start()
            self.mode = 'rocket'
            self.last_mode = 'ion'
            return True
        elif target_mode == 'ion':
            if self.ion.xenon_mass_kg <= 0:
                return False  # Insufficient xenon
            if context and context.get('power_available', 0.0) < self.ion.current_power_W:
                return False  # Not enough power
            self.rocket.shutdown()
            self.ion.start()
            self.mode = 'ion'
            self.last_mode = 'rocket'
            return True
        elif target_mode == 'off':
            self.rocket.shutdown()
            self.ion.shutdown()
            self.mode = 'off'
            self.last_mode = self.mode
            return True
        else:
            return False  # Invalid mode

    def get_mode(self) -> str:
        return self.mode

    def update(self, dt: float, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.mode == 'rocket':
            return {'mode': 'rocket', **self.rocket.update(dt, context.get('ambient_pressure_Pa', 101325.0) if context else 101325.0)}
        elif self.mode == 'ion':
            return {'mode': 'ion', **self.ion.update(dt)}
        else:
            return {'mode': 'off'}
