import numpy as np
from typing import Dict, Any, Optional

class EmergencyReturnPlanner:
    """
    Plans emergency return-to-base (RTB) trajectories, accounting for degraded navigation, propulsion, or control.
    Supports fallback logic and degraded mode selection.
    """
    def __init__(self, base_pos: np.ndarray):
        self.base_pos = base_pos

    def plan_rtb(self, current_state: Dict[str, Any], degraded: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        current_state: {'pos': np.array, 'vel': np.array, ...}
        degraded: {'navigation': bool, 'propulsion': bool, 'control': bool}
        """
        degraded = degraded or {}
        nav_ok = not degraded.get('navigation', False)
        prop_ok = not degraded.get('propulsion', False)
        ctrl_ok = not degraded.get('control', False)
        # If navigation degraded, use last known heading
        if not nav_ok:
            heading = current_state.get('last_heading', np.array([1.0, 0.0, 0.0]))
            rtb_vector = heading / np.linalg.norm(heading)
            mode = 'blind'
        else:
            rtb_vector = self.base_pos - current_state['pos']
            rtb_vector /= np.linalg.norm(rtb_vector)
            mode = 'normal'
        # If propulsion degraded, limit speed
        speed = np.linalg.norm(current_state['vel'])
        max_speed = 10.0 if prop_ok else 2.0
        rtb_vel = rtb_vector * min(speed, max_speed)
        # If control degraded, add safety margin
        safety_margin = 100.0 if ctrl_ok else 500.0
        return {
            'rtb_vector': rtb_vector,
            'rtb_velocity': rtb_vel,
            'mode': mode,
            'safety_margin_m': safety_margin
        }
