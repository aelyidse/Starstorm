from typing import Dict, Any, List, Optional

class FallbackCommunicationManager:
    """
    Implements fallback communication strategies for contested or denied environments.
    Supports mode downgrading, message prioritization, and robust emergency signaling.
    """
    def __init__(self, available_modes: List[str]):
        self.available_modes = available_modes
        self.fallback_order = self._default_fallback_order()
        self.current_mode = None

    def _default_fallback_order(self) -> List[str]:
        # Prioritize most robust/stealth modes last
        priority = ['secure', 'lpi', 'fhss', 'satcom', 'emergency']
        return [m for m in priority if m in self.available_modes]

    def select_fallback_mode(self, failed_modes: Optional[List[str]] = None) -> Optional[str]:
        failed_modes = failed_modes or []
        for mode in self.fallback_order:
            if mode not in failed_modes:
                self.current_mode = mode
                return mode
        self.current_mode = None
        return None

    def prioritize_messages(self, messages: List[Dict[str, Any]], critical_only: bool = False) -> List[Dict[str, Any]]:
        # Prioritize or filter messages for fallback mode
        if critical_only:
            return [m for m in messages if m.get('priority', 10) <= 1 or m.get('type') == 'emergency']
        return sorted(messages, key=lambda m: m.get('priority', 10))

    def get_current_mode(self) -> Optional[str]:
        return self.current_mode
