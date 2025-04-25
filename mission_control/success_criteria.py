from typing import Dict, Any, Callable, Optional

class SuccessCriteriaValidator:
    """
    Validates mission success based on mission type and scenario outcome.
    Supports registration of criteria functions for different mission types.
    """
    def __init__(self):
        self.criteria_funcs: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self.last_result: Optional[Dict[str, Any]] = None

    def register_criteria(self, mission_type: str, func: Callable[[Dict[str, Any]], bool]):
        self.criteria_funcs[mission_type] = func

    def validate(self, mission_type: str, final_state: Dict[str, Any]) -> bool:
        if mission_type not in self.criteria_funcs:
            raise ValueError(f"No success criteria registered for mission type: {mission_type}")
        result = self.criteria_funcs[mission_type](final_state)
        self.last_result = {
            'mission_type': mission_type,
            'final_state': final_state,
            'success': result
        }
        return result

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        return self.last_result
