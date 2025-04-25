from typing import Dict, Any, List, Optional

class DataCollectionPrioritizer:
    """
    Prioritizes sensor and payload data collection based on mission objectives, constraints, and available resources.
    Supports dynamic re-prioritization and integration with mission logic.
    """
    def __init__(self, objectives: List[Dict[str, Any]]):
        """
        objectives: List of dicts, e.g., [{'type': 'imagery', 'area': ..., 'priority': 1}, ...]
        """
        self.objectives = objectives
        self.priority_map = self._build_priority_map(objectives)

    def _build_priority_map(self, objectives: List[Dict[str, Any]]) -> Dict[str, int]:
        # Map data types/areas to priorities
        pmap = {}
        for obj in objectives:
            key = obj.get('type') + str(obj.get('area', ''))
            pmap[key] = obj.get('priority', 10)
        return pmap

    def prioritize(self, data_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # data_requests: [{'type': ..., 'area': ..., 'data': ...}, ...]
        def keyfunc(req):
            k = req.get('type') + str(req.get('area', ''))
            return self.priority_map.get(k, 100)
        return sorted(data_requests, key=keyfunc)

    def update_objectives(self, new_objectives: List[Dict[str, Any]]):
        self.objectives = new_objectives
        self.priority_map = self._build_priority_map(new_objectives)
