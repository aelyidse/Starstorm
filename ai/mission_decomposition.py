from typing import Dict, Any, List, Optional

class MissionObjectiveDecomposer:
    """
    Decomposes high-level mission objectives into executable tasks for autonomous systems.
    Supports hierarchical breakdown, dependency analysis, and task parameterization.
    """
    def __init__(self):
        self.last_objectives: Optional[List[Dict[str, Any]]] = None
        self.last_tasks: Optional[List[Dict[str, Any]]] = None

    def decompose(self, objectives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        objectives: List of dicts, e.g., [{'objective': 'surveil_area', 'area': ..., 'priority': 1}, ...]
        Returns: List of executable task dicts
        """
        tasks = []
        for obj in objectives:
            if obj['objective'] == 'surveil_area':
                area = obj.get('area')
                pattern = obj.get('pattern', 'lawnmower')
                tasks.append({'task': 'generate_waypoints', 'area': area, 'pattern': pattern, 'priority': obj.get('priority', 10)})
                tasks.append({'task': 'collect_sensor_data', 'area': area, 'priority': obj.get('priority', 10)})
            elif obj['objective'] == 'track_target':
                target_id = obj.get('target_id')
                tasks.append({'task': 'initiate_tracking', 'target_id': target_id, 'priority': obj.get('priority', 10)})
                tasks.append({'task': 'handoff_tracking', 'target_id': target_id, 'priority': obj.get('priority', 10)})
            elif obj['objective'] == 'deploy_payload':
                payload = obj.get('payload')
                location = obj.get('location')
                tasks.append({'task': 'deploy_payload', 'payload': payload, 'location': location, 'priority': obj.get('priority', 10)})
            elif obj['objective'] == 'communicate':
                mode = obj.get('mode', 'secure')
                tasks.append({'task': 'select_comm_mode', 'mode': mode, 'priority': obj.get('priority', 10)})
                tasks.append({'task': 'transmit_data', 'priority': obj.get('priority', 10)})
            # Extend for other objectives as needed
        self.last_objectives = objectives
        self.last_tasks = tasks
        return tasks

    def get_last_decomposition(self) -> Optional[List[Dict[str, Any]]]:
        return self.last_tasks
