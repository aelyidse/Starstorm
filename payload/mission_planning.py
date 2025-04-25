from typing import Dict, Any, List, Optional
import heapq
import time

class PayloadMissionPlanner:
    """
    Plans and schedules payload operations based on mission objectives, constraints, and system state.
    Supports priority-based tasking, resource allocation, and temporal sequencing.
    """
    def __init__(self, payloads: Dict[str, Dict[str, Any]]):
        self.payloads = payloads
        self.task_queue: List[Any] = []  # Min-heap: (priority, start_time, task_dict)
        self.completed_tasks: List[Dict[str, Any]] = []

    def add_task(self, payload: str, action: str, start_time: float, duration: float, priority: int = 10, params: Optional[Dict[str, Any]] = None):
        task = {
            'payload': payload,
            'action': action,  # 'deploy' or 'retract' or custom
            'start_time': start_time,
            'end_time': start_time + duration,
            'priority': priority,
            'params': params or {},
            'status': 'pending'
        }
        heapq.heappush(self.task_queue, (priority, start_time, task))

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        if not self.task_queue:
            return None
        return self.task_queue[0][2]

    def advance_time(self, current_time: float):
        # Execute tasks whose time has come
        while self.task_queue and self.task_queue[0][2]['start_time'] <= current_time:
            _, _, task = heapq.heappop(self.task_queue)
            # Simulate execution (could call deployment controller here)
            task['status'] = 'completed'
            task['actual_start'] = current_time
            self.completed_tasks.append(task)

    def get_task_queue(self) -> List[Dict[str, Any]]:
        return [t[2] for t in sorted(self.task_queue)]

    def get_completed_tasks(self) -> List[Dict[str, Any]]:
        return self.completed_tasks
