from typing import Dict, Any, List, Optional
import copy

class RealTimeMissionReplanner:
    """
    Enables real-time mission replanning in response to dynamic events, failures, or new intelligence.
    Supports objective reprioritization, task adjustment, and plan versioning.
    """
    def __init__(self):
        self.current_plan: Optional[List[Dict[str, Any]]] = None
        self.plan_history: List[List[Dict[str, Any]]] = []
        self.replan_log: List[Dict[str, Any]] = []

    def set_initial_plan(self, plan: List[Dict[str, Any]]):
        self.current_plan = copy.deepcopy(plan)
        self.plan_history.append(copy.deepcopy(plan))

    def replan(self, event: Dict[str, Any], objectives: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        event: dict describing the trigger (e.g., {'type': 'failure', 'component': 'sensor1'})
        objectives: optional new/revised objectives
        Returns: new plan (list of tasks)
        """
        new_plan = copy.deepcopy(self.current_plan) if self.current_plan else []
        # Example: reprioritize or remove tasks based on event
        if event['type'] == 'failure' and 'component' in event:
            # Remove or replace tasks involving failed component
            new_plan = [t for t in new_plan if event['component'] not in str(t)]
        elif event['type'] == 'new_intel' and objectives:
            # Add new tasks for updated objectives
            for obj in objectives:
                new_plan.append({'task': 'address_new_objective', **obj})
        # Log and update
        self.plan_history.append(copy.deepcopy(new_plan))
        self.replan_log.append({'event': event, 'resulting_plan': copy.deepcopy(new_plan)})
        self.current_plan = new_plan
        return new_plan

    def get_current_plan(self) -> Optional[List[Dict[str, Any]]]:
        return self.current_plan

    def get_plan_history(self) -> List[List[Dict[str, Any]]]:
        return self.plan_history

    def get_replan_log(self) -> List[Dict[str, Any]]:
        return self.replan_log
