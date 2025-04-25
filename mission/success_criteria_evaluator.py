from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from enum import Enum
import time

class SuccessCriteriaResult(Enum):
    """Result of a success criteria evaluation"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    UNKNOWN = "unknown"

class MissionSuccessCriteriaEvaluator:
    """
    Evaluates mission success based on defined criteria and mission outcomes.
    Provides quantitative and qualitative assessment of mission performance.
    """
    def __init__(self):
        self.criteria_functions: Dict[str, Callable[[Dict[str, Any]], SuccessCriteriaResult]] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        
    def register_criteria(self, mission_type: str, 
                         criteria_func: Callable[[Dict[str, Any]], SuccessCriteriaResult]):
        """Register success criteria for a specific mission type"""
        self.criteria_functions[mission_type] = criteria_func
        
    def evaluate(self, mission_type: str, mission_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate mission success based on registered criteria
        
        Args:
            mission_type: Type of mission to evaluate
            mission_state: Final mission state including resources, events, and outcomes
            
        Returns:
            Evaluation result with success status and details
        """
        if mission_type not in self.criteria_functions:
            result = {
                "mission_type": mission_type,
                "timestamp": time.time(),
                "result": SuccessCriteriaResult.UNKNOWN,
                "details": "No success criteria registered for this mission type"
            }
        else:
            criteria_result = self.criteria_functions[mission_type](mission_state)
            result = {
                "mission_type": mission_type,
                "timestamp": time.time(),
                "result": criteria_result,
                "details": self._generate_details(mission_type, mission_state, criteria_result)
            }
            
        # Record evaluation in history
        self.evaluation_history.append(result)
        return result
    
    def _generate_details(self, mission_type: str, 
                         mission_state: Dict[str, Any], 
                         result: SuccessCriteriaResult) -> Dict[str, Any]:
        """Generate detailed evaluation information"""
        # Extract key metrics from mission state based on mission type
        details = {
            "resource_utilization": self._calculate_resource_utilization(mission_state),
            "objective_completion": self._calculate_objective_completion(mission_state),
            "time_efficiency": self._calculate_time_efficiency(mission_state),
            "anomalies": self._extract_anomalies(mission_state)
        }
        return details
    
    def _calculate_resource_utilization(self, mission_state: Dict[str, Any]) -> float:
        """Calculate resource utilization efficiency"""
        # Extract resource usage from mission state if available
        if "resource_usage" not in mission_state:
            return 0.0
            
        usage = mission_state.get("resource_usage", {})
        allocated = sum(usage.get("allocated", {}).values())
        consumed = sum(usage.get("consumed", {}).values())
        
        if allocated == 0:
            return 0.0
            
        return consumed / allocated
    
    def _calculate_objective_completion(self, mission_state: Dict[str, Any]) -> float:
        """Calculate percentage of objectives completed"""
        objectives = mission_state.get("objectives", [])
        if not objectives:
            return 0.0
            
        completed = sum(1 for obj in objectives if obj.get("status") == "completed")
        return completed / len(objectives)
    
    def _calculate_time_efficiency(self, mission_state: Dict[str, Any]) -> float:
        """Calculate time efficiency of mission execution"""
        planned_duration = mission_state.get("planned_duration", 0)
        actual_duration = mission_state.get("actual_duration", 0)
        
        if planned_duration == 0 or actual_duration == 0:
            return 0.0
            
        return planned_duration / actual_duration
    
    def _extract_anomalies(self, mission_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract anomalies from mission execution"""
        return mission_state.get("anomalies", [])
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of all evaluations"""
        return self.evaluation_history