from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import time
import logging
from enum import Enum

class MissionPhaseStatus(Enum):
    """Status of a mission phase execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ABORTED = "aborted"

class BranchCondition:
    """Represents a condition that determines which branch to take in a mission scenario"""
    
    def __init__(self, 
                condition_func: Callable[[Dict[str, Any]], bool], 
                description: str):
        """
        Initialize a branch condition.
        
        Args:
            condition_func: Function that evaluates to True/False based on system state
            description: Human-readable description of the condition
        """
        self.condition_func = condition_func
        self.description = description
        
    def evaluate(self, system_state: Dict[str, Any]) -> bool:
        """
        Evaluate the condition based on current system state.
        
        Args:
            system_state: Current system state
            
        Returns:
            True if condition is met, False otherwise
        """
        return self.condition_func(system_state)

class MissionPhase:
    """Represents a single phase in a mission scenario"""
    
    def __init__(self, 
                 name: str,
                 actions: List[Callable[[Dict[str, Any]], Dict[str, Any]]],
                 success_criteria: Optional[BranchCondition] = None,
                 timeout_seconds: Optional[float] = None,
                 required: bool = True):
        """
        Initialize a mission phase.
        
        Args:
            name: Phase name
            actions: List of action functions to execute
            success_criteria: Optional condition that determines if phase was successful
            timeout_seconds: Optional timeout for phase execution
            required: Whether this phase is required for mission success
        """
        self.name = name
        self.actions = actions
        self.success_criteria = success_criteria
        self.timeout_seconds = timeout_seconds
        self.required = required
        self.status = MissionPhaseStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Dict[str, Any] = {}
        
    def execute(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the phase actions.
        
        Args:
            system_state: Current system state
            
        Returns:
            Result of phase execution
        """
        self.start_time = time.time()
        self.status = MissionPhaseStatus.IN_PROGRESS
        
        result = {"phase": self.name, "status": self.status.value}
        
        try:
            # Execute each action in sequence
            for i, action in enumerate(self.actions):
                action_result = action(system_state)
                result[f"action_{i}"] = action_result
                
                # Update system state with action results
                system_state.update(action_result.get("state_updates", {}))
                
                # Check if we need to abort after this action
                if action_result.get("abort", False):
                    self.status = MissionPhaseStatus.ABORTED
                    result["aborted_after_action"] = i
                    break
            
            # Check success criteria if provided and not aborted
            if self.status != MissionPhaseStatus.ABORTED and self.success_criteria:
                if self.success_criteria.evaluate(system_state):
                    self.status = MissionPhaseStatus.COMPLETED
                else:
                    self.status = MissionPhaseStatus.FAILED
            elif self.status != MissionPhaseStatus.ABORTED:
                # No success criteria provided, assume success
                self.status = MissionPhaseStatus.COMPLETED
                
        except Exception as e:
            self.status = MissionPhaseStatus.FAILED
            result["error"] = str(e)
            
        # Check timeout
        if self.timeout_seconds and time.time() - self.start_time > self.timeout_seconds:
            self.status = MissionPhaseStatus.FAILED
            result["timeout"] = True
            
        self.end_time = time.time()
        result["status"] = self.status.value
        result["duration"] = self.end_time - self.start_time
        
        self.result = result
        return result

class ConditionalBranch:
    """Represents a conditional branch in a mission scenario"""
    
    def __init__(self, 
                 condition: BranchCondition,
                 phases: List[MissionPhase],
                 else_phases: Optional[List[MissionPhase]] = None):
        """
        Initialize a conditional branch.
        
        Args:
            condition: Condition that determines which branch to take
            phases: Phases to execute if condition is True
            else_phases: Phases to execute if condition is False
        """
        self.condition = condition
        self.phases = phases
        self.else_phases = else_phases or []
        self.taken_branch: Optional[str] = None
        
    def evaluate(self, system_state: Dict[str, Any]) -> List[MissionPhase]:
        """
        Evaluate the condition and return the appropriate branch.
        
        Args:
            system_state: Current system state
            
        Returns:
            List of phases to execute
        """
        if self.condition.evaluate(system_state):
            self.taken_branch = "if"
            return self.phases
        else:
            self.taken_branch = "else"
            return self.else_phases

class MissionConstraint:
    """
    Represents a constraint that must be satisfied for a mission to be valid.
    """
    def __init__(self, 
                 constraint_func: Callable[[Dict[str, Any]], bool],
                 description: str,
                 error_message: str):
        """
        Initialize a mission constraint.
        
        Args:
            constraint_func: Function that evaluates to True if constraint is satisfied
            description: Human-readable description of the constraint
            error_message: Error message to display if constraint is violated
        """
        self.constraint_func = constraint_func
        self.description = description
        self.error_message = error_message
        
    def validate(self, mission_config: Dict[str, Any]) -> bool:
        """
        Validate if the mission configuration satisfies this constraint.
        
        Args:
            mission_config: Mission configuration to validate
            
        Returns:
            True if constraint is satisfied, False otherwise
        """
        return self.constraint_func(mission_config)


class MissionConstraintValidator:
    """
    Validates mission configurations against a set of constraints.
    Ensures missions meet required criteria before execution.
    """
    def __init__(self):
        self.constraints: List[MissionConstraint] = []
        self.validation_results: List[Dict[str, Any]] = []
        
    def add_constraint(self, constraint: MissionConstraint) -> None:
        """
        Add a constraint to the validator.
        
        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)
        
    def validate_mission(self, mission_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a mission configuration against all registered constraints.
        
        Args:
            mission_config: Mission configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        self.validation_results = []
        error_messages = []
        
        for constraint in self.constraints:
            is_valid = constraint.validate(mission_config)
            
            result = {
                "constraint": constraint.description,
                "is_valid": is_valid
            }
            
            if not is_valid:
                error_messages.append(constraint.error_message)
                result["error"] = constraint.error_message
                
            self.validation_results.append(result)
            
        return len(error_messages) == 0, error_messages
    
    def get_validation_results(self) -> List[Dict[str, Any]]:
        """
        Get the results of the last validation.
        
        Returns:
            List of validation results
        """
        return self.validation_results


# Update MissionScenarioExecutor to include constraint validation
class MissionScenarioExecutor:
    """
    Executes mission scenarios with support for conditional branching.
    Allows for dynamic mission paths based on system state and conditions.
    """
    
    def __init__(self, 
                 mission_name: str,
                 system_state_provider: Callable[[], Dict[str, Any]],
                 logger: Optional[logging.Logger] = None,
                 constraint_validator: Optional[MissionConstraintValidator] = None):
        """
        Initialize the mission scenario executor.
        
        Args:
            mission_name: Name of the mission
            system_state_provider: Function that returns the current system state
            logger: Optional logger for mission execution
            constraint_validator: Optional validator for mission constraints
        """
        self.mission_name = mission_name
        self.system_state_provider = system_state_provider
        self.logger = logger or logging.getLogger(__name__)
        self.constraint_validator = constraint_validator
        
        self.phases: List[Union[MissionPhase, ConditionalBranch]] = []
        self.execution_path: List[Dict[str, Any]] = []
        self.mission_status = MissionPhaseStatus.PENDING
        self.current_phase_index = 0
        self.mission_config: Dict[str, Any] = {"name": mission_name}
        
    def add_phase(self, phase: MissionPhase) -> None:
        """
        Add a phase to the mission scenario.
        
        Args:
            phase: Phase to add
        """
        self.phases.append(phase)
        
    def add_conditional_branch(self, branch: ConditionalBranch) -> None:
        """
        Add a conditional branch to the mission scenario.
        
        Args:
            branch: Conditional branch to add
        """
        self.phases.append(branch)
        
    def set_mission_config(self, config: Dict[str, Any]) -> None:
        """
        Set the mission configuration.
        
        Args:
            config: Mission configuration
        """
        self.mission_config.update(config)
        
    def validate_constraints(self) -> Tuple[bool, List[str]]:
        """
        Validate mission against constraints.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not self.constraint_validator:
            return True, []
            
        return self.constraint_validator.validate_mission(self.mission_config)
    
    def execute_mission(self) -> Dict[str, Any]:
        """
        Execute the entire mission scenario.
        
        Returns:
            Mission execution results
        """
        self.logger.info(f"Starting mission: {self.mission_name}")
        
        # Validate mission constraints before execution
        if self.constraint_validator:
            is_valid, error_messages = self.validate_constraints()
            if not is_valid:
                self.logger.error(f"Mission validation failed: {', '.join(error_messages)}")
                return {
                    "mission_name": self.mission_name,
                    "status": MissionPhaseStatus.FAILED.value,
                    "validation_errors": error_messages,
                    "message": "Mission failed constraint validation"
                }
        
        start_time = time.time()
        
        # Reset execution state
        self.execution_path = []
        self.mission_status = MissionPhaseStatus.IN_PROGRESS
        self.current_phase_index = 0
        
        # Flatten phases by evaluating conditionals
        flattened_phases = self._flatten_phases()
        
        # Execute each phase in sequence
        for i, phase in enumerate(flattened_phases):
            self.current_phase_index = i
            self.logger.info(f"Executing phase {i+1}/{len(flattened_phases)}: {phase.name}")
            
            # Get current system state
            system_state = self.system_state_provider()
            
            # Execute the phase
            result = phase.execute(system_state)
            
            # Record execution
            self.execution_path.append({
                "phase": phase.name,
                "status": phase.status.value,
                "result": result
            })
            
            # Check if we need to abort the mission
            if phase.status in [MissionPhaseStatus.FAILED, MissionPhaseStatus.ABORTED] and phase.required:
                self.mission_status = MissionPhaseStatus.ABORTED
                self.logger.warning(f"Mission aborted due to failure in required phase: {phase.name}")
                break
                
        # Determine overall mission status if not already aborted
        if self.mission_status != MissionPhaseStatus.ABORTED:
            required_phases = [p for p in flattened_phases if p.required]
            if all(p.status == MissionPhaseStatus.COMPLETED for p in required_phases):
                self.mission_status = MissionPhaseStatus.COMPLETED
                self.logger.info("Mission completed successfully")
            else:
                self.mission_status = MissionPhaseStatus.FAILED
                self.logger.warning("Mission failed due to incomplete required phases")
                
        # Prepare mission summary
        end_time = time.time()
        mission_summary = {
            "mission_name": self.mission_name,
            "status": self.mission_status.value,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "phases_executed": len(self.execution_path),
            "execution_path": self.execution_path
        }
        
        return mission_summary
    
    def _flatten_phases(self) -> List[MissionPhase]:
        """
        Flatten the mission phases by evaluating conditionals.
        
        Returns:
            Flattened list of phases to execute
        """
        flattened = []
        system_state = self.system_state_provider()
        
        for item in self.phases:
            if isinstance(item, MissionPhase):
                flattened.append(item)
            elif isinstance(item, ConditionalBranch):
                # Evaluate the condition and add the appropriate phases
                branch_phases = item.evaluate(system_state)
                flattened.extend(branch_phases)
                
                # Log which branch was taken
                self.logger.info(f"Conditional branch taken: {item.taken_branch} - {item.condition.description}")
                
        return flattened
    
    def get_execution_status(self) -> Dict[str, Any]:
        """
        Get the current execution status.
        
        Returns:
            Current execution status
        """
        return {
            "mission_name": self.mission_name,
            "status": self.mission_status.value,
            "current_phase": self.current_phase_index,
            "total_phases": len(self._flatten_phases()),
            "execution_path": self.execution_path
        }
    
    def abort_mission(self) -> Dict[str, Any]:
        """
        Abort the mission execution.
        
        Returns:
            Abort status
        """
        self.mission_status = MissionPhaseStatus.ABORTED
        self.logger.warning("Mission manually aborted")
        
        return {
            "mission_name": self.mission_name,
            "status": self.mission_status.value,
            "aborted_at_phase": self.current_phase_index,
            "message": "Mission manually aborted"
        }