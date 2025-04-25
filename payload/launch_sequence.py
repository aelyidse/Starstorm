from typing import Dict, Any, List, Optional, Callable, Tuple
import time
import datetime
from enum import Enum

class LaunchCondition(Enum):
    """Enum for launch condition status"""
    GO = "go"
    NO_GO = "no_go"
    HOLD = "hold"
    UNKNOWN = "unknown"

class LaunchSequenceManager:
    """
    Manages launch sequences for different deployment scenarios (vertical, air-launch, orbital, etc.).
    Supports stepwise sequencing, scenario-specific logic, and status reporting.
    """
    def __init__(self, scenario: str):
        self.scenario = scenario  # e.g., 'vertical', 'air_launch', 'orbital', etc.
        self.sequence: List[Dict[str, Any]] = self.build_sequence(scenario)
        self.current_step = 0
        self.status_log: List[Dict[str, Any]] = []
        self.completed = False
        self.aborted = False
        self.step_handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.error_handlers: Dict[str, Callable[[Dict[str, Any], Exception], Dict[str, Any]]] = {}
        self.pre_launch_checks_completed = False
        self.critical_systems = {
            "propulsion": False,
            "guidance": False,
            "communications": False,
            "power": False,
            "payload": False
        }
        
        # Hold and abort capabilities
        self.hold_active = False
        self.hold_reason = ""
        self.hold_start_time = None
        self.hold_duration = 0.0
        self.hold_history: List[Dict[str, Any]] = []
        
        # Launch window optimization
        self.launch_window_start = None
        self.launch_window_end = None
        self.optimal_launch_time = None
        self.launch_window_factors: Dict[str, float] = {}
        
        # Launch condition monitoring
        self.conditions: Dict[str, LaunchCondition] = {
            "weather": LaunchCondition.UNKNOWN,
            "range_safety": LaunchCondition.UNKNOWN,
            "telemetry": LaunchCondition.UNKNOWN,
            "ground_systems": LaunchCondition.UNKNOWN,
            "vehicle_systems": LaunchCondition.UNKNOWN
        }
        self.condition_thresholds: Dict[str, Dict[str, Any]] = {}
        self.condition_history: List[Dict[str, Any]] = []
        
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default step handlers for common launch steps"""
        self.step_handlers["init_checks"] = self._handle_init_checks
        self.step_handlers["fuel_pressurization"] = self._handle_fuel_pressurization
        self.step_handlers["engine_ignition"] = self._handle_engine_ignition
        self.step_handlers["liftoff"] = self._handle_liftoff
        self.step_handlers["stage_sep"] = self._handle_stage_separation
        self.step_handlers["orbit_insertion"] = self._handle_orbit_insertion
        self.step_handlers["release"] = self._handle_air_release
        self.step_handlers["climb"] = self._handle_climb
        self.step_handlers["deployment"] = self._handle_orbital_deployment
        self.step_handlers["system_activation"] = self._handle_system_activation
        self.step_handlers["mission_start"] = self._handle_mission_start

    def build_sequence(self, scenario: str) -> List[Dict[str, Any]]:
        # Define scenario-specific sequences
        if scenario == 'vertical':
            return [
                {'step': 'init_checks', 'desc': 'Initialize and check all systems', 'timeout': 120, 'critical': True},
                {'step': 'fuel_pressurization', 'desc': 'Pressurize fuel tanks', 'timeout': 60, 'critical': True},
                {'step': 'engine_ignition', 'desc': 'Ignite engines', 'timeout': 30, 'critical': True},
                {'step': 'liftoff', 'desc': 'Liftoff and initial ascent', 'timeout': 60, 'critical': True},
                {'step': 'stage_sep', 'desc': 'Stage separation', 'timeout': 45, 'critical': True},
                {'step': 'orbit_insertion', 'desc': 'Insert into target orbit', 'timeout': 180, 'critical': True},
            ]
        elif scenario == 'air_launch':
            return [
                {'step': 'init_checks', 'desc': 'Initialize and check all systems', 'timeout': 120, 'critical': True},
                {'step': 'release', 'desc': 'Release from carrier aircraft', 'timeout': 30, 'critical': True},
                {'step': 'engine_ignition', 'desc': 'Ignite engines', 'timeout': 30, 'critical': True},
                {'step': 'climb', 'desc': 'Climb to target altitude', 'timeout': 120, 'critical': True},
                {'step': 'orbit_insertion', 'desc': 'Insert into target orbit', 'timeout': 180, 'critical': True},
            ]
        elif scenario == 'orbital':
            return [
                {'step': 'init_checks', 'desc': 'Initialize and check all systems', 'timeout': 120, 'critical': True},
                {'step': 'deployment', 'desc': 'Deploy from orbital platform', 'timeout': 60, 'critical': True},
                {'step': 'system_activation', 'desc': 'Activate onboard systems', 'timeout': 90, 'critical': True},
                {'step': 'mission_start', 'desc': 'Begin mission operations', 'timeout': 60, 'critical': False},
            ]
        else:
            return [{'step': 'init_checks', 'desc': 'Initialize and check all systems', 'timeout': 120, 'critical': True}]

    def register_step_handler(self, step_name: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Register a custom handler for a specific launch step"""
        self.step_handlers[step_name] = handler

    def register_error_handler(self, step_name: str, handler: Callable[[Dict[str, Any], Exception], Dict[str, Any]]):
        """Register a custom error handler for a specific launch step"""
        self.error_handlers[step_name] = handler

    def hold_sequence(self, reason: str) -> Dict[str, Any]:
        """
        Place the launch sequence on hold
        
        Args:
            reason: Reason for the hold
            
        Returns:
            Hold status information
        """
        if self.completed or self.aborted:
            return {"status": "error", "message": "Cannot hold: sequence already completed or aborted"}
            
        self.hold_active = True
        self.hold_reason = reason
        self.hold_start_time = time.time()
        
        hold_status = {
            'step': 'hold',
            'desc': 'Launch sequence on hold',
            'status': 'hold',
            'reason': reason,
            'timestamp': self.hold_start_time,
            'at_step': self.current_step,
            'at_step_name': self.sequence[self.current_step]['step'] if self.current_step < len(self.sequence) else None
        }
        
        self.status_log.append(hold_status)
        self.hold_history.append(hold_status)
        
        return hold_status
        
    def resume_sequence(self) -> Dict[str, Any]:
        """
        Resume the launch sequence from hold
        
        Returns:
            Resume status information
        """
        if not self.hold_active:
            return {"status": "error", "message": "Cannot resume: sequence not on hold"}
            
        if self.completed or self.aborted:
            return {"status": "error", "message": "Cannot resume: sequence already completed or aborted"}
            
        resume_time = time.time()
        self.hold_duration += (resume_time - self.hold_start_time)
        self.hold_active = False
        
        resume_status = {
            'step': 'resume',
            'desc': 'Launch sequence resumed',
            'status': 'resumed',
            'hold_reason': self.hold_reason,
            'hold_duration': resume_time - self.hold_start_time,
            'total_hold_duration': self.hold_duration,
            'timestamp': resume_time,
            'at_step': self.current_step,
            'at_step_name': self.sequence[self.current_step]['step'] if self.current_step < len(self.sequence) else None
        }
        
        self.status_log.append(resume_status)
        self.hold_reason = ""
        
        return resume_status
    
    def set_launch_window(self, start_time: datetime.datetime, end_time: datetime.datetime, 
                         factors: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Set the launch window parameters
        
        Args:
            start_time: Launch window start time
            end_time: Launch window end time
            factors: Dictionary of factors and their weights for optimization
                    (e.g., {'weather': 0.4, 'visibility': 0.2, 'trajectory': 0.4})
                    
        Returns:
            Launch window information
        """
        if start_time >= end_time:
            return {"status": "error", "message": "Launch window start must be before end time"}
            
        self.launch_window_start = start_time
        self.launch_window_end = end_time
        self.launch_window_factors = factors or {
            "weather": 0.3,
            "visibility": 0.2,
            "trajectory": 0.3,
            "range_safety": 0.2
        }
        
        # Calculate window duration in minutes
        window_duration = (end_time - start_time).total_seconds() / 60
        
        window_info = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_minutes': window_duration,
            'optimization_factors': self.launch_window_factors
        }
        
        return window_info
    
    def optimize_launch_time(self, conditions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate the optimal launch time within the window based on conditions
        
        Args:
            conditions: Dictionary of condition forecasts over time
                      (e.g., {'weather': {'12:00': 0.8, '12:30': 0.9}, 'visibility': {...}})
                      
        Returns:
            Optimal launch time and score information
        """
        if not self.launch_window_start or not self.launch_window_end:
            return {"status": "error", "message": "Launch window not set"}
            
        # Generate time points at 5-minute intervals within the launch window
        time_points = []
        current_time = self.launch_window_start
        interval = datetime.timedelta(minutes=5)
        
        while current_time <= self.launch_window_end:
            time_points.append(current_time)
            current_time += interval
            
        # Calculate scores for each time point
        scores = []
        for time_point in time_points:
            time_str = time_point.strftime("%H:%M")
            score = 0.0
            
            for factor, weight in self.launch_window_factors.items():
                if factor in conditions and time_str in conditions[factor]:
                    score += conditions[factor][time_str] * weight
                    
            scores.append((time_point, score))
            
        # Find optimal time (highest score)
        if not scores:
            return {"status": "error", "message": "No valid time points in launch window"}
            
        optimal_time, optimal_score = max(scores, key=lambda x: x[1])
        self.optimal_launch_time = optimal_time
        
        return {
            'optimal_launch_time': optimal_time.isoformat(),
            'score': optimal_score,
            'all_scores': {t.isoformat(): s for t, s in scores},
            'window_start': self.launch_window_start.isoformat(),
            'window_end': self.launch_window_end.isoformat()
        }
    
    def set_condition_threshold(self, condition: str, threshold: Dict[str, Any]) -> None:
        """
        Set threshold values for a launch condition
        
        Args:
            condition: Condition name
            threshold: Threshold parameters (e.g., {'min': 0.7, 'max': 1.0, 'unit': 'km'})
        """
        self.condition_thresholds[condition] = threshold
    
    def update_condition(self, condition: str, value: Any, status: LaunchCondition = None) -> Dict[str, Any]:
        """
        Update a launch condition status
        
        Args:
            condition: Condition name
            value: Current value
            status: Optional explicit status override
            
        Returns:
            Updated condition information
        """
        if condition not in self.conditions:
            self.conditions[condition] = LaunchCondition.UNKNOWN
            
        # Determine status based on thresholds if not explicitly provided
        if status is None:
            if condition in self.condition_thresholds:
                threshold = self.condition_thresholds[condition]
                
                if 'min' in threshold and value < threshold['min']:
                    status = LaunchCondition.NO_GO
                elif 'max' in threshold and value > threshold['max']:
                    status = LaunchCondition.NO_GO
                else:
                    status = LaunchCondition.GO
            else:
                # No threshold defined, keep as unknown
                status = LaunchCondition.UNKNOWN
                
        # Update condition
        self.conditions[condition] = status
        
        # Record in history
        condition_update = {
            'condition': condition,
            'value': value,
            'status': status.value,
            'timestamp': time.time()
        }
        
        self.condition_history.append(condition_update)
        
        return condition_update
    
    def get_launch_status(self) -> Dict[str, Any]:
        """
        Get comprehensive launch status including conditions
        
        Returns:
            Launch status information
        """
        # Check if all monitored conditions are GO
        all_conditions_go = all(status == LaunchCondition.GO for status in self.conditions.values())
        
        # Count conditions by status
        condition_counts = {status.value: 0 for status in LaunchCondition}
        for status in self.conditions.values():
            condition_counts[status.value] += 1
            
        # Get current sequence status
        sequence_status = self.get_current_status()
        
        return {
            'sequence_status': sequence_status,
            'hold_active': self.hold_active,
            'hold_reason': self.hold_reason if self.hold_active else None,
            'hold_duration': self.hold_duration,
            'conditions': {k: v.value for k, v in self.conditions.items()},
            'condition_counts': condition_counts,
            'all_conditions_go': all_conditions_go,
            'launch_window': {
                'start': self.launch_window_start.isoformat() if self.launch_window_start else None,
                'end': self.launch_window_end.isoformat() if self.launch_window_end else None,
                'optimal_time': self.optimal_launch_time.isoformat() if self.optimal_launch_time else None
            }
        }
    
    def execute_next_step(self, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Execute the next step in the launch sequence with optional parameters"""
        # Check if sequence is on hold
        if self.hold_active:
            return {
                'status': 'hold',
                'message': f"Sequence on hold: {self.hold_reason}",
                'hold_duration': time.time() - self.hold_start_time
            }
            
        if self.completed or self.aborted or self.current_step >= len(self.sequence):
            self.completed = True
            return None
            
        step = self.sequence[self.current_step]
        step_name = step['step']
        step_params = params or {}
        
        # Record start time for timeout tracking
        start_time = time.time()
        status = {
            'step': step_name, 
            'desc': step['desc'], 
            'status': 'in_progress', 
            'start_time': start_time
        }
        
        try:
            # Check if we have a handler for this step
            if step_name in self.step_handlers:
                result = self.step_handlers[step_name](step_params)
                status.update(result)
            else:
                # Default behavior if no handler is registered
                time.sleep(0.1)  # Simulate execution delay
                status['details'] = "Default execution (no handler registered)"
            
            # Check for timeout
            if 'timeout' in step and (time.time() - start_time) > step['timeout']:
                status['status'] = 'timeout'
                if step.get('critical', False):
                    self.aborted = True
                    status['abort_reason'] = f"Critical step {step_name} timed out"
            else:
                status['status'] = 'completed'
                
        except Exception as e:
            # Handle errors
            status['status'] = 'failed'
            status['error'] = str(e)
            
            # Use custom error handler if available
            if step_name in self.error_handlers:
                error_result = self.error_handlers[step_name](step_params, e)
                status.update(error_result)
                
            # Abort if this is a critical step
            if step.get('critical', False):
                self.aborted = True
                status['abort_reason'] = f"Critical step {step_name} failed: {str(e)}"
        
        # Record end time and duration
        status['end_time'] = time.time()
        status['duration'] = status['end_time'] - start_time
        
        # Update status log
        self.status_log.append(status)
        
        # Move to next step if not aborted
        if not self.aborted:
            self.current_step += 1
            if self.current_step >= len(self.sequence):
                self.completed = True
                
        return status

    def execute_all(self, step_params: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Execute all remaining steps in the sequence"""
        results = []
        while not (self.completed or self.aborted):
            # Get params for this step if provided
            params = None
            if step_params and self.current_step < len(self.sequence):
                current_step_name = self.sequence[self.current_step]['step']
                params = step_params.get(current_step_name)
                
            result = self.execute_next_step(params)
            if result:
                results.append(result)
        return results

    def abort_sequence(self, reason: str) -> Dict[str, Any]:
        """Abort the launch sequence"""
        self.aborted = True
        abort_status = {
            'step': 'abort',
            'desc': 'Launch sequence aborted',
            'status': 'aborted',
            'reason': reason,
            'timestamp': time.time()
        }
        self.status_log.append(abort_status)
        return abort_status

    def reset_sequence(self) -> None:
        """Reset the launch sequence to the beginning"""
        self.current_step = 0
        self.completed = False
        self.aborted = False
        self.pre_launch_checks_completed = False
        self.critical_systems = {k: False for k in self.critical_systems}
        # Keep the status log for history

    def get_status_log(self) -> List[Dict[str, Any]]:
        """Get the complete status log"""
        return self.status_log

    def get_current_status(self) -> Dict[str, Any]:
        """Get the current status of the launch sequence"""
        if self.aborted:
            return {'status': 'aborted', 'current_step': self.current_step, 'reason': self.status_log[-1].get('reason', 'Unknown')}
        elif self.completed:
            return {'status': 'completed', 'steps_executed': len(self.status_log)}
        elif self.current_step < len(self.sequence):
            return {
                'status': 'in_progress',
                'current_step': self.current_step,
                'current_step_name': self.sequence[self.current_step]['step'],
                'progress_percentage': (self.current_step / len(self.sequence)) * 100
            }
        else:
            return {'status': 'unknown'}

    def is_completed(self) -> bool:
        """Check if the launch sequence is completed"""
        return self.completed

    def is_aborted(self) -> bool:
        """Check if the launch sequence was aborted"""
        return self.aborted

    # Step handler implementations
    def _handle_init_checks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization and system checks"""
        # Simulate system checks
        systems_checked = params.get('systems_to_check', list(self.critical_systems.keys()))
        check_results = {}
        
        for system in systems_checked:
            if system in self.critical_systems:
                # Simulate check result (would be actual system checks in real implementation)
                check_passed = params.get(f"{system}_status", True)
                self.critical_systems[system] = check_passed
                check_results[system] = check_passed
        
        # All critical systems must pass checks
        all_systems_go = all(self.critical_systems.values())
        self.pre_launch_checks_completed = all_systems_go
        
        return {
            'details': "System initialization and checks",
            'check_results': check_results,
            'all_systems_go': all_systems_go
        }

    def _handle_fuel_pressurization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fuel pressurization"""
        if not self.pre_launch_checks_completed:
            raise ValueError("Cannot pressurize fuel tanks: pre-launch checks not completed")
            
        target_pressure = params.get('target_pressure', 35.0)  # PSI
        current_pressure = params.get('initial_pressure', 0.0)
        pressurization_rate = params.get('pressurization_rate', 1.0)  # PSI per second
        
        # Simulate pressurization
        time_required = (target_pressure - current_pressure) / pressurization_rate
        time.sleep(min(time_required, 0.5))  # Simulate but cap at 0.5 seconds
        
        return {
            'details': "Fuel tank pressurization",
            'target_pressure': target_pressure,
            'achieved_pressure': target_pressure,
            'time_required': time_required
        }

    def _handle_engine_ignition(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle engine ignition"""
        if not self.critical_systems['propulsion']:
            raise ValueError("Cannot ignite engines: propulsion system check failed")
            
        engine_count = params.get('engine_count', 1)
        ignition_sequence = params.get('sequence', 'simultaneous')  # or 'sequential'
        
        # Simulate ignition
        time.sleep(0.2)
        
        return {
            'details': "Engine ignition",
            'engine_count': engine_count,
            'ignition_sequence': ignition_sequence,
            'all_engines_nominal': True
        }

    def _handle_liftoff(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle liftoff and initial ascent"""
        thrust_percentage = params.get('thrust_percentage', 100)
        target_velocity = params.get('target_velocity', 100)  # m/s
        
        # Simulate liftoff
        time.sleep(0.3)
        
        return {
            'details': "Liftoff and initial ascent",
            'thrust_percentage': thrust_percentage,
            'achieved_velocity': target_velocity,
            'trajectory': 'nominal'
        }

    def _handle_stage_separation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stage separation"""
        separation_altitude = params.get('separation_altitude', 100)  # km
        separation_velocity = params.get('separation_velocity', 2000)  # m/s
        
        # Simulate separation
        time.sleep(0.2)
        
        return {
            'details': "Stage separation",
            'separation_altitude': separation_altitude,
            'separation_velocity': separation_velocity,
            'separation_clean': True
        }

    def _handle_orbit_insertion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle orbit insertion"""
        target_apogee = params.get('target_apogee', 500)  # km
        target_perigee = params.get('target_perigee', 500)  # km
        target_inclination = params.get('target_inclination', 51.6)  # degrees
        
        # Simulate orbit insertion
        time.sleep(0.4)
        
        return {
            'details': "Orbit insertion",
            'achieved_apogee': target_apogee,
            'achieved_perigee': target_perigee,
            'achieved_inclination': target_inclination,
            'orbit_stable': True
        }

    def _handle_air_release(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle release from carrier aircraft"""
        release_altitude = params.get('release_altitude', 12)  # km
        release_velocity = params.get('release_velocity', 250)  # m/s
        
        # Simulate release
        time.sleep(0.2)
        
        return {
            'details': "Release from carrier aircraft",
            'release_altitude': release_altitude,
            'release_velocity': release_velocity,
            'release_clean': True
        }

    def _handle_climb(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle climb to target altitude"""
        target_altitude = params.get('target_altitude', 100)  # km
        climb_rate = params.get('climb_rate', 50)  # m/s
        
        # Simulate climb
        time.sleep(0.3)
        
        return {
            'details': "Climb to target altitude",
            'target_altitude': target_altitude,
            'achieved_altitude': target_altitude,
            'climb_rate': climb_rate,
            'trajectory': 'nominal'
        }

    def _handle_orbital_deployment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle deployment from orbital platform"""
        deployment_method = params.get('deployment_method', 'mechanical')
        relative_velocity = params.get('relative_velocity', 0.5)  # m/s
        
        # Simulate deployment
        time.sleep(0.2)
        
        return {
            'details': "Deployment from orbital platform",
            'deployment_method': deployment_method,
            'relative_velocity': relative_velocity,
            'deployment_clean': True
        }

    def _handle_system_activation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle activation of onboard systems"""
        systems_to_activate = params.get('systems', ['power', 'comms', 'payload', 'guidance'])
        activation_sequence = params.get('sequence', 'standard')
        
        # Simulate activation
        time.sleep(0.3)
        
        return {
            'details': "Activation of onboard systems",
            'systems_activated': systems_to_activate,
            'activation_sequence': activation_sequence,
            'all_systems_nominal': True
        }

    def _handle_mission_start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mission start"""
        mission_mode = params.get('mission_mode', 'standard')
        initial_objective = params.get('initial_objective', 'data_collection')
        
        # Simulate mission start
        time.sleep(0.1)
        
        return {
            'details': "Mission start",
            'mission_mode': mission_mode,
            'initial_objective': initial_objective,
            'mission_started': True
        }
