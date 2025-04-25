from typing import Dict, Any, List, Optional, Tuple
import time
import math

class EngineSequenceController:
    """
    Controls engine startup and shutdown sequences with proper timing and safety checks.
    Manages valve operations, ignition, and propellant flow during critical transition phases.
    """
    def __init__(self, 
                 engine_type: str = "bipropellant",
                 pre_ignition_duration_s: float = 2.0,
                 ignition_duration_s: float = 0.5,
                 ramp_up_duration_s: float = 3.0,
                 shutdown_duration_s: float = 2.5,
                 safety_checks: bool = True):
        """
        Initialize engine sequence controller.
        
        Args:
            engine_type: Engine type (bipropellant, monopropellant, etc.)
            pre_ignition_duration_s: Pre-ignition phase duration in seconds
            ignition_duration_s: Ignition phase duration in seconds
            ramp_up_duration_s: Thrust ramp-up duration in seconds
            shutdown_duration_s: Shutdown sequence duration in seconds
            safety_checks: Whether to perform safety checks during sequences
        """
        self.engine_type = engine_type
        self.pre_ignition_duration = pre_ignition_duration_s
        self.ignition_duration = ignition_duration_s
        self.ramp_up_duration = ramp_up_duration_s
        self.shutdown_duration = shutdown_duration_s
        self.safety_checks = safety_checks
        
        # Sequence state
        self.current_phase = "idle"
        self.sequence_start_time = 0.0
        self.current_sequence_time = 0.0
        self.target_throttle = 0.0
        self.current_throttle = 0.0
        
        # Valve positions
        self.fuel_valve_position = 0.0
        self.oxidizer_valve_position = 0.0
        self.igniter_state = False
        
        # Sequence history
        self.sequence_log = []
        
    def start_engine(self, target_throttle_pct: float = 100.0) -> Dict[str, Any]:
        """
        Initiate engine startup sequence.
        
        Args:
            target_throttle_pct: Target throttle percentage (0-100)
            
        Returns:
            Startup sequence initial state
        """
        # Validate input
        if target_throttle_pct < 0 or target_throttle_pct > 100:
            return {"status": "error", "message": "Target throttle must be between 0-100%"}
            
        # Check if already in a sequence
        if self.current_phase != "idle" and self.current_phase != "shutdown_complete":
            return {"status": "error", "message": f"Engine already in {self.current_phase} phase"}
            
        # Initialize sequence
        self.sequence_start_time = time.time()
        self.current_sequence_time = 0.0
        self.target_throttle = target_throttle_pct / 100.0
        self.current_throttle = 0.0
        self.current_phase = "pre_ignition"
        
        # Set initial valve positions
        if self.engine_type == "bipropellant":
            # For bipropellant, start with oxidizer valve slightly open
            self.fuel_valve_position = 0.0
            self.oxidizer_valve_position = 0.05
        else:
            # For monopropellant, prepare fuel valve
            self.fuel_valve_position = 0.05
            self.oxidizer_valve_position = 0.0
            
        # Log sequence start
        log_entry = {
            "time": self.current_sequence_time,
            "phase": self.current_phase,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "igniter": False,
            "throttle": self.current_throttle
        }
        self.sequence_log.append(log_entry)
        
        return {
            "status": "started",
            "phase": self.current_phase,
            "target_throttle": self.target_throttle,
            "sequence_time": self.current_sequence_time
        }
        
    def shutdown_engine(self) -> Dict[str, Any]:
        """
        Initiate engine shutdown sequence.
        
        Returns:
            Shutdown sequence initial state
        """
        # Check if engine is running
        if self.current_phase == "idle" or self.current_phase == "shutdown_complete":
            return {"status": "error", "message": "Engine not running"}
            
        # Initialize shutdown sequence
        self.sequence_start_time = time.time()
        self.current_sequence_time = 0.0
        self.current_phase = "shutdown_initiated"
        
        # Log sequence start
        log_entry = {
            "time": self.current_sequence_time,
            "phase": self.current_phase,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "igniter": self.igniter_state,
            "throttle": self.current_throttle
        }
        self.sequence_log.append(log_entry)
        
        return {
            "status": "shutdown_initiated",
            "phase": self.current_phase,
            "current_throttle": self.current_throttle,
            "sequence_time": self.current_sequence_time
        }
        
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update engine sequence controller.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Current sequence state
        """
        # Update sequence time
        self.current_sequence_time += dt
        
        # Process current phase
        if self.current_phase == "pre_ignition":
            return self._update_pre_ignition(dt)
        elif self.current_phase == "ignition":
            return self._update_ignition(dt)
        elif self.current_phase == "ramp_up":
            return self._update_ramp_up(dt)
        elif self.current_phase == "steady_state":
            return self._update_steady_state(dt)
        elif self.current_phase == "shutdown_initiated":
            return self._update_shutdown_initiated(dt)
        elif self.current_phase == "shutdown_in_progress":
            return self._update_shutdown_in_progress(dt)
        elif self.current_phase == "shutdown_complete":
            return self._update_shutdown_complete(dt)
        else:
            return {"status": "idle"}
            
    def _update_pre_ignition(self, dt: float) -> Dict[str, Any]:
        """Update pre-ignition phase."""
        # Check if pre-ignition phase is complete
        if self.current_sequence_time >= self.pre_ignition_duration:
            # Transition to ignition phase
            self.current_phase = "ignition"
            self.igniter_state = True
            
            # For bipropellant, open fuel valve slightly
            if self.engine_type == "bipropellant":
                self.fuel_valve_position = 0.1
                self.oxidizer_valve_position = 0.15
            else:
                self.fuel_valve_position = 0.15
        else:
            # Gradually open valves during pre-ignition
            progress = self.current_sequence_time / self.pre_ignition_duration
            
            if self.engine_type == "bipropellant":
                self.oxidizer_valve_position = 0.05 + progress * 0.1
            else:
                self.fuel_valve_position = 0.05 + progress * 0.1
                
        # Log state
        self._log_state()
        
        return {
            "status": "in_progress",
            "phase": self.current_phase,
            "progress": self.current_sequence_time / self.pre_ignition_duration,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "igniter": self.igniter_state,
            "throttle": self.current_throttle
        }
        
    def _update_ignition(self, dt: float) -> Dict[str, Any]:
        """Update ignition phase."""
        # Check if ignition phase is complete
        if self.current_sequence_time >= (self.pre_ignition_duration + self.ignition_duration):
            # Transition to ramp-up phase
            self.current_phase = "ramp_up"
            self.current_throttle = 0.1  # Start at 10% throttle
        else:
            # During ignition, maintain valve positions but igniter is on
            self.igniter_state = True
            
            # Small initial throttle
            self.current_throttle = 0.05
                
        # Log state
        self._log_state()
        
        return {
            "status": "in_progress",
            "phase": self.current_phase,
            "progress": (self.current_sequence_time - self.pre_ignition_duration) / self.ignition_duration,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "igniter": self.igniter_state,
            "throttle": self.current_throttle
        }
        
    def _update_ramp_up(self, dt: float) -> Dict[str, Any]:
        """Update thrust ramp-up phase."""
        ramp_start_time = self.pre_ignition_duration + self.ignition_duration
        
        # Check if ramp-up phase is complete
        if self.current_sequence_time >= (ramp_start_time + self.ramp_up_duration):
            # Transition to steady state
            self.current_phase = "steady_state"
            self.current_throttle = self.target_throttle
            
            # Fully open valves proportional to throttle
            if self.engine_type == "bipropellant":
                self.fuel_valve_position = self.target_throttle
                self.oxidizer_valve_position = self.target_throttle
            else:
                self.fuel_valve_position = self.target_throttle
                self.oxidizer_valve_position = 0.0
                
            # Turn off igniter once main combustion is established
            self.igniter_state = False
        else:
            # Gradually increase throttle during ramp-up
            progress = (self.current_sequence_time - ramp_start_time) / self.ramp_up_duration
            self.current_throttle = 0.1 + progress * (self.target_throttle - 0.1)
            
            # Adjust valve positions based on throttle
            if self.engine_type == "bipropellant":
                self.fuel_valve_position = self.current_throttle
                self.oxidizer_valve_position = self.current_throttle
            else:
                self.fuel_valve_position = self.current_throttle
                
            # Keep igniter on during early ramp-up, then turn off
            if progress > 0.7:
                self.igniter_state = False
                
        # Log state
        self._log_state()
        
        return {
            "status": "in_progress",
            "phase": self.current_phase,
            "progress": (self.current_sequence_time - ramp_start_time) / self.ramp_up_duration,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "igniter": self.igniter_state,
            "throttle": self.current_throttle
        }
        
    def _update_steady_state(self, dt: float) -> Dict[str, Any]:
        """Update steady state operation."""
        # In steady state, maintain target throttle
        self.current_throttle = self.target_throttle
        
        # Log state periodically (not every update)
        if len(self.sequence_log) % 10 == 0:
            self._log_state()
        
        return {
            "status": "running",
            "phase": self.current_phase,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "igniter": self.igniter_state,
            "throttle": self.current_throttle
        }
        
    def _update_shutdown_initiated(self, dt: float) -> Dict[str, Any]:
        """Update shutdown initiated phase."""
        # Transition to shutdown in progress
        self.current_phase = "shutdown_in_progress"
        
        # Log state
        self._log_state()
        
        return {
            "status": "in_progress",
            "phase": self.current_phase,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "throttle": self.current_throttle
        }
        
    def _update_shutdown_in_progress(self, dt: float) -> Dict[str, Any]:
        """Update shutdown in progress phase."""
        # Check if shutdown is complete
        if self.current_sequence_time >= self.shutdown_duration:
            # Transition to shutdown complete
            self.current_phase = "shutdown_complete"
            self.current_throttle = 0.0
            self.fuel_valve_position = 0.0
            self.oxidizer_valve_position = 0.0
            self.igniter_state = False
        else:
            # Gradually decrease throttle and close valves
            progress = self.current_sequence_time / self.shutdown_duration
            
            # For bipropellant engines, close fuel valve faster than oxidizer
            # to ensure complete combustion of remaining fuel
            if self.engine_type == "bipropellant":
                self.current_throttle = self.target_throttle * (1.0 - progress)
                self.fuel_valve_position = self.target_throttle * (1.0 - progress * 1.2)
                self.oxidizer_valve_position = self.target_throttle * (1.0 - progress * 0.8)
                
                # Ensure valves don't go negative
                self.fuel_valve_position = max(0.0, self.fuel_valve_position)
                self.oxidizer_valve_position = max(0.0, self.oxidizer_valve_position)
            else:
                # For monopropellant, simply ramp down
                self.current_throttle = self.target_throttle * (1.0 - progress)
                self.fuel_valve_position = self.target_throttle * (1.0 - progress)
                
        # Log state
        self._log_state()
        
        return {
            "status": "in_progress",
            "phase": self.current_phase,
            "progress": self.current_sequence_time / self.shutdown_duration,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "throttle": self.current_throttle
        }
        
    def _update_shutdown_complete(self, dt: float) -> Dict[str, Any]:
        """Update shutdown complete phase."""
        # Reset to idle after a short delay
        if self.current_sequence_time >= (self.shutdown_duration + 5.0):
            self.current_phase = "idle"
            
        return {
            "status": "complete",
            "phase": self.current_phase,
            "fuel_valve": 0.0,
            "oxidizer_valve": 0.0,
            "throttle": 0.0
        }
        
    def _log_state(self):
        """Log current state to sequence history."""
        log_entry = {
            "time": self.current_sequence_time,
            "phase": self.current_phase,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "igniter": self.igniter_state,
            "throttle": self.current_throttle
        }
        self.sequence_log.append(log_entry)
        
    def get_sequence_log(self) -> List[Dict[str, Any]]:
        """Get sequence log history."""
        return self.sequence_log
        
    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        return {
            "phase": self.current_phase,
            "sequence_time": self.current_sequence_time,
            "fuel_valve": self.fuel_valve_position,
            "oxidizer_valve": self.oxidizer_valve_position,
            "igniter": self.igniter_state,
            "throttle": self.current_throttle,
            "target_throttle": self.target_throttle
        }