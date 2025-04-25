from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import math
from .thrust_vectoring import ThrustVectoringSystem

class ThrustController:
    """
    Manages thrust vectoring and control for spacecraft attitude control.
    Integrates with engine systems and provides high-level control interfaces.
    """
    def __init__(self, 
                 engine_system: Any,
                 vectoring_system: Optional[ThrustVectoringSystem] = None,
                 control_frequency_hz: float = 10.0):
        """
        Initialize thrust controller.
        
        Args:
            engine_system: Reference to engine system
            vectoring_system: Thrust vectoring system (or None to create default)
            control_frequency_hz: Control loop frequency in Hz
        """
        self.engine = engine_system
        self.vectoring = vectoring_system or ThrustVectoringSystem()
        self.control_dt = 1.0 / control_frequency_hz
        self.time_since_last_control = 0.0
        
        # PID controller gains for attitude control
        self.pid_gains = {
            'roll': {'kp': 0.5, 'ki': 0.1, 'kd': 0.2},
            'pitch': {'kp': 0.5, 'ki': 0.1, 'kd': 0.2},
            'yaw': {'kp': 0.5, 'ki': 0.1, 'kd': 0.2}
        }
        
        # Error tracking for PID
        self.errors = {
            'roll': {'current': 0.0, 'previous': 0.0, 'integral': 0.0},
            'pitch': {'current': 0.0, 'previous': 0.0, 'integral': 0.0},
            'yaw': {'current': 0.0, 'previous': 0.0, 'integral': 0.0}
        }
        
        # Target attitude
        self.target_attitude = (0.0, 0.0, 0.0)  # roll, pitch, yaw in degrees
        
    def set_target_attitude(self, roll_deg: float, pitch_deg: float, yaw_deg: float):
        """Set target attitude in degrees."""
        self.target_attitude = (roll_deg, pitch_deg, yaw_deg)
        
    def update_attitude_control(self, current_attitude: Tuple[float, float, float], dt: float) -> Dict[str, Any]:
        """
        Update attitude control based on current and target attitudes.
        
        Args:
            current_attitude: Current (roll, pitch, yaw) in degrees
            dt: Time step in seconds
            
        Returns:
            Control commands and status
        """
        # Update errors
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            self.errors[axis]['previous'] = self.errors[axis]['current']
            self.errors[axis]['current'] = self.target_attitude[i] - current_attitude[i]
            self.errors[axis]['integral'] += self.errors[axis]['current'] * dt
            
            # Anti-windup: limit integral term
            self.errors[axis]['integral'] = max(-10.0, min(10.0, self.errors[axis]['integral']))
        
        # Calculate PID outputs
        control_outputs = {}
        for axis in ['roll', 'pitch', 'yaw']:
            error = self.errors[axis]['current']
            error_integral = self.errors[axis]['integral']
            error_derivative = (self.errors[axis]['current'] - self.errors[axis]['previous']) / dt
            
            # PID formula
            control = (
                self.pid_gains[axis]['kp'] * error +
                self.pid_gains[axis]['ki'] * error_integral +
                self.pid_gains[axis]['kd'] * error_derivative
            )
            
            control_outputs[axis] = control
        
        # Apply controls to vectoring system
        if self.vectoring.vectoring_type == "gimbal":
            # For gimbal control, apply pitch and yaw
            for i in range(self.vectoring.num_engines):
                self.vectoring.set_gimbal_angle(
                    i, 
                    control_outputs['pitch'], 
                    control_outputs['yaw']
                )
        elif self.vectoring.vectoring_type == "differential":
            # For differential throttling
            # This is a simplified approach - real implementation would be more complex
            base_throttle = 0.7  # Base throttle level
            
            if self.vectoring.num_engines == 4:
                # Quad engine configuration
                throttles = [base_throttle] * 4
                
                # Roll control (opposite corners)
                roll_adj = control_outputs['roll'] * 0.3
                throttles[0] += roll_adj  # Front-right
                throttles[3] += roll_adj  # Rear-left
                throttles[1] -= roll_adj  # Front-left
                throttles[2] -= roll_adj  # Rear-right
                
                # Pitch control (front vs back)
                pitch_adj = control_outputs['pitch'] * 0.3
                throttles[0] += pitch_adj  # Front-right
                throttles[1] += pitch_adj  # Front-left
                throttles[2] -= pitch_adj  # Rear-right
                throttles[3] -= pitch_adj  # Rear-left
                
                # Yaw control (right vs left)
                yaw_adj = control_outputs['yaw'] * 0.3
                throttles[0] += yaw_adj  # Front-right
                throttles[2] += yaw_adj  # Rear-right
                throttles[1] -= yaw_adj  # Front-left
                throttles[3] -= yaw_adj  # Rear-left
                
                # Ensure throttles are within bounds
                throttles = [max(0.1, min(1.0, t)) for t in throttles]
                self.vectoring.set_differential_throttle(throttles)
        
        return {
            'control_outputs': control_outputs,
            'errors': self.errors,
            'vectoring_state': self.vectoring.get_state()
        }
        
    def update(self, current_attitude: Tuple[float, float, float], dt: float) -> Dict[str, Any]:
        """
        Main update function for thrust controller.
        
        Args:
            current_attitude: Current (roll, pitch, yaw) in degrees
            dt: Time step in seconds
            
        Returns:
            Controller state
        """
        self.time_since_last_control += dt
        
        # Run control loop at specified frequency
        control_output = None
        if self.time_since_last_control >= self.control_dt:
            control_output = self.update_attitude_control(current_attitude, self.time_since_last_control)
            self.time_since_last_control = 0.0
            
        # Update vectoring system physics
        vectoring_state = self.vectoring.update(dt)
        
        # Get current thrust from engine
        if hasattr(self.engine, 'compute_thrust'):
            current_thrust = self.engine.compute_thrust()
        else:
            current_thrust = 0.0
            
        # Calculate resulting torque
        torque = self.vectoring.calculate_torque(current_thrust)
        
        return {
            'torque': torque,
            'vectoring_state': vectoring_state,
            'control_output': control_output,
            'current_thrust': current_thrust
        }