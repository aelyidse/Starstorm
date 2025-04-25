from typing import Dict, Any, Optional, Callable
import time

class PIDController:
    """
    PID Controller with anti-windup protection for feedback control systems.
    Implements a complete PID control loop with configurable parameters.
    """
    def __init__(self, 
                 kp: float, 
                 ki: float, 
                 kd: float, 
                 setpoint: float,
                 output_limits: Optional[tuple] = None,
                 anti_windup: bool = True,
                 sample_time: float = 0.01):
        # Controller gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Setpoint
        self.setpoint = setpoint
        
        # Output limits for anti-windup
        self.output_limits = output_limits
        self.anti_windup = anti_windup
        
        # Controller memory
        self.prev_error = 0.0
        self.integral = 0.0
        
        # Timing
        self.sample_time = sample_time  # seconds
        self.last_time = time.time()
        
        # Performance metrics
        self.metrics = {
            'overshoot': 0.0,
            'rise_time': 0.0,
            'settling_time': 0.0,
            'steady_state_error': 0.0
        }
    
    def compute(self, process_variable: float) -> float:
        """
        Compute PID control action based on process variable and setpoint.
        
        Args:
            process_variable: Current value of the process variable
            
        Returns:
            Control output value
        """
        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Use fixed sample time if actual dt is too small
        if dt < self.sample_time:
            return self.prev_output if hasattr(self, 'prev_output') else 0.0
            
        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term (on measurement, not error, to avoid derivative kick)
        d_term = 0.0
        if dt > 0:  # Avoid division by zero
            d_term = -self.kd * (process_variable - self.prev_process_variable) / dt if hasattr(self, 'prev_process_variable') else 0.0
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits if specified
        if self.output_limits:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])
            
            # Anti-windup - back-calculate integral term
            if self.anti_windup and ((output == self.output_limits[1] and error > 0) or 
                                    (output == self.output_limits[0] and error < 0)):
                # Adjust integral to prevent further windup
                self.integral = (output - p_term - d_term) / self.ki if self.ki != 0 else 0.0
        
        # Store state for next iteration
        self.prev_error = error
        self.prev_process_variable = process_variable
        self.prev_output = output
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        if hasattr(self, 'prev_process_variable'):
            delattr(self, 'prev_process_variable')
        if hasattr(self, 'prev_output'):
            delattr(self, 'prev_output')
    
    def set_tunings(self, kp: float, ki: float, kd: float):
        """Update controller tuning parameters"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def set_setpoint(self, setpoint: float):
        """Update controller setpoint"""
        self.setpoint = setpoint
    
    def set_output_limits(self, min_output: float, max_output: float):
        """Set output limits for anti-windup protection"""
        if min_output > max_output:
            raise ValueError("Min output must be less than max output")
        self.output_limits = (min_output, max_output)