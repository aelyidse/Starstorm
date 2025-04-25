from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import math

class ThrustProfileManager:
    """
    Manages thrust profiles for rocket engines.
    Supports constant, step, polynomial, and custom profiles.
    """
    def __init__(self, 
                 engine_max_thrust_N: float,
                 engine_min_throttle: float = 0.1,
                 engine_max_throttle: float = 1.0):
        """
        Initialize thrust profile manager.
        
        Args:
            engine_max_thrust_N: Maximum engine thrust in Newtons
            engine_min_throttle: Minimum throttle setting (0.0-1.0)
            engine_max_throttle: Maximum throttle setting (0.0-1.0)
        """
        self.engine_max_thrust_N = engine_max_thrust_N
        self.engine_min_throttle = engine_min_throttle
        self.engine_max_throttle = engine_max_throttle
        
        # Default profile is constant at max throttle
        self.profile_type = "constant"
        self.profile_params = {"throttle": 1.0}
        
        # For step and custom profiles
        self.time_points = [0.0]
        self.throttle_points = [1.0]
        
        # For polynomial profiles
        self.polynomial_coeffs = [1.0]
        
        # For tracking
        self.current_time = 0.0
        self.profile_duration = float('inf')  # Default to infinite duration
        
    def set_constant_profile(self, throttle: float) -> None:
        """
        Set a constant throttle profile.
        
        Args:
            throttle: Throttle level (0.0-1.0)
        """
        throttle = max(self.engine_min_throttle, min(self.engine_max_throttle, throttle))
        self.profile_type = "constant"
        self.profile_params = {"throttle": throttle}
        self.time_points = [0.0]
        self.throttle_points = [throttle]
        self.profile_duration = float('inf')
        
    def set_step_profile(self, time_points: List[float], throttle_points: List[float]) -> None:
        """
        Set a step profile with specific time points and throttle levels.
        
        Args:
            time_points: List of time points in seconds
            throttle_points: List of throttle levels (0.0-1.0) at each time point
        """
        if len(time_points) != len(throttle_points):
            raise ValueError("Time points and throttle points must have the same length")
            
        # Ensure time points are in ascending order
        if not all(time_points[i] <= time_points[i+1] for i in range(len(time_points)-1)):
            raise ValueError("Time points must be in ascending order")
            
        # Clamp throttle values to valid range
        throttle_points = [max(self.engine_min_throttle, min(self.engine_max_throttle, t)) 
                          for t in throttle_points]
        
        self.profile_type = "step"
        self.time_points = time_points
        self.throttle_points = throttle_points
        self.profile_duration = time_points[-1] if time_points else 0.0
        
    def set_polynomial_profile(self, coefficients: List[float], duration: float) -> None:
        """
        Set a polynomial throttle profile: throttle = sum(c[i] * t^i).
        
        Args:
            coefficients: Polynomial coefficients [c0, c1, c2, ...]
            duration: Profile duration in seconds
        """
        self.profile_type = "polynomial"
        self.polynomial_coeffs = coefficients
        self.profile_duration = duration
        
        # Generate sample points for visualization
        sample_times = np.linspace(0, duration, 20)
        self.time_points = sample_times.tolist()
        self.throttle_points = [self.evaluate_polynomial(t) for t in sample_times]
        
    def set_custom_profile(self, 
                          time_points: List[float], 
                          throttle_points: List[float], 
                          interpolation: str = "linear") -> None:
        """
        Set a custom profile with interpolation between points.
        
        Args:
            time_points: List of time points in seconds
            throttle_points: List of throttle levels at each time point
            interpolation: Interpolation method ("linear", "cubic")
        """
        if len(time_points) != len(throttle_points):
            raise ValueError("Time points and throttle points must have the same length")
            
        # Ensure time points are in ascending order
        if not all(time_points[i] <= time_points[i+1] for i in range(len(time_points)-1)):
            raise ValueError("Time points must be in ascending order")
            
        # Clamp throttle values to valid range
        throttle_points = [max(self.engine_min_throttle, min(self.engine_max_throttle, t)) 
                          for t in throttle_points]
        
        self.profile_type = "custom"
        self.profile_params = {"interpolation": interpolation}
        self.time_points = time_points
        self.throttle_points = throttle_points
        self.profile_duration = time_points[-1] if time_points else 0.0
        
    def evaluate_polynomial(self, t: float) -> float:
        """Evaluate polynomial at time t."""
        result = sum(c * (t ** i) for i, c in enumerate(self.polynomial_coeffs))
        return max(self.engine_min_throttle, min(self.engine_max_throttle, result))
        
    def get_throttle_at_time(self, time: float) -> float:
        """
        Get throttle level at specified time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Throttle level (0.0-1.0)
        """
        self.current_time = time
        
        if self.profile_type == "constant":
            return self.profile_params["throttle"]
            
        elif self.profile_type == "polynomial":
            if time > self.profile_duration:
                return self.engine_min_throttle
            return self.evaluate_polynomial(time)
            
        elif self.profile_type in ["step", "custom"]:
            # Handle time before first point or after last point
            if time <= self.time_points[0]:
                return self.throttle_points[0]
                
            if time >= self.time_points[-1]:
                return self.throttle_points[-1]
                
            # Find surrounding points and interpolate
            for i in range(1, len(self.time_points)):
                if time <= self.time_points[i]:
                    t0, throttle0 = self.time_points[i-1], self.throttle_points[i-1]
                    t1, throttle1 = self.time_points[i], self.throttle_points[i]
                    
                    # Linear interpolation
                    if self.profile_type == "step" or self.profile_params.get("interpolation") == "linear":
                        fraction = (time - t0) / (t1 - t0)
                        return throttle0 + fraction * (throttle1 - throttle0)
                    
                    # Cubic interpolation (simplified)
                    elif self.profile_params.get("interpolation") == "cubic":
                        # Need at least 4 points for cubic interpolation
                        if len(self.time_points) < 4:
                            # Fall back to linear
                            fraction = (time - t0) / (t1 - t0)
                            return throttle0 + fraction * (throttle1 - throttle0)
                        
                        # Simple cubic interpolation
                        x = (time - t0) / (t1 - t0)
                        return (2*x*x*x - 3*x*x + 1) * throttle0 + (x*x*x - 2*x*x + x) * (t1 - t0) * 0.5 + (-2*x*x*x + 3*x*x) * throttle1
        
        # Fallback
        return self.engine_min_throttle
        
    def get_thrust_at_time(self, time: float) -> float:
        """
        Get thrust in Newtons at specified time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Thrust in Newtons
        """
        throttle = self.get_throttle_at_time(time)
        return self.engine_max_thrust_N * throttle
        
    def get_profile_summary(self) -> Dict[str, Any]:
        """
        Get summary of current profile.
        
        Returns:
            Profile summary
        """
        return {
            "profile_type": self.profile_type,
            "duration": self.profile_duration,
            "min_throttle": min(self.throttle_points) if self.throttle_points else self.engine_min_throttle,
            "max_throttle": max(self.throttle_points) if self.throttle_points else self.engine_max_throttle,
            "current_time": self.current_time,
            "current_throttle": self.get_throttle_at_time(self.current_time),
            "current_thrust_N": self.get_thrust_at_time(self.current_time)
        }