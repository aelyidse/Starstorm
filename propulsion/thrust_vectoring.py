from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import math

class ThrustVectoringSystem:
    """
    Thrust vectoring control system for precise attitude control of spacecraft.
    Supports gimbal control, differential throttling, and jet vanes.
    """
    def __init__(self, 
                 vectoring_type: str = "gimbal",
                 max_gimbal_angle_deg: float = 15.0,
                 response_time_s: float = 0.2,
                 num_engines: int = 1,
                 engine_positions: Optional[List[Tuple[float, float, float]]] = None):
        """
        Initialize thrust vectoring control system.
        
        Args:
            vectoring_type: Type of thrust vectoring ("gimbal", "differential", "vanes")
            max_gimbal_angle_deg: Maximum gimbal angle in degrees
            response_time_s: Actuator response time in seconds
            num_engines: Number of engines in the system
            engine_positions: List of (x,y,z) positions of each engine relative to COM
        """
        self.vectoring_type = vectoring_type
        self.max_gimbal_angle = max_gimbal_angle_deg
        self.response_time = response_time_s
        self.num_engines = num_engines
        
        # Default engine positions if not provided
        if engine_positions is None:
            if num_engines == 1:
                self.engine_positions = [(0.0, 0.0, 0.0)]  # Single center engine
            elif num_engines == 3:
                self.engine_positions = [
                    (0.0, 0.0, 0.0),    # Center
                    (1.0, 0.0, 0.0),    # Right
                    (-1.0, 0.0, 0.0)    # Left
                ]
            elif num_engines == 4:
                self.engine_positions = [
                    (1.0, 0.0, 1.0),    # Front-right
                    (-1.0, 0.0, 1.0),   # Front-left
                    (1.0, 0.0, -1.0),   # Rear-right
                    (-1.0, 0.0, -1.0)   # Rear-left
                ]
            else:
                self.engine_positions = [(0.0, 0.0, 0.0)] * num_engines
        else:
            self.engine_positions = engine_positions
            
        # Current state
        self.current_gimbal_angles = [(0.0, 0.0) for _ in range(num_engines)]  # (pitch, yaw) in degrees
        self.target_gimbal_angles = [(0.0, 0.0) for _ in range(num_engines)]
        self.throttle_levels = [1.0] * num_engines
        
        # Performance history
        self.command_history = []
        
    def set_gimbal_angle(self, engine_idx: int, pitch_deg: float, yaw_deg: float) -> bool:
        """
        Set target gimbal angle for a specific engine.
        
        Args:
            engine_idx: Engine index
            pitch_deg: Pitch angle in degrees
            yaw_deg: Yaw angle in degrees
            
        Returns:
            Success status
        """
        if engine_idx < 0 or engine_idx >= self.num_engines:
            return False
            
        # Limit to max gimbal angle
        pitch = max(-self.max_gimbal_angle, min(self.max_gimbal_angle, pitch_deg))
        yaw = max(-self.max_gimbal_angle, min(self.max_gimbal_angle, yaw_deg))
        
        self.target_gimbal_angles[engine_idx] = (pitch, yaw)
        self.command_history.append({
            'time': None,  # Would be set in a real system
            'type': 'gimbal',
            'engine': engine_idx,
            'pitch': pitch,
            'yaw': yaw
        })
        
        return True
        
    def set_differential_throttle(self, throttle_levels: List[float]) -> bool:
        """
        Set differential throttle levels for engines.
        
        Args:
            throttle_levels: List of throttle levels (0.0-1.0) for each engine
            
        Returns:
            Success status
        """
        if len(throttle_levels) != self.num_engines:
            return False
            
        # Limit throttle levels
        self.throttle_levels = [max(0.0, min(1.0, t)) for t in throttle_levels]
        
        self.command_history.append({
            'time': None,
            'type': 'differential',
            'throttle_levels': self.throttle_levels.copy()
        })
        
        return True
        
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update gimbal angles based on targets and response time.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Current state
        """
        # Simple first-order response model
        rate = dt / self.response_time
        
        for i in range(self.num_engines):
            current_pitch, current_yaw = self.current_gimbal_angles[i]
            target_pitch, target_yaw = self.target_gimbal_angles[i]
            
            # Move current angles toward target
            new_pitch = current_pitch + (target_pitch - current_pitch) * rate
            new_yaw = current_yaw + (target_yaw - current_yaw) * rate
            
            self.current_gimbal_angles[i] = (new_pitch, new_yaw)
            
        return self.get_state()
        
    def calculate_torque(self, thrust_N: float) -> Tuple[float, float, float]:
        """
        Calculate torque generated by current thrust vector configuration.
        
        Args:
            thrust_N: Current thrust in Newtons
            
        Returns:
            (roll, pitch, yaw) torque in N·m
        """
        if self.vectoring_type == "differential":
            # For differential throttling
            thrust_per_engine = thrust_N / self.num_engines
            
            roll_torque = 0.0
            pitch_torque = 0.0
            yaw_torque = 0.0
            
            for i in range(self.num_engines):
                pos_x, pos_y, pos_z = self.engine_positions[i]
                engine_thrust = thrust_per_engine * self.throttle_levels[i]
                
                # Simplified torque calculation
                roll_torque += engine_thrust * pos_y  # y-position creates roll
                pitch_torque += engine_thrust * pos_z  # z-position creates pitch
                yaw_torque += engine_thrust * pos_x  # x-position creates yaw
                
            return (roll_torque, pitch_torque, yaw_torque)
            
        else:  # Gimbal or vanes
            total_torque_x = 0.0
            total_torque_y = 0.0
            total_torque_z = 0.0
            
            thrust_per_engine = thrust_N / self.num_engines
            
            for i in range(self.num_engines):
                pos_x, pos_y, pos_z = self.engine_positions[i]
                pitch, yaw = self.current_gimbal_angles[i]
                
                # Convert angles to radians
                pitch_rad = math.radians(pitch)
                yaw_rad = math.radians(yaw)
                
                # Calculate thrust vector components
                thrust_x = thrust_per_engine * math.sin(yaw_rad) * math.cos(pitch_rad)
                thrust_y = thrust_per_engine * math.sin(pitch_rad)
                thrust_z = thrust_per_engine * math.cos(yaw_rad) * math.cos(pitch_rad)
                
                # Calculate torques
                torque_x = pos_y * thrust_z - pos_z * thrust_y
                torque_y = pos_z * thrust_x - pos_x * thrust_z
                torque_z = pos_x * thrust_y - pos_y * thrust_x
                
                total_torque_x += torque_x
                total_torque_y += torque_y
                total_torque_z += torque_z
                
            return (total_torque_x, total_torque_y, total_torque_z)
            
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the thrust vectoring system."""
        return {
            'vectoring_type': self.vectoring_type,
            'current_gimbal_angles': self.current_gimbal_angles,
            'target_gimbal_angles': self.target_gimbal_angles,
            'throttle_levels': self.throttle_levels,
            'num_engines': self.num_engines
        }
        
    def get_command_history(self) -> List[Dict[str, Any]]:
        """Get history of commands sent to the system."""
        return self.command_history
