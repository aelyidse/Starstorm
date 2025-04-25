import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .sensor_fusion import NavigationSensorFusion

class PositionEstimator:
    """
    Position estimation system using sensor fusion.
    Combines data from GPS, IMU, star tracker, and quantum sensors
    to provide robust position estimates even in GPS-denied environments.
    """
    def __init__(self, sensor_weights: Optional[Dict[str, float]] = None):
        # Initialize default sensor weights
        self.sensor_weights = sensor_weights or {
            'gps': 0.7,
            'imu': 0.2,
            'star_tracker': 0.05,
            'quantum': 0.05
        }
        
        # Initialize state for EKF
        initial_state = {
            'pos': np.zeros(3),
            'vel': np.zeros(3),
            'quat': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            'bgx': 0.0, 'bgy': 0.0, 'bgz': 0.0  # Gyro biases
        }
        
        # Create EKF fusion engine
        self.ekf = NavigationSensorFusion(initial_state)
        
        # Position history for smoothing
        self.position_history = []
        self.max_history_length = 10
        
        # Uncertainty estimates
        self.position_uncertainty = np.ones(3) * 100.0  # High initial uncertainty
        
        # GPS-denied mode flag
        self.gps_denied_mode = False
    
    def update(self, sensor_data: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Update position estimate with new sensor data
        
        Args:
            sensor_data: Dictionary containing sensor readings
            dt: Time delta since last update
            
        Returns:
            Dictionary with position estimate and uncertainty
        """
        # Check GPS availability
        gps_available = 'gps' in sensor_data and sensor_data['gps'] is not None
        self.gps_denied_mode = not gps_available
        
        # Prediction step with IMU
        if 'imu' in sensor_data and sensor_data['imu'] is not None:
            self.ekf.predict(sensor_data['imu'], dt)
        
        # Update steps with available sensors
        if gps_available:
            self.ekf.update_gps(sensor_data['gps'])
            
        if 'star_tracker' in sensor_data and sensor_data['star_tracker'] is not None:
            self.ekf.update_star(sensor_data['star_tracker'])
            
        # Get current state estimate
        current_state = self.ekf.get_state()
        
        # Store position in history
        self.position_history.append(current_state['pos'].copy())
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # Calculate position uncertainty based on sensor availability
        self._update_uncertainty(gps_available)
        
        # Return position estimate with uncertainty
        return {
            'position': current_state['pos'].copy(),
            'velocity': current_state['vel'].copy(),
            'orientation': current_state['quat'].copy(),
            'uncertainty': self.position_uncertainty.copy(),
            'gps_denied': self.gps_denied_mode
        }
    
    def _update_uncertainty(self, gps_available: bool) -> None:
        """Update position uncertainty based on available sensors"""
        if gps_available:
            # GPS available - lower uncertainty
            self.position_uncertainty = np.maximum(self.position_uncertainty * 0.8, 1.0)
        else:
            # GPS denied - uncertainty grows over time
            self.position_uncertainty = np.minimum(self.position_uncertainty * 1.2, 1000.0)
    
    def get_smoothed_position(self) -> np.ndarray:
        """Get position estimate smoothed over recent history"""
        if not self.position_history:
            return np.zeros(3)
        
        # Simple weighted average with more weight on recent positions
        weights = np.linspace(0.5, 1.0, len(self.position_history))
        weights /= np.sum(weights)
        
        smoothed_pos = np.zeros(3)
        for i, pos in enumerate(self.position_history):
            smoothed_pos += pos * weights[i]
            
        return smoothed_pos