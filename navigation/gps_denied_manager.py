from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import time

class GPSDeniedNavigationManager:
    """
    Manages multiple GPS-denied navigation systems and provides optimal position estimates.
    Dynamically selects and fuses navigation sources based on availability and confidence.
    """
    def __init__(self, navigation_systems: Optional[Dict[str, Any]] = None):
        self.navigation_systems = navigation_systems or {}
        self.system_weights = {}
        self.position = None
        self.orientation = None
        self.uncertainty = None
        self.history = []
        self.last_update = None
        
    def register_system(self, name: str, system: Any, initial_weight: float = 1.0) -> None:
        """Register a navigation system"""
        self.navigation_systems[name] = system
        self.system_weights[name] = initial_weight
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update system weights based on performance metrics"""
        for system, metric in performance_metrics.items():
            if system in self.system_weights:
                # Higher metric = better performance = higher weight
                self.system_weights[system] = max(0.1, min(10.0, metric))
                
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0"""
        total = sum(self.system_weights.values())
        if total > 0:
            for system in self.system_weights:
                self.system_weights[system] /= total
        
    def update(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update position using all available navigation systems"""
        timestamp = time.time()
        system_results = {}
        
        # Update each navigation system
        for name, system in self.navigation_systems.items():
            # Check if required sensor data is available
            if self._has_required_data(name, sensor_data):
                try:
                    # Get relevant sensor data for this system
                    system_data = self._extract_system_data(name, sensor_data)
                    
                    # Update the navigation system
                    result = system.update(**system_data)
                    system_results[name] = result
                except Exception as e:
                    print(f"Error updating {name}: {e}")
        
        # Fuse results if we have any
        if system_results:
            self._fuse_results(system_results)
            
        # Record history
        if self.position is not None:
            self.history.append({
                'position': self.position.copy(),
                'orientation': self.orientation.copy() if self.orientation is not None else None,
                'uncertainty': self.uncertainty,
                'active_systems': list(system_results.keys()),
                'timestamp': timestamp
            })
            
        self.last_update = timestamp
        
        return {
            'position': self.position.copy() if self.position is not None else None,
            'orientation': self.orientation.copy() if self.orientation is not None else None,
            'uncertainty': self.uncertainty,
            'active_systems': list(system_results.keys()),
            'timestamp': timestamp
        }
        
    def _has_required_data(self, system_name: str, sensor_data: Dict[str, Any]) -> bool:
        """Check if required sensor data is available for a system"""
        # Define required sensors for each system type
        requirements = {
            'terrain': ['camera', 'lidar'],
            'visual_inertial': ['camera', 'imu'],
            'magnetic': ['magnetometer'],
            'celestial': ['star_camera']
        }
        
        # Get requirements for this system
        for sys_type, required in requirements.items():
            if sys_type in system_name.lower():
                # Check if at least one required sensor is available
                return any(sensor in sensor_data for sensor in required)
                
        # Default: assume no special requirements
        return True
        
    def _extract_system_data(self, system_name: str, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant sensor data for a specific navigation system"""
        # Customize data extraction based on system type
        if 'terrain' in system_name.lower():
            return {'sensor_data': {k: v for k, v in sensor_data.items() 
                                  if k in ['camera', 'lidar', 'radar']}}
                                  
        elif 'visual_inertial' in system_name.lower():
            return {
                'image': sensor_data.get('camera'),
                'imu_data': sensor_data.get('imu', {})
            }
            
        elif 'magnetic' in system_name.lower():
            return {'magnetometer_data': sensor_data.get('magnetometer', np.array([]))}
            
        elif 'celestial' in system_name.lower():
            return {'star_camera_image': sensor_data.get('star_camera')}
            
        # Default: pass all sensor data
        return {'sensor_data': sensor_data}
        
    def _fuse_results(self, system_results: Dict[str, Dict[str, Any]]) -> None:
        """Fuse results from multiple navigation systems"""
        positions = []
        orientations = []
        weights = []
        
        # Collect valid results
        for name, result in system_results.items():
            if 'position' in result and result['position'] is not None:
                positions.append(result['position'])
                weights.append(self.system_weights.get(name, 1.0))
                
                if 'orientation' in result and result['orientation'] is not None:
                    orientations.append(result['orientation'])
        
        # Fuse positions if we have any
        if positions:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Weighted average of positions
            self.position = np.zeros(3)
            for i, pos in enumerate(positions):
                self.position += pos * weights[i]
                
            # Calculate uncertainty based on weighted standard deviation
            if len(positions) > 1:
                variance = np.zeros(3)
                for i, pos in enumerate(positions):
                    variance += weights[i] * (pos - self.position)**2
                self.uncertainty = float(np.sqrt(np.mean(variance)))
            else:
                # Single system: use its uncertainty if available
                for name, result in system_results.items():
                    if 'position' in result and result['position'] is not None:
                        self.uncertainty = result.get('uncertainty', 100.0)
                        break
            
            # Fuse orientations if we have any
            if orientations:
                # Simple average of quaternions
                # In practice, implement proper quaternion averaging
                self.orientation = np.mean(orientations, axis=0)
                self.orientation /= np.linalg.norm(self.orientation)