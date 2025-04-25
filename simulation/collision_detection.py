import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class CollisionDetector:
    """
    Detects potential collisions between objects in space.
    Supports various collision detection algorithms and optimization techniques.
    """
    def __init__(self, safety_margin_m: float = 100.0):
        self.safety_margin_m = safety_margin_m
        self.collision_pairs = []
        
    def detect_sphere_collision(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """
        Detect collision between two spherical objects using simple distance check.
        
        Args:
            obj1: First object with 'position' (x,y,z) and 'radius' keys
            obj2: Second object with 'position' (x,y,z) and 'radius' keys
            
        Returns:
            True if collision detected, False otherwise
        """
        pos1 = np.array(obj1['position'])
        pos2 = np.array(obj2['position'])
        
        # Calculate distance between objects
        distance = np.linalg.norm(pos2 - pos1)
        
        # Check if distance is less than sum of radii plus safety margin
        collision_threshold = obj1['radius'] + obj2['radius'] + self.safety_margin_m
        return distance < collision_threshold
    
    def detect_aabb_collision(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """
        Detect collision using Axis-Aligned Bounding Boxes (AABB).
        
        Args:
            obj1: First object with 'min_bound' and 'max_bound' (x,y,z) keys
            obj2: Second object with 'min_bound' and 'max_bound' (x,y,z) keys
            
        Returns:
            True if collision detected, False otherwise
        """
        # Check for overlap in all three axes
        return (obj1['min_bound'][0] <= obj2['max_bound'][0] and obj1['max_bound'][0] >= obj2['min_bound'][0] and
                obj1['min_bound'][1] <= obj2['max_bound'][1] and obj1['max_bound'][1] >= obj2['min_bound'][1] and
                obj1['min_bound'][2] <= obj2['max_bound'][2] and obj1['max_bound'][2] >= obj2['min_bound'][2])
    
    def predict_closest_approach(self, obj1: Dict[str, Any], obj2: Dict[str, Any], 
                                time_horizon_s: float = 3600.0) -> Tuple[float, float]:
        """
        Predict time and distance of closest approach between two objects.
        
        Args:
            obj1: First object with 'position' and 'velocity' (x,y,z) keys
            obj2: Second object with 'position' and 'velocity' (x,y,z) keys
            time_horizon_s: Time horizon for prediction in seconds
            
        Returns:
            Tuple of (time_to_closest_approach_s, closest_distance_m)
        """
        pos1 = np.array(obj1['position'])
        vel1 = np.array(obj1['velocity'])
        pos2 = np.array(obj2['position'])
        vel2 = np.array(obj2['velocity'])
        
        # Relative position and velocity
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1
        
        # Time of closest approach: t = -dot(rel_pos, rel_vel) / dot(rel_vel, rel_vel)
        rel_vel_squared = np.dot(rel_vel, rel_vel)
        
        if rel_vel_squared < 1e-10:  # Objects moving in parallel
            time_to_closest = 0.0
        else:
            time_to_closest = -np.dot(rel_pos, rel_vel) / rel_vel_squared
        
        # Constrain to time horizon
        time_to_closest = max(0.0, min(time_to_closest, time_horizon_s))
        
        # Calculate closest distance
        closest_pos1 = pos1 + vel1 * time_to_closest
        closest_pos2 = pos2 + vel2 * time_to_closest
        closest_distance = np.linalg.norm(closest_pos2 - closest_pos1)
        
        return time_to_closest, closest_distance
    
    def scan_for_collisions(self, objects: List[Dict[str, Any]], 
                           time_horizon_s: float = 3600.0) -> List[Dict[str, Any]]:
        """
        Scan a list of objects for potential collisions within time horizon.
        
        Args:
            objects: List of objects with position, velocity, and radius
            time_horizon_s: Time horizon for prediction in seconds
            
        Returns:
            List of collision pairs with collision data
        """
        collision_pairs = []
        
        # O(n²) comparison of all object pairs
        for i in range(len(objects)):
            for j in range(i+1, len(objects)):
                obj1 = objects[i]
                obj2 = objects[j]
                
                # Predict closest approach
                time_to_closest, closest_distance = self.predict_closest_approach(
                    obj1, obj2, time_horizon_s
                )
                
                # Check if collision is predicted
                collision_threshold = obj1['radius'] + obj2['radius'] + self.safety_margin_m
                if closest_distance < collision_threshold:
                    collision_pairs.append({
                        'object1_id': obj1['id'],
                        'object2_id': obj2['id'],
                        'time_to_collision_s': time_to_closest,
                        'closest_distance_m': closest_distance,
                        'collision_probability': self._calculate_collision_probability(
                            closest_distance, collision_threshold
                        )
                    })
        
        # Sort by time to collision
        collision_pairs.sort(key=lambda x: x['time_to_collision_s'])
        self.collision_pairs = collision_pairs
        return collision_pairs
    
    def _calculate_collision_probability(self, distance: float, threshold: float) -> float:
        """Calculate collision probability based on distance vs threshold"""
        if distance >= threshold:
            return 0.0
        
        # Simple linear model: probability increases as distance decreases
        return 1.0 - (distance / threshold)