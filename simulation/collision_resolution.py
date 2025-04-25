import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

class CollisionAvoidanceStrategy(Enum):
    """Strategies for collision avoidance"""
    ALTITUDE_CHANGE = "Altitude Change"
    INCLINATION_CHANGE = "Inclination Change"
    PHASING_MANEUVER = "Phasing Maneuver"
    COMBINED_MANEUVER = "Combined Maneuver"

class CollisionResolver:
    """
    Resolves potential collisions by calculating avoidance maneuvers.
    Supports various resolution strategies and optimization for fuel efficiency.
    """
    def __init__(self, mu: float = 3.986004418e14):
        self.mu = mu  # Earth's gravitational parameter (m³/s²)
        
    def calculate_avoidance_maneuver(self, 
                                    spacecraft: Dict[str, Any], 
                                    collision_object: Dict[str, Any],
                                    time_to_collision_s: float,
                                    strategy: CollisionAvoidanceStrategy = CollisionAvoidanceStrategy.ALTITUDE_CHANGE
                                    ) -> Dict[str, Any]:
        """
        Calculate avoidance maneuver for spacecraft to avoid collision.
        
        Args:
            spacecraft: Spacecraft state with position, velocity, mass
            collision_object: Object to avoid with position, velocity
            time_to_collision_s: Time to predicted collision
            strategy: Avoidance strategy to use
            
        Returns:
            Maneuver details including delta-v, fuel required, new trajectory
        """
        if strategy == CollisionAvoidanceStrategy.ALTITUDE_CHANGE:
            return self._altitude_change_maneuver(spacecraft, collision_object, time_to_collision_s)
        elif strategy == CollisionAvoidanceStrategy.INCLINATION_CHANGE:
            return self._inclination_change_maneuver(spacecraft, collision_object, time_to_collision_s)
        elif strategy == CollisionAvoidanceStrategy.PHASING_MANEUVER:
            return self._phasing_maneuver(spacecraft, collision_object, time_to_collision_s)
        else:
            return self._combined_maneuver(spacecraft, collision_object, time_to_collision_s)
    
    def _altitude_change_maneuver(self, spacecraft: Dict[str, Any], 
                                collision_object: Dict[str, Any],
                                time_to_collision_s: float) -> Dict[str, Any]:
        """Calculate altitude change maneuver (raise or lower orbit)"""
        # Extract spacecraft parameters
        pos = np.array(spacecraft['position'])
        vel = np.array(spacecraft['velocity'])
        
        # Current orbital parameters
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Determine if we should raise or lower orbit
        # Simple heuristic: if collision object is higher, lower our orbit
        collision_pos = np.array(collision_object['position'])
        collision_r = np.linalg.norm(collision_pos)
        
        # Calculate new target radius (30km difference is usually sufficient)
        altitude_change = 30000.0  # 30 km
        new_r = r - altitude_change if r < collision_r else r + altitude_change
        
        # Calculate delta-v for Hohmann transfer
        delta_v1 = np.sqrt(self.mu/r) * (np.sqrt(2*new_r/(r+new_r)) - 1)
        delta_v2 = np.sqrt(self.mu/new_r) * (1 - np.sqrt(2*r/(r+new_r)))
        total_delta_v = abs(delta_v1) + abs(delta_v2)
        
        # Calculate transfer time
        transfer_time = np.pi * np.sqrt((r+new_r)**3/(8*self.mu))
        
        return {
            'strategy': CollisionAvoidanceStrategy.ALTITUDE_CHANGE.value,
            'delta_v_mps': total_delta_v,
            'transfer_time_s': transfer_time,
            'new_radius_m': new_r,
            'burn_1': {
                'delta_v_mps': delta_v1,
                'time_s': 0  # Immediate execution
            },
            'burn_2': {
                'delta_v_mps': delta_v2,
                'time_s': transfer_time
            }
        }
    
    def _inclination_change_maneuver(self, spacecraft: Dict[str, Any], 
                                   collision_object: Dict[str, Any],
                                   time_to_collision_s: float) -> Dict[str, Any]:
        """Calculate inclination change maneuver"""
        # Extract spacecraft parameters
        pos = np.array(spacecraft['position'])
        vel = np.array(spacecraft['velocity'])
        
        # Current orbital parameters
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Calculate inclination change (1 degree is usually sufficient)
        inclination_change_rad = np.radians(1.0)
        
        # Delta-v for inclination change: 2*v*sin(i/2)
        delta_v = 2 * v * np.sin(inclination_change_rad/2)
        
        return {
            'strategy': CollisionAvoidanceStrategy.INCLINATION_CHANGE.value,
            'delta_v_mps': delta_v,
            'inclination_change_deg': np.degrees(inclination_change_rad),
            'burn': {
                'delta_v_mps': delta_v,
                'time_s': 0  # Immediate execution
            }
        }
    
    def _phasing_maneuver(self, spacecraft: Dict[str, Any], 
                        collision_object: Dict[str, Any],
                        time_to_collision_s: float) -> Dict[str, Any]:
        """Calculate phasing maneuver (temporarily change orbital period)"""
        # Extract spacecraft parameters
        pos = np.array(spacecraft['position'])
        vel = np.array(spacecraft['velocity'])
        
        # Current orbital parameters
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Current orbital period
        current_period = 2 * np.pi * np.sqrt(r**3/self.mu)
        
        # Calculate new period for phasing (5% slower for half an orbit)
        new_period = current_period * 1.05
        
        # New semi-major axis
        new_a = ((new_period/(2*np.pi))**2 * self.mu)**(1/3)
        
        # Delta-v for transfer to phasing orbit and back
        delta_v1 = np.sqrt(self.mu/r) * (np.sqrt(2*new_a/(r+new_a)) - 1)
        delta_v2 = np.sqrt(self.mu/new_a) * (1 - np.sqrt(2*r/(r+new_a)))
        
        # Time in phasing orbit (half an orbit)
        phasing_time = new_period / 2
        
        return {
            'strategy': CollisionAvoidanceStrategy.PHASING_MANEUVER.value,
            'delta_v_mps': abs(delta_v1) + abs(delta_v2),
            'phasing_time_s': phasing_time,
            'burn_1': {
                'delta_v_mps': delta_v1,
                'time_s': 0  # Immediate execution
            },
            'burn_2': {
                'delta_v_mps': delta_v2,
                'time_s': phasing_time
            }
        }
    
    def _combined_maneuver(self, spacecraft: Dict[str, Any], 
                         collision_object: Dict[str, Any],
                         time_to_collision_s: float) -> Dict[str, Any]:
        """Calculate combined maneuver (small changes to multiple orbital elements)"""
        # For combined maneuvers, we'll do a small altitude change and small inclination change
        
        # Calculate a smaller altitude change (15km)
        altitude_change = 15000.0  # 15 km
        pos = np.array(spacecraft['position'])
        vel = np.array(spacecraft['velocity'])
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Determine direction based on collision object
        collision_pos = np.array(collision_object['position'])
        collision_r = np.linalg.norm(collision_pos)
        new_r = r - altitude_change if r < collision_r else r + altitude_change
        
        # Delta-v for altitude change
        delta_v_altitude = np.sqrt(self.mu/r) * (np.sqrt(2*new_r/(r+new_r)) - 1)
        
        # Small inclination change (0.5 degrees)
        inclination_change_rad = np.radians(0.5)
        delta_v_inclination = 2 * v * np.sin(inclination_change_rad/2)
        
        # Combined delta-v (simplified - in reality would need vector addition)
        total_delta_v = abs(delta_v_altitude) + abs(delta_v_inclination)
        
        return {
            'strategy': CollisionAvoidanceStrategy.COMBINED_MANEUVER.value,
            'delta_v_mps': total_delta_v,
            'altitude_change_m': altitude_change,
            'inclination_change_deg': np.degrees(inclination_change_rad),
            'burn': {
                'delta_v_mps': total_delta_v,
                'time_s': 0  # Immediate execution
            }
        }
    
    def select_optimal_strategy(self, spacecraft: Dict[str, Any], 
                              collision_object: Dict[str, Any],
                              time_to_collision_s: float) -> Dict[str, Any]:
        """
        Select the optimal collision avoidance strategy based on fuel efficiency.
        
        Args:
            spacecraft: Spacecraft state
            collision_object: Object to avoid
            time_to_collision_s: Time to predicted collision
            
        Returns:
            Optimal maneuver details
        """
        # Calculate maneuvers for all strategies
        maneuvers = []
        for strategy in CollisionAvoidanceStrategy:
            maneuver = self.calculate_avoidance_maneuver(
                spacecraft, collision_object, time_to_collision_s, strategy
            )
            maneuvers.append(maneuver)
        
        # Select maneuver with lowest delta-v (most fuel efficient)
        optimal_maneuver = min(maneuvers, key=lambda x: x['delta_v_mps'])
        return optimal_maneuver