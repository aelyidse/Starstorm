import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
import math

class AtmosphericPathPlanner:
    """
    Plans optimal paths through the atmosphere, considering vehicle dynamics and constraints.
    Supports Dubins paths, minimum-turn, and energy-efficient trajectories.
    """
    def __init__(self, min_turn_radius_m: float = 100.0):
        self.min_turn_radius_m = min_turn_radius_m

    def plan_path(self, start: np.ndarray, end: np.ndarray, heading_start: float, heading_end: float) -> List[np.ndarray]:
        """
        Plan a path from start to end with specified headings using 2D Dubins path.
        
        Args:
            start: Starting position [x, y, z]
            end: Ending position [x, y, z]
            heading_start: Starting heading in radians
            heading_end: Ending heading in radians
            
        Returns:
            List of waypoints from start to end
        """
        # For 3D path planning, we'll use the 3D Dubins path method
        return self.plan_3d_dubins_path(start, end, heading_start, 0.0, heading_end, 0.0)
    
    def plan_3d_dubins_path(self, 
                           start: np.ndarray, 
                           end: np.ndarray, 
                           heading_start: float, 
                           pitch_start: float,
                           heading_end: float, 
                           pitch_end: float,
                           resolution: float = 0.1) -> List[np.ndarray]:
        """
        Plan a full 3D Dubins path considering both heading and pitch constraints.
        
        Args:
            start: Starting position [x, y, z]
            end: Ending position [x, y, z]
            heading_start: Starting heading in radians
            pitch_start: Starting pitch in radians
            heading_end: Ending heading in radians
            pitch_end: Ending pitch in radians
            resolution: Path resolution parameter (lower = more waypoints)
            
        Returns:
            List of waypoints from start to end
        """
        # Project start and end points to 2D for horizontal path planning
        start_2d = start[:2]
        end_2d = end[:2]
        
        # Get 2D Dubins path for horizontal movement
        horizontal_path = self._compute_2d_dubins_path(
            start_2d, end_2d, heading_start, heading_end, resolution)
        
        # Calculate the total horizontal path length
        horizontal_length = self._calculate_path_length(horizontal_path)
        
        # Create full 3D path by adding altitude profile
        path_3d = []
        
        # Initial and final altitudes
        z_start = start[2]
        z_end = end[2]
        
        # Generate altitude profile based on pitch constraints
        max_climb_rate = math.tan(max(abs(pitch_start), abs(pitch_end)))
        
        # Check if we need a more complex altitude profile due to pitch constraints
        altitude_diff = z_end - z_start
        min_horizontal_dist_needed = abs(altitude_diff) / max_climb_rate if max_climb_rate > 0 else 0
        
        if horizontal_length >= min_horizontal_dist_needed:
            # Simple altitude profile - constant climb/descent rate
            for i, point in enumerate(horizontal_path):
                # Calculate progress along path (0 to 1)
                progress = i / (len(horizontal_path) - 1) if len(horizontal_path) > 1 else 1
                
                # Interpolate altitude
                altitude = z_start + progress * altitude_diff
                
                # Create 3D point
                point_3d = np.array([point[0], point[1], altitude])
                path_3d.append(point_3d)
        else:
            # Complex altitude profile needed - implement climb/descent with pitch constraints
            # This is a simplified approach - a more sophisticated solution would use 3D curves
            
            # Calculate required horizontal distance for climb/descent
            horizontal_dist_for_altitude = min_horizontal_dist_needed
            
            # Calculate horizontal distance available for level flight
            level_flight_dist = horizontal_length - horizontal_dist_for_altitude
            
            # If level flight distance is negative, we need to extend the path
            if level_flight_dist < 0:
                # Extend the path by adding a detour
                # This is a simplified approach - in practice, you'd want to generate a new path
                midpoint = (start_2d + end_2d) / 2
                offset = np.array([-1, 1]) * abs(altitude_diff) / max_climb_rate / 2
                detour_point = midpoint + offset
                
                # Recalculate path with detour
                path_to_detour = self._compute_2d_dubins_path(
                    start_2d, detour_point, heading_start, heading_end, resolution)
                path_from_detour = self._compute_2d_dubins_path(
                    detour_point, end_2d, heading_start, heading_end, resolution)
                
                # Combine paths
                horizontal_path = path_to_detour[:-1] + path_from_detour
                horizontal_length = self._calculate_path_length(horizontal_path)
            
            # Now create the 3D path with proper altitude profile
            climb_dist = horizontal_dist_for_altitude / 2
            
            for i, point in enumerate(horizontal_path):
                # Calculate distance traveled along path
                if i == 0:
                    dist_traveled = 0
                else:
                    dist_traveled = self._calculate_path_length(horizontal_path[:i+1])
                
                # Calculate altitude based on position along path
                if dist_traveled < climb_dist:
                    # Initial climb/descent phase
                    progress = dist_traveled / climb_dist
                    altitude = z_start + progress * altitude_diff / 2
                elif dist_traveled > horizontal_length - climb_dist:
                    # Final climb/descent phase
                    remaining = horizontal_length - dist_traveled
                    progress = 1 - (remaining / climb_dist)
                    altitude = z_start + altitude_diff / 2 + progress * altitude_diff / 2
                else:
                    # Level flight phase
                    altitude = z_start + altitude_diff / 2
                
                # Create 3D point
                point_3d = np.array([point[0], point[1], altitude])
                path_3d.append(point_3d)
        
        return path_3d
    
    def _compute_2d_dubins_path(self, 
                               start: np.ndarray, 
                               end: np.ndarray, 
                               heading_start: float, 
                               heading_end: float,
                               resolution: float = 0.1) -> List[np.ndarray]:
        """
        Compute a 2D Dubins path between two points with specified headings.
        
        Args:
            start: Starting position [x, y]
            end: Ending position [x, y]
            heading_start: Starting heading in radians
            heading_end: Ending heading in radians
            resolution: Path resolution parameter
            
        Returns:
            List of waypoints for the Dubins path
        """
        # Define the six possible Dubins path types
        path_types = ["LSL", "LSR", "RSL", "RSR", "RLR", "LRL"]
        
        # Calculate the best path
        best_path = None
        best_length = float('inf')
        
        for path_type in path_types:
            path, length = self._calculate_dubins_path(
                start, end, heading_start, heading_end, path_type)
            
            if path is not None and length < best_length:
                best_path = path
                best_length = length
        
        # If no valid path found, return straight line
        if best_path is None:
            return [start, end]
        
        # Sample the path at the specified resolution
        sampled_path = self._sample_dubins_path(best_path, resolution)
        
        return sampled_path
    
    def _calculate_dubins_path(self, 
                              start: np.ndarray, 
                              end: np.ndarray, 
                              heading_start: float, 
                              heading_end: float,
                              path_type: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Calculate a specific type of Dubins path.
        
        Args:
            start: Starting position [x, y]
            end: Ending position [x, y]
            heading_start: Starting heading in radians
            heading_end: Ending heading in radians
            path_type: Type of Dubins path ("LSL", "LSR", "RSL", "RSR", "RLR", "LRL")
            
        Returns:
            Tuple of (path parameters, path length)
        """
        # Normalize the problem - translate and rotate so that start is at origin
        # and heading_start is along the x-axis
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Distance between points
        d = math.sqrt(dx*dx + dy*dy)
        
        # Normalize by turning radius
        d = d / self.min_turn_radius_m
        
        # If points are too close, return None
        if d < 1e-10:
            return None, float('inf')
        
        # Rotate to align with x-axis
        theta = math.atan2(dy, dx) - heading_start
        
        # Normalized end point
        x = d * math.cos(theta)
        y = d * math.sin(theta)
        
        # Normalized end heading
        phi = heading_end - heading_start
        
        # Normalize angle to [-pi, pi]
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        
        # Calculate path based on type
        if path_type == "LSL":
            return self._dubins_LSL(x, y, phi)
        elif path_type == "LSR":
            return self._dubins_LSR(x, y, phi)
        elif path_type == "RSL":
            return self._dubins_RSL(x, y, phi)
        elif path_type == "RSR":
            return self._dubins_RSR(x, y, phi)
        elif path_type == "RLR":
            return self._dubins_RLR(x, y, phi)
        elif path_type == "LRL":
            return self._dubins_LRL(x, y, phi)
        else:
            return None, float('inf')
    
    def _dubins_LSL(self, x: float, y: float, phi: float) -> Tuple[Optional[Dict[str, Any]], float]:
        """Calculate LSL (Left-Straight-Left) Dubins path"""
        u = math.sin(phi)
        v = math.cos(phi)
        
        # Center of first circle
        c1 = np.array([0, 1])
        
        # Center of second circle
        c2 = np.array([x - math.sin(phi), y + 1 - math.cos(phi)])
        
        # Distance between circle centers
        d = np.linalg.norm(c2 - c1)
        
        # If circles are too close, no valid path
        if d < 2:
            return None, float('inf')
        
        # Calculate angles
        theta = math.atan2(c2[1] - c1[1], c2[0] - c1[0])
        alpha = math.acos(2 / d)
        
        # Calculate arc angles
        t = self._mod2pi(theta + alpha)
        p = self._mod2pi(phi - t + math.pi)
        
        # Calculate segment lengths
        segment1 = t
        segment2 = d * math.sin(alpha)
        segment3 = p
        
        # Total length
        length = segment1 + segment2 + segment3
        
        path = {
            'type': 'LSL',
            'start': np.array([0, 0]),
            'end': np.array([x, y]),
            'segments': [segment1, segment2, segment3],
            'centers': [c1, c2],
            'radius': 1.0  # Normalized radius
        }
        
        return path, length
    
    def _dubins_RSR(self, x: float, y: float, phi: float) -> Tuple[Optional[Dict[str, Any]], float]:
        """Calculate RSR (Right-Straight-Right) Dubins path"""
        # Mirror the problem and use LSL
        path, length = self._dubins_LSL(x, -y, -phi)
        
        if path is not None:
            path['type'] = 'RSR'
            path['centers'] = [np.array([0, -1]), np.array([x - math.sin(phi), y - 1 + math.cos(phi)])]
        
        return path, length
    
    def _dubins_LSR(self, x: float, y: float, phi: float) -> Tuple[Optional[Dict[str, Any]], float]:
        """Calculate LSR (Left-Straight-Right) Dubins path"""
        u = math.sin(phi)
        v = math.cos(phi)
        
        # Center of first circle
        c1 = np.array([0, 1])
        
        # Center of second circle
        c2 = np.array([x - math.sin(phi), y - 1 + math.cos(phi)])
        
        # Distance between circle centers
        d = np.linalg.norm(c2 - c1)
        
        # If circles are too close, no valid path
        if d < 2:
            return None, float('inf')
        
        # Calculate angles
        theta = math.atan2(c2[1] - c1[1], c2[0] - c1[0])
        alpha = math.acos(2 / d)
        
        # Calculate arc angles
        t = self._mod2pi(theta - alpha)
        p = self._mod2pi(t - phi)
        
        # Calculate segment lengths
        segment1 = t
        segment2 = d * math.sin(alpha)
        segment3 = p
        
        # Total length
        length = segment1 + segment2 + segment3
        
        path = {
            'type': 'LSR',
            'start': np.array([0, 0]),
            'end': np.array([x, y]),
            'segments': [segment1, segment2, segment3],
            'centers': [c1, c2],
            'radius': 1.0  # Normalized radius
        }
        
        return path, length
    
    def _dubins_RSL(self, x: float, y: float, phi: float) -> Tuple[Optional[Dict[str, Any]], float]:
        """Calculate RSL (Right-Straight-Left) Dubins path"""
        # Mirror the problem and use LSR
        path, length = self._dubins_LSR(x, -y, -phi)
        
        if path is not None:
            path['type'] = 'RSL'
            path['centers'] = [np.array([0, -1]), np.array([x - math.sin(phi), y + 1 - math.cos(phi)])]
        
        return path, length
    
    def _dubins_RLR(self, x: float, y: float, phi: float) -> Tuple[Optional[Dict[str, Any]], float]:
        """Calculate RLR (Right-Left-Right) Dubins path"""
        # Center of first circle
        c1 = np.array([0, -1])
        
        # Center of second circle
        c2 = np.array([x - math.sin(phi), y - 1 + math.cos(phi)])
        
        # Distance between circle centers
        d = np.linalg.norm(c2 - c1)
        
        # If circles are too far, no valid path
        if d > 4:
            return None, float('inf')
        
        # Calculate angles
        theta = math.atan2(c2[1] - c1[1], c2[0] - c1[0])
        alpha = math.acos(d / 4)
        
        # Calculate arc angles
        t = self._mod2pi(theta + alpha + math.pi)
        p = self._mod2pi(phi - t - math.pi)
        q = self._mod2pi(2 * alpha)
        
        # Calculate segment lengths
        segment1 = t
        segment2 = q
        segment3 = p
        
        # Total length
        length = segment1 + segment2 + segment3
        
        path = {
            'type': 'RLR',
            'start': np.array([0, 0]),
            'end': np.array([x, y]),
            'segments': [segment1, segment2, segment3],
            'centers': [c1, np.array([x - 2 * math.sin(phi), y - 2 * math.cos(phi)]), c2],
            'radius': 1.0  # Normalized radius
        }
        
        return path, length
    
    def _dubins_LRL(self, x: float, y: float, phi: float) -> Tuple[Optional[Dict[str, Any]], float]:
        """Calculate LRL (Left-Right-Left) Dubins path"""
        # Mirror the problem and use RLR
        path, length = self._dubins_RLR(x, -y, -phi)
        
        if path is not None:
            path['type'] = 'LRL'
            path['centers'] = [np.array([0, 1]), np.array([x - 2 * math.sin(phi), y + 2 * math.cos(phi)]), 
                              np.array([x - math.sin(phi), y + 1 - math.cos(phi)])]
        
        return path, length
    
    def _sample_dubins_path(self, path: Dict[str, Any], resolution: float) -> List[np.ndarray]:
        """
        Sample points along a Dubins path at the specified resolution.
        
        Args:
            path: Dubins path parameters
            resolution: Path resolution parameter
            
        Returns:
            List of waypoints along the path
        """
        # Extract path parameters
        path_type = path['type']
        segments = path['segments']
        centers = path['centers']
        radius = path['radius'] * self.min_turn_radius_m
        
        # Initialize waypoints list with start point
        waypoints = [path['start'] * self.min_turn_radius_m]
        
        # Current position and heading
        pos = path['start'] * self.min_turn_radius_m
        heading = 0.0  # Initial heading is along x-axis due to normalization
        
        # Sample points based on path type
        for i, segment in enumerate(segments):
            if i == 0:  # First segment (arc)
                if path_type[0] == 'L':
                    # Left turn
                    center = centers[0] * self.min_turn_radius_m
                    start_angle = math.atan2(pos[1] - center[1], pos[0] - center[0])
                    end_angle = start_angle + segment
                    
                    # Sample points along arc
                    num_samples = max(2, int(segment * radius / resolution))
                    for j in range(1, num_samples + 1):
                        angle = start_angle + j * segment / num_samples
                        x = center[0] + radius * math.cos(angle)
                        y = center[1] + radius * math.sin(angle)
                        waypoints.append(np.array([x, y]))
                    
                    # Update position and heading
                    pos = waypoints[-1]
                    heading = self._mod2pi(end_angle + math.pi/2)
                else:
                    # Right turn
                    center = centers[0] * self.min_turn_radius_m
                    start_angle = math.atan2(pos[1] - center[1], pos[0] - center[0])
                    end_angle = start_angle - segment
                    
                    # Sample points along arc
                    num_samples = max(2, int(segment * radius / resolution))
                    for j in range(1, num_samples + 1):
                        angle = start_angle - j * segment / num_samples
                        x = center[0] + radius * math.cos(angle)
                        y = center[1] + radius * math.sin(angle)
                        waypoints.append(np.array([x, y]))
                    
                    # Update position and heading
                    pos = waypoints[-1]
                    heading = self._mod2pi(end_angle - math.pi/2)
            
            elif i == 1:  # Second segment (straight or arc)
                if len(path_type) == 3 and path_type[1] == 'S':
                    # Straight segment
                    distance = segment * self.min_turn_radius_m
                    
                    # Sample points along straight line
                    num_samples = max(2, int(distance / resolution))
                    for j in range(1, num_samples + 1):
                        x = pos[0] + j * distance / num_samples * math.cos(heading)
                        y = pos[1] + j * distance / num_samples * math.sin(heading)
                        waypoints.append(np.array([x, y]))
                    
                    # Update position
                    pos = waypoints[-1]
                else:
                    # Arc segment (for RLR or LRL)
                    center = centers[1] * self.min_turn_radius_m
                    start_angle = math.atan2(pos[1] - center[1], pos[0] - center[0])
                    
                    if (path_type == 'RLR' and i == 1) or (path_type == 'LRL' and i == 1):
                        # Middle segment has opposite direction
                        end_angle = start_angle + segment if path_type[1] == 'L' else start_angle - segment
                        
                        # Sample points along arc
                        num_samples = max(2, int(segment * radius / resolution))
                        for j in range(1, num_samples + 1):
                            if path_type[1] == 'L':
                                angle = start_angle + j * segment / num_samples
                            else:
                                angle = start_angle - j * segment / num_samples
                            x = center[0] + radius * math.cos(angle)
                            y = center[1] + radius * math.sin(angle)
                            waypoints.append(np.array([x, y]))
                        
                        # Update position and heading
                        pos = waypoints[-1]
                        if path_type[1] == 'L':
                            heading = self._mod2pi(end_angle + math.pi/2)
                        else:
                            heading = self._mod2pi(end_angle - math.pi/2)
            
            else:  # Third segment (arc)
                center = centers[-1] * self.min_turn_radius_m
                start_angle = math.atan2(pos[1] - center[1], pos[0] - center[0])
                
                if path_type[2] == 'L':
                    # Left turn
                    end_angle = start_angle + segment
                    
                    # Sample points along arc
                    num_samples = max(2, int(segment * radius / resolution))
                    for j in range(1, num_samples + 1):
                        angle = start_angle + j * segment / num_samples
                        x = center[0] + radius * math.cos(angle)
                        y = center[1] + radius * math.sin(angle)
                        waypoints.append(np.array([x, y]))
                else:
                    # Right turn
                    end_angle = start_angle - segment
                    
                    # Sample points along arc
                    num_samples = max(2, int(segment * radius / resolution))
                    for j in range(1, num_samples + 1):
                        angle = start_angle - j * segment / num_samples
                        x = center[0] + radius * math.cos(angle)
                        y = center[1] + radius * math.sin(angle)
                        waypoints.append(np.array([x, y]))
        
        return waypoints
    
    def _mod2pi(self, theta: float) -> float:
        """Normalize angle to [0, 2*pi]"""
        return theta % (2 * math.pi)
    
    def _calculate_path_length(self, path: List[np.ndarray]) -> float:
        """Calculate the length of a path"""
        length = 0.0
        for i in range(1, len(path)):
            length += np.linalg.norm(path[i] - path[i-1])
        return length
    
    def generate_energy_efficient_path(self, 
                                      start: np.ndarray, 
                                      end: np.ndarray,
                                      heading_start: float,
                                      heading_end: float,
                                      wind_field: Callable[[np.ndarray], np.ndarray],
                                      energy_model: Callable[[np.ndarray, np.ndarray], float]) -> List[np.ndarray]:
        """
        Generate an energy-efficient path considering wind field and vehicle energy model.
        
        Args:
            start: Starting position [x, y, z]
            end: Ending position [x, y, z]
            heading_start: Starting heading in radians
            heading_end: Ending heading in radians
            wind_field: Function that returns wind vector at a given position
            energy_model: Function that calculates energy consumption for a path segment
            
        Returns:
            List of waypoints for the energy-efficient path
        """
        # First, get a baseline Dubins path
        baseline_path = self.plan_3d_dubins_path(
            start, end, heading_start, 0.0, heading_end, 0.0)
        
        # TODO: Implement energy optimization algorithm
        # This would involve modifying the path to minimize energy consumption
        # considering the wind field and energy model
        
        return baseline_path

class OrbitalPathPlanner:
    """
    Plans optimal orbital transfer trajectories (e.g., Hohmann, bi-elliptic, low-thrust).
    Computes waypoints and burns for maneuver execution.
    """
    def __init__(self, mu: float = 3.986004418e14):
        self.mu = mu  # Gravitational parameter (Earth)

    def hohmann_transfer(self, r1: float, r2: float) -> Dict[str, Any]:
        # Computes delta-v and timing for Hohmann transfer
        v1 = np.sqrt(self.mu / r1)
        v2 = np.sqrt(self.mu / r2)
        va = np.sqrt(self.mu * (2/r1 - 1/((r1 + r2)/2)))
        vb = np.sqrt(self.mu * (2/r2 - 1/((r1 + r2)/2)))
        delta_v1 = va - v1
        delta_v2 = v2 - vb
        transfer_time = np.pi * np.sqrt(((r1 + r2)/2)**3 / self.mu)
        return {
            'delta_v1_mps': delta_v1,
            'delta_v2_mps': delta_v2,
            'transfer_time_s': transfer_time
        }

    def plan_orbital_path(self, state: Dict[str, Any], target_orbit: Dict[str, float]) -> Dict[str, Any]:
        # Example: plan Hohmann transfer from current to target orbit
        r1 = np.linalg.norm(state['pos'])
        r2 = target_orbit['radius_m']
        return self.hohmann_transfer(r1, r2)
