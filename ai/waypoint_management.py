from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import time
import heapq
from collections import deque

class Waypoint:
    """Represents a navigation waypoint with position and metadata."""
    def __init__(self, 
                 waypoint_id: str,
                 position: Tuple[float, float, float],
                 arrival_time: Optional[float] = None,
                 required: bool = False,
                 metadata: Optional[Dict[str, Any]] = None):
        self.waypoint_id = waypoint_id
        self.position = position  # (x, y, z) coordinates
        self.arrival_time = arrival_time  # Optional scheduled arrival time
        self.required = required  # If True, waypoint cannot be skipped during replanning
        self.metadata = metadata or {}
        self.visited = False
        self.visit_time = None

    def mark_visited(self, visit_time: Optional[float] = None):
        """Mark waypoint as visited with timestamp."""
        self.visited = True
        self.visit_time = visit_time or time.time()

    def distance_to(self, other_waypoint) -> float:
        """Calculate Euclidean distance to another waypoint."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other_waypoint.position)))


class WaypointManager:
    """
    Manages waypoints and provides dynamic route replanning capabilities.
    Supports waypoint creation, route optimization, and adaptive replanning.
    """
    def __init__(self):
        self.waypoints: Dict[str, Waypoint] = {}
        self.current_route: List[str] = []  # Ordered list of waypoint IDs
        self.route_history: List[Dict[str, Any]] = []
        self.current_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.replanning_triggers: Dict[str, Any] = {
            'obstacle_detected': False,
            'time_deviation': 0.0,  # seconds
            'resource_change': False,
            'priority_change': False
        }
        self.replanning_threshold = {
            'time_deviation': 30.0,  # seconds
        }
        self.last_replan_time = None

    def add_waypoint(self, waypoint: Waypoint) -> None:
        """Add a waypoint to the manager."""
        self.waypoints[waypoint.waypoint_id] = waypoint

    def remove_waypoint(self, waypoint_id: str) -> Optional[Waypoint]:
        """Remove a waypoint from the manager."""
        if waypoint_id in self.waypoints:
            waypoint = self.waypoints.pop(waypoint_id)
            # Remove from current route if present
            if waypoint_id in self.current_route:
                self.current_route.remove(waypoint_id)
            return waypoint
        return None

    def set_current_position(self, position: Tuple[float, float, float]) -> None:
        """Update current position."""
        self.current_position = position

    def plan_route(self, start_waypoint_id: Optional[str] = None, 
                  end_waypoint_id: Optional[str] = None,
                  required_waypoint_ids: Optional[List[str]] = None) -> List[str]:
        """
        Plan an optimal route through waypoints.
        Uses a simplified greedy nearest-neighbor approach.
        """
        available_waypoints = set(self.waypoints.keys())
        required_waypoints = set(required_waypoint_ids or [])
        
        # Ensure required waypoints exist
        required_waypoints = required_waypoints.intersection(available_waypoints)
        
        # Start from current position or specified waypoint
        current_pos = self.current_position
        if start_waypoint_id and start_waypoint_id in self.waypoints:
            current_pos = self.waypoints[start_waypoint_id].position
            
        route = []
        remaining = list(available_waypoints)
        
        # Process required waypoints first using nearest neighbor
        while required_waypoints:
            nearest_id = None
            nearest_dist = float('inf')
            
            for wp_id in required_waypoints:
                wp = self.waypoints[wp_id]
                # Calculate distance from current position to waypoint
                if start_waypoint_id and wp_id == start_waypoint_id:
                    dist = 0  # Start waypoint has zero distance
                else:
                    x1, y1, z1 = current_pos
                    x2, y2, z2 = wp.position
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = wp_id
            
            if nearest_id:
                route.append(nearest_id)
                current_pos = self.waypoints[nearest_id].position
                required_waypoints.remove(nearest_id)
                remaining.remove(nearest_id)
        
        # Process remaining waypoints
        while remaining:
            nearest_id = None
            nearest_dist = float('inf')
            
            for wp_id in remaining:
                wp = self.waypoints[wp_id]
                # Calculate distance from current position to waypoint
                x1, y1, z1 = current_pos
                x2, y2, z2 = wp.position
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = wp_id
            
            if nearest_id:
                route.append(nearest_id)
                current_pos = self.waypoints[nearest_id].position
                remaining.remove(nearest_id)
        
        # If end waypoint specified, ensure it's the last one
        if end_waypoint_id and end_waypoint_id in self.waypoints:
            if end_waypoint_id in route:
                route.remove(end_waypoint_id)
            route.append(end_waypoint_id)
        
        # Save the new route
        self.current_route = route
        self.route_history.append({
            'route': route.copy(),
            'timestamp': time.time(),
            'reason': 'initial_planning'
        })
        
        return route

    def check_replanning_needed(self) -> Tuple[bool, str]:
        """Check if replanning is needed based on triggers."""
        if self.replanning_triggers['obstacle_detected']:
            return True, 'obstacle_detected'
            
        if abs(self.replanning_triggers['time_deviation']) > self.replanning_threshold['time_deviation']:
            return True, 'time_deviation'
            
        if self.replanning_triggers['resource_change']:
            return True, 'resource_change'
            
        if self.replanning_triggers['priority_change']:
            return True, 'priority_change'
            
        return False, ''

    def replan_route(self, reason: str = 'manual') -> List[str]:
        """
        Dynamically replan the route based on current conditions.
        Preserves required waypoints while optimizing the path.
        """
        # Skip if no current route or too soon after last replan
        if not self.current_route:
            return self.current_route
            
        if self.last_replan_time and (time.time() - self.last_replan_time) < 5.0:
            return self.current_route  # Avoid too frequent replanning
        
        # Identify current position and remaining waypoints
        current_position = self.current_position
        
        # Get unvisited waypoints from current route
        unvisited_waypoints = []
        required_waypoints = []
        
        for wp_id in self.current_route:
            wp = self.waypoints.get(wp_id)
            if wp and not wp.visited:
                unvisited_waypoints.append(wp_id)
                if wp.required:
                    required_waypoints.append(wp_id)
        
        # If no unvisited waypoints, return empty route
        if not unvisited_waypoints:
            return []
        
        # Plan new route from current position through remaining waypoints
        new_route = self.plan_route(
            required_waypoint_ids=required_waypoints
        )
        
        # Record replanning event
        self.route_history.append({
            'route': new_route.copy(),
            'timestamp': time.time(),
            'reason': reason
        })
        
        self.last_replan_time = time.time()
        self.current_route = new_route
        
        # Reset triggers
        self.replanning_triggers = {
            'obstacle_detected': False,
            'time_deviation': 0.0,
            'resource_change': False,
            'priority_change': False
        }
        
        return new_route

    def set_replanning_trigger(self, trigger_name: str, value: Any) -> None:
        """Set a trigger that may cause route replanning."""
        if trigger_name in self.replanning_triggers:
            self.replanning_triggers[trigger_name] = value
            
            # Check if replanning is needed immediately
            needed, reason = self.check_replanning_needed()
            if needed:
                self.replan_route(reason=reason)

    def get_next_waypoint(self) -> Optional[Waypoint]:
        """Get the next waypoint in the current route."""
        if not self.current_route:
            return None
            
        next_wp_id = self.current_route[0]
        return self.waypoints.get(next_wp_id)

    def mark_waypoint_visited(self, waypoint_id: str) -> None:
        """Mark a waypoint as visited."""
        if waypoint_id in self.waypoints:
            self.waypoints[waypoint_id].mark_visited()
            
            # Remove from current route
            if waypoint_id in self.current_route:
                self.current_route.remove(waypoint_id)

    def get_route_progress(self) -> Dict[str, Any]:
        """Get progress information about the current route."""
        total_waypoints = len(self.current_route) + sum(1 for wp in self.waypoints.values() if wp.visited)
        visited_waypoints = sum(1 for wp in self.waypoints.values() if wp.visited)
        
        if total_waypoints == 0:
            completion_percentage = 0.0
        else:
            completion_percentage = (visited_waypoints / total_waypoints) * 100.0
            
        return {
            'total_waypoints': total_waypoints,
            'visited_waypoints': visited_waypoints,
            'remaining_waypoints': len(self.current_route),
            'completion_percentage': completion_percentage
        }

    def get_route_history(self) -> List[Dict[str, Any]]:
        """Get the history of route changes."""
        return self.route_history