import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
import time

class EnergyOptimalTrajectory:
    """
    Generates energy-optimal trajectories for spacecraft maneuvers.
    Minimizes propellant usage while satisfying mission constraints.
    """
    
    def __init__(self, mu: float = 3.986004418e14):
        """
        Initialize the energy-optimal trajectory generator.
        
        Args:
            mu: Gravitational parameter (Earth default)
        """
        self.mu = mu  # Standard gravitational parameter
        self.obstacles = []  # List to store detected obstacles
        self.current_trajectory = None  # Current planned trajectory
        self.replanning_threshold = 1000.0  # Distance threshold for replanning (meters)
        
    def generate_optimal_trajectory(self, 
                                   initial_state: Dict[str, Any],
                                   target_state: Dict[str, Any],
                                   time_constraint: float,
                                   max_thrust: float,
                                   isp: float,
                                   mass: float,
                                   timestep: float = 60.0) -> Dict[str, Any]:
        """
        Generate an energy-optimal trajectory between initial and target states.
        
        Args:
            initial_state: Initial orbital state {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            target_state: Target orbital state {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            time_constraint: Maximum time allowed for the maneuver (seconds)
            max_thrust: Maximum available thrust (N)
            isp: Specific impulse (s)
            mass: Initial spacecraft mass (kg)
            timestep: Simulation timestep (seconds)
            
        Returns:
            Dictionary containing trajectory and control history
        """
        # Constants
        g0 = 9.80665  # Standard gravity (m/s²)
        
        # Calculate number of steps
        num_steps = int(time_constraint / timestep)
        
        # Initialize trajectory arrays
        positions = np.zeros((num_steps + 1, 3))
        velocities = np.zeros((num_steps + 1, 3))
        thrusts = np.zeros((num_steps, 3))
        thrust_magnitudes = np.zeros(num_steps)
        mass_history = np.zeros(num_steps + 1)
        times = np.linspace(0, time_constraint, num_steps + 1)
        
        # Set initial conditions
        positions[0] = np.array(initial_state['pos'])
        velocities[0] = np.array(initial_state['vel'])
        mass_history[0] = mass
        
        # Indirect optimization method (primer vector theory)
        # Initialize costates (adjoint variables)
        lambda_r = np.zeros((num_steps + 1, 3))
        lambda_v = np.zeros((num_steps + 1, 3))
        
        # Initial guess for costates (can be improved with shooting method)
        lambda_r[0] = np.array([0.1, 0.1, 0.1])
        lambda_v[0] = np.array([1.0, 1.0, 1.0])
        
        # Iterative optimization
        max_iterations = 10
        convergence_threshold = 1e-3
        
        for iteration in range(max_iterations):
            # Forward propagation with current costates
            for step in range(num_steps):
                # Current state
                r = positions[step]
                v = velocities[step]
                m = mass_history[step]
                
                # Compute thrust direction from primer vector
                primer_vector = lambda_v[step]
                primer_magnitude = np.linalg.norm(primer_vector)
                
                # Switching function determines if thrust is on or off
                if primer_magnitude > 1.0:
                    # Thrust is on - maximum thrust in optimal direction
                    thrust_direction = -primer_vector / primer_magnitude
                    thrust_magnitude = max_thrust
                else:
                    # Thrust is off (coasting)
                    thrust_direction = np.zeros(3)
                    thrust_magnitude = 0.0
                
                # Calculate thrust vector
                thrust = thrust_magnitude * thrust_direction
                
                # Store thrust
                thrusts[step] = thrust
                thrust_magnitudes[step] = thrust_magnitude
                
                # Calculate mass flow rate
                mdot = thrust_magnitude / (isp * g0) if thrust_magnitude > 0 else 0
                
                # Update mass
                new_mass = m - mdot * timestep
                
                # Gravitational acceleration
                r_norm = np.linalg.norm(r)
                g = -self.mu * r / (r_norm ** 3)
                
                # State propagation (simple Euler integration)
                new_r = r + v * timestep
                new_v = v + (g + thrust / m) * timestep
                
                # Store new state
                positions[step + 1] = new_r
                velocities[step + 1] = new_v
                mass_history[step + 1] = new_mass
                
                # Propagate costates (adjoint equations)
                r_squared = r_norm ** 2
                r_cubed = r_norm ** 3
                
                # Gravity gradient matrix
                G = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            G[i, j] = 3 * r[i] * r[j] / r_squared - 1
                        else:
                            G[i, j] = 3 * r[i] * r[j] / r_squared
                
                G = G * self.mu / r_cubed
                
                # Costate propagation
                lambda_r[step + 1] = lambda_r[step] - G @ lambda_v[step] * timestep
                lambda_v[step + 1] = lambda_v[step] - lambda_r[step] * timestep
            
            # Check terminal constraints
            final_pos_error = np.linalg.norm(positions[-1] - target_state['pos'])
            final_vel_error = np.linalg.norm(velocities[-1] - target_state['vel'])
            total_error = final_pos_error + final_vel_error
            
            # Check convergence
            if total_error < convergence_threshold:
                break
            
            # Update initial costates using simple gradient method
            sensitivity = 0.1
            lambda_r[0] -= sensitivity * (positions[-1] - target_state['pos'])
            lambda_v[0] -= sensitivity * (velocities[-1] - target_state['vel'])
        
        # Calculate propellant usage
        propellant_used = mass - mass_history[-1]
        
        # Calculate total delta-v
        delta_v = isp * g0 * np.log(mass / mass_history[-1])
        
        return {
            'success': final_pos_error < 1000 and final_vel_error < 10,
            'trajectory': {
                'times': times.tolist(),
                'positions': positions.tolist(),
                'velocities': velocities.tolist(),
                'mass': mass_history.tolist()
            },
            'controls': {
                'thrust_vectors': thrusts.tolist(),
                'thrust_magnitudes': thrust_magnitudes.tolist()
            },
            'performance': {
                'propellant_used_kg': propellant_used,
                'delta_v_mps': delta_v,
                'final_position_error_m': final_pos_error,
                'final_velocity_error_mps': final_vel_error
            }
        }
    
    def low_thrust_transfer(self, 
                           initial_orbit: Dict[str, float],
                           target_orbit: Dict[str, float],
                           spacecraft: Dict[str, float],
                           max_time_days: float = 30.0) -> Dict[str, Any]:
        """
        Generate an energy-optimal low-thrust transfer between orbits.
        
        Args:
            initial_orbit: Initial orbit parameters {'a': semi-major axis, 'e': eccentricity, 'i': inclination}
            target_orbit: Target orbit parameters {'a': semi-major axis, 'e': eccentricity, 'i': inclination}
            spacecraft: Spacecraft parameters {'mass': kg, 'max_thrust': N, 'isp': s}
            max_time_days: Maximum transfer time in days
            
        Returns:
            Optimal trajectory and control history
        """
        # Convert orbital elements to state vectors (simplified)
        initial_state = self._orbit_to_state(initial_orbit)
        target_state = self._orbit_to_state(target_orbit)
        
        # Convert time to seconds
        max_time_seconds = max_time_days * 86400
        
        # Generate optimal trajectory
        return self.generate_optimal_trajectory(
            initial_state,
            target_state,
            max_time_seconds,
            spacecraft['max_thrust'],
            spacecraft['isp'],
            spacecraft['mass']
        )
    
    def _orbit_to_state(self, orbit: Dict[str, float]) -> Dict[str, List[float]]:
        """
        Convert orbital elements to position and velocity vectors.
        Simplified implementation for circular/near-circular orbits.
        
        Args:
            orbit: Orbital elements {'a': semi-major axis, 'e': eccentricity, 'i': inclination}
            
        Returns:
            State vectors {'pos': [x,y,z], 'vel': [vx,vy,vz]}
        """
        a = orbit['a']
        e = orbit.get('e', 0.0)
        i = orbit.get('i', 0.0)
        
        # For simplicity, assume position at perigee
        r = a * (1 - e)
        v = np.sqrt(self.mu * (2/r - 1/a))
        
        # Position in orbital plane
        pos = np.array([r, 0, 0])
        vel = np.array([0, v, 0])
        
        # Rotate by inclination
        if i != 0:
            cos_i = np.cos(np.radians(i))
            sin_i = np.sin(np.radians(i))
            rot_matrix = np.array([
                [1, 0, 0],
                [0, cos_i, -sin_i],
                [0, sin_i, cos_i]
            ])
            pos = rot_matrix @ pos
            vel = rot_matrix @ vel
        
        return {'pos': pos.tolist(), 'vel': vel.tolist()}
    
    def add_obstacle(self, position: List[float], radius: float, velocity: Optional[List[float]] = None):
        """
        Add an obstacle to the environment for collision avoidance.
        
        Args:
            position: [x, y, z] position of obstacle center (m)
            radius: Obstacle radius/safety margin (m)
            velocity: Optional [vx, vy, vz] velocity vector (m/s)
        """
        self.obstacles.append({
            'position': np.array(position),
            'radius': radius,
            'velocity': np.array(velocity) if velocity else np.zeros(3)
        })
        
        # If we have a current trajectory, check if replanning is needed
        if self.current_trajectory:
            self.check_trajectory_safety()
    
    def clear_obstacles(self):
        """Clear all obstacles from the environment."""
        self.obstacles = []
    
    def check_trajectory_safety(self) -> bool:
        """
        Check if current trajectory is safe from all obstacles.
        
        Returns:
            True if trajectory is safe, False if collision detected
        """
        if not self.current_trajectory or not self.obstacles:
            return True
            
        positions = np.array(self.current_trajectory['trajectory']['positions'])
        times = np.array(self.current_trajectory['trajectory']['times'])
        
        for obstacle in self.obstacles:
            for i, pos in enumerate(positions):
                # Update obstacle position based on its velocity and time
                obstacle_pos = obstacle['position'] + obstacle['velocity'] * times[i]
                
                # Check distance to obstacle
                distance = np.linalg.norm(pos - obstacle_pos)
                
                # If distance is less than obstacle radius plus safety margin, collision detected
                if distance < obstacle['radius'] + 500.0:  # 500m safety margin
                    return False
                    
        return True
    
    def avoid_obstacles(self, 
                       initial_state: Dict[str, Any],
                       target_state: Dict[str, Any],
                       spacecraft: Dict[str, float],
                       max_time_seconds: float = 86400.0) -> Dict[str, Any]:
        """
        Generate an obstacle-avoiding trajectory with dynamic replanning.
        
        Args:
            initial_state: Initial state {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            target_state: Target state {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            spacecraft: Spacecraft parameters {'mass': kg, 'max_thrust': N, 'isp': s}
            max_time_seconds: Maximum transfer time in seconds
            
        Returns:
            Optimal trajectory that avoids obstacles
        """
        # First attempt at trajectory generation without considering obstacles
        trajectory = self.generate_optimal_trajectory(
            initial_state,
            target_state,
            max_time_seconds,
            spacecraft['max_thrust'],
            spacecraft['isp'],
            spacecraft['mass']
        )
        
        self.current_trajectory = trajectory
        
        # Check if trajectory is safe
        if self.check_trajectory_safety():
            return trajectory
            
        # If not safe, try to find a safe trajectory with waypoints
        return self._replan_with_waypoints(
            initial_state,
            target_state,
            spacecraft,
            max_time_seconds
        )
    
    def _replan_with_waypoints(self,
                              initial_state: Dict[str, Any],
                              target_state: Dict[str, Any],
                              spacecraft: Dict[str, float],
                              max_time_seconds: float) -> Dict[str, Any]:
        """
        Replan trajectory using intermediate waypoints to avoid obstacles.
        
        Args:
            initial_state: Initial state {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            target_state: Target state {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            spacecraft: Spacecraft parameters
            max_time_seconds: Maximum transfer time
            
        Returns:
            Safe trajectory that avoids obstacles
        """
        # Find closest obstacle that causes collision
        closest_obstacle = None
        min_distance = float('inf')
        
        positions = np.array(self.current_trajectory['trajectory']['positions'])
        
        for obstacle in self.obstacles:
            for pos in positions:
                distance = np.linalg.norm(pos - obstacle['position'])
                if distance < obstacle['radius'] + 500.0 and distance < min_distance:
                    min_distance = distance
                    closest_obstacle = obstacle
        
        if not closest_obstacle:
            return self.current_trajectory
            
        # Generate avoidance waypoint
        avoidance_direction = np.cross(
            np.array(target_state['pos']) - np.array(initial_state['pos']),
            np.array([0, 0, 1])  # Use z-axis for cross product
        )
        
        if np.linalg.norm(avoidance_direction) < 1e-6:
            # If vectors are parallel, use a different axis
            avoidance_direction = np.cross(
                np.array(target_state['pos']) - np.array(initial_state['pos']),
                np.array([0, 1, 0])
            )
            
        avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
        
        # Calculate avoidance distance based on obstacle size
        avoidance_distance = closest_obstacle['radius'] * 2.5
        
        # Create waypoint position
        waypoint_pos = closest_obstacle['position'] + avoidance_direction * avoidance_distance
        
        # Create waypoint state (position and zero velocity for simplicity)
        waypoint_state = {
            'pos': waypoint_pos.tolist(),
            'vel': [0, 0, 0]
        }
        
        # Plan trajectory to waypoint
        trajectory_to_waypoint = self.generate_optimal_trajectory(
            initial_state,
            waypoint_state,
            max_time_seconds / 2,
            spacecraft['max_thrust'],
            spacecraft['isp'],
            spacecraft['mass']
        )
        
        # Plan trajectory from waypoint to target
        final_mass = trajectory_to_waypoint['trajectory']['mass'][-1]
        spacecraft_updated = spacecraft.copy()
        spacecraft_updated['mass'] = final_mass
        
        trajectory_from_waypoint = self.generate_optimal_trajectory(
            waypoint_state,
            target_state,
            max_time_seconds / 2,
            spacecraft['max_thrust'],
            spacecraft['isp'],
            final_mass
        )
        
        # Combine trajectories
        combined_trajectory = self._combine_trajectories(
            trajectory_to_waypoint,
            trajectory_from_waypoint
        )
        
        self.current_trajectory = combined_trajectory
        
        # Check if new trajectory is safe
        if self.check_trajectory_safety():
            return combined_trajectory
            
        # If still not safe, try with more waypoints or different approach
        # For simplicity, we'll just return the combined trajectory
        return combined_trajectory
    
    def _combine_trajectories(self, traj1: Dict[str, Any], traj2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine two trajectory segments into one continuous trajectory.
        
        Args:
            traj1: First trajectory segment
            traj2: Second trajectory segment
            
        Returns:
            Combined trajectory
        """
        # Get trajectory data
        times1 = np.array(traj1['trajectory']['times'])
        positions1 = np.array(traj1['trajectory']['positions'])
        velocities1 = np.array(traj1['trajectory']['velocities'])
        mass1 = np.array(traj1['trajectory']['mass'])
        thrust_vectors1 = np.array(traj1['controls']['thrust_vectors'])
        thrust_magnitudes1 = np.array(traj1['controls']['thrust_magnitudes'])
        
        times2 = np.array(traj2['trajectory']['times'])
        positions2 = np.array(traj2['trajectory']['positions'])
        velocities2 = np.array(traj2['trajectory']['velocities'])
        mass2 = np.array(traj2['trajectory']['mass'])
        thrust_vectors2 = np.array(traj2['controls']['thrust_vectors'])
        thrust_magnitudes2 = np.array(traj2['controls']['thrust_magnitudes'])
        
        # Adjust times for second trajectory to continue from first
        times2 = times2 + times1[-1]
        
        # Combine arrays
        times = np.concatenate([times1, times2[1:]])
        positions = np.concatenate([positions1, positions2[1:]])
        velocities = np.concatenate([velocities1, velocities2[1:]])
        mass = np.concatenate([mass1, mass2[1:]])
        thrust_vectors = np.concatenate([thrust_vectors1, thrust_vectors2])
        thrust_magnitudes = np.concatenate([thrust_magnitudes1, thrust_magnitudes2])
        
        # Calculate combined performance metrics
        propellant_used = traj1['performance']['propellant_used_kg'] + traj2['performance']['propellant_used_kg']
        delta_v = traj1['performance']['delta_v_mps'] + traj2['performance']['delta_v_mps']
        
        # Create combined trajectory
        combined = {
            'success': traj1['success'] and traj2['success'],
            'trajectory': {
                'times': times.tolist(),
                'positions': positions.tolist(),
                'velocities': velocities.tolist(),
                'mass': mass.tolist()
            },
            'controls': {
                'thrust_vectors': thrust_vectors.tolist(),
                'thrust_magnitudes': thrust_magnitudes.tolist()
            },
            'performance': {
                'propellant_used_kg': propellant_used,
                'delta_v_mps': delta_v,
                'final_position_error_m': traj2['performance']['final_position_error_m'],
                'final_velocity_error_mps': traj2['performance']['final_velocity_error_mps']
            }
        }
        
        return combined
    
    def multi_constraint_optimization(self,
                                     initial_state: Dict[str, Any],
                                     target_state: Dict[str, Any],
                                     spacecraft: Dict[str, float],
                                     constraints: Dict[str, Any],
                                     max_time_seconds: float = 86400.0) -> Dict[str, Any]:
        """
        Generate a trajectory that optimizes for multiple constraints simultaneously.
        
        Args:
            initial_state: Initial state {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            target_state: Target state {'pos': [x,y,z], 'vel': [vx,vy,vz]}
            spacecraft: Spacecraft parameters {'mass': kg, 'max_thrust': N, 'isp': s}
            constraints: Dictionary of constraints with weights:
                         {'energy': weight, 'time': weight, 'safety': weight, 'radiation': weight}
            max_time_seconds: Maximum transfer time in seconds
            
        Returns:
            Optimal trajectory satisfying multiple constraints
        """
        # Normalize constraint weights
        total_weight = sum(constraints.values())
        normalized_constraints = {k: v/total_weight for k, v in constraints.items()}
        
        # Generate baseline trajectory
        baseline = self.generate_optimal_trajectory(
            initial_state,
            target_state,
            max_time_seconds,
            spacecraft['max_thrust'],
            spacecraft['isp'],
            spacecraft['mass']
        )
        
        # If energy is the only constraint with significant weight, return baseline
        if normalized_constraints.get('energy', 0) > 0.8:
            return baseline
        
        # Initialize best trajectory and score
        best_trajectory = baseline
        best_score = self._evaluate_trajectory(baseline, normalized_constraints)
        
        # Try different time constraints if time is a factor
        if normalized_constraints.get('time', 0) > 0.1:
            time_factors = [0.7, 0.5, 0.3]
            for factor in time_factors:
                candidate = self.generate_optimal_trajectory(
                    initial_state,
                    target_state,
                    max_time_seconds * factor,
                    spacecraft['max_thrust'],
                    spacecraft['isp'],
                    spacecraft['mass']
                )
                
                score = self._evaluate_trajectory(candidate, normalized_constraints)
                if score > best_score and candidate['success']:
                    best_trajectory = candidate
                    best_score = score
        
        # Consider safety if it's a constraint
        if normalized_constraints.get('safety', 0) > 0.1 and self.obstacles:
            # Try trajectory with obstacle avoidance
            safety_trajectory = self.avoid_obstacles(
                initial_state,
                target_state,
                spacecraft,
                max_time_seconds
            )
            
            safety_score = self._evaluate_trajectory(safety_trajectory, normalized_constraints)
            if safety_score > best_score and safety_trajectory['success']:
                best_trajectory = safety_trajectory
                best_score = safety_score
        
        # Consider radiation exposure if it's a constraint
        if normalized_constraints.get('radiation', 0) > 0.1:
            # Generate trajectory that minimizes radiation exposure
            radiation_trajectory = self._minimize_radiation_exposure(
                initial_state,
                target_state,
                spacecraft,
                max_time_seconds
            )
            
            radiation_score = self._evaluate_trajectory(radiation_trajectory, normalized_constraints)
            if radiation_score > best_score and radiation_trajectory['success']:
                best_trajectory = radiation_trajectory
                best_score = radiation_score
        
        return best_trajectory
    
    def _evaluate_trajectory(self, trajectory: Dict[str, Any], constraints: Dict[str, float]) -> float:
        """
        Evaluate a trajectory against multiple constraints.
        
        Args:
            trajectory: Trajectory data
            constraints: Normalized constraint weights
            
        Returns:
            Score (higher is better)
        """
        score = 0.0
        
        # Energy efficiency score (inverse of propellant used)
        if 'energy' in constraints:
            propellant_used = trajectory['performance']['propellant_used_kg']
            energy_score = 1.0 / (1.0 + propellant_used)
            score += constraints['energy'] * energy_score
        
        # Time efficiency score
        if 'time' in constraints:
            total_time = trajectory['trajectory']['times'][-1]
            time_score = 1.0 / (1.0 + total_time / 86400.0)  # Normalize by day
            score += constraints['time'] * time_score
        
        # Safety score (distance from obstacles)
        if 'safety' in constraints and self.obstacles:
            min_distance = float('inf')
            positions = np.array(trajectory['trajectory']['positions'])
            
            for obstacle in self.obstacles:
                for pos in positions:
                    distance = np.linalg.norm(pos - obstacle['position'])
                    min_distance = min(min_distance, distance)
            
            # Normalize by obstacle radius (higher is better)
            safety_margin = min_distance / 1000.0  # Normalize by km
            safety_score = 1.0 - 1.0 / (1.0 + safety_margin)
            score += constraints['safety'] * safety_score
        
        # Radiation exposure score
        if 'radiation' in constraints:
            # Simplified radiation model based on trajectory
            radiation_score = self._calculate_radiation_score(trajectory)
            score += constraints['radiation'] * radiation_score
        
        return score
    
    def _calculate_radiation_score(self, trajectory: Dict[str, Any]) -> float:
        """
        Calculate radiation exposure score for a trajectory.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Radiation score (higher is better, less exposure)
        """
        # Simplified radiation model
        # Assume radiation decreases with distance from Earth
        positions = np.array(trajectory['trajectory']['positions'])
        distances = np.linalg.norm(positions, axis=1)
        
        # Calculate average distance (higher is better for radiation)
        avg_distance = np.mean(distances)
        
        # Normalize by Earth radius (6371 km)
        normalized_distance = avg_distance / 6371000.0
        
        # Score inversely proportional to radiation exposure
        return 1.0 - 1.0 / (1.0 + normalized_distance)
    
    def _minimize_radiation_exposure(self,
                                    initial_state: Dict[str, Any],
                                    target_state: Dict[str, Any],
                                    spacecraft: Dict[str, float],
                                    max_time_seconds: float) -> Dict[str, Any]:
        """
        Generate trajectory that minimizes radiation exposure.
        
        Args:
            initial_state: Initial state
            target_state: Target state
            spacecraft: Spacecraft parameters
            max_time_seconds: Maximum transfer time
            
        Returns:
            Trajectory with minimized radiation exposure
        """
        # For radiation minimization, we want to stay farther from Earth
        # Create a modified target state that takes a higher path
        
        # Calculate direction vector from Earth to target
        target_pos = np.array(target_state['pos'])
        target_dir = target_pos / np.linalg.norm(target_pos)
        
        # Create waypoint at 20% higher altitude
        waypoint_pos = target_pos * 1.2
        waypoint_state = {
            'pos': waypoint_pos.tolist(),
            'vel': [0, 0, 0]  # Zero velocity at waypoint
        }
        
        # Plan trajectory to waypoint
        trajectory_to_waypoint = self.generate_optimal_trajectory(
            initial_state,
            waypoint_state,
            max_time_seconds / 2,
            spacecraft['max_thrust'],
            spacecraft['isp'],
            spacecraft['mass']
        )
        
        # Plan trajectory from waypoint to target
        final_mass = trajectory_to_waypoint['trajectory']['mass'][-1]
        spacecraft_updated = spacecraft.copy()
        spacecraft_updated['mass'] = final_mass
        
        trajectory_from_waypoint = self.generate_optimal_trajectory(
            waypoint_state,
            target_state,
            max_time_seconds / 2,
            spacecraft['max_thrust'],
            spacecraft['isp'],
            final_mass
        )
        
        # Combine trajectories
        combined_trajectory = self._combine_trajectories(
            trajectory_to_waypoint,
            trajectory_from_waypoint
        )
        
        return combined_trajectory