from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import numpy as np
import time
from scipy.optimize import minimize

class ModelPredictiveController:
    """
    Model Predictive Controller (MPC) for advanced control applications.
    Predicts future system behavior and optimizes control actions over a receding horizon.
    """
    def __init__(self, 
                 prediction_horizon: int,
                 control_horizon: int,
                 sample_time: float,
                 model_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 constraints: Optional[Dict[str, Any]] = None,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None):
        """
        Initialize the MPC controller.
        
        Args:
            prediction_horizon: Number of steps to predict into the future
            control_horizon: Number of control moves to optimize
            sample_time: Time between control updates (seconds)
            model_function: Function that predicts next state given current state and input
            constraints: Dictionary containing state and input constraints
            Q: State error cost matrix
            R: Control effort cost matrix
        """
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.sample_time = sample_time
        self.model_function = model_function
        
        # Default constraints if none provided
        self.constraints = constraints or {
            'input_min': None,
            'input_max': None,
            'state_min': None,
            'state_max': None
        }
        
        # Cost matrices
        self.Q = Q  # State error cost
        self.R = R  # Control effort cost
        
        # Controller memory
        self.last_control = None
        self.last_state = None
        self.last_reference = None
        self.last_solution = None
        self.last_time = time.time()
        
        # Performance metrics
        self.solve_times = []
        self.prediction_errors = []
    
    def set_cost_matrices(self, Q: np.ndarray, R: np.ndarray):
        """Set the cost matrices for the MPC optimization"""
        self.Q = Q
        self.R = R
    
    def set_constraints(self, constraints: Dict[str, Any]):
        """Set the constraints for the MPC optimization"""
        self.constraints = constraints
    
    def _objective_function(self, u_sequence: np.ndarray, current_state: np.ndarray, 
                           reference_trajectory: np.ndarray) -> float:
        """
        Objective function to minimize in the MPC optimization.
        
        Args:
            u_sequence: Flattened sequence of control inputs to optimize
            current_state: Current system state
            reference_trajectory: Desired state trajectory
            
        Returns:
            Cost value to minimize
        """
        # Reshape control sequence
        u_sequence = u_sequence.reshape(self.control_horizon, -1)
        
        # Initialize cost and state
        cost = 0.0
        state = current_state.copy()
        
        # Simulate system over prediction horizon
        for k in range(self.prediction_horizon):
            # Get control input (use last calculated input if beyond control horizon)
            u_k = u_sequence[min(k, self.control_horizon-1)]
            
            # State error cost
            state_error = state - reference_trajectory[k]
            cost += state_error.T @ self.Q @ state_error
            
            # Control effort cost
            if k < self.control_horizon:
                cost += u_k.T @ self.R @ u_k
            
            # Predict next state
            state = self.model_function(state, u_k)
            
        return cost
    
    def _constraint_function(self, u_sequence: np.ndarray, current_state: np.ndarray) -> List[float]:
        """
        Constraint function for the MPC optimization.
        
        Args:
            u_sequence: Flattened sequence of control inputs
            current_state: Current system state
            
        Returns:
            List of constraint violations (negative values indicate satisfied constraints)
        """
        constraints = []
        u_sequence = u_sequence.reshape(self.control_horizon, -1)
        
        # Input constraints
        if self.constraints['input_min'] is not None:
            for k in range(self.control_horizon):
                constraints.extend(self.constraints['input_min'] - u_sequence[k])
                
        if self.constraints['input_max'] is not None:
            for k in range(self.control_horizon):
                constraints.extend(u_sequence[k] - self.constraints['input_max'])
        
        # State constraints require simulating the system
        if self.constraints['state_min'] is not None or self.constraints['state_max'] is not None:
            state = current_state.copy()
            
            for k in range(self.prediction_horizon):
                u_k = u_sequence[min(k, self.control_horizon-1)]
                
                if self.constraints['state_min'] is not None:
                    constraints.extend(self.constraints['state_min'] - state)
                    
                if self.constraints['state_max'] is not None:
                    constraints.extend(state - self.constraints['state_max'])
                
                # Predict next state
                state = self.model_function(state, u_k)
                
        return constraints
    
    def optimize(self, current_state: np.ndarray, reference_trajectory: np.ndarray, 
                initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve the MPC optimization problem.
        
        Args:
            current_state: Current system state
            reference_trajectory: Desired state trajectory over prediction horizon
            initial_guess: Initial guess for control sequence
            
        Returns:
            Tuple of (optimal control sequence, optimization info)
        """
        start_time = time.time()
        
        # Ensure reference trajectory has correct length
        if len(reference_trajectory) < self.prediction_horizon:
            # Pad with last value if reference is too short
            last_ref = reference_trajectory[-1]
            reference_trajectory = np.vstack([
                reference_trajectory, 
                np.tile(last_ref, (self.prediction_horizon - len(reference_trajectory), 1))
            ])
        
        # Prepare initial guess
        input_dim = reference_trajectory.shape[1] if self.R.shape[0] == reference_trajectory.shape[1] else self.R.shape[0]
        if initial_guess is None:
            if self.last_solution is not None:
                # Use previous solution shifted by one step
                initial_guess = np.vstack([
                    self.last_solution[1:],
                    self.last_solution[-1:]
                ])
            else:
                # Start with zeros
                initial_guess = np.zeros((self.control_horizon, input_dim))
        
        # Flatten for optimizer
        initial_guess_flat = initial_guess.flatten()
        
        # Define constraints for optimizer
        constraints = []
        if any(key in self.constraints for key in ['state_min', 'state_max', 'input_min', 'input_max']):
            constraints.append({
                'type': 'ineq',
                'fun': lambda u: -np.array(self._constraint_function(u, current_state))
            })
        
        # Solve optimization problem
        result = minimize(
            self._objective_function,
            initial_guess_flat,
            args=(current_state, reference_trajectory),
            method='SLSQP',
            constraints=constraints,
            options={'disp': False}
        )
        
        # Reshape solution
        optimal_sequence = result.x.reshape(self.control_horizon, -1)
        
        # Store solution for warm start
        self.last_solution = optimal_sequence
        self.last_state = current_state
        self.last_reference = reference_trajectory
        
        # Record solve time
        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)
        
        # Return first control action and info
        return optimal_sequence, {
            'success': result.success,
            'solve_time': solve_time,
            'cost': result.fun,
            'message': result.message
        }
    
    def compute_control(self, current_state: np.ndarray, reference_trajectory: np.ndarray) -> np.ndarray:
        """
        Compute the optimal control action for the current state.
        
        Args:
            current_state: Current system state
            reference_trajectory: Desired state trajectory
            
        Returns:
            Optimal control action (first step of sequence)
        """
        # Solve MPC problem
        optimal_sequence, _ = self.optimize(current_state, reference_trajectory)
        
        # Return first control action
        self.last_control = optimal_sequence[0]
        self.last_time = time.time()
        
        return self.last_control
    
    def get_prediction(self, current_state: np.ndarray, control_sequence: np.ndarray) -> np.ndarray:
        """
        Get the predicted state trajectory for a given control sequence.
        
        Args:
            current_state: Current system state
            control_sequence: Sequence of control inputs
            
        Returns:
            Predicted state trajectory
        """
        state = current_state.copy()
        states = [state]
        
        for k in range(self.prediction_horizon):
            u_k = control_sequence[min(k, len(control_sequence)-1)]
            state = self.model_function(state, u_k)
            states.append(state)
            
        return np.array(states)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get controller performance metrics"""
        if not self.solve_times:
            return {
                'avg_solve_time': 0.0,
                'max_solve_time': 0.0,
                'min_solve_time': 0.0
            }
            
        return {
            'avg_solve_time': np.mean(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'min_solve_time': np.min(self.solve_times),
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0
        }