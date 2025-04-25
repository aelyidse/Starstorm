import numpy as np
import time
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import deque

class ControlSystemTuner:
    """
    Provides utilities for tuning control systems automatically.
    Implements various tuning methods for PID controllers and other control systems.
    """
    def __init__(self):
        self.tuning_history = {}
        self.performance_metrics = {}
        
    def tune_pid_ziegler_nichols(self, 
                                process_function: Callable[[float], float],
                                initial_gain: float = 0.1,
                                max_gain: float = 100.0,
                                settling_time: float = 5.0,
                                sample_time: float = 0.01) -> Dict[str, float]:
        """
        Tune PID controller using the Ziegler-Nichols method
        
        Args:
            process_function: Function that simulates the process response to a control input
            initial_gain: Initial proportional gain to start with
            max_gain: Maximum allowable gain
            settling_time: Time to wait for process to settle
            sample_time: Sampling time for the simulation
            
        Returns:
            Dictionary with tuned PID parameters (kp, ki, kd)
        """
        # Step 1: Find the ultimate gain (Ku) and ultimate period (Tu)
        ku, tu = self._find_ultimate_gain_and_period(
            process_function, initial_gain, max_gain, settling_time, sample_time
        )
        
        # Step 2: Calculate PID parameters using Ziegler-Nichols rules
        kp = 0.6 * ku
        ki = 1.2 * ku / tu
        kd = 0.075 * ku * tu
        
        # Store tuning results
        method = "ziegler_nichols"
        self.tuning_history[method] = {
            'ku': ku,
            'tu': tu,
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'timestamp': time.time()
        }
        
        return {'kp': kp, 'ki': ki, 'kd': kd}
    
    def _find_ultimate_gain_and_period(self, 
                                      process_function: Callable[[float], float],
                                      initial_gain: float,
                                      max_gain: float,
                                      settling_time: float,
                                      sample_time: float) -> Tuple[float, float]:
        """
        Find the ultimate gain and period using relay feedback method
        
        Args:
            process_function: Process simulation function
            initial_gain: Initial gain to start with
            max_gain: Maximum allowable gain
            settling_time: Time to wait for process to settle
            sample_time: Sampling time
            
        Returns:
            Tuple of (ultimate gain, ultimate period)
        """
        # Implementation of relay feedback method
        relay_amplitude = 1.0
        setpoint = 0.0
        process_value = 0.0
        last_process_value = 0.0
        relay_output = relay_amplitude
        
        # For detecting oscillations
        crossings = []
        
        # Run simulation
        t = 0.0
        while t < settling_time:
            # Apply relay feedback
            error = setpoint - process_value
            relay_output = relay_amplitude if error > 0 else -relay_amplitude
            
            # Get process response
            process_value = process_function(relay_output)
            
            # Detect zero crossings (when process crosses setpoint)
            if (last_process_value < setpoint and process_value >= setpoint) or \
               (last_process_value > setpoint and process_value <= setpoint):
                crossings.append(t)
            
            last_process_value = process_value
            t += sample_time
        
        # Need at least 3 crossings to calculate period
        if len(crossings) < 3:
            return 1.0, 1.0  # Default values if method fails
        
        # Calculate average period from zero crossings
        periods = [crossings[i+1] - crossings[i] for i in range(len(crossings)-1)]
        tu = 2 * sum(periods) / len(periods)  # Multiply by 2 because half-period
        
        # Calculate ultimate gain
        ku = (4 * relay_amplitude) / (np.pi * process_value)
        
        return abs(ku), tu
    
    def tune_pid_cohen_coon(self, 
                           process_function: Callable[[float], float],
                           step_size: float = 1.0,
                           settling_time: float = 10.0,
                           sample_time: float = 0.01) -> Dict[str, float]:
        """
        Tune PID controller using the Cohen-Coon method
        
        Args:
            process_function: Function that simulates the process response to a control input
            step_size: Size of step input for process identification
            settling_time: Time to wait for process to settle
            sample_time: Sampling time for the simulation
            
        Returns:
            Dictionary with tuned PID parameters (kp, ki, kd)
        """
        # Step 1: Apply step input and record response
        t_values = []
        pv_values = []
        
        t = 0.0
        while t < settling_time:
            process_value = process_function(step_size)
            t_values.append(t)
            pv_values.append(process_value)
            t += sample_time
        
        # Step 2: Identify process parameters (gain, time constant, delay)
        k, tau, theta = self._identify_fopdt_model(t_values, pv_values, step_size)
        
        # Step 3: Calculate PID parameters using Cohen-Coon rules
        r = theta / tau
        
        kp = (1.35 / k) * (1 + 0.18 * r)
        ki = kp / (tau * (2.5 - 2 * r))
        kd = kp * tau * 0.37 * r / (1 + 0.2 * r)
        
        # Store tuning results
        method = "cohen_coon"
        self.tuning_history[method] = {
            'k': k,
            'tau': tau,
            'theta': theta,
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'timestamp': time.time()
        }
        
        return {'kp': kp, 'ki': ki, 'kd': kd}
    
    def _identify_fopdt_model(self, 
                             t_values: List[float], 
                             pv_values: List[float],
                             step_size: float) -> Tuple[float, float, float]:
        """
        Identify First Order Plus Dead Time (FOPDT) model parameters
        
        Args:
            t_values: Time values
            pv_values: Process values
            step_size: Size of step input
            
        Returns:
            Tuple of (process gain, time constant, delay)
        """
        # Convert to numpy arrays
        t = np.array(t_values)
        pv = np.array(pv_values)
        
        # Calculate steady-state gain
        k = (pv[-1] - pv[0]) / step_size
        
        # Find time when process reaches 28.3% and 63.2% of final value
        pv_28 = pv[0] + 0.283 * (pv[-1] - pv[0])
        pv_63 = pv[0] + 0.632 * (pv[-1] - pv[0])
        
        t_28 = t[np.argmin(np.abs(pv - pv_28))]
        t_63 = t[np.argmin(np.abs(pv - pv_63))]
        
        # Calculate time constant and delay
        theta = 1.5 * (t_28 - t[0])
        tau = 1.5 * (t_63 - t_28)
        
        return k, tau, theta
    
    def tune_pid_iterative(self, 
                          process_function: Callable[[float, float, float, float], float],
                          objective_function: Callable[[List[float]], float],
                          initial_params: Dict[str, float] = None,
                          bounds: Dict[str, Tuple[float, float]] = None,
                          max_iterations: int = 50) -> Dict[str, float]:
        """
        Tune PID controller using iterative optimization
        
        Args:
            process_function: Function that simulates process with PID parameters
            objective_function: Function that evaluates performance of parameters
            initial_params: Initial PID parameters to start optimization
            bounds: Parameter bounds as dictionary of (min, max) tuples
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary with optimized PID parameters
        """
        # Default initial parameters
        if initial_params is None:
            initial_params = {'kp': 1.0, 'ki': 0.1, 'kd': 0.1}
            
        # Default bounds
        if bounds is None:
            bounds = {
                'kp': (0.01, 100.0),
                'ki': (0.0, 50.0),
                'kd': (0.0, 50.0)
            }
            
        # Convert to format for scipy optimizer
        x0 = [initial_params['kp'], initial_params['ki'], initial_params['kd']]
        param_bounds = [bounds['kp'], bounds['ki'], bounds['kd']]
        
        # Define objective function wrapper
        def objective_wrapper(x):
            return objective_function([x[0], x[1], x[2]])
        
        # Run optimization
        result = minimize(
            objective_wrapper,
            x0,
            method='L-BFGS-B',
            bounds=param_bounds,
            options={'maxiter': max_iterations}
        )
        
        # Extract optimized parameters
        kp, ki, kd = result.x
        
        # Store tuning results
        method = "iterative_optimization"
        self.tuning_history[method] = {
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'objective_value': result.fun,
            'iterations': result.nit,
            'timestamp': time.time()
        }
        
        return {'kp': kp, 'ki': ki, 'kd': kd}
    
    def auto_tune_mpc_weights(self, 
                             mpc_controller,
                             simulation_function: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
                             initial_state: np.ndarray,
                             reference: np.ndarray,
                             state_weight_range: Tuple[float, float] = (0.1, 10.0),
                             control_weight_range: Tuple[float, float] = (0.01, 1.0),
                             max_iterations: int = 20) -> Dict[str, np.ndarray]:
        """
        Automatically tune MPC controller weights (Q and R matrices)
        
        Args:
            mpc_controller: MPC controller instance
            simulation_function: Function to simulate system with MPC controller
            initial_state: Initial system state
            reference: Reference trajectory
            state_weight_range: Range for state weights
            control_weight_range: Range for control weights
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimized Q and R matrices
        """
        # Get dimensions
        state_dim = initial_state.shape[0]
        if hasattr(mpc_controller, 'R'):
            control_dim = mpc_controller.R.shape[0]
        else:
            control_dim = 1  # Default if not available
            
        # Define objective function for optimization
        def objective(weights):
            # Extract weights for Q and R matrices
            q_weights = weights[:state_dim]
            r_weights = weights[state_dim:state_dim+control_dim]
            
            # Create diagonal Q and R matrices
            Q = np.diag(q_weights)
            R = np.diag(r_weights)
            
            # Update controller
            mpc_controller.set_cost_matrices(Q, R)
            
            # Simulate system and get performance metric
            performance, _ = simulation_function(initial_state, reference)
            
            # Return negative performance (minimize)
            return -performance
            
        # Initial weights (diagonal elements of Q and R)
        initial_q = np.ones(state_dim) * (state_weight_range[0] + state_weight_range[1]) / 2
        initial_r = np.ones(control_dim) * (control_weight_range[0] + control_weight_range[1]) / 2
        initial_weights = np.concatenate([initial_q, initial_r])
        
        # Define bounds
        q_bounds = [(state_weight_range[0], state_weight_range[1]) for _ in range(state_dim)]
        r_bounds = [(control_weight_range[0], control_weight_range[1]) for _ in range(control_dim)]
        bounds = q_bounds + r_bounds
        
        # Run optimization
        result = minimize(
            objective,
            initial_weights,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations}
        )
        
        # Extract optimized weights
        optimized_weights = result.x
        q_weights = optimized_weights[:state_dim]
        r_weights = optimized_weights[state_dim:state_dim+control_dim]
        
        # Create final Q and R matrices
        Q_optimal = np.diag(q_weights)
        R_optimal = np.diag(r_weights)
        
        # Update controller with optimal weights
        mpc_controller.set_cost_matrices(Q_optimal, R_optimal)
        
        # Store tuning results
        method = "mpc_weight_optimization"
        self.tuning_history[method] = {
            'Q': Q_optimal.tolist(),
            'R': R_optimal.tolist(),
            'objective_value': result.fun,
            'iterations': result.nit,
            'timestamp': time.time()
        }
        
        return {'Q': Q_optimal, 'R': R_optimal}
    
    def evaluate_controller_performance(self, 
                                       simulation_function: Callable[[], Tuple[List[float], List[float], List[float]]],
                                       metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate controller performance using standard metrics
        
        Args:
            simulation_function: Function that returns time, setpoint, and process value arrays
            metrics: List of metrics to calculate (defaults to all)
            
        Returns:
            Dictionary of performance metrics
        """
        # Default metrics
        if metrics is None:
            metrics = ['iae', 'ise', 'itae', 'rise_time', 'settling_time', 'overshoot']
            
        # Run simulation
        time_values, setpoint_values, process_values = simulation_function()
        
        # Calculate requested metrics
        results = {}
        
        if 'iae' in metrics:
            # Integral of Absolute Error
            error = np.abs(np.array(setpoint_values) - np.array(process_values))
            results['iae'] = np.trapz(error, time_values)
            
        if 'ise' in metrics:
            # Integral of Squared Error
            error = np.square(np.array(setpoint_values) - np.array(process_values))
            results['ise'] = np.trapz(error, time_values)
            
        if 'itae' in metrics:
            # Integral of Time-weighted Absolute Error
            error = np.array(time_values) * np.abs(np.array(setpoint_values) - np.array(process_values))
            results['itae'] = np.trapz(error, time_values)
            
        if 'rise_time' in metrics:
            # Rise time (10% to 90%)
            sp_change = setpoint_values[-1] - setpoint_values[0]
            if sp_change != 0:
                threshold_10 = setpoint_values[0] + 0.1 * sp_change
                threshold_90 = setpoint_values[0] + 0.9 * sp_change
                
                t_10 = time_values[0]
                t_90 = time_values[-1]
                
                for i in range(len(time_values)):
                    if process_values[i] >= threshold_10 and t_10 == time_values[0]:
                        t_10 = time_values[i]
                    if process_values[i] >= threshold_90:
                        t_90 = time_values[i]
                        break
                        
                results['rise_time'] = t_90 - t_10
            else:
                results['rise_time'] = 0.0
                
        if 'settling_time' in metrics:
            # Settling time (time to reach and stay within 2% of final value)
            final_value = setpoint_values[-1]
            threshold = 0.02 * abs(final_value)
            
            settled = False
            settling_time = time_values[-1]
            
            for i in range(len(time_values) - 1, 0, -1):
                if abs(process_values[i] - final_value) > threshold:
                    settling_time = time_values[i]
                    break
                    
            results['settling_time'] = settling_time
            
        if 'overshoot' in metrics:
            # Percent overshoot
            sp_change = setpoint_values[-1] - setpoint_values[0]
            if sp_change != 0:
                max_value = max(process_values)
                if max_value > setpoint_values[-1]:
                    results['overshoot'] = 100.0 * (max_value - setpoint_values[-1]) / sp_change
                else:
                    results['overshoot'] = 0.0
            else:
                results['overshoot'] = 0.0
                
        # Store performance metrics
        self.performance_metrics = results
        
        return results
    
    def plot_tuning_comparison(self, 
                              simulation_functions: Dict[str, Callable[[], Tuple[List[float], List[float], List[float]]]],
                              save_path: Optional[str] = None):
        """
        Plot comparison of different tuning methods
        
        Args:
            simulation_functions: Dictionary mapping tuning method names to simulation functions
            save_path: Optional path to save the plot
            
        Returns:
            True if successful, False otherwise
        """
        plt.figure(figsize=(12, 8))
        
        # Plot each tuning method
        for method_name, sim_func in simulation_functions.items():
            time_values, setpoint_values, process_values = sim_func()
            plt.plot(time_values, process_values, label=f"Method: {method_name}")
            
        # Plot setpoint
        time_values, setpoint_values, _ = next(iter(simulation_functions.values()))()
        plt.plot(time_values, setpoint_values, 'k--', label="Setpoint")
        
        plt.title("Tuning Method Comparison")
        plt.xlabel("Time")
        plt.ylabel("Process Variable")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return True
        else:
            plt.show()
            return True
    
    def get_tuning_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the history of all tuning operations"""
        return self.tuning_history
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get the last calculated performance metrics"""
        return self.performance_metrics


class AdaptiveTuner:
    """
    Implements adaptive tuning for control systems that automatically
    adjusts parameters based on performance metrics and operating conditions.
    """
    def __init__(self, adaptation_rate: float = 0.1, history_length: int = 20):
        self.adaptation_rate = adaptation_rate
        self.parameter_history = {}
        self.performance_history = {}
        self.history_length = history_length
        self.operating_regions = []
        
    def register_parameter(self, name: str, initial_value: float, 
                          min_value: float, max_value: float):
        """Register a control parameter for adaptive tuning"""
        self.parameter_history[name] = deque(maxlen=self.history_length)
        self.parameter_history[name].append(initial_value)
        self.operating_regions.append({
            'parameter': name,
            'min_value': min_value,
            'max_value': max_value,
            'optimal_values': {}  # Will store optimal values for different operating regions
        })
        
    def record_performance(self, metric_name: str, value: float):
        """Record a performance metric for adaptation decisions"""
        if metric_name not in self.performance_history:
            self.performance_history[metric_name] = deque(maxlen=self.history_length)
        self.performance_history[metric_name].append(value)
        
    def adapt_parameters(self, 
                        current_operating_point: Dict[str, float],
                        performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Adapt control parameters based on current performance and operating point
        
        Args:
            current_operating_point: Current operating conditions
            performance_metrics: Current performance metrics
            
        Returns:
            Dictionary of adapted parameters
        """
        adapted_params = {}
        
        # Find closest operating region or create new one
        region_idx = self._find_operating_region(current_operating_point)
        
        for param_name in self.parameter_history:
            current_value = self.parameter_history[param_name][-1]
            
            # Check if we have optimal values for this region
            if region_idx is not None and param_name in self.operating_regions[region_idx]['optimal_values']:
                # Move toward known optimal value for this region
                optimal_value = self.operating_regions[region_idx]['optimal_values'][param_name]
                new_value = current_value + self.adaptation_rate * (optimal_value - current_value)
            else:
                # Use performance gradient to adapt
                new_value = self._adapt_using_performance_gradient(param_name, current_value, performance_metrics)
            
            # Ensure value is within bounds
            for region in self.operating_regions:
                if region['parameter'] == param_name:
                    new_value = max(region['min_value'], min(region['max_value'], new_value))
            
            # Store new value
            self.parameter_history[param_name].append(new_value)
            adapted_params[param_name] = new_value
            
            # Update optimal value for this region if performance improved
            if region_idx is not None and self._is_performance_improved(performance_metrics):
                self.operating_regions[region_idx]['optimal_values'][param_name] = new_value
        
        return adapted_params
    
    def _find_operating_region(self, operating_point: Dict[str, float]) -> Optional[int]:
        """Find the closest operating region or return None if none exists"""
        # Implementation would depend on how operating regions are defined
        # This is a placeholder for a more sophisticated implementation
        return 0 if self.operating_regions else None
    
    def _adapt_using_performance_gradient(self, 
                                         param_name: str, 
                                         current_value: float,
                                         performance_metrics: Dict[str, float]) -> float:
        """Adapt parameter using performance gradient"""
        # Simple implementation - could be more sophisticated
        # Use IAE or ISE as primary metric if available
        metric_name = 'iae' if 'iae' in performance_metrics else 'ise' if 'ise' in performance_metrics else None
        
        if metric_name is None or len(self.performance_history.get(metric_name, [])) < 2:
            return current_value  # Not enough history
            
        # Get last two performance values
        current_perf = self.performance_history[metric_name][-1]
        prev_perf = self.performance_history[metric_name][-2]
        
        # Get last parameter change
        if len(self.parameter_history[param_name]) < 2:
            return current_value  # Not enough history
            
        current_param = self.parameter_history[param_name][-1]
        prev_param = self.parameter_history[param_name][-2]
        
        # Calculate gradients
        param_change = current_param - prev_param
        perf_change = current_perf - prev_perf
        
        # If performance improved (lower error), continue in same direction
        if perf_change < 0 and param_change != 0:
            # Continue in same direction with adaptation rate
            direction = param_change / abs(param_change)
            return current_value + direction * self.adaptation_rate * abs(current_value)
        elif perf_change > 0 and param_change != 0:
            # Reverse direction with adaptation rate
            direction = -param_change / abs(param_change)
            return current_value + direction * self.adaptation_rate * abs(current_value)
        else:
            # No change or no clear direction, make small random change
            import random
            direction = 1 if random.random() > 0.5 else -1
            return current_value + direction * self.adaptation_rate * abs(current_value) * 0.1
    
    def _is_performance_improved(self, performance_metrics: Dict[str, float]) -> bool:
        """Check if performance has improved compared to history"""
        # Use IAE or ISE as primary metric if available
        metric_name = 'iae' if 'iae' in performance_metrics else 'ise' if 'ise' in performance_metrics else None
        
        if metric_name is None or metric_name not in self.performance_history or len(self.performance_history[metric_name]) < 2:
            return False  # Not enough history
            
        current_perf = performance_metrics[metric_name]
        prev_perf = self.performance_history[metric_name][-1]
        
        # Lower error metrics indicate better performance
        return current_perf < prev_perf
    
    def get_parameter_history(self, param_name: str) -> List[float]:
        """Get history of a specific parameter"""
        if param_name in self.parameter_history:
            return list(self.parameter_history[param_name])
        return []
    
    def get_performance_history(self, metric_name: str) -> List[float]:
        """Get history of a specific performance metric"""
        if metric_name in self.performance_history:
            return list(self.performance_history[metric_name])
        return []
    
    def plot_adaptation_history(self, 
                               param_names: List[str], 
                               metric_names: List[str],
                               save_path: Optional[str] = None):
        """
        Plot the history of parameter adaptations and performance metrics
        
        Args:
            param_names: List of parameter names to plot
            metric_names: List of metric names to plot
            save_path: Optional path to save the plot
            
        Returns:
            True if successful, False otherwise
        """
        if not param_names or not all(p in self.parameter_history for p in param_names):
            return False
            
        if not metric_names or not all(m in self.performance_history for m in metric_names):
            return False
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot parameters
        for param_name in param_names:
            values = list(self.parameter_history[param_name])
            ax1.plot(range(len(values)), values, label=param_name)
            
        ax1.set_title("Parameter Adaptation History")
        ax1.set_ylabel("Parameter Value")
        ax1.legend()
        ax1.grid(True)
        
        # Plot metrics
        for metric_name in metric_names:
            values = list(self.performance_history[metric_name])
            ax2.plot(range(len(values)), values, label=metric_name)
            
        ax2.set_title("Performance Metric History")
        ax2.set_xlabel("Adaptation Step")
        ax2.set_ylabel("Metric Value")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return True
        else:
            plt.show()
            return True