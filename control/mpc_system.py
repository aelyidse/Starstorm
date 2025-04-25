from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import numpy as np
from .model_predictive_control import ModelPredictiveController

class MPCSystem:
    """
    System for managing multiple Model Predictive Controllers.
    Provides a framework for predictive control of multiple system parameters.
    """
    def __init__(self):
        self.controllers: Dict[str, ModelPredictiveController] = {}
        self.state_functions: Dict[str, Callable[[], np.ndarray]] = {}
        self.reference_functions: Dict[str, Callable[[], np.ndarray]] = {}
        self.control_functions: Dict[str, Callable[[np.ndarray], None]] = {}
        # Adaptive parameters
        self.adaptation_enabled: Dict[str, bool] = {}
        self.adaptation_metrics: Dict[str, List[float]] = {}
        self.adaptation_thresholds: Dict[str, float] = {}
    
    def add_controller(self,
                      name: str,
                      controller: ModelPredictiveController,
                      state_function: Callable[[], np.ndarray],
                      reference_function: Callable[[], np.ndarray],
                      control_function: Callable[[np.ndarray], None]):
        """
        Add an MPC controller to the system
        
        Args:
            name: Unique name for the controller
            controller: MPC controller instance
            state_function: Function that returns the current state vector
            reference_function: Function that returns the reference trajectory
            control_function: Function that applies the control output
        """
        self.controllers[name] = controller
        self.state_functions[name] = state_function
        self.reference_functions[name] = reference_function
        self.control_functions[name] = control_function
    
    def update(self, controller_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Update one or all controllers
        
        Args:
            controller_name: Name of specific controller to update, or None for all
            
        Returns:
            Dictionary of controller names and their outputs
        """
        results = {}
        
        if controller_name:
            # Update specific controller
            if controller_name not in self.controllers:
                raise ValueError(f"Controller '{controller_name}' not found")
                
            controller = self.controllers[controller_name]
            state = self.state_functions[controller_name]()
            reference = self.reference_functions[controller_name]()
            control = controller.compute_control(state, reference)
            self.control_functions[controller_name](control)
            results[controller_name] = control
        else:
            # Update all controllers
            for name, controller in self.controllers.items():
                state = self.state_functions[name]()
                reference = self.reference_functions[name]()
                control = controller.compute_control(state, reference)
                self.control_functions[name](control)
                results[name] = control
                
        return results
    
    def get_predictions(self, controller_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Get state predictions for one or all controllers
        
        Args:
            controller_name: Name of specific controller, or None for all
            
        Returns:
            Dictionary of controller names and their predicted state trajectories
        """
        predictions = {}
        
        if controller_name:
            # Get prediction for specific controller
            if controller_name not in self.controllers:
                raise ValueError(f"Controller '{controller_name}' not found")
                
            controller = self.controllers[controller_name]
            state = self.state_functions[controller_name]()
            reference = self.reference_functions[controller_name]()
            
            # Optimize to get control sequence
            control_sequence, _ = controller.optimize(state, reference)
            
            # Get prediction
            prediction = controller.get_prediction(state, control_sequence)
            predictions[controller_name] = prediction
        else:
            # Get predictions for all controllers
            for name, controller in self.controllers.items():
                state = self.state_functions[name]()
                reference = self.reference_functions[name]()
                
                # Optimize to get control sequence
                control_sequence, _ = controller.optimize(state, reference)
                
                # Get prediction
                prediction = controller.get_prediction(state, control_sequence)
                predictions[name] = prediction
                
        return predictions
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all controllers"""
        return {name: controller.get_metrics() for name, controller in self.controllers.items()}
    
    def enable_adaptation(self, controller_name: str, threshold: float = 0.1):
        """
        Enable adaptive control for a specific controller
        
        Args:
            controller_name: Name of the controller to adapt
            threshold: Performance threshold that triggers adaptation
        """
        if controller_name not in self.controllers:
            raise ValueError(f"Controller '{controller_name}' not found")
            
        self.adaptation_enabled[controller_name] = True
        self.adaptation_thresholds[controller_name] = threshold
        self.adaptation_metrics[controller_name] = []
    
    def disable_adaptation(self, controller_name: str):
        """Disable adaptive control for a specific controller"""
        if controller_name in self.adaptation_enabled:
            self.adaptation_enabled[controller_name] = False
    
    def update_with_adaptation(self, controller_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Update controllers with adaptive parameter tuning
        
        Args:
            controller_name: Name of specific controller to update, or None for all
            
        Returns:
            Dictionary of controller names and their outputs
        """
        results = {}
        
        controllers_to_update = [controller_name] if controller_name else self.controllers.keys()
        
        for name in controllers_to_update:
            if name not in self.controllers:
                continue
                
            controller = self.controllers[name]
            state = self.state_functions[name]()
            reference = self.reference_functions[name]()
            
            # Check if adaptation is enabled for this controller
            if self.adaptation_enabled.get(name, False):
                # Get current performance metrics
                metrics = controller.get_metrics()
                
                # Store performance metric (using prediction error as example)
                if 'avg_prediction_error' in metrics:
                    self.adaptation_metrics[name].append(metrics['avg_prediction_error'])
                
                # Adapt controller parameters if needed
                if len(self.adaptation_metrics[name]) >= 5:  # Need enough history
                    self._adapt_controller_parameters(name, controller)
            
            # Compute control action and apply it
            control = controller.compute_control(state, reference)
            self.control_functions[name](control)
            results[name] = control
                
        return results
    
    def _adapt_controller_parameters(self, name: str, controller: ModelPredictiveController):
        """
        Adapt controller parameters based on performance metrics
        
        Args:
            name: Controller name
            controller: MPC controller instance
        """
        # Get recent performance metrics
        recent_metrics = self.adaptation_metrics[name][-5:]
        avg_metric = sum(recent_metrics) / len(recent_metrics)
        
        # Check if adaptation is needed
        if avg_metric > self.adaptation_thresholds.get(name, 0.1):
            # Example adaptation: adjust cost matrices
            if hasattr(controller, 'Q') and hasattr(controller, 'R'):
                # Increase state tracking weight if error is high
                controller.Q = controller.Q * 1.05
                
                # Decrease control effort penalty
                controller.R = controller.R * 0.95
                
                # Ensure matrices remain positive definite
                min_eigenvalue_Q = np.min(np.linalg.eigvals(controller.Q))
                if min_eigenvalue_Q <= 0:
                    controller.Q = controller.Q - min_eigenvalue_Q * np.eye(controller.Q.shape[0]) + 0.01 * np.eye(controller.Q.shape[0])
                
                min_eigenvalue_R = np.min(np.linalg.eigvals(controller.R))
                if min_eigenvalue_R <= 0:
                    controller.R = controller.R - min_eigenvalue_R * np.eye(controller.R.shape[0]) + 0.01 * np.eye(controller.R.shape[0])
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get the status of all adaptive controllers"""
        status = {}
        for name in self.controllers:
            if name in self.adaptation_enabled and self.adaptation_enabled[name]:
                status[name] = {
                    'enabled': True,
                    'threshold': self.adaptation_thresholds.get(name, 0.0),
                    'metrics_history': self.adaptation_metrics.get(name, []),
                    'current_metric': self.adaptation_metrics[name][-1] if name in self.adaptation_metrics and self.adaptation_metrics[name] else None
                }
            else:
                status[name] = {'enabled': False}
        return status