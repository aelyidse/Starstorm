from typing import Dict, Any, Optional, Callable, List
from .pid_controller import PIDController

class FeedbackControlSystem:
    """
    Feedback control system that manages multiple PID controllers.
    Provides a framework for closed-loop control of system parameters.
    """
    def __init__(self):
        self.controllers: Dict[str, PIDController] = {}
        self.process_variables: Dict[str, Callable[[], float]] = {}
        self.control_outputs: Dict[str, Callable[[float], None]] = {}
    
    def add_controller(self, 
                      name: str, 
                      controller: PIDController,
                      process_variable_func: Callable[[], float],
                      control_output_func: Callable[[float], None]):
        """
        Add a PID controller to the feedback control system
        
        Args:
            name: Unique name for the controller
            controller: PID controller instance
            process_variable_func: Function that returns the current process variable
            control_output_func: Function that applies the control output
        """
        self.controllers[name] = controller
        self.process_variables[name] = process_variable_func
        self.control_outputs[name] = control_output_func
    
    def update(self, controller_name: Optional[str] = None) -> Dict[str, float]:
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
            pv = self.process_variables[controller_name]()
            output = controller.compute(pv)
            self.control_outputs[controller_name](output)
            results[controller_name] = output
        else:
            # Update all controllers
            for name, controller in self.controllers.items():
                pv = self.process_variables[name]()
                output = controller.compute(pv)
                self.control_outputs[name](output)
                results[name] = output
                
        return results
    
    def reset_controllers(self, controller_names: Optional[List[str]] = None):
        """
        Reset one or more controllers
        
        Args:
            controller_names: List of controller names to reset, or None for all
        """
        if controller_names:
            for name in controller_names:
                if name in self.controllers:
                    self.controllers[name].reset()
        else:
            for controller in self.controllers.values():
                controller.reset()