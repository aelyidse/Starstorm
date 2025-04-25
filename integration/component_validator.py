from typing import Dict, Any, List, Optional, Set, Tuple, Type
import inspect
import importlib
import sys
import os
from pathlib import Path

class ComponentInterfaceDefinition:
    """Defines the expected interface for a component."""
    
    def __init__(self, name: str, required_methods: Set[str], required_attributes: Set[str],
                 expected_inputs: Dict[str, Type], expected_outputs: Dict[str, Type]):
        self.name = name
        self.required_methods = required_methods
        self.required_attributes = required_attributes
        self.expected_inputs = expected_inputs
        self.expected_outputs = expected_outputs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interface definition to dictionary."""
        return {
            'name': self.name,
            'required_methods': list(self.required_methods),
            'required_attributes': list(self.required_attributes),
            'expected_inputs': {k: str(v) for k, v in self.expected_inputs.items()},
            'expected_outputs': {k: str(v) for k, v in self.expected_outputs.items()}
        }

class ComponentValidator:
    """
    Validates components against their expected interfaces.
    Ensures components implement required methods and attributes.
    """
    
    def __init__(self):
        self.interface_definitions: Dict[str, ComponentInterfaceDefinition] = {}
    
    def register_interface(self, interface: ComponentInterfaceDefinition) -> None:
        """Register an interface definition for validation."""
        self.interface_definitions[interface.name] = interface
    
    def validate_component(self, component_instance: Any, interface_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a component against a registered interface.
        
        Args:
            component_instance: Instance of the component to validate
            interface_name: Name of the interface to validate against
            
        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        if interface_name not in self.interface_definitions:
            return False, [f"Interface '{interface_name}' not registered"]
        
        interface = self.interface_definitions[interface_name]
        errors = []
        
        # Check required methods
        for method_name in interface.required_methods:
            if not hasattr(component_instance, method_name):
                errors.append(f"Missing required method: {method_name}")
            elif not callable(getattr(component_instance, method_name)):
                errors.append(f"Attribute {method_name} is not callable")
        
        # Check required attributes
        for attr_name in interface.required_attributes:
            if not hasattr(component_instance, attr_name):
                errors.append(f"Missing required attribute: {attr_name}")
        
        # Check method signatures for expected inputs and outputs
        for method_name, expected_type in interface.expected_inputs.items():
            if hasattr(component_instance, method_name) and callable(getattr(component_instance, method_name)):
                method = getattr(component_instance, method_name)
                sig = inspect.signature(method)
                
                # Skip self/cls parameter for instance/class methods
                params = list(sig.parameters.values())[1:] if len(sig.parameters) > 0 else []
                
                if len(params) == 0 and expected_type is not None:
                    errors.append(f"Method {method_name} expects input of type {expected_type} but takes no parameters")
        
        return len(errors) == 0, errors
    
    def discover_components(self, module_path: str) -> Dict[str, Any]:
        """
        Discover components in a module path.
        
        Args:
            module_path: Dot-separated path to module
            
        Returns:
            Dictionary of component name to component class
        """
        components = {}
        try:
            module = importlib.import_module(module_path)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module_path:
                    components[name] = obj
        except ImportError as e:
            print(f"Error importing module {module_path}: {e}")
        
        return components
    
    def generate_interface_from_component(self, component_class: Type) -> ComponentInterfaceDefinition:
        """
        Generate an interface definition from a component class.
        
        Args:
            component_class: Class to generate interface from
            
        Returns:
            ComponentInterfaceDefinition
        """
        name = component_class.__name__
        methods = set()
        attributes = set()
        inputs = {}
        outputs = {}
        
        for attr_name, attr in inspect.getmembers(component_class):
            if attr_name.startswith('_'):
                continue
                
            if inspect.isfunction(attr) or inspect.ismethod(attr):
                methods.add(attr_name)
                sig = inspect.signature(attr)
                
                # Get return type annotation if available
                if sig.return_annotation != inspect.Signature.empty:
                    outputs[attr_name] = sig.return_annotation
            else:
                attributes.add(attr_name)
        
        return ComponentInterfaceDefinition(name, methods, attributes, inputs, outputs)