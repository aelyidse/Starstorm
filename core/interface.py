import abc
from typing import Dict, Any, List, Optional, Type, Callable, Set

class InterfaceMethod:
    """Represents a method in an interface contract with its signature and validation rules."""
    def __init__(self, name: str, return_type: Type, parameters: Dict[str, Type], 
                 description: str = "", required: bool = True):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.description = description
        self.required = required
        
    def describe(self) -> Dict[str, Any]:
        """Return a dictionary describing this method."""
        return {
            'name': self.name,
            'return_type': str(self.return_type),
            'parameters': {name: str(param_type) for name, param_type in self.parameters.items()},
            'description': self.description,
            'required': self.required
        }

class InterfaceProperty:
    """Represents a property in an interface contract with its type and validation rules."""
    def __init__(self, name: str, property_type: Type, description: str = "", 
                 required: bool = True, read_only: bool = False):
        self.name = name
        self.property_type = property_type
        self.description = description
        self.required = required
        self.read_only = read_only
        
    def describe(self) -> Dict[str, Any]:
        """Return a dictionary describing this property."""
        return {
            'name': self.name,
            'type': str(self.property_type),
            'description': self.description,
            'required': self.required,
            'read_only': self.read_only
        }

class ComponentInterface(abc.ABC):
    """
    Abstract base for explicit component interfaces.
    Each interface should define the methods and data contracts required for interaction.
    """
    def __init__(self):
        self._methods: Dict[str, InterfaceMethod] = {}
        self._properties: Dict[str, InterfaceProperty] = {}
        self._events: Set[str] = set()
        
    def register_method(self, method: InterfaceMethod) -> None:
        """Register a method in this interface contract."""
        self._methods[method.name] = method
        
    def register_property(self, prop: InterfaceProperty) -> None:
        """Register a property in this interface contract."""
        self._properties[prop.name] = prop
        
    def register_event(self, event_name: str) -> None:
        """Register an event that this interface can emit."""
        self._events.add(event_name)
        
    def validate_implementation(self, implementation: Any) -> List[str]:
        """
        Validate that an implementation satisfies this interface.
        Returns a list of validation errors, or an empty list if valid.
        """
        errors = []
        
        # Check methods
        for method_name, method_def in self._methods.items():
            if not hasattr(implementation, method_name):
                if method_def.required:
                    errors.append(f"Required method '{method_name}' not implemented")
                continue
                
            impl_method = getattr(implementation, method_name)
            if not callable(impl_method):
                errors.append(f"'{method_name}' is not callable")
                
        # Check properties
        for prop_name, prop_def in self._properties.items():
            if not hasattr(implementation, prop_name):
                if prop_def.required:
                    errors.append(f"Required property '{prop_name}' not implemented")
                    
        return errors
    
    @abc.abstractmethod
    def interface_version(self) -> str:
        pass

    @abc.abstractmethod
    def describe(self) -> Dict[str, Any]:
        """Return a dictionary describing the interface contract."""
        contract = {
            'version': self.interface_version(),
            'methods': {name: method.describe() for name, method in self._methods.items()},
            'properties': {name: prop.describe() for name, prop in self._properties.items()},
            'events': list(self._events)
        }
        return contract
