from typing import Dict, Type, Any, Optional, List
from .component import Component
from .enhanced_component import EnhancedComponent
from .system_manager import SystemManager
from .exceptions import ComponentError

class ComponentFactory:
    """
    Factory for creating and registering components.
    Supports both standard and enhanced components.
    """
    
    def __init__(self, system_manager: SystemManager):
        self.system_manager = system_manager
        self._component_types: Dict[str, Type[Component]] = {}
        
    def register_component_type(self, component_type: Type[Component], type_name: Optional[str] = None) -> None:
        """Register a component type for later instantiation."""
        name = type_name or component_type.__name__
        self._component_types[name] = component_type
        
    def create_component(self, 
                         type_name: str, 
                         name: str, 
                         dependencies: Optional[List[str]] = None,
                         **kwargs) -> Component:
        """Create a component of the specified type."""
        if type_name not in self._component_types:
            raise ComponentError(f"Unknown component type: {type_name}")
            
        component_class = self._component_types[type_name]
        
        # Create the component
        if issubclass(component_class, EnhancedComponent):
            # Enhanced component with additional parameters
            component = component_class(name, dependencies, **kwargs)
        else:
            # Standard component
            component = component_class(name, dependencies)
            
        # Register with system manager
        self.system_manager.register_component(component)
        
        # Inject dependencies if available
        if dependencies:
            injector = self.system_manager.get_dependency_injector()
            injector.inject_dependencies(component)
            
        return component
        
    def create_and_configure(self, 
                             type_name: str, 
                             name: str, 
                             config: Dict[str, Any],
                             dependencies: Optional[List[str]] = None) -> Component:
        """Create and configure a component in one step."""
        component = self.create_component(type_name, name, dependencies)
        
        # Apply configuration if component supports it
        if isinstance(component, EnhancedComponent):
            for key, value in config.items():
                component.set_config(key, value)
                
        return component