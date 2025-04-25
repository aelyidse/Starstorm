import inspect
from typing import Dict, Any, Type, Optional, List, Set, Callable, TypeVar, Generic, cast
from .component import Component
from .enhanced_component import EnhancedComponent
from .exceptions import ComponentError
from .system_manager import SystemManager

T = TypeVar('T')

class DependencyProvider(Generic[T]):
    """
    A provider class that encapsulates dependency resolution logic.
    Can provide dependencies through various strategies (singleton, factory, etc.)
    """
    def __init__(self, dependency_type: Type[T], factory: Optional[Callable[[], T]] = None):
        self.dependency_type = dependency_type
        self.factory = factory
        self._instance: Optional[T] = None
        
    def get(self) -> T:
        """Get the dependency instance."""
        if self._instance is None:
            if self.factory:
                self._instance = self.factory()
            else:
                self._instance = self.dependency_type()
        return self._instance
    
    def set_instance(self, instance: T) -> None:
        """Set a specific instance to be provided."""
        if not isinstance(instance, self.dependency_type):
            raise ComponentError(f"Instance must be of type {self.dependency_type.__name__}")
        self._instance = instance


class DependencyInjector:
    """
    Main dependency injection container that manages dependencies and their resolution.
    """
    def __init__(self, system_manager: Optional[SystemManager] = None):
        self._providers: Dict[str, DependencyProvider] = {}
        self._system_manager = system_manager
        self._component_dependencies: Dict[str, Set[str]] = {}
        
    def register_provider(self, name: str, provider: DependencyProvider) -> None:
        """Register a dependency provider."""
        self._providers[name] = provider
        
    def register_type(self, name: str, dependency_type: Type[T], 
                     factory: Optional[Callable[[], T]] = None) -> None:
        """Register a type as a dependency."""
        provider = DependencyProvider(dependency_type, factory)
        self.register_provider(name, provider)
        
    def register_instance(self, name: str, instance: Any) -> None:
        """Register an existing instance as a dependency."""
        provider = DependencyProvider(type(instance))
        provider.set_instance(instance)
        self.register_provider(name, provider)
        
    def get(self, name: str) -> Any:
        """Get a dependency by name."""
        if name not in self._providers:
            if self._system_manager and name in self._system_manager.components:
                # Auto-register components from system manager
                component = self._system_manager.components[name]
                self.register_instance(name, component)
                return component
            raise ComponentError(f"Dependency not registered: {name}")
        return self._providers[name].get()
    
    def inject_dependencies(self, component: Component) -> None:
        """
        Inject dependencies into a component based on its declared dependencies.
        This method resolves and injects all dependencies for a component.
        """
        for dep_name in component.get_dependencies():
            try:
                dependency = self.get(dep_name)
                
                # For enhanced components, use the dependency resolution mechanism
                if isinstance(component, EnhancedComponent):
                    component.resolve_dependency(dep_name)
                    
                    # Try to find a setter method for this dependency
                    setter_name = f"set_{dep_name}"
                    if hasattr(component, setter_name) and callable(getattr(component, setter_name)):
                        setter = getattr(component, setter_name)
                        setter(dependency)
                    else:
                        # Try to set as attribute if no setter exists
                        setattr(component, f"_{dep_name}", dependency)
                
                # Track this dependency relationship
                if component.name not in self._component_dependencies:
                    self._component_dependencies[component.name] = set()
                self._component_dependencies[component.name].add(dep_name)
                
            except ComponentError as e:
                raise ComponentError(f"Failed to inject dependency '{dep_name}' into '{component.name}': {str(e)}")
    
    def inject_by_type(self, obj: Any) -> None:
        """
        Inject dependencies by examining type annotations.
        This allows for constructor or property injection based on types.
        """
        if not hasattr(obj, "__annotations__"):
            return
            
        for attr_name, attr_type in obj.__annotations__.items():
            # Skip if already has a value
            if hasattr(obj, attr_name) and getattr(obj, attr_name) is not None:
                continue
                
            # Find a provider that matches this type
            for provider_name, provider in self._providers.items():
                if issubclass(provider.dependency_type, attr_type):
                    setattr(obj, attr_name, provider.get())
                    break
    
    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """Get the current dependency relationships graph."""
        return {k: v.copy() for k, v in self._component_dependencies.items()}
    
    def clear(self) -> None:
        """Clear all registered dependencies."""
        self._providers.clear()
        self._component_dependencies.clear()