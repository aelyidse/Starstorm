import importlib
import inspect
import os
import pkgutil
from typing import Dict, List, Type, Set, Optional
from .component import Component
from .enhanced_component import EnhancedComponent
from .exceptions import ComponentError

class ComponentRegistry:
    """
    Registry for component discovery and auto-registration.
    Supports scanning packages for component types.
    """
    
    def __init__(self):
        self._component_types: Dict[str, Type[Component]] = {}
        self._scanned_packages: Set[str] = set()
        
    def register_component_type(self, component_class: Type[Component]) -> None:
        """Register a component type."""
        name = component_class.__name__
        self._component_types[name] = component_class
        
    def get_component_type(self, name: str) -> Type[Component]:
        """Get a registered component type by name."""
        if name not in self._component_types:
            raise ComponentError(f"Component type not found: {name}")
        return self._component_types[name]
        
    def get_all_component_types(self) -> Dict[str, Type[Component]]:
        """Get all registered component types."""
        return self._component_types.copy()
        
    def scan_package(self, package_name: str) -> None:
        """
        Scan a package for component classes.
        Automatically registers any subclass of Component found.
        """
        if package_name in self._scanned_packages:
            return
            
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            raise ComponentError(f"Could not import package: {package_name}")
            
        self._scanned_packages.add(package_name)
        
        # Get the package path
        if not hasattr(package, '__path__'):
            return
            
        # Scan all modules in the package
        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            full_module_name = f"{package_name}.{module_name}"
            
            try:
                module = importlib.import_module(full_module_name)
                
                # Find all Component subclasses in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Component) and 
                        obj != Component and 
                        obj != EnhancedComponent):
                        self.register_component_type(obj)
                        
                # Recursively scan subpackages
                if is_pkg:
                    self.scan_package(full_module_name)
                    
            except ImportError:
                # Skip modules that can't be imported
                continue
                
    def get_enhanced_component_types(self) -> Dict[str, Type[EnhancedComponent]]:
        """Get all registered enhanced component types."""
        return {
            name: comp_type for name, comp_type in self._component_types.items()
            if issubclass(comp_type, EnhancedComponent)
        }