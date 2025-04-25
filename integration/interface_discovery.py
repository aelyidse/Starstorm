from typing import Dict, Any, List, Set, Type, Optional
import inspect
import importlib
import pkgutil
import sys
from core.interface import ComponentInterface

class InterfaceDiscovery:
    """
    Discovers and catalogs interfaces throughout the system.
    Provides mechanisms to find, document, and report on available interfaces.
    """
    def __init__(self):
        self.discovered_interfaces: Dict[str, Type[ComponentInterface]] = {}
        self.interface_implementations: Dict[str, List[str]] = {}
        
    def discover_interfaces(self, package_path: str) -> Dict[str, Type[ComponentInterface]]:
        """
        Recursively discover all ComponentInterface subclasses in the given package.
        
        Args:
            package_path: Dot-notation path to the package to scan
            
        Returns:
            Dictionary mapping interface names to interface classes
        """
        discovered = {}
        package = importlib.import_module(package_path)
        
        # Find all ComponentInterface subclasses in this package
        for name, obj in inspect.getmembers(package):
            if (inspect.isclass(obj) and 
                issubclass(obj, ComponentInterface) and 
                obj != ComponentInterface):
                discovered[name] = obj
                
        # Recursively search subpackages
        if hasattr(package, '__path__'):
            for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
                full_name = f"{package_path}.{name}"
                try:
                    sub_discovered = self.discover_interfaces(full_name)
                    discovered.update(sub_discovered)
                except ImportError:
                    pass
                    
        self.discovered_interfaces.update(discovered)
        return discovered
    
    def find_implementations(self, interface_class: Type[ComponentInterface]) -> List[str]:
        """
        Find all classes that implement a specific interface.
        
        Args:
            interface_class: The interface class to search for implementations
            
        Returns:
            List of implementation class names
        """
        implementations = []
        for module_name, module in sys.modules.items():
            if module and not module_name.startswith('_'):
                for name, obj in inspect.getmembers(module):
                    # Check if it's a class and has the interface's methods
                    if (inspect.isclass(obj) and 
                        hasattr(obj, 'interface') and 
                        isinstance(obj.interface, interface_class)):
                        implementations.append(f"{module_name}.{name}")
        
        interface_name = interface_class.__name__
        self.interface_implementations[interface_name] = implementations
        return implementations
    
    def generate_interface_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of all discovered interfaces and their implementations.
        
        Returns:
            Dictionary containing interface documentation
        """
        report = {
            'interfaces': {},
            'implementation_count': {},
            'orphaned_interfaces': []
        }
        
        for name, interface_class in self.discovered_interfaces.items():
            # Create an instance to access its methods
            try:
                interface_instance = interface_class()
                description = interface_instance.describe()
                
                # Find implementations
                implementations = self.find_implementations(interface_class)
                
                report['interfaces'][name] = {
                    'description': description,
                    'implementations': implementations
                }
                
                report['implementation_count'][name] = len(implementations)
                
                if not implementations:
                    report['orphaned_interfaces'].append(name)
            except Exception as e:
                report['interfaces'][name] = {
                    'error': f"Could not document interface: {str(e)}"
                }
                
        return report