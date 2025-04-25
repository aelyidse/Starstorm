from typing import Dict, Any, List, Tuple, Optional
from semver import Version
from core.interface import ComponentInterface

class InterfaceCompatibilityChecker:
    """
    Checks compatibility between interface versions to ensure components can work together.
    """
    def __init__(self):
        self.interfaces: Dict[str, Dict[str, ComponentInterface]] = {}
        self.compatibility_results: List[Dict[str, Any]] = []
        
    def register_interface(self, name: str, version: str, interface: ComponentInterface) -> None:
        """Register an interface with a specific name and version."""
        if name not in self.interfaces:
            self.interfaces[name] = {}
        self.interfaces[name][version] = interface
        
    def check_compatibility(self, interface_name: str, required_version: str, 
                           available_version: str) -> Tuple[bool, str]:
        """
        Check if a required interface version is compatible with an available version.
        Returns (is_compatible, reason).
        """
        if interface_name not in self.interfaces:
            return False, f"Interface '{interface_name}' not registered"
            
        if available_version not in self.interfaces[interface_name]:
            return False, f"Version '{available_version}' of interface '{interface_name}' not registered"
            
        interface = self.interfaces[interface_name][available_version]
        return interface.is_compatible_with(required_version)
        
    def validate_component_dependencies(self, component_name: str, 
                                      dependencies: Dict[str, str],
                                      available_interfaces: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Validate that a component's interface dependencies can be satisfied.
        
        Args:
            component_name: Name of the component being validated
            dependencies: Dict mapping interface names to required versions
            available_interfaces: Dict mapping interface names to available versions
            
        Returns:
            List of validation results
        """
        results = []
        
        for interface_name, required_version in dependencies.items():
            if interface_name not in available_interfaces:
                results.append({
                    'component': component_name,
                    'interface': interface_name,
                    'required_version': required_version,
                    'available_version': None,
                    'compatible': False,
                    'reason': f"Interface '{interface_name}' not available"
                })
                continue
                
            available_version = available_interfaces[interface_name]
            compatible, reason = self.check_compatibility(
                interface_name, required_version, available_version
            )
            
            results.append({
                'component': component_name,
                'interface': interface_name,
                'required_version': required_version,
                'available_version': available_version,
                'compatible': compatible,
                'reason': reason
            })
            
        self.compatibility_results.extend(results)
        return results
        
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all compatibility check results."""
        return self.compatibility_results