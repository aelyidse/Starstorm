from typing import Dict, Any, List, Optional, Set, Tuple, Type, Callable
import inspect
from .component_validator import ComponentInterfaceDefinition

class InterfaceCompatibilityChecker:
    """
    Checks compatibility between component interfaces.
    Validates that components can interact correctly based on their interfaces.
    """
    
    def __init__(self):
        self.interfaces: Dict[str, ComponentInterfaceDefinition] = {}
    
    def register_interface(self, interface: ComponentInterfaceDefinition) -> None:
        """Register an interface for compatibility checking."""
        self.interfaces[interface.name] = interface
    
    def check_compatibility(self, provider_name: str, consumer_name: str) -> Tuple[bool, List[str]]:
        """
        Check if a provider interface is compatible with a consumer interface.
        
        Args:
            provider_name: Name of the provider interface
            consumer_name: Name of the consumer interface
            
        Returns:
            Tuple of (is_compatible, list_of_compatibility_issues)
        """
        if provider_name not in self.interfaces:
            return False, [f"Provider interface '{provider_name}' not registered"]
        
        if consumer_name not in self.interfaces:
            return False, [f"Consumer interface '{consumer_name}' not registered"]
        
        provider = self.interfaces[provider_name]
        consumer = self.interfaces[consumer_name]
        issues = []
        
        # Check if provider implements all methods required by consumer
        for method in consumer.required_methods:
            if method not in provider.required_methods:
                issues.append(f"Provider '{provider_name}' missing method '{method}' required by consumer '{consumer_name}'")
        
        # Check if provider's outputs match consumer's expected inputs
        for method, expected_type in consumer.expected_inputs.items():
            if method in provider.expected_outputs:
                provider_type = provider.expected_outputs[method]
                if not self._is_type_compatible(provider_type, expected_type):
                    issues.append(f"Type mismatch for method '{method}': provider returns {provider_type}, consumer expects {expected_type}")
        
        return len(issues) == 0, issues
    
    def _is_type_compatible(self, provider_type: Type, consumer_type: Type) -> bool:
        """Check if provider type is compatible with consumer type."""
        # Simple case: exact match
        if provider_type == consumer_type:
            return True
        
        # Check for subclass relationship
        try:
            return issubclass(provider_type, consumer_type)
        except TypeError:
            # Handle non-class types (e.g., Union, Optional)
            return False
    
    def generate_adapter(self, provider_name: str, consumer_name: str) -> Optional[Dict[str, Callable]]:
        """
        Generate adapter functions to make incompatible interfaces work together.
        
        Args:
            provider_name: Name of the provider interface
            consumer_name: Name of the consumer interface
            
        Returns:
            Dictionary of adapter functions or None if adaptation is not possible
        """
        is_compatible, issues = self.check_compatibility(provider_name, consumer_name)
        if is_compatible:
            return {}  # No adapters needed
        
        if provider_name not in self.interfaces or consumer_name not in self.interfaces:
            return None
        
        provider = self.interfaces[provider_name]
        consumer = self.interfaces[consumer_name]
        adapters = {}
        
        # For each method required by consumer but missing or incompatible in provider
        for method in consumer.required_methods:
            if method not in provider.required_methods:
                # Cannot generate adapter for missing method
                continue
            
            if method in consumer.expected_inputs and method in provider.expected_outputs:
                provider_type = provider.expected_outputs[method]
                consumer_type = consumer.expected_inputs[method]
                
                if not self._is_type_compatible(provider_type, consumer_type):
                    # Generate adapter function for type conversion
                    adapters[method] = self._create_adapter_function(method, provider_type, consumer_type)
        
        return adapters if adapters else None
    
    def _create_adapter_function(self, method_name: str, from_type: Type, to_type: Type) -> Callable:
        """Create an adapter function to convert between types."""
        def adapter(provider_instance, *args, **kwargs):
            result = getattr(provider_instance, method_name)(*args, **kwargs)
            # This is a simplified conversion - in practice, you'd need more sophisticated logic
            return to_type(result)
        
        return adapter