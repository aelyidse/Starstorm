import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

T = TypeVar('T')

def inject(dependency_name: Optional[str] = None):
    """
    Decorator for injecting dependencies into method parameters.
    
    Usage:
        @inject()
        def method(self, dependency: DependencyType):
            # dependency will be automatically injected
            
        @inject('specific_dependency')
        def method(self, param: Any):
            # param will be injected with the named dependency
    """
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Skip if this is not a component with _injected_dependencies
            if not hasattr(self, '_injected_dependencies'):
                return func(self, *args, **kwargs)
                
            # Get the parameter that needs injection
            params = list(sig.parameters.values())[1:]  # Skip 'self'
            
            if len(args) >= len(params):
                # All parameters are already provided
                return func(self, *args, **kwargs)
                
            # Find parameters that need injection
            args_list = list(args)
            for i, param in enumerate(params[len(args):]):
                param_name = param.name
                
                # Skip if already in kwargs
                if param_name in kwargs:
                    continue
                    
                # Determine which dependency to inject
                dep_name = dependency_name
                if dep_name is None:
                    # Use parameter name as dependency name
                    dep_name = param_name
                    
                # Get the dependency if available
                if dep_name in self._injected_dependencies:
                    args_list.append(self._injected_dependencies[dep_name])
                    
            return func(self, *args_list, **kwargs)
            
        return wrapper
        
    return decorator


def component_method(func: Callable) -> Callable:
    """
    Decorator for component methods that automatically injects dependencies.
    
    Usage:
        @component_method
        async def process_data(self, data_processor: DataProcessor):
            # data_processor will be automatically injected
    """
    sig = inspect.signature(func)
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Skip if this is not a component with _injected_dependencies
        if not hasattr(self, '_injected_dependencies'):
            return func(self, *args, **kwargs)
            
        # Get the parameter that needs injection
        params = list(sig.parameters.values())[1:]  # Skip 'self'
        
        # Find parameters that need injection
        for param in params:
            param_name = param.name
            
            # Skip if already provided
            if param_name in kwargs or len(args) > 0:
                args = args[1:] if len(args) > 0 else args
                continue
                
            # Try to find a matching dependency by name
            if param_name in self._injected_dependencies:
                kwargs[param_name] = self._injected_dependencies[param_name]
                
        return func(self, *args, **kwargs)
        
    return wrapper