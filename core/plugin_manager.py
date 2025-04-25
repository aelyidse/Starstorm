from typing import Dict, Type, Any, Optional
from pathlib import Path
import importlib
import inspect
from .dependency_injector import DependencyInjector
from .enhanced_component import EnhancedComponent

class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle for SDK extensions.
    """
    def __init__(self, injector: DependencyInjector):
        self._injector = injector
        self._plugins: Dict[str, Type[EnhancedComponent]] = {}
        self._plugin_instances: Dict[str, EnhancedComponent] = {}
        
    def discover_plugins(self, package_name: str) -> None:
        """
        Discover all plugins in a package.
        Plugins are classes that inherit from EnhancedComponent and are decorated with @plugin.
        """
        try:
            package = importlib.import_module(package_name)
            for _, obj in inspect.getmembers(package):
                if (inspect.isclass(obj) and 
                    issubclass(obj, EnhancedComponent) and 
                    hasattr(obj, '_is_plugin')):
                    self._plugins[obj.__name__] = obj
        except ImportError as e:
            raise ImportError(f"Failed to discover plugins in {package_name}: {str(e)}")
            
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> EnhancedComponent:
        """
        Load and initialize a plugin instance.
        """
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")
            
        plugin_class = self._plugins[plugin_name]
        instance = plugin_class(
            name=plugin_name,
            config=config or {}
        )
        
        # Inject dependencies
        self._injector.inject_dependencies(instance)
        
        self._plugin_instances[plugin_name] = instance
        return instance
        
    def get_plugin(self, plugin_name: str) -> EnhancedComponent:
        """Get a loaded plugin instance."""
        if plugin_name not in self._plugin_instances:
            raise ValueError(f"Plugin not loaded: {plugin_name}")
        return self._plugin_instances[plugin_name]
        
    def list_plugins(self) -> Dict[str, Type[EnhancedComponent]]:
        """List all discovered plugins."""
        return self._plugins.copy()
        
    def list_loaded_plugins(self) -> Dict[str, EnhancedComponent]:
        """List all loaded plugin instances."""
        return self._plugin_instances.copy()

def plugin(cls: Type[EnhancedComponent]) -> Type[EnhancedComponent]:
    """
    Decorator to mark a class as a plugin.
    """
    cls._is_plugin = True
    return cls