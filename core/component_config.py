import json
import os
from typing import Dict, Any, Optional
from .exceptions import ComponentError

class ComponentConfigManager:
    """
    Manages configuration for components and SDK.
    Supports loading from files, environment variables, and direct setting.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._sdk_config: Dict[str, Any] = {}
        
    def set_sdk_config(self, config: Dict[str, Any]) -> None:
        """Set SDK-wide configuration."""
        self._sdk_config.update(config)
        
    def get_sdk_config(self) -> Dict[str, Any]:
        """Get the current SDK configuration."""
        return self._sdk_config.copy()
        
    def update_sdk_config(self, updates: Dict[str, Any]) -> None:
        """Update SDK configuration with new values."""
        self._sdk_config.update(updates)
        
    def load_sdk_config_file(self, filename: str = "sdk_config.json") -> None:
        """Load SDK configuration from a JSON file."""
        if not self.config_dir:
            raise ComponentError("Config directory not set")
            
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise ComponentError(f"SDK config file not found: {filepath}")
            
        try:
            with open(filepath, 'r') as f:
                self._sdk_config.update(json.load(f))
        except json.JSONDecodeError:
            raise ComponentError(f"Invalid JSON in SDK config file: {filepath}")
        
    def load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        if not self.config_dir:
            raise ComponentError("Config directory not set")
            
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise ComponentError(f"Config file not found: {filepath}")
            
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ComponentError(f"Invalid JSON in config file: {filepath}")
            
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component."""
        if component_name not in self._configs:
            # Try to load from file
            try:
                if self.config_dir:
                    filename = f"{component_name}.json"
                    self._configs[component_name] = self.load_config_file(filename)
                else:
                    self._configs[component_name] = {}
            except ComponentError:
                # No config file found, use empty config
                self._configs[component_name] = {}
                
        return self._configs[component_name].copy()
        
    def set_component_config(self, component_name: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific component."""
        self._configs[component_name] = config
        
    def update_component_config(self, component_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a specific component."""
        if component_name not in self._configs:
            self._configs[component_name] = {}
            
        self._configs[component_name].update(updates)
        
    def load_from_env(self, prefix: str = "COMPONENT_") -> None:
        """
        Load configuration from environment variables.
        Format: PREFIX_COMPONENT_KEY=value
        Example: COMPONENT_LOGGER_LEVEL=DEBUG
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                parts = key[len(prefix):].split('_', 1)
                if len(parts) == 2:
                    component_name, config_key = parts
                    component_name = component_name.lower()
                    
                    if component_name not in self._configs:
                        self._configs[component_name] = {}
                        
                    # Try to parse as JSON, fall back to string
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        parsed_value = value
                        
                    self._configs[component_name][config_key] = parsed_value