from ..core.plugin_manager import plugin
from ..core.enhanced_component import EnhancedComponent

@plugin
class ExamplePlugin(EnhancedComponent):
    """
    Example plugin demonstrating plugin architecture.
    """
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config=config)
        
    def plugin_initialize(self) -> None:
        print(f"Initializing plugin: {self.name}")
        
    def plugin_start(self) -> None:
        print(f"Starting plugin: {self.name}")
        
    def plugin_stop(self) -> None:
        print(f"Stopping plugin: {self.name}")