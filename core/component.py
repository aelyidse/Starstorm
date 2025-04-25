import abc
from typing import Any, Dict, Optional, List, Type
from .events import Event
from .interface import ComponentInterface

class Component(abc.ABC):
    """
    Abstract base class for all system components.
    Enforces interface contracts, dependency declaration, lifecycle hooks, health monitoring, and self-description.
    """
    def __init__(self, name: str, dependencies: Optional[List[str]] = None):
        self.name = name
        self.active = False
        self.state: Dict[str, Any] = {}
        self.dependencies = dependencies or []
        self._interface: Optional[ComponentInterface] = None
        self._health: str = 'unknown'

    @property
    def interface(self) -> Optional[ComponentInterface]:
        return self._interface

    @interface.setter
    def interface(self, iface: ComponentInterface):
        self._interface = iface

    @abc.abstractmethod
    async def on_event(self, event: Event):
        """Handle incoming events. Must be implemented by all components."""
        pass

    @abc.abstractmethod
    async def start(self):
        """Start the component and perform initialization."""
        self.active = True
        self._health = 'starting'

    @abc.abstractmethod
    async def stop(self):
        """Stop the component and perform teardown."""
        self.active = False
        self._health = 'stopped'

    def get_state(self) -> Dict[str, Any]:
        return self.state.copy()

    def get_dependencies(self) -> List[str]:
        return list(self.dependencies)

    def set_health(self, status: str):
        self._health = status

    def get_health(self) -> str:
        return self._health

    def describe(self) -> Dict[str, Any]:
        """Return a self-description of the component, including interface and dependencies."""
        return {
            'name': self.name,
            'active': self.active,
            'health': self._health,
            'dependencies': self.dependencies,
            'interface': self._interface.describe() if self._interface else None,
        }
