import asyncio
import logging
from typing import Dict, Type, Any, Optional, List, Set
from .event_bus import EventBus
from .events import Event, LifecycleEvent, EventPriority
from .component import Component
from .exceptions import SystemManagerError
from .dependency_injector import DependencyInjector
from .events import Event, EventPriority, PropagationPhase, EventResult
from .lifecycle_component_registry import LifecycleComponentRegistry, ComponentState

class SystemManager:
    def __init__(self):
        self.event_bus = EventBus()
        self.components: Dict[str, Component] = {}
        self._interfaces: Dict[str, Any] = {}
        self._logger = logging.getLogger("core.system_manager")
        self._running = False
        self._dependency_injector = DependencyInjector(self)
        self._component_hierarchy: Dict[str, str] = {}  # child -> parent mapping
        self._component_registry = LifecycleComponentRegistry(self)

    def register_component(self, component: Component):
        if component.name in self.components:
            raise SystemManagerError(f"Component {component.name} already registered.")
        # Register interface if present
        if component.interface:
            iface_name = type(component.interface).__name__
            self._interfaces[iface_name] = component.interface
        self.components[component.name] = component
        component.system = self  # Set the system reference
        self._logger.info(f"Component registered: {component.name}")
        
        # Register with component registry
        if component.name not in self._component_registry._components:
            self._component_registry.register_component(component)
        
        # Register with event manager
        if component.parent:
            self.set_component_parent(component.name, component.parent)

    def set_component_parent(self, component_name: str, parent_name: str) -> None:
        """Set the parent-child relationship for event propagation"""
        if component_name not in self.components:
            raise SystemManagerError(f"Component {component_name} not registered.")
        if parent_name not in self.components:
            raise SystemManagerError(f"Parent component {parent_name} not registered.")
            
        self._component_hierarchy[component_name] = parent_name
        self.event_bus.event_manager.set_parent(component_name, parent_name)
        
        # Update the component's parent reference
        component = self.components[component_name]
        component.parent = parent_name

    def resolve_dependencies(self):
        for comp_name, comp in self.components.items():
            for dep in comp.get_dependencies():
                if dep not in self.components:
                    raise SystemManagerError(f"Dependency {dep} for component {comp.name} not found.")
                # Inject dependency using the injector
                self._dependency_injector.inject_dependencies(comp)

    def validate_interfaces(self):
        for comp in self.components.values():
            iface = comp.interface
            if iface is not None:
                iface_name = type(iface).__name__
                if iface_name not in self._interfaces:
                    raise SystemManagerError(f"Interface {iface_name} required by {comp.name} not registered.")

    async def start(self):
        self._logger.info("SystemManager starting all components...")
        
        # Start the event bus first
        await self.event_bus.run()
        
        # Resolve dependencies and validate interfaces
        self.resolve_dependencies()
        self.validate_interfaces()
        
        # Start all components using the lifecycle registry
        await self._component_registry.start_all()
        
        self._running = True

    async def stop(self):
        self._logger.info("SystemManager stopping all components...")
        
        # Stop all components using the lifecycle registry
        await self._component_registry.stop_all()
        
        # Stop the event bus last
        self.event_bus.stop()
        self._running = False

    def get_component(self, name: str) -> Component:
        if name not in self.components:
            raise SystemManagerError(f"Component not found: {name}")
        return self.components[name]

    def get_interface(self, interface_name: str) -> Any:
        if interface_name not in self._interfaces:
            raise SystemManagerError(f"Interface not found: {interface_name}")
        return self._interfaces.get(interface_name)

    def is_running(self) -> bool:
        return self._running
        
    def get_dependency_injector(self) -> DependencyInjector:
        """Get the dependency injector instance."""
        return self._dependency_injector
        
    def register_dependency(self, name: str, instance: Any) -> None:
        """Register a dependency directly with the injector."""
        self._dependency_injector.register_instance(name, instance)
        
    async def publish_event(self, event: Event) -> None:
        """Publish an event to the event bus."""
        await self.event_bus.publish(event)
        
    async def broadcast_to_components(self, event: Event) -> None:
        """Broadcast an event to all components directly."""
        for component in self.components.values():
            await component.on_event(event)
