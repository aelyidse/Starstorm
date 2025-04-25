import asyncio
import logging
from typing import Dict, List, Type, Optional, Set, Callable, Any
from enum import Enum, auto

from .component import Component
from .enhanced_component import EnhancedComponent
from .events import Event, LifecycleEvent, EventPriority
from .exceptions import ComponentError

class ComponentState(Enum):
    """States a component can be in during its lifecycle"""
    UNREGISTERED = auto()
    REGISTERED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()

class LifecycleComponentRegistry:
    """
    Registry for components with enhanced lifecycle management.
    Tracks component states and manages transitions through the component lifecycle.
    """
    
    def __init__(self, system_manager=None):
        self._components: Dict[str, Component] = {}
        self._component_states: Dict[str, ComponentState] = {}
        self._component_types: Dict[str, Type[Component]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._dependents: Dict[str, Set[str]] = {}
        self._system_manager = system_manager
        self._logger = logging.getLogger("core.lifecycle_registry")
        self._lifecycle_hooks: Dict[ComponentState, List[Callable]] = {
            state: [] for state in ComponentState
        }
        
    def register_component(self, component: Component) -> None:
        """Register a component with the registry"""
        if component.name in self._components:
            raise ComponentError(f"Component {component.name} already registered")
            
        self._components[component.name] = component
        self._component_states[component.name] = ComponentState.REGISTERED
        
        # Track dependencies
        dependencies = component.get_dependencies()
        self._dependencies[component.name] = set(dependencies)
        
        # Update dependents tracking
        for dep in dependencies:
            if dep not in self._dependents:
                self._dependents[dep] = set()
            self._dependents[dep].add(component.name)
            
        # Register with system manager if available
        if self._system_manager and component.name not in self._system_manager.components:
            self._system_manager.register_component(component)
            
        # Fire lifecycle event
        self._fire_lifecycle_event(component.name, ComponentState.REGISTERED)
        
    def register_component_type(self, component_class: Type[Component]) -> None:
        """Register a component type for later instantiation"""
        name = component_class.__name__
        self._component_types[name] = component_class
        
    def create_component(self, 
                        type_name: str, 
                        name: str, 
                        dependencies: Optional[List[str]] = None,
                        **kwargs) -> Component:
        """Create and register a component of the specified type"""
        if type_name not in self._component_types:
            raise ComponentError(f"Component type not found: {type_name}")
            
        component_class = self._component_types[type_name]
        
        # Create the component
        if issubclass(component_class, EnhancedComponent):
            component = component_class(name, dependencies, **kwargs)
        else:
            component = component_class(name, dependencies)
            
        # Register the component
        self.register_component(component)
        return component
        
    def get_component(self, name: str) -> Component:
        """Get a registered component by name"""
        if name not in self._components:
            raise ComponentError(f"Component not found: {name}")
        return self._components[name]
        
    def get_component_state(self, name: str) -> ComponentState:
        """Get the current state of a component"""
        if name not in self._component_states:
            raise ComponentError(f"Component not found: {name}")
        return self._component_states[name]
        
    def set_component_state(self, name: str, state: ComponentState) -> None:
        """Set the state of a component"""
        if name not in self._components:
            raise ComponentError(f"Component not found: {name}")
            
        old_state = self._component_states.get(name)
        self._component_states[name] = state
        
        # Fire lifecycle event
        self._fire_lifecycle_event(name, state, old_state)
        
    def register_lifecycle_hook(self, state: ComponentState, hook: Callable) -> None:
        """Register a hook to be called when a component enters a state"""
        self._lifecycle_hooks[state].append(hook)
        
    def _fire_lifecycle_event(self, 
                             component_name: str, 
                             new_state: ComponentState,
                             old_state: Optional[ComponentState] = None) -> None:
        """Fire lifecycle hooks and events for a state change"""
        # Call hooks
        for hook in self._lifecycle_hooks[new_state]:
            try:
                hook(component_name, old_state, new_state)
            except Exception as e:
                self._logger.error(f"Error in lifecycle hook: {str(e)}")
                
        # Publish event if system manager is available
        if self._system_manager:
            lifecycle_type = new_state.name.lower()
            event = LifecycleEvent(
                source="lifecycle_registry",
                payload={
                    "component": component_name,
                    "old_state": old_state.name if old_state else None,
                    "new_state": new_state.name
                },
                lifecycle_type=lifecycle_type,
                priority=EventPriority.HIGH
            )
            asyncio.create_task(self._system_manager.publish_event(event))
            
    async def initialize_component(self, name: str) -> None:
        """Initialize a component and its dependencies"""
        if name not in self._components:
            raise ComponentError(f"Component not found: {name}")
            
        # Skip if already initialized or initializing
        current_state = self._component_states[name]
        if current_state in (ComponentState.INITIALIZING, 
                            ComponentState.INITIALIZED,
                            ComponentState.STARTING,
                            ComponentState.RUNNING):
            return
            
        # Set state to initializing
        self.set_component_state(name, ComponentState.INITIALIZING)
        
        # Initialize dependencies first
        for dep_name in self._dependencies[name]:
            if dep_name not in self._components:
                self.set_component_state(name, ComponentState.FAILED)
                raise ComponentError(f"Dependency {dep_name} not found for {name}")
                
            # Initialize the dependency
            await self.initialize_component(dep_name)
            
        # Now initialize this component
        try:
            component = self._components[name]
            
            # For enhanced components, call pre-start hooks
            if isinstance(component, EnhancedComponent):
                for hook in component._lifecycle_hooks['pre_start']:
                    await component._call_maybe_async(hook)
                    
            # Set as initialized
            self.set_component_state(name, ComponentState.INITIALIZED)
            
        except Exception as e:
            self.set_component_state(name, ComponentState.FAILED)
            self._logger.error(f"Failed to initialize component {name}: {str(e)}")
            raise ComponentError(f"Failed to initialize component {name}: {str(e)}")
            
    async def start_component(self, name: str) -> None:
        """Start a component and its dependencies"""
        if name not in self._components:
            raise ComponentError(f"Component not found: {name}")
            
        # Skip if already running
        if self._component_states[name] == ComponentState.RUNNING:
            return
            
        # Initialize first if needed
        if self._component_states[name] not in (ComponentState.INITIALIZED, ComponentState.STOPPED):
            await self.initialize_component(name)
            
        # Set state to starting
        self.set_component_state(name, ComponentState.STARTING)
        
        # Start dependencies first
        for dep_name in self._dependencies[name]:
            await self.start_component(dep_name)
            
        # Now start this component
        try:
            component = self._components[name]
            await component.start()
            self.set_component_state(name, ComponentState.RUNNING)
            
        except Exception as e:
            self.set_component_state(name, ComponentState.FAILED)
            self._logger.error(f"Failed to start component {name}: {str(e)}")
            raise ComponentError(f"Failed to start component {name}: {str(e)}")
            
    async def stop_component(self, name: str, stop_dependents: bool = True) -> None:
        """Stop a component and optionally its dependents"""
        if name not in self._components:
            raise ComponentError(f"Component not found: {name}")
            
        # Skip if already stopped
        if self._component_states[name] in (ComponentState.STOPPED, ComponentState.UNREGISTERED):
            return
            
        # Stop dependents first if requested
        if stop_dependents and name in self._dependents:
            for dependent in self._dependents[name]:
                await self.stop_component(dependent, stop_dependents=False)
                
        # Set state to stopping
        self.set_component_state(name, ComponentState.STOPPING)
        
        # Stop the component
        try:
            component = self._components[name]
            await component.stop()
            self.set_component_state(name, ComponentState.STOPPED)
            
        except Exception as e:
            self.set_component_state(name, ComponentState.FAILED)
            self._logger.error(f"Failed to stop component {name}: {str(e)}")
            raise ComponentError(f"Failed to stop component {name}: {str(e)}")
            
    async def restart_component(self, name: str) -> None:
        """Restart a component and its dependencies"""
        await self.stop_component(name)
        await self.start_component(name)
        
    def unregister_component(self, name: str) -> None:
        """Unregister a component from the registry"""
        if name not in self._components:
            raise ComponentError(f"Component not found: {name}")
            
        # Can't unregister if dependents exist
        if name in self._dependents and self._dependents[name]:
            raise ComponentError(f"Cannot unregister {name}, it has dependents: {self._dependents[name]}")
            
        # Remove from registry
        component = self._components.pop(name)
        self._component_states.pop(name)
        
        # Update dependency tracking
        if name in self._dependencies:
            for dep in self._dependencies[name]:
                if dep in self._dependents and name in self._dependents[dep]:
                    self._dependents[dep].remove(name)
            del self._dependencies[name]
            
        if name in self._dependents:
            del self._dependents[name]
            
        # Fire lifecycle event
        self._fire_lifecycle_event(name, ComponentState.UNREGISTERED)
        
    def get_all_components(self) -> Dict[str, Component]:
        """Get all registered components"""
        return self._components.copy()
        
    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """Get the dependency graph"""
        return {k: v.copy() for k, v in self._dependencies.items()}
        
    def get_dependent_graph(self) -> Dict[str, Set[str]]:
        """Get the dependent graph"""
        return {k: v.copy() for k, v in self._dependents.items()}
        
    def get_components_by_state(self, state: ComponentState) -> List[str]:
        """Get all components in a specific state"""
        return [name for name, s in self._component_states.items() if s == state]
        
    async def start_all(self) -> None:
        """Start all registered components in dependency order"""
        # Build a list of components with no dependents (leaf nodes)
        start_order = []
        visited = set()
        
        def visit(component_name):
            if component_name in visited:
                return
            visited.add(component_name)
            
            # Visit dependencies first
            for dep in self._dependencies.get(component_name, set()):
                if dep in self._components:
                    visit(dep)
                    
            start_order.append(component_name)
            
        # Visit all components
        for name in self._components:
            visit(name)
            
        # Start components in order
        for name in start_order:
            await self.start_component(name)
            
    async def stop_all(self) -> None:
        """Stop all registered components in reverse dependency order"""
        # Use the reverse of the start order
        stop_order = []
        visited = set()
        
        def visit(component_name):
            if component_name in visited:
                return
            visited.add(component_name)
            
            # Visit dependents first
            for dep in self._dependents.get(component_name, set()):
                if dep in self._components:
                    visit(dep)
                    
            stop_order.append(component_name)
            
        # Visit all components
        for name in self._components:
            visit(name)
            
        # Stop components in order
        for name in stop_order:
            await self.stop_component(name, stop_dependents=False)