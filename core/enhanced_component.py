import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Set, Type, Callable, Union, TypeVar
from .component import Component
from .events import Event, ComponentEvent, EventFilter, EventPriority
from .interface import ComponentInterface
from .exceptions import ComponentError

T = TypeVar('T')

class EnhancedComponent(Component):
    """
    Enhanced component base class that extends the core Component with additional features:
    - Versioning
    - Configuration management
    - Enhanced dependency injection
    - Lifecycle hooks
    - Event filtering
    - Metrics collection
    """
    
    def __init__(self, 
                 name: str, 
                 dependencies: Optional[List[str]] = None,
                 version: str = "1.0.0",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, dependencies)
        self.version = version
        self._config = config or {}
        self._default_config: Dict[str, Any] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._lifecycle_hooks: Dict[str, List[Callable]] = {
            'pre_start': [],
            'post_start': [],
            'pre_stop': [],
            'post_stop': [],
        }
        self._metrics: Dict[str, Any] = {
            'events_processed': 0,
            'last_event_time': None,
            'errors': 0,
        }
        self._logger = logging.getLogger(f"component.{name}")
        self._dependencies_resolved: Set[str] = set()
        self._injected_dependencies: Dict[str, Any] = {}
        self._event_subscriptions: List[str] = []
        self._event_filters: Dict[str, EventFilter] = {}
        
    def register_event_handler(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        
    def register_lifecycle_hook(self, hook_type: str, hook: Callable[[], None]) -> None:
        """Register a lifecycle hook."""
        if hook_type not in self._lifecycle_hooks:
            raise ComponentError(f"Unknown lifecycle hook type: {hook_type}")
        self._lifecycle_hooks[hook_type].append(hook)
        
    async def on_event(self, event: Event) -> None:
        """Enhanced event handling with type-based dispatch."""
        self._metrics['events_processed'] += 1
        self._metrics['last_event_time'] = event.timestamp
        
        # Add this component to the event's propagation path
        event.add_to_path(self.name)
        
        try:
            # Call specific handlers for this event type
            event_type = event.event_type()
            if event_type in self._event_handlers:
                for handler in self._event_handlers[event_type]:
                    await self._call_maybe_async(handler, event)
                    
            # Call the component-specific implementation
            await self._handle_event(event)
        except Exception as e:
            self._metrics['errors'] += 1
            self._logger.error(f"Error handling event {event.id}: {str(e)}")
            raise
            
    async def _handle_event(self, event: Event) -> None:
        """
        Component-specific event handling implementation.
        This replaces the direct override of on_event.
        """
        pass
        
    async def start(self) -> None:
        """Enhanced start with lifecycle hooks."""
        # Run pre-start hooks
        for hook in self._lifecycle_hooks['pre_start']:
            await self._call_maybe_async(hook)
            
        # Validate configuration
        self._validate_config()
        
        # Check dependencies are resolved
        self._check_dependencies_resolved()
        
        # Subscribe to events
        await self._subscribe_to_events()
        
        # Call parent implementation
        await super().start()
        
        # Component-specific startup
        await self._start_component()
        
        # Run post-start hooks
        for hook in self._lifecycle_hooks['post_start']:
            await self._call_maybe_async(hook)
            
        self.set_health('running')
        
    async def _start_component(self) -> None:
        """
        Component-specific startup implementation.
        This replaces the direct override of start.
        """
        pass
        
    async def stop(self) -> None:
        """Enhanced stop with lifecycle hooks."""
        # Run pre-stop hooks
        for hook in self._lifecycle_hooks['pre_stop']:
            await self._call_maybe_async(hook)
            
        # Unsubscribe from events
        self._unsubscribe_from_events()
        
        # Call parent implementation
        await super().stop()
        
        # Component-specific shutdown
        await self._stop_component()
        
        # Run post-stop hooks
        for hook in self._lifecycle_hooks['post_stop']:
            await self._call_maybe_async(hook)
            
    async def _stop_component(self) -> None:
        """
        Component-specific shutdown implementation.
        This replaces the direct override of stop.
        """
        pass
        
    def set_default_config(self, config: Dict[str, Any]) -> None:
        """Set default configuration values."""
        self._default_config = config
        
        # Apply defaults for any missing config
        for key, value in self._default_config.items():
            if key not in self._config:
                self._config[key] = value
                
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        return self._config.get(key, default)
        
    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
        
    def _validate_config(self) -> None:
        """Validate the configuration against defaults and requirements."""
        # This is a simple implementation that could be extended
        # with schema validation or type checking
        for key, value in self._default_config.items():
            if key not in self._config:
                self._config[key] = value
                
    def resolve_dependency(self, dependency_name: str) -> None:
        """Mark a dependency as resolved."""
        if dependency_name in self.dependencies:
            self._dependencies_resolved.add(dependency_name)
            
    def _check_dependencies_resolved(self) -> None:
        """Check if all dependencies are resolved."""
        unresolved = set(self.dependencies) - self._dependencies_resolved
        if unresolved:
            raise ComponentError(f"Unresolved dependencies: {unresolved}")
            
    async def _call_maybe_async(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function that may be async or sync."""
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics."""
        return self._metrics.copy()
        
    def describe(self) -> Dict[str, Any]:
        """Enhanced component description."""
        base_desc = super().describe()
        return {
            **base_desc,
            'version': self.version,
            'metrics': self.get_metrics(),
            'config': self._config,
        }
        
    def get_dependency(self, name: str) -> Any:
        """Get an injected dependency by name."""
        if name not in self._injected_dependencies:
            raise ComponentError(f"Dependency not injected: {name}")
        return self._injected_dependencies[name]
    
    def set_dependency(self, name: str, dependency: Any) -> None:
        """Set a dependency manually."""
        self._injected_dependencies[name] = dependency
        self.resolve_dependency(name)
        
    # Event subscription methods
    
    def create_event_filter(self, filter_name: str) -> EventFilter:
        """Create and register a named event filter."""
        filter = EventFilter()
        self._event_filters[filter_name] = filter
        return filter
    
    def get_event_filter(self, filter_name: str) -> EventFilter:
        """Get a registered event filter."""
        if filter_name not in self._event_filters:
            raise ComponentError(f"Event filter not found: {filter_name}")
        return self._event_filters[filter_name]
    
    async def _subscribe_to_events(self) -> None:
        """Subscribe to events based on registered handlers."""
        if not hasattr(self.system, 'event_bus'):
            self._logger.warning("No event bus available for subscriptions")
            return
            
        event_bus = self.system.event_bus
        
        # Create a subscription for each event type with handlers
        for event_type in self._event_handlers:
            # Create a filter for this event type
            filter = EventFilter().of_types(event_type)
            
            # Subscribe to the event bus
            subscription_id = event_bus.subscribe(
                self.name,
                self.on_event,
                filter
            )
            
            self._event_subscriptions.append(subscription_id)
            self._logger.debug(f"Subscribed to events of type {event_type}")
    
    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        if not hasattr(self.system, 'event_bus'):
            return
            
        event_bus = self.system.event_bus
        
        # Unsubscribe from all subscriptions
        for subscription_id in self._event_subscriptions:
            event_bus.unsubscribe(subscription_id)
            
        # Or unsubscribe all at once
        # event_bus.unsubscribe_all(self.name)
        
        self._event_subscriptions.clear()
        self._logger.debug("Unsubscribed from all events")
    
    async def publish_event(self, event: Event) -> None:
        """Publish an event to the event bus."""
        if not hasattr(self.system, 'event_bus'):
            self._logger.warning("No event bus available for publishing events")
            return
            
        # Set the source if not already set
        if not event.source:
            event.source = self.name
            
        await self.system.event_bus.publish(event)
    
    async def create_and_publish_event(self, 
                                      event_type: str, 
                                      payload: Dict[str, Any],
                                      priority: EventPriority = EventPriority.NORMAL) -> Event:
        """Create and publish an event in one step."""
        event = ComponentEvent(
            source=self.name,
            payload=payload,
            priority=priority
        )
        
        await self.publish_event(event)
        return event