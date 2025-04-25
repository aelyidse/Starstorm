import asyncio
import logging
from typing import Dict, List, Set, Type, Optional, Callable, Any, TypeVar
from .events import Event, EventPriority, PropagationPhase, EventResult, EventSubscription
from .exceptions import ComponentError

T = TypeVar('T', bound=Event)

class EventManager:
    """
    Central manager for the event system.
    Handles event registration, subscription, and propagation.
    """
    def __init__(self):
        self._subscriptions: Dict[str, List[EventSubscription]] = {}
        self._global_subscriptions: List[EventSubscription] = []
        self._logger = logging.getLogger("core.event_manager")
        self._component_hierarchy: Dict[str, str] = {}  # child -> parent mapping
        
    def set_parent(self, component_name: str, parent_name: str) -> None:
        """Set the parent of a component for event bubbling"""
        self._component_hierarchy[component_name] = parent_name
        
    def get_parent(self, component_name: str) -> Optional[str]:
        """Get the parent of a component"""
        return self._component_hierarchy.get(component_name)
        
    def get_component_path(self, component_name: str) -> List[str]:
        """Get the path from root to this component"""
        path = [component_name]
        current = component_name
        
        while current in self._component_hierarchy:
            parent = self._component_hierarchy[current]
            path.insert(0, parent)
            current = parent
            
            # Prevent infinite loops in case of circular references
            if len(path) > 100 or parent in path:
                break
                
        return path
        
    def subscribe(self, 
                  component_name: str,
                  handler: Callable[[Event], EventResult],
                  event_type: Optional[Type[Event]] = None,
                  priority: EventPriority = EventPriority.NORMAL,
                  source_filter: Optional[str] = None,
                  target_filter: Optional[str] = None,
                  phases: Optional[Set[PropagationPhase]] = None) -> None:
        """
        Subscribe to events.
        
        Args:
            component_name: Name of the component subscribing
            handler: Function to call when event occurs
            event_type: Type of event to subscribe to (None for all events)
            priority: Priority of this handler
            source_filter: Only handle events from this source
            target_filter: Only handle events for this target
            phases: Which propagation phases to handle
        """
        subscription = EventSubscription(
            handler=handler,
            component_name=component_name,
            priority=priority,
            event_type=event_type,
            source_filter=source_filter,
            target_filter=target_filter,
            phases=phases or {PropagationPhase.CAPTURING, PropagationPhase.TARGET, PropagationPhase.BUBBLING}
        )
        
        if event_type:
            event_type_name = event_type.__name__
            if event_type_name not in self._subscriptions:
                self._subscriptions[event_type_name] = []
            self._subscriptions[event_type_name].append(subscription)
            self._subscriptions[event_type_name].sort(key=lambda s: s.priority.value, reverse=True)
        else:
            # Global subscription (all events)
            self._global_subscriptions.append(subscription)
            self._global_subscriptions.sort(key=lambda s: s.priority.value, reverse=True)
            
    def unsubscribe(self, component_name: str, event_type: Optional[Type[Event]] = None) -> None:
        """Unsubscribe from events"""
        if event_type:
            event_type_name = event_type.__name__
            if event_type_name in self._subscriptions:
                self._subscriptions[event_type_name] = [
                    s for s in self._subscriptions[event_type_name] 
                    if s.component_name != component_name
                ]
        else:
            # Unsubscribe from all events
            for event_type_name in self._subscriptions:
                self._subscriptions[event_type_name] = [
                    s for s in self._subscriptions[event_type_name] 
                    if s.component_name != component_name
                ]
            self._global_subscriptions = [
                s for s in self._global_subscriptions 
                if s.component_name != component_name
            ]
            
    async def dispatch(self, event: Event) -> Event:
        """
        Dispatch an event through the system.
        
        This handles the full event propagation cycle:
        1. Capturing phase (top-down)
        2. Target phase (at the target component)
        3. Bubbling phase (bottom-up)
        
        Returns the (potentially modified) event after propagation.
        """
        if event.target:
            # Get the path from root to target for propagation
            path = self.get_component_path(event.target)
            
            # Capturing phase (top-down)
            event.current_phase = PropagationPhase.CAPTURING
            for component in path[:-1]:  # All except the target
                await self._dispatch_to_component(event, component)
                if event.is_propagation_stopped():
                    break
                    
            # Target phase
            if not event.is_propagation_stopped():
                event.current_phase = PropagationPhase.TARGET
                if path:  # Make sure we have a target
                    await self._dispatch_to_component(event, path[-1])
                    
            # Bubbling phase (bottom-up)
            if not event.is_propagation_stopped():
                event.current_phase = PropagationPhase.BUBBLING
                for component in reversed(path[:-1]):  # All except the target, in reverse
                    await self._dispatch_to_component(event, component)
                    if event.is_propagation_stopped():
                        break
        else:
            # No target, broadcast to all components
            await self._dispatch_to_all(event)
            
        return event
        
    async def _dispatch_to_component(self, event: Event, component_name: str) -> None:
        """Dispatch an event to a specific component"""
        # Get relevant subscriptions
        subscriptions = []
        
        # Add type-specific subscriptions
        event_type_name = event.event_type()
        if event_type_name in self._subscriptions:
            subscriptions.extend([
                s for s in self._subscriptions[event_type_name]
                if s.component_name == component_name and s.matches(event)
            ])
            
        # Add global subscriptions for this component
        subscriptions.extend([
            s for s in self._global_subscriptions
            if s.component_name == component_name and s.matches(event)
        ])
        
        # Sort by priority
        subscriptions.sort(key=lambda s: s.priority.value, reverse=True)
        
        # Call handlers
        for subscription in subscriptions:
            try:
                result = subscription.handler(event)
                
                # Mark as processed
                event.mark_processed_by(component_name)
                
                # Handle result
                if result == EventResult.CANCELLED:
                    event.cancel()
                    break
                elif result == EventResult.MODIFIED:
                    # Continue but with modified event
                    pass
                # Otherwise continue to next handler
                
                if event.is_propagation_stopped():
                    break
                    
            except Exception as e:
                self._logger.error(f"Error in event handler for {component_name}: {str(e)}")
                
    async def _dispatch_to_all(self, event: Event) -> None:
        """Broadcast an event to all components"""
        # Get all unique component names from subscriptions
        component_names = set()
        
        for subscriptions in self._subscriptions.values():
            for subscription in subscriptions:
                component_names.add(subscription.component_name)
                
        for subscription in self._global_subscriptions:
            component_names.add(subscription.component_name)
            
        # Dispatch to each component
        for component_name in component_names:
            if not event.is_propagation_stopped():
                await self._dispatch_to_component(event, component_name)
                
    def create_event(self, event_type: Type[T], **kwargs) -> T:
        """Helper to create an event of the specified type"""
        return event_type(**kwargs)