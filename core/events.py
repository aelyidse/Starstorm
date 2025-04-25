import asyncio
import uuid
import time
from typing import Any, Dict, List, Optional, Set, Type, Callable, Union, TypeVar
from enum import Enum, auto
from dataclasses import dataclass, field

class EventPriority(Enum):
    """Priority levels for event processing."""
    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()
    BACKGROUND = auto()

class EventType(Enum):
    """Standard event types for system-wide use."""
    SYSTEM = "system"
    COMPONENT = "component"
    COMMAND = "command"
    STATE_CHANGE = "state_change"
    NOTIFICATION = "notification"
    ERROR = "error"
    TELEMETRY = "telemetry"
    LIFECYCLE = "lifecycle"
    CUSTOM = "custom"

@dataclass
class Event:
    """
    Base event class for the event propagation system.
    All events in the system should inherit from this class.
    """
    source: str
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = field(default=EventPriority.NORMAL)
    propagation_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "base"
    
    def add_to_path(self, component_name: str) -> None:
        """Add a component to the propagation path."""
        self.propagation_path.append(component_name)
    
    def has_visited(self, component_name: str) -> bool:
        """Check if the event has already visited a component."""
        return component_name in self.propagation_path
    
    def with_metadata(self, key: str, value: Any) -> 'Event':
        """Add metadata to the event and return self for chaining."""
        self.metadata[key] = value
        return self
    
    def clone(self) -> 'Event':
        """Create a copy of this event."""
        return Event(
            source=self.source,
            payload=self.payload.copy(),
            id=self.id,
            timestamp=self.timestamp,
            priority=self.priority,
            propagation_path=self.propagation_path.copy(),
            metadata=self.metadata.copy()
        )

@dataclass
class ComponentEvent(Event):
    """Event specific to component communication."""
    
    def event_type(self) -> str:
        return "component"

@dataclass
class CommandEvent(Event):
    """Event representing a command to be executed."""
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def event_type(self) -> str:
        return "command"

@dataclass
class StateChangeEvent(Event):
    """Event representing a state change in a component."""
    previous_state: Dict[str, Any]
    current_state: Dict[str, Any]
    changed_fields: List[str] = field(default_factory=list)
    
    def event_type(self) -> str:
        return "state_change"

@dataclass
class LifecycleEvent(Event):
    """Event representing a component lifecycle change."""
    lifecycle_type: str  # "starting", "started", "stopping", "stopped", etc.
    
    def event_type(self) -> str:
        return "lifecycle"

@dataclass
class ErrorEvent(Event):
    """Event representing an error in the system."""
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    
    def event_type(self) -> str:
        return "error"

class EventFilter:
    """
    Filter for events based on various criteria.
    Used by the event bus to determine which events to deliver to subscribers.
    """
    def __init__(self):
        self.source_filter: Optional[Set[str]] = None
        self.type_filter: Optional[Set[str]] = None
        self.priority_filter: Optional[Set[EventPriority]] = None
        self.custom_filter: Optional[Callable[[Event], bool]] = None
    
    def from_sources(self, *sources: str) -> 'EventFilter':
        """Filter events from specific sources."""
        self.source_filter = set(sources)
        return self
    
    def of_types(self, *types: str) -> 'EventFilter':
        """Filter events of specific types."""
        self.type_filter = set(types)
        return self
    
    def with_priorities(self, *priorities: EventPriority) -> 'EventFilter':
        """Filter events with specific priorities."""
        self.priority_filter = set(priorities)
        return self
    
    def with_custom_filter(self, filter_func: Callable[[Event], bool]) -> 'EventFilter':
        """Add a custom filter function."""
        self.custom_filter = filter_func
        return self
    
    def matches(self, event: Event) -> bool:
        """Check if an event matches this filter."""
        if self.source_filter and event.source not in self.source_filter:
            return False
        
        if self.type_filter and event.event_type() not in self.type_filter:
            return False
        
        if self.priority_filter and event.priority not in self.priority_filter:
            return False
        
        if self.custom_filter and not self.custom_filter(event):
            return False
        
        return True
