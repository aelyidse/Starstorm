from typing import Dict, Any, List, Callable, TypeVar, Generic, Set, Union, Optional, Tuple
import copy
import json
import os
import datetime
from collections import deque

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

def update_field(state: Dict[str, Any], field: str, value: Any) -> Dict[str, Any]:
    """
    Update a field in a dictionary state.
    
    Args:
        state: The current state dictionary
        field: The field to update
        value: The new value
        
    Returns:
        A new state dictionary with the updated field
    """
    new_state = copy.deepcopy(state)
    new_state[field] = value
    return new_state

def update_nested_field(state: Dict[str, Any], path: List[str], value: Any) -> Dict[str, Any]:
    """
    Update a nested field in a dictionary state.
    
    Args:
        state: The current state dictionary
        path: List of keys to traverse
        value: The new value
        
    Returns:
        A new state dictionary with the updated nested field
    """
    if not path:
        return value
    
    new_state = copy.deepcopy(state)
    current = new_state
    
    # Navigate to the parent of the field to update
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Update the field
    current[path[-1]] = value
    return new_state

def append_to_list(state: Dict[str, Any], field: str, item: Any) -> Dict[str, Any]:
    """
    Append an item to a list field in the state.
    
    Args:
        state: The current state dictionary
        field: The list field to update
        item: The item to append
        
    Returns:
        A new state dictionary with the updated list
    """
    new_state = copy.deepcopy(state)
    if field not in new_state or not isinstance(new_state[field], list):
        new_state[field] = []
    new_state[field].append(item)
    return new_state

def remove_from_list(state: Dict[str, Any], field: str, 
                    predicate: Callable[[Any], bool]) -> Dict[str, Any]:
    """
    Remove items from a list field in the state based on a predicate.
    
    Args:
        state: The current state dictionary
        field: The list field to update
        predicate: Function that returns True for items to remove
        
    Returns:
        A new state dictionary with the updated list
    """
    new_state = copy.deepcopy(state)
    if field in new_state and isinstance(new_state[field], list):
        new_state[field] = [item for item in new_state[field] if not predicate(item)]
    return new_state

def merge_dictionaries(state: Dict[str, Any], field: str, 
                      updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge a dictionary into a dictionary field in the state.
    
    Args:
        state: The current state dictionary
        field: The dictionary field to update
        updates: The dictionary to merge
        
    Returns:
        A new state dictionary with the merged dictionary field
    """
    new_state = copy.deepcopy(state)
    if field not in new_state or not isinstance(new_state[field], dict):
        new_state[field] = {}
    new_state[field].update(updates)
    return new_state

class StateHistory(Generic[T]):
    """
    Manages a history of states with a configurable maximum depth.
    Provides methods to add states, retrieve previous states, and roll back to a specific state.
    
    Args:
        max_depth: Maximum number of historical states to keep (default: 10)
        initial_state: Optional initial state to add to history
    """
    def __init__(self, max_depth: int = 10, initial_state: Optional[T] = None):
        self.max_depth = max_depth
        self.history: deque = deque(maxlen=max_depth)
        if initial_state is not None:
            self.add_state(initial_state)
    
    def add_state(self, state: T) -> None:
        """
        Add a new state to the history.
        
        Args:
            state: The state to add
        """
        self.history.append(copy.deepcopy(state))
    
    def get_previous_state(self, steps_back: int = 1) -> Optional[T]:
        """
        Get a previous state from the history.
        
        Args:
            steps_back: Number of steps to go back in history (default: 1)
            
        Returns:
            The previous state, or None if not available
        """
        if steps_back <= 0 or steps_back > len(self.history):
            return None
        return copy.deepcopy(self.history[-steps_back])
    
    def rollback(self, steps_back: int = 1) -> Optional[T]:
        """
        Roll back to a previous state and remove all newer states.
        
        Args:
            steps_back: Number of steps to roll back (default: 1)
            
        Returns:
            The state after rollback, or None if rollback not possible
        """
        if steps_back <= 0 or steps_back >= len(self.history):
            return None
        
        # Remove newer states
        for _ in range(steps_back):
            self.history.pop()
        
        # Return the current state after rollback
        return self.get_current_state()
    
    def get_current_state(self) -> Optional[T]:
        """
        Get the current (most recent) state.
        
        Returns:
            The current state, or None if history is empty
        """
        if not self.history:
            return None
        return copy.deepcopy(self.history[-1])
    
    def clear(self) -> None:
        """
        Clear the state history.
        """
        self.history.clear()
    
    def get_history(self) -> List[T]:
        """
        Get the entire state history as a list.
        
        Returns:
            List of states in chronological order
        """
        return list(self.history)
    
    def __len__(self) -> int:
        """
        Get the number of states in the history.
        
        Returns:
            Number of states
        """
        return len(self.history)

def create_state_history(max_depth: int = 10, initial_state: Optional[Dict[str, Any]] = None) -> StateHistory:
    """
    Create a new StateHistory instance for dictionary states.
    
    Args:
        max_depth: Maximum number of historical states to keep (default: 10)
        initial_state: Optional initial state to add to history
        
    Returns:
        A new StateHistory instance
    """
    return StateHistory(max_depth, initial_state)

# State Diffing Utilities

def diff_states(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two state dictionaries and return the differences.
    
    Args:
        old_state: The previous state dictionary
        new_state: The current state dictionary
        
    Returns:
        A dictionary containing only the fields that changed, with their new values
    """
    if not isinstance(old_state, dict) or not isinstance(new_state, dict):
        return new_state if old_state != new_state else {}
    
    diff = {}
    # Find added or modified fields
    for key, value in new_state.items():
        if key not in old_state:
            diff[key] = value
        elif isinstance(value, dict) and isinstance(old_state[key], dict):
            nested_diff = diff_states(old_state[key], value)
            if nested_diff:
                diff[key] = nested_diff
        elif value != old_state[key]:
            diff[key] = value
    
    # Find removed fields
    for key in old_state:
        if key not in new_state:
            diff[key] = None
    
    return diff

def get_changed_fields(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> List[str]:
    """
    Get a list of field names that changed between two states.
    
    Args:
        old_state: The previous state dictionary
        new_state: The current state dictionary
        
    Returns:
        A list of field names that were added, modified, or removed
    """
    changes = []
    
    # Check for added or modified fields
    for key in new_state:
        if key not in old_state or new_state[key] != old_state[key]:
            changes.append(key)
    
    # Check for removed fields
    for key in old_state:
        if key not in new_state and key not in changes:
            changes.append(key)
    
    return changes

def get_nested_changed_fields(old_state: Dict[str, Any], new_state: Dict[str, Any], 
                             prefix: str = "") -> List[str]:
    """
    Get a list of field paths that changed between two states, including nested fields.
    
    Args:
        old_state: The previous state dictionary
        new_state: The current state dictionary
        prefix: Prefix for nested field paths
        
    Returns:
        A list of field paths that were added, modified, or removed
    """
    changes = []
    
    if not isinstance(old_state, dict) or not isinstance(new_state, dict):
        if old_state != new_state:
            return [prefix] if prefix else []
        return []
    
    # Check for added or modified fields
    for key in new_state:
        current_path = f"{prefix}.{key}" if prefix else key
        
        if key not in old_state:
            changes.append(current_path)
        elif isinstance(new_state[key], dict) and isinstance(old_state[key], dict):
            # Recursively check nested dictionaries
            nested_changes = get_nested_changed_fields(old_state[key], new_state[key], current_path)
            changes.extend(nested_changes)
        elif new_state[key] != old_state[key]:
            changes.append(current_path)
    
    # Check for removed fields
    for key in old_state:
        current_path = f"{prefix}.{key}" if prefix else key
        if key not in new_state:
            changes.append(current_path)
    
    return changes

def create_state_change_event(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a state change event with details about what changed.
    
    Args:
        old_state: The previous state dictionary
        new_state: The current state dictionary
        
    Returns:
        A dictionary representing a state change event
    """
    changes = get_changed_fields(old_state, new_state)
    diff = diff_states(old_state, new_state)
    
    return {
        "type": "state_change",
        "previous_state": old_state,
        "current_state": new_state,
        "changed_fields": changes,
        "diff": diff
    }

def detect_threshold_changes(old_state: Dict[str, Any], new_state: Dict[str, Any], 
                           thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Detect changes that exceed specified thresholds.
    
    Args:
        old_state: The previous state dictionary
        new_state: The current state dictionary
        thresholds: Dictionary mapping field names to threshold values
        
    Returns:
        List of dictionaries describing threshold violations
    """
    violations = []
    
    for field, threshold in thresholds.items():
        # Handle nested fields using dot notation
        path = field.split('.')
        old_value = old_state
        new_value = new_state
        
        # Navigate to the nested field
        valid_path = True
        for key in path:
            if key in old_value and key in new_value:
                old_value = old_value[key]
                new_value = new_value[key]
            else:
                valid_path = False
                break
        
        if not valid_path:
            continue
            
        # Check if numeric and if change exceeds threshold
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            change = abs(new_value - old_value)
            if change > threshold:
                violations.append({
                    "field": field,
                    "old_value": old_value,
                    "new_value": new_value,
                    "change": change,
                    "threshold": threshold
                })
    
    return violations

def track_state_changes(state_history: StateHistory) -> List[Dict[str, Any]]:
    """
    Analyze a state history to track changes across multiple states.
    
    Args:
        state_history: A StateHistory object containing state history
        
    Returns:
        List of state change events
    """
    history = state_history.get_history()
    if len(history) < 2:
        return []
    
    changes = []
    for i in range(1, len(history)):
        old_state = history[i-1]
        new_state = history[i]
        event = create_state_change_event(old_state, new_state)
        changes.append(event)
    
    return changes


def serialize_state(state: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
    """
    Serialize a state dictionary to a JSON-compatible format.
    
    Args:
        state: The state dictionary to serialize
        include_metadata: Whether to include metadata like timestamp
        
    Returns:
        A JSON-serializable dictionary
    """
    serialized = copy.deepcopy(state)
    
    # Add metadata if requested
    if include_metadata:
        serialized['__metadata__'] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'version': '1.0.0'
        }
    
    # Handle non-serializable types
    _process_non_serializable(serialized)
    
    return serialized

def _process_non_serializable(obj: Any) -> None:
    """
    Recursively process a data structure to make it JSON serializable.
    Modifies the object in place.
    
    Args:
        obj: The object to process
    """
    if isinstance(obj, dict):
        # Process each key-value pair in dictionaries
        for key, value in list(obj.items()):
            if isinstance(value, (set, tuple)):
                obj[key] = list(value)
            elif hasattr(value, 'as_dict') and callable(value.as_dict):
                obj[key] = value.as_dict()
            elif hasattr(value, '__dict__'):
                obj[key] = value.__dict__
            else:
                _process_non_serializable(value)
    elif isinstance(obj, list):
        # Process each item in lists
        for i, item in enumerate(obj):
            if isinstance(item, (set, tuple)):
                obj[i] = list(item)
            elif hasattr(item, 'as_dict') and callable(item.as_dict):
                obj[i] = item.as_dict()
            elif hasattr(item, '__dict__'):
                obj[i] = item.__dict__
            else:
                _process_non_serializable(item)

def deserialize_state(serialized_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize a state dictionary from a JSON-compatible format.
    
    Args:
        serialized_state: The serialized state dictionary
        
    Returns:
        A deserialized state dictionary
    """
    state = copy.deepcopy(serialized_state)
    
    # Remove metadata if present
    if '__metadata__' in state:
        del state['__metadata__']
    
    return state

def save_state_to_file(state: Dict[str, Any], filepath: str, pretty: bool = False) -> None:
    """
    Save a state dictionary to a JSON file.
    
    Args:
        state: The state dictionary to save
        filepath: Path to the file
        pretty: Whether to format the JSON with indentation
    """
    serialized = serialize_state(state)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    with open(filepath, 'w') as f:
        if pretty:
            json.dump(serialized, f, indent=2)
        else:
            json.dump(serialized, f)

def load_state_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load a state dictionary from a JSON file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        The loaded state dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(filepath, 'r') as f:
        serialized = json.load(f)
    
    return deserialize_state(serialized)

class StateSerializer:
    """
    Handles serialization and deserialization of state objects with versioning support.
    """
    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.custom_serializers: Dict[type, Callable[[Any], Any]] = {}
        self.custom_deserializers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
    
    def register_serializer(self, type_class: type, serializer: Callable[[Any], Any]) -> None:
        """
        Register a custom serializer for a specific type.
        
        Args:
            type_class: The type to register a serializer for
            serializer: Function that converts an instance to a serializable dict
        """
        self.custom_serializers[type_class] = serializer
    
    def register_deserializer(self, type_name: str, deserializer: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Register a custom deserializer for a specific type.
        
        Args:
            type_name: The type name as stored in serialized data
            deserializer: Function that converts a dict back to the original type
        """
        self.custom_deserializers[type_name] = deserializer
    
    def serialize(self, obj: Any) -> Dict[str, Any]:
        """
        Serialize an object to a JSON-compatible dictionary.
        
        Args:
            obj: The object to serialize
            
        Returns:
            A JSON-serializable dictionary
        """
        if isinstance(obj, dict):
            result = {k: self.serialize(v) for k, v in obj.items()}
            result['__metadata__'] = {
                'timestamp': datetime.datetime.now().isoformat(),
                'version': self.version
            }
            return result
        
        # Check for custom serializers
        for cls, serializer in self.custom_serializers.items():
            if isinstance(obj, cls):
                serialized = serializer(obj)
                if isinstance(serialized, dict):
                    serialized['__type__'] = cls.__name__
                return serialized
        
        # Handle built-in types
        if isinstance(obj, (list, tuple, set)):
            return [self.serialize(item) for item in obj]
        elif hasattr(obj, 'as_dict') and callable(obj.as_dict):
            result = obj.as_dict()
            result['__type__'] = obj.__class__.__name__
            return result
        elif hasattr(obj, '__dict__'):
            result = {k: self.serialize(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            result['__type__'] = obj.__class__.__name__
            return result
        
        # Return primitive types as is
        return obj
    
    def deserialize(self, data: Any) -> Any:
        """
        Deserialize data from a JSON-compatible format.
        
        Args:
            data: The data to deserialize
            
        Returns:
            The deserialized object
        """
        if not isinstance(data, dict):
            if isinstance(data, list):
                return [self.deserialize(item) for item in data]
            return data
        
        # Handle type information
        if '__type__' in data:
            type_name = data['__type__']
            if type_name in self.custom_deserializers:
                # Create a copy without the type info
                data_copy = {k: v for k, v in data.items() if k != '__type__'}
                return self.custom_deserializers[type_name](data_copy)
        
        # Remove metadata if present
        result = {k: self.deserialize(v) for k, v in data.items() if k != '__metadata__'}
        return result
    
    def save_to_file(self, obj: Any, filepath: str, pretty: bool = False) -> None:
        """
        Serialize an object and save it to a JSON file.
        
        Args:
            obj: The object to serialize
            filepath: Path to the file
            pretty: Whether to format the JSON with indentation
        """
        serialized = self.serialize(obj)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(serialized, f, indent=2)
            else:
                json.dump(serialized, f)
    
    def load_from_file(self, filepath: str) -> Any:
        """
        Load and deserialize an object from a JSON file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            The deserialized object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        with open(filepath, 'r') as f:
            serialized = json.load(f)
        
        return self.deserialize(serialized)