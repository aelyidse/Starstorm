from typing import Dict, Any, List, Tuple, Optional, Callable
import copy
import math
import numpy as np
import time
import hashlib
import json

class DeterministicPhysicsCore:
    """
    A deterministic physics simulation core that ensures reproducible results
    across different runs with the same initial conditions and inputs.
    
    Features:
    - Fixed timestep physics updates
    - Deterministic calculations
    - State serialization and deserialization
    - Reproducible random number generation
    - Collision detection and resolution
    - Support for various force models
    """
    
    def __init__(self, timestep: float = 0.01, gravity: float = 9.81):
        """
        Initialize the physics core.
        
        Args:
            timestep: Fixed physics timestep in seconds
            gravity: Gravity acceleration in m/s²
        """
        self.timestep = timestep
        self.gravity = gravity
        self.time = 0.0
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.forces: List[Callable] = []
        self.collision_handlers: Dict[Tuple[str, str], Callable] = {}
        self.seed = int(time.time())
        self.rng = np.random.RandomState(self.seed)
        
        # Add default forces
        self.add_force(self._apply_gravity)
    
    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for deterministic randomness.
        
        Args:
            seed: Integer seed value
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def add_entity(self, entity_id: str, properties: Dict[str, Any]) -> None:
        """
        Add an entity to the physics simulation.
        
        Args:
            entity_id: Unique identifier for the entity
            properties: Physical properties of the entity
        """
        required_props = ['position', 'velocity', 'mass']
        for prop in required_props:
            if prop not in properties:
                raise ValueError(f"Entity must have '{prop}' property")
        
        # Ensure position and velocity are numpy arrays for consistent math
        if not isinstance(properties['position'], np.ndarray):
            properties['position'] = np.array(properties['position'], dtype=np.float64)
        if not isinstance(properties['velocity'], np.ndarray):
            properties['velocity'] = np.array(properties['velocity'], dtype=np.float64)
        
        # Add default properties if not present
        if 'acceleration' not in properties:
            properties['acceleration'] = np.zeros_like(properties['position'])
        if 'forces' not in properties:
            properties['forces'] = np.zeros_like(properties['position'])
        if 'fixed' not in properties:
            properties['fixed'] = False
        
        self.entities[entity_id] = copy.deepcopy(properties)
    
    def remove_entity(self, entity_id: str) -> None:
        """
        Remove an entity from the simulation.
        
        Args:
            entity_id: ID of the entity to remove
        """
        if entity_id in self.entities:
            del self.entities[entity_id]
    
    def add_force(self, force_func: Callable) -> None:
        """
        Add a force function to the simulation.
        
        Args:
            force_func: Function that applies forces to entities
        """
        self.forces.append(force_func)
    
    def add_collision_handler(self, type1: str, type2: str, handler: Callable) -> None:
        """
        Add a collision handler for specific entity types.
        
        Args:
            type1: First entity type
            type2: Second entity type
            handler: Collision resolution function
        """
        self.collision_handlers[(type1, type2)] = handler
        # Also add the reverse order
        self.collision_handlers[(type2, type1)] = handler
    
    def _apply_gravity(self, entity: Dict[str, Any]) -> np.ndarray:
        """
        Apply gravitational force to an entity.
        
        Args:
            entity: Entity to apply gravity to
            
        Returns:
            Force vector from gravity
        """
        if entity.get('fixed', False):
            return np.zeros_like(entity['position'])
        
        # Apply gravity in the negative y direction (assuming y is up)
        gravity_force = np.zeros_like(entity['position'])
        gravity_force[1] = -self.gravity * entity['mass']
        return gravity_force
    
    def _detect_collisions(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Detect collisions between entities.
        
        Returns:
            List of collision data (entity1_id, entity2_id, collision_info)
        """
        collisions = []
        entity_ids = list(self.entities.keys())
        
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                id1, id2 = entity_ids[i], entity_ids[j]
                entity1, entity2 = self.entities[id1], self.entities[id2]
                
                # Simple sphere collision detection
                if 'radius' in entity1 and 'radius' in entity2:
                    distance = np.linalg.norm(entity1['position'] - entity2['position'])
                    if distance < (entity1['radius'] + entity2['radius']):
                        collision_info = {
                            'distance': distance,
                            'normal': (entity2['position'] - entity1['position']) / distance,
                            'overlap': entity1['radius'] + entity2['radius'] - distance
                        }
                        collisions.append((id1, id2, collision_info))
        
        return collisions
    
    def _resolve_collisions(self, collisions: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        """
        Resolve detected collisions.
        
        Args:
            collisions: List of collision data
        """
        for id1, id2, collision_info in collisions:
            entity1, entity2 = self.entities[id1], self.entities[id2]
            
            # Check if we have a specific handler for these entity types
            handler = None
            if 'type' in entity1 and 'type' in entity2:
                handler = self.collision_handlers.get((entity1['type'], entity2['type']))
            
            if handler:
                # Use custom handler
                handler(id1, id2, entity1, entity2, collision_info, self)
            else:
                # Default elastic collision
                if not entity1.get('fixed', False) and not entity2.get('fixed', False):
                    # Calculate impulse
                    relative_velocity = entity2['velocity'] - entity1['velocity']
                    normal = collision_info['normal']
                    
                    # Project relative velocity onto collision normal
                    velocity_along_normal = np.dot(relative_velocity, normal)
                    
                    # Do not resolve if objects are moving apart
                    if velocity_along_normal > 0:
                        continue
                    
                    # Calculate restitution (bounciness)
                    restitution = min(
                        entity1.get('restitution', 0.8),
                        entity2.get('restitution', 0.8)
                    )
                    
                    # Calculate impulse scalar
                    j = -(1 + restitution) * velocity_along_normal
                    j /= (1 / entity1['mass']) + (1 / entity2['mass'])
                    
                    # Apply impulse
                    impulse = j * normal
                    entity1['velocity'] -= impulse / entity1['mass']
                    entity2['velocity'] += impulse / entity2['mass']
                
                # Position correction to prevent sinking
                correction = collision_info['overlap'] * 0.5 * collision_info['normal']
                if not entity1.get('fixed', False):
                    entity1['position'] -= correction
                if not entity2.get('fixed', False):
                    entity2['position'] += correction
    
    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance the simulation by one timestep.
        
        Args:
            dt: Optional custom timestep (defaults to self.timestep)
        """
        if dt is None:
            dt = self.timestep
        
        # Reset forces
        for entity_id, entity in self.entities.items():
            entity['forces'] = np.zeros_like(entity['position'])
        
        # Apply forces
        for entity_id, entity in self.entities.items():
            if entity.get('fixed', False):
                continue
                
            for force_func in self.forces:
                force = force_func(entity)
                entity['forces'] += force
        
        # Update velocities and positions
        for entity_id, entity in self.entities.items():
            if entity.get('fixed', False):
                continue
                
            # Update acceleration
            entity['acceleration'] = entity['forces'] / entity['mass']
            
            # Update velocity (semi-implicit Euler integration)
            entity['velocity'] += entity['acceleration'] * dt
            
            # Update position
            entity['position'] += entity['velocity'] * dt
        
        # Handle collisions
        collisions = self._detect_collisions()
        self._resolve_collisions(collisions)
        
        # Update simulation time
        self.time += dt
    
    def run(self, duration: float) -> None:
        """
        Run the simulation for a specified duration.
        
        Args:
            duration: Duration to run in seconds
        """
        steps = int(duration / self.timestep)
        for _ in range(steps):
            self.step()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the physics simulation.
        
        Returns:
            Dictionary containing the simulation state
        """
        # Convert numpy arrays to lists for serialization
        serializable_entities = {}
        for entity_id, entity in self.entities.items():
            serializable_entity = {}
            for key, value in entity.items():
                if isinstance(value, np.ndarray):
                    serializable_entity[key] = value.tolist()
                else:
                    serializable_entity[key] = value
            serializable_entities[entity_id] = serializable_entity
        
        return {
            'time': self.time,
            'timestep': self.timestep,
            'gravity': self.gravity,
            'seed': self.seed,
            'entities': serializable_entities
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the simulation state from a state dictionary.
        
        Args:
            state: State dictionary to load
        """
        self.time = state['time']
        self.timestep = state['timestep']
        self.gravity = state['gravity']
        self.seed = state['seed']
        self.rng = np.random.RandomState(self.seed)
        
        # Convert lists back to numpy arrays
        self.entities = {}
        for entity_id, entity in state['entities'].items():
            processed_entity = {}
            for key, value in entity.items():
                if key in ['position', 'velocity', 'acceleration', 'forces'] and isinstance(value, list):
                    processed_entity[key] = np.array(value, dtype=np.float64)
                else:
                    processed_entity[key] = value
            self.entities[entity_id] = processed_entity
    
    def serialize(self) -> str:
        """
        Serialize the simulation state to a JSON string.
        
        Returns:
            JSON string representation of the state
        """
        state = self.get_state()
        return json.dumps(state)
    
    def deserialize(self, serialized_state: str) -> None:
        """
        Deserialize a JSON string to set the simulation state.
        
        Args:
            serialized_state: JSON string representation of the state
        """
        state = json.loads(serialized_state)
        self.set_state(state)
    
    def compute_state_hash(self) -> str:
        """
        Compute a deterministic hash of the current simulation state.
        Useful for verifying determinism across runs.
        
        Returns:
            Hash string of the current state
        """
        state_json = self.serialize()
        return hashlib.sha256(state_json.encode()).hexdigest()