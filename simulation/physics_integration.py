from typing import Dict, Any, Optional, List
import numpy as np
from simulation.physics_core import DeterministicPhysicsCore
from core.state_container import StateContainer
from core.state_operations import update_field, merge_dictionaries

class PhysicsIntegration:
    """
    Integrates the deterministic physics core with the state management system.
    Provides methods to synchronize physics state with the application state.
    """
    
    def __init__(self, physics_core: Optional[DeterministicPhysicsCore] = None):
        """
        Initialize the physics integration.
        
        Args:
            physics_core: Optional existing physics core instance
        """
        self.physics = physics_core or DeterministicPhysicsCore()
        self.entity_mappings: Dict[str, str] = {}  # Maps state IDs to physics entity IDs
    
    def register_entity(self, state_id: str, physics_id: str, properties: Dict[str, Any]) -> None:
        """
        Register an entity from the state system with the physics system.
        
        Args:
            state_id: ID in the state management system
            physics_id: ID to use in the physics system
            properties: Physical properties for the entity
        """
        self.physics.add_entity(physics_id, properties)
        self.entity_mappings[state_id] = physics_id
    
    def update_from_state(self, state_container: StateContainer) -> None:
        """
        Update physics entities from the application state.
        
        Args:
            state_container: The application state container
        """
        state = state_container.state
        
        for state_id, physics_id in self.entity_mappings.items():
            # Extract entity data from state using path notation (e.g., "vehicles.rover1")
            entity_data = self._get_nested_value(state, state_id.split('.'))
            
            if entity_data and physics_id in self.physics.entities:
                # Update physics entity with state data
                physics_entity = self.physics.entities[physics_id]
                
                # Map state properties to physics properties
                if 'position' in entity_data:
                    physics_entity['position'] = np.array(entity_data['position'], dtype=np.float64)
                if 'velocity' in entity_data:
                    physics_entity['velocity'] = np.array(entity_data['velocity'], dtype=np.float64)
                if 'mass' in entity_data:
                    physics_entity['mass'] = entity_data['mass']
                # Add other properties as needed
    
    def update_state(self, state_container: StateContainer) -> StateContainer:
        """
        Update application state with physics simulation results.
        
        Args:
            state_container: The application state container
            
        Returns:
            Updated state container
        """
        # Start a transaction for atomic state update
        transaction = state_container.begin_transaction()
        
        for state_id, physics_id in self.entity_mappings.items():
            if physics_id in self.physics.entities:
                physics_entity = self.physics.entities[physics_id]
                
                # Prepare state updates
                updates = {}
                if 'position' in physics_entity:
                    updates['position'] = physics_entity['position'].tolist()
                if 'velocity' in physics_entity:
                    updates['velocity'] = physics_entity['velocity'].tolist()
                if 'acceleration' in physics_entity:
                    updates['acceleration'] = physics_entity['acceleration'].tolist()
                
                # Apply updates to the state path
                path_parts = state_id.split('.')
                
                # Define an updater function for the transaction
                def update_entity_state(current_state):
                    return self._update_nested_value(current_state, path_parts, updates)
                
                # Apply the update in the transaction
                transaction.apply(update_entity_state)
        
        # Commit the transaction
        return transaction.commit()
    
    def step_simulation(self, state_container: StateContainer, dt: Optional[float] = None) -> StateContainer:
        """
        Perform one step of physics simulation and update the application state.
        
        Args:
            state_container: The application state container
            dt: Optional timestep duration
            
        Returns:
            Updated state container
        """
        # Update physics from current state
        self.update_from_state(state_container)
        
        # Step the physics simulation
        self.physics.step(dt)
        
        # Update state with new physics data
        return self.update_state(state_container)
    
    def _get_nested_value(self, data: Dict[str, Any], path: List[str]) -> Any:
        """
        Get a value from a nested dictionary using a path.
        
        Args:
            data: Dictionary to extract from
            path: List of keys forming the path
            
        Returns:
            The value at the path, or None if not found
        """
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _update_nested_value(self, data: Dict[str, Any], path: List[str], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary at a specific path.
        
        Args:
            data: Dictionary to update
            path: List of keys forming the path
            updates: Dictionary of updates to apply
            
        Returns:
            Updated dictionary
        """
        result = data.copy()
        current = result
        
        # Navigate to the correct nested dictionary
        for i, key in enumerate(path[:-1]):
            if key not in current:
                current[key] = {}
            if not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Update the final dictionary
        last_key = path[-1]
        if last_key not in current:
            current[last_key] = {}
        if isinstance(current[last_key], dict):
            current[last_key].update(updates)
        else:
            current[last_key] = updates
        
        return result