from typing import Dict, Any, List, Optional
import asyncio
from manufacturing.digital_twin import ComponentDigitalTwin, ComponentState

class DigitalTwinFactory:
    """
    Factory for creating and managing component digital twins.
    Provides centralized management of multiple digital twins.
    """
    def __init__(self):
        self.twins: Dict[str, ComponentDigitalTwin] = {}
        
    def create_twin(self, 
                   name: str,
                   component_type: str,
                   dependencies: Optional[List[str]] = None,
                   initial_state: ComponentState = ComponentState.DESIGN,
                   properties: Optional[Dict[str, Any]] = None) -> ComponentDigitalTwin:
        """Create a new digital twin"""
        twin = ComponentDigitalTwin(
            name=name,
            component_type=component_type,
            dependencies=dependencies,
            initial_state=initial_state,
            properties=properties
        )
        self.twins[name] = twin
        return twin
        
    def get_twin(self, name: str) -> Optional[ComponentDigitalTwin]:
        """Get a digital twin by name"""
        return self.twins.get(name)
        
    def list_twins(self) -> List[str]:
        """List all digital twins"""
        return list(self.twins.keys())
        
    def get_twins_by_type(self, component_type: str) -> List[ComponentDigitalTwin]:
        """Get all digital twins of a specific type"""
        return [twin for twin in self.twins.values() if twin.component_type == component_type]
        
    def get_twins_by_state(self, state: ComponentState) -> List[ComponentDigitalTwin]:
        """Get all digital twins in a specific state"""
        return [twin for twin in self.twins.values() if twin.state == state]
        
    async def start_all(self) -> None:
        """Start all digital twins"""
        start_tasks = [twin.start() for twin in self.twins.values()]
        await asyncio.gather(*start_tasks)
        
    async def stop_all(self) -> None:
        """Stop all digital twins"""
        stop_tasks = [twin.stop() for twin in self.twins.values()]
        await asyncio.gather(*stop_tasks)
        
    def delete_twin(self, name: str) -> bool:
        """Delete a digital twin"""
        if name in self.twins:
            del self.twins[name]
            return True
        return False
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all digital twins"""
        return {
            'total_twins': len(self.twins),
            'twins_by_state': {state.value: len(self.get_twins_by_state(state)) 
                              for state in ComponentState},
            'twins_by_type': {component_type: len(twins) 
                             for component_type, twins in 
                             self._group_by_type().items()}
        }
        
    def _group_by_type(self) -> Dict[str, List[ComponentDigitalTwin]]:
        """Group twins by component type"""
        result = {}
        for twin in self.twins.values():
            if twin.component_type not in result:
                result[twin.component_type] = []
            result[twin.component_type].append(twin)
        return result