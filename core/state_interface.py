from typing import Dict, Any, List, Callable, Optional, TypeVar, Generic, Tuple
import abc
from core.interface import ComponentInterface, InterfaceMethod, InterfaceProperty

T = TypeVar('T')

class StateContainerInterface(ComponentInterface, Generic[T]):
    """
    Interface for state containers that provide immutable state with transactional updates.
    """
    def __init__(self):
        super().__init__()
        
        # Register methods
        self.register_method(InterfaceMethod(
            name="begin_transaction",
            return_type="Transaction",
            parameters={},
            description="Begin a new transaction for updating the state"
        ))
        
        self.register_method(InterfaceMethod(
            name="add_validator",
            return_type=None,
            parameters={"validator": Callable[[T], List[str]]},
            description="Add a validator function that will be called before any state update"
        ))
        
        self.register_method(InterfaceMethod(
            name="subscribe",
            return_type=str,
            parameters={"callback": Callable[[T, T], None]},
            description="Subscribe to state changes"
        ))
        
        self.register_method(InterfaceMethod(
            name="unsubscribe",
            return_type=None,
            parameters={"subscription_id": str},
            description="Unsubscribe from state changes"
        ))
        
        self.register_method(InterfaceMethod(
            name="get_history",
            return_type=List[Tuple[T, str]],
            parameters={},
            description="Get the history of state changes"
        ))
        
        # Register properties
        self.register_property(InterfaceProperty(
            name="state",
            property_type=T,
            description="The current state (immutable)",
            read_only=True
        ))
        
    def interface_version(self) -> str:
        return "1.0.0"
    
    def describe(self) -> Dict[str, Any]:
        contract = super().describe()
        contract.update({
            "description": "Interface for state containers that provide immutable state with transactional updates"
        })
        return contract