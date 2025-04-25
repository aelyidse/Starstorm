from typing import Dict, Any, Callable, List, Optional, TypeVar, Generic, Set, Tuple
import copy
import uuid
from enum import Enum

T = TypeVar('T')

class TransactionStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

class StateTransactionError(Exception):
    """Exception raised when a state transaction fails."""
    pass

class Transaction:
    """
    Represents a transaction that can be applied to a state container.
    Transactions are atomic - either all changes are applied or none are.
    """
    def __init__(self, state_container: 'StateContainer'):
        self.id = str(uuid.uuid4())
        self.state_container = state_container
        self.changes: List[Tuple[Callable[[Any], Any], List, Dict]] = []
        self.status = TransactionStatus.PENDING
        self._working_state = None
    
    def apply(self, updater_func: Callable[[T], T], *args, **kwargs) -> 'Transaction':
        """
        Register a state update function to be applied when the transaction is committed.
        
        Args:
            updater_func: Function that takes the current state and returns a new state
            *args, **kwargs: Arguments to pass to the updater function
            
        Returns:
            Self for method chaining
        """
        self.changes.append((updater_func, args, kwargs))
        return self
    
    def _get_working_state(self) -> Any:
        """Get a working copy of the state for validation."""
        if self._working_state is None:
            self._working_state = copy.deepcopy(self.state_container.state)
        return self._working_state
    
    def validate(self) -> List[str]:
        """
        Validate all changes in the transaction without applying them.
        
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        working_state = self._get_working_state()
        
        for updater_func, args, kwargs in self.changes:
            try:
                # Apply the update to our working copy
                working_state = updater_func(working_state, *args, **kwargs)
                
                # Run the state container's validators
                for validator in self.state_container.validators:
                    validator_errors = validator(working_state)
                    if validator_errors:
                        errors.extend(validator_errors)
            except Exception as e:
                errors.append(f"Update function error: {str(e)}")
                
        return errors
    
    def commit(self) -> Any:
        """
        Apply all changes in the transaction to the state container.
        
        Returns:
            The new state after all changes are applied
            
        Raises:
            StateTransactionError: If the transaction cannot be committed
        """
        if self.status != TransactionStatus.PENDING:
            raise StateTransactionError(f"Cannot commit transaction with status {self.status}")
        
        # Validate the transaction
        errors = self.validate()
        if errors:
            self.status = TransactionStatus.FAILED
            raise StateTransactionError(f"Transaction validation failed: {errors}")
        
        # Apply all changes
        new_state = self._get_working_state()
        
        # Update the state container
        try:
            self.state_container._update_state(new_state, self.id)
            self.status = TransactionStatus.COMMITTED
            return new_state
        except Exception as e:
            self.status = TransactionStatus.FAILED
            raise StateTransactionError(f"Failed to commit transaction: {str(e)}")
    
    def rollback(self) -> None:
        """
        Discard all changes in the transaction.
        """
        self.status = TransactionStatus.ROLLED_BACK
        self.changes = []
        self._working_state = None


class StateContainer(Generic[T]):
    """
    An immutable state container that supports transactional updates.
    All state updates create a new state object rather than modifying the existing one.
    """
    def __init__(self, initial_state: T):
        self._state = initial_state
        self._history: List[Tuple[T, str]] = [(initial_state, "initial")]
        self.validators: List[Callable[[T], List[str]]] = []
        self._subscribers: Dict[str, Callable[[T, T], None]] = {}
    
    @property
    def state(self) -> T:
        """Get the current state (immutable)."""
        return copy.deepcopy(self._state)
    
    def begin_transaction(self) -> Transaction:
        """
        Begin a new transaction for updating the state.
        
        Returns:
            A new Transaction object
        """
        return Transaction(self)
    
    def _update_state(self, new_state: T, transaction_id: str) -> None:
        """
        Internal method to update the state.
        
        Args:
            new_state: The new state to set
            transaction_id: ID of the transaction that created this state
        """
        old_state = self._state
        self._state = new_state
        self._history.append((new_state, transaction_id))
        
        # Notify subscribers
        for callback in self._subscribers.values():
            callback(old_state, new_state)
    
    def add_validator(self, validator: Callable[[T], List[str]]) -> None:
        """
        Add a validator function that will be called before any state update.
        
        Args:
            validator: Function that takes a state and returns a list of validation errors
        """
        self.validators.append(validator)
    
    def subscribe(self, callback: Callable[[T, T], None]) -> str:
        """
        Subscribe to state changes.
        
        Args:
            callback: Function that takes (old_state, new_state) and is called on every state change
            
        Returns:
            Subscription ID that can be used to unsubscribe
        """
        subscription_id = str(uuid.uuid4())
        self._subscribers[subscription_id] = callback
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> None:
        """
        Unsubscribe from state changes.
        
        Args:
            subscription_id: The subscription ID returned from subscribe()
        """
        if subscription_id in self._subscribers:
            del self._subscribers[subscription_id]
    
    def get_history(self) -> List[Tuple[T, str]]:
        """
        Get the history of state changes.
        
        Returns:
            List of (state, transaction_id) tuples
        """
        return copy.deepcopy(self._history)