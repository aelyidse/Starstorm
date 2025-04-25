from typing import Dict, Any, List, Optional, Callable, Tuple
import copy
from .command_validation import CommandValidationError, ContextualCommandValidator

class TransactionalCommandExecutor:
    """
    Executes commands with transactional semantics.
    Ensures commands are executed atomically - either completely succeeds or fails with no side effects.
    """
    def __init__(self, validator: ContextualCommandValidator):
        self.validator = validator
        self.command_handlers: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}
        self.executed_commands: List[Dict[str, Any]] = []
        self.system_snapshots: List[Dict[str, Any]] = []
        self.history_manager = None
    
    def register_command_handler(self, command_type: str, 
                               handler: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]):
        """Register a handler function for a specific command type"""
        self.command_handlers[command_type] = handler
    
    def set_history_manager(self, history_manager) -> None:
        """Set the history manager for recording command execution"""
        self.history_manager = history_manager
    
    def execute(self, command: Dict[str, Any], system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a command with transactional semantics
        
        Args:
            command: Command to execute
            system_state: Current system state
            
        Returns:
            Result of command execution
            
        Raises:
            CommandValidationError: If validation fails
            ValueError: If command handler is not registered
        """
        # Validate command
        validation_result = self.validator.validate(command, system_state)
        if not validation_result['valid']:
            error = CommandValidationError(f"Command validation failed: {validation_result['details']}")
            
            # Record failed command in history
            if self.history_manager:
                self.history_manager.record_command(command, {'error': str(error)}, False)
                
            raise error
        
        # Create system state snapshot for rollback
        state_snapshot = copy.deepcopy(system_state)
        self.system_snapshots.append(state_snapshot)
        
        command_type = command.get('type')
        if command_type not in self.command_handlers:
            self.system_snapshots.pop()  # Remove snapshot on failure
            error = ValueError(f"No handler registered for command type: {command_type}")
            
            # Record failed command in history
            if self.history_manager:
                self.history_manager.record_command(command, {'error': str(error)}, False)
                
            raise error
        
        try:
            # Execute command
            result = self.command_handlers[command_type](command, system_state)
            # Record successful execution
            self.executed_commands.append(command)
            
            # Record in history
            if self.history_manager:
                self.history_manager.record_command(command, result, True)
                
            return result
        except Exception as e:
            # Rollback to snapshot on failure
            self._rollback(system_state)
            
            # Record failed command in history
            if self.history_manager:
                self.history_manager.record_command(command, {'error': str(e)}, False)
                
            raise RuntimeError(f"Command execution failed: {str(e)}")
    
    def execute_batch(self, commands: List[Dict[str, Any]], system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute multiple commands as a single transaction
        
        Args:
            commands: List of commands to execute
            system_state: Current system state
            
        Returns:
            List of execution results
            
        Raises:
            Exception: If any command fails, all changes are rolled back
        """
        # Create system state snapshot for rollback
        state_snapshot = copy.deepcopy(system_state)
        self.system_snapshots.append(state_snapshot)
        
        results = []
        try:
            for cmd in commands:
                result = self.execute(cmd, system_state)
                results.append(result)
            return results
        except Exception as e:
            # Rollback all commands in the batch
            self._rollback(system_state)
            # Remove the commands executed in this batch from history
            self.executed_commands = self.executed_commands[:-len(results)]
            raise RuntimeError(f"Batch execution failed: {str(e)}")
    
    def _rollback(self, system_state: Dict[str, Any]) -> None:
        """Rollback system state to last snapshot"""
        if not self.system_snapshots:
            return
            
        snapshot = self.system_snapshots.pop()
        # Clear current state and restore from snapshot
        system_state.clear()
        system_state.update(snapshot)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get history of successfully executed commands"""
        return self.executed_commands