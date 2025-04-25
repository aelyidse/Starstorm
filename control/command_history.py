from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import time
import json
import os
from datetime import datetime
from .command_execution import TransactionalCommandExecutor
from .command_validation import CommandValidationError

class CommandHistoryManager:
    """
    Manages command history with replay capabilities.
    Records command execution, provides filtering, and supports replaying command sequences.
    """
    def __init__(self, executor: TransactionalCommandExecutor, history_size: int = 1000):
        self.executor = executor
        self.history_size = history_size
        self.command_history: List[Dict[str, Any]] = []
        self.replay_hooks: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        
    def record_command(self, command: Dict[str, Any], result: Dict[str, Any], success: bool) -> None:
        """Record a command execution in history"""
        history_entry = {
            'command': command,
            'result': result,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'success': success
        }
        
        self.command_history.append(history_entry)
        
        # Trim history if it exceeds the maximum size
        if len(self.command_history) > self.history_size:
            self.command_history = self.command_history[-self.history_size:]
    
    def get_history(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get command history with optional filtering
        
        Args:
            filters: Optional filters to apply (e.g., command_type, time_range, success)
            
        Returns:
            Filtered command history
        """
        if not filters:
            return self.command_history
            
        filtered_history = self.command_history
        
        # Filter by command type
        if 'command_type' in filters:
            filtered_history = [
                entry for entry in filtered_history 
                if entry['command'].get('type') == filters['command_type']
            ]
            
        # Filter by time range
        if 'time_start' in filters:
            filtered_history = [
                entry for entry in filtered_history 
                if entry['timestamp'] >= filters['time_start']
            ]
            
        if 'time_end' in filters:
            filtered_history = [
                entry for entry in filtered_history 
                if entry['timestamp'] <= filters['time_end']
            ]
            
        # Filter by success status
        if 'success' in filters:
            filtered_history = [
                entry for entry in filtered_history 
                if entry['success'] == filters['success']
            ]
            
        return filtered_history
    
    def save_history(self, filepath: str) -> bool:
        """
        Save command history to a file
        
        Args:
            filepath: Path to save history file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.command_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving command history: {e}")
            return False
    
    def load_history(self, filepath: str) -> bool:
        """
        Load command history from a file
        
        Args:
            filepath: Path to history file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            return False
            
        try:
            with open(filepath, 'r') as f:
                loaded_history = json.load(f)
                
            # Validate loaded history format
            for entry in loaded_history:
                if not all(k in entry for k in ['command', 'result', 'timestamp', 'success']):
                    return False
                    
            self.command_history = loaded_history
            return True
        except Exception as e:
            print(f"Error loading command history: {e}")
            return False
    
    def register_replay_hook(self, command_type: str, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Register a hook function to be called during command replay"""
        self.replay_hooks[command_type] = hook
    
    def replay_commands(self, commands: List[Dict[str, Any]], 
                       system_state: Dict[str, Any],
                       simulate: bool = False) -> Dict[str, Any]:
        """
        Replay a sequence of commands
        
        Args:
            commands: List of commands to replay
            system_state: Current system state
            simulate: If True, simulate execution without modifying state
            
        Returns:
            Replay results
        """
        results = []
        success_count = 0
        failed_commands = []
        
        # Create a snapshot for simulation mode
        if simulate:
            original_state = system_state.copy()
            
        for i, command in enumerate(commands):
            try:
                # Call pre-execution hook if registered
                command_type = command.get('type')
                if command_type in self.replay_hooks:
                    self.replay_hooks[command_type](command)
                
                # Execute command
                result = self.executor.execute(command, system_state)
                results.append({
                    'command': command,
                    'result': result,
                    'success': True,
                    'index': i
                })
                success_count += 1
            except (CommandValidationError, ValueError, RuntimeError) as e:
                results.append({
                    'command': command,
                    'error': str(e),
                    'success': False,
                    'index': i
                })
                failed_commands.append((i, command, str(e)))
        
        # Restore original state if in simulation mode
        if simulate:
            system_state.clear()
            system_state.update(original_state)
            
        return {
            'total_commands': len(commands),
            'success_count': success_count,
            'failure_count': len(failed_commands),
            'results': results,
            'failed_commands': failed_commands
        }
    
    def create_macro(self, name: str, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a named macro from a sequence of commands
        
        Args:
            name: Name of the macro
            commands: List of commands in the macro
            
        Returns:
            Macro definition
        """
        macro = {
            'name': name,
            'commands': commands,
            'created_at': time.time(),
            'command_count': len(commands)
        }
        return macro
    
    def replay_macro(self, macro: Dict[str, Any], system_state: Dict[str, Any], 
                    simulate: bool = False) -> Dict[str, Any]:
        """
        Replay a macro
        
        Args:
            macro: Macro definition
            system_state: Current system state
            simulate: If True, simulate execution without modifying state
            
        Returns:
            Replay results
        """
        return self.replay_commands(macro['commands'], system_state, simulate)