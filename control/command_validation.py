from typing import Dict, Any, List, Optional, Callable
import re
from enum import Enum

class CommandValidationError(Exception):
    """Exception raised for command validation errors"""
    pass

class ValidationLevel(Enum):
    """Validation levels for contextual command validation"""
    SYNTAX = 1      # Basic syntax validation
    SEMANTIC = 2    # Semantic validation (types, required params)
    CONTEXTUAL = 3  # Full contextual validation (system state)

class CommandInterpreter:
    """
    Validates and interprets commands against a schema and system state.
    Provides semantic validation beyond syntax checking.
    """
    def __init__(self, command_schema: Dict[str, Any]):
        self.command_schema = command_schema
        self.validators: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], bool]] = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validators for common commands"""
        # Example: Validate 'set' commands
        self.validators['set'] = self._validate_set_command
    
    def _validate_set_command(self, command: Dict[str, Any], system_state: Dict[str, Any]) -> bool:
        """Validate a 'set' command against system state"""
        target = command.get('target')
        if not target:
            return False
        
        # Check if target exists in system state
        if '.' in target:
            # Handle nested targets (e.g., 'subsystem.parameter')
            parts = target.split('.')
            current = system_state
            for part in parts[:-1]:
                if part not in current:
                    return False
                current = current[part]
            return parts[-1] in current
        else:
            return target in system_state
    
    def register_validator(self, command_type: str, validator: Callable[[Dict[str, Any], Dict[str, Any]], bool]):
        """Register a custom validator for a command type"""
        self.validators[command_type] = validator
    
    def interpret_and_validate(self, command: Dict[str, Any], system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret and validate a command against the system state
        
        Args:
            command: Command object to validate
            system_state: Current system state
            
        Returns:
            Validated and possibly transformed command
            
        Raises:
            CommandValidationError: If validation fails
        """
        command_type = command.get('type')
        if not command_type:
            raise CommandValidationError("Missing command type")
            
        # Check if command type is supported
        if command_type not in self.command_schema:
            raise CommandValidationError(f"Unsupported command type: {command_type}")
            
        # Validate required parameters
        schema = self.command_schema[command_type]
        for param in schema.get('required_params', []):
            if param not in command:
                raise CommandValidationError(f"Missing required parameter: {param}")
                
        # Validate parameter types
        for param, value in command.items():
            if param == 'type':
                continue
                
            if param in schema.get('param_types', {}):
                expected_type = schema['param_types'][param]
                if not self._validate_type(value, expected_type):
                    raise CommandValidationError(
                        f"Invalid type for parameter '{param}'. Expected {expected_type}, got {type(value).__name__}"
                    )
                    
        # Apply custom validator if available
        if command_type in self.validators:
            if not self.validators[command_type](command, system_state):
                raise CommandValidationError(f"Command validation failed for {command_type}")
                
        # Apply transformations if needed
        transformed = command.copy()
        if 'transform' in schema:
            for param, transform_func in schema['transform'].items():
                if param in transformed:
                    transformed[param] = transform_func(transformed[param])
                    
        return transformed
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate that a value matches the expected type"""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'array':
            return isinstance(value, list)
        elif expected_type == 'object':
            return isinstance(value, dict)
        elif expected_type.startswith('regex:'):
            pattern = expected_type[6:]
            return isinstance(value, str) and bool(re.match(pattern, value))
        return False


class ContextualCommandValidator:
    """
    Validates commands with contextual awareness.
    Considers system state, operational mode, and command dependencies.
    """
    def __init__(self, command_schema: Dict[str, Any]):
        self.command_schema = command_schema
        self.validators: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], bool]] = {}
        self.context_validators: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], bool]] = {}
        self.dependency_map: Dict[str, List[str]] = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validators for common commands"""
        # Basic validators
        self.validators['set'] = self._validate_set_command
        
        # Contextual validators
        self.context_validators['set'] = self._context_validate_set_command
    
    def _validate_set_command(self, command: Dict[str, Any], system_state: Dict[str, Any]) -> bool:
        """Validate a 'set' command against system state"""
        target = command.get('target')
        if not target:
            return False
        
        # Check if target exists in system state
        if '.' in target:
            # Handle nested targets (e.g., 'subsystem.parameter')
            parts = target.split('.')
            current = system_state
            for part in parts[:-1]:
                if part not in current:
                    return False
                current = current[part]
            return parts[-1] in current
        else:
            return target in system_state
    
    def _context_validate_set_command(self, command: Dict[str, Any], system_state: Dict[str, Any]) -> bool:
        """Contextual validation for set command based on system mode and constraints"""
        target = command.get('target')
        value = command.get('value')
        
        # Check if system is in a mode that allows this parameter to be changed
        if 'mode' in system_state and target in system_state.get('locked_parameters', []):
            current_mode = system_state['mode']
            if current_mode in ['CRITICAL', 'EMERGENCY']:
                return False
        
        # Check value against allowed ranges if defined
        if target in system_state.get('parameter_constraints', {}):
            constraints = system_state['parameter_constraints'][target]
            if 'min' in constraints and value < constraints['min']:
                return False
            if 'max' in constraints and value > constraints['max']:
                return False
        
        return True
    
    def register_validator(self, command_type: str, validator: Callable[[Dict[str, Any], Dict[str, Any]], bool]):
        """Register a custom validator for a command type"""
        self.validators[command_type] = validator
    
    def register_context_validator(self, command_type: str, validator: Callable[[Dict[str, Any], Dict[str, Any]], bool]):
        """Register a contextual validator for a command type"""
        self.context_validators[command_type] = validator
    
    def register_command_dependency(self, command_type: str, dependencies: List[str]):
        """Register dependencies for a command type"""
        self.dependency_map[command_type] = dependencies
    
    def validate(self, command: Dict[str, Any], system_state: Dict[str, Any], 
                level: ValidationLevel = ValidationLevel.CONTEXTUAL) -> Dict[str, Any]:
        """
        Validate a command against the system state with specified validation level
        
        Args:
            command: Command object to validate
            system_state: Current system state
            level: Validation level to apply
            
        Returns:
            Validation result with status and details
            
        Raises:
            CommandValidationError: If validation fails
        """
        result = {'valid': True, 'details': []}
        
        # Basic validation
        command_type = command.get('type')
        if not command_type:
            result['valid'] = False
            result['details'].append("Missing command type")
            return result
            
        # Check if command type is supported
        if command_type not in self.command_schema:
            result['valid'] = False
            result['details'].append(f"Unsupported command type: {command_type}")
            return result
        
        # Syntax and semantic validation
        if level.value >= ValidationLevel.SEMANTIC.value:
            # Validate required parameters
            schema = self.command_schema[command_type]
            for param in schema.get('required_params', []):
                if param not in command:
                    result['valid'] = False
                    result['details'].append(f"Missing required parameter: {param}")
            
            # Validate parameter types
            for param, value in command.items():
                if param == 'type':
                    continue
                    
                if param in schema.get('param_types', {}):
                    expected_type = schema['param_types'][param]
                    if not self._validate_type(value, expected_type):
                        result['valid'] = False
                        result['details'].append(
                            f"Invalid type for parameter '{param}'. Expected {expected_type}, got {type(value).__name__}"
                        )
        
        # Contextual validation
        if level.value >= ValidationLevel.CONTEXTUAL.value and result['valid']:
            # Check command dependencies
            if command_type in self.dependency_map:
                for dep in self.dependency_map[command_type]:
                    if not system_state.get('executed_commands', {}).get(dep, False):
                        result['valid'] = False
                        result['details'].append(f"Dependency not satisfied: {dep} must be executed before {command_type}")
            
            # Apply basic validator if available
            if command_type in self.validators:
                if not self.validators[command_type](command, system_state):
                    result['valid'] = False
                    result['details'].append(f"Basic validation failed for {command_type}")
            
            # Apply contextual validator if available
            if command_type in self.context_validators:
                if not self.context_validators[command_type](command, system_state):
                    result['valid'] = False
                    result['details'].append(f"Contextual validation failed for {command_type}")
        
        return result
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate that a value matches the expected type"""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'array':
            return isinstance(value, list)
        elif expected_type == 'object':
            return isinstance(value, dict)
        elif expected_type.startswith('regex:'):
            pattern = expected_type[6:]
            return isinstance(value, str) and bool(re.match(pattern, value))
        return False
