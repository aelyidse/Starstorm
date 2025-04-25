from typing import Dict, Any, List, Optional, Union, Callable
import re
from enum import Enum
import json

class TokenType(Enum):
    """Token types for command lexical analysis"""
    COMMAND = 'COMMAND'
    PARAMETER = 'PARAMETER'
    VALUE = 'VALUE'
    OPERATOR = 'OPERATOR'
    SEPARATOR = 'SEPARATOR'
    IDENTIFIER = 'IDENTIFIER'
    NUMBER = 'NUMBER'
    STRING = 'STRING'
    BOOLEAN = 'BOOLEAN'
    EOF = 'EOF'

class Token:
    """Represents a lexical token in the command language"""
    def __init__(self, token_type: TokenType, value: str, position: int):
        self.type = token_type
        self.value = value
        self.position = position
    
    def __repr__(self):
        return f"Token({self.type}, '{self.value}', {self.position})"

class CommandLexer:
    """
    Lexical analyzer for command processing.
    Converts raw command strings into tokens for parsing.
    """
    def __init__(self):
        self.command_patterns = {
            TokenType.COMMAND: r'^(set|get|execute|query|abort|pause|resume)',
            TokenType.PARAMETER: r'^--[a-zA-Z][a-zA-Z0-9_]*',
            TokenType.OPERATOR: r'^(==|!=|>=|<=|>|<|=)',
            TokenType.SEPARATOR: r'^[,;:]',
            TokenType.NUMBER: r'^-?\d+(\.\d+)?',
            TokenType.STRING: r'^"[^"]*"',
            TokenType.BOOLEAN: r'^(true|false)',
            TokenType.IDENTIFIER: r'^[a-zA-Z][a-zA-Z0-9_]*',
        }
    
    def tokenize(self, command_str: str) -> List[Token]:
        """Convert a command string into a list of tokens"""
        tokens = []
        position = 0
        
        # Strip leading/trailing whitespace
        command_str = command_str.strip()
        
        while position < len(command_str):
            # Skip whitespace
            if command_str[position].isspace():
                position += 1
                continue
            
            match_found = False
            for token_type, pattern in self.command_patterns.items():
                regex = re.compile(pattern)
                match = regex.match(command_str[position:])
                
                if match:
                    value = match.group(0)
                    tokens.append(Token(token_type, value, position))
                    position += len(value)
                    match_found = True
                    break
            
            if not match_found:
                raise ValueError(f"Invalid token at position {position}: '{command_str[position:]}'")
        
        # Add EOF token
        tokens.append(Token(TokenType.EOF, "", position))
        return tokens

class CommandSyntaxError(Exception):
    """Exception raised for command syntax errors"""
    def __init__(self, message: str, position: int):
        self.message = message
        self.position = position
        super().__init__(f"{message} at position {position}")

class CommandParser:
    """
    Parser for command grammar.
    Validates command syntax and builds command objects.
    """
    def __init__(self, grammar_rules: Optional[Dict[str, Any]] = None):
        self.grammar_rules = grammar_rules or self._default_grammar()
        self.tokens: List[Token] = []
        self.current_index = 0
    
    def _default_grammar(self) -> Dict[str, Any]:
        """Define default grammar rules"""
        return {
            'commands': {
                'set': {
                    'parameters': ['target', 'value'],
                    'optional_parameters': ['timeout', 'units']
                },
                'get': {
                    'parameters': ['target'],
                    'optional_parameters': ['format']
                },
                'execute': {
                    'parameters': ['action'],
                    'optional_parameters': ['priority', 'timeout']
                },
                'query': {
                    'parameters': ['resource'],
                    'optional_parameters': ['filter', 'limit']
                },
                'abort': {
                    'parameters': ['target'],
                    'optional_parameters': []
                },
                'pause': {
                    'parameters': ['target'],
                    'optional_parameters': []
                },
                'resume': {
                    'parameters': ['target'],
                    'optional_parameters': []
                }
            },
            'parameter_types': {
                'target': 'IDENTIFIER',
                'value': ['NUMBER', 'STRING', 'BOOLEAN', 'IDENTIFIER'],
                'timeout': 'NUMBER',
                'units': 'STRING',
                'format': 'STRING',
                'action': 'IDENTIFIER',
                'priority': 'NUMBER',
                'resource': 'IDENTIFIER',
                'filter': 'STRING',
                'limit': 'NUMBER'
            }
        }
    
    def parse(self, tokens: List[Token]) -> Dict[str, Any]:
        """Parse a list of tokens into a command object"""
        self.tokens = tokens
        self.current_index = 0
        
        # Expect a command token first
        if self.current_token.type != TokenType.COMMAND:
            raise CommandSyntaxError("Expected command", self.current_token.position)
        
        command_name = self.current_token.value
        if command_name not in self.grammar_rules['commands']:
            raise CommandSyntaxError(f"Unknown command: {command_name}", self.current_token.position)
        
        command_obj = {'command': command_name, 'parameters': {}}
        self.advance()  # Move past command token
        
        # Parse parameters
        while self.current_token.type != TokenType.EOF:
            if self.current_token.type != TokenType.PARAMETER:
                raise CommandSyntaxError("Expected parameter", self.current_token.position)
            
            # Extract parameter name without the -- prefix
            param_name = self.current_token.value[2:]
            self.advance()  # Move past parameter token
            
            # Check if parameter is valid for this command
            command_rules = self.grammar_rules['commands'][command_name]
            if param_name not in command_rules['parameters'] and param_name not in command_rules['optional_parameters']:
                raise CommandSyntaxError(f"Invalid parameter '{param_name}' for command '{command_name}'", 
                                        self.tokens[self.current_index-1].position)
            
            # Expect a value
            if self.current_token.type == TokenType.EOF:
                raise CommandSyntaxError(f"Expected value for parameter '{param_name}'", 
                                        self.tokens[self.current_index-1].position)
            
            # Validate parameter type
            expected_types = self.grammar_rules['parameter_types'][param_name]
            if isinstance(expected_types, str):
                expected_types = [expected_types]
            
            if self.current_token.type.name not in expected_types:
                raise CommandSyntaxError(
                    f"Invalid type for parameter '{param_name}'. Expected {expected_types}, got {self.current_token.type.name}",
                    self.current_token.position
                )
            
            # Store parameter value
            value = self.current_token.value
            # Convert value based on token type
            if self.current_token.type == TokenType.NUMBER:
                value = float(value) if '.' in value else int(value)
            elif self.current_token.type == TokenType.STRING:
                value = value[1:-1]  # Remove quotes
            elif self.current_token.type == TokenType.BOOLEAN:
                value = value.lower() == 'true'
            
            command_obj['parameters'][param_name] = value
            self.advance()  # Move past value token
        
        # Check for required parameters
        for required_param in self.grammar_rules['commands'][command_name]['parameters']:
            if required_param not in command_obj['parameters']:
                raise CommandSyntaxError(f"Missing required parameter '{required_param}'", 
                                        self.tokens[-1].position)
        
        return command_obj
    
    @property
    def current_token(self) -> Token:
        """Get the current token"""
        return self.tokens[self.current_index]
    
    def advance(self):
        """Move to the next token"""
        if self.current_index < len(self.tokens) - 1:
            self.current_index += 1

class CommandInterpreter:
    """
    Interprets and validates commands using formal grammar.
    Provides a pipeline for command processing from string to executable form.
    """
    def __init__(self, grammar_file: Optional[str] = None):
        self.grammar = self._load_grammar(grammar_file)
        self.lexer = CommandLexer()
        self.parser = CommandParser(self.grammar)
        self.validators: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], bool]] = {}
    
    def _load_grammar(self, grammar_file: Optional[str]) -> Dict[str, Any]:
        """Load grammar from file or use default"""
        if grammar_file:
            try:
                with open(grammar_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading grammar file: {e}")
                return {}
        return {}
    
    def register_validator(self, command: str, validator_func: Callable[[Dict[str, Any], Dict[str, Any]], bool]):
        """Register a validator function for a specific command"""
        self.validators[command] = validator_func
    
    def interpret_command(self, command_str: str, system_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a command string through the entire pipeline:
        1. Lexical analysis (tokenization)
        2. Syntax parsing
        3. Semantic validation
        4. Command object creation
        
        Returns a validated command object ready for execution
        """
        try:
            # Lexical analysis
            tokens = self.lexer.tokenize(command_str)
            
            # Syntax parsing
            command_obj = self.parser.parse(tokens)
            
            # Semantic validation
            if command_obj['command'] in self.validators and system_state is not None:
                if not self.validators[command_obj['command']](command_obj, system_state):
                    raise ValueError(f"Command validation failed for {command_obj['command']}")
            
            return command_obj
            
        except CommandSyntaxError as e:
            # Enhance error with position indicator
            error_indicator = command_str + '\n' + ' ' * e.position + '^'
            raise ValueError(f"Syntax error: {e.message}\n{error_indicator}")
        except Exception as e:
            raise ValueError(f"Command interpretation error: {str(e)}")