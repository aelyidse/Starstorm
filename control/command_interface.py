import hashlib
import hmac
from typing import Dict, Any, Optional

class SecureCommandValidator:
    """
    Implements secure command validation and authentication for command and control interfaces.
    Supports HMAC-based authentication and command structure validation.
    """
    def __init__(self, secret_key: bytes, allowed_commands: Optional[Dict[str, Any]] = None):
        self.secret_key = secret_key
        self.allowed_commands = allowed_commands or {}
        self.last_validation_result: Optional[Dict[str, Any]] = None

    def validate_command(self, command: Dict[str, Any], signature: str) -> bool:
        # Validate command structure
        if command['type'] not in self.allowed_commands:
            self.last_validation_result = {'valid': False, 'reason': 'Unknown command type'}
            return False
        # Compute HMAC signature
        msg = str(command).encode('utf-8')
        expected_sig = hmac.new(self.secret_key, msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, signature):
            self.last_validation_result = {'valid': False, 'reason': 'Invalid signature'}
            return False
        self.last_validation_result = {'valid': True, 'reason': 'Authenticated'}
        return True

    def get_last_validation_result(self) -> Optional[Dict[str, Any]]:
        return self.last_validation_result
