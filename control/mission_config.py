import json
from typing import Dict, Any, Optional

class MissionConfigManager:
    """
    Manages mission configuration and upload functionality for autonomous systems.
    Supports loading, validating, and uploading mission configs.
    """
    def __init__(self, allowed_keys: Optional[list] = None):
        self.allowed_keys = allowed_keys or ['objectives', 'constraints', 'resources', 'timing']
        self.current_config: Optional[Dict[str, Any]] = None
        self.last_upload_status: Optional[str] = None

    def load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if not self.validate_config(config):
                self.last_upload_status = 'Invalid config structure'
                return None
            self.current_config = config
            return config
        except Exception as e:
            self.last_upload_status = f'Load error: {e}'
            return None

    def validate_config(self, config: Dict[str, Any]) -> bool:
        # Check for required keys
        for k in self.allowed_keys:
            if k not in config:
                return False
        return True

    def upload_config(self, upload_func) -> bool:
        """
        upload_func: function to handle actual upload (e.g., to vehicle or remote server)
        Returns True if upload succeeds.
        """
        if not self.current_config:
            self.last_upload_status = 'No config loaded'
            return False
        try:
            upload_func(self.current_config)
            self.last_upload_status = 'Upload successful'
            return True
        except Exception as e:
            self.last_upload_status = f'Upload error: {e}'
            return False

    def get_last_status(self) -> Optional[str]:
        return self.last_upload_status
