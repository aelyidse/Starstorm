import hashlib
import json
from typing import Dict, Any, List, Optional
import os

class TamperEvidentTelemetryStorage:
    """
    Provides secure telemetry storage with tamper-evident features.
    Stores telemetry as a hash-chained log, enabling detection of unauthorized modifications.
    """
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.hash_chain: List[str] = []
        self.last_hash: str = '0' * 64  # Genesis hash
        self.log_index = 0
        # Create file if it doesn't exist
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, 'w') as f:
                f.write('')

    def store(self, telemetry: Dict[str, Any]):
        record = {
            'index': self.log_index,
            'telemetry': telemetry,
            'prev_hash': self.last_hash
        }
        record_str = json.dumps(record, sort_keys=True)
        record_hash = hashlib.sha256(record_str.encode('utf-8')).hexdigest()
        self.hash_chain.append(record_hash)
        self.last_hash = record_hash
        self.log_index += 1
        # Append to file
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps({'record': record, 'hash': record_hash}) + '\n')

    def verify_integrity(self) -> bool:
        # Recompute hash chain from file
        prev_hash = '0' * 64
        with open(self.storage_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                record = entry['record']
                stored_hash = entry['hash']
                if record['prev_hash'] != prev_hash:
                    return False
                record_str = json.dumps(record, sort_keys=True)
                computed_hash = hashlib.sha256(record_str.encode('utf-8')).hexdigest()
                if computed_hash != stored_hash:
                    return False
                prev_hash = stored_hash
        return True

    def get_hash_chain(self) -> List[str]:
        return self.hash_chain
