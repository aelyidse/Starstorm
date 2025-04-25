import json
from typing import Dict, Any, List, Optional
import time

class CertificationEvidenceGenerator:
    """
    Generates certification evidence artifacts for compliance requirements.
    Supports evidence collection, formatting, export, and traceability.
    """
    def __init__(self):
        self.evidence: List[Dict[str, Any]] = []
        self.export_log: List[str] = []

    def collect(self, requirement_id: str, result: Any, description: Optional[str] = None, source: Optional[str] = None):
        entry = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'requirement_id': requirement_id,
            'result': result,
            'description': description or '',
            'source': source or ''
        }
        self.evidence.append(entry)

    def export(self, path: str) -> bool:
        try:
            with open(path, 'w') as f:
                json.dump(self.evidence, f, indent=2)
            self.export_log.append(f"Exported to {path} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
            return True
        except Exception as e:
            self.export_log.append(f"Export failed: {e}")
            return False

    def get_evidence(self) -> List[Dict[str, Any]]:
        return self.evidence

    def get_export_log(self) -> List[str]:
        return self.export_log
