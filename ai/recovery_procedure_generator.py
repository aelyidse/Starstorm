from typing import Dict, Any, List, Optional, Tuple
import time

class RecoveryProcedureGenerator:
    """
    Generates automated recovery procedures based on fault diagnostics.
    Creates step-by-step recovery plans tailored to specific failure modes.
    """
    def __init__(self, subsystem_dependencies: Optional[Dict[str, List[str]]] = None):
        self.subsystem_dependencies = subsystem_dependencies or {}
        self.recovery_templates: Dict[str, List[Dict[str, str]]] = {}
        self.generated_procedures: List[Dict[str, Any]] = []
        
    def register_recovery_template(self, failure_type: str, steps: List[Dict[str, str]]):
        """
        Register a template for a specific failure type
        
        Args:
            failure_type: The type of failure (e.g., 'power_loss', 'comms_failure')
            steps: List of recovery steps with 'action' and 'description' keys
        """
        self.recovery_templates[failure_type] = steps
        
    def generate_procedure(self, 
                          subsystem: str, 
                          failure_type: str, 
                          diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a recovery procedure for a specific failure
        
        Args:
            subsystem: The affected subsystem
            failure_type: The type of failure detected
            diagnostics: Diagnostic data about the failure
            
        Returns:
            Complete recovery procedure with steps and metadata
        """
        # Start with template if available, otherwise create generic steps
        steps = []
        if failure_type in self.recovery_templates:
            steps = self.recovery_templates[failure_type].copy()
        else:
            # Generate generic recovery steps
            steps = [
                {'action': 'diagnose', 'description': f'Run full diagnostics on {subsystem}'},
                {'action': 'isolate', 'description': f'Isolate {subsystem} from other systems'},
                {'action': 'reset', 'description': f'Attempt soft reset of {subsystem}'},
                {'action': 'restore', 'description': f'Restore {subsystem} to default configuration'},
                {'action': 'verify', 'description': f'Verify {subsystem} functionality'}
            ]
            
        # Add dependency checks if applicable
        if subsystem in self.subsystem_dependencies:
            for dep in self.subsystem_dependencies[subsystem]:
                steps.insert(0, {
                    'action': 'check_dependency',
                    'description': f'Verify {dep} is operational before proceeding'
                })
                
        # Create the complete procedure
        procedure = {
            'id': f"{subsystem}_{failure_type}_{int(time.time())}",
            'subsystem': subsystem,
            'failure_type': failure_type,
            'creation_time': time.time(),
            'severity': self._determine_severity(failure_type, diagnostics),
            'steps': steps,
            'estimated_recovery_time': self._estimate_recovery_time(steps),
            'diagnostics': diagnostics
        }
        
        self.generated_procedures.append(procedure)
        return procedure
    
    def _determine_severity(self, failure_type: str, diagnostics: Dict[str, Any]) -> str:
        """Determine the severity of the failure based on diagnostics"""
        # This could be expanded with more sophisticated logic
        if 'critical' in failure_type.lower():
            return 'critical'
        elif 'major' in failure_type.lower():
            return 'major'
        elif diagnostics.get('impact', '') == 'mission_threatening':
            return 'critical'
        elif diagnostics.get('redundancy_available', True) == False:
            return 'major'
        else:
            return 'minor'
    
    def _estimate_recovery_time(self, steps: List[Dict[str, str]]) -> float:
        """Estimate recovery time based on number and type of steps"""
        # Simple estimation - could be made more sophisticated
        base_time = 60.0  # Base time in seconds
        return base_time * len(steps)
    
    def get_procedure_by_id(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a previously generated procedure by ID"""
        for proc in self.generated_procedures:
            if proc['id'] == procedure_id:
                return proc
        return None
    
    def get_procedures_for_subsystem(self, subsystem: str) -> List[Dict[str, Any]]:
        """Get all procedures generated for a specific subsystem"""
        return [p for p in self.generated_procedures if p['subsystem'] == subsystem]