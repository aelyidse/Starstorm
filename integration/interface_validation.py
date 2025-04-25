from typing import Callable, Dict, Any, List, Optional, Tuple, Type
from core.interface import ComponentInterface

class InterfaceValidator:
    """
    Validates interfaces between system components for compatibility and contract adherence.
    Supports signature checks, type validation, and custom test execution.
    """
    def __init__(self):
        self.interface_tests: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []

    def register_interface_test(self, name: str, producer: Callable, consumer: Callable, 
                              test_func: Optional[Callable] = None,
                              version_check: Optional[Tuple[str, str]] = None):
        """
        Register an interface test between a producer and consumer.
        
        Args:
            name: Test name
            producer: Function that produces data
            consumer: Function that consumes data
            test_func: Optional custom validation function
            version_check: Optional tuple of (required_version, available_version) for version checking
        """
        self.interface_tests.append({
            'name': name,
            'producer': producer,
            'consumer': consumer,
            'test_func': test_func,
            'version_check': version_check
        })
    
    def validate_interface_contract(self, interface_class: Type[ComponentInterface], 
                                  implementation: Any) -> List[str]:
        """
        Validate that an implementation satisfies an interface contract.
        
        Args:
            interface_class: The interface class to validate against
            implementation: The implementation to validate
            
        Returns:
            List of validation errors, empty if valid
        """
        try:
            interface_instance = interface_class()
            return interface_instance.validate_implementation(implementation)
        except Exception as e:
            return [f"Error validating interface: {str(e)}"]

    def validate_all(self) -> List[Dict[str, Any]]:
        self.results = []
        for test in self.interface_tests:
            name = test['name']
            try:
                # Version compatibility check if specified
                if test['version_check']:
                    required, available = test['version_check']
                    req_version = Version.parse(required)
                    avail_version = Version.parse(available)
                    
                    # Basic semver compatibility check
                    if req_version.major != avail_version.major:
                        raise ValueError(f"Incompatible major versions: {required} vs {available}")
                    if req_version.major == 0 and req_version.minor != avail_version.minor:
                        raise ValueError(f"Incompatible minor versions in development release: {required} vs {available}")
                
                # Basic signature and type check
                produced = test['producer']()
                consumed = test['consumer'](produced)
                
                # Custom logic if provided
                if test['test_func']:
                    assert test['test_func'](produced, consumed)
                    
                self.results.append({'name': name, 'status': 'PASS'})
            except Exception as e:
                self.results.append({'name': name, 'status': 'FAIL', 'error': str(e)})
        return self.results

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results
