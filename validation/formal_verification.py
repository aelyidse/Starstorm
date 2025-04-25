from typing import Callable, Dict, Any, List, Optional
import inspect

class FormalVerifier:
    """
    Provides formal verification methods for critical algorithms.
    Supports property registration, assertion checking, and verification reporting.
    """
    def __init__(self):
        self.properties: Dict[str, Callable[[Any], bool]] = {}
        self.results: List[Dict[str, Any]] = []

    def register_property(self, name: str, prop_func: Callable[[Any], bool]):
        self.properties[name] = prop_func

    def verify(self, func: Callable, test_cases: List[Any], property_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        to_check = property_names or list(self.properties.keys())
        verification_results = []
        for case in test_cases:
            output = func(*case) if isinstance(case, (tuple, list)) else func(case)
            for pname in to_check:
                prop_func = self.properties[pname]
                try:
                    result = prop_func(output)
                except Exception as e:
                    result = False
                verification_results.append({
                    'case': case,
                    'property': pname,
                    'result': result
                })
        self.results.extend(verification_results)
        return verification_results

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results

    def get_properties(self) -> List[str]:
        return list(self.properties.keys())
