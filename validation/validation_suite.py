from typing import Callable, Dict, Any, List, Optional

class Requirement:
    def __init__(self, req_id: str, description: str, validator: Callable[[Any], bool]):
        self.req_id = req_id
        self.description = description
        self.validator = validator

class ValidationSuite:
    """
    Comprehensive validation suite for all system requirements.
    Supports requirement registration, batch validation, result aggregation, and reporting.
    """
    def __init__(self):
        self.requirements: Dict[str, Requirement] = {}
        self.results: List[Dict[str, Any]] = []

    def register_requirement(self, req_id: str, description: str, validator: Callable[[Any], bool]):
        self.requirements[req_id] = Requirement(req_id, description, validator)

    def validate(self, test_subject: Any) -> List[Dict[str, Any]]:
        self.results = []
        for req_id, req in self.requirements.items():
            try:
                passed = req.validator(test_subject)
            except Exception as e:
                passed = False
            self.results.append({
                'requirement': req_id,
                'description': req.description,
                'passed': passed
            })
        return self.results

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results

    def get_requirements(self) -> List[Dict[str, str]]:
        return [{'req_id': r.req_id, 'description': r.description} for r in self.requirements.values()]
