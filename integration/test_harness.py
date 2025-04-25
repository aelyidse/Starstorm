from typing import Callable, Dict, Any, List
import traceback

class SubsystemTestHarness:
    """
    Comprehensive test harness for subsystem integration and validation.
    Supports registration, execution, and result logging for all subsystem tests.
    """
    def __init__(self):
        self.tests: Dict[str, Callable[[], Any]] = {}
        self.results: List[Dict[str, Any]] = []

    def register_test(self, name: str, test_func: Callable[[], Any]):
        self.tests[name] = test_func

    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        for name, test in self.tests.items():
            try:
                result = test()
                self.results.append({'name': name, 'result': result, 'status': 'PASS'})
            except Exception as e:
                self.results.append({'name': name, 'result': str(e), 'traceback': traceback.format_exc(), 'status': 'FAIL'})
        return self.results

    def run_test(self, name: str) -> Dict[str, Any]:
        if name not in self.tests:
            return {'name': name, 'status': 'NOT_FOUND'}
        try:
            result = self.tests[name]()
            entry = {'name': name, 'result': result, 'status': 'PASS'}
        except Exception as e:
            entry = {'name': name, 'result': str(e), 'traceback': traceback.format_exc(), 'status': 'FAIL'}
        self.results.append(entry)
        return entry

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results
