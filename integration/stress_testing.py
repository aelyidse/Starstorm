import time
from typing import Callable, Dict, Any, List, Optional

class StressTester:
    """
    Executes stress testing routines to explore system limits under extreme conditions.
    Supports test registration, repeated execution, parameter sweeps, and result logging.
    """
    def __init__(self):
        self.tests: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self.results: List[Dict[str, Any]] = []

    def register_test(self, name: str, test_func: Callable[[Dict[str, Any]], Any]):
        self.tests[name] = test_func

    def run(self, name: str, params: Dict[str, Any], repetitions: int = 100) -> List[Any]:
        if name not in self.tests:
            raise ValueError(f"Test '{name}' not registered.")
        results = []
        for _ in range(repetitions):
            result = self.tests[name](params)
            results.append(result)
        self.results.append({'name': name, 'params': params, 'repetitions': repetitions, 'results': results})
        return results

    def parameter_sweep(self, name: str, param_grid: List[Dict[str, Any]], repetitions: int = 10) -> List[Dict[str, Any]]:
        if name not in self.tests:
            raise ValueError(f"Test '{name}' not registered.")
        sweep_results = []
        for params in param_grid:
            results = self.run(name, params, repetitions)
            sweep_results.append({'params': params, 'results': results})
        return sweep_results

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results
