import random
from typing import Callable, Dict, Any, List, Optional

class MonteCarloSimulator:
    """
    Runs Monte Carlo simulations for reliability assessment and probabilistic analysis.
    Supports scenario function registration, repeated sampling, and result aggregation.
    """
    def __init__(self):
        self.scenarios: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self.results: List[Dict[str, Any]] = []

    def register_scenario(self, name: str, scenario_func: Callable[[Dict[str, Any]], Any]):
        self.scenarios[name] = scenario_func

    def run(self, name: str, params: Dict[str, Any], runs: int = 1000) -> List[Any]:
        if name not in self.scenarios:
            raise ValueError(f"Scenario '{name}' not registered.")
        scenario_func = self.scenarios[name]
        run_results = []
        for _ in range(runs):
            # Inject randomization for each run
            randomized_params = {k: v() if callable(v) else v for k, v in params.items()}
            result = scenario_func(randomized_params)
            run_results.append(result)
        self.results.append({'name': name, 'params': params, 'runs': runs, 'results': run_results})
        return run_results

    def aggregate_results(self, results: List[Any], metric: Optional[Callable[[List[Any]], Any]] = None) -> Any:
        if metric:
            return metric(results)
        # Default: return mean if results are numeric
        if results and isinstance(results[0], (int, float)):
            return sum(results) / len(results)
        return results

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results
