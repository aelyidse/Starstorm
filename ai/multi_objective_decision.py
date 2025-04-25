from typing import Dict, Any, List, Optional
import numpy as np

class MultiObjectiveDecisionMaker:
    """
    Implements decision-making with multiple objective optimization for autonomous mission management.
    Supports weighted sum, Pareto ranking, and constraint satisfaction.
    """
    def __init__(self, objectives: List[str], weights: Optional[Dict[str, float]] = None):
        self.objectives = objectives
        self.weights = weights or {obj: 1.0 for obj in objectives}
        self.last_options: Optional[List[Dict[str, Any]]] = None
        self.last_scores: Optional[List[float]] = None
        self.last_selection: Optional[Dict[str, Any]] = None

    def evaluate(self, options: List[Dict[str, Any]]) -> List[float]:
        # Each option: {'metrics': {'obj1': val1, 'obj2': val2, ...}, ...}
        scores = []
        for opt in options:
            score = 0.0
            for obj in self.objectives:
                val = opt['metrics'].get(obj, 0.0)
                score += self.weights.get(obj, 1.0) * val
            scores.append(score)
        self.last_options = options
        self.last_scores = scores
        return scores

    def select(self, options: List[Dict[str, Any]], method: str = 'weighted_sum', constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if method == 'weighted_sum':
            scores = self.evaluate(options)
            idx = int(np.argmax(scores))
            self.last_selection = options[idx]
            return options[idx]
        elif method == 'pareto':
            # Simple Pareto front selection (maximally non-dominated)
            pareto = self.pareto_front(options)
            self.last_selection = pareto[0] if pareto else None
            return pareto[0] if pareto else None
        elif method == 'constraint':
            # Select first option that satisfies all constraints
            for opt in options:
                if all(opt['metrics'].get(k, 0.0) >= v for k, v in (constraints or {}).items()):
                    self.last_selection = opt
                    return opt
            return None
        else:
            # Default: weighted sum
            return self.select(options, method='weighted_sum')

    def pareto_front(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Returns non-dominated options (Pareto front)
        metrics = np.array([list(opt['metrics'].values()) for opt in options])
        is_dominated = np.zeros(len(options), dtype=bool)
        for i, m in enumerate(metrics):
            for j, n in enumerate(metrics):
                if i != j and np.all(n >= m) and np.any(n > m):
                    is_dominated[i] = True
                    break
        return [options[i] for i, dom in enumerate(is_dominated) if not dom]

    def get_last_selection(self) -> Optional[Dict[str, Any]]:
        return self.last_selection
