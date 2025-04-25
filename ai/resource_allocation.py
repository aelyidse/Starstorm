from typing import Dict, Any, List, Optional
import numpy as np

class ResourceAllocationOptimizer:
    """
    Optimizes allocation of power, computation, and bandwidth resources for autonomous systems.
    Supports weighted objective optimization, constraints, and dynamic reallocation.
    """
    def __init__(self, total_resources: Dict[str, float], weights: Optional[Dict[str, float]] = None):
        """
        total_resources: {'power': W, 'compute': GFLOPS, 'bandwidth': Mbps}
        weights: {'power': w1, 'compute': w2, 'bandwidth': w3}
        """
        self.total_resources = total_resources
        self.weights = weights or {k: 1.0 for k in total_resources}
        self.last_allocation: Optional[Dict[str, Dict[str, float]]] = None

    def optimize(self, demands: List[Dict[str, Any]], constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        """
        demands: List of {'id': subsystem, 'power': W, 'compute': GFLOPS, 'bandwidth': Mbps, 'priority': int}
        constraints: Optional resource constraints per subsystem
        Returns: {subsystem: {'power': x, 'compute': y, 'bandwidth': z}}
        """
        n = len(demands)
        alloc = {d['id']: {'power': 0, 'compute': 0, 'bandwidth': 0} for d in demands}
        # Normalize priorities
        priorities = np.array([d.get('priority', 1) for d in demands])
        norm_p = priorities / priorities.sum() if priorities.sum() > 0 else np.ones(n) / n
        for i, d in enumerate(demands):
            for res in ['power', 'compute', 'bandwidth']:
                req = float(d.get(res, 0))
                avail = self.total_resources[res]
                alloc[d['id']][res] = min(req, avail * norm_p[i])
                # Apply constraints if present
                if constraints and d['id'] in constraints and res in constraints[d['id']]:
                    alloc[d['id']][res] = min(alloc[d['id']][res], constraints[d['id']][res])
        self.last_allocation = alloc
        return alloc

    def get_last_allocation(self) -> Optional[Dict[str, Dict[str, float]]]:
        return self.last_allocation
