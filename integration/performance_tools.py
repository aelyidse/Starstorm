import time
import functools
from typing import Callable, Any, Dict, List

class PerformanceProfiler:
    """
    Provides decorators and utilities for profiling execution time and identifying bottlenecks.
    Aggregates timing statistics and supports reporting for optimization.
    """
    def __init__(self):
        self.stats: Dict[str, List[float]] = {}

    def profile(self, name: str):
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                self.stats.setdefault(name, []).append(elapsed)
                return result
            return wrapper
        return decorator

    def get_stats(self) -> Dict[str, List[float]]:
        return self.stats

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, times in self.stats.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'min': min(times),
                    'max': max(times),
                    'avg': sum(times) / len(times)
                }
        return summary

class Optimizer:
    """
    Provides basic optimization tools, such as timing-based suggestions and function comparison.
    """
    def compare(self, func_a: Callable, func_b: Callable, args=(), kwargs=None, runs=10) -> Dict[str, Any]:
        kwargs = kwargs or {}
        times_a = []
        times_b = []
        for _ in range(runs):
            start = time.perf_counter()
            func_a(*args, **kwargs)
            times_a.append(time.perf_counter() - start)
            start = time.perf_counter()
            func_b(*args, **kwargs)
            times_b.append(time.perf_counter() - start)
        return {
            'func_a_avg': sum(times_a) / runs,
            'func_b_avg': sum(times_b) / runs,
            'faster': 'A' if sum(times_a) < sum(times_b) else 'B'
        }
