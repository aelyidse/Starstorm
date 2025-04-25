import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, List, Optional, Callable, Tuple
from functools import wraps

class PerformanceBenchmark:
    """Performance benchmarking tool for measuring execution time and resource usage"""
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
        self.baseline_results = {}
        
    def benchmark(self, iterations: int = 100, warmup: int = 10):
        """Decorator for benchmarking a function"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Warmup runs
                for _ in range(warmup):
                    func(*args, **kwargs)
                
                # Timed runs
                execution_times = []
                for _ in range(iterations):
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    end_time = time.perf_counter()
                    execution_times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Calculate statistics
                stats = {
                    'min': min(execution_times),
                    'max': max(execution_times),
                    'mean': statistics.mean(execution_times),
                    'median': statistics.median(execution_times),
                    'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    'iterations': iterations
                }
                
                # Store results
                func_name = func.__name__
                self.results[func_name] = stats
                
                return result
            return wrapper
        return decorator
    
    def set_baseline(self, func_name: str):
        """Set current results as baseline for comparison"""
        if func_name in self.results:
            self.baseline_results[func_name] = self.results[func_name].copy()
    
    def compare_to_baseline(self, func_name: str) -> Dict[str, Any]:
        """Compare current results to baseline"""
        if func_name not in self.results or func_name not in self.baseline_results:
            return {'error': 'No baseline or current results found'}
        
        current = self.results[func_name]
        baseline = self.baseline_results[func_name]
        
        comparison = {}
        for metric in ['min', 'max', 'mean', 'median']:
            if baseline[metric] == 0:
                comparison[f'{metric}_change_percent'] = float('inf')
            else:
                change = ((current[metric] - baseline[metric]) / baseline[metric]) * 100
                comparison[f'{metric}_change_percent'] = change
        
        comparison['current'] = current
        comparison['baseline'] = baseline
        
        return comparison
    
    def generate_report(self, output_dir: str = 'benchmark_reports') -> str:
        """Generate performance report with visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"{self.name}_report.html")
        
        # Create plots
        for func_name, stats in self.results.items():
            plt.figure(figsize=(10, 6))
            
            # If we have baseline, add comparison
            if func_name in self.baseline_results:
                baseline = self.baseline_results[func_name]
                plt.axhline(y=baseline['mean'], color='r', linestyle='--', label=f'Baseline Mean: {baseline["mean"]:.2f}ms')
            
            # Plot current results
            plt.bar(['Min', 'Mean', 'Median', 'Max'], 
                   [stats['min'], stats['mean'], stats['median'], stats['max']])
            
            plt.title(f'Performance Metrics for {func_name}')
            plt.ylabel('Time (ms)')
            plt.savefig(os.path.join(output_dir, f"{func_name}_metrics.png"))
            plt.close()
        
        # Generate HTML report
        with open(report_path, 'w') as f:
            f.write("<html><head><title>Performance Benchmark Report</title></head><body>")
            f.write(f"<h1>Performance Report: {self.name}</h1>")
            
            for func_name, stats in self.results.items():
                f.write(f"<h2>Function: {func_name}</h2>")
                f.write("<table border='1'><tr><th>Metric</th><th>Value</th></tr>")
                for metric, value in stats.items():
                    if isinstance(value, (int, float)):
                        f.write(f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>")
                    else:
                        f.write(f"<tr><td>{metric}</td><td>{value}</td></tr>")
                f.write("</table>")
                
                # Add comparison if baseline exists
                if func_name in self.baseline_results:
                    comparison = self.compare_to_baseline(func_name)
                    f.write("<h3>Comparison to Baseline</h3>")
                    f.write("<table border='1'><tr><th>Metric</th><th>Change %</th></tr>")
                    for metric, value in comparison.items():
                        if 'change_percent' in metric:
                            f.write(f"<tr><td>{metric}</td><td>{value:.2f}%</td></tr>")
                    f.write("</table>")
                
                # Add plot
                f.write(f"<img src='{func_name}_metrics.png' alt='Performance metrics'>")
            
            f.write("</body></html>")
        
        return report_path