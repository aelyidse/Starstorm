import argparse
import os
import sys
import json
import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from testing.scenario_testing import ScenarioTestCase, ScenarioTestSuite
from testing.performance_benchmarking import PerformanceBenchmark
from testing.regression_testing import RegressionTest, RegressionTestRunner, GitIntegrationTest

def run_all_tests(config_path: str) -> Dict[str, Any]:
    """
    Run all tests based on configuration
    
    Args:
        config_path: Path to test configuration file
        
    Returns:
        Aggregated test results
    """
    # Load test configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'scenario_tests': None,
        'performance_tests': None,
        'regression_tests': None
    }
    
    # Run scenario tests if configured
    if 'scenario_tests' in config:
        scenario_suite = ScenarioTestSuite(config['scenario_tests'].get('name', 'ScenarioTests'))
        # Load and run scenario tests
        # Implementation depends on specific test cases
        results['scenario_tests'] = scenario_suite.run_all()
    
    # Run performance tests if configured
    if 'performance_tests' in config:
        perf_tests = config['performance_tests']
        benchmark = PerformanceBenchmark(perf_tests.get('name', 'PerformanceBenchmarks'))
        # Run performance tests
        # Implementation depends on specific benchmarks
        report_path = benchmark.generate_report()
        results['performance_tests'] = {
            'report_path': report_path,
            'benchmarks': list(benchmark.results.keys())
        }
    
    # Run regression tests if configured
    if 'regression_tests' in config:
        regression_config = config['regression_tests']
        runner = RegressionTestRunner(regression_config.get('output_dir', 'regression_reports'))
        # Load and run regression tests
        # Implementation depends on specific test cases
        results['regression_tests'] = runner.run_all(None, regression_config.get('test_inputs', {}))
    
    # Save aggregated results
    output_dir = config.get('output_dir', 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, f"test_results_{timestamp}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated tests for Starstorm")
    parser.add_argument('--config', type=str, default='testing/test_config.json',
                        help='Path to test configuration file')
    args = parser.parse_args()
    
    results = run_all_tests(args.config)
    
    # Print summary
    print("Test Run Summary:")
    print(f"Timestamp: {results['timestamp']}")
    
    if results['scenario_tests']:
        scenario = results['scenario_tests']
        print(f"\nScenario Tests: {scenario['success']}")
        print(f"  Tests: {scenario['test_count']}")
        print(f"  Failures: {scenario['failures']}")
        print(f"  Errors: {scenario['errors']}")
    
    if results['performance_tests']:
        perf = results['performance_tests']
        print(f"\nPerformance Tests:")
        print(f"  Report: {perf['report_path']}")
        print(f"  Benchmarks: {', '.join(perf['benchmarks'])}")
    
    if results['regression_tests']:
        reg = results['regression_tests']
        print(f"\nRegression Tests:")
        print(f"  Total: {reg['total_tests']}")
        print(f"  Passed: {reg['passed']}")
        print(f"  Failed: {reg['failed']}")
        print(f"  No Snapshot: {reg['no_snapshot']}")