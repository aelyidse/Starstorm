import unittest
import coverage
import json
import os
from typing import Dict, Any, List, Optional, Callable

class ScenarioTestCase(unittest.TestCase):
    """Base class for scenario-based testing with coverage analysis"""
    
    def setUp(self):
        self.cov = coverage.Coverage(source=['ai', 'control', 'communications', 'propulsion'])
        self.cov.start()
        
    def tearDown(self):
        self.cov.stop()
        self.cov.save()
        
    def run_scenario(self, scenario_config: Dict[str, Any], system_under_test: Any) -> Dict[str, Any]:
        """
        Run a test scenario based on configuration
        
        Args:
            scenario_config: Scenario configuration with inputs and expected outputs
            system_under_test: The system component being tested
            
        Returns:
            Test results including actual outputs and metrics
        """
        # Extract scenario parameters
        inputs = scenario_config.get('inputs', {})
        expected_outputs = scenario_config.get('expected_outputs', {})
        
        # Execute the scenario
        actual_outputs = {}
        for input_name, input_value in inputs.items():
            if hasattr(system_under_test, input_name):
                method = getattr(system_under_test, input_name)
                if callable(method):
                    result = method(input_value)
                    actual_outputs[input_name] = result
        
        # Validate outputs against expectations
        validation_results = {}
        for output_name, expected_value in expected_outputs.items():
            if output_name in actual_outputs:
                actual_value = actual_outputs[output_name]
                is_valid = actual_value == expected_value
                validation_results[output_name] = {
                    'expected': expected_value,
                    'actual': actual_value,
                    'valid': is_valid
                }
        
        return {
            'inputs': inputs,
            'outputs': actual_outputs,
            'validation': validation_results,
            'success': all(r.get('valid', False) for r in validation_results.values())
        }
    
    def generate_coverage_report(self, output_dir: str = 'coverage_reports') -> Dict[str, Any]:
        """Generate and return coverage metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate reports
        self.cov.html_report(directory=os.path.join(output_dir, 'html'))
        self.cov.xml_report(outfile=os.path.join(output_dir, 'coverage.xml'))
        
        # Get coverage data
        data = self.cov.get_data()
        report = self.cov.report(show_missing=False)
        
        return {
            'total_coverage_percent': report,
            'report_dir': output_dir
        }

class ScenarioTestSuite:
    """Manages collections of scenario tests with aggregated reporting"""
    
    def __init__(self, name: str):
        self.name = name
        self.test_cases = []
        self.results = []
        
    def add_test_case(self, test_case: ScenarioTestCase):
        self.test_cases.append(test_case)
        
    def run_all(self) -> Dict[str, Any]:
        """Run all test cases and aggregate results"""
        suite = unittest.TestSuite()
        for test_case in self.test_cases:
            suite.addTest(test_case)
            
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        
        # Aggregate coverage from all test cases
        combined_coverage = coverage.Coverage()
        for test_case in self.test_cases:
            if hasattr(test_case, 'cov') and test_case.cov:
                combined_coverage.combine([test_case.cov])
        
        # Generate combined report
        report_dir = f'coverage_reports/{self.name}'
        os.makedirs(report_dir, exist_ok=True)
        combined_coverage.html_report(directory=os.path.join(report_dir, 'html'))
        combined_coverage.xml_report(outfile=os.path.join(report_dir, 'coverage.xml'))
        
        return {
            'test_count': suite.countTestCases(),
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': result.wasSuccessful(),
            'coverage_report': report_dir
        }