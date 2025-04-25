import unittest
import json
import os
import datetime
import difflib
import git
from typing import Dict, Any, List, Optional, Callable, Union

class RegressionTest:
    """Base class for regression testing with snapshot comparison"""
    
    def __init__(self, test_name: str, snapshot_dir: str = 'regression_snapshots'):
        self.test_name = test_name
        self.snapshot_dir = snapshot_dir
        self.current_results = None
        os.makedirs(snapshot_dir, exist_ok=True)
    
    def run_test(self, system_under_test: Any, test_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the test and capture outputs
        
        Args:
            system_under_test: The system component being tested
            test_inputs: Input parameters for the test
            
        Returns:
            Test results
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement run_test")
    
    def _get_snapshot_path(self) -> str:
        """Get path to the snapshot file"""
        return os.path.join(self.snapshot_dir, f"{self.test_name}_snapshot.json")
    
    def save_snapshot(self) -> None:
        """Save current results as a snapshot for future comparison"""
        if self.current_results is None:
            raise ValueError("No test results to save. Run the test first.")
        
        with open(self._get_snapshot_path(), 'w') as f:
            json.dump({
                'timestamp': datetime.datetime.now().isoformat(),
                'results': self.current_results
            }, f, indent=2)
    
    def compare_with_snapshot(self) -> Dict[str, Any]:
        """Compare current results with saved snapshot"""
        if self.current_results is None:
            raise ValueError("No test results to compare. Run the test first.")
        
        snapshot_path = self._get_snapshot_path()
        if not os.path.exists(snapshot_path):
            return {
                'status': 'no_snapshot',
                'message': 'No snapshot exists for comparison'
            }
        
        with open(snapshot_path, 'r') as f:
            snapshot_data = json.load(f)
        
        snapshot_results = snapshot_data['results']
        
        # Compare results
        differences = self._compare_objects(snapshot_results, self.current_results)
        
        if differences:
            return {
                'status': 'regression',
                'differences': differences,
                'snapshot_timestamp': snapshot_data['timestamp']
            }
        else:
            return {
                'status': 'success',
                'message': 'Current results match snapshot',
                'snapshot_timestamp': snapshot_data['timestamp']
            }
    
    def _compare_objects(self, obj1: Any, obj2: Any, path: str = '') -> List[Dict[str, Any]]:
        """
        Recursively compare two objects and return differences
        
        Args:
            obj1: First object (snapshot)
            obj2: Second object (current)
            path: Current path in the object structure
            
        Returns:
            List of differences with paths
        """
        differences = []
        
        if type(obj1) != type(obj2):
            differences.append({
                'path': path,
                'type': 'type_mismatch',
                'snapshot_type': type(obj1).__name__,
                'current_type': type(obj2).__name__
            })
            return differences
        
        if isinstance(obj1, dict):
            # Compare dictionaries
            all_keys = set(obj1.keys()) | set(obj2.keys())
            for key in all_keys:
                key_path = f"{path}.{key}" if path else key
                
                if key not in obj1:
                    differences.append({
                        'path': key_path,
                        'type': 'key_added',
                        'value': obj2[key]
                    })
                elif key not in obj2:
                    differences.append({
                        'path': key_path,
                        'type': 'key_removed',
                        'value': obj1[key]
                    })
                else:
                    differences.extend(self._compare_objects(obj1[key], obj2[key], key_path))
        
        elif isinstance(obj1, list):
            # Compare lists
            if len(obj1) != len(obj2):
                differences.append({
                    'path': path,
                    'type': 'length_mismatch',
                    'snapshot_length': len(obj1),
                    'current_length': len(obj2)
                })
            
            # Compare elements
            for i in range(min(len(obj1), len(obj2))):
                item_path = f"{path}[{i}]"
                differences.extend(self._compare_objects(obj1[i], obj2[i], item_path))
        
        elif obj1 != obj2:
            # Compare primitive values
            differences.append({
                'path': path,
                'type': 'value_mismatch',
                'snapshot_value': obj1,
                'current_value': obj2
            })
        
        return differences

class RegressionTestRunner:
    """Manages and runs collections of regression tests"""
    
    def __init__(self, output_dir: str = 'regression_reports'):
        self.tests: List[RegressionTest] = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def add_test(self, test: RegressionTest) -> None:
        """Add a regression test to the runner"""
        self.tests.append(test)
    
    def run_all(self, system_under_test: Any, test_inputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run all regression tests
        
        Args:
            system_under_test: The system component being tested
            test_inputs: Dictionary mapping test names to input parameters
            
        Returns:
            Aggregated test results
        """
        results = {
            'total_tests': len(self.tests),
            'passed': 0,
            'failed': 0,
            'no_snapshot': 0,
            'test_results': {}
        }
        
        for test in self.tests:
            # Run the test
            test_inputs_dict = test_inputs.get(test.test_name, {})
            test.current_results = test.run_test(system_under_test, test_inputs_dict)
            
            # Compare with snapshot
            comparison = test.compare_with_snapshot()
            
            # Update aggregated results
            if comparison['status'] == 'success':
                results['passed'] += 1
            elif comparison['status'] == 'regression':
                results['failed'] += 1
            else:  # no_snapshot
                results['no_snapshot'] += 1
            
            results['test_results'][test.test_name] = comparison
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    def _generate_report(self, results: Dict[str, Any]) -> None:
        """Generate HTML report for regression test results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"regression_report_{timestamp}.html")
        
        with open(report_path, 'w') as f:
            f.write("<html><head><title>Regression Test Report</title></head><body>")
            f.write("<h1>Regression Test Report</h1>")
            f.write(f"<p>Generated: {datetime.datetime.now().isoformat()}</p>")
            
            # Summary
            f.write("<h2>Summary</h2>")
            f.write("<table border='1'>")
            f.write(f"<tr><td>Total Tests</td><td>{results['total_tests']}</td></tr>")
            f.write(f"<tr><td>Passed</td><td>{results['passed']}</td></tr>")
            f.write(f"<tr><td>Failed</td><td>{results['failed']}</td></tr>")
            f.write(f"<tr><td>No Snapshot</td><td>{results['no_snapshot']}</td></tr>")
            f.write("</table>")
            
            # Detailed results
            f.write("<h2>Test Details</h2>")
            for test_name, test_result in results['test_results'].items():
                f.write(f"<h3>Test: {test_name}</h3>")
                f.write(f"<p>Status: {test_result['status']}</p>")
                
                if test_result['status'] == 'regression':
                    f.write("<h4>Differences</h4>")
                    f.write("<table border='1'><tr><th>Path</th><th>Type</th><th>Details</th></tr>")
                    
                    for diff in test_result['differences']:
                        f.write(f"<tr><td>{diff['path']}</td><td>{diff['type']}</td><td>")
                        
                        if diff['type'] == 'value_mismatch':
                            f.write(f"Snapshot: {diff['snapshot_value']}<br>")
                            f.write(f"Current: {diff['current_value']}")
                        elif diff['type'] == 'type_mismatch':
                            f.write(f"Snapshot type: {diff['snapshot_type']}<br>")
                            f.write(f"Current type: {diff['current_type']}")
                        elif diff['type'] == 'length_mismatch':
                            f.write(f"Snapshot length: {diff['snapshot_length']}<br>")
                            f.write(f"Current length: {diff['current_length']}")
                        elif diff['type'] in ['key_added', 'key_removed']:
                            f.write(f"Value: {diff['value']}")
                        
                        f.write("</td></tr>")
                    
                    f.write("</table>")
            
            f.write("</body></html>")

class GitIntegrationTest(RegressionTest):
    """Regression test with Git integration for tracking changes"""
    
    def __init__(self, test_name: str, snapshot_dir: str = 'regression_snapshots', repo_path: str = '.'):
        super().__init__(test_name, snapshot_dir)
        self.repo_path = repo_path
        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            self.repo = None
    
    def save_snapshot(self) -> None:
        """Save snapshot and record Git commit information"""
        super().save_snapshot()
        
        if self.repo:
            # Add Git commit information to snapshot metadata
            snapshot_path = self._get_snapshot_path()
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            # Add Git metadata
            snapshot_data['git_metadata'] = {
                'commit_hash': self.repo.head.commit.hexsha,
                'commit_message': self.repo.head.commit.message,
                'branch': self.repo.active_branch.name,
                'author': f"{self.repo.head.commit.author.name} <{self.repo.head.commit.author.email}>",
                'commit_date': self.repo.head.commit.committed_datetime.isoformat()
            }
            
            # Save updated snapshot
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot_data, f, indent=2)