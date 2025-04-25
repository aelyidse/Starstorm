from typing import Dict, Any, List, Optional, Set, Tuple, Type, Callable
import inspect
import importlib
import os
import sys
from pathlib import Path
import json
from .dependency_analyzer import DependencyAnalyzer

class IntegrationTestGenerator:
    """
    Generates integration tests for component interactions.
    Creates test scenarios based on component interfaces and dependencies.
    """
    
    def __init__(self, dependency_analyzer: Optional[DependencyAnalyzer] = None):
        self.dependency_analyzer = dependency_analyzer or DependencyAnalyzer()
        self.test_templates: Dict[str, str] = {}
        self.component_mocks: Dict[str, Dict[str, Any]] = {}
    
    def register_test_template(self, template_name: str, template: str) -> None:
        """Register a test template for a specific integration scenario."""
        self.test_templates[template_name] = template
    
    def register_component_mock(self, component_name: str, mock_config: Dict[str, Any]) -> None:
        """Register mock configuration for a component."""
        self.component_mocks[component_name] = mock_config
    
    def generate_integration_test(self, components: List[str], template_name: str = "default") -> str:
        """
        Generate an integration test for a set of components.
        
        Args:
            components: List of component names to include in the test
            template_name: Name of the template to use
            
        Returns:
            Generated test code as string
        """
        if template_name not in self.test_templates:
            raise ValueError(f"Test template '{template_name}' not found")
        
        template = self.test_templates[template_name]
        
        # Get dependencies between components
        dependencies = {}
        for component in components:
            if self.dependency_analyzer and component in self.dependency_analyzer.component_registry:
                deps = list(self.dependency_analyzer.dependency_graph.successors(component))
                dependencies[component] = [d for d in deps if d in components]
        
        # Generate mock setup code
        mock_setup = []
        for component in components:
            if component in self.component_mocks:
                mock_config = self.component_mocks[component]
                mock_setup.append(f"# Mock setup for {component}")
                for method, return_value in mock_config.get('methods', {}).items():
                    mock_setup.append(f"{component.lower()}_mock.{method}.return_value = {repr(return_value)}")
        
        # Fill in template
        test_code = template.format(
            components=", ".join(components),
            dependencies=json.dumps(dependencies, indent=4),
            mock_setup="\n".join(mock_setup)
        )
        
        return test_code
    
    def generate_test_suite(self, output_dir: str) -> Dict[str, str]:
        """
        Generate a complete test suite for all registered components.
        
        Args:
            output_dir: Directory to write test files
            
        Returns:
            Dictionary mapping test file paths to generated code
        """
        if not self.dependency_analyzer or not self.dependency_analyzer.component_registry:
            return {}
        
        os.makedirs(output_dir, exist_ok=True)
        generated_tests = {}
        
        # Generate tests for each component and its direct dependencies
        for component in self.dependency_analyzer.component_registry:
            dependencies = list(self.dependency_analyzer.dependency_graph.successors(component))
            if not dependencies:
                continue
            
            test_components = [component] + dependencies
            test_code = self.generate_integration_test(test_components)
            
            test_file = os.path.join(output_dir, f"test_integration_{component.lower()}.py")
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            generated_tests[test_file] = test_code
        
        # Generate tests for identified dependency cycles
        cycles = self.dependency_analyzer.find_cycles()
        for i, cycle in enumerate(cycles):
            if not cycle:
                continue
            
            test_code = self.generate_integration_test(cycle, template_name="cycle")
            
            test_file = os.path.join(output_dir, f"test_cycle_{i+1}.py")
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            generated_tests[test_file] = test_code
        
        return generated_tests
    
    def generate_default_templates(self) -> None:
        """Generate default test templates."""
        self.test_templates["default"] = """
import unittest
from unittest.mock import MagicMock, patch

# Integration test for components: {components}
class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        # Setup component mocks
        {mock_setup}
        
        # Component dependency information
        self.dependencies = {dependencies}
    
    def test_component_interaction(self):
        # Test that components interact correctly
        pass
    
    def test_error_handling(self):
        # Test error handling between components
        pass
    
    def test_data_flow(self):
        # Test data flow between components
        pass

if __name__ == '__main__':
    unittest.main()
"""
        
        self.test_templates["cycle"] = """
import unittest
from unittest.mock import MagicMock, patch

# Integration test for dependency cycle: {components}
class TestDependencyCycle(unittest.TestCase):
    
    def setUp(self):
        # Setup component mocks
        {mock_setup}
        
        # Component dependency cycle information
        self.dependencies = {dependencies}
    
    def test_cycle_initialization(self):
        # Test that components in the cycle can be initialized
        pass
    
    def test_cycle_operation(self):
        # Test that components in the cycle operate correctly
        pass
    
    def test_cycle_termination(self):
        # Test that components in the cycle can be terminated
        pass

if __name__ == '__main__':
    unittest.main()
"""