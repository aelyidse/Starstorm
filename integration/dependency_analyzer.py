from typing import Dict, Any, List, Optional, Set, Tuple, Type, Union
import networkx as nx
import matplotlib.pyplot as plt
import os
import importlib
import inspect
import sys
from pathlib import Path

class DependencyAnalyzer:
    """
    Analyzes system-wide dependencies between components.
    Identifies dependency chains, cycles, and potential integration issues.
    """
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.component_registry: Dict[str, Any] = {}
    
    def register_component(self, name: str, component: Any, dependencies: List[str] = None) -> None:
        """
        Register a component and its dependencies.
        
        Args:
            name: Component name
            component: Component instance or class
            dependencies: List of dependency component names
        """
        self.component_registry[name] = component
        self.dependency_graph.add_node(name)
        
        if dependencies:
            for dep in dependencies:
                self.dependency_graph.add_edge(name, dep)
    
    def discover_dependencies(self, module_path: str, base_class: Optional[Type] = None) -> None:
        """
        Discover components and their dependencies in a module.
        
        Args:
            module_path: Dot-separated path to module
            base_class: Optional base class to filter components
        """
        try:
            module = importlib.import_module(module_path)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and (base_class is None or issubclass(obj, base_class)) and obj.__module__ == module_path:
                    # Look for dependencies in __init__ method
                    dependencies = []
                    if hasattr(obj, '__init__'):
                        init_sig = inspect.signature(obj.__init__)
                        for param_name, param in init_sig.parameters.items():
                            if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                                dependencies.append(str(param.annotation))
                    
                    self.register_component(name, obj, dependencies)
        except ImportError as e:
            print(f"Error importing module {module_path}: {e}")
    
    def find_cycles(self) -> List[List[str]]:
        """Find dependency cycles in the system."""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return cycles
        except nx.NetworkXNoCycle:
            return []
    
    def get_dependency_chain(self, start_component: str, end_component: str) -> Optional[List[str]]:
        """
        Find the dependency chain between two components.
        
        Args:
            start_component: Starting component name
            end_component: Ending component name
            
        Returns:
            List of components in the dependency chain or None if no path exists
        """
        if start_component not in self.dependency_graph or end_component not in self.dependency_graph:
            return None
        
        try:
            path = nx.shortest_path(self.dependency_graph, start_component, end_component)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_all_dependencies(self, component_name: str) -> Set[str]:
        """Get all dependencies (direct and indirect) for a component."""
        if component_name not in self.dependency_graph:
            return set()
        
        dependencies = set()
        for node in self.dependency_graph:
            if node == component_name:
                continue
            
            try:
                if nx.has_path(self.dependency_graph, component_name, node):
                    dependencies.add(node)
            except:
                pass
        
        return dependencies
    
    def get_dependents(self, component_name: str) -> Set[str]:
        """Get all components that depend on the specified component."""
        if component_name not in self.dependency_graph:
            return set()
        
        dependents = set()
        for node in self.dependency_graph:
            if node == component_name:
                continue
            
            try:
                if nx.has_path(self.dependency_graph, node, component_name):
                    dependents.add(node)
            except:
                pass
        
        return dependents
    
    def visualize_dependencies(self, output_file: str = "dependency_graph.png") -> None:
        """
        Visualize the dependency graph.
        
        Args:
            output_file: Path to save the visualization
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.dependency_graph)
        nx.draw(self.dependency_graph, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, edge_color='gray', arrows=True, font_size=10)
        plt.title("System Component Dependencies")
        plt.savefig(output_file)
        plt.close()
    
    def generate_dependency_report(self) -> Dict[str, Any]:
        """Generate a comprehensive dependency report."""
        report = {
            'components': list(self.component_registry.keys()),
            'total_components': len(self.component_registry),
            'dependency_count': self.dependency_graph.number_of_edges(),
            'cycles': self.find_cycles(),
            'has_cycles': len(self.find_cycles()) > 0,
            'component_details': {}
        }
        
        for component in self.component_registry:
            dependencies = list(self.dependency_graph.successors(component))
            dependents = list(self.dependency_graph.predecessors(component))
            
            report['component_details'][component] = {
                'direct_dependencies': dependencies,
                'dependency_count': len(dependencies),
                'dependents': dependents,
                'dependent_count': len(dependents),
                'is_leaf': len(dependencies) == 0,
                'is_root': len(dependents) == 0
            }
        
        return report