"""
Failure Propagation System for Starstorm

This module implements cascading failure models with dependency tracking,
allowing failures to propagate through the system based on component dependencies.
"""

import logging
from typing import Dict, Set, List, Tuple, Optional, Any
import networkx as nx
from enum import Enum, auto
import time

class FailureSeverity(Enum):
    """Enumeration of failure severity levels"""
    MINOR = auto()
    MAJOR = auto()
    CRITICAL = auto()
    CATASTROPHIC = auto()

class FailurePropagationManager:
    """
    Manages the propagation of failures through a system based on component dependencies.
    
    This class tracks dependencies between components, propagates failures through
    the dependency graph, and provides analysis of failure impacts.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        # Dependency graph: component -> set of components that depend on it
        self._dependency_graph: Dict[str, Set[str]] = {}
        # Reverse dependency graph: component -> set of components it depends on
        self._reverse_dependency_graph: Dict[str, Set[str]] = {}
        # Track active failures and their propagation paths
        self._active_failures: Dict[str, Dict[str, List[List[str]]]] = {}
        # Track failure timestamps
        self._failure_timestamps: Dict[str, Dict[str, float]] = {}
        # Track failure severities
        self._failure_severities: Dict[str, Dict[str, FailureSeverity]] = {}
        # Track components affected by cascading failures
        self._affected_components: Dict[str, Set[str]] = {}
        
    def register_dependency(self, dependent: str, dependency: str) -> None:
        """
        Register a dependency relationship between components.
        
        Args:
            dependent: The component that depends on another
            dependency: The component that is depended upon
        """
        # Initialize if not exists
        if dependency not in self._dependency_graph:
            self._dependency_graph[dependency] = set()
        if dependent not in self._reverse_dependency_graph:
            self._reverse_dependency_graph[dependent] = set()
            
        # Add to graphs
        self._dependency_graph[dependency].add(dependent)
        self._reverse_dependency_graph[dependent].add(dependency)
        
        self._logger.debug(f"Registered dependency: {dependent} depends on {dependency}")
        
    def register_dependencies_from_registry(self, dependency_graph: Dict[str, Set[str]]) -> None:
        """
        Register dependencies from a component registry's dependency graph.
        
        Args:
            dependency_graph: Dictionary mapping components to their dependencies
        """
        # The input graph is reversed from what we need (it maps components to what they depend on)
        # We need to build the reverse to get what depends on each component
        for component, dependencies in dependency_graph.items():
            for dependency in dependencies:
                self.register_dependency(component, dependency)
                
    def register_subsystem_dependency(self, dependent_subsystem: str, dependency_subsystem: str) -> None:
        """
        Register a dependency between subsystems.
        
        Args:
            dependent_subsystem: The subsystem that depends on another
            dependency_subsystem: The subsystem that is depended upon
        """
        key_dependent = f"subsystem:{dependent_subsystem}"
        key_dependency = f"subsystem:{dependency_subsystem}"
        self.register_dependency(key_dependent, key_dependency)
        
    def inject_failure(self, component: str, failure_mode: str, 
                      severity: FailureSeverity = FailureSeverity.MAJOR) -> List[str]:
        """
        Inject a failure and propagate it through the system.
        
        Args:
            component: The component where the failure originates
            failure_mode: The type of failure
            severity: The severity of the failure
            
        Returns:
            List of affected components
        """
        # Initialize tracking for this component if needed
        if component not in self._active_failures:
            self._active_failures[component] = {}
            self._failure_timestamps[component] = {}
            self._failure_severities[component] = {}
            
        # Record the failure
        self._active_failures[component][failure_mode] = [[component]]
        self._failure_timestamps[component][failure_mode] = time.time()
        self._failure_severities[component][failure_mode] = severity
        
        # Propagate the failure
        affected = self._propagate_failure(component, failure_mode, severity)
        
        # Track affected components
        if component not in self._affected_components:
            self._affected_components[component] = set()
        self._affected_components[component].update(affected)
        
        self._logger.info(f"Injected failure '{failure_mode}' in {component}, affected: {affected}")
        return affected
        
    def _propagate_failure(self, component: str, failure_mode: str, 
                          severity: FailureSeverity) -> List[str]:
        """
        Propagate a failure through the dependency graph.
        
        Args:
            component: The component where the failure originates
            failure_mode: The type of failure
            severity: The severity of the failure
            
        Returns:
            List of affected components
        """
        affected = []
        
        # Only propagate if component is in the dependency graph
        if component not in self._dependency_graph:
            return affected
            
        # Get components that depend on the failed component
        dependents = self._dependency_graph[component]
        
        # Propagate to each dependent
        for dependent in dependents:
            # Skip if already affected by this failure
            if (component in self._active_failures and 
                failure_mode in self._active_failures[component] and
                any(dependent in path for path in self._active_failures[component][failure_mode])):
                continue
                
            # Add to affected list
            affected.append(dependent)
            
            # Record propagation path
            if component in self._active_failures and failure_mode in self._active_failures[component]:
                for path in self._active_failures[component][failure_mode]:
                    new_path = path.copy()
                    new_path.append(dependent)
                    
                    # Initialize if needed
                    if dependent not in self._active_failures:
                        self._active_failures[dependent] = {}
                    if f"cascaded:{failure_mode}" not in self._active_failures[dependent]:
                        self._active_failures[dependent][f"cascaded:{failure_mode}"] = []
                        
                    # Add propagation path
                    self._active_failures[dependent][f"cascaded:{failure_mode}"].append(new_path)
            
            # Recursively propagate
            sub_affected = self._propagate_failure(dependent, f"cascaded:{failure_mode}", severity)
            affected.extend(sub_affected)
            
        return affected
        
    def clear_failure(self, component: str, failure_mode: str) -> List[str]:
        """
        Clear a failure and its cascading effects.
        
        Args:
            component: The component where the failure originated
            failure_mode: The type of failure
            
        Returns:
            List of components where failures were cleared
        """
        cleared = []
        
        # Check if this failure exists
        if (component not in self._active_failures or 
            failure_mode not in self._active_failures[component]):
            return cleared
            
        # Find all cascaded failures from this one
        cascaded_failures = []
        for comp, failures in self._active_failures.items():
            for mode, paths in failures.items():
                if mode == f"cascaded:{failure_mode}" or mode == failure_mode:
                    for path in paths:
                        if component in path:
                            cascaded_failures.append((comp, mode))
        
        # Clear the original failure
        if component in self._active_failures and failure_mode in self._active_failures[component]:
            del self._active_failures[component][failure_mode]
            if component in self._failure_timestamps and failure_mode in self._failure_timestamps[component]:
                del self._failure_timestamps[component][failure_mode]
            if component in self._failure_severities and failure_mode in self._failure_severities[component]:
                del self._failure_severities[component][failure_mode]
            cleared.append(component)
            
        # Clear all cascaded failures
        for comp, mode in cascaded_failures:
            if comp in self._active_failures and mode in self._active_failures[comp]:
                del self._active_failures[comp][mode]
                cleared.append(comp)
                
        self._logger.info(f"Cleared failure '{failure_mode}' in {component}, affected: {cleared}")
        return cleared
        
    def get_active_failures(self, component: Optional[str] = None) -> Dict[str, Dict[str, List[List[str]]]]:
        """
        Get active failures, optionally filtered by component.
        
        Args:
            component: Optional component to filter by
            
        Returns:
            Dictionary of active failures with propagation paths
        """
        if component:
            return {component: self._active_failures.get(component, {})}
        return self._active_failures.copy()
        
    def get_failure_graph(self) -> nx.DiGraph:
        """
        Get a directed graph representation of the current failure propagation.
        
        Returns:
            NetworkX DiGraph of failure propagation
        """
        G = nx.DiGraph()
        
        # Add all components as nodes
        all_components = set()
        for comp in self._dependency_graph:
            all_components.add(comp)
        for comp in self._reverse_dependency_graph:
            all_components.add(comp)
            
        for comp in all_components:
            # Add node with attributes
            failures = self._active_failures.get(comp, {})
            has_failure = len(failures) > 0
            G.add_node(comp, has_failure=has_failure, failures=list(failures.keys()))
            
        # Add dependency edges
        for comp, dependents in self._dependency_graph.items():
            for dependent in dependents:
                G.add_edge(comp, dependent)
                
        return G
        
    def analyze_critical_components(self) -> List[Tuple[str, int]]:
        """
        Analyze and identify critical components based on dependency structure.
        
        Returns:
            List of (component, impact_score) tuples sorted by impact
        """
        G = nx.DiGraph()
        
        # Build the graph
        for comp, dependents in self._dependency_graph.items():
            for dependent in dependents:
                G.add_edge(comp, dependent)
                
        # Calculate centrality measures
        centrality = nx.betweenness_centrality(G)
        out_degree = dict(G.out_degree())
        
        # Combine measures for impact score
        impact_scores = []
        for comp in G.nodes():
            score = centrality.get(comp, 0) * 10 + out_degree.get(comp, 0)
            impact_scores.append((comp, score))
            
        # Sort by impact score
        impact_scores.sort(key=lambda x: x[1], reverse=True)
        return impact_scores
        
    def get_failure_impact_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on failure impacts.
        
        Returns:
            Dictionary with failure impact analysis
        """
        report = {
            "active_failures_count": sum(len(failures) for failures in self._active_failures.values()),
            "affected_components_count": sum(len(comps) for comps in self._affected_components.values()),
            "critical_components": self.analyze_critical_components()[:5],
            "failure_propagation_paths": {},
            "component_vulnerability_scores": {},
        }
        
        # Add propagation paths for active failures
        for comp, failures in self._active_failures.items():
            for mode, paths in failures.items():
                if paths:  # Only include if there are propagation paths
                    key = f"{comp}:{mode}"
                    report["failure_propagation_paths"][key] = paths
                    
        # Calculate vulnerability scores (how many failures affect each component)
        vulnerability = {}
        for comp, failures in self._active_failures.items():
            for mode, paths in failures.items():
                for path in paths:
                    for affected in path:
                        if affected not in vulnerability:
                            vulnerability[affected] = 0
                        vulnerability[affected] += 1
                        
        report["component_vulnerability_scores"] = vulnerability
        
        return report