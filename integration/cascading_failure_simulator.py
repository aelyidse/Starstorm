"""
Cascading Failure Simulator for Starstorm

This module extends the SystemFailureSimulator with cascading failure capabilities,
allowing failures to propagate through the system based on component dependencies.
"""

import logging
from typing import Dict, Set, List, Tuple, Optional, Any
import numpy as np
from copy import deepcopy

from core.failure_propagation import FailurePropagationManager, FailureSeverity
from integration.system_simulation import SystemFailureSimulator

class CascadingFailureSimulator(SystemFailureSimulator):
    """
    Enhanced failure simulator that implements cascading failures with dependency tracking.
    
    This class extends SystemFailureSimulator to propagate failures through dependent
    subsystems based on a dependency graph.
    """
    
    def __init__(self, failure_rates: Optional[Dict[str, Dict[str, float]]] = None,
                subsystem_dependencies: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the cascading failure simulator.
        
        Args:
            failure_rates: Dictionary mapping subsystems to their failure modes and rates
            subsystem_dependencies: Dictionary mapping subsystems to their dependencies
        """
        super().__init__(failure_rates)
        self._logger = logging.getLogger(__name__)
        
        # Initialize the failure propagation manager
        self._propagation_manager = FailurePropagationManager()
        
        # Default subsystem dependencies if none provided
        self._subsystem_dependencies = subsystem_dependencies or {
            'propulsion': ['power'],
            'navigation': ['power', 'command'],
            'comms': ['power', 'command'],
            'payload': ['power', 'command'],
            'thermal': ['power'],
            'command': ['power'],
        }
        
        # Register subsystem dependencies
        self._register_subsystem_dependencies()
        
        # Track which failures have been propagated
        self._propagated_failures: Dict[str, Set[str]] = {
            subsystem: set() for subsystem in self.failure_rates.keys()
        }
        
    def _register_subsystem_dependencies(self) -> None:
        """Register all subsystem dependencies with the propagation manager."""
        for subsystem, dependencies in self._subsystem_dependencies.items():
            for dependency in dependencies:
                self._propagation_manager.register_subsystem_dependency(subsystem, dependency)
                self._logger.debug(f"Registered subsystem dependency: {subsystem} depends on {dependency}")
                
    def inject_failure(self, subsystem: str, mode: str) -> List[str]:
        """
        Inject a failure and propagate it to dependent subsystems.
        
        Args:
            subsystem: The subsystem to affect
            mode: The failure mode to inject
            
        Returns:
            List of affected subsystems
        """
        # First, inject the failure in the base simulator
        super().inject_failure(subsystem, mode)
        
        # Determine severity based on failure mode
        severity = self._get_failure_severity(subsystem, mode)
        
        # Then propagate through the dependency graph
        affected = self._propagation_manager.inject_failure(
            f"subsystem:{subsystem}", 
            f"{mode}", 
            severity
        )
        
        # Convert affected components back to subsystem names and inject cascaded failures
        cascaded_subsystems = []
        for component in affected:
            if component.startswith("subsystem:"):
                affected_subsystem = component.replace("subsystem:", "")
                cascaded_subsystems.append(affected_subsystem)
                
                # Inject a cascaded failure in this subsystem if not the original
                if affected_subsystem != subsystem:
                    cascaded_mode = f"cascaded_from_{subsystem}_{mode}"
                    super().inject_failure(affected_subsystem, cascaded_mode)
                    self._propagated_failures[affected_subsystem].add(cascaded_mode)
        
        self._logger.info(f"Injected failure '{mode}' in {subsystem}, cascaded to: {cascaded_subsystems}")
        return cascaded_subsystems
        
    def clear_failure(self, subsystem: str, mode: str) -> List[str]:
        """
        Clear a failure and its cascading effects.
        
        Args:
            subsystem: The subsystem to affect
            mode: The failure mode to clear
            
        Returns:
            List of subsystems where failures were cleared
        """
        # First clear from the base simulator
        super().clear_failure(subsystem, mode)
        
        # Then clear from the propagation manager
        cleared = self._propagation_manager.clear_failure(
            f"subsystem:{subsystem}", 
            f"{mode}"
        )
        
        # Convert cleared components back to subsystem names and clear cascaded failures
        cleared_subsystems = []
        for component in cleared:
            if component.startswith("subsystem:"):
                cleared_subsystem = component.replace("subsystem:", "")
                cleared_subsystems.append(cleared_subsystem)
                
                # Clear any cascaded failures in this subsystem
                if cleared_subsystem != subsystem:
                    cascaded_mode = f"cascaded_from_{subsystem}_{mode}"
                    if cascaded_mode in self._propagated_failures[cleared_subsystem]:
                        super().clear_failure(cleared_subsystem, cascaded_mode)
                        self._propagated_failures[cleared_subsystem].remove(cascaded_mode)
        
        self._logger.info(f"Cleared failure '{mode}' in {subsystem}, cascaded clearing: {cleared_subsystems}")
        return cleared_subsystems
        
    def _get_failure_severity(self, subsystem: str, mode: str) -> FailureSeverity:
        """
        Determine the severity of a failure mode.
        
        Args:
            subsystem: The affected subsystem
            mode: The failure mode
            
        Returns:
            FailureSeverity level
        """
        # Define critical failure modes by subsystem
        critical_failures = {
            'propulsion': ['engine_shutdown', 'fuel_leak'],
            'power': ['power_bus_failure', 'battery_failure'],
            'command': ['processor_reset', 'memory_corruption'],
            'navigation': ['star_tracker_failure', 'imu_failure'],
        }
        
        # Define catastrophic failure modes
        catastrophic_failures = {
            'propulsion': ['explosion'],
            'power': ['total_power_loss'],
            'structure': ['hull_breach', 'structural_failure'],
        }
        
        # Check if this is a catastrophic failure
        if subsystem in catastrophic_failures and mode in catastrophic_failures[subsystem]:
            return FailureSeverity.CATASTROPHIC
            
        # Check if this is a critical failure
        if subsystem in critical_failures and mode in critical_failures[subsystem]:
            return FailureSeverity.CRITICAL
            
        # Default to MAJOR for most failures
        return FailureSeverity.MAJOR
        
    def apply_failures(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all active failures to the system state, including cascaded failures.
        
        Args:
            system_state: Current system state
            
        Returns:
            Modified system state with failures applied
        """
        # First apply failures using the base implementation
        state = super().apply_failures(system_state)
        
        # Then apply any additional cascading effects not covered by the base implementation
        # For example, propagate power failures to all dependent systems
        for subsystem, failures in self.active_failures.items():
            if subsystem == 'power' and any(mode in failures for mode in ['power_bus_failure', 'battery_failure']):
                # Apply more severe effects to dependent systems
                for dependent in self._subsystem_dependencies.get('power', []):
                    if dependent in state:
                        state[dependent]['powered'] = False
                        state[dependent]['operational'] = False
        
        return state
        
    def get_failure_impact_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on failure impacts.
        
        Returns:
            Dictionary with failure impact analysis
        """
        return self._propagation_manager.get_failure_impact_report()
        
    def get_critical_components(self) -> List[Tuple[str, float]]:
        """
        Get a list of critical components sorted by impact score.
        
        Returns:
            List of (component, impact_score) tuples
        """
        return self._propagation_manager.analyze_critical_components()
        
    def get_active_propagation_paths(self) -> Dict[str, Any]:
        """
        Get the active failure propagation paths.
        
        Returns:
            Dictionary of active failures with propagation paths
        """
        return self._propagation_manager.get_active_failures()
    
    def detect_anomalies(self, system_state: Dict[str, Any], 
                         baseline_state: Optional[Dict[str, Any]] = None,
                         thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
        """
        Detect anomalies in system state by comparing against baseline or thresholds.
        
        Args:
            system_state: Current system state
            baseline_state: Optional baseline state for comparison
            thresholds: Optional dictionary of anomaly thresholds by subsystem and parameter
            
        Returns:
            Dictionary mapping subsystems to detected anomalies and their deviation scores
        """
        anomalies = {subsystem: {} for subsystem in self.failure_rates.keys()}
        
        # Use default thresholds if none provided
        if not thresholds:
            thresholds = {
                'propulsion': {'thrust_N': 0.1, 'fuel_leak_kg': 0.01, 'temperature_K': 50.0},
                'power': {'bus_voltage': 0.5, 'battery_capacity': 0.2, 'solar_efficiency': 0.15},
                'comms': {'signal_strength': 0.3, 'packet_error_rate': 0.1},
                'navigation': {'position_error_m': 10.0, 'attitude_error_rad': 0.05},
                'thermal': {'temperature_K': 15.0},
                'command': {'memory_errors': 1, 'processor_uptime_s': 10.0}
            }
        
        # Compare against baseline if provided
        if baseline_state:
            for subsystem in system_state:
                if subsystem not in baseline_state or subsystem not in thresholds:
                    continue
                    
                for param, value in system_state[subsystem].items():
                    if param not in baseline_state[subsystem] or param not in thresholds[subsystem]:
                        continue
                        
                    baseline_value = baseline_state[subsystem][param]
                    threshold = thresholds[subsystem][param]
                    
                    # Skip non-numeric values
                    if not isinstance(value, (int, float)) or not isinstance(baseline_value, (int, float)):
                        continue
                        
                    # Calculate deviation
                    if baseline_value != 0:
                        deviation = abs((value - baseline_value) / baseline_value)
                    else:
                        deviation = abs(value) if abs(value) > threshold else 0
                        
                    # Record anomaly if deviation exceeds threshold
                    if deviation > threshold:
                        anomalies[subsystem][param] = deviation
        
        # Check against absolute thresholds for critical parameters
        critical_limits = {
            'propulsion': {'fuel_leak_kg': (0, 0), 'thrust_N': (0.1, None)},
            'power': {'bus_voltage': (4.5, 5.5), 'battery_capacity': (0.1, None)},
            'thermal': {'temperature_K': (233, 323)},
        }
        
        for subsystem, limits in critical_limits.items():
            if subsystem not in system_state:
                continue
                
            for param, (min_val, max_val) in limits.items():
                if param not in system_state[subsystem]:
                    continue
                    
                value = system_state[subsystem][param]
                
                # Skip non-numeric values
                if not isinstance(value, (int, float)):
                    continue
                    
                # Check against limits
                if min_val is not None and value < min_val:
                    anomalies[subsystem][param] = (min_val - value) / min_val if min_val != 0 else abs(value)
                elif max_val is not None and value > max_val:
                    anomalies[subsystem][param] = (value - max_val) / max_val
        
        return anomalies
    
    def classify_anomalies(self, anomalies: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, str]]:
        """
        Classify detected anomalies into potential failure modes.
        
        Args:
            anomalies: Dictionary of detected anomalies by subsystem
            
        Returns:
            Dictionary mapping subsystems to potential failure modes and confidence levels
        """
        failure_signatures = {
            'propulsion': {
                'engine_shutdown': {'thrust_N': 1.0, 'engine_running': 1.0},
                'fuel_leak': {'fuel_leak_kg': 1.0, 'fuel_remaining_kg': 0.5},
                'thrust_vector_failure': {'vectoring_active': 1.0, 'gimbal_deg': 0.7},
                'throttle_stuck': {'throttle_position': 0.8}
            },
            'power': {
                'solar_panel_degradation': {'solar_efficiency': 1.0},
                'battery_failure': {'battery_capacity': 1.0},
                'power_bus_failure': {'bus_voltage': 1.0}
            },
            # Add signatures for other subsystems as needed
        }
        
        classifications = {subsystem: {} for subsystem in anomalies.keys()}
        
        for subsystem, params in anomalies.items():
            if not params or subsystem not in failure_signatures:
                continue
                
            # Calculate match score for each failure mode
            for mode, signature in failure_signatures[subsystem].items():
                score = 0.0
                total_weight = 0.0
                
                for param, weight in signature.items():
                    if param in params:
                        score += weight
                    total_weight += weight
                
                if total_weight > 0 and score > 0:
                    confidence = score / total_weight
                    if confidence >= 0.5:  # Only include if confidence is reasonable
                        classifications[subsystem][mode] = self._confidence_level(confidence)
        
        return classifications
    
    def _confidence_level(self, score: float) -> str:
        """Convert numerical confidence score to descriptive level."""
        if score >= 0.9:
            return "high"
        elif score >= 0.7:
            return "medium"
        else:
            return "low"
    
    def isolate_root_failures(self, detected_failures: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """
        Isolate root causes from a set of detected failures.
        
        Args:
            detected_failures: Dictionary mapping subsystems to detected failure modes
            
        Returns:
            List of (subsystem, failure_mode, confidence) tuples representing root causes
        """
        # Get the dependency graph
        dependency_graph = self._subsystem_dependencies
        
        # Build reverse dependency graph
        reverse_dependencies = {}
        for subsystem, dependencies in dependency_graph.items():
            for dependency in dependencies:
                if dependency not in reverse_dependencies:
                    reverse_dependencies[dependency] = []
                reverse_dependencies[dependency].append(subsystem)
        
        # Collect all subsystems with failures
        failed_subsystems = []
        for subsystem, failures in detected_failures.items():
            if failures:
                failed_subsystems.append(subsystem)
        
        # Find potential root causes (subsystems with failures that have no failed dependencies)
        root_causes = []
        for subsystem in failed_subsystems:
            has_failed_dependency = False
            
            # Check if any of this subsystem's dependencies have failures
            for dependency in dependency_graph.get(subsystem, []):
                if dependency in failed_subsystems:
                    has_failed_dependency = True
                    break
            
            # If no failed dependencies, this could be a root cause
            if not has_failed_dependency:
                # Add each failure mode from this subsystem
                for mode, confidence in detected_failures[subsystem].items():
                    # Convert string confidence to numeric
                    conf_value = 0.6 if confidence == "low" else 0.8 if confidence == "medium" else 0.95
                    root_causes.append((subsystem, mode, conf_value))
        
        # Sort by confidence (highest first)
        root_causes.sort(key=lambda x: x[2], reverse=True)
        return root_causes
    
    def generate_isolation_report(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive failure isolation report.
        
        Args:
            system_state: Current system state
            
        Returns:
            Dictionary with failure isolation analysis
        """
        # Detect anomalies
        anomalies = self.detect_anomalies(system_state)
        
        # Classify anomalies into failure modes
        classifications = self.classify_anomalies(anomalies)
        
        # Isolate root causes
        root_causes = self.isolate_root_failures(classifications)
        
        # Get propagation paths from failure manager
        propagation_paths = self._propagation_manager.get_active_failures()
        
        # Build the report
        report = {
            "detected_anomalies": anomalies,
            "classified_failures": classifications,
            "root_causes": root_causes,
            "propagation_paths": propagation_paths,
            "affected_subsystems": list(set([subsystem for subsystem, failures in classifications.items() if failures])),
            "isolation_confidence": self._calculate_isolation_confidence(root_causes, classifications)
        }
        
        return report
    
    def _calculate_isolation_confidence(self, root_causes: List[Tuple[str, str, float]], 
                                       classifications: Dict[str, Dict[str, str]]) -> float:
        """Calculate overall confidence in the isolation results."""
        if not root_causes:
            return 0.0
            
        # Count total failures detected
        total_failures = sum(len(failures) for failures in classifications.values())
        
        # If we have exactly one root cause for all failures, high confidence
        if len(root_causes) == 1 and total_failures > 1:
            return 0.9
            
        # If we have multiple root causes, lower confidence
        return min(0.7, root_causes[0][2])