from typing import Dict, Any, List, Optional, Tuple, Set, Callable, Union
import numpy as np
import random
from enum import Enum
from manufacturing.digital_twin import ComponentDigitalTwin, ComponentState
from manufacturing.digital_twin_factory import DigitalTwinFactory
from manufacturing.tolerance_simulation import ToleranceSimulator

class QualityDefectType(Enum):
    """Types of quality defects that can occur in manufacturing"""
    DIMENSIONAL = "dimensional"  # Out of tolerance dimensions
    SURFACE = "surface"          # Surface defects (scratches, dents)
    MATERIAL = "material"        # Material defects (voids, inclusions)
    ASSEMBLY = "assembly"        # Assembly errors (misalignment, missing parts)
    FUNCTIONAL = "functional"    # Functional failures
    COSMETIC = "cosmetic"        # Cosmetic issues


class InspectionMethod(Enum):
    """Methods for quality inspection"""
    VISUAL = "visual"                # Visual inspection
    DIMENSIONAL = "dimensional"      # Dimensional measurement
    FUNCTIONAL = "functional"        # Functional testing
    NDT = "non_destructive_testing"  # Non-destructive testing
    AUTOMATED = "automated"          # Automated inspection systems
    SAMPLING = "sampling"            # Statistical sampling


class QualityControlSimulator:
    """
    Simulates quality control processes in manufacturing.
    Models inspection methods, defect detection, and statistical process control.
    """
    def __init__(self, 
                 digital_twin_factory: Optional[DigitalTwinFactory] = None,
                 tolerance_simulator: Optional[ToleranceSimulator] = None):
        self.digital_twin_factory = digital_twin_factory
        self.tolerance_simulator = tolerance_simulator
        self.inspection_methods: Dict[str, Dict[str, Any]] = {}
        self.defect_models: Dict[str, Dict[str, Any]] = {}
        self.inspection_results: Dict[str, List[Dict[str, Any]]] = {}
        self.process_capability: Dict[str, Dict[str, float]] = {}
        
    def register_inspection_method(self, 
                                  name: str,
                                  method_type: InspectionMethod,
                                  detection_rates: Dict[QualityDefectType, float],
                                  false_positive_rate: float = 0.05,
                                  time_per_unit: float = 1.0,
                                  cost_per_unit: float = 1.0) -> None:
        """
        Register an inspection method
        
        Args:
            name: Unique name for this inspection method
            method_type: Type of inspection method
            detection_rates: Dictionary mapping defect types to detection probabilities (0-1)
            false_positive_rate: Rate of false positives (0-1)
            time_per_unit: Time required per unit (minutes)
            cost_per_unit: Cost per unit inspected
        """
        self.inspection_methods[name] = {
            'type': method_type,
            'detection_rates': detection_rates,
            'false_positive_rate': false_positive_rate,
            'time_per_unit': time_per_unit,
            'cost_per_unit': cost_per_unit
        }
        
    def register_defect_model(self,
                             component_type: str,
                             defect_rates: Dict[QualityDefectType, float],
                             severity_distribution: Optional[Callable] = None) -> None:
        """
        Register a defect model for a component type
        
        Args:
            component_type: Type of component
            defect_rates: Dictionary mapping defect types to occurrence rates (0-1)
            severity_distribution: Optional function to generate defect severity (0-1)
        """
        if severity_distribution is None:
            # Default severity distribution - beta distribution favoring minor defects
            severity_distribution = lambda: np.random.beta(2, 5)
            
        self.defect_models[component_type] = {
            'defect_rates': defect_rates,
            'severity_distribution': severity_distribution
        }
        
    def simulate_defects(self, 
                        component_type: str, 
                        batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Simulate manufacturing defects for a batch of components
        
        Args:
            component_type: Type of component to simulate
            batch_size: Number of components to simulate
            
        Returns:
            List of components with simulated defects
        """
        if component_type not in self.defect_models:
            raise ValueError(f"No defect model registered for component type: {component_type}")
            
        model = self.defect_models[component_type]
        defect_rates = model['defect_rates']
        severity_func = model['severity_distribution']
        
        batch = []
        for i in range(batch_size):
            component = {
                'id': f"{component_type}_{i}",
                'type': component_type,
                'defects': []
            }
            
            # Check each defect type
            for defect_type, rate in defect_rates.items():
                if random.random() < rate:
                    severity = severity_func()
                    component['defects'].append({
                        'type': defect_type,
                        'severity': severity,
                        'description': f"{defect_type.value} defect with severity {severity:.2f}"
                    })
            
            # Add property to indicate if component has any defects
            component['has_defects'] = len(component['defects']) > 0
            batch.append(component)
            
        return batch
    
    def simulate_inspection(self,
                           components: List[Dict[str, Any]],
                           inspection_method: str) -> Dict[str, Any]:
        """
        Simulate quality inspection for a batch of components
        
        Args:
            components: List of components to inspect
            inspection_method: Name of inspection method to use
            
        Returns:
            Inspection results including detected defects and metrics
        """
        if inspection_method not in self.inspection_methods:
            raise ValueError(f"Unknown inspection method: {inspection_method}")
            
        method = self.inspection_methods[inspection_method]
        detection_rates = method['detection_rates']
        false_positive_rate = method['false_positive_rate']
        
        results = {
            'method': inspection_method,
            'total_components': len(components),
            'inspection_time': method['time_per_unit'] * len(components),
            'inspection_cost': method['cost_per_unit'] * len(components),
            'components': [],
            'summary': {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'defects_by_type': {defect_type.value: 0 for defect_type in QualityDefectType}
            }
        }
        
        for component in components:
            component_result = {
                'id': component['id'],
                'type': component['type'],
                'actual_defects': component['defects'],
                'detected_defects': [],
                'false_positives': []
            }
            
            # Process actual defects - some may be missed based on detection rate
            for defect in component['defects']:
                defect_type = QualityDefectType(defect['type'].value)
                detection_rate = detection_rates.get(defect_type, 0.5)  # Default 50% if not specified
                
                # Higher severity defects are easier to detect
                adjusted_rate = min(1.0, detection_rate * (1 + defect['severity']))
                
                if random.random() < adjusted_rate:
                    # Defect detected
                    component_result['detected_defects'].append(defect)
                    results['summary']['true_positives'] += 1
                    results['summary']['defects_by_type'][defect_type.value] += 1
                else:
                    # Defect missed
                    results['summary']['false_negatives'] += 1
            
            # Check for false positives
            if random.random() < false_positive_rate:
                # Generate a false positive
                false_defect_type = random.choice(list(QualityDefectType))
                false_defect = {
                    'type': false_defect_type,
                    'severity': random.random() * 0.5,  # Lower severity for false positives
                    'description': f"False positive {false_defect_type.value} defect"
                }
                component_result['false_positives'].append(false_defect)
                results['summary']['false_positives'] += 1
            
            # Count true negatives
            if not component['defects'] and not component_result['false_positives']:
                results['summary']['true_negatives'] += 1
                
            # Add to results
            results['components'].append(component_result)
        
        # Calculate quality metrics
        total_inspected = len(components)
        total_defects = sum(len(c['defects']) for c in components)
        detected_defects = results['summary']['true_positives']
        
        if total_defects > 0:
            results['summary']['detection_rate'] = detected_defects / total_defects
        else:
            results['summary']['detection_rate'] = 1.0
            
        results['summary']['false_positive_rate'] = results['summary']['false_positives'] / total_inspected
        
        # Store results
        if inspection_method not in self.inspection_results:
            self.inspection_results[inspection_method] = []
        self.inspection_results[inspection_method].append(results)
        
        return results
    
    def calculate_process_capability(self,
                                    component_type: str,
                                    property_name: str,
                                    specification_limits: Tuple[float, float],
                                    sample_data: List[float]) -> Dict[str, float]:
        """
        Calculate process capability indices (Cp, Cpk)
        
        Args:
            component_type: Type of component
            property_name: Name of the property to analyze
            specification_limits: Tuple of (lower_limit, upper_limit)
            sample_data: List of measured values
            
        Returns:
            Dictionary with capability metrics
        """
        if not sample_data:
            return {'cp': 0, 'cpk': 0, 'sigma': 0}
            
        # Calculate statistics
        mean = np.mean(sample_data)
        sigma = np.std(sample_data)
        
        if sigma == 0:
            return {'cp': float('inf'), 'cpk': float('inf'), 'sigma': 0}
            
        lower_limit, upper_limit = specification_limits
        
        # Calculate Cp - process capability
        cp = (upper_limit - lower_limit) / (6 * sigma)
        
        # Calculate Cpk - process capability index
        cpu = (upper_limit - mean) / (3 * sigma)
        cpl = (mean - lower_limit) / (3 * sigma)
        cpk = min(cpu, cpl)
        
        # Store results
        if component_type not in self.process_capability:
            self.process_capability[component_type] = {}
            
        self.process_capability[component_type][property_name] = {
            'cp': cp,
            'cpk': cpk,
            'sigma': sigma,
            'mean': mean,
            'specification_limits': specification_limits
        }
        
        return {
            'cp': cp,
            'cpk': cpk,
            'sigma': sigma,
            'mean': mean,
            'specification_limits': specification_limits
        }
    
    def run_statistical_process_control(self,
                                       component_type: str,
                                       property_name: str,
                                       measurements: List[List[float]]) -> Dict[str, Any]:
        """
        Run Statistical Process Control (SPC) analysis on measurement data
        
        Args:
            component_type: Type of component
            property_name: Name of the property being measured
            measurements: List of sample groups (each a list of measurements)
            
        Returns:
            SPC analysis results
        """
        if not measurements or not measurements[0]:
            return {'status': 'error', 'message': 'No measurement data provided'}
            
        # Calculate control limits
        subgroup_means = [np.mean(group) for group in measurements]
        subgroup_ranges = [max(group) - min(group) for group in measurements]
        
        overall_mean = np.mean(subgroup_means)
        mean_range = np.mean(subgroup_ranges)
        
        # Constants for control limit calculations (for sample size n)
        n = len(measurements[0])
        if n <= 1:
            return {'status': 'error', 'message': 'Subgroup size must be greater than 1'}
            
        # A2, D3, D4 are constants that depend on subgroup size
        # These are approximations - in practice, use a lookup table
        a2 = 1.880 if n == 2 else 1.023 if n == 3 else 0.729 if n == 4 else 0.577 if n == 5 else 0.483
        d3 = 0 if n <= 5 else 0.076 if n == 6 else 0.136 if n == 7 else 0.184
        d4 = 3.267 if n == 2 else 2.575 if n == 3 else 2.282 if n == 4 else 2.115 if n == 5 else 2.004
        
        # Calculate control limits
        ucl_x = overall_mean + a2 * mean_range
        lcl_x = overall_mean - a2 * mean_range
        ucl_r = d4 * mean_range
        lcl_r = d3 * mean_range
        
        # Check for out of control points
        x_out_of_control = [i for i, mean in enumerate(subgroup_means) 
                           if mean > ucl_x or mean < lcl_x]
        r_out_of_control = [i for i, range_val in enumerate(subgroup_ranges) 
                           if range_val > ucl_r or range_val < lcl_r]
        
        # Check for trends (7 consecutive points on same side of mean)
        x_trends = []
        current_trend = {'direction': None, 'start': 0, 'length': 0}
        
        for i, mean in enumerate(subgroup_means):
            if mean > overall_mean:
                direction = 'above'
            elif mean < overall_mean:
                direction = 'below'
            else:
                direction = current_trend['direction']  # Continue current trend
                
            if direction == current_trend['direction']:
                current_trend['length'] += 1
            else:
                # Check if previous trend was significant
                if current_trend['length'] >= 7:
                    x_trends.append({
                        'start': current_trend['start'],
                        'end': i - 1,
                        'direction': current_trend['direction'],
                        'length': current_trend['length']
                    })
                
                # Start new trend
                current_trend = {'direction': direction, 'start': i, 'length': 1}
        
        # Check if final trend is significant
        if current_trend['length'] >= 7:
            x_trends.append({
                'start': current_trend['start'],
                'end': len(subgroup_means) - 1,
                'direction': current_trend['direction'],
                'length': current_trend['length']
            })
        
        return {
            'component_type': component_type,
            'property_name': property_name,
            'overall_mean': overall_mean,
            'mean_range': mean_range,
            'control_limits': {
                'x_chart': {'ucl': ucl_x, 'lcl': lcl_x, 'center': overall_mean},
                'r_chart': {'ucl': ucl_r, 'lcl': lcl_r, 'center': mean_range}
            },
            'out_of_control': {
                'x_chart': x_out_of_control,
                'r_chart': r_out_of_control
            },
            'trends': x_trends,
            'process_status': 'out_of_control' if (x_out_of_control or r_out_of_control or x_trends) else 'in_control'
        }