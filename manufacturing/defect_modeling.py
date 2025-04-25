from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
import random
from enum import Enum
from manufacturing.quality_control_simulation import QualityDefectType

class DefectSeverity(Enum):
    """Severity levels for manufacturing defects"""
    MINOR = "minor"          # Cosmetic issues, no functional impact
    MODERATE = "moderate"    # Minor functional impact, within acceptable limits
    MAJOR = "major"          # Significant functional impact, may require rework
    CRITICAL = "critical"    # Safety or mission-critical impact, requires rejection

class DefectModel:
    """
    Models manufacturing defects and their occurrence probabilities.
    Provides statistical generation of defects based on process parameters.
    """
    def __init__(self, process_name: str, base_defect_rate: float = 0.05):
        self.process_name = process_name
        self.base_defect_rate = base_defect_rate
        self.defect_types: Dict[QualityDefectType, float] = {}
        self.severity_distributions: Dict[QualityDefectType, Callable[[], DefectSeverity]] = {}
        self.process_factors: Dict[str, float] = {}
        
    def add_defect_type(self, 
                       defect_type: QualityDefectType, 
                       relative_frequency: float,
                       severity_distribution: Optional[Dict[DefectSeverity, float]] = None):
        """
        Add a defect type with its relative frequency and severity distribution
        
        Args:
            defect_type: Type of defect
            relative_frequency: Relative frequency compared to base rate
            severity_distribution: Optional distribution of severity levels
        """
        self.defect_types[defect_type] = relative_frequency
        
        # Default severity distribution if not provided
        if severity_distribution is None:
            severity_distribution = {
                DefectSeverity.MINOR: 0.6,
                DefectSeverity.MODERATE: 0.3,
                DefectSeverity.MAJOR: 0.08,
                DefectSeverity.CRITICAL: 0.02
            }
            
        # Create a function that returns a severity based on the distribution
        def severity_generator():
            r = random.random()
            cumulative = 0
            for severity, prob in severity_distribution.items():
                cumulative += prob
                if r <= cumulative:
                    return severity
            return DefectSeverity.MINOR  # Default fallback
            
        self.severity_distributions[defect_type] = severity_generator
        
    def add_process_factor(self, factor_name: str, impact_multiplier: float):
        """
        Add a process factor that affects defect rates
        
        Args:
            factor_name: Name of the process factor
            impact_multiplier: Multiplier for defect rate (>1 increases, <1 decreases)
        """
        self.process_factors[factor_name] = impact_multiplier
        
    def generate_defects(self, 
                        component_id: str,
                        process_conditions: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Generate defects for a component based on process conditions
        
        Args:
            component_id: Identifier for the component
            process_conditions: Optional dictionary of process condition values
            
        Returns:
            List of defects generated for this component
        """
        defects = []
        process_conditions = process_conditions or {}
        
        # Calculate adjusted defect rate based on process conditions
        adjusted_rate = self.base_defect_rate
        for factor, value in process_conditions.items():
            if factor in self.process_factors:
                # Apply the impact of this factor
                factor_impact = self.process_factors[factor]
                # Higher value means more impact
                adjusted_rate *= (1 + (value - 1) * factor_impact)
        
        # Generate defects for each defect type
        for defect_type, relative_freq in self.defect_types.items():
            defect_probability = adjusted_rate * relative_freq
            
            if random.random() < defect_probability:
                # Generate a defect of this type
                severity = self.severity_distributions[defect_type]()
                
                defects.append({
                    'component_id': component_id,
                    'defect_type': defect_type,
                    'severity': severity,
                    'timestamp': np.datetime64('now'),
                    'process_conditions': process_conditions.copy()
                })
                
        return defects


class DefectPropagationModel:
    """
    Models how defects propagate through the manufacturing process.
    Tracks defect evolution and compound effects across process steps.
    """
    def __init__(self):
        self.propagation_rules: List[Dict[str, Any]] = []
        
    def add_propagation_rule(self, 
                            source_defect: QualityDefectType,
                            target_defect: QualityDefectType,
                            probability: float,
                            severity_modifier: float = 1.0):
        """
        Add a rule for defect propagation
        
        Args:
            source_defect: Original defect type
            target_defect: Resulting defect type
            probability: Probability of propagation
            severity_modifier: Multiplier for severity (>1 worsens, <1 improves)
        """
        self.propagation_rules.append({
            'source': source_defect,
            'target': target_defect,
            'probability': probability,
            'severity_modifier': severity_modifier
        })
        
    def propagate_defects(self, 
                         input_defects: List[Dict[str, Any]],
                         process_step: str) -> List[Dict[str, Any]]:
        """
        Propagate defects through a process step
        
        Args:
            input_defects: List of existing defects
            process_step: Current manufacturing process step
            
        Returns:
            Updated list of defects after propagation
        """
        output_defects = input_defects.copy()
        new_defects = []
        
        for defect in input_defects:
            source_type = defect['defect_type']
            source_severity = defect['severity']
            
            # Find applicable propagation rules
            for rule in self.propagation_rules:
                if rule['source'] == source_type and random.random() < rule['probability']:
                    # Create a new propagated defect
                    severity_value = self._get_severity_value(source_severity)
                    modified_severity = severity_value * rule['severity_modifier']
                    new_severity = self._get_severity_from_value(modified_severity)
                    
                    new_defect = defect.copy()
                    new_defect['defect_type'] = rule['target']
                    new_defect['severity'] = new_severity
                    new_defect['parent_defect'] = defect['component_id']
                    new_defect['process_step'] = process_step
                    
                    new_defects.append(new_defect)
        
        # Add new defects to the output
        output_defects.extend(new_defects)
        return output_defects
    
    def _get_severity_value(self, severity: DefectSeverity) -> float:
        """Convert severity enum to numerical value"""
        severity_values = {
            DefectSeverity.MINOR: 0.25,
            DefectSeverity.MODERATE: 0.5,
            DefectSeverity.MAJOR: 0.75,
            DefectSeverity.CRITICAL: 1.0
        }
        return severity_values.get(severity, 0.25)
    
    def _get_severity_from_value(self, value: float) -> DefectSeverity:
        """Convert numerical value to severity enum"""
        if value < 0.3:
            return DefectSeverity.MINOR
        elif value < 0.6:
            return DefectSeverity.MODERATE
        elif value < 0.85:
            return DefectSeverity.MAJOR
        else:
            return DefectSeverity.CRITICAL


class ManufacturingDefectSimulator:
    """
    Simulates defects across the entire manufacturing process.
    Integrates defect models, propagation, and process parameters.
    """
    def __init__(self):
        self.process_steps: List[str] = []
        self.defect_models: Dict[str, DefectModel] = {}
        self.propagation_model = DefectPropagationModel()
        self.process_parameters: Dict[str, Dict[str, Any]] = {}
        
    def add_process_step(self, step_name: str, defect_model: DefectModel):
        """Add a process step with its defect model"""
        self.process_steps.append(step_name)
        self.defect_models[step_name] = defect_model
        
    def set_process_parameters(self, step_name: str, parameters: Dict[str, Any]):
        """Set process parameters for a step"""
        self.process_parameters[step_name] = parameters
        
    def simulate_component_manufacturing(self, component_id: str) -> Dict[str, Any]:
        """
        Simulate manufacturing of a single component through all process steps
        
        Args:
            component_id: Identifier for the component
            
        Returns:
            Simulation results including all defects
        """
        all_defects = []
        step_defects = {}
        
        for step in self.process_steps:
            # Get process parameters for this step
            params = self.process_parameters.get(step, {})
            
            # Generate new defects for this step
            new_defects = self.defect_models[step].generate_defects(component_id, params)
            
            # Propagate existing defects
            if all_defects:
                all_defects = self.propagation_model.propagate_defects(all_defects, step)
            
            # Add new defects from this step
            all_defects.extend(new_defects)
            
            # Record defects by step
            step_defects[step] = new_defects
            
        # Determine overall component quality
        quality_score = self._calculate_quality_score(all_defects)
        
        return {
            'component_id': component_id,
            'defects': all_defects,
            'defects_by_step': step_defects,
            'quality_score': quality_score,
            'pass': quality_score >= 0.7  # Example threshold
        }
    
    def simulate_batch(self, batch_size: int, prefix: str = "COMP") -> List[Dict[str, Any]]:
        """
        Simulate manufacturing of a batch of components
        
        Args:
            batch_size: Number of components to simulate
            prefix: Prefix for component IDs
            
        Returns:
            List of simulation results for each component
        """
        results = []
        for i in range(batch_size):
            component_id = f"{prefix}_{i+1:04d}"
            result = self.simulate_component_manufacturing(component_id)
            results.append(result)
        
        return results
    
    def _calculate_quality_score(self, defects: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score based on defects"""
        if not defects:
            return 1.0  # Perfect score for no defects
            
        # Calculate penalty for each defect based on severity
        severity_penalties = {
            DefectSeverity.MINOR: 0.05,
            DefectSeverity.MODERATE: 0.15,
            DefectSeverity.MAJOR: 0.3,
            DefectSeverity.CRITICAL: 0.6
        }
        
        # Start with perfect score and subtract penalties
        score = 1.0
        for defect in defects:
            severity = defect['severity']
            penalty = severity_penalties.get(severity, 0.05)
            score -= penalty
            
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))


class DefectAnalyzer:
    """
    Analyzes defect patterns and provides statistical insights.
    Identifies common defect patterns and correlations with process parameters.
    """
    def __init__(self):
        pass
        
    def analyze_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze results from a batch simulation
        
        Args:
            batch_results: Results from batch simulation
            
        Returns:
            Analysis results
        """
        total_components = len(batch_results)
        passing_components = sum(1 for r in batch_results if r['pass'])
        
        # Count defects by type and severity
        defect_counts = {}
        severity_counts = {s.value: 0 for s in DefectSeverity}
        
        for result in batch_results:
            for defect in result['defects']:
                defect_type = defect['defect_type'].value
                severity = defect['severity'].value
                
                if defect_type not in defect_counts:
                    defect_counts[defect_type] = 0
                defect_counts[defect_type] += 1
                severity_counts[severity] += 1
        
        # Calculate defect rates
        defect_rates = {k: v / total_components for k, v in defect_counts.items()}
        
        # Calculate average quality score
        avg_quality = sum(r['quality_score'] for r in batch_results) / total_components
        
        return {
            'total_components': total_components,
            'passing_components': passing_components,
            'yield_rate': passing_components / total_components,
            'defect_counts': defect_counts,
            'defect_rates': defect_rates,
            'severity_counts': severity_counts,
            'average_quality_score': avg_quality
        }
        
    def identify_critical_process_parameters(self, 
                                           batch_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Identify process parameters with strongest correlation to defects
        
        Args:
            batch_results: Results from batch simulation
            
        Returns:
            Dictionary of parameters and their correlation coefficients
        """
        # This would implement correlation analysis between process parameters and defect rates
        # Simplified implementation for demonstration
        correlations = {}
        
        # Extract all process parameters across all steps
        all_params = set()
        for result in batch_results:
            for defect in result['defects']:
                if 'process_conditions' in defect:
                    all_params.update(defect['process_conditions'].keys())
        
        # Calculate correlation for each parameter
        for param in all_params:
            # This is a placeholder for actual correlation calculation
            correlations[param] = random.uniform(-1, 1)
            
        return correlations