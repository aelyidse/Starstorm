from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import random
from enum import Enum
from manufacturing.digital_twin import ComponentDigitalTwin, ComponentState
from manufacturing.digital_twin_factory import DigitalTwinFactory

class ToleranceDistribution(Enum):
    """Types of statistical distributions for manufacturing tolerances"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    TRIANGULAR = "triangular"
    CUSTOM = "custom"

class WearModel:
    """
    Models component wear and degradation over time.
    Supports various wear mechanisms and lifetime prediction.
    """
    def __init__(self, initial_properties: Dict[str, Any]):
        self.initial_properties = initial_properties.copy()
        self.current_properties = initial_properties.copy()
        self.operating_hours = 0
        self.wear_rates: Dict[str, float] = {}
        self.wear_models: Dict[str, Callable[[float, Dict[str, Any]], float]] = {}
        self.failure_thresholds: Dict[str, float] = {}
        
    def register_wear_rate(self, property_name: str, rate_per_hour: float):
        """Register linear wear rate for a property"""
        self.wear_rates[property_name] = rate_per_hour
        
    def register_wear_model(self, property_name: str, 
                           model_func: Callable[[float, Dict[str, Any]], float],
                           failure_threshold: Optional[float] = None):
        """
        Register custom wear model function for a property
        
        Args:
            property_name: Property that experiences wear
            model_func: Function(hours, properties) that returns new property value
            failure_threshold: Optional value at which component is considered failed
        """
        self.wear_models[property_name] = model_func
        if failure_threshold is not None:
            self.failure_thresholds[property_name] = failure_threshold
            
    def simulate_wear(self, hours: float) -> Dict[str, Any]:
        """
        Simulate wear over a specified number of operating hours
        
        Args:
            hours: Number of operating hours to simulate
            
        Returns:
            Updated component properties after wear
        """
        self.operating_hours += hours
        
        # Apply linear wear rates
        for prop, rate in self.wear_rates.items():
            if prop in self.current_properties:
                # For simplicity, assume wear reduces property value
                self.current_properties[prop] -= rate * hours
                
        # Apply custom wear models
        for prop, model in self.wear_models.items():
            if prop in self.current_properties:
                self.current_properties[prop] = model(self.operating_hours, self.current_properties)
                
        return self.current_properties
    
    def predict_remaining_life(self) -> Dict[str, float]:
        """
        Predict remaining useful life based on wear rates and failure thresholds
        
        Returns:
            Dictionary of property names and their predicted remaining hours
        """
        remaining_life = {}
        
        # Check linear wear properties
        for prop, rate in self.wear_rates.items():
            if prop in self.failure_thresholds and prop in self.current_properties:
                if rate <= 0:
                    remaining_life[prop] = float('inf')  # No wear
                else:
                    current = self.current_properties[prop]
                    threshold = self.failure_thresholds[prop]
                    # Calculate remaining hours until threshold is reached
                    remaining_hours = (current - threshold) / rate
                    remaining_life[prop] = max(0, remaining_hours)
        
        # Overall remaining life is the minimum of all properties
        if remaining_life:
            remaining_life['overall'] = min(remaining_life.values())
        else:
            remaining_life['overall'] = float('inf')
            
        return remaining_life
    
    def reset(self):
        """Reset to initial state"""
        self.current_properties = self.initial_properties.copy()
        self.operating_hours = 0


class ToleranceSimulator:
    """
    Simulates manufacturing tolerances and their effects on component performance.
    Allows for statistical analysis of manufacturing variations.
    """
    def __init__(self, digital_twin_factory: Optional[DigitalTwinFactory] = None):
        self.digital_twin_factory = digital_twin_factory
        self.tolerance_specs: Dict[str, Dict[str, Any]] = {}
        self.simulation_results: Dict[str, List[Dict[str, Any]]] = {}
        
    def register_tolerance(self, 
                          component_type: str, 
                          property_name: str, 
                          nominal_value: Union[float, List[float]],
                          distribution: ToleranceDistribution = ToleranceDistribution.NORMAL,
                          parameters: Dict[str, Any] = None,
                          custom_generator: Optional[Callable] = None) -> None:
        """
        Register a manufacturing tolerance specification
        
        Args:
            component_type: Type of component this tolerance applies to
            property_name: Name of the property with tolerance
            nominal_value: Nominal (ideal) value of the property
            distribution: Statistical distribution to use
            parameters: Parameters for the distribution (e.g. std_dev for NORMAL)
            custom_generator: Custom function for CUSTOM distribution
        """
        if component_type not in self.tolerance_specs:
            self.tolerance_specs[component_type] = {}
            
        self.tolerance_specs[component_type][property_name] = {
            'nominal': nominal_value,
            'distribution': distribution,
            'parameters': parameters or {},
            'custom_generator': custom_generator
        }
    
    def generate_value(self, spec: Dict[str, Any]) -> Union[float, List[float]]:
        """Generate a value based on tolerance specification"""
        nominal = spec['nominal']
        distribution = spec['distribution']
        params = spec['parameters']
        
        # Handle array/vector values
        if isinstance(nominal, list):
            return [self._generate_single_value(n, distribution, params) for n in nominal]
        else:
            return self._generate_single_value(nominal, distribution, params)
    
    def _generate_single_value(self, nominal: float, distribution: ToleranceDistribution, 
                              params: Dict[str, Any]) -> float:
        """Generate a single value based on distribution"""
        if distribution == ToleranceDistribution.UNIFORM:
            # Default to ±1% if not specified
            tolerance = params.get('tolerance', 0.01)
            min_val = nominal * (1 - tolerance)
            max_val = nominal * (1 + tolerance)
            return random.uniform(min_val, max_val)
            
        elif distribution == ToleranceDistribution.NORMAL:
            # Default to 0.33% std dev if not specified (3-sigma = 1%)
            std_dev = params.get('std_dev', nominal * 0.0033)
            return random.normalvariate(nominal, std_dev)
            
        elif distribution == ToleranceDistribution.TRIANGULAR:
            # Default to ±1% if not specified
            tolerance = params.get('tolerance', 0.01)
            min_val = nominal * (1 - tolerance)
            max_val = nominal * (1 + tolerance)
            return random.triangular(min_val, nominal, max_val)
            
        elif distribution == ToleranceDistribution.CUSTOM:
            if 'custom_generator' in params and callable(params['custom_generator']):
                return params['custom_generator'](nominal)
            else:
                # Fallback to normal if custom generator not provided
                return random.normalvariate(nominal, nominal * 0.0033)
                
        # Default fallback
        return nominal
    
    def simulate_batch(self, 
                      component_type: str, 
                      batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Simulate a batch of components with manufacturing variations
        
        Args:
            component_type: Type of component to simulate
            batch_size: Number of components to simulate
            
        Returns:
            List of simulated component properties
        """
        if component_type not in self.tolerance_specs:
            raise ValueError(f"No tolerance specifications registered for {component_type}")
            
        batch_results = []
        for i in range(batch_size):
            component = {'id': f"{component_type}_{i+1}"}
            
            # Generate values for each property with tolerance
            for prop_name, spec in self.tolerance_specs[component_type].items():
                component[prop_name] = self.generate_value(spec)
                
            batch_results.append(component)
            
        # Store results
        self.simulation_results[component_type] = batch_results
        return batch_results
    
    def create_digital_twins_from_batch(self, 
                                       component_type: str, 
                                       batch_results: Optional[List[Dict[str, Any]]] = None) -> List[ComponentDigitalTwin]:
        """
        Create digital twins from batch simulation results
        
        Args:
            component_type: Type of component
            batch_results: Optional batch results (uses last simulation if None)
            
        Returns:
            List of created digital twins
        """
        if not self.digital_twin_factory:
            raise ValueError("Digital twin factory not provided")
            
        if batch_results is None:
            if component_type not in self.simulation_results:
                raise ValueError(f"No simulation results for {component_type}")
            batch_results = self.simulation_results[component_type]
            
        twins = []
        for component in batch_results:
            twin_name = component['id']
            properties = {k: v for k, v in component.items() if k != 'id'}
            
            twin = self.digital_twin_factory.create_twin(
                name=twin_name,
                component_type=component_type,
                properties=properties
            )
            twins.append(twin)
            
        return twins
    
    def analyze_batch(self, 
                     component_type: str, 
                     batch_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze batch simulation results
        
        Args:
            component_type: Type of component
            batch_results: Optional batch results (uses last simulation if None)
            
        Returns:
            Statistical analysis of the batch
        """
        if batch_results is None:
            if component_type not in self.simulation_results:
                raise ValueError(f"No simulation results for {component_type}")
            batch_results = self.simulation_results[component_type]
            
        analysis = {'component_type': component_type, 'batch_size': len(batch_results)}
        
        # Get all property names (excluding id)
        properties = set()
        for component in batch_results:
            properties.update(k for k in component.keys() if k != 'id')
            
        # Calculate statistics for each property
        for prop in properties:
            # Extract values for this property
            values = []
            for component in batch_results:
                if prop in component:
                    val = component[prop]
                    if isinstance(val, list):
                        # For vector properties, analyze each dimension
                        for i, dim_val in enumerate(val):
                            if f"{prop}_dim{i}" not in analysis:
                                analysis[f"{prop}_dim{i}"] = []
                            analysis[f"{prop}_dim{i}"].append(dim_val)
                    else:
                        values.append(val)
            
            if values:
                # Calculate statistics
                analysis[f"{prop}_mean"] = np.mean(values)
                analysis[f"{prop}_std"] = np.std(values)
                analysis[f"{prop}_min"] = np.min(values)
                analysis[f"{prop}_max"] = np.max(values)
                
                # Calculate nominal value deviation
                if component_type in self.tolerance_specs and prop in self.tolerance_specs[component_type]:
                    nominal = self.tolerance_specs[component_type][prop]['nominal']
                    if not isinstance(nominal, list):
                        analysis[f"{prop}_nominal"] = nominal
                        analysis[f"{prop}_deviation"] = analysis[f"{prop}_mean"] - nominal
                        analysis[f"{prop}_percent_deviation"] = (analysis[f"{prop}_deviation"] / nominal) * 100
        
        return analysis
    
    def predict_performance(self, 
                           component_type: str, 
                           performance_model: Callable[[Dict[str, Any]], float],
                           batch_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Predict performance metrics based on manufacturing variations
        
        Args:
            component_type: Type of component
            performance_model: Function that calculates performance from component properties
            batch_results: Optional batch results (uses last simulation if None)
            
        Returns:
            Performance prediction statistics
        """
        if batch_results is None:
            if component_type not in self.simulation_results:
                raise ValueError(f"No simulation results for {component_type}")
            batch_results = self.simulation_results[component_type]
            
        # Calculate performance for each component in batch
        performances = []
        for component in batch_results:
            perf = performance_model(component)
            performances.append(perf)
            
        # Calculate performance statistics
        perf_stats = {
            'mean': np.mean(performances),
            'std_dev': np.std(performances),
            'min': np.min(performances),
            'max': np.max(performances),
            'range': np.max(performances) - np.min(performances),
            'yield': sum(1 for p in performances if self._meets_performance_threshold(p)) / len(performances)
        }
        
        return perf_stats
    
    def _meets_performance_threshold(self, performance: float, threshold: float = 0.9) -> bool:
        """Check if performance meets minimum threshold (default 90%)"""
        return performance >= threshold
    
    def sensitivity_analysis(self,
                            component_type: str,
                            performance_model: Callable[[Dict[str, Any]], float],
                            property_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Perform sensitivity analysis to determine which properties most affect performance
        
        Args:
            component_type: Type of component
            performance_model: Function that calculates performance from component properties
            property_ranges: Dictionary of property names to (min, max) ranges to test
            
        Returns:
            Dictionary of properties and their sensitivity coefficients
        """
        if component_type not in self.tolerance_specs:
            raise ValueError(f"No tolerance specifications for {component_type}")
            
        # Create baseline component with nominal values
        baseline = {'id': f"{component_type}_baseline"}
        for prop_name, spec in self.tolerance_specs[component_type].items():
            baseline[prop_name] = spec['nominal']
            
        # Calculate baseline performance
        baseline_perf = performance_model(baseline)
        
        # Calculate sensitivity for each property
        sensitivities = {}
        for prop_name, (min_val, max_val) in property_ranges.items():
            if prop_name not in self.tolerance_specs[component_type]:
                continue
                
            # Create test component with property at min value
            test_min = baseline.copy()
            test_min[prop_name] = min_val
            perf_min = performance_model(test_min)
            
            # Create test component with property at max value
            test_max = baseline.copy()
            test_max[prop_name] = max_val
            perf_max = performance_model(test_max)
            
            # Calculate sensitivity coefficient
            prop_range = max_val - min_val
            perf_range = perf_max - perf_min
            
            if prop_range == 0:
                sensitivity = 0
            else:
                # Normalized sensitivity coefficient
                sensitivity = (perf_range / baseline_perf) / (prop_range / baseline[prop_name])
                
            sensitivities[prop_name] = sensitivity
            
        return sensitivities
    
    def monte_carlo_performance_analysis(self,
                                        component_type: str,
                                        performance_model: Callable[[Dict[str, Any]], float],
                                        num_iterations: int = 1000) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis to predict performance distribution
        
        Args:
            component_type: Type of component
            performance_model: Function that calculates performance from component properties
            num_iterations: Number of Monte Carlo iterations
            
        Returns:
            Performance distribution statistics and confidence intervals
        """
        # Generate a large batch of components
        batch = self.simulate_batch(component_type, num_iterations)
        
        # Calculate performance for each component
        performances = [performance_model(component) for component in batch]
        
        # Calculate statistics
        mean = np.mean(performances)
        std_dev = np.std(performances)
        
        # Calculate confidence intervals
        sorted_perfs = sorted(performances)
        ci_90 = (sorted_perfs[int(0.05 * num_iterations)], sorted_perfs[int(0.95 * num_iterations)])
        ci_95 = (sorted_perfs[int(0.025 * num_iterations)], sorted_perfs[int(0.975 * num_iterations)])
        ci_99 = (sorted_perfs[int(0.005 * num_iterations)], sorted_perfs[int(0.995 * num_iterations)])
        
        # Calculate yield at different thresholds
        yield_90 = sum(1 for p in performances if p >= 0.9) / num_iterations
        yield_95 = sum(1 for p in performances if p >= 0.95) / num_iterations
        yield_99 = sum(1 for p in performances if p >= 0.99) / num_iterations
        
        return {
            'mean': mean,
            'std_dev': std_dev,
            'min': min(performances),
            'max': max(performances),
            'ci_90': ci_90,
            'ci_95': ci_95,
            'ci_99': ci_99,
            'yield_90': yield_90,
            'yield_95': yield_95,
            'yield_99': yield_99,
            'histogram': np.histogram(performances, bins=20)
        }


    def create_wear_model(self, component: Dict[str, Any]) -> WearModel:
        """
        Create a wear model for a component
        
        Args:
            component: Component properties
            
        Returns:
            Configured WearModel instance
        """
        return WearModel(component)
    
    def batch_lifetime_analysis(self, 
                               component_type: str,
                               wear_config: Dict[str, Any],
                               batch_results: Optional[List[Dict[str, Any]]] = None,
                               operating_hours: float = 10000) -> Dict[str, Any]:
        """
        Perform lifetime analysis on a batch of components
        
        Args:
            component_type: Type of component
            wear_config: Configuration for wear models
            batch_results: Optional batch results (uses last simulation if None)
            operating_hours: Maximum operating hours to simulate
            
        Returns:
            Statistical analysis of component lifetimes
        """
        if batch_results is None:
            if component_type not in self.simulation_results:
                raise ValueError(f"No simulation results for {component_type}")
            batch_results = self.simulation_results[component_type]
        
        # Configure wear models for each component
        wear_models = []
        for component in batch_results:
            model = self.create_wear_model(component)
            
            # Apply wear configuration
            for prop, config in wear_config.items():
                if 'rate' in config:
                    model.register_wear_rate(prop, config['rate'])
                if 'threshold' in config:
                    model.failure_thresholds[prop] = config['threshold']
                if 'model' in config and callable(config['model']):
                    model.register_wear_model(prop, config['model'], 
                                            config.get('threshold'))
            
            wear_models.append(model)
        
        # Simulate wear and collect lifetime data
        lifetimes = []
        for model in wear_models:
            # Simulate in increments to find failure point
            hours_step = operating_hours / 100
            total_hours = 0
            failed = False
            
            while total_hours < operating_hours and not failed:
                model.simulate_wear(hours_step)
                total_hours += hours_step
                
                # Check if any property has reached failure threshold
                for prop, threshold in model.failure_thresholds.items():
                    if prop in model.current_properties:
                        if model.current_properties[prop] <= threshold:
                            failed = True
                            break
            
            lifetimes.append(total_hours if failed else operating_hours)
        
        # Calculate lifetime statistics
        lifetime_stats = {
            'mean': np.mean(lifetimes),
            'median': np.median(lifetimes),
            'std_dev': np.std(lifetimes),
            'min': np.min(lifetimes),
            'max': np.max(lifetimes),
            'reliability_1000h': sum(1 for lt in lifetimes if lt >= 1000) / len(lifetimes),
            'reliability_5000h': sum(1 for lt in lifetimes if lt >= 5000) / len(lifetimes),
            'reliability_10000h': sum(1 for lt in lifetimes if lt >= 10000) / len(lifetimes)
        }
        
        return lifetime_stats