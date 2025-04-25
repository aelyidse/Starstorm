from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ManufacturingParameters:
    """Parameters for manufacturing process simulation."""
    tolerance: float  # Manufacturing tolerance in meters
    material_yield_strength: float  # Material strength in MPa
    thermal_expansion_coeff: float  # Thermal expansion coefficient
    max_thermal_stress: float  # Maximum allowable thermal stress in MPa

@dataclass
class DesignSpecifications:
    """Design specifications for a component."""
    dimensions: Tuple[float, float, float]  # Length, width, height in meters
    weight: float  # Target weight in kg
    material: str  # Material type
    critical_tolerances: Dict[str, float]  # Critical dimensions and their tolerances

class ManufacturingProcess:
    """Simulates a specific manufacturing process with precision characteristics."""
    
    def __init__(self, name: str, params: ManufacturingParameters):
        self.name = name
        self.params = params
        self.is_operational: bool = True
        self.process_variability: float = 0.001  # Base variability in process

    def set_operational(self, status: bool) -> None:
        """Set the operational status of the manufacturing process."""
        self.is_operational = status

    def simulate_process(self, spec: DesignSpecifications, 
                        env_temp: float = 298.0) -> Dict[str, float]:
        """Simulate the manufacturing process for a given design spec."""
        if not self.is_operational:
            return {"deviation": float('inf'), "thermal_stress": float('inf'), 
                   "yield_probability": 0.0}
        
        # Simulate dimensional deviation based on tolerance and variability
        deviation = np.random.normal(0, self.params.tolerance + self.process_variability)
        for dim, tol in spec.critical_tolerances.items():
            dev = np.random.normal(0, tol / 2.0)
            deviation = max(abs(deviation), abs(dev))  # Worst-case critical dimension
        
        # Calculate thermal stress from temperature change (simplified)
        temp_change = abs(env_temp - 298.0)  # Deviation from standard temp (25C)
        thermal_stress = self.params.thermal_expansion_coeff * temp_change * self.params.material_yield_strength
        
        # Probability of successful yield based on deviation and stress
        dev_factor = max(0.0, 1.0 - abs(deviation) / self.params.tolerance)
        stress_factor = max(0.0, 1.0 - thermal_stress / self.params.max_thermal_stress)
        yield_prob = dev_factor * stress_factor * 0.95  # Base 95% success rate
        
        return {
            "deviation": deviation,
            "thermal_stress": thermal_stress,
            "yield_probability": max(0.01, min(0.99, yield_prob))
        }

class DigitalTwinFactory:
    """Digital twin simulation of factory with bi-directional design feedback."""
    
    def __init__(self):
        self.processes: Dict[str, ManufacturingProcess] = {}
        self.designs: Dict[str, DesignSpecifications] = {}
        self.feedback_history: List[Dict] = []
        self.environment_temperature: float = 298.0  # Kelvin (25C)
        self.maintenance_status: Dict[str, bool] = {}

    def add_process(self, name: str, params: ManufacturingParameters) -> None:
        """Add a manufacturing process to the digital twin."""
        self.processes[name] = ManufacturingProcess(name, params)
        self.maintenance_status[name] = True

    def add_design(self, name: str, spec: DesignSpecifications) -> None:
        """Add a design specification to the digital twin."""
        self.designs[name] = spec

    def set_environment_temperature(self, temp: float) -> None:
        """Set the factory environment temperature."""
        self.environment_temperature = max(273.0, min(373.0, temp))  # Reasonable bounds

    def set_maintenance_status(self, process_name: str, status: bool) -> None:
        """Set maintenance status for a process (affects variability)."""
        if process_name in self.processes:
            self.maintenance_status[process_name] = status
            self.processes[process_name].process_variability *= (1.5 if not status else 1.0)
            self.processes[process_name].set_operational(status)

    def simulate_production_run(self, design_name: str, process_name: str, 
                              num_items: int = 100) -> Dict[str, float]:
        """Simulate a production run for a design using a specific process."""
        if design_name not in self.designs or process_name not in self.processes:
            return {"average_deviation": float('inf'), "success_rate": 0.0, 
                   "average_thermal_stress": float('inf')}
        
        design = self.designs[design_name]
        results = []
        for _ in range(num_items):
            result = self.processes[process_name].simulate_process(
                design, self.environment_temperature)
            results.append(result)
        
        avg_deviation = np.mean([r['deviation'] for r in results])
        success_rate = np.mean([r['yield_probability'] for r in results])
        avg_stress = np.mean([r['thermal_stress'] for r in results])
        
        feedback = {
            "design_name": design_name,
            "process_name": process_name,
            "average_deviation": avg_deviation,
            "success_rate": success_rate,
            "average_thermal_stress": avg_stress,
            "environment_temperature": self.environment_temperature
        }
        self.feedback_history.append(feedback)
        if len(self.feedback_history) > 50:  # Limit history size
            self.feedback_history.pop(0)
        
        return {
            "average_deviation": avg_deviation,
            "success_rate": success_rate,
            "average_thermal_stress": avg_stress
        }

    def generate_design_feedback(self, design_name: str) -> Dict[str, any]:
        """Generate bi-directional feedback for design improvement based on manufacturing data."""
        relevant_feedback = [f for f in self.feedback_history if f['design_name'] == design_name]
        if not relevant_feedback:
            return {"status": "No data available", "suggested_tolerance_adjustment": 0.0, 
                   "material_suggestion": "N/A"}
        
        avg_success_rate = np.mean([f['success_rate'] for f in relevant_feedback])
        avg_deviation = np.mean([f['average_deviation'] for f in relevant_feedback])
        avg_stress = np.mean([f['average_thermal_stress'] for f in relevant_feedback])
        
        # Suggest tolerance adjustments
        tolerance_adj = avg_deviation * 1.2 if avg_success_rate < 0.8 else avg_deviation * 0.8
        material_sug = "Consider stronger material" if avg_stress > 100.0 else "Material adequate"
        if avg_stress > 200.0:
            material_sug = "Urgent: High thermal stress, change material or design"
        
        return {
            "status": "Feedback generated",
            "average_success_rate": avg_success_rate,
            "average_deviation": avg_deviation,
            "average_thermal_stress": avg_stress,
            "suggested_tolerance_adjustment": tolerance_adj,
            "material_suggestion": material_sug
        }

    def validate_precision(self, design_name: str, process_name: str, 
                          required_tolerance: float) -> Dict[str, any]:
        """Validate if manufacturing process meets precision requirements."""
        if design_name not in self.designs or process_name not in self.processes:
            return {"validation_status": "Invalid design or process", "within_tolerance": False}
        
        # Run a small simulation batch to test precision
        result = self.simulate_production_run(design_name, process_name, num_items=10)
        deviation = result['average_deviation']
        within_tol = abs(deviation) <= required_tolerance
        
        return {
            "validation_status": "Validated",
            "measured_deviation": deviation,
            "required_tolerance": required_tolerance,
            "within_tolerance": within_tol,
            "success_rate": result['success_rate']
        }

    def get_factory_status(self) -> Dict[str, any]:
        """Get the current status of the digital twin factory."""
        status = {
            "environment_temperature": self.environment_temperature,
            "number_of_designs": len(self.designs),
            "number_of_processes": len(self.processes),
            "maintenance_status": self.maintenance_status.copy(),
            "feedback_history_length": len(self.feedback_history)
        }
        return status

# Example usage
if __name__ == "__main__":
    # Create a digital twin factory
    factory = DigitalTwinFactory()
    
    # Add manufacturing processes
    additive_params = ManufacturingParameters(
        tolerance=1e-4, material_yield_strength=400.0,
        thermal_expansion_coeff=1.2e-5, max_thermal_stress=200.0
    )
    subtractive_params = ManufacturingParameters(
        tolerance=5e-5, material_yield_strength=450.0,
        thermal_expansion_coeff=1.1e-5, max_thermal_stress=250.0
    )
    factory.add_process("Additive Manufacturing", additive_params)
    factory.add_process("Subtractive Manufacturing", subtractive_params)
    
    # Add design specifications
    drone_frame = DesignSpecifications(
        dimensions=(0.5, 0.5, 0.1), weight=2.0, material="Carbon Composite",
        critical_tolerances={"frame_thickness": 1e-4, "mounting_hole": 5e-5}
    )
    factory.add_design("Drone Frame", drone_frame)
    
    # Simulate production runs
    additive_result = factory.simulate_production_run("Drone Frame", "Additive Manufacturing")
    print("Additive Manufacturing Results:")
    for key, value in additive_result.items():
        print(f"  {key}: {value:.2e}")
    
    subtractive_result = factory.simulate_production_run("Drone Frame", "Subtractive Manufacturing")
    print("\nSubtractive Manufacturing Results:")
    for key, value in subtractive_result.items():
        print(f"  {key}: {value:.2e}")
    
    # Generate design feedback
    feedback = factory.generate_design_feedback("Drone Frame")
    print("\nDesign Feedback for Drone Frame:")
    for key, value in feedback.items():
        print(f"  {key}: {value}")
    
    # Validate manufacturing precision
    validation = factory.validate_precision("Drone Frame", "Additive Manufacturing", 2e-4)
    print("\nPrecision Validation for Additive Manufacturing:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Test environmental impact
    factory.set_environment_temperature(308.0)  # 35C
    hot_result = factory.simulate_production_run("Drone Frame", "Additive Manufacturing")
    print("\nAdditive Manufacturing Results at 35C:")
    for key, value in hot_result.items():
        print(f"  {key}: {value:.2e}")
    
    # Test maintenance impact
    factory.set_maintenance_status("Additive Manufacturing", False)
    maintenance_result = factory.simulate_production_run("Drone Frame", "Additive Manufacturing")
    print("\nAdditive Manufacturing Results During Maintenance:")
    for key, value in maintenance_result.items():
        print(f"  {key}: {value:.2e}")
    
    # Check factory status
    status = factory.get_factory_status()
    print("\nFactory Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
