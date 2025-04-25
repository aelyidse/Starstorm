from typing import Dict, List, Optional, Tuple
import numpy as np

class MetamaterialControlSystem:
    """Controls and dynamically adjusts metamaterial properties for stealth optimization."""
    
    def __init__(self):
        self.threat_frequencies: List[float] = []
        self.current_config: Dict[str, float] = {
            "permittivity": -2.5,
            "permeability": -1.8,
            "refractive_index": -2.0
        }
        self.power_consumption: float = 0.0  # in watts
        self.adaptation_state: str = "PASSIVE"

    def analyze_threat_environment(self, sensor_data: Dict[str, float]) -> List[float]:
        """Analyze sensor data to identify threat frequencies."""
        self.threat_frequencies = []
        for freq, power in sensor_data.items():
            if power > 0.1:  # Threshold for significant threat
                self.threat_frequencies.append(float(freq))
        return self.threat_frequencies

    def adjust_metamaterial_properties(self, target_frequency: float) -> Dict[str, float]:
        """Adjust metamaterial properties to counter specific frequency threats."""
        self.adaptation_state = "ACTIVE"
        self.power_consumption = 15.0  # Active adaptation power draw
        
        # Adjust properties based on frequency band
        if target_frequency < 18.0:  # X, Ku, K bands
            self.current_config["permittivity"] = -2.8
            self.current_config["permeability"] = -2.0
            self.current_config["refractive_index"] = -2.2
        else:  # Millimeter wave
            self.current_config["permittivity"] = -1.5
            self.current_config["permeability"] = -1.0
            self.current_config["refractive_index"] = -1.3
            
        return self.current_config

    def set_passive_mode(self) -> None:
        """Set system to passive mode to conserve power."""
        self.adaptation_state = "PASSIVE"
        self.power_consumption = 0.0
        self.current_config = {
            "permittivity": -2.5,
            "permeability": -1.8,
            "refractive_index": -2.0
        }

    def get_power_draw(self) -> float:
        """Return current power consumption of the system."""
        return self.power_consumption

    def get_adaptation_status(self) -> str:
        """Return current adaptation state."""
        return self.adaptation_state

    def predict_threat_evolution(self, historical_data: List[Dict[str, float]]) -> List[float]:
        """Predict future threat frequencies based on historical sensor data."""
        if len(historical_data) < 3:
            return self.threat_frequencies
            
        # Simple linear prediction based on last few data points
        predicted_frequencies = []
        latest_frequencies = [d.keys() for d in historical_data[-3:]]
        for freq_band in latest_frequencies[-1]:
            freq = float(freq_band)
            trend = sum(1 for d in latest_frequencies if freq in d)
            if trend >= 2:  # Frequency appears consistently
                predicted_frequencies.append(freq)
                # Predict potential harmonics
                if freq * 2 <= 300.0:
                    predicted_frequencies.append(freq * 2)
        return predicted_frequencies

    def optimize_for_multiple_threats(self, frequencies: List[float]) -> Dict[str, float]:
        """Optimize metamaterial properties for multiple simultaneous threats."""
        if not frequencies:
            return self.current_config
            
        # Use average frequency as target for multiple threats
        avg_freq = sum(frequencies) / len(frequencies)
        return self.adjust_metamaterial_properties(avg_freq)

# Example usage
if __name__ == "__main__":
    control = MetamaterialControlSystem()
    
    # Simulate sensor data (frequency: power level)
    sensor_reading = {
        "10.0": 0.5,  # Strong signal at 10 GHz
        "35.0": 0.05, # Weak signal at 35 GHz
        "100.0": 0.2  # Moderate signal at 100 GHz
    }
    
    # Analyze threats
    threats = control.analyze_threat_environment(sensor_reading)
    print(f"Detected threat frequencies: {threats} GHz")
    
    # Optimize for multiple threats
    config = control.optimize_for_multiple_threats(threats)
    print(f"Adjusted configuration: {config}")
    print(f"Power draw: {control.get_power_draw()} W")
    print(f"Adaptation state: {control.get_adaptation_status()}")
    
    # Return to passive mode
    control.set_passive_mode()
    print(f"After passive mode - Power draw: {control.get_power_draw()} W")
    print(f"Adaptation state: {control.get_adaptation_status()}")
