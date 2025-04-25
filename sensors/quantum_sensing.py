from typing import Dict, List, Tuple
import numpy as np

class QuantumSensorProperties:
    """Class to store properties of quantum sensors."""
    
    def __init__(self, sensitivity: float, operating_range: Tuple[float, float], noise_floor: float):
        self.sensitivity = sensitivity  # Sensor specific units (e.g., femtotesla/sqrt(Hz) for magnetometer)
        self.operating_range = operating_range  # Operational limits (min, max)
        self.noise_floor = noise_floor  # Base noise level

class QuantumSensor:
    """Base class representing a quantum sensor with specific properties."""
    
    def __init__(self, name: str, properties: QuantumSensorProperties):
        self.name = name
        self.properties = properties
        self.current_reading: float = 0.0
        self.is_operational: bool = True

    def read(self, true_value: float, interference: float = 0.0) -> float:
        """Simulate reading from the sensor with noise and interference."""
        if not self.is_operational:
            return 0.0
            
        noise = np.random.normal(0, self.properties.noise_floor)
        self.current_reading = (true_value + noise + interference) * self.properties.sensitivity
        return self.current_reading

    def set_operational_status(self, status: bool) -> None:
        """Set the operational status of the sensor."""
        self.is_operational = status

class QuantumMagnetometer(QuantumSensor):
    """Simulates a quantum magnetometer using nitrogen-vacancy centers in diamond."""
    
    def __init__(self):
        props = QuantumSensorProperties(
            sensitivity=1e-6,  # 10 femtotesla/sqrt(Hz) scaled for simulation
            operating_range=(1e-9, 1e-3),  # Tesla range
            noise_floor=1e-8  # Base noise level
        )
        super().__init__(name="NV-Diamond Magnetometer", properties=props)

    def detect_magnetic_signature(self, field_strength: float, distance: float) -> float:
        """Detect magnetic field signature with falloff by distance."""
        # Magnetic field strength decreases with cube of distance (simplified dipole model)
        effective_field = field_strength / (distance ** 3) if distance > 0 else field_strength
        return self.read(effective_field)

class QuantumGravimeter(QuantumSensor):
    """Simulates a quantum gravimeter using ultracold atom interferometry."""
    
    def __init__(self):
        props = QuantumSensorProperties(
            sensitivity=1e-8,  # Microgal/sqrt(Hz) scaled
            operating_range=(9.0, 10.0),  # m/s^2 range
            noise_floor=1e-7
        )
        super().__init__(name="Ultracold Atom Gravimeter", properties=props)

    def measure_gravitational_field(self, field_strength: float, altitude: float) -> float:
        """Measure gravitational field with altitude variation."""
        # Simplified gravitational variation with altitude
        effective_field = field_strength * (1 - 2.255e-7 * altitude) if altitude >= 0 else field_strength
        return self.read(effective_field)

class QuantumEMFieldSensor(QuantumSensor):
    """Simulates a quantum electromagnetic field sensor."""
    
    def __init__(self):
        props = QuantumSensorProperties(
            sensitivity=1e-5,  # V/m sensitivity scaled
            operating_range=(1e-6, 1e3),  # V/m range
            noise_floor=1e-6
        )
        super().__init__(name="Quantum EM Field Sensor", properties=props)

    def detect_em_field(self, field_strength: float, frequency: float) -> float:
        """Detect electromagnetic field with frequency dependency."""
        # Better sensitivity at lower frequencies (simplified)
        freq_factor = min(1.0, 100.0 / frequency) if frequency > 0 else 1.0
        effective_field = field_strength * freq_factor
        return self.read(effective_field)

class QuantumImagingSystem(QuantumSensor):
    """Simulates a quantum imaging system for high-resolution detection."""
    
    def __init__(self):
        props = QuantumSensorProperties(
            sensitivity=1e-3,  # Photon count sensitivity scaled
            operating_range=(1.0, 1e6),  # Photon flux range
            noise_floor=1e-4
        )
        super().__init__(name="Quantum Imaging System", properties=props)
        self.resolution: Tuple[int, int] = (4096, 3072)  # Pixel resolution as per RTF

    def capture_image(self, photon_flux: float, distance: float) -> Dict[str, float]:
        """Simulate capturing an image with resolution degradation by distance."""
        effective_flux = photon_flux / (distance ** 2) if distance > 0 else photon_flux
        quality = min(1.0, 1000.0 / distance) if distance > 0 else 1.0
        return {
            "flux_reading": self.read(effective_flux),
            "image_quality": quality
        }

class QuantumSensingFramework:
    """Framework to manage multiple quantum sensors for enhanced situational awareness."""
    
    def __init__(self):
        self.sensors: Dict[str, QuantumSensor] = {}
        self.cooling_system_active: bool = True
        self.shielding_effectiveness: float = 0.95  # 95% interference reduction
        self.add_sensor(QuantumMagnetometer())
        self.add_sensor(QuantumGravimeter())
        self.add_sensor(QuantumEMFieldSensor())
        self.add_sensor(QuantumImagingSystem())

    def add_sensor(self, sensor: QuantumSensor) -> None:
        """Add a quantum sensor to the framework."""
        self.sensors[sensor.name] = sensor

    def set_cooling_system(self, active: bool) -> None:
        """Set status of cryogenic cooling system (affects sensor performance)."""
        self.cooling_system_active = active
        for sensor in self.sensors.values():
            # Simplified: cooling failure disables sensors
            sensor.set_operational_status(active)

    def set_shielding_effectiveness(self, effectiveness: float) -> None:
        """Set effectiveness of electromagnetic shielding (0.0 to 1.0)."""
        self.shielding_effectiveness = max(0.0, min(1.0, effectiveness))

    def get_magnetic_reading(self, field_strength: float, distance: float) -> float:
        """Get reading from quantum magnetometer."""
        sensor = self.sensors.get("NV-Diamond Magnetometer")
        if not sensor:
            return 0.0
        interference = 0.0 if self.shielding_effectiveness == 1.0 else field_strength * 0.1
        interference *= (1.0 - self.shielding_effectiveness)
        return sensor.detect_magnetic_signature(field_strength, distance) + interference

    def get_gravitational_reading(self, field_strength: float, altitude: float) -> float:
        """Get reading from quantum gravimeter."""
        sensor = self.sensors.get("Ultracold Atom Gravimeter")
        if not sensor:
            return 0.0
        return sensor.measure_gravitational_field(field_strength, altitude)

    def get_em_field_reading(self, field_strength: float, frequency: float) -> float:
        """Get reading from quantum EM field sensor."""
        sensor = self.sensors.get("Quantum EM Field Sensor")
        if not sensor:
            return 0.0
        interference = field_strength * 0.05 * (1.0 - self.shielding_effectiveness)
        return sensor.detect_em_field(field_strength, frequency) + interference

    def get_quantum_image(self, photon_flux: float, distance: float) -> Dict[str, float]:
        """Get quantum imaging data."""
        sensor = self.sensors.get("Quantum Imaging System")
        if not sensor:
            return {"flux_reading": 0.0, "image_quality": 0.0}
        return sensor.capture_image(photon_flux, distance)

    def get_system_status(self) -> Dict[str, bool]:
        """Return status of all sensors and supporting systems."""
        status = {
            "cooling_system": self.cooling_system_active,
            "shielding_effectiveness": self.shielding_effectiveness
        }
        for name, sensor in self.sensors.items():
            status[f"{name}_operational"] = sensor.is_operational
        return status

# Example usage
if __name__ == "__main__":
    framework = QuantumSensingFramework()
    
    # Test magnetic field detection
    mag_reading = framework.get_magnetic_reading(field_strength=1e-6, distance=100.0)
    print(f"Magnetic Field Reading: {mag_reading:.2e} scaled Tesla")
    
    # Test gravitational field measurement
    grav_reading = framework.get_gravitational_reading(field_strength=9.81, altitude=100000.0)
    print(f"Gravitational Field Reading: {grav_reading:.2e} scaled m/s^2")
    
    # Test EM field detection
    em_reading = framework.get_em_field_reading(field_strength=1e-3, frequency=10.0)
    print(f"EM Field Reading: {em_reading:.2e} scaled V/m")
    
    # Test quantum imaging
    img_data = framework.get_quantum_image(photon_flux=1000.0, distance=500.0)
    print(f"Quantum Imaging Data:")
    print(f"  Flux Reading: {img_data['flux_reading']:.2e} scaled photons")
    print(f"  Image Quality: {img_data['image_quality']*100:.1f}%")
    
    # Check system status
    status = framework.get_system_status()
    print("\nSystem Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Simulate cooling system failure
    framework.set_cooling_system(False)
    print("\nAfter Cooling System Failure:")
    status = framework.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    mag_reading_failed = framework.get_magnetic_reading(field_strength=1e-6, distance=100.0)
    print(f"Magnetic Field Reading (cooling failed): {mag_reading_failed:.2e} scaled Tesla")
