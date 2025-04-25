import math
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from core.enhanced_component import EnhancedComponent

class RadiationType(Enum):
    """Types of radiation in space environment"""
    SOLAR_PARTICLE = "Solar Particle"
    COSMIC_RAY = "Cosmic Ray"
    TRAPPED_PARTICLE = "Trapped Particle"
    SOLAR_FLARE = "Solar Flare"
    GAMMA_RAY = "Gamma Ray"

class MagneticFieldModel:
    """
    Models Earth's magnetic field at different altitudes and latitudes.
    """
    # Earth's magnetic dipole moment (A·m²)
    EARTH_DIPOLE_MOMENT = 7.94e22
    
    def __init__(self, altitude_m: float, latitude_deg: float, longitude_deg: float):
        self.altitude_m = altitude_m
        self.latitude_rad = math.radians(latitude_deg)
        self.longitude_rad = math.radians(longitude_deg)
        self.earth_radius_m = 6371000.0
        
    def field_strength_T(self) -> float:
        """Calculate magnetic field strength in Tesla"""
        # Distance from Earth's center
        r = self.earth_radius_m + self.altitude_m
        
        # Simplified dipole model
        mu0 = 4 * math.pi * 1e-7  # Vacuum permeability
        m = self.EARTH_DIPOLE_MOMENT
        
        # Field strength varies with latitude (strongest at poles)
        theta = math.pi/2 - self.latitude_rad  # Colatitude
        
        # Dipole field equation
        B = (mu0 * m / (4 * math.pi * r**3)) * math.sqrt(1 + 3 * math.cos(theta)**2)
        return B
    
    def describe(self) -> Dict[str, Any]:
        return {
            'altitude_m': self.altitude_m,
            'latitude_deg': math.degrees(self.latitude_rad),
            'longitude_deg': math.degrees(self.longitude_rad),
            'field_strength_T': self.field_strength_T()
        }

class RadiationModel:
    """
    Models space radiation environment including cosmic rays, solar particles,
    and trapped radiation in Earth's magnetic field.
    """
    def __init__(self, altitude_m: float, latitude_deg: float, solar_activity: float = 0.5):
        """
        Initialize radiation model
        
        Args:
            altitude_m: Altitude in meters
            latitude_deg: Latitude in degrees
            solar_activity: Solar activity level (0.0-1.0, where 1.0 is solar maximum)
        """
        self.altitude_m = altitude_m
        self.latitude_rad = math.radians(latitude_deg)
        self.solar_activity = max(0.0, min(1.0, solar_activity))
        self.earth_radius_m = 6371000.0
        
    def total_dose_rate_Gy_s(self) -> float:
        """Calculate total radiation dose rate in Gray/second"""
        return (self.cosmic_ray_dose_rate() + 
                self.trapped_particle_dose_rate() + 
                self.solar_particle_dose_rate())
    
    def cosmic_ray_dose_rate(self) -> float:
        """Calculate cosmic ray dose rate"""
        # Base cosmic ray dose increases with altitude and decreases with solar activity
        base_rate = 5.0e-8  # Gy/s
        altitude_factor = min(1.0, self.altitude_m / 400000.0)
        solar_factor = 1.0 - 0.3 * self.solar_activity  # Cosmic rays decrease during solar max
        
        return base_rate * (1.0 + altitude_factor) * solar_factor
    
    def trapped_particle_dose_rate(self) -> float:
        """Calculate trapped particle (Van Allen belts) dose rate"""
        # Simplified model of Van Allen belts
        inner_belt_center = 4000000.0  # m from Earth's center
        outer_belt_center = 17000000.0  # m from Earth's center
        belt_width = 4000000.0  # m
        
        r = self.earth_radius_m + self.altitude_m
        
        # Calculate distance from belt centers
        inner_belt_dist = abs(r - inner_belt_center)
        outer_belt_dist = abs(r - outer_belt_center)
        
        # Dose peaks in the belts and falls off with distance
        inner_belt_factor = math.exp(-inner_belt_dist**2 / (2 * belt_width**2))
        outer_belt_factor = 0.1 * math.exp(-outer_belt_dist**2 / (2 * (2*belt_width)**2))
        
        # Latitude effect (belts are strongest at equator)
        latitude_factor = math.cos(self.latitude_rad)**2
        
        base_rate = 1.0e-6  # Gy/s
        return base_rate * (inner_belt_factor + outer_belt_factor) * latitude_factor
    
    def solar_particle_dose_rate(self) -> float:
        """Calculate solar particle dose rate"""
        # Solar particle events increase with solar activity
        base_rate = 1.0e-9  # Gy/s
        solar_factor = self.solar_activity**2  # Nonlinear increase with solar activity
        
        # Decreases with Earth's magnetic shielding (stronger at poles)
        magnetic_shielding = 1.0 - 0.9 * math.cos(self.latitude_rad)**2
        
        # Altitude factor (increases in deep space)
        altitude_factor = min(1.0, self.altitude_m / 1000000.0)
        
        return base_rate * solar_factor * magnetic_shielding * (1.0 + altitude_factor)
    
    def radiation_types(self) -> List[Tuple[RadiationType, float]]:
        """Return list of radiation types and their relative contributions"""
        cosmic = self.cosmic_ray_dose_rate()
        trapped = self.trapped_particle_dose_rate()
        solar = self.solar_particle_dose_rate()
        total = cosmic + trapped + solar
        
        if total == 0:
            return []
            
        return [
            (RadiationType.COSMIC_RAY, cosmic/total),
            (RadiationType.TRAPPED_PARTICLE, trapped/total),
            (RadiationType.SOLAR_PARTICLE, solar/total)
        ]
    
    def describe(self) -> Dict[str, Any]:
        return {
            'altitude_m': self.altitude_m,
            'latitude_deg': math.degrees(self.latitude_rad),
            'solar_activity': self.solar_activity,
            'total_dose_rate_Gy_s': self.total_dose_rate_Gy_s(),
            'radiation_types': [(rt.value, contrib) for rt, contrib in self.radiation_types()]
        }

class VacuumModel:
    """
    Models vacuum conditions for outgassing, microdebris, and residual atmosphere effects.
    """
    def __init__(self, altitude_m: float):
        self.altitude_m = altitude_m

    def pressure_Pa(self) -> float:
        # Exponential decay of atmospheric pressure with altitude (beyond thermosphere)
        if self.altitude_m < 600e3:
            return 1e-7  # Pa, upper thermosphere
        else:
            # LEO and beyond
            return 1e-10 * math.exp(-(self.altitude_m - 600e3) / 100e3)

    def mean_free_path_m(self) -> float:
        # Approximate mean free path in meters
        p = self.pressure_Pa()
        T = 1000.0  # K, typical
        k = 1.380649e-23  # J/K
        d = 3.7e-10  # m, effective diameter of O2/N2
        n = p / (k * T)
        if n == 0:
            return float('inf')
        return 1.0 / (math.sqrt(2) * math.pi * d ** 2 * n)

    def describe(self) -> Dict[str, Any]:
        return {
            'altitude_m': self.altitude_m,
            'pressure_Pa': self.pressure_Pa(),
            'mean_free_path_m': self.mean_free_path_m(),
        }

class SpaceEnvironmentModel(EnhancedComponent):
    """
    Comprehensive space environment model that combines vacuum, radiation,
    and magnetic field models into a single component.
    """
    def __init__(self, 
                 name: str, 
                 dependencies: Optional[List[str]] = None,
                 altitude_m: float = 500000.0,
                 latitude_deg: float = 0.0,
                 longitude_deg: float = 0.0,
                 solar_activity: float = 0.5):
        super().__init__(name, dependencies)
        self.altitude_m = altitude_m
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.solar_activity = solar_activity
        
        # Initialize sub-models
        self.vacuum_model = VacuumModel(altitude_m)
        self.radiation_model = RadiationModel(altitude_m, latitude_deg, solar_activity)
        self.magnetic_model = MagneticFieldModel(altitude_m, latitude_deg, longitude_deg)
        
    async def _start_component(self) -> None:
        """Component-specific startup implementation."""
        self._logger.info(f"Space Environment Model started at altitude: {self.altitude_m}m")
        
    def update_position(self, altitude_m: float, latitude_deg: float, longitude_deg: float) -> None:
        """Update the position of the environment model"""
        self.altitude_m = altitude_m
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        
        # Update sub-models
        self.vacuum_model = VacuumModel(altitude_m)
        self.radiation_model = RadiationModel(altitude_m, latitude_deg, self.solar_activity)
        self.magnetic_model = MagneticFieldModel(altitude_m, latitude_deg, longitude_deg)
        
    def update_solar_activity(self, solar_activity: float) -> None:
        """Update the solar activity level"""
        self.solar_activity = max(0.0, min(1.0, solar_activity))
        self.radiation_model = RadiationModel(
            self.altitude_m, self.latitude_deg, self.solar_activity
        )
        
    def get_environment_data(self) -> Dict[str, Any]:
        """Get comprehensive environment data"""
        return {
            'position': {
                'altitude_m': self.altitude_m,
                'latitude_deg': self.latitude_deg,
                'longitude_deg': self.longitude_deg
            },
            'solar_activity': self.solar_activity,
            'vacuum': self.vacuum_model.describe(),
            'radiation': self.radiation_model.describe(),
            'magnetic_field': self.magnetic_model.describe()
        }
        
    def describe(self) -> Dict[str, Any]:
        """Enhanced component description"""
        base_desc = super().describe()
        base_desc.update({
            'environment_data': self.get_environment_data()
        })
        return base_desc
