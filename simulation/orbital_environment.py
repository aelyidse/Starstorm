import math
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from core.enhanced_component import EnhancedComponent
from simulation.collision_detection import CollisionDetector
from simulation.collision_resolution import CollisionResolver, CollisionAvoidanceStrategy

class OrbitalRegime(Enum):
    """Classification of orbital regimes"""
    LEO = "Low Earth Orbit"
    MEO = "Medium Earth Orbit"
    GEO = "Geostationary Orbit"
    HEO = "Highly Elliptical Orbit"
    CISLUNAR = "Cislunar Space"
    INTERPLANETARY = "Interplanetary Space"

class DebrisModel:
    """
    Models space debris and micrometeoroid environment.
    """
    def __init__(self, altitude_m: float, inclination_deg: float = 0.0):
        self.altitude_m = altitude_m
        self.inclination_rad = math.radians(inclination_deg)
        self.earth_radius_m = 6371000.0
        
    def get_orbital_regime(self) -> OrbitalRegime:
        """Determine the orbital regime based on altitude"""
        if self.altitude_m < 2000000:  # 2000 km
            return OrbitalRegime.LEO
        elif self.altitude_m < 35786000:  # GEO altitude
            return OrbitalRegime.MEO
        elif 35700000 <= self.altitude_m <= 35900000:  # GEO band
            return OrbitalRegime.GEO
        elif self.altitude_m < 400000000:  # Lunar distance
            return OrbitalRegime.CISLUNAR
        else:
            return OrbitalRegime.INTERPLANETARY
    
    def debris_flux_m2_s(self) -> float:
        """Calculate debris flux (impacts per m² per second)"""
        regime = self.get_orbital_regime()
        
        # Base flux values by regime (particles/m²/s)
        base_flux = {
            OrbitalRegime.LEO: 1e-6,
            OrbitalRegime.MEO: 1e-8,
            OrbitalRegime.GEO: 5e-9,
            OrbitalRegime.CISLUNAR: 1e-9,
            OrbitalRegime.INTERPLANETARY: 5e-10
        }
        
        # LEO has highest debris concentration, especially at certain altitudes
        if regime == OrbitalRegime.LEO:
            # Peaks at ~800-900km
            altitude_factor = math.exp(-(self.altitude_m - 850000)**2 / (2 * 300000**2))
            # Higher inclination orbits (sun-synchronous, polar) have more debris
            inclination_factor = 1.0 + 0.5 * math.sin(self.inclination_rad)
            return base_flux[regime] * (1.0 + altitude_factor) * inclination_factor
        
        # GEO has concentration at geostationary altitude
        elif regime == OrbitalRegime.GEO:
            # Concentrated at GEO
            altitude_factor = math.exp(-(self.altitude_m - 35786000)**2 / (2 * 50000**2))
            return base_flux[regime] * (1.0 + 5.0 * altitude_factor)
            
        return base_flux[regime]
    
    def micrometeoroid_flux_m2_s(self) -> float:
        """Calculate micrometeoroid flux (impacts per m² per second)"""
        # Base flux decreases with distance from Earth (shielding effect)
        base_flux = 1e-8
        
        # Earth shielding factor
        r = self.earth_radius_m + self.altitude_m
        earth_shielding = 0.5 * (1.0 + math.sqrt(1.0 - (self.earth_radius_m / r)**2))
        
        return base_flux * earth_shielding
    
    def total_impact_flux_m2_s(self) -> float:
        """Calculate total impact flux from debris and micrometeoroids"""
        return self.debris_flux_m2_s() + self.micrometeoroid_flux_m2_s()
    
    def describe(self) -> Dict[str, Any]:
        return {
            'altitude_m': self.altitude_m,
            'inclination_deg': math.degrees(self.inclination_rad),
            'orbital_regime': self.get_orbital_regime().value,
            'debris_flux_m2_s': self.debris_flux_m2_s(),
            'micrometeoroid_flux_m2_s': self.micrometeoroid_flux_m2_s(),
            'total_impact_flux_m2_s': self.total_impact_flux_m2_s()
        }

class OrbitalEnvironmentModel(EnhancedComponent):
    """
    Models the orbital environment including debris, thermal cycles,
    and other orbit-specific environmental factors.
    """
    # Add to imports
    from simulation.coordinate_system import CoordinateSystem, ReferenceFrame
    
    # Add to OrbitalEnvironmentModel class
    def __init__(self, 
                 name: str, 
                 dependencies: Optional[List[str]] = None,
                 altitude_m: float = 500000.0,
                 inclination_deg: float = 0.0,
                 eccentricity: float = 0.0):
        super().__init__(name, dependencies)
        self.altitude_m = altitude_m
        self.inclination_deg = inclination_deg
        self.eccentricity = eccentricity
        
        # Initialize sub-models
        self.debris_model = DebrisModel(altitude_m, inclination_deg)
        
        # Initialize collision detection and resolution
        self.collision_detector = CollisionDetector(safety_margin_m=100.0)
        self.collision_resolver = CollisionResolver()
        self.space_objects = []
        self.collision_warnings = []
        
        # Initialize coordinate system
        self.coordinate_system = CoordinateSystem()
        
    async def _start_component(self) -> None:
        """Component-specific startup implementation."""
        self._logger.info(f"Orbital Environment Model started at altitude: {self.altitude_m}m")
        
    def update_orbit(self, altitude_m: float, inclination_deg: float, eccentricity: float) -> None:
        """Update the orbital parameters"""
        self.altitude_m = altitude_m
        self.inclination_deg = inclination_deg
        self.eccentricity = eccentricity
        
        # Update sub-models
        self.debris_model = DebrisModel(altitude_m, inclination_deg)
        
    def thermal_cycle_period_s(self) -> float:
        """Calculate thermal cycle period in seconds"""
        # For circular orbits, this is the orbital period
        # For eccentric orbits, thermal cycles can be more complex
        
        # Orbital period calculation (Kepler's Third Law)
        mu = 3.986004418e14  # Earth's gravitational parameter (m³/s²)
        r = self.earth_radius_m + self.altitude_m
        
        # Semi-major axis for eccentric orbit
        a = r / (1.0 - self.eccentricity)
        
        # Orbital period
        T = 2.0 * math.pi * math.sqrt(a**3 / mu)
        return T
    
    def eclipse_duration_s(self) -> float:
        """Calculate eclipse duration in seconds"""
        # Simplified model for circular orbits
        if self.eccentricity > 0.1:
            # For eccentric orbits, this is a rough approximation
            return self.thermal_cycle_period_s() * 0.3 * (1.0 - self.eccentricity)
        
        # For circular orbits
        r = self.earth_radius_m + self.altitude_m
        
        # Earth's shadow radius at this distance
        shadow_radius = self.earth_radius_m
        
        # Orbital velocity
        mu = 3.986004418e14  # Earth's gravitational parameter
        v = math.sqrt(mu / r)
        
        # Chord length through Earth's shadow
        chord = 2.0 * shadow_radius * math.cos(math.asin(shadow_radius / r))
        
        # Time to traverse the chord
        return chord / v
    
    def add_space_object(self, object_data: Dict[str, Any]) -> None:
        """Add a space object to the environment for collision detection"""
        self.space_objects.append(object_data)
        
    def remove_space_object(self, object_id: str) -> None:
        """Remove a space object from the environment"""
        self.space_objects = [obj for obj in self.space_objects if obj['id'] != object_id]
        
    def update_space_object(self, object_id: str, new_data: Dict[str, Any]) -> None:
        """Update a space object's data"""
        for i, obj in enumerate(self.space_objects):
            if obj['id'] == object_id:
                self.space_objects[i].update(new_data)
                break
                
    def detect_collisions(self, time_horizon_s: float = 3600.0) -> List[Dict[str, Any]]:
        """Detect potential collisions within time horizon"""
        self.collision_warnings = self.collision_detector.scan_for_collisions(
            self.space_objects, time_horizon_s
        )
        return self.collision_warnings
        
    def get_avoidance_maneuver(self, spacecraft_id: str, collision_id: str) -> Optional[Dict[str, Any]]:
        """Get avoidance maneuver for a specific collision warning"""
        # Find the spacecraft and collision object
        spacecraft = None
        collision_object = None
        
        for obj in self.space_objects:
            if obj['id'] == spacecraft_id:
                spacecraft = obj
            elif obj['id'] == collision_id:
                collision_object = obj
                
        if not spacecraft or not collision_object:
            return None
            
        # Find the collision warning
        collision_warning = None
        for warning in self.collision_warnings:
            if ((warning['object1_id'] == spacecraft_id and warning['object2_id'] == collision_id) or
                (warning['object1_id'] == collision_id and warning['object2_id'] == spacecraft_id)):
                collision_warning = warning
                break
                
        if not collision_warning:
            return None
            
        # Calculate optimal avoidance maneuver
        return self.collision_resolver.select_optimal_strategy(
            spacecraft, 
            collision_object,
            collision_warning['time_to_collision_s']
        )
        
    def get_environment_data(self) -> Dict[str, Any]:
        """Get comprehensive orbital environment data"""
        data = {
            'orbit': {
                'altitude_m': self.altitude_m,
                'inclination_deg': self.inclination_deg,
                'eccentricity': self.eccentricity,
                'regime': self.debris_model.get_orbital_regime().value
            },
            'debris': self.debris_model.describe(),
            'thermal_cycle': {
                'period_s': self.thermal_cycle_period_s(),
                'eclipse_duration_s': self.eclipse_duration_s()
            },
            'collision_warnings': len(self.collision_warnings)
        }
        
        # Add collision warning details if any exist
        if self.collision_warnings:
            data['collision_warnings_details'] = self.collision_warnings[:5]  # Top 5 most imminent
            
        return data
        
    def describe(self) -> Dict[str, Any]:
        """Enhanced component description"""
        base_desc = super().describe()
        base_desc.update({
            'environment_data': self.get_environment_data()
        })
        return base_desc