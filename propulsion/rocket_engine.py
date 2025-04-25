import math
from typing import Dict, Any, Optional, List, Tuple
from .thrust_profile import ThrustProfileManager

class RocketEngine:
    """
    High-fidelity rocket engine simulation with variable thrust and fuel consumption models.
    Supports throttle, mixture ratio, and transient effects.
    """
    def __init__(self,
                 max_thrust_N: float,
                 isp_s: float,
                 fuel_density_kg_m3: float,
                 chamber_pressure_Pa: float,
                 throttle_min: float = 0.1,
                 throttle_max: float = 1.0):
        self.max_thrust_N = max_thrust_N
        self.isp_s = isp_s
        self.fuel_density_kg_m3 = fuel_density_kg_m3
        self.chamber_pressure_Pa = chamber_pressure_Pa
        self.throttle_min = throttle_min
        self.throttle_max = throttle_max
        self.current_throttle = 0.0
        self.fuel_mass_kg = 0.0
        self.oxidizer_mass_kg = 0.0
        self.running = False
        
        # Initialize thrust profile manager
        self.thrust_profile = ThrustProfileManager(
            engine_max_thrust_N=max_thrust_N,
            engine_min_throttle=throttle_min,
            engine_max_throttle=throttle_max
        )
        
        # Simulation time
        self.simulation_time = 0.0

    def load_propellants(self, fuel_mass_kg: float, oxidizer_mass_kg: float):
        self.fuel_mass_kg = fuel_mass_kg
        self.oxidizer_mass_kg = oxidizer_mass_kg

    def set_throttle(self, throttle: float):
        self.current_throttle = max(self.throttle_min, min(self.throttle_max, throttle))

    def start(self):
        self.running = True

    def shutdown(self):
        self.running = False
        self.current_throttle = 0.0

    def compute_thrust(self, ambient_pressure_Pa: float = 101325.0) -> float:
        if not self.running or self.fuel_mass_kg <= 0 or self.current_throttle <= 0:
            return 0.0
        # Thrust = max_thrust * throttle * pressure correction
        pressure_ratio = max(0.7, 1.0 - (ambient_pressure_Pa / self.chamber_pressure_Pa))
        thrust = self.max_thrust_N * self.current_throttle * pressure_ratio
        return max(0.0, thrust)

    def fuel_consumption(self, dt: float) -> Dict[str, float]:
        if not self.running or self.current_throttle <= 0:
            return {'fuel_used_kg': 0.0, 'oxidizer_used_kg': 0.0}
        g0 = 9.80665
        thrust = self.compute_thrust()
        mdot = thrust / (self.isp_s * g0)
        # Assume mixture ratio O/F = 2.5 (typical for LOX/Kerosene)
        of_ratio = 2.5
        oxidizer_used = mdot * dt * (of_ratio / (1 + of_ratio))
        fuel_used = mdot * dt * (1 / (1 + of_ratio))
        self.fuel_mass_kg = max(0.0, self.fuel_mass_kg - fuel_used)
        self.oxidizer_mass_kg = max(0.0, self.oxidizer_mass_kg - oxidizer_used)
        return {'fuel_used_kg': fuel_used, 'oxidizer_used_kg': oxidizer_used}

    def update(self, dt: float, ambient_pressure_Pa: float = 101325.0) -> Dict[str, Any]:
        thrust = self.compute_thrust(ambient_pressure_Pa)
        consumption = self.fuel_consumption(dt)
        return {
            'thrust_N': thrust,
            'fuel_mass_kg': self.fuel_mass_kg,
            'oxidizer_mass_kg': self.oxidizer_mass_kg,
            **consumption
        }

    def initialize_sequence_controller(self, 
                                      pre_ignition_duration_s: float = 2.0,
                                      ignition_duration_s: float = 0.5,
                                      ramp_up_duration_s: float = 3.0,
                                      shutdown_duration_s: float = 2.5) -> None:
        """
        Initialize engine sequence controller.
        
        Args:
            pre_ignition_duration_s: Pre-ignition phase duration in seconds
            ignition_duration_s: Ignition phase duration in seconds
            ramp_up_duration_s: Thrust ramp-up duration in seconds
            shutdown_duration_s: Shutdown sequence duration in seconds
        """
        from engine_sequence import EngineSequenceController
        
        self.sequence_controller = EngineSequenceController(
            engine_type="bipropellant",
            pre_ignition_duration_s=pre_ignition_duration_s,
            ignition_duration_s=ignition_duration_s,
            ramp_up_duration_s=ramp_up_duration_s,
            shutdown_duration_s=shutdown_duration_s
        )
        
    def start_with_sequence(self, target_throttle: float = 1.0) -> Dict[str, Any]:
        """
        Start engine using sequence controller.
        
        Args:
            target_throttle: Target throttle level (0.0-1.0)
            
        Returns:
            Sequence status
        """
        if not hasattr(self, 'sequence_controller'):
            self.initialize_sequence_controller()
            
        # Start sequence
        result = self.sequence_controller.start_engine(target_throttle_pct=target_throttle * 100.0)
        
        return result
        
    def shutdown_with_sequence(self) -> Dict[str, Any]:
        """
        Shutdown engine using sequence controller.
        
        Returns:
            Sequence status
        """
        if not hasattr(self, 'sequence_controller'):
            return {"status": "error", "message": "Sequence controller not initialized"}
            
        # Start shutdown sequence
        result = self.sequence_controller.shutdown_engine()
        
        return result
        
    def update_sequence(self, dt: float) -> Dict[str, Any]:
        """
        Update engine sequence.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Sequence state
        """
        if not hasattr(self, 'sequence_controller'):
            return {"status": "error", "message": "Sequence controller not initialized"}
            
        # Update sequence
        state = self.sequence_controller.update(dt)
        
        # Apply sequence state to engine
        if state["phase"] != "idle" and state["phase"] != "shutdown_complete":
            self.current_throttle = state["throttle"]
            self.running = state["throttle"] > 0.0
            
        return state

    def set_thrust_profile(self, profile_type: str, **profile_params) -> None:
        """
        Set thrust profile for the engine.
        
        Args:
            profile_type: Type of profile ("constant", "step", "polynomial", "custom")
            **profile_params: Profile-specific parameters
        """
        if profile_type == "constant":
            throttle = profile_params.get("throttle", 1.0)
            self.thrust_profile.set_constant_profile(throttle)
            
        elif profile_type == "step":
            time_points = profile_params.get("time_points", [0.0])
            throttle_points = profile_params.get("throttle_points", [1.0])
            self.thrust_profile.set_step_profile(time_points, throttle_points)
            
        elif profile_type == "polynomial":
            coefficients = profile_params.get("coefficients", [1.0])
            duration = profile_params.get("duration", 100.0)
            self.thrust_profile.set_polynomial_profile(coefficients, duration)
            
        elif profile_type == "custom":
            time_points = profile_params.get("time_points", [0.0])
            throttle_points = profile_params.get("throttle_points", [1.0])
            interpolation = profile_params.get("interpolation", "linear")
            self.thrust_profile.set_custom_profile(time_points, throttle_points, interpolation)
            
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
    
    def update_with_profile(self, dt: float, ambient_pressure_Pa: float = 101325.0) -> Dict[str, Any]:
        """
        Update engine state using current thrust profile.
        
        Args:
            dt: Time step in seconds
            ambient_pressure_Pa: Ambient pressure in Pa
            
        Returns:
            Engine state
        """
        # Update simulation time
        self.simulation_time += dt
        
        # Get throttle from profile
        profile_throttle = self.thrust_profile.get_throttle_at_time(self.simulation_time)
        
        # Set engine throttle
        if self.running:
            self.set_throttle(profile_throttle)
        
        # Update engine state
        return self.update(dt, ambient_pressure_Pa)
