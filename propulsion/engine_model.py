from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import math

class PropulsionSystem:
    """
    High-fidelity engine model with thermodynamic simulation.
    Models propellant flow, combustion, and performance characteristics.
    """
    def __init__(self, 
                 engine_type: str = "bipropellant",
                 thrust_max_N: float = 500.0,
                 isp_vacuum_s: float = 320.0,
                 mixture_ratio: float = 2.1,
                 chamber_pressure_MPa: float = 2.0,
                 expansion_ratio: float = 150.0):
        """
        Initialize propulsion system with engine parameters.
        
        Args:
            engine_type: Engine type (monopropellant, bipropellant, ion, etc.)
            thrust_max_N: Maximum thrust in Newtons
            isp_vacuum_s: Specific impulse in vacuum (seconds)
            mixture_ratio: Oxidizer to fuel ratio (for bipropellant)
            chamber_pressure_MPa: Chamber pressure in MPa
            expansion_ratio: Nozzle expansion ratio
        """
        self.engine_type = engine_type
        self.thrust_max_N = thrust_max_N
        self.isp_vacuum_s = isp_vacuum_s
        self.mixture_ratio = mixture_ratio
        self.chamber_pressure_MPa = chamber_pressure_MPa
        self.expansion_ratio = expansion_ratio
        
        # Constants
        self.g0 = 9.80665  # Standard gravity (m/s²)
        self.R_universal = 8314.0  # Universal gas constant (J/kmol·K)
        
        # Derived parameters
        self.calculate_derived_parameters()
        
        # State variables
        self.current_thrust_N = 0.0
        self.current_isp_s = 0.0
        self.current_mass_flow_kgps = 0.0
        self.current_chamber_temp_K = 0.0
        self.current_exit_pressure_Pa = 0.0
        
        # Performance history
        self.performance_history = []
        
    def calculate_derived_parameters(self) -> None:
        """Calculate derived engine parameters based on inputs."""
        # Molecular weight of exhaust (estimated based on engine type)
        if self.engine_type == "hydrolox":
            self.exhaust_molecular_weight = 12.0  # kg/kmol
            self.gamma = 1.26  # Ratio of specific heats
        elif self.engine_type == "kerolox":
            self.exhaust_molecular_weight = 22.0  # kg/kmol
            self.gamma = 1.22
        elif self.engine_type == "methalox":
            self.exhaust_molecular_weight = 16.0  # kg/kmol
            self.gamma = 1.24
        elif self.engine_type == "hypergolic":
            self.exhaust_molecular_weight = 24.0  # kg/kmol
            self.gamma = 1.25
        else:  # Default/monopropellant
            self.exhaust_molecular_weight = 20.0  # kg/kmol
            self.gamma = 1.23
            
        # Calculate characteristic velocity (c*)
        self.chamber_pressure_Pa = self.chamber_pressure_MPa * 1e6
        self.R_specific = self.R_universal / self.exhaust_molecular_weight
        
        # Estimate chamber temperature based on engine type and chamber pressure
        if self.engine_type == "hydrolox":
            self.ideal_chamber_temp_K = 3500.0
        elif self.engine_type == "kerolox":
            self.ideal_chamber_temp_K = 3300.0
        elif self.engine_type == "methalox":
            self.ideal_chamber_temp_K = 3400.0
        elif self.engine_type == "ion":
            self.ideal_chamber_temp_K = 500.0
        else:
            self.ideal_chamber_temp_K = 3000.0
            
        # Calculate characteristic velocity
        self.c_star = math.sqrt(self.gamma * self.R_specific * self.ideal_chamber_temp_K) / (
            self.gamma * math.sqrt((2 / (self.gamma + 1))**((self.gamma + 1)/(self.gamma - 1))))
            
        # Calculate throat area based on max thrust
        self.mass_flow_max = self.thrust_max_N / (self.isp_vacuum_s * self.g0)
        self.throat_area = self.mass_flow_max * self.c_star / self.chamber_pressure_Pa
        
        # Calculate exit area
        self.exit_area = self.throat_area * self.expansion_ratio
        
        # Calculate exit Mach number (from expansion ratio)
        # Solve iteratively for exit Mach number
        M_exit = 2.0  # Initial guess
        for _ in range(10):  # Usually converges quickly
            term1 = (2 / (self.gamma + 1)) * (1 + ((self.gamma - 1) / 2) * M_exit**2)
            term2 = (self.gamma + 1) / (2 * (self.gamma - 1))
            er_calc = (1 / M_exit) * term1**term2
            if abs(er_calc - self.expansion_ratio) < 0.01:
                break
            M_exit += 0.1
        self.exit_mach = M_exit
        
    def calculate_performance(self, 
                             throttle_pct: float, 
                             ambient_pressure_Pa: float = 0.0,
                             ambient_temp_K: float = 293.15) -> Dict[str, float]:
        """
        Calculate engine performance at given throttle setting and ambient conditions.
        
        Args:
            throttle_pct: Throttle setting (0-100%)
            ambient_pressure_Pa: Ambient pressure in Pa
            ambient_temp_K: Ambient temperature in K
            
        Returns:
            Dictionary of performance parameters
        """
        # Limit throttle to valid range
        throttle = max(0.0, min(100.0, throttle_pct)) / 100.0
        
        # Calculate chamber pressure at current throttle
        chamber_pressure = self.chamber_pressure_Pa * throttle
        
        # Calculate chamber temperature (simplified model)
        chamber_temp = self.ideal_chamber_temp_K * (0.8 + 0.2 * throttle)
        
        # Calculate exit pressure
        exit_pressure = chamber_pressure * (1 + (self.gamma - 1) / 2 * self.exit_mach**2)**(
            -self.gamma / (self.gamma - 1))
        
        # Calculate thrust coefficient
        term1 = self.gamma * math.sqrt((2 / (self.gamma + 1))**((self.gamma + 1)/(self.gamma - 1)))
        term2 = math.sqrt(1 - (exit_pressure / chamber_pressure)**((self.gamma - 1) / self.gamma))
        term3 = (exit_pressure - ambient_pressure_Pa) * self.exit_area / (chamber_pressure * self.throat_area)
        thrust_coef = term1 * term2 + term3
        
        # Calculate actual thrust
        thrust = chamber_pressure * self.throat_area * thrust_coef
        
        # Calculate actual Isp
        isp = thrust / (self.mass_flow_max * throttle * self.g0)
        
        # Calculate mass flow rate
        mass_flow = self.mass_flow_max * throttle
        
        # Calculate fuel and oxidizer flow rates (for bipropellant)
        if self.engine_type in ["bipropellant", "hydrolox", "kerolox", "methalox", "hypergolic"]:
            oxidizer_flow = mass_flow * self.mixture_ratio / (1 + self.mixture_ratio)
            fuel_flow = mass_flow / (1 + self.mixture_ratio)
        else:
            oxidizer_flow = 0.0
            fuel_flow = mass_flow
            
        # Calculate exhaust velocity
        exhaust_velocity = isp * self.g0
        
        # Store current state
        self.current_thrust_N = thrust
        self.current_isp_s = isp
        self.current_mass_flow_kgps = mass_flow
        self.current_chamber_temp_K = chamber_temp
        self.current_exit_pressure_Pa = exit_pressure
        
        # Record performance data
        performance = {
            'thrust_N': thrust,
            'isp_s': isp,
            'mass_flow_kgps': mass_flow,
            'chamber_pressure_Pa': chamber_pressure,
            'chamber_temp_K': chamber_temp,
            'exit_pressure_Pa': exit_pressure,
            'exit_velocity_mps': exhaust_velocity,
            'fuel_flow_kgps': fuel_flow,
            'oxidizer_flow_kgps': oxidizer_flow,
            'ambient_pressure_Pa': ambient_pressure_Pa,
            'throttle_pct': throttle_pct
        }
        
        self.performance_history.append(performance)
        return performance
        
    def calculate_altitude_compensation(self, altitude_m: float) -> Dict[str, float]:
        """
        Calculate performance adjustments for a given altitude.
        
        Args:
            altitude_m: Current altitude in meters
            
        Returns:
            Performance at the specified altitude
        """
        # Simple atmospheric model
        if altitude_m < 0:
            altitude_m = 0
            
        # Approximate atmospheric pressure based on altitude
        if altitude_m < 11000:  # Troposphere
            temp_K = 288.15 - 0.00649 * altitude_m
            pressure_ratio = (temp_K / 288.15) ** 5.256
            ambient_pressure_Pa = 101325.0 * pressure_ratio
        elif altitude_m < 20000:  # Lower Stratosphere
            temp_K = 216.65
            ambient_pressure_Pa = 22632.0 * math.exp(-0.00015769 * (altitude_m - 11000))
        elif altitude_m < 32000:  # Upper Stratosphere
            temp_K = 216.65 + 0.001 * (altitude_m - 20000)
            pressure_ratio = (temp_K / 216.65) ** -34.163
            ambient_pressure_Pa = 5474.9 * pressure_ratio
        else:  # Mesosphere and above
            ambient_pressure_Pa = 0.0
            temp_K = 240.0
            
        # Calculate performance at this ambient pressure
        return self.calculate_performance(100.0, ambient_pressure_Pa, temp_K)
        
    def calculate_thrust_vector(self, 
                               gimbal_angle_y: float = 0.0, 
                               gimbal_angle_z: float = 0.0) -> Tuple[float, float, float]:
        """
        Calculate thrust vector components based on gimbal angles.
        
        Args:
            gimbal_angle_y: Pitch gimbal angle in degrees
            gimbal_angle_z: Yaw gimbal angle in degrees
            
        Returns:
            Tuple of (thrust_x, thrust_y, thrust_z) components in Newtons
        """
        # Convert angles to radians
        y_rad = math.radians(gimbal_angle_y)
        z_rad = math.radians(gimbal_angle_z)
        
        # Calculate thrust components
        thrust_x = self.current_thrust_N * math.cos(y_rad) * math.cos(z_rad)
        thrust_y = self.current_thrust_N * math.sin(y_rad)
        thrust_z = self.current_thrust_N * math.cos(y_rad) * math.sin(z_rad)
        
        return (thrust_x, thrust_y, thrust_z)