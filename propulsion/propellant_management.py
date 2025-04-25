from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import math

class PropellantTank:
    """
    Models a propellant tank with thermal and pressure characteristics.
    """
    def __init__(self, 
                 propellant_type: str,
                 capacity_kg: float,
                 initial_fill_pct: float = 100.0,
                 initial_pressure_MPa: float = 0.3,
                 initial_temp_K: float = 293.15):
        """
        Initialize propellant tank.
        
        Args:
            propellant_type: Type of propellant (e.g., "LOX", "LH2", "RP-1")
            capacity_kg: Maximum capacity in kg
            initial_fill_pct: Initial fill percentage (0-100)
            initial_pressure_MPa: Initial pressure in MPa
            initial_temp_K: Initial temperature in Kelvin
        """
        self.propellant_type = propellant_type
        self.capacity_kg = capacity_kg
        self.current_mass_kg = capacity_kg * (initial_fill_pct / 100.0)
        self.pressure_MPa = initial_pressure_MPa
        self.temperature_K = initial_temp_K
        
        # Set properties based on propellant type
        self._set_propellant_properties()
        
        # History tracking
        self.history = []
        self._record_state()
        
    def _set_propellant_properties(self):
        """Set physical properties based on propellant type."""
        properties = {
            "LOX": {
                "density_kgm3": 1141.0,
                "boiling_point_K": 90.2,
                "specific_heat_JkgK": 918.0,
                "thermal_conductivity_WmK": 0.152,
                "is_cryogenic": True
            },
            "LH2": {
                "density_kgm3": 70.8,
                "boiling_point_K": 20.3,
                "specific_heat_JkgK": 9690.0,
                "thermal_conductivity_WmK": 0.099,
                "is_cryogenic": True
            },
            "RP-1": {
                "density_kgm3": 820.0,
                "boiling_point_K": 490.0,
                "specific_heat_JkgK": 2010.0,
                "thermal_conductivity_WmK": 0.145,
                "is_cryogenic": False
            },
            "N2O4": {
                "density_kgm3": 1440.0,
                "boiling_point_K": 294.3,
                "specific_heat_JkgK": 1550.0,
                "thermal_conductivity_WmK": 0.13,
                "is_cryogenic": False
            },
            "MMH": {
                "density_kgm3": 870.0,
                "boiling_point_K": 360.7,
                "specific_heat_JkgK": 2930.0,
                "thermal_conductivity_WmK": 0.17,
                "is_cryogenic": False
            }
        }
        
        # Set default properties if propellant type not found
        if self.propellant_type not in properties:
            self.properties = properties["RP-1"]
            self.properties["name"] = "Generic"
        else:
            self.properties = properties[self.propellant_type]
            self.properties["name"] = self.propellant_type
            
        # Calculate volume based on density
        self.volume_m3 = self.capacity_kg / self.properties["density_kgm3"]
        
    def withdraw(self, requested_mass_kg: float) -> float:
        """
        Withdraw propellant from tank.
        
        Args:
            requested_mass_kg: Requested mass to withdraw in kg
            
        Returns:
            Actual mass withdrawn in kg
        """
        available = min(requested_mass_kg, self.current_mass_kg)
        self.current_mass_kg -= available
        
        # Update pressure (simplified model)
        fill_ratio = self.current_mass_kg / self.capacity_kg
        if fill_ratio > 0:
            # Simple linear pressure model based on fill level
            self.pressure_MPa = self.pressure_MPa * (0.5 + 0.5 * fill_ratio)
        else:
            self.pressure_MPa = 0.0
            
        self._record_state()
        return available
        
    def update_thermal(self, dt: float, ambient_temp_K: float = 293.15) -> float:
        """
        Update thermal state and calculate boiloff for cryogenic propellants.
        
        Args:
            dt: Time step in seconds
            ambient_temp_K: Ambient temperature in Kelvin
            
        Returns:
            Mass of boiloff in kg
        """
        boiloff = 0.0
        
        # Only apply thermal effects to cryogenic propellants
        if self.properties.get("is_cryogenic", False) and self.current_mass_kg > 0:
            # Simple thermal model
            temp_difference = ambient_temp_K - self.temperature_K
            
            # Heat transfer rate (simplified)
            heat_transfer_coefficient = 0.05  # W/m²K
            surface_area = math.pow(self.volume_m3, 2/3) * 6  # Approximation for a cube
            
            # Heat transfer
            heat_transfer_rate = heat_transfer_coefficient * surface_area * temp_difference
            
            # Temperature change
            if self.current_mass_kg > 0:
                temp_change = (heat_transfer_rate * dt) / (self.current_mass_kg * self.properties["specific_heat_JkgK"])
                self.temperature_K += temp_change
                
                # Boiloff if temperature exceeds boiling point
                if self.temperature_K > self.properties["boiling_point_K"]:
                    # Calculate boiloff (simplified)
                    latent_heat = 200000.0  # J/kg (approximate)
                    excess_temp = self.temperature_K - self.properties["boiling_point_K"]
                    
                    # Energy available for boiloff
                    boiloff_energy = excess_temp * self.current_mass_kg * self.properties["specific_heat_JkgK"]
                    boiloff = min(boiloff_energy / latent_heat, self.current_mass_kg)
                    
                    # Remove boiloff mass
                    self.current_mass_kg -= boiloff
                    
                    # Reset temperature to boiling point
                    self.temperature_K = self.properties["boiling_point_K"]
        
        self._record_state()
        return boiloff
        
    def _record_state(self):
        """Record current state to history."""
        state = {
            "mass_kg": self.current_mass_kg,
            "fill_pct": (self.current_mass_kg / self.capacity_kg) * 100.0,
            "pressure_MPa": self.pressure_MPa,
            "temperature_K": self.temperature_K
        }
        self.history.append(state)
        
    def get_state(self) -> Dict[str, Any]:
        """Get current tank state."""
        return {
            "propellant_type": self.propellant_type,
            "current_mass_kg": self.current_mass_kg,
            "capacity_kg": self.capacity_kg,
            "fill_percentage": (self.current_mass_kg / self.capacity_kg) * 100.0,
            "pressure_MPa": self.pressure_MPa,
            "temperature_K": self.temperature_K,
            "is_cryogenic": self.properties.get("is_cryogenic", False),
            "volume_m3": self.volume_m3
        }


class PropellantManagementSystem:
    """
    Manages propellant flow, pressurization, and thermal control for rocket engines.
    """
    def __init__(self, tanks: List[PropellantTank], mixture_ratio: float = 2.1):
        """
        Initialize propellant management system.
        
        Args:
            tanks: List of propellant tanks
            mixture_ratio: Oxidizer to fuel ratio (for bipropellant engines)
        """
        self.tanks = tanks
        self.mixture_ratio = mixture_ratio
        self.flow_rates = [0.0] * len(tanks)
        self.valve_positions = [0.0] * len(tanks)
        self.pressurization_active = [False] * len(tanks)
        
        # Performance history
        self.consumption_history = []
        
    def set_valve_positions(self, positions: List[float]) -> bool:
        """
        Set valve positions for each tank.
        
        Args:
            positions: List of valve positions (0.0-1.0) for each tank
            
        Returns:
            Success status
        """
        if len(positions) != len(self.tanks):
            return False
            
        self.valve_positions = [max(0.0, min(1.0, p)) for p in positions]
        return True
        
    def set_pressurization(self, tank_indices: List[int], active: bool) -> bool:
        """
        Set pressurization state for specified tanks.
        
        Args:
            tank_indices: List of tank indices to modify
            active: Whether pressurization should be active
            
        Returns:
            Success status
        """
        for idx in tank_indices:
            if idx < 0 or idx >= len(self.tanks):
                return False
            self.pressurization_active[idx] = active
        return True
        
    def calculate_flow_rates(self, engine_demand_kgps: float) -> Dict[str, Any]:
        """
        Calculate flow rates based on engine demand and valve positions.
        
        Args:
            engine_demand_kgps: Total propellant demand in kg/s
            
        Returns:
            Flow rate information
        """
        # Identify fuel and oxidizer tanks
        oxidizer_indices = []
        fuel_indices = []
        
        for i, tank in enumerate(self.tanks):
            if tank.propellant_type in ["LOX", "N2O4"]:
                oxidizer_indices.append(i)
            else:
                fuel_indices.append(i)
                
        # Calculate required flow rates
        if len(oxidizer_indices) > 0 and len(fuel_indices) > 0:
            # Bipropellant mode
            total_oxidizer_demand = engine_demand_kgps * self.mixture_ratio / (1 + self.mixture_ratio)
            total_fuel_demand = engine_demand_kgps / (1 + self.mixture_ratio)
            
            # Distribute demand proportionally to valve positions
            oxidizer_valve_sum = sum(self.valve_positions[i] for i in oxidizer_indices)
            fuel_valve_sum = sum(self.valve_positions[i] for i in fuel_indices)
            
            # Reset flow rates
            self.flow_rates = [0.0] * len(self.tanks)
            
            # Set oxidizer flow rates
            if oxidizer_valve_sum > 0:
                for i in oxidizer_indices:
                    self.flow_rates[i] = total_oxidizer_demand * (self.valve_positions[i] / oxidizer_valve_sum)
            
            # Set fuel flow rates
            if fuel_valve_sum > 0:
                for i in fuel_indices:
                    self.flow_rates[i] = total_fuel_demand * (self.valve_positions[i] / fuel_valve_sum)
        else:
            # Monopropellant mode
            valve_sum = sum(self.valve_positions)
            if valve_sum > 0:
                for i in range(len(self.tanks)):
                    self.flow_rates[i] = engine_demand_kgps * (self.valve_positions[i] / valve_sum)
            else:
                self.flow_rates = [0.0] * len(self.tanks)
                
        return {
            "flow_rates_kgps": self.flow_rates,
            "total_flow_kgps": sum(self.flow_rates),
            "requested_flow_kgps": engine_demand_kgps
        }
        
    def update(self, dt: float, engine_demand_kgps: float, ambient_temp_K: float = 293.15) -> Dict[str, Any]:
        """
        Update propellant management system.
        
        Args:
            dt: Time step in seconds
            engine_demand_kgps: Engine propellant demand in kg/s
            ambient_temp_K: Ambient temperature in Kelvin
            
        Returns:
            Update results
        """
        # Calculate flow rates
        flow_info = self.calculate_flow_rates(engine_demand_kgps)
        
        # Withdraw propellant from tanks
        actual_flows = []
        for i, tank in enumerate(self.tanks):
            requested_mass = self.flow_rates[i] * dt
            actual_mass = tank.withdraw(requested_mass)
            actual_flows.append(actual_mass / dt)
            
            # Update thermal state
            boiloff = tank.update_thermal(dt, ambient_temp_K)
            
            # Apply pressurization effects if active
            if self.pressurization_active[i]:
                # Simplified pressurization model
                tank.pressure_MPa = min(tank.pressure_MPa * 1.01, 5.0)  # Limit to 5 MPa
                
        # Record consumption
        consumption = {
            "time_step": dt,
            "requested_flows_kgps": self.flow_rates,
            "actual_flows_kgps": actual_flows,
            "total_flow_kgps": sum(actual_flows),
            "tank_states": [tank.get_state() for tank in self.tanks]
        }
        self.consumption_history.append(consumption)
        
        return consumption
        
    def get_state(self) -> Dict[str, Any]:
        """Get current system state."""
        return {
            "tanks": [tank.get_state() for tank in self.tanks],
            "valve_positions": self.valve_positions,
            "flow_rates_kgps": self.flow_rates,
            "pressurization_active": self.pressurization_active,
            "mixture_ratio": self.mixture_ratio
        }


class PropellantSimulation:
    """
    Simulation for propellant management during rocket operation.
    """
    def __init__(self, 
                 propellant_system: PropellantManagementSystem,
                 engine_throttle_profile: List[Tuple[float, float]] = None):
        """
        Initialize propellant simulation.
        
        Args:
            propellant_system: Propellant management system
            engine_throttle_profile: List of (time_s, throttle_pct) tuples
        """
        self.propellant_system = propellant_system
        
        # Default throttle profile if none provided
        self.throttle_profile = engine_throttle_profile or [
            (0.0, 0.0),    # Start
            (5.0, 100.0),  # Full throttle at 5s
            (100.0, 100.0), # Maintain full throttle
            (105.0, 0.0)   # Shutdown at 105s
        ]
        
        # Simulation parameters
        self.time = 0.0
        self.max_thrust_N = 500000.0  # 500 kN
        self.isp_s = 320.0  # seconds
        self.g0 = 9.80665  # m/s²
        
        # Results storage
        self.results = []
        
    def get_throttle_at_time(self, time_s: float) -> float:
        """Get throttle percentage at a specific time."""
        if time_s <= self.throttle_profile[0][0]:
            return self.throttle_profile[0][1]
            
        if time_s >= self.throttle_profile[-1][0]:
            return self.throttle_profile[-1][1]
            
        # Find surrounding points and interpolate
        for i in range(1, len(self.throttle_profile)):
            if time_s <= self.throttle_profile[i][0]:
                t0, throttle0 = self.throttle_profile[i-1]
                t1, throttle1 = self.throttle_profile[i]
                
                # Linear interpolation
                fraction = (time_s - t0) / (t1 - t0)
                return throttle0 + fraction * (throttle1 - throttle0)
                
        return 0.0  # Fallback
        
    def run_simulation(self, duration_s: float, dt: float = 0.1, ambient_temp_K: float = 293.15) -> List[Dict[str, Any]]:
        """
        Run propellant simulation for specified duration.
        
        Args:
            duration_s: Simulation duration in seconds
            dt: Time step in seconds
            ambient_temp_K: Ambient temperature in Kelvin
            
        Returns:
            Simulation results
        """
        self.time = 0.0
        self.results = []
        
        while self.time < duration_s:
            # Get throttle at current time
            throttle_pct = self.get_throttle_at_time(self.time)
            
            # Calculate propellant demand
            max_flow_rate = self.max_thrust_N / (self.isp_s * self.g0)
            engine_demand = max_flow_rate * (throttle_pct / 100.0)
            
            # Update propellant system
            update_result = self.propellant_system.update(dt, engine_demand, ambient_temp_K)
            
            # Calculate actual thrust based on propellant flow
            actual_flow = update_result["total_flow_kgps"]
            actual_thrust = actual_flow * self.isp_s * self.g0
            
            # Record results
            result = {
                "time_s": self.time,
                "throttle_pct": throttle_pct,
                "requested_flow_kgps": engine_demand,
                "actual_flow_kgps": actual_flow,
                "requested_thrust_N": self.max_thrust_N * (throttle_pct / 100.0),
                "actual_thrust_N": actual_thrust,
                "tank_states": [tank.get_state() for tank in self.propellant_system.tanks]
            }
            self.results.append(result)
            
            # Advance time
            self.time += dt
            
        return self.results