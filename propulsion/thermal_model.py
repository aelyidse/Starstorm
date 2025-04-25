from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import math

class FuelConsumptionModel:
    """
    Models fuel consumption with variable efficiency based on operating conditions.
    Accounts for throttle settings, environmental factors, and engine characteristics.
    """
    def __init__(self, 
                 base_efficiency: float = 0.85,
                 engine_type: str = "bipropellant",
                 efficiency_curve: Optional[Dict[float, float]] = None):
        """
        Initialize fuel consumption model.
        
        Args:
            base_efficiency: Baseline efficiency at optimal conditions
            engine_type: Engine type (bipropellant, monopropellant, ion, etc.)
            efficiency_curve: Optional mapping of throttle levels to efficiency multipliers
        """
        self.base_efficiency = base_efficiency
        self.engine_type = engine_type
        
        # Default efficiency curve if none provided (throttle_pct: efficiency_multiplier)
        self.efficiency_curve = efficiency_curve or {
            0.1: 0.65,  # Low throttle is inefficient
            0.2: 0.80,
            0.4: 0.90,
            0.6: 0.95,
            0.8: 1.0,
            1.0: 0.98   # Slight inefficiency at max throttle
        }
        
        # Temperature effects on efficiency (temp_K: efficiency_multiplier)
        self.temp_efficiency = {
            500: 0.85,   # Cold start
            1000: 0.92,
            2000: 1.0,   # Optimal temperature
            3000: 0.97,
            3500: 0.90   # Overheating
        }
        
        # Pressure effects on efficiency (pressure_ratio: efficiency_multiplier)
        # pressure_ratio = chamber_pressure / ambient_pressure
        self.pressure_efficiency = {
            10: 0.88,
            50: 0.94,
            100: 0.98,
            500: 1.0,
            1000: 0.99
        }
        
    def get_throttle_efficiency(self, throttle_pct: float) -> float:
        """Calculate efficiency multiplier based on throttle setting"""
        throttle = throttle_pct / 100.0
        
        # Find the closest throttle points in our curve
        keys = sorted(self.efficiency_curve.keys())
        
        if throttle <= keys[0]:
            return self.efficiency_curve[keys[0]]
        if throttle >= keys[-1]:
            return self.efficiency_curve[keys[-1]]
            
        # Linear interpolation between points
        for i in range(1, len(keys)):
            if throttle < keys[i]:
                k0, k1 = keys[i-1], keys[i]
                e0, e1 = self.efficiency_curve[k0], self.efficiency_curve[k1]
                return e0 + (e1 - e0) * (throttle - k0) / (k1 - k0)
                
        return 1.0  # Fallback
        
    def get_temperature_efficiency(self, chamber_temp_K: float) -> float:
        """Calculate efficiency multiplier based on chamber temperature"""
        keys = sorted(self.temp_efficiency.keys())
        
        if chamber_temp_K <= keys[0]:
            return self.temp_efficiency[keys[0]]
        if chamber_temp_K >= keys[-1]:
            return self.temp_efficiency[keys[-1]]
            
        # Linear interpolation
        for i in range(1, len(keys)):
            if chamber_temp_K < keys[i]:
                k0, k1 = keys[i-1], keys[i]
                e0, e1 = self.temp_efficiency[k0], self.temp_efficiency[k1]
                return e0 + (e1 - e0) * (chamber_temp_K - k0) / (k1 - k0)
                
        return 1.0  # Fallback
        
    def get_pressure_efficiency(self, chamber_pressure_Pa: float, ambient_pressure_Pa: float) -> float:
        """Calculate efficiency multiplier based on pressure ratio"""
        if ambient_pressure_Pa < 1.0:
            ambient_pressure_Pa = 1.0  # Prevent division by zero
            
        pressure_ratio = chamber_pressure_Pa / ambient_pressure_Pa
        keys = sorted(self.pressure_efficiency.keys())
        
        if pressure_ratio <= keys[0]:
            return self.pressure_efficiency[keys[0]]
        if pressure_ratio >= keys[-1]:
            return self.pressure_efficiency[keys[-1]]
            
        # Linear interpolation
        for i in range(1, len(keys)):
            if pressure_ratio < keys[i]:
                k0, k1 = keys[i-1], keys[i]
                e0, e1 = self.pressure_efficiency[k0], self.pressure_efficiency[k1]
                return e0 + (e1 - e0) * (pressure_ratio - k0) / (k1 - k0)
                
        return 1.0  # Fallback
    
    def calculate_consumption(self, 
                             thrust_N: float,
                             isp_s: float,
                             throttle_pct: float,
                             chamber_temp_K: float,
                             chamber_pressure_Pa: float,
                             ambient_pressure_Pa: float,
                             dt: float) -> Dict[str, Any]:
        """
        Calculate fuel consumption with variable efficiency.
        
        Args:
            thrust_N: Current thrust in Newtons
            isp_s: Specific impulse in seconds
            throttle_pct: Throttle percentage (0-100)
            chamber_temp_K: Chamber temperature in Kelvin
            chamber_pressure_Pa: Chamber pressure in Pascals
            ambient_pressure_Pa: Ambient pressure in Pascals
            dt: Time step in seconds
            
        Returns:
            Dictionary with consumption data
        """
        # Standard gravity
        g0 = 9.80665  # m/s²
        
        # Calculate base mass flow rate (kg/s)
        base_mdot = thrust_N / (isp_s * g0)
        
        # Calculate efficiency multipliers
        throttle_eff = self.get_throttle_efficiency(throttle_pct)
        temp_eff = self.get_temperature_efficiency(chamber_temp_K)
        pressure_eff = self.get_pressure_efficiency(chamber_pressure_Pa, ambient_pressure_Pa)
        
        # Combined efficiency
        total_efficiency = self.base_efficiency * throttle_eff * temp_eff * pressure_eff
        
        # Adjust mass flow rate based on efficiency
        actual_mdot = base_mdot / total_efficiency
        
        # Calculate fuel used in this time step
        fuel_used = actual_mdot * dt
        
        # Calculate specific fuel consumption (kg/N·s)
        sfc = actual_mdot / thrust_N if thrust_N > 0 else 0.0
        
        return {
            "mass_flow_rate_kgps": actual_mdot,
            "fuel_used_kg": fuel_used,
            "efficiency": total_efficiency,
            "specific_fuel_consumption": sfc,
            "efficiency_factors": {
                "throttle": throttle_eff,
                "temperature": temp_eff,
                "pressure": pressure_eff
            }
        }

class ThermalModel:
    """
    Thermal model for propulsion systems.
    Simulates heat transfer, cooling, and thermal stresses in engine components.
    """
    def __init__(self, 
                 chamber_wall_thickness_mm: float = 5.0,
                 chamber_material: str = "inconel",
                 cooling_type: str = "regenerative",
                 coolant_flow_rate_kgps: float = 2.0):
        """
        Initialize thermal model with engine parameters.
        
        Args:
            chamber_wall_thickness_mm: Chamber wall thickness in mm
            chamber_material: Chamber material (inconel, copper, etc.)
            cooling_type: Cooling method (regenerative, film, radiative)
            coolant_flow_rate_kgps: Coolant flow rate in kg/s
        """
        self.chamber_wall_thickness = chamber_wall_thickness_mm / 1000.0  # Convert to meters
        self.chamber_material = chamber_material
        self.cooling_type = cooling_type
        self.coolant_flow_rate = coolant_flow_rate_kgps
        
        # Material properties
        self.material_properties = {
            "inconel": {
                "thermal_conductivity": 12.0,  # W/m·K
                "density": 8400.0,  # kg/m³
                "specific_heat": 440.0,  # J/kg·K
                "melting_point": 1600.0,  # K
                "thermal_expansion": 13.0e-6  # 1/K
            },
            "copper": {
                "thermal_conductivity": 400.0,
                "density": 8960.0,
                "specific_heat": 385.0,
                "melting_point": 1360.0,
                "thermal_expansion": 17.0e-6
            },
            "stainless_steel": {
                "thermal_conductivity": 16.0,
                "density": 8000.0,
                "specific_heat": 500.0,
                "melting_point": 1700.0,
                "thermal_expansion": 17.3e-6
            }
        }
        
        # Default to inconel if material not found
        if chamber_material not in self.material_properties:
            self.chamber_material = "inconel"
            
        # Current state
        self.current_wall_temp = 300.0  # K
        self.current_coolant_temp = 290.0  # K
        self.thermal_stress = 0.0  # MPa
        self.heat_flux = 0.0  # MW/m²
        
        # Cooling efficiency factors
        self.cooling_efficiency = {
            "regenerative": 0.85,
            "film": 0.70,
            "radiative": 0.40,
            "ablative": 0.60
        }
        
    def calculate_heat_transfer(self, 
                               chamber_temp_K: float, 
                               chamber_pressure_Pa: float,
                               throat_area_m2: float,
                               burn_duration_s: float) -> Dict[str, float]:
        """
        Calculate heat transfer and thermal effects.
        
        Args:
            chamber_temp_K: Combustion chamber temperature in K
            chamber_pressure_Pa: Chamber pressure in Pa
            throat_area_m2: Throat area in m²
            burn_duration_s: Engine burn duration in seconds
            
        Returns:
            Dictionary of thermal parameters
        """
        # Calculate chamber inner surface area (simplified as cylinder)
        chamber_diameter = math.sqrt(throat_area_m2 * 4 / math.pi) * 2.5  # Estimate chamber diameter
        chamber_length = chamber_diameter * 1.5  # Typical L/D ratio
        chamber_surface_area = math.pi * chamber_diameter * chamber_length
        
        # Calculate heat flux using Bartz equation (simplified)
        # q = h * (Taw - Tw)
        # where h is heat transfer coefficient, Taw is adiabatic wall temp, Tw is wall temp
        
        # Simplified heat transfer coefficient (Bartz)
        gas_viscosity = 0.00007  # Pa·s (approximate for hot combustion gases)
        gas_conductivity = 0.5  # W/m·K
        pr = 0.8  # Prandtl number
        
        # Simplified Bartz coefficient
        h_g = 0.026 * (chamber_pressure_Pa**0.8) * (chamber_diameter**-0.2) * (gas_conductivity**0.6) * (
            gas_viscosity**0.2) * (pr**0.4)
        
        # Adiabatic wall temperature (recovery factor ~ 0.9)
        T_aw = 0.9 * chamber_temp_K
        
        # Heat flux
        q = h_g * (T_aw - self.current_wall_temp)
        self.heat_flux = q / 1e6  # Convert to MW/m²
        
        # Apply cooling efficiency
        cooling_factor = self.cooling_efficiency.get(self.cooling_type, 0.5)
        effective_heat_flux = q * (1.0 - cooling_factor)
        
        # Calculate wall temperature rise
        material = self.material_properties[self.chamber_material]
        thermal_diffusivity = material["thermal_conductivity"] / (
            material["density"] * material["specific_heat"])
        
        # Simplified 1D heat conduction
        delta_T = effective_heat_flux * self.chamber_wall_thickness / material["thermal_conductivity"]
        
        # Update wall temperature (with time factor)
        time_factor = min(1.0, burn_duration_s / 10.0)  # Approach steady state after ~10s
        self.current_wall_temp += delta_T * time_factor
        
        # Calculate thermal stress
        thermal_gradient = delta_T / self.chamber_wall_thickness
        self.thermal_stress = material["thermal_expansion"] * material["thermal_conductivity"] * thermal_gradient
        
        # Calculate coolant temperature rise (if using active cooling)
        coolant_temp_rise = 0.0
        if self.cooling_type in ["regenerative", "film"]:
            # Q = m * cp * ΔT
            coolant_specific_heat = 4200.0  # J/kg·K (water)
            heat_absorbed = q * chamber_surface_area * cooling_factor
            coolant_temp_rise = heat_absorbed / (self.coolant_flow_rate * coolant_specific_heat)
            self.current_coolant_temp += coolant_temp_rise
        
        # Calculate safety margin
        temp_margin = material["melting_point"] - self.current_wall_temp
        safety_factor = temp_margin / material["melting_point"]
        
        return {
            "wall_temperature_K": self.current_wall_temp,
            "coolant_temperature_K": self.current_coolant_temp,
            "heat_flux_MW_m2": self.heat_flux,
            "thermal_stress_MPa": self.thermal_stress,
            "temperature_margin_K": temp_margin,
            "safety_factor": safety_factor,
            "is_safe": safety_factor > 0.2  # 20% safety margin
        }