"""
SolarPowerSystem: Models a deployable solar panel array for vehicle power supply.
"""
class SolarPowerSystem:
    """
    Represents a solar power system with deployable panels, battery storage, health, and diagnostics.
    """
    def __init__(self, max_output_watts: float = 500.0, battery_capacity_wh: float = 2000.0):
        self.max_output_watts = max_output_watts
        self.deployed = False
        self.current_output_watts = 0.0
        self.battery_capacity_wh = battery_capacity_wh
        self.battery_charge_wh = battery_capacity_wh * 0.8  # Start at 80%
        self.health = 1.0  # 1.0 = new, 0.0 = failed
        self.degradation_rate = 0.0001  # per hour
        self.fault = False
        self.last_sunlight_angle_deg = 90.0
        self.last_sunlight_fraction = 1.0

    def deploy(self):
        self.deployed = True
        return 'Solar panels deployed.'

    def retract(self):
        self.deployed = False
        self.current_output_watts = 0.0
        return 'Solar panels retracted.'

    def get_power_output(self, sunlight_fraction: float = 1.0, sunlight_angle_deg: float = 90.0, hours_elapsed: float = 0.0) -> float:
        if self.fault or self.health <= 0.0:
            self.current_output_watts = 0.0
            return 0.0
        # Sunlight angle: 90 = direct, 0 = edge-on
        angle_factor = max(0.0, np.sin(np.radians(sunlight_angle_deg)))
        output = self.max_output_watts * sunlight_fraction * angle_factor * self.health
        self.current_output_watts = output if self.deployed else 0.0
        self.last_sunlight_angle_deg = sunlight_angle_deg
        self.last_sunlight_fraction = sunlight_fraction
        # Battery charging
        if self.deployed and output > 0:
            self.battery_charge_wh = min(self.battery_capacity_wh, self.battery_charge_wh + output * hours_elapsed)
        # Degrade health
        self.health = max(0.0, self.health - self.degradation_rate * hours_elapsed)
        return self.current_output_watts

    def draw_power(self, watt: float, hours: float):
        # Draw from battery if not enough solar
        available = self.current_output_watts * hours
        deficit = max(0.0, watt * hours - available)
        if deficit > 0:
            if self.battery_charge_wh >= deficit:
                self.battery_charge_wh -= deficit
                return True
            else:
                self.battery_charge_wh = 0.0
                self.fault = True
                return False
        return True

    def induce_fault(self):
        self.fault = True
        return 'Solar panel fault induced.'

    def repair(self):
        self.fault = False
        self.health = min(1.0, self.health + 0.2)
        return 'Solar panel repaired.'

    def get_status(self):
        return {
            'deployed': self.deployed,
            'current_output_watts': self.current_output_watts,
            'battery_charge_wh': self.battery_charge_wh,
            'battery_capacity_wh': self.battery_capacity_wh,
            'health': self.health,
            'fault': self.fault,
            'last_sunlight_angle_deg': self.last_sunlight_angle_deg,
            'last_sunlight_fraction': self.last_sunlight_fraction,
        }
