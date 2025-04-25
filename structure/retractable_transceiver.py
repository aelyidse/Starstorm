"""
RetractableTransceiver: Models a transceiver that can be extended/retracted for communication and jamming.
"""
class RetractableTransceiver:
    """
    Represents a wireless transceiver that can be retracted/extended for communication and jamming.
    Enhanced with frequency, power, antenna gain, diagnostics, and deployment checks.
    """
    def __init__(self, supported_freqs=("UHF", "VHF", "X", "Ka"), max_power_watt: float = 100.0, antenna_gain_dbi: float = 8.0):
        self.state = 'retracted'
        self.mode = 'idle'  # 'idle', 'comm', 'jam'
        self.frequency_band = None
        self.supported_freqs = supported_freqs
        self.tx_power_watt = 0.0
        self.max_power_watt = max_power_watt
        self.antenna_gain_dbi = antenna_gain_dbi
        self.diagnostics = {'last_check': None, 'status': 'OK'}

    def extend(self):
        self.state = 'extended'
        return 'Transceiver extended.'

    def retract(self):
        self.state = 'retracted'
        self.tx_power_watt = 0.0
        self.frequency_band = None
        self.mode = 'idle'
        return 'Transceiver retracted.'

    def set_mode(self, mode: str):
        if self.state != 'extended':
            return 'Cannot set mode: transceiver not extended.'
        if mode in ['idle', 'comm', 'jam']:
            self.mode = mode
            return f"Transceiver mode set to {mode}."
        return 'Invalid mode.'

    def set_frequency(self, band: str):
        if band not in self.supported_freqs:
            return f"Unsupported frequency band: {band}"
        self.frequency_band = band
        return f"Frequency band set to {band}."

    def set_power(self, watt: float):
        if watt < 0 or watt > self.max_power_watt:
            return f"Power out of range (0-{self.max_power_watt} W)."
        self.tx_power_watt = watt
        return f"Transmit power set to {watt} W."

    def jam(self, target_band: str, duration_s: float):
        if self.state != 'extended' or self.mode != 'jam':
            return 'Transceiver must be extended and in jam mode.'
        if target_band not in self.supported_freqs:
            return f"Cannot jam unsupported band: {target_band}"
        # Simulate jamming
        return f"Jamming {target_band} for {duration_s} seconds at {self.tx_power_watt} W."

    def run_diagnostics(self):
        # Simulate a check
        import random, time
        self.diagnostics['last_check'] = time.time()
        self.diagnostics['status'] = 'OK' if random.random() > 0.02 else 'FAULT'
        return self.diagnostics

    def get_status(self):
        return {
            'state': self.state,
            'mode': self.mode,
            'frequency_band': self.frequency_band,
            'tx_power_watt': self.tx_power_watt,
            'antenna_gain_dbi': self.antenna_gain_dbi,
            'supported_freqs': self.supported_freqs,
            'diagnostics': self.diagnostics,
        }
