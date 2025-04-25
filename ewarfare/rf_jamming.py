import numpy as np
from typing import List, Dict, Any, Optional

class RFJammer:
    """
    Simulates and controls RF jamming across multiple frequency bands.
    Supports barrage, spot, sweep, and smart jamming modes.
    """
    def __init__(self, bands: List[Dict[str, Any]]):
        """
        bands: List of dicts with 'name', 'center_freq_Hz', 'bandwidth_Hz', 'power_W'
        """
        self.bands = bands
        self.active_mode = 'barrage'  # default mode
        self.target_freqs: Optional[List[float]] = None
        self.sweep_rate_Hz = 1e6
        self.smart_targets: Optional[List[float]] = None

    def set_mode(self, mode: str, **kwargs):
        assert mode in ['barrage', 'spot', 'sweep', 'smart']
        self.active_mode = mode
        if mode == 'spot':
            self.target_freqs = kwargs.get('target_freqs')
        elif mode == 'sweep':
            self.sweep_rate_Hz = kwargs.get('sweep_rate_Hz', self.sweep_rate_Hz)
        elif mode == 'smart':
            self.smart_targets = kwargs.get('smart_targets')

    def generate_jamming_signal(self, duration_s: float, sample_rate: float) -> Dict[str, np.ndarray]:
        t = np.arange(0, duration_s, 1/sample_rate)
        signals = {}
        for band in self.bands:
            if self.active_mode == 'barrage':
                # White noise across the band
                signals[band['name']] = np.random.normal(0, np.sqrt(band['power_W']), len(t))
            elif self.active_mode == 'spot' and self.target_freqs:
                # Jam only at specific frequencies
                sig = np.zeros(len(t))
                for f in self.target_freqs:
                    sig += np.sqrt(band['power_W']) * np.sin(2 * np.pi * f * t)
                signals[band['name']] = sig
            elif self.active_mode == 'sweep':
                # Sweep a tone across the band
                f_start = band['center_freq_Hz'] - band['bandwidth_Hz']/2
                f_end = band['center_freq_Hz'] + band['bandwidth_Hz']/2
                sweep_freqs = np.linspace(f_start, f_end, len(t))
                sig = np.sqrt(band['power_W']) * np.sin(2 * np.pi * sweep_freqs * t)
                signals[band['name']] = sig
            elif self.active_mode == 'smart' and self.smart_targets:
                # Jam only detected threat frequencies
                sig = np.zeros(len(t))
                for f in self.smart_targets:
                    sig += np.sqrt(band['power_W']) * np.sin(2 * np.pi * f * t)
                signals[band['name']] = sig
            else:
                signals[band['name']] = np.zeros(len(t))
        return signals
