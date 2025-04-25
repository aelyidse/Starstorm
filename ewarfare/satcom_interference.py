import numpy as np
from typing import List, Dict, Any, Optional

class SatcomInterferenceAlgorithms:
    """
    Algorithms for interfering with satellite communication links (uplink, downlink, crosslink).
    Supports spot, barrage, and protocol-aware interference in GEO, MEO, and LEO bands.
    """
    def __init__(self, bands: List[Dict[str, Any]]):
        """
        bands: List of dicts with 'name', 'center_freq_Hz', 'bandwidth_Hz', 'power_W'
        """
        self.bands = bands
        self.active_mode = 'spot'  # default mode
        self.target_sat_freqs: Optional[List[float]] = None
        self.protocol = None
        self.sweep_rate_Hz = 1e6

    def set_mode(self, mode: str, **kwargs):
        assert mode in ['spot', 'barrage', 'sweep', 'protocol-aware']
        self.active_mode = mode
        if mode == 'spot':
            self.target_sat_freqs = kwargs.get('target_sat_freqs')
        elif mode == 'protocol-aware':
            self.protocol = kwargs.get('protocol')
        elif mode == 'sweep':
            self.sweep_rate_Hz = kwargs.get('sweep_rate_Hz', self.sweep_rate_Hz)

    def interfere(self, duration_s: float, sample_rate: float) -> Dict[str, np.ndarray]:
        t = np.arange(0, duration_s, 1/sample_rate)
        signals = {}
        for band in self.bands:
            if self.active_mode == 'barrage':
                # White noise across the band
                signals[band['name']] = np.random.normal(0, np.sqrt(band['power_W']), len(t))
            elif self.active_mode == 'spot' and self.target_sat_freqs:
                # Jam only at specific satellite frequencies
                sig = np.zeros(len(t))
                for f in self.target_sat_freqs:
                    sig += np.sqrt(band['power_W']) * np.sin(2 * np.pi * f * t)
                signals[band['name']] = sig
            elif self.active_mode == 'sweep':
                # Sweep a tone across the band
                f_start = band['center_freq_Hz'] - band['bandwidth_Hz']/2
                f_end = band['center_freq_Hz'] + band['bandwidth_Hz']/2
                sweep_freqs = np.linspace(f_start, f_end, len(t))
                sig = np.sqrt(band['power_W']) * np.sin(2 * np.pi * sweep_freqs * t)
                signals[band['name']] = sig
            elif self.active_mode == 'protocol-aware' and self.protocol:
                # Protocol-specific interference (e.g., BPSK, QPSK)
                if self.protocol == 'BPSK':
                    bits = np.random.choice([-1, 1], size=len(t))
                    sig = np.sin(2 * np.pi * band['center_freq_Hz'] * t) * bits
                elif self.protocol == 'QPSK':
                    bits_i = np.random.choice([-1, 1], size=len(t))
                    bits_q = np.random.choice([-1, 1], size=len(t))
                    sig = (np.sin(2 * np.pi * band['center_freq_Hz'] * t) * bits_i +
                           np.cos(2 * np.pi * band['center_freq_Hz'] * t) * bits_q)
                else:
                    sig = np.random.normal(0, 1, len(t))
                signals[band['name']] = sig * np.sqrt(band['power_W'])
            else:
                signals[band['name']] = np.zeros(len(t))
        return signals
