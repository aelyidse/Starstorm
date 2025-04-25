import numpy as np
from typing import List, Dict, Any

class DisruptionWaveformGenerator:
    """
    Generates targeted disruption waveforms for electronic attack against enemy communications.
    Supports matched interference, protocol-aware jamming, and adaptive waveform synthesis.
    """
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def matched_interference(self, target_signal: np.ndarray, snr_db: float = 0.0) -> np.ndarray:
        # Generate interference matched to target signal (coherent jamming)
        power_target = np.mean(target_signal ** 2)
        power_jam = power_target / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(power_jam), len(target_signal))
        return target_signal + noise

    def protocol_aware_jam(self, carrier_freq: float, duration_s: float, modulation: str = 'psk', params: Dict[str, Any] = None) -> np.ndarray:
        # Generate a jamming waveform that mimics or disrupts a given protocol
        t = np.arange(0, duration_s, 1 / self.sample_rate)
        if modulation == 'psk':
            bits = np.random.choice([-1, 1], size=len(t))
            jam = np.sin(2 * np.pi * carrier_freq * t) * bits
        elif modulation == 'fm':
            freq_dev = params.get('freq_dev', 1e3) if params else 1e3
            mod_signal = np.cumsum(np.random.normal(0, 1, len(t)))
            jam = np.sin(2 * np.pi * (carrier_freq * t + freq_dev * mod_signal / self.sample_rate))
        else:
            jam = np.random.normal(0, 1, len(t))
        return jam

    def adaptive_waveform(self, threat_features: Dict[str, Any], duration_s: float) -> np.ndarray:
        # Generate a waveform that adapts to threat features (e.g., hopping, periodicity)
        carrier_freqs = threat_features.get('carrier_freqs', [1e6])
        modulation = threat_features.get('modulation', 'noise')
        period = threat_features.get('periodicity_s', 0.0)
        t = np.arange(0, duration_s, 1 / self.sample_rate)
        jam = np.zeros_like(t)
        for i, f in enumerate(carrier_freqs):
            if modulation == 'PSK/FHSS/noise-like' and period > 0:
                # Hopping jammer
                hop_len = int(period * self.sample_rate)
                for k in range(0, len(t), hop_len):
                    bit = np.random.choice([-1, 1])
                    jam[k:k + hop_len] += np.sin(2 * np.pi * f * t[k:k + hop_len]) * bit
            else:
                jam += np.sin(2 * np.pi * f * t)
        # Add noise for masking
        jam += np.random.normal(0, 0.5, len(t))
        return jam
