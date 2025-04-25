import numpy as np
from typing import Tuple, Optional

class LPISignalGenerator:
    """
    Generates low-probability-of-intercept (LPI) signals for stealth communications.
    Supports DSSS, noise-like waveforms, and power shaping for minimal detectability.
    """
    def __init__(self, sample_rate: float, spreading_factor: int = 128, seed: Optional[int] = None):
        self.sample_rate = sample_rate
        self.spreading_factor = spreading_factor
        self.seed = seed if seed is not None else np.random.randint(1, 1e6)
        self.prng = np.random.RandomState(self.seed)
        self.code = self.prng.choice([-1, 1], size=self.spreading_factor)

    def dsss_modulate(self, data_bits: np.ndarray) -> np.ndarray:
        # Direct Sequence Spread Spectrum (DSSS) modulation
        spread_data = np.repeat(data_bits, self.spreading_factor) * np.tile(self.code, len(data_bits))
        return spread_data

    def noise_like_waveform(self, length: int) -> np.ndarray:
        # Generate a noise-like (Gaussian) waveform
        return self.prng.normal(0, 1, length)

    def power_shape(self, signal: np.ndarray, shape: str = 'flat') -> np.ndarray:
        # Apply power shaping to minimize spectral features
        if shape == 'flat':
            return signal / np.max(np.abs(signal) + 1e-8)
        elif shape == 'tapered':
            window = np.hanning(len(signal))
            return signal * window / (np.max(np.abs(signal * window)) + 1e-8)
        else:
            raise ValueError(f"Unknown power shape: {shape}")

    def generate_lpi_signal(self, data_bits: np.ndarray, length: int = 1024, shape: str = 'tapered') -> np.ndarray:
        # Full LPI signal pipeline: DSSS + power shaping
        dsss = self.dsss_modulate(data_bits)
        if len(dsss) < length:
            dsss = np.pad(dsss, (0, length - len(dsss)), 'constant')
        else:
            dsss = dsss[:length]
        shaped = self.power_shape(dsss, shape=shape)
        return shaped
