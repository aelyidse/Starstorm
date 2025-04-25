import numpy as np
from typing import List, Tuple, Optional
import secrets

class FHSSController:
    """
    Implements Frequency-Hopping Spread Spectrum (FHSS) transmission algorithms for secure, anti-jam communications.
    Supports pseudorandom hop sequence generation, synchronization, and channel selection.
    """
    def __init__(self, center_freq_Hz: float, bandwidth_Hz: float, n_channels: int, hop_rate_Hz: float, seed: Optional[int] = None):
        self.center_freq_Hz = center_freq_Hz
        self.bandwidth_Hz = bandwidth_Hz
        self.n_channels = n_channels
        self.hop_rate_Hz = hop_rate_Hz
        self.seed = seed if seed is not None else secrets.randbits(32)
        self.channel_freqs = np.linspace(
            center_freq_Hz - bandwidth_Hz / 2,
            center_freq_Hz + bandwidth_Hz / 2,
            n_channels
        )
        self.prng = np.random.RandomState(self.seed)
        self.hop_sequence = self._generate_hop_sequence()
        self.current_hop = 0

    def _generate_hop_sequence(self) -> List[int]:
        # Pseudorandom permutation of channel indices
        seq = np.arange(self.n_channels)
        self.prng.shuffle(seq)
        return seq.tolist()

    def get_next_channel(self) -> Tuple[float, int]:
        # Returns next frequency and channel index
        idx = self.hop_sequence[self.current_hop % self.n_channels]
        freq = self.channel_freqs[idx]
        self.current_hop += 1
        return freq, idx

    def synchronize(self, remote_seed: int):
        # Synchronize hop sequence with remote party
        self.seed = remote_seed
        self.prng = np.random.RandomState(self.seed)
        self.hop_sequence = self._generate_hop_sequence()
        self.current_hop = 0

    def get_hop_sequence(self) -> List[float]:
        # Returns the full hop frequency sequence
        return [self.channel_freqs[i] for i in self.hop_sequence]
