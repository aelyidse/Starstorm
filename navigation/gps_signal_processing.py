import numpy as np
from typing import Dict, Any, Optional

class GPSSignalProcessor:
    """
    Processes GPS signals, simulates receiver solution, and models errors.
    Supports pseudorange, carrier phase, and error sources (ionospheric, multipath, clock, noise).
    """
    def __init__(self, receiver_position: np.ndarray, clock_bias_s: float = 0.0):
        self.receiver_position = receiver_position  # np.array([x, y, z])
        self.clock_bias_s = clock_bias_s
        self.c = 299792458.0  # m/s

    def simulate_pseudorange(self, sat_position: np.ndarray, true_range: Optional[float] = None, error_model: Optional[Dict[str, float]] = None) -> float:
        # Compute geometric range
        if true_range is None:
            true_range = np.linalg.norm(sat_position - self.receiver_position)
        # Error sources
        iono = error_model.get('iono', 0.0) if error_model else 0.0
        multipath = error_model.get('multipath', 0.0) if error_model else 0.0
        clock = error_model.get('clock', 0.0) if error_model else 0.0
        noise = error_model.get('noise', 0.0) if error_model else 0.0
        pseudorange = true_range + iono + multipath + clock * self.c + noise
        return pseudorange

    def process_observations(self, sats: Dict[str, np.ndarray], error_model: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        # Returns pseudorange for each satellite
        return {prn: self.simulate_pseudorange(pos, error_model=error_model) for prn, pos in sats.items()}

    def estimate_position(self, pseudoranges: Dict[str, float], sat_positions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        # Simple least-squares position solution (iterative, not robust to poor geometry)
        x = self.receiver_position.copy()
        for _ in range(5):
            H = []
            y = []
            for prn, pr in pseudoranges.items():
                s = sat_positions[prn]
                r = np.linalg.norm(s - x)
                H.append((x - s) / r)
                y.append(pr - r)
            H = np.vstack(H)
            y = np.array(y)
            dx, *_ = np.linalg.lstsq(H, y, rcond=None)
            x = x + dx
        return {'position_m': x}

class GPSErrorModel:
    """
    Models GPS error sources (ionospheric, multipath, clock, noise) for simulation and analysis.
    """
    def __init__(self, iono_std: float = 5.0, multipath_std: float = 2.0, clock_std: float = 1e-8, noise_std: float = 1.0):
        self.iono_std = iono_std
        self.multipath_std = multipath_std
        self.clock_std = clock_std
        self.noise_std = noise_std

    def sample(self) -> Dict[str, float]:
        return {
            'iono': np.random.normal(0, self.iono_std),
            'multipath': np.random.normal(0, self.multipath_std),
            'clock': np.random.normal(0, self.clock_std),
            'noise': np.random.normal(0, self.noise_std)
        }
