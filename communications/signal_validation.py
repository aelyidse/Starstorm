import numpy as np
from typing import Tuple, Dict, Any

class SignalIntegrityValidator:
    """
    Validates signal integrity in noisy environments using SNR estimation, error detection, and confidence metrics.
    Supports thresholding, CRC, and soft-decision logic for robust communications.
    """
    def __init__(self, snr_threshold_db: float = 6.0):
        self.snr_threshold_db = snr_threshold_db

    def estimate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        # Estimate SNR in dB
        power_signal = np.mean(signal ** 2)
        power_noise = np.mean(noise ** 2) + 1e-12
        snr = 10 * np.log10(power_signal / power_noise)
        return snr

    def crc_check(self, data: bytes, crc_func) -> bool:
        # Check CRC using user-supplied function
        return crc_func(data)

    def validate(self, received: np.ndarray, expected: np.ndarray, noise: np.ndarray, crc_func = None, data: bytes = None) -> Dict[str, Any]:
        snr = self.estimate_snr(received, noise)
        passed_snr = snr >= self.snr_threshold_db
        # Soft-decision: correlation
        corr = np.corrcoef(received, expected)[0, 1]
        passed_corr = corr > 0.9
        # CRC check if provided
        passed_crc = True
        if crc_func is not None and data is not None:
            passed_crc = self.crc_check(data, crc_func)
        return {
            'snr_db': snr,
            'correlation': corr,
            'passed_snr': passed_snr,
            'passed_corr': passed_corr,
            'passed_crc': passed_crc,
            'integrity_ok': passed_snr and passed_corr and passed_crc
        }
