import numpy as np
from typing import Dict, Any, List
from scipy.signal import find_peaks, periodogram

class EnemySignalAnalyzer:
    """
    Analyzes intercepted RF signals to identify enemy communication patterns.
    Extracts features such as modulation, periodicity, and hopping behavior.
    """
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def analyze_spectrum(self, signal: np.ndarray) -> Dict[str, Any]:
        # Compute power spectral density
        freqs, psd = periodogram(signal, fs=self.sample_rate)
        # Identify peaks (potential carrier frequencies)
        peaks, _ = find_peaks(psd, height=np.max(psd) * 0.1)
        carrier_freqs = freqs[peaks]
        return {'carrier_freqs': carrier_freqs, 'psd': psd, 'freqs': freqs}

    def detect_periodicity(self, signal: np.ndarray) -> float:
        # Autocorrelation to find periodicity (e.g., TDMA frame, hopping interval)
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[corr.size // 2:]
        peaks, _ = find_peaks(corr, height=0.5 * np.max(corr))
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sample_rate
            avg_period = np.mean(intervals)
            return avg_period
        return 0.0

    def classify_modulation(self, signal: np.ndarray) -> str:
        # Simple modulation classifier: AM, FM, PSK, noise-like
        std = np.std(signal)
        if std < 0.2:
            return 'constant'
        elif np.mean(np.abs(np.diff(signal))) > 0.5:
            return 'PSK/FHSS/noise-like'
        elif np.std(np.abs(signal)) > 0.5:
            return 'AM/FM'
        else:
            return 'unknown'

    def extract_features(self, signal: np.ndarray) -> Dict[str, Any]:
        spectrum = self.analyze_spectrum(signal)
        period = self.detect_periodicity(signal)
        modulation = self.classify_modulation(signal)
        return {
            'carrier_freqs': spectrum['carrier_freqs'],
            'modulation': modulation,
            'periodicity_s': period
        }
