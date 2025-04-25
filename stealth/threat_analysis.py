import numpy as np
from typing import List, Dict, Any

class ThreatFrequencyAnalyzer:
    """
    Analyzes detected threat signals and determines dominant frequencies for countermeasure selection.
    Supports multi-band, multi-threat environments.
    """
    def __init__(self):
        self.detected_frequencies: List[float] = []
        self.threat_db: List[Dict[str, Any]] = []  # [{'freq_Hz': float, 'type': str, ...}]

    def analyze_spectrum(self, spectrum: np.ndarray, freqs: np.ndarray, threshold: float = 0.7) -> List[float]:
        # spectrum: power spectral density
        # freqs: corresponding frequency bins (Hz)
        # threshold: fraction of max to consider as threat
        max_val = np.max(spectrum)
        detected = freqs[spectrum > threshold * max_val]
        self.detected_frequencies = detected.tolist()
        return self.detected_frequencies

    def add_threat(self, freq_Hz: float, threat_type: str):
        self.threat_db.append({'freq_Hz': freq_Hz, 'type': threat_type})

    def get_threats(self) -> List[Dict[str, Any]]:
        return self.threat_db.copy()

class CountermeasureSelector:
    """
    Selects optimal countermeasures based on detected threat frequencies and system capabilities.
    Supports metamaterial, jamming, and emission control responses.
    """
    def __init__(self, available_cms: List[str]):
        self.available_cms = available_cms  # e.g., ['metamaterial', 'jammer', 'emission_control']

    def select(self, detected_freqs: List[float], threat_db: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Simple logic: prioritize metamaterial for radar, jammer for comms, emission control for IR
        response = {}
        for f in detected_freqs:
            threat = next((t for t in threat_db if abs(t['freq_Hz'] - f) < 1e3), None)
            if threat:
                if threat['type'] == 'radar' and 'metamaterial' in self.available_cms:
                    response[f] = 'metamaterial'
                elif threat['type'] == 'comms' and 'jammer' in self.available_cms:
                    response[f] = 'jammer'
                elif threat['type'] == 'IR' and 'emission_control' in self.available_cms:
                    response[f] = 'emission_control'
                else:
                    response[f] = 'none'
            else:
                response[f] = 'none'
        return response
