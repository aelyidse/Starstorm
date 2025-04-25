import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from collections import deque

class StabilityAnalyzer:
    """
    Provides tools for analyzing control system stability.
    Implements methods for time-domain and frequency-domain stability analysis.
    """
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.signal_history: Dict[str, deque] = {}
        self.stability_metrics: Dict[str, Dict[str, float]] = {}
        
    def record_signal(self, name: str, value: float, timestamp: Optional[float] = None):
        """Record a signal value for stability analysis"""
        if name not in self.signal_history:
            self.signal_history[name] = deque(maxlen=self.history_length)
            
        self.signal_history[name].append((timestamp or time.time(), value))
        
    def analyze_stability(self, signal_name: str) -> Dict[str, float]:
        """
        Analyze stability of a recorded signal
        
        Args:
            signal_name: Name of the signal to analyze
            
        Returns:
            Dictionary of stability metrics
        """
        if signal_name not in self.signal_history or len(self.signal_history[signal_name]) < 10:
            return {'error': 'Insufficient data for analysis'}
            
        # Extract signal values
        timestamps, values = zip(*list(self.signal_history[signal_name]))
        values = np.array(values)
        
        # Calculate stability metrics
        metrics = {
            'mean': float(np.mean(values)),
            'std_dev': float(np.std(values)),
            'variance': float(np.var(values)),
            'max_deviation': float(np.max(np.abs(values - np.mean(values)))),
            'settling_time': self._estimate_settling_time(values),
            'overshoot': self._calculate_overshoot(values),
            'oscillation_frequency': self._detect_oscillation_frequency(timestamps, values),
            'damping_ratio': self._estimate_damping_ratio(values)
        }
        
        # Store metrics
        self.stability_metrics[signal_name] = metrics
        return metrics
        
    def _estimate_settling_time(self, values: np.ndarray) -> float:
        """Estimate settling time (time to reach within 2% of final value)"""
        if len(values) < 10:
            return 0.0
            
        final_value = values[-1]
        threshold = 0.02 * abs(final_value)  # 2% threshold
        
        # Find first point where all subsequent values are within threshold
        for i in range(len(values) - 1, 0, -1):
            if abs(values[i] - final_value) > threshold:
                return float(i)  # Return index as proxy for time
                
        return 0.0  # Already settled
        
    def _calculate_overshoot(self, values: np.ndarray) -> float:
        """Calculate percent overshoot"""
        if len(values) < 3:
            return 0.0
            
        final_value = values[-1]
        if final_value == 0:
            return 0.0
            
        max_value = np.max(values)
        if max_value <= final_value:
            return 0.0
            
        return float((max_value - final_value) / final_value * 100.0)
        
    def _detect_oscillation_frequency(self, timestamps: Tuple[float, ...], values: np.ndarray) -> float:
        """Detect oscillation frequency using FFT"""
        if len(values) < 10:
            return 0.0
            
        # Detrend the signal
        trend = np.polyfit(range(len(values)), values, 1)
        detrended = values - np.polyval(trend, range(len(values)))
        
        # Compute FFT
        fft_values = np.abs(np.fft.rfft(detrended))
        fft_freqs = np.fft.rfftfreq(len(detrended), d=np.mean(np.diff(timestamps)))
        
        # Find dominant frequency (excluding DC component)
        if len(fft_values) <= 1:
            return 0.0
            
        dominant_idx = np.argmax(fft_values[1:]) + 1
        return float(fft_freqs[dominant_idx])
        
    def _estimate_damping_ratio(self, values: np.ndarray) -> float:
        """Estimate damping ratio from decay envelope"""
        if len(values) < 10:
            return 1.0  # Assume critically damped
            
        # Detrend
        trend = np.polyfit(range(len(values)), values, 1)
        detrended = values - np.polyval(trend, range(len(values)))
        
        # Find peaks
        peaks = []
        for i in range(1, len(detrended) - 1):
            if detrended[i] > detrended[i-1] and detrended[i] > detrended[i+1]:
                peaks.append((i, detrended[i]))
                
        if len(peaks) < 2:
            return 1.0  # Not enough peaks to estimate
            
        # Calculate logarithmic decrement
        peak_values = [p[1] for p in peaks]
        if peak_values[0] == 0:
            return 1.0
            
        log_decrement = np.log(peak_values[0] / peak_values[-1]) / (len(peak_values) - 1)
        
        # Convert to damping ratio
        damping_ratio = log_decrement / (2 * np.pi)
        return float(min(1.0, damping_ratio))  # Cap at 1.0
        
    def plot_stability_analysis(self, signal_name: str, save_path: Optional[str] = None):
        """Generate stability analysis plots for a signal"""
        if signal_name not in self.signal_history or len(self.signal_history[signal_name]) < 10:
            return False
            
        timestamps, values = zip(*list(self.signal_history[signal_name]))
        rel_timestamps = [t - timestamps[0] for t in timestamps]
        
        plt.figure(figsize=(12, 8))
        
        # Time domain plot
        plt.subplot(2, 1, 1)
        plt.plot(rel_timestamps, values)
        plt.title(f'Stability Analysis: {signal_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.grid(True)
        
        # Add stability metrics as text
        if signal_name in self.stability_metrics:
            metrics = self.stability_metrics[signal_name]
            text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            plt.figtext(0.02, 0.02, text, fontsize=9)
        
        # Frequency domain plot (FFT)
        plt.subplot(2, 1, 2)
        values_array = np.array(values)
        trend = np.polyfit(range(len(values_array)), values_array, 1)
        detrended = values_array - np.polyval(trend, range(len(values_array)))
        
        fft_values = np.abs(np.fft.rfft(detrended))
        fft_freqs = np.fft.rfftfreq(len(detrended), d=np.mean(np.diff(timestamps)))
        
        plt.plot(fft_freqs, fft_values)
        plt.title('Frequency Domain Analysis')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return True
        else:
            plt.show()
            return True


class ControlSystemStabilityMonitor:
    """
    Real-time monitor for control system stability.
    Tracks control signals, detects instability, and provides early warnings.
    """
    def __init__(self, warning_threshold: float = 0.7, critical_threshold: float = 0.9):
        self.analyzer = StabilityAnalyzer()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.stability_status: Dict[str, str] = {}
        self.alerts: List[Dict[str, Any]] = []
        
    def update_signal(self, name: str, value: float):
        """Update a control signal and analyze stability"""
        self.analyzer.record_signal(name, value)
        
        # Only analyze if we have enough data
        if len(self.analyzer.signal_history.get(name, [])) >= 10:
            metrics = self.analyzer.analyze_stability(name)
            self._evaluate_stability(name, metrics)
            
    def _evaluate_stability(self, signal_name: str, metrics: Dict[str, float]):
        """Evaluate stability based on metrics and generate alerts if needed"""
        # Skip if error in metrics
        if 'error' in metrics:
            return
            
        # Calculate instability score (higher = more unstable)
        instability_score = (
            0.3 * min(1.0, metrics['std_dev'] / max(0.001, abs(metrics['mean']))) +
            0.3 * min(1.0, metrics['overshoot'] / 100.0) +
            0.2 * min(1.0, 1.0 - metrics['damping_ratio']) +
            0.2 * min(1.0, metrics['max_deviation'] / max(0.001, abs(metrics['mean'])))
        )
        
        # Update status
        old_status = self.stability_status.get(signal_name, 'stable')
        
        if instability_score >= self.critical_threshold:
            new_status = 'critical'
        elif instability_score >= self.warning_threshold:
            new_status = 'warning'
        else:
            new_status = 'stable'
            
        self.stability_status[signal_name] = new_status
        
        # Generate alert if status worsened
        if self._status_worsened(old_status, new_status):
            self.alerts.append({
                'signal': signal_name,
                'timestamp': time.time(),
                'status': new_status,
                'instability_score': instability_score,
                'metrics': metrics
            })
            
    def _status_worsened(self, old_status: str, new_status: str) -> bool:
        """Check if status has worsened"""
        status_rank = {'stable': 0, 'warning': 1, 'critical': 2}
        return status_rank.get(new_status, 0) > status_rank.get(old_status, 0)
        
    def get_alerts(self, clear: bool = False) -> List[Dict[str, Any]]:
        """Get all alerts and optionally clear the alert list"""
        alerts = self.alerts.copy()
        if clear:
            self.alerts = []
        return alerts
        
    def get_stability_report(self) -> Dict[str, Any]:
        """Get comprehensive stability report for all signals"""
        return {
            'signals': {
                name: {
                    'status': self.stability_status.get(name, 'unknown'),
                    'metrics': self.analyzer.stability_metrics.get(name, {})
                }
                for name in self.analyzer.signal_history.keys()
            },
            'alert_count': len(self.alerts),
            'critical_signals': [
                name for name, status in self.stability_status.items()
                if status == 'critical'
            ]
        }


class RootLocusAnalyzer:
    """
    Implements root locus analysis for control system stability.
    Analyzes system stability across different gain values.
    """
    def __init__(self):
        pass
        
    def analyze_transfer_function(self, numerator: List[float], denominator: List[float], 
                                 gain_range: Tuple[float, float], points: int = 100) -> Dict[str, Any]:
        """
        Analyze stability of a transfer function using root locus
        
        Args:
            numerator: Coefficients of transfer function numerator (highest power first)
            denominator: Coefficients of transfer function denominator (highest power first)
            gain_range: Range of gain values to analyze (min, max)
            points: Number of gain points to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Create gain values
        gains = np.linspace(gain_range[0], gain_range[1], points)
        
        # Calculate poles for each gain
        poles_list = []
        stable_gains = []
        
        for gain in gains:
            # Apply gain to numerator
            scaled_num = [gain * coeff for coeff in numerator]
            
            # Calculate closed-loop denominator: den + K*num
            if len(scaled_num) < len(denominator):
                # Pad with zeros
                scaled_num = [0] * (len(denominator) - len(scaled_num)) + scaled_num
            
            # Calculate closed-loop poles
            closed_loop_den = [n + d for n, d in zip(scaled_num, denominator)]
            poles = np.roots(closed_loop_den)
            poles_list.append(poles)
            
            # Check stability (all poles in left half-plane)
            if all(pole.real < 0 for pole in poles):
                stable_gains.append(gain)
        
        # Find stability margins
        stability_range = (min(stable_gains) if stable_gains else None, 
                          max(stable_gains) if stable_gains else None)
        
        return {
            'gains': gains.tolist(),
            'poles': [[complex(p.real, p.imag) for p in poles] for poles in poles_list],
            'stable_gains': stable_gains,
            'stability_range': stability_range,
            'is_stable': len(stable_gains) > 0
        }
        
    def plot_root_locus(self, analysis_result: Dict[str, Any], save_path: Optional[str] = None):
        """Generate root locus plot from analysis results"""
        gains = analysis_result['gains']
        poles_list = analysis_result['poles']
        
        plt.figure(figsize=(10, 8))
        
        # Plot imaginary axis
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot poles for each gain
        for i, poles in enumerate(poles_list):
            # Use color gradient based on gain
            color = plt.cm.viridis(i / len(gains))
            
            for pole in poles:
                plt.plot(pole.real, pole.imag, 'o', color=color, markersize=4, alpha=0.7)
        
        # Add colorbar for gain reference
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=gains[0], vmax=gains[-1]))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Gain (K)')
        
        # Add stability information
        if analysis_result['stability_range'][0] is not None:
            plt.title(f"Root Locus Analysis\nStable for K in [{analysis_result['stability_range'][0]:.4f}, {analysis_result['stability_range'][1]:.4f}]")
        else:
            plt.title("Root Locus Analysis\nNo stable gain values found")
        
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.grid(True)
        
        # Make equal aspect ratio
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return True
        else:
            plt.show()
            return True