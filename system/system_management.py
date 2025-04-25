from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import time
from collections import deque

class SystemManagementModule:
    """
    Manages system health and predicts potential failures before they occur.
    Implements predictive failure analysis using statistical models and machine learning.
    """
    def __init__(self, subsystems: List[str], history_length: int = 100):
        self.subsystems = subsystems
        self.history_length = history_length
        self.telemetry_history: Dict[str, Dict[str, deque]] = {s: {} for s in subsystems}
        self.failure_predictions: Dict[str, Dict[str, float]] = {s: {} for s in subsystems}
        self.last_analysis_time: Optional[float] = None
        self.maintenance_recommendations: List[Dict[str, Any]] = []
        
    def record_telemetry(self, subsystem: str, metrics: Dict[str, float], timestamp: Optional[float] = None) -> None:
        """
        Record telemetry data for a subsystem for later analysis
        
        Args:
            subsystem: Name of the subsystem
            metrics: Dictionary of metric names to values
            timestamp: Optional timestamp (uses current time if not provided)
        """
        if subsystem not in self.subsystems:
            raise ValueError(f"Unknown subsystem: {subsystem}")
            
        now = timestamp or time.time()
        
        # Initialize metrics if they don't exist
        for metric, value in metrics.items():
            if metric not in self.telemetry_history[subsystem]:
                self.telemetry_history[subsystem][metric] = deque(maxlen=self.history_length)
            
            # Store value with timestamp
            self.telemetry_history[subsystem][metric].append((now, value))
    
    def analyze_failure_patterns(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze telemetry data to predict potential failures
        
        Returns:
            Dictionary mapping subsystems to metrics and their failure probabilities
        """
        self.last_analysis_time = time.time()
        
        for subsystem in self.subsystems:
            for metric, history in self.telemetry_history[subsystem].items():
                if len(history) < 10:  # Need enough data for analysis
                    continue
                    
                # Extract values and timestamps
                timestamps, values = zip(*list(history))
                
                # Calculate trend using linear regression
                trend = self._calculate_trend(timestamps, values)
                
                # Calculate volatility (standard deviation)
                volatility = np.std(values)
                
                # Calculate anomaly score
                anomaly_score = self._detect_anomalies(values)
                
                # Combine factors to estimate failure probability
                failure_prob = self._estimate_failure_probability(trend, volatility, anomaly_score)
                
                # Store prediction
                if metric not in self.failure_predictions[subsystem]:
                    self.failure_predictions[subsystem][metric] = 0.0
                    
                # Smooth prediction with previous value (exponential smoothing)
                alpha = 0.3  # Smoothing factor
                old_prob = self.failure_predictions[subsystem][metric]
                self.failure_predictions[subsystem][metric] = alpha * failure_prob + (1 - alpha) * old_prob
        
        return self.failure_predictions
    
    def _calculate_trend(self, timestamps: Tuple[float, ...], values: Tuple[float, ...]) -> float:
        """Calculate the trend of a time series using linear regression"""
        x = np.array(timestamps)
        y = np.array(values)
        
        # Normalize x to avoid numerical issues
        x = x - x[0]
        
        # Simple linear regression
        if np.std(x) == 0:
            return 0.0
            
        slope = np.cov(x, y)[0, 1] / np.var(x)
        
        # Normalize slope by the mean value to get relative trend
        mean_y = np.mean(y)
        if mean_y == 0:
            return 0.0
            
        return slope / mean_y
    
    def _detect_anomalies(self, values: Tuple[float, ...]) -> float:
        """Detect anomalies using Z-score method"""
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            return 0.0
            
        # Calculate Z-scores
        z_scores = np.abs((values_array - mean) / std)
        
        # Return the maximum Z-score from recent values (last 20%)
        recent_z = z_scores[-max(1, len(z_scores)//5):]
        return float(np.max(recent_z))
    
    def _estimate_failure_probability(self, trend: float, volatility: float, anomaly_score: float) -> float:
        """Combine multiple factors to estimate failure probability"""
        # Negative trend (decreasing values) often indicates degradation
        trend_factor = max(0.0, -trend * 10)  # Scale trend appropriately
        
        # High volatility can indicate instability
        volatility_factor = min(1.0, volatility * 2)
        
        # Anomalies are strong indicators of potential failures
        anomaly_factor = min(1.0, anomaly_score / 3)  # Z-score > 3 is significant
        
        # Combine factors (weights could be tuned based on domain knowledge)
        probability = 0.4 * trend_factor + 0.3 * volatility_factor + 0.3 * anomaly_factor
        
        return min(1.0, max(0.0, probability))
    
    def generate_maintenance_recommendations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Generate maintenance recommendations based on failure predictions
        
        Args:
            threshold: Probability threshold for generating recommendations
            
        Returns:
            List of maintenance recommendations
        """
        recommendations = []
        
        for subsystem in self.subsystems:
            for metric, probability in self.failure_predictions[subsystem].items():
                if probability >= threshold:
                    # Calculate urgency based on probability
                    if probability >= 0.9:
                        urgency = "critical"
                    elif probability >= 0.8:
                        urgency = "high"
                    else:
                        urgency = "medium"
                        
                    recommendations.append({
                        'subsystem': subsystem,
                        'metric': metric,
                        'failure_probability': probability,
                        'urgency': urgency,
                        'timestamp': time.time()
                    })
        
        self.maintenance_recommendations = recommendations
        return recommendations
    
    def get_subsystem_health(self, subsystem: str) -> Dict[str, Any]:
        """
        Get the health status of a specific subsystem
        
        Args:
            subsystem: Name of the subsystem
            
        Returns:
            Health status including metrics and failure probabilities
        """
        if subsystem not in self.subsystems:
            raise ValueError(f"Unknown subsystem: {subsystem}")
            
        # Calculate overall health score (inverse of average failure probability)
        metrics = self.failure_predictions[subsystem]
        if not metrics:
            health_score = 1.0  # Perfect health if no metrics
        else:
            avg_failure_prob = sum(metrics.values()) / len(metrics)
            health_score = 1.0 - avg_failure_prob
            
        # Determine status based on health score
        if health_score >= 0.9:
            status = "optimal"
        elif health_score >= 0.7:
            status = "good"
        elif health_score >= 0.4:
            status = "degraded"
        else:
            status = "critical"
            
        return {
            'subsystem': subsystem,
            'health_score': health_score,
            'status': status,
            'metrics': self.failure_predictions[subsystem],
            'last_updated': self.last_analysis_time
        }