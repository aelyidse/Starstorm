from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import time

class TacticalAnalysisModule:
    """
    Provides tactical situation analysis through multi-sensor data fusion.
    Integrates data from various sensors to create a unified tactical picture,
    detect threats, and recommend appropriate responses.
    """
    def __init__(self, sensor_types: List[str], fusion_method: str = 'bayesian'):
        self.sensor_types = sensor_types
        self.fusion_method = fusion_method
        self.sensor_data: Dict[str, Dict[str, Any]] = {s: {} for s in sensor_types}
        self.fused_state: Dict[str, Any] = {}
        self.confidence_levels: Dict[str, float] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.last_update_time: Optional[float] = None
        
        # Sensor reliability weights (could be learned over time)
        self.sensor_weights: Dict[str, float] = {s: 1.0 for s in sensor_types}
        
    def ingest_sensor_data(self, sensor_type: str, data: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        """
        Ingest new data from a specific sensor
        
        Args:
            sensor_type: Type of sensor providing the data
            data: Sensor readings and metadata
            timestamp: Optional timestamp (uses current time if not provided)
        """
        if sensor_type not in self.sensor_types:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
            
        now = timestamp or time.time()
        self.sensor_data[sensor_type] = {
            'data': data,
            'timestamp': now
        }
        
    def perform_fusion(self) -> Dict[str, Any]:
        """
        Perform sensor fusion using the configured method
        
        Returns:
            Fused tactical state representation
        """
        now = time.time()
        self.last_update_time = now
        
        if self.fusion_method == 'bayesian':
            fused_state = self._bayesian_fusion()
        elif self.fusion_method == 'kalman':
            fused_state = self._kalman_fusion()
        elif self.fusion_method == 'weighted':
            fused_state = self._weighted_fusion()
        else:
            fused_state = self._simple_fusion()
            
        # Store history
        self.analysis_history.append({
            'timestamp': now,
            'fused_state': fused_state.copy(),
            'confidence_levels': self.confidence_levels.copy()
        })
        
        # Limit history size
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]
            
        self.fused_state = fused_state
        return fused_state
    
    def _simple_fusion(self) -> Dict[str, Any]:
        """Simple averaging fusion method"""
        fused = {}
        confidence = {}
        
        # Collect all keys from all sensors
        all_keys = set()
        for sensor in self.sensor_types:
            if 'data' in self.sensor_data[sensor]:
                all_keys.update(self.sensor_data[sensor]['data'].keys())
        
        # Fuse each key
        for key in all_keys:
            values = []
            weights = []
            
            for sensor in self.sensor_types:
                if 'data' in self.sensor_data[sensor] and key in self.sensor_data[sensor]['data']:
                    value = self.sensor_data[sensor]['data'][key]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        weights.append(self.sensor_weights[sensor])
            
            if values:
                # Weighted average for numeric values
                fused[key] = np.average(values, weights=weights)
                # Confidence based on agreement between sensors
                if len(values) > 1:
                    confidence[key] = 1.0 - (np.std(values) / (np.max(values) - np.min(values) + 1e-10))
                else:
                    confidence[key] = 0.5  # Single sensor = medium confidence
            
        self.confidence_levels = confidence
        return fused
    
    def _bayesian_fusion(self) -> Dict[str, Any]:
        """Bayesian fusion method for probabilistic state estimation"""
        # Simplified implementation - would be more complex in a real system
        fused = self._simple_fusion()  # Start with simple fusion
        
        # Apply Bayesian update using prior knowledge
        if self.fused_state:  # If we have a prior state
            for key in fused:
                if key in self.fused_state:
                    # Simple Bayesian update (would use proper distributions in real system)
                    prior = self.fused_state[key]
                    likelihood = fused[key]
                    # Simplified posterior calculation
                    fused[key] = (prior * 0.3 + likelihood * 0.7)
                    
                    # Increase confidence due to temporal consistency
                    if key in self.confidence_levels:
                        self.confidence_levels[key] = min(1.0, self.confidence_levels[key] * 1.2)
                        
        return fused
    
    def _kalman_fusion(self) -> Dict[str, Any]:
        """Kalman filter based fusion for state tracking"""
        # Simplified placeholder - real implementation would use proper Kalman filter
        return self._simple_fusion()
    
    def _weighted_fusion(self) -> Dict[str, Any]:
        """Weighted fusion based on sensor reliability"""
        return self._simple_fusion()  # Already implements weighting
    
    def analyze_tactical_situation(self) -> Dict[str, Any]:
        """
        Analyze the current tactical situation based on fused sensor data
        
        Returns:
            Tactical analysis with threat assessments and recommendations
        """
        if not self.fused_state:
            self.perform_fusion()
            
        # Example tactical analysis logic
        threats = self._identify_threats()
        priorities = self._prioritize_threats(threats)
        recommendations = self._generate_recommendations(priorities)
        
        analysis = {
            'timestamp': time.time(),
            'threats': threats,
            'threat_priorities': priorities,
            'recommendations': recommendations,
            'confidence': self.confidence_levels
        }
        
        return analysis
    
    def _identify_threats(self) -> List[Dict[str, Any]]:
        """Identify potential threats from fused sensor data"""
        threats = []
        
        # Example threat identification logic
        # In a real system, this would use more sophisticated algorithms
        if 'radar_contacts' in self.fused_state:
            for contact in self.fused_state.get('radar_contacts', []):
                if isinstance(contact, dict) and contact.get('closing_rate', 0) > 50:
                    threats.append({
                        'type': 'approaching_object',
                        'source': 'radar',
                        'details': contact,
                        'severity': min(1.0, contact.get('closing_rate', 0) / 200)
                    })
                    
        if 'radiation_level' in self.fused_state and self.fused_state['radiation_level'] > 0.5:
            threats.append({
                'type': 'radiation',
                'source': 'radiation_sensor',
                'details': {'level': self.fused_state['radiation_level']},
                'severity': self.fused_state['radiation_level']
            })
            
        return threats
    
    def _prioritize_threats(self, threats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize identified threats based on severity and confidence"""
        for threat in threats:
            # Calculate priority based on severity and confidence
            confidence = self.confidence_levels.get(threat['type'], 0.5)
            threat['priority'] = threat['severity'] * confidence
            
        # Sort by priority (highest first)
        return sorted(threats, key=lambda t: t['priority'], reverse=True)
    
    def _generate_recommendations(self, prioritized_threats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate tactical recommendations based on prioritized threats"""
        recommendations = []
        
        for threat in prioritized_threats:
            if threat['type'] == 'approaching_object':
                if threat['severity'] > 0.8:
                    recommendations.append({
                        'action': 'evasive_maneuver',
                        'parameters': {'direction': 'away', 'urgency': 'high'},
                        'threat_id': id(threat)
                    })
                elif threat['severity'] > 0.4:
                    recommendations.append({
                        'action': 'increase_monitoring',
                        'parameters': {'target': threat['details']},
                        'threat_id': id(threat)
                    })
            elif threat['type'] == 'radiation':
                if threat['severity'] > 0.7:
                    recommendations.append({
                        'action': 'radiation_protection',
                        'parameters': {'level': 'maximum'},
                        'threat_id': id(threat)
                    })
                    
        return recommendations
    
    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the most recent tactical analysis"""
        if not self.analysis_history:
            return None
        return self.analysis_history[-1]
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get the complete analysis history"""
        return self.analysis_history