from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import statistics
from ai.recovery_procedure_generator import RecoveryProcedureGenerator
from ai.decision_making import (
    HierarchicalDecisionSystem, DecisionMaker, Decision, 
    DecisionLevel, DecisionPriority, StrategicDecisionMaker,
    TacticalDecisionMaker, OperationalDecisionMaker, EmergencyDecisionMaker
)

class SelfDiagnosticsAndHealthManager:
    """
    Implements self-diagnostics and health management for autonomous operations.
    Monitors subsystem status, detects faults, performs health checks, and coordinates recovery.
    Includes predictive maintenance to anticipate and prevent failures.
    """
    def __init__(self, subsystems: List[str], check_interval: float = 60.0):
        self.subsystems = subsystems
        self.status: Dict[str, str] = {s: 'unknown' for s in subsystems}
        self.health_log: List[Dict[str, Any]] = []
        self.last_check_time: Optional[float] = None
        self.check_interval = check_interval
        # Predictive maintenance additions
        self.performance_metrics: Dict[str, List[float]] = {s: [] for s in subsystems}
        self.failure_predictions: Dict[str, float] = {}  # subsystem -> probability of failure
        self.maintenance_schedule: List[Dict[str, Any]] = []
        # Anomaly detection additions
        self.metric_history: Dict[str, Dict[str, List[float]]] = {s: {} for s in subsystems}
        self.anomaly_log: List[Dict[str, Any]] = []
        self.anomaly_thresholds: Dict[str, Dict[str, float]] = {s: {} for s in subsystems}
        # Recovery procedure generator
        self.recovery_generator = RecoveryProcedureGenerator()
        self.recovery_procedures: Dict[str, Dict[str, Any]] = {}
        
        # Decision-making system integration
        self.decision_system = HierarchicalDecisionSystem()
        self._setup_decision_makers()
        self._register_action_handlers()

    def perform_health_check(self, current_time: Optional[float] = None) -> Dict[str, str]:
        # Simulate health check for all subsystems
        now = current_time or time.time()
        self.last_check_time = now
        import random
        for s in self.subsystems:
            # Simulate random failure with small probability
            if self.status[s] != 'failed':
                self.status[s] = 'operational' if random.random() > 0.05 else 'failed'
        self.health_log.append({'time': now, 'status': self.status.copy()})
        return self.status.copy()

    def detect_faults(self) -> List[str]:
        # Return list of failed subsystems
        return [s for s, st in self.status.items() if st == 'failed']

    def attempt_recovery(self, subsystem: str) -> bool:
        # Generate recovery procedure if not already available
        if subsystem not in self.recovery_procedures:
            diagnostics = self._gather_diagnostics(subsystem)
            failure_type = self._determine_failure_type(subsystem, diagnostics)
            procedure = self.recovery_generator.generate_procedure(
                subsystem, failure_type, diagnostics
            )
            self.recovery_procedures[subsystem] = procedure
        
        # Execute recovery procedure
        success = self._execute_recovery_procedure(self.recovery_procedures[subsystem])
        
        if success:
            self.status[subsystem] = 'operational'
        
        return success
    
    def _gather_diagnostics(self, subsystem: str) -> Dict[str, Any]:
        """Gather diagnostic information for a subsystem"""
        diagnostics = {
            'timestamp': time.time(),
            'recent_anomalies': [a for a in self.anomaly_log[-10:] 
                               if a['subsystem'] == subsystem],
            'performance_history': self.performance_metrics.get(subsystem, [])[-10:],
            'failure_probability': self.failure_predictions.get(subsystem, 0.0),
            'redundancy_available': subsystem.endswith('_redundant')  # Simple example logic
        }
        return diagnostics
    
    def _determine_failure_type(self, subsystem: str, diagnostics: Dict[str, Any]) -> str:
        """Determine the type of failure based on diagnostics"""
        # This could be expanded with more sophisticated failure classification
        if not diagnostics['recent_anomalies']:
            return 'unknown_failure'
        
        # Use the most recent anomaly type as failure type
        return diagnostics['recent_anomalies'][-1].get('metric', 'general_failure')
    
    def _execute_recovery_procedure(self, procedure: Dict[str, Any]) -> bool:
        """Execute a recovery procedure and return success status"""
        # In a real system, this would actually execute the steps
        # For simulation, we'll just log and return success probabilistically
        import random
        
        # Log execution attempt
        self.health_log.append({
            'time': time.time(),
            'action': 'recovery_attempt',
            'procedure_id': procedure['id'],
            'subsystem': procedure['subsystem']
        })
        
        # Simulate success probability based on severity
        success_probability = {
            'minor': 0.9,
            'major': 0.6,
            'critical': 0.3
        }.get(procedure['severity'], 0.5)
        
        success = random.random() < success_probability
        
        # Log result
        self.health_log.append({
            'time': time.time(),
            'action': 'recovery_result',
            'procedure_id': procedure['id'],
            'subsystem': procedure['subsystem'],
            'success': success
        })
        
        return success
    
    def get_recovery_procedures(self) -> Dict[str, Dict[str, Any]]:
        """Get all generated recovery procedures"""
        return self.recovery_procedures

    def get_health_log(self) -> List[Dict[str, Any]]:
        return self.health_log

    def get_status(self) -> Dict[str, str]:
        return self.status.copy()
        
    # Predictive maintenance methods
    def record_performance_metric(self, subsystem: str, metric_value: float) -> None:
        """Record a performance metric for predictive analysis"""
        if subsystem in self.performance_metrics:
            self.performance_metrics[subsystem].append(metric_value)
            # Keep only the last 100 measurements for efficiency
            if len(self.performance_metrics[subsystem]) > 100:
                self.performance_metrics[subsystem] = self.performance_metrics[subsystem][-100:]
    
    def detect_anomalies(self, subsystem: str, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in subsystem metrics using statistical models.
        
        Args:
            subsystem: The subsystem to check
            metrics: Dictionary of metric names to current values
            
        Returns:
            List of detected anomalies with details
        """
        anomalies = []
        
        for metric_name, current_value in metrics.items():
            # Initialize metric history if not present
            if metric_name not in self.metric_history[subsystem]:
                self.metric_history[subsystem][metric_name] = []
            
            history = self.metric_history[subsystem][metric_name]
            history.append(current_value)
            
            # Keep history to a reasonable size
            if len(history) > 100:
                history = history[-100:]
                self.metric_history[subsystem][metric_name] = history
            
            # Need enough data for statistical analysis
            if len(history) >= 10:
                anomaly = self._check_statistical_anomaly(subsystem, metric_name, current_value, history)
                if anomaly:
                    anomalies.append(anomaly)
                    self.anomaly_log.append(anomaly)
        
        return anomalies
    
    def _check_statistical_anomaly(self, subsystem: str, metric_name: str, 
                                  current_value: float, history: List[float]) -> Optional[Dict[str, Any]]:
        """
        Check if current value is a statistical anomaly using Z-score method.
        
        Args:
            subsystem: The subsystem being checked
            metric_name: The name of the metric
            current_value: The current value to check
            history: Historical values of this metric
            
        Returns:
            Anomaly details if detected, None otherwise
        """
        try:
            # Calculate mean and standard deviation
            mean = statistics.mean(history[:-1])  # Exclude current value
            stdev = statistics.stdev(history[:-1]) if len(history) > 2 else 0.001
            
            # Calculate z-score (how many standard deviations from mean)
            z_score = abs(current_value - mean) / stdev if stdev > 0 else 0
            
            # Get threshold or use default of 3 (common statistical threshold)
            threshold = self.anomaly_thresholds.get(subsystem, {}).get(metric_name, 3.0)
            
            if z_score > threshold:
                return {
                    'subsystem': subsystem,
                    'metric': metric_name,
                    'value': current_value,
                    'mean': mean,
                    'stdev': stdev,
                    'z_score': z_score,
                    'threshold': threshold,
                    'time': time.time()
                }
            return None
        except (statistics.StatisticsError, ZeroDivisionError):
            return None
    
    def set_anomaly_threshold(self, subsystem: str, metric_name: str, threshold: float) -> None:
        """Set the anomaly detection threshold for a specific metric"""
        if subsystem in self.anomaly_thresholds:
            self.anomaly_thresholds[subsystem][metric_name] = threshold
    
    def get_anomaly_log(self) -> List[Dict[str, Any]]:
        """Get the log of detected anomalies"""
        return self.anomaly_log
    
    def aggregate_health_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate health metrics across all subsystems.
        
        Returns:
            Dictionary with aggregated health metrics by subsystem
        """
        aggregated = {}
        
        for subsystem in self.subsystems:
            # Get basic status
            status = self.status.get(subsystem, 'unknown')
            
            # Calculate failure probability
            failure_prob = self.failure_predictions.get(subsystem, 0.0)
            
            # Count anomalies for this subsystem
            anomaly_count = sum(1 for a in self.anomaly_log 
                               if a['subsystem'] == subsystem)
            
            # Get performance metrics statistics if available
            perf_stats = {}
            metrics = self.performance_metrics.get(subsystem, [])
            if metrics:
                try:
                    perf_stats = {
                        'mean': statistics.mean(metrics),
                        'median': statistics.median(metrics),
                        'stdev': statistics.stdev(metrics) if len(metrics) > 1 else 0,
                        'min': min(metrics),
                        'max': max(metrics),
                        'trend': sum(metrics[-5:]) / 5 - sum(metrics[-10:-5]) / 5 
                                if len(metrics) >= 10 else 0
                    }
                except statistics.StatisticsError:
                    pass
            
            # Combine all metrics
            aggregated[subsystem] = {
                'status': status,
                'failure_probability': failure_prob,
                'anomaly_count': anomaly_count,
                'performance_stats': perf_stats,
                'last_updated': time.time()
            }
        
        return aggregated
    
    def get_health_trends(self, subsystem: str, days: int = 7) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get health metric trends for a specific subsystem.
        
        Args:
            subsystem: The subsystem to get trends for
            days: Number of days of history to include
            
        Returns:
            Dictionary of metric names to list of (timestamp, value) tuples
        """
        now = time.time()
        cutoff = now - (days * 24 * 3600)
        
        # Extract relevant metrics from logs
        trends = {
            'status': [],  # 1 for operational, 0 for failed
            'anomalies': [],  # Cumulative anomaly count
            'performance': []  # Average performance metric
        }
        
        # Process health logs for status
        anomaly_count = 0
        for entry in self.health_log:
            if entry['time'] >= cutoff:
                status_val = 1 if entry['status'].get(subsystem) == 'operational' else 0
                trends['status'].append((entry['time'], status_val))
        
        # Process anomaly logs
        for entry in self.anomaly_log:
            if entry['subsystem'] == subsystem and entry['time'] >= cutoff:
                anomaly_count += 1
                trends['anomalies'].append((entry['time'], anomaly_count))
        
        # Process performance metrics if available
        if subsystem in self.performance_metrics:
            # This is simplified - in a real system you'd store timestamps with metrics
            metrics = self.performance_metrics[subsystem]
            if metrics:
                # Create synthetic timestamps for demonstration
                interval = (now - cutoff) / len(metrics)
                for i, metric in enumerate(metrics):
                    timestamp = cutoff + i * interval
                    trends['performance'].append((timestamp, metric))
        
        return trends

    def predict_failures(self) -> Dict[str, float]:
        """Analyze performance metrics to predict potential failures"""
        import statistics
        
        for subsystem in self.subsystems:
            metrics = self.performance_metrics.get(subsystem, [])
            if len(metrics) >= 10:  # Need sufficient data for prediction
                # Simple prediction model: detect degrading performance trends
                try:
                    mean = statistics.mean(metrics[-10:])
                    stdev = statistics.stdev(metrics[-10:]) if len(metrics) > 1 else 0
                    trend = sum(metrics[-5:]) / 5 - sum(metrics[-10:-5]) / 5
                    
                    # Higher probability if negative trend and high variability
                    failure_prob = 0.0
                    if trend < 0:
                        failure_prob = min(0.9, abs(trend) / mean * (1 + stdev / mean))
                    
                    self.failure_predictions[subsystem] = failure_prob
                except (statistics.StatisticsError, ZeroDivisionError):
                    self.failure_predictions[subsystem] = 0.0
        
        return self.failure_predictions
    
    def schedule_maintenance(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Schedule maintenance for subsystems with high failure probability"""
        predictions = self.predict_failures()
        now = time.time()
        
        # Clear old schedule
        self.maintenance_schedule = [m for m in self.maintenance_schedule 
                                    if m['scheduled_time'] > now]
        
        # Add new maintenance tasks
        for subsystem, probability in predictions.items():
            if probability > threshold and not any(m['subsystem'] == subsystem 
                                                for m in self.maintenance_schedule):
                # Schedule maintenance with urgency based on probability
                urgency = "high" if probability > 0.8 else "medium"
                scheduled_time = now + (24 * 3600 if urgency == "high" else 72 * 3600)
                
                self.maintenance_schedule.append({
                    'subsystem': subsystem,
                    'failure_probability': probability,
                    'urgency': urgency,
                    'scheduled_time': scheduled_time,
                    'created_at': now
                })
        
        return self.maintenance_schedule
    
    def get_maintenance_recommendations(self) -> List[Dict[str, Any]]:
        """Get current maintenance recommendations"""
        # Update schedule first
        self.schedule_maintenance()
        return sorted(self.maintenance_schedule, 
                     key=lambda x: (x['scheduled_time'], -x['failure_probability']))

    def _setup_decision_makers(self):
        """Set up the decision makers for the hierarchical system"""
        # Strategic decision maker for long-term health management
        strategic_dm = StrategicDecisionMaker("HealthStrategic")
        strategic_dm.set_mission_objectives({
            "maximize_lifespan": 0.8,
            "maintain_functionality": 0.9,
            "optimize_performance": 0.7
        })
        self.decision_system.register_decision_maker(strategic_dm)
        
        # Tactical decision maker for resource allocation
        tactical_dm = TacticalDecisionMaker("HealthTactical")
        self.decision_system.register_decision_maker(tactical_dm)
        
        # Operational decision maker for immediate actions
        operational_dm = OperationalDecisionMaker("HealthOperational")
        self.decision_system.register_decision_maker(operational_dm)
        
        # Emergency decision maker for critical health issues
        emergency_dm = EmergencyDecisionMaker("HealthEmergency")
        emergency_dm.set_emergency_thresholds({
            "radiation_level": 1000.0,
            "temperature": 85.0,
            "power_remaining": 0.1  # 10% remaining
        })
        self.decision_system.register_decision_maker(emergency_dm)
    
    def _register_action_handlers(self):
        """Register handlers for different action types"""
        self.decision_system.register_action_handler(
            "attempt_recovery", 
            lambda action: self.attempt_recovery(action["subsystem"])
        )
        
        self.decision_system.register_action_handler(
            "schedule_maintenance",
            lambda action: self.schedule_maintenance(action.get("threshold", 0.7))
        )
        
        self.decision_system.register_action_handler(
            "update_anomaly_threshold",
            lambda action: self.set_anomaly_threshold(
                action["subsystem"], 
                action["metric"], 
                action["threshold"]
            )
        )
        
        self.decision_system.register_action_handler(
            "emergency_procedure",
            self._handle_emergency_procedure
        )
    
    def _handle_emergency_procedure(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency procedures"""
        procedure = action.get("procedure")
        parameters = action.get("parameters", {})
        
        if procedure == "emergency_shutdown":
            system = parameters.get("system")
            # Simulate emergency shutdown of the system
            if system in self.status:
                self.status[system] = "shutdown"
                self.health_log.append({
                    "time": time.time(),
                    "action": "emergency_shutdown",
                    "system": system
                })
                return {"success": True, "system": system, "status": "shutdown"}
        
        elif procedure == "emergency_mitigation":
            sensor = parameters.get("sensor")
            # Simulate emergency mitigation actions
            self.health_log.append({
                "time": time.time(),
                "action": "emergency_mitigation",
                "sensor": sensor,
                "value": parameters.get("value")
            })
            return {"success": True, "sensor": sensor, "mitigated": True}
        
        return {"success": False, "error": "Unknown emergency procedure"}
    
    def update_decision_context(self):
        """Update the decision system context with current health data"""
        context = {
            "system_status": self.status.copy(),
            "failure_predictions": self.failure_predictions.copy(),
            "anomaly_count": len(self.anomaly_log),
            "last_health_check_time": self.last_check_time,
            "maintenance_schedule": self.maintenance_schedule.copy(),
            "sensor_data": {}  # Would be populated with actual sensor data
        }
        
        # Add performance metrics
        for subsystem, metrics in self.performance_metrics.items():
            if metrics:
                context["sensor_data"][f"{subsystem}_performance"] = metrics[-1]
        
        self.decision_system.update_context(context)
    
    def execute_decision_cycle(self) -> List[Decision]:
        """Execute a decision cycle and return the decisions made"""
        # Update context first
        self.update_decision_context()
        
        # Execute the decision cycle
        decisions = self.decision_system.execute_decision_cycle()
        
        # Log decisions
        for decision in decisions:
            self.health_log.append({
                "time": time.time(),
                "action": "decision_made",
                "level": decision.level.name,
                "description": decision.description,
                "priority": decision.priority.name
            })
        
        return decisions
    
    def get_recent_decisions(self, count: int = 10) -> List[Decision]:
        """Get the most recent decisions made by the system"""
        return self.decision_system.get_recent_decisions(count)
