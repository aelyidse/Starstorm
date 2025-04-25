from typing import Dict, Any, List, Optional, Tuple
import math
import time
import numpy as np

class FaultDetector:
    """
    Detects anomalies in propulsion system parameters.
    """
    def __init__(self, parameter_limits: Dict[str, Tuple[float, float]]):
        """
        Initialize fault detector with parameter limits.
        
        Args:
            parameter_limits: Dictionary of parameter names and their (min, max) limits
        """
        self.parameter_limits = parameter_limits
        self.fault_history = []
        self.active_faults = {}
        
    def check_parameter(self, param_name: str, value: float) -> bool:
        """Check if parameter is within limits."""
        if param_name not in self.parameter_limits:
            return True
            
        min_val, max_val = self.parameter_limits[param_name]
        return min_val <= value <= max_val
        
    def detect_faults(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect faults in current parameters.
        
        Args:
            parameters: Dictionary of current parameter values
            
        Returns:
            Dictionary of detected faults
        """
        detected_faults = {}
        
        for param_name, value in parameters.items():
            if not self.check_parameter(param_name, value):
                min_val, max_val = self.parameter_limits.get(param_name, (None, None))
                detected_faults[param_name] = {
                    "value": value,
                    "min_limit": min_val,
                    "max_limit": max_val,
                    "timestamp": time.time()
                }
                
        # Update active faults
        self.active_faults.update(detected_faults)
        
        # Record to history if faults detected
        if detected_faults:
            self.fault_history.append({
                "timestamp": time.time(),
                "faults": detected_faults.copy()
            })
            
        return detected_faults


class FaultHandler:
    """
    Handles fault recovery actions for propulsion system.
    """
    def __init__(self):
        self.recovery_actions = {
            "chamber_pressure_high": self.handle_high_pressure,
            "chamber_pressure_low": self.handle_low_pressure,
            "temperature_high": self.handle_high_temperature,
            "temperature_low": self.handle_low_temperature,
            "flow_rate_high": self.handle_high_flow_rate,
            "flow_rate_low": self.handle_low_flow_rate,
            "mixture_ratio_off": self.handle_mixture_ratio,
            "valve_stuck": self.handle_valve_stuck,
            "ignition_failure": self.handle_ignition_failure,
            "thrust_low": self.handle_low_thrust,
            "sensor_failure": self.handle_sensor_failure
        }
        
    def handle_high_pressure(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle high chamber pressure fault."""
        # Reduce throttle to safe level
        safe_throttle = max(0.3, engine.current_throttle * 0.7)
        engine.set_throttle(safe_throttle)
        
        return {
            "action": "reduce_throttle",
            "new_throttle": safe_throttle,
            "status": "recovering"
        }
        
    def handle_low_pressure(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle low chamber pressure fault."""
        # Check if we need to shut down or can recover
        if engine.current_throttle > 0.3:
            # Try reducing throttle to stabilize
            new_throttle = max(0.2, engine.current_throttle * 0.8)
            engine.set_throttle(new_throttle)
            return {
                "action": "reduce_throttle",
                "new_throttle": new_throttle,
                "status": "recovering"
            }
        else:
            # Pressure too low at low throttle, shut down
            engine.shutdown()
            return {
                "action": "shutdown",
                "status": "safe_shutdown"
            }
        
    def handle_high_temperature(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle high temperature fault."""
        # Reduce throttle to cool down
        new_throttle = max(0.2, engine.current_throttle * 0.6)
        engine.set_throttle(new_throttle)
        
        return {
            "action": "reduce_throttle_for_cooling",
            "new_throttle": new_throttle,
            "status": "recovering"
        }
        
    def handle_low_temperature(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle low temperature fault."""
        # Might need preheating or gradual throttle increase
        if not engine.running:
            return {
                "action": "preheat_required",
                "status": "waiting"
            }
        else:
            # Gradual increase if already running
            new_throttle = min(0.4, engine.current_throttle * 1.1)
            engine.set_throttle(new_throttle)
            return {
                "action": "increase_throttle_gradually",
                "new_throttle": new_throttle,
                "status": "recovering"
            }
        
    def handle_high_flow_rate(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle high flow rate fault."""
        # Reduce throttle to decrease flow
        new_throttle = max(0.2, engine.current_throttle * 0.7)
        engine.set_throttle(new_throttle)
        
        return {
            "action": "reduce_throttle",
            "new_throttle": new_throttle,
            "status": "recovering"
        }
        
    def handle_low_flow_rate(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle low flow rate fault."""
        # Check if we have a blockage or need to increase pressure
        if engine.current_throttle > 0.7:
            # Likely blockage, shut down for safety
            engine.shutdown()
            return {
                "action": "shutdown",
                "reason": "suspected_blockage",
                "status": "safe_shutdown"
            }
        else:
            # Try increasing throttle
            new_throttle = min(0.7, engine.current_throttle * 1.2)
            engine.set_throttle(new_throttle)
            return {
                "action": "increase_throttle",
                "new_throttle": new_throttle,
                "status": "recovering"
            }
        
    def handle_mixture_ratio(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle off-nominal mixture ratio."""
        # This would typically adjust valve positions in a real system
        # For this simulation, we'll just reduce throttle for safety
        new_throttle = max(0.3, engine.current_throttle * 0.8)
        engine.set_throttle(new_throttle)
        
        return {
            "action": "adjust_mixture_ratio",
            "new_throttle": new_throttle,
            "status": "recovering"
        }
        
    def handle_valve_stuck(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stuck valve fault."""
        # In a real system, might try cycling the valve
        # For simulation, we'll shut down if critical
        if fault_data.get("is_critical", True):
            engine.shutdown()
            return {
                "action": "shutdown",
                "reason": "critical_valve_stuck",
                "status": "safe_shutdown"
            }
        else:
            # Non-critical valve, continue with reduced throttle
            new_throttle = max(0.4, engine.current_throttle * 0.8)
            engine.set_throttle(new_throttle)
            return {
                "action": "continue_with_reduced_performance",
                "new_throttle": new_throttle,
                "status": "degraded_operation"
            }
        
    def handle_ignition_failure(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ignition failure."""
        # Try reignition sequence once
        if not fault_data.get("retry_attempted", False):
            # In a real system, would reset and retry ignition sequence
            return {
                "action": "retry_ignition",
                "retry_count": 1,
                "status": "retrying"
            }
        else:
            # Already tried once, abort
            return {
                "action": "abort_ignition",
                "status": "failed"
            }
        
    def handle_low_thrust(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle low thrust fault."""
        # Check if we can increase throttle
        current_thrust = fault_data.get("current_thrust", 0)
        expected_thrust = fault_data.get("expected_thrust", 0)
        
        if current_thrust < 0.7 * expected_thrust:
            # Significant thrust loss, might be serious
            if engine.current_throttle > 0.8:
                # Already at high throttle, can't compensate
                engine.shutdown()
                return {
                    "action": "shutdown",
                    "reason": "cannot_achieve_required_thrust",
                    "status": "safe_shutdown"
                }
            else:
                # Try increasing throttle to compensate
                new_throttle = min(1.0, engine.current_throttle * 1.3)
                engine.set_throttle(new_throttle)
                return {
                    "action": "increase_throttle",
                    "new_throttle": new_throttle,
                    "status": "compensating"
                }
        else:
            # Minor thrust loss, adjust throttle slightly
            new_throttle = min(1.0, engine.current_throttle * 1.1)
            engine.set_throttle(new_throttle)
            return {
                "action": "adjust_throttle",
                "new_throttle": new_throttle,
                "status": "compensating"
            }
        
    def handle_sensor_failure(self, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sensor failure."""
        sensor_name = fault_data.get("sensor_name", "unknown")
        is_critical = fault_data.get("is_critical", False)
        
        if is_critical:
            # Critical sensor failure, safe shutdown
            engine.shutdown()
            return {
                "action": "shutdown",
                "reason": f"critical_sensor_failure_{sensor_name}",
                "status": "safe_shutdown"
            }
        else:
            # Non-critical sensor, switch to redundant or estimated value
            return {
                "action": "use_redundant_sensor",
                "sensor": sensor_name,
                "status": "operating_with_redundancy"
            }
    
    def handle_fault(self, fault_type: str, engine, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a specific fault type.
        
        Args:
            fault_type: Type of fault to handle
            engine: Engine reference
            fault_data: Fault data
            
        Returns:
            Action result
        """
        if fault_type in self.recovery_actions:
            return self.recovery_actions[fault_type](engine, fault_data)
        else:
            # Unknown fault type, use conservative approach
            if engine.current_throttle > 0.5:
                # Reduce throttle for safety
                new_throttle = engine.current_throttle * 0.7
                engine.set_throttle(new_throttle)
                return {
                    "action": "reduce_throttle",
                    "new_throttle": new_throttle,
                    "status": "unknown_fault_safety_measure"
                }
            else:
                # Already at low throttle, shut down
                engine.shutdown()
                return {
                    "action": "shutdown",
                    "reason": "unknown_fault_safety_measure",
                    "status": "safe_shutdown"
                }


class PropulsionHealthMonitor:
    """
    Monitors propulsion system health and performance metrics.
    Tracks parameter trends, detects anomalies, and predicts potential failures.
    """
    def __init__(self, parameter_thresholds: Dict[str, Tuple[float, float]], history_length: int = 100):
        self.parameter_thresholds = parameter_thresholds
        self.history_length = history_length
        self.parameter_history: Dict[str, List[Tuple[float, float]]] = {}  # param_name -> [(timestamp, value), ...]
        self.health_metrics: Dict[str, float] = {}
        self.anomaly_scores: Dict[str, float] = {}
        self.last_update_time = time.time()
        
    def record_parameters(self, parameters: Dict[str, float]) -> None:
        """Record current parameter values with timestamp"""
        current_time = time.time()
        
        for param_name, value in parameters.items():
            if param_name not in self.parameter_history:
                self.parameter_history[param_name] = []
                
            self.parameter_history[param_name].append((current_time, value))
            
            # Keep history within length limit
            if len(self.parameter_history[param_name]) > self.history_length:
                self.parameter_history[param_name] = self.parameter_history[param_name][-self.history_length:]
                
        self.last_update_time = current_time
    
    def calculate_health_metrics(self) -> Dict[str, float]:
        """Calculate health metrics for all parameters"""
        for param_name, history in self.parameter_history.items():
            if len(history) < 10:  # Need enough data points
                continue
                
            # Extract values and timestamps
            timestamps, values = zip(*history)
            values_array = np.array(values)
            
            # Calculate trend (slope of linear regression)
            trend = self._calculate_trend(timestamps, values)
            
            # Calculate volatility
            volatility = np.std(values_array)
            
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(values_array)
            self.anomaly_scores[param_name] = anomaly_score
            
            # Calculate health score (1.0 = perfect health, 0.0 = critical)
            if param_name in self.parameter_thresholds:
                min_val, max_val = self.parameter_thresholds[param_name]
                latest_value = values[-1]
                
                # Distance from optimal range (middle of the range)
                optimal = (min_val + max_val) / 2
                range_size = max_val - min_val
                
                if range_size > 0:
                    # Normalize distance from optimal
                    distance = abs(latest_value - optimal) / (range_size / 2)
                    
                    # Combine factors for health score
                    trend_factor = max(0.0, -trend * 5.0)  # Negative trend reduces health
                    volatility_factor = min(1.0, volatility * 2.0)
                    anomaly_factor = min(1.0, anomaly_score / 3.0)
                    distance_factor = min(1.0, distance)
                    
                    health_score = 1.0 - (0.3 * trend_factor + 0.2 * volatility_factor + 
                                         0.3 * anomaly_factor + 0.2 * distance_factor)
                    
                    # Ensure health score is in [0, 1]
                    health_score = max(0.0, min(1.0, health_score))
                    self.health_metrics[param_name] = health_score
        
        return self.health_metrics
    
    def _calculate_trend(self, timestamps: Tuple[float, ...], values: Tuple[float, ...]) -> float:
        """Calculate trend using linear regression"""
        if len(timestamps) < 2:
            return 0.0
            
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
    
    def _calculate_anomaly_score(self, values: np.ndarray) -> float:
        """Calculate anomaly score using Z-score method"""
        if len(values) < 10:
            return 0.0
            
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
            
        # Calculate Z-scores
        z_scores = np.abs((values - mean) / std)
        
        # Return the maximum Z-score from recent values
        recent_z = z_scores[-max(1, len(z_scores)//5):]
        return float(np.max(recent_z))
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall propulsion system health status"""
        # Calculate latest metrics
        self.calculate_health_metrics()
        
        # Calculate overall health as average of parameter health scores
        if not self.health_metrics:
            overall_health = 1.0  # Default to perfect health if no data
        else:
            overall_health = sum(self.health_metrics.values()) / len(self.health_metrics)
        
        # Determine status based on health score
        if overall_health >= 0.9:
            status = "optimal"
        elif overall_health >= 0.7:
            status = "good"
        elif overall_health >= 0.4:
            status = "degraded"
        else:
            status = "critical"
        
        # Identify parameters with lowest health scores
        param_health = [(param, score) for param, score in self.health_metrics.items()]
        param_health.sort(key=lambda x: x[1])
        
        # Identify parameters with highest anomaly scores
        anomalies = [(param, score) for param, score in self.anomaly_scores.items() if score > 2.0]
        anomalies.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'overall_health': overall_health,
            'status': status,
            'parameter_health': dict(self.health_metrics),
            'concerning_parameters': param_health[:3] if param_health else [],
            'anomalies': anomalies[:3] if anomalies else [],
            'last_updated': self.last_update_time
        }
    
    def predict_failures(self, prediction_horizon: float = 3600.0) -> Dict[str, Any]:
        """
        Predict potential failures within the specified time horizon
        
        Args:
            prediction_horizon: Time horizon in seconds
            
        Returns:
            Dictionary with failure predictions
        """
        predictions = {}
        
        for param_name, history in self.parameter_history.items():
            if len(history) < 20:  # Need enough data for prediction
                continue
                
            if param_name not in self.parameter_thresholds:
                continue
                
            min_val, max_val = self.parameter_thresholds[param_name]
            timestamps, values = zip(*history)
            
            # Convert to numpy arrays
            timestamps_array = np.array(timestamps)
            values_array = np.array(values)
            
            # Normalize timestamps to start from 0
            timestamps_array = timestamps_array - timestamps_array[0]
            
            # Fit a 2nd degree polynomial for prediction
            coeffs = np.polyfit(timestamps_array, values_array, 2)
            poly = np.poly1d(coeffs)
            
            # Current time relative to history start
            current_time = timestamps_array[-1]
            
            # Check if parameter will exceed limits within horizon
            will_exceed = False
            time_to_failure = None
            
            # Check at multiple points within the horizon
            check_points = np.linspace(current_time, current_time + prediction_horizon, 20)
            for t in check_points:
                predicted_value = poly(t)
                if predicted_value < min_val or predicted_value > max_val:
                    will_exceed = True
                    time_to_failure = t - current_time
                    break
            
            if will_exceed and time_to_failure is not None:
                # Calculate confidence based on R² of the fit
                residuals = values_array - poly(timestamps_array)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((values_array - np.mean(values_array))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                predictions[param_name] = {
                    'time_to_failure': time_to_failure,
                    'predicted_value': float(poly(current_time + time_to_failure)),
                    'threshold_exceeded': 'min' if poly(current_time + time_to_failure) < min_val else 'max',
                    'confidence': r_squared
                }
        
        return {
            'predicted_failures': predictions,
            'prediction_horizon': prediction_horizon,
            'timestamp': time.time()
        }

class FaultTolerantController:
    """
    Fault-tolerant propulsion control system.
    Monitors engine parameters, detects faults, and applies recovery actions.
    """
    def __init__(self, engine, redundancy_level: int = 1):
        """
        Initialize fault-tolerant controller.
        
        Args:
            engine: Reference to engine
            redundancy_level: Level of redundancy (0=none, 1=critical only, 2=full)
        """
        self.engine = engine
        self.redundancy_level = redundancy_level
        
        # Initialize fault detector with parameter limits
        self.fault_detector = FaultDetector({
            "chamber_pressure_Pa": (0.5 * engine.chamber_pressure_Pa, 1.2 * engine.chamber_pressure_Pa),
            "chamber_temp_K": (500, 3500),
            "thrust_N": (0, 1.1 * engine.max_thrust_N),
            "mixture_ratio": (1.8, 3.2),  # Typical range for LOX/Kerosene
            "fuel_flow_kgps": (0, 50),
            "oxidizer_flow_kgps": (0, 150)
        })
        
        # Initialize fault handler
        self.fault_handler = FaultHandler()
        
        # Initialize health monitor with same parameter thresholds
        self.health_monitor = PropulsionHealthMonitor({
            "chamber_pressure_Pa": (0.5 * engine.chamber_pressure_Pa, 1.2 * engine.chamber_pressure_Pa),
            "chamber_temp_K": (500, 3500),
            "thrust_N": (0, 1.1 * engine.max_thrust_N),
            "mixture_ratio": (1.8, 3.2),
            "fuel_flow_kgps": (0, 50),
            "oxidizer_flow_kgps": (0, 150)
        })
        
        # Fault classification
        self.fault_classifier = {
            "chamber_pressure_Pa": lambda v, limits: "chamber_pressure_high" if v > limits[1] else "chamber_pressure_low",
            "chamber_temp_K": lambda v, limits: "temperature_high" if v > limits[1] else "temperature_low",
            "fuel_flow_kgps": lambda v, limits: "flow_rate_high" if v > limits[1] else "flow_rate_low",
            "oxidizer_flow_kgps": lambda v, limits: "flow_rate_high" if v > limits[1] else "flow_rate_low",
            "mixture_ratio": lambda v, limits: "mixture_ratio_off" if v < limits[0] or v > limits[1] else None,
            "thrust_N": lambda v, expected: "thrust_low" if v < 0.9 * expected else None
        }
        
        # Controller state
        self.state = "nominal"
        self.fault_recovery_actions = {}
        self.last_update_time = time.time()
        self.control_history = []
        
    def update(self, engine_state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Update fault-tolerant controller.
        
        Args:
            engine_state: Current engine state
            dt: Time step in seconds
            
        Returns:
            Controller actions and state
        """
        # Record update time
        current_time = time.time()
        self.last_update_time = current_time
        
        # Extract parameters to monitor
        parameters = {
            "chamber_pressure_Pa": engine_state.get("chamber_pressure_Pa", self.engine.chamber_pressure_Pa * self.engine.current_throttle),
            "chamber_temp_K": engine_state.get("chamber_temp_K", 300 + 2000 * self.engine.current_throttle),
            "thrust_N": engine_state.get("thrust_N", 0),
            "fuel_flow_kgps": engine_state.get("fuel_flow_kgps", 0),
            "oxidizer_flow_kgps": engine_state.get("oxidizer_flow_kgps", 0)
        }
        
        # Calculate mixture ratio if both flows are present
        if parameters["fuel_flow_kgps"] > 0 and parameters["oxidizer_flow_kgps"] > 0:
            parameters["mixture_ratio"] = parameters["oxidizer_flow_kgps"] / parameters["fuel_flow_kgps"]
        
        # Update health monitor with current parameters
        self.health_monitor.record_parameters(parameters)
        
        # Detect faults
        detected_faults = self.fault_detector.detect_faults(parameters)
        
        # Process any detected faults
        actions = {}
        if detected_faults:
            self.state = "fault_detected"
            
            # Classify and handle each fault
            for param_name, fault_data in detected_faults.items():
                if param_name in self.fault_classifier:
                    classifier_func = self.fault_classifier[param_name]
                    fault_type = classifier_func(fault_data["value"], (fault_data["min_limit"], fault_data["max_limit"]))
                    
                    if fault_type:
                        # Handle the fault
                        action_result = self.fault_handler.handle_fault(fault_type, self.engine, {
                            **fault_data,
                            "parameter": param_name,
                            "fault_type": fault_type
                        })
                        
                        actions[fault_type] = action_result
                        self.fault_recovery_actions[fault_type] = action_result
        else:
            # No active faults, check if we're recovering
            if self.state == "fault_recovery":
                # Check if we can return to nominal operation
                all_recovered = True
                for fault_type, action in self.fault_recovery_actions.items():
                    if action["status"] in ["recovering", "compensating", "degraded_operation"]:
                        # Still recovering, check if we can improve
                        if fault_type in ["chamber_pressure_low", "temperature_low", "flow_rate_low", "thrust_low"]:
                            # These faults might allow increasing throttle during recovery
                            if self.engine.current_throttle < 0.9:
                                new_throttle = min(0.9, self.engine.current_throttle * 1.05)
                                self.engine.set_throttle(new_throttle)
                                actions[f"recovery_{fault_type}"] = {
                                    "action": "increase_throttle_during_recovery",
                                    "new_throttle": new_throttle
                                }
                        
                        all_recovered = False
                
                if all_recovered:
                    self.state = "nominal"
                    self.fault_recovery_actions = {}
            else:
                self.state = "nominal"
        
        # Get health status
        health_status = self.health_monitor.get_system_health()
        
        # Record control action
        control_record = {
            "timestamp": current_time,
            "state": self.state,
            "detected_faults": detected_faults,
            "actions": actions,
            "engine_throttle": self.engine.current_throttle,
            "engine_running": self.engine.running,
            "health_status": health_status
        }
        self.control_history.append(control_record)
        
        return {
            "controller_state": self.state,
            "detected_faults": detected_faults,
            "actions": actions,
            "engine_state": {
                "throttle": self.engine.current_throttle,
                "running": self.engine.running
            },
            "health_status": health_status
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report for propulsion system.
        
        Returns:
            Dictionary with health metrics and predictions
        """
        health_status = self.health_monitor.get_system_health()
        failure_predictions = self.health_monitor.predict_failures()
        
        return {
            "health_status": health_status,
            "failure_predictions": failure_predictions,
            "fault_history": self.fault_detector.fault_history[-10:],
            "controller_state": self.state,
            "active_faults": self.fault_detector.active_faults,
            "recovery_actions": self.fault_recovery_actions,
            "timestamp": time.time()
        }

    def get_redundant_sensor_value(self, primary_value: float, sensor_name: str) -> float:
        """
        Get value from redundant sensor or estimate.
        In a real system, would read from physical redundant sensors.
        
        Args:
            primary_value: Primary sensor value
            sensor_name: Sensor name
            
        Returns:
            Redundant or estimated value
        """
        if self.redundancy_level == 0:
            # No redundancy
            return primary_value
            
        # Simulate redundant sensor with small variation
        if self.redundancy_level >= 2:
            # Full redundancy - simulate physical redundant sensor
            redundant_value = primary_value * (1 + np.random.normal(0, 0.02))
            return redundant_value
        else:
            # Critical redundancy only - check if this is a critical sensor
            if sensor_name in ["chamber_pressure_Pa", "thrust_N"]:
                # Critical sensor, simulate redundant
                redundant_value = primary_value * (1 + np.random.normal(0, 0.05))
                return redundant_value
            else:
                # Non-critical, use primary
                return primary_value
    
    def emergency_shutdown(self) -> Dict[str, Any]:
        """
        Perform emergency shutdown sequence.
        
        Returns:
            Shutdown status
        """
        # Immediate engine shutdown
        self.engine.shutdown()
        
        # Record emergency action
        shutdown_record = {
            "timestamp": time.time(),
            "action": "emergency_shutdown",
            "reason": "manual_command",
            "engine_state": {
                "throttle": self.engine.current_throttle,
                "running": self.engine.running
            }
        }
        self.control_history.append(shutdown_record)
        
        return {
            "status": "emergency_shutdown_complete",
            "engine_running": self.engine.running
        }