from typing import Dict, Any, List, Optional, Set, Tuple, Callable
import time
import json
import logging
import os
from datetime import datetime

class AutonomyAuditLogger:
    """
    Provides comprehensive audit logging for autonomy decisions.
    Tracks level transitions, permission checks, and boundary violations
    with detailed context for accountability and traceability.
    """
    def __init__(self, log_file: Optional[str] = None, console_output: bool = True):
        self.log_entries: List[Dict[str, Any]] = []
        self.console_output = console_output
        self.log_file = log_file
        
        # Set up logging
        self.logger = logging.getLogger("autonomy_audit")
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_transition(self, from_level: str, to_level: str, success: bool, 
                      role: Optional[str] = None, reason: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> None:
        """Log an autonomy level transition attempt"""
        entry = {
            'event_type': 'transition',
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'from_level': from_level,
            'to_level': to_level,
            'success': success,
            'role': role,
            'reason': reason,
            'context': context or {}
        }
        self.log_entries.append(entry)
        
        # Log to configured outputs
        message = f"AUTONOMY TRANSITION: {from_level} -> {to_level} | Success: {success}"
        if role:
            message += f" | Role: {role}"
        if reason:
            message += f" | Reason: {reason}"
            
        if success:
            self.logger.info(message)
        else:
            self.logger.warning(message)
    
    def log_permission_check(self, operation: str, level: str, allowed: bool,
                           role: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a permission check for an operation"""
        entry = {
            'event_type': 'permission_check',
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'operation': operation,
            'level': level,
            'allowed': allowed,
            'role': role,
            'context': context or {}
        }
        self.log_entries.append(entry)
        
        # Log to configured outputs
        message = f"PERMISSION CHECK: {operation} at level {level} | Allowed: {allowed}"
        if role:
            message += f" | Role: {role}"
            
        if allowed:
            self.logger.info(message)
        else:
            self.logger.warning(message)
    
    def log_boundary_violation(self, operation: str, level: str, 
                             role: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a boundary violation attempt"""
        entry = {
            'event_type': 'boundary_violation',
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'operation': operation,
            'level': level,
            'role': role,
            'context': context or {}
        }
        self.log_entries.append(entry)
        
        # Log to configured outputs
        message = f"BOUNDARY VIOLATION: {operation} at level {level}"
        if role:
            message += f" | Role: {role}"
            
        self.logger.warning(message)
    
    def log_safety_check(self, from_level: str, to_level: str, check_name: str, 
                        passed: bool, details: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None) -> None:
        """Log a safety check result"""
        entry = {
            'event_type': 'safety_check',
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'from_level': from_level,
            'to_level': to_level,
            'check_name': check_name,
            'passed': passed,
            'details': details,
            'context': context or {}
        }
        self.log_entries.append(entry)
        
        # Log to configured outputs
        message = f"SAFETY CHECK: {from_level} -> {to_level} | Check: {check_name} | Passed: {passed}"
        if details:
            message += f" | Details: {details}"
            
        if passed:
            self.logger.info(message)
        else:
            self.logger.warning(message)
    
    def get_all_logs(self) -> List[Dict[str, Any]]:
        """Get all log entries"""
        return self.log_entries
    
    def get_logs_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get logs filtered by event type"""
        return [entry for entry in self.log_entries if entry['event_type'] == event_type]
    
    def export_logs(self, file_path: str, format: str = 'json') -> bool:
        """Export logs to a file in the specified format"""
        try:
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(self.log_entries, f, indent=2)
                return True
            elif format.lower() == 'csv':
                import csv
                with open(file_path, 'w', newline='') as f:
                    if not self.log_entries:
                        return True
                    
                    # Get all possible fields from all entries
                    fieldnames = set()
                    for entry in self.log_entries:
                        fieldnames.update(entry.keys())
                    
                    writer = csv.DictWriter(f, fieldnames=list(fieldnames))
                    writer.writeheader()
                    for entry in self.log_entries:
                        writer.writerow(entry)
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Failed to export logs: {str(e)}")
            return False

class AutonomyLevelManager:
    """
    Manages variable autonomy levels and enables smooth transitions between manual, supervised, and fully autonomous modes.
    Supports transition logic, status reporting, mode enforcement, and fine-grained permission models.
    """
    def __init__(self, levels: Optional[List[str]] = None, audit_log_file: Optional[str] = None):
        self.levels = levels or ['manual', 'supervised', 'autonomous']
        self.current_level = self.levels[0]
        self.transition_log: List[Dict[str, Any]] = []
        self.transitioning = False
        self.transition_start_time: Optional[float] = None
        self.transition_duration = 2.0  # seconds for smooth transition
        
        # Permission model - maps autonomy levels to sets of allowed operations
        self.permissions: Dict[str, Set[str]] = {
            'manual': {'sensor_read', 'status_report'},
            'supervised': {'sensor_read', 'status_report', 'navigation', 'data_collection'},
            'autonomous': {'sensor_read', 'status_report', 'navigation', 'data_collection', 
                          'mission_planning', 'resource_allocation', 'self_healing'}
        }
        
        # Role-based access control
        self.roles: Dict[str, Set[str]] = {
            'operator': {'manual', 'supervised'},
            'mission_commander': {'manual', 'supervised', 'autonomous'},
            'system': {'autonomous'}
        }
        
        # Current user role
        self.current_role: Optional[str] = None
        
        # Decision boundary enforcement
        self.boundary_violations: List[Dict[str, Any]] = []
        self.enforce_boundaries = True
        
        # Safety checks for transitions
        self.safety_checks: Dict[str, List[Callable[[], bool]]] = {}
        self.validation_failures: List[Dict[str, Any]] = []
        self.last_validation_result: Optional[Dict[str, Any]] = None
        
        # Initialize audit logger
        self.audit_logger = AutonomyAuditLogger(log_file=audit_log_file)
        
    def register_safety_check(self, transition: str, check_func: Callable[[], bool], check_name: Optional[str] = None) -> None:
        """
        Register a safety check function for a specific transition
        transition: format is 'from_level:to_level' or 'any:to_level' or 'from_level:any'
        check_func: function that returns True if safe to transition, False otherwise
        check_name: optional name for the check (for audit logging)
        """
        if transition not in self.safety_checks:
            self.safety_checks[transition] = []
        
        # Store the check name with the function for audit logging
        func_name = check_name or getattr(check_func, '__name__', f"check_{len(self.safety_checks[transition])}")
        self.safety_checks[transition].append((check_func, func_name))
    
    def validate_transition(self, from_level: str, to_level: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a transition from one level to another is safe
        Returns: (is_valid, reason_if_invalid)
        """
        # Check specific transition
        specific = f"{from_level}:{to_level}"
        # Check 'from any level' transition
        from_any = f"any:{to_level}"
        # Check 'to any level' transition
        to_any = f"{from_level}:any"
        
        # Collect all applicable checks
        checks = []
        if specific in self.safety_checks:
            checks.extend(self.safety_checks[specific])
        if from_any in self.safety_checks:
            checks.extend(self.safety_checks[from_any])
        if to_any in self.safety_checks:
            checks.extend(self.safety_checks[to_any])
            
        # If no checks defined, transition is valid by default
        if not checks:
            return True, None
            
        # Run all checks
        for i, (check, check_name) in enumerate(checks):
            try:
                check_result = check()
                # Log the safety check result
                self.audit_logger.log_safety_check(
                    from_level=from_level,
                    to_level=to_level,
                    check_name=check_name,
                    passed=check_result,
                    details=None if check_result else f"Check {check_name} failed"
                )
                
                if not check_result:
                    return False, f"Safety check {check_name} failed for transition {from_level} to {to_level}"
            except Exception as e:
                error_msg = f"Error in safety check {check_name}: {str(e)}"
                # Log the safety check error
                self.audit_logger.log_safety_check(
                    from_level=from_level,
                    to_level=to_level,
                    check_name=check_name,
                    passed=False,
                    details=error_msg
                )
                return False, error_msg
                
        return True, None
        
    def set_level(self, target_level: str, role: Optional[str] = None, system_state: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """
        Set the autonomy level with validation and safety checks
        Returns: (success, reason_if_failed)
        """
        context = {
            'system_state': system_state or {},
            'timestamp': time.time()
        }
        
        if target_level not in self.levels:
            reason = f"Unknown autonomy level: {target_level}"
            self._log_validation_failure(self.current_level, target_level, reason, role)
            # Audit log the failed transition
            self.audit_logger.log_transition(
                from_level=self.current_level,
                to_level=target_level,
                success=False,
                role=role,
                reason=reason,
                context=context
            )
            raise ValueError(reason)
            
        # Check role-based permissions
        if role and role in self.roles:
            if target_level not in self.roles[role]:
                reason = f"Role '{role}' cannot set autonomy level to '{target_level}'"
                self._log_validation_failure(self.current_level, target_level, reason, role)
                # Audit log the failed transition
                self.audit_logger.log_transition(
                    from_level=self.current_level,
                    to_level=target_level,
                    success=False,
                    role=role,
                    reason=reason,
                    context=context
                )
                raise PermissionError(reason)
            self.current_role = role
            
        if target_level == self.current_level:
            return True, None
            
        # Validate the transition
        is_valid, reason = self.validate_transition(self.current_level, target_level)
        if not is_valid:
            self._log_validation_failure(self.current_level, target_level, reason, role)
            # Audit log the failed transition
            self.audit_logger.log_transition(
                from_level=self.current_level,
                to_level=target_level,
                success=False,
                role=role,
                reason=reason,
                context=context
            )
            return False, reason
            
        # Begin transition
        self.transitioning = True
        self.transition_start_time = time.time()
        previous_level = self.current_level
        
        try:
            # Simulate smooth transition
            time.sleep(self.transition_duration)
            self.current_level = target_level
            
            # Log successful transition
            transition_data = {
                'from': previous_level,
                'to': target_level,
                'time': time.time(),
                'duration': self.transition_duration,
                'role': self.current_role,
                'validated': True
            }
            self.transition_log.append(transition_data)
            self.last_validation_result = {
                'success': True,
                'from': previous_level,
                'to': target_level,
                'time': time.time()
            }
            
            # Audit log the successful transition
            self.audit_logger.log_transition(
                from_level=previous_level,
                to_level=target_level,
                success=True,
                role=self.current_role,
                context=context
            )
            
            return True, None
            
        except Exception as e:
            # Rollback on error
            self.current_level = previous_level
            reason = f"Transition failed: {str(e)}"
            self._log_validation_failure(previous_level, target_level, reason, role)
            
            # Audit log the failed transition
            self.audit_logger.log_transition(
                from_level=previous_level,
                to_level=target_level,
                success=False,
                role=role,
                reason=reason,
                context=context
            )
            
            return False, reason
            
        finally:
            self.transitioning = False
    
    def _log_validation_failure(self, from_level: str, to_level: str, reason: Optional[str], role: Optional[str]) -> None:
        """Log a validation failure"""
        failure = {
            'from': from_level,
            'to': to_level,
            'time': time.time(),
            'reason': reason,
            'role': role
        }
        self.validation_failures.append(failure)
        self.last_validation_result = {
            'success': False,
            'from': from_level,
            'to': to_level,
            'time': time.time(),
            'reason': reason
        }

    def get_current_level(self) -> str:
        return self.current_level

    def is_transitioning(self) -> bool:
        return self.transitioning

    def get_transition_log(self) -> List[Dict[str, Any]]:
        return self.transition_log
        
    def get_validation_failures(self) -> List[Dict[str, Any]]:
        """Get the history of validation failures"""
        return self.validation_failures
        
    def get_last_validation_result(self) -> Optional[Dict[str, Any]]:
        """Get the result of the last validation attempt"""
        return self.last_validation_result
        
    def check_permission(self, operation: str) -> bool:
        """Check if the current autonomy level allows a specific operation"""
        if not self.current_level or self.transitioning:
            return False
        allowed = operation in self.permissions.get(self.current_level, set())
        
        # Audit log the permission check
        self.audit_logger.log_permission_check(
            operation=operation,
            level=self.current_level,
            allowed=allowed,
            role=self.current_role
        )
        
        if not allowed and self.enforce_boundaries:
            self.log_boundary_violation(operation)
        return allowed
    
    def enforce_decision_boundaries(self, enforce: bool = True) -> None:
        """Enable or disable decision boundary enforcement"""
        self.enforce_boundaries = enforce
    
    def log_boundary_violation(self, operation: str) -> None:
        """Log attempted boundary violations for auditing"""
        violation = {
            'operation': operation,
            'attempted_level': self.current_level,
            'time': time.time(),
            'role': self.current_role
        }
        self.boundary_violations.append(violation)
        
        # Audit log the boundary violation
        self.audit_logger.log_boundary_violation(
            operation=operation,
            level=self.current_level,
            role=self.current_role
        )
    
    def get_boundary_violations(self) -> List[Dict[str, Any]]:
        """Retrieve log of boundary violation attempts"""
        return self.boundary_violations
        
    def get_allowed_operations(self) -> Set[str]:
        """Get all operations allowed at the current autonomy level"""
        if not self.current_level:
            return set()
        return self.permissions.get(self.current_level, set()).copy()
        
    def set_role_permissions(self, role: str, allowed_levels: Set[str]) -> None:
        """Update role-based permissions"""
        # Validate all levels exist
        for level in allowed_levels:
            if level not in self.levels:
                raise ValueError(f"Unknown autonomy level: {level}")
        self.roles[role] = allowed_levels
        
    def attempt_operation(self, operation: str) -> bool:
        """Attempt to perform an operation with boundary enforcement"""
        if self.check_permission(operation):
            return True
        return False
        
    def export_audit_logs(self, file_path: str, format: str = 'json') -> bool:
        """Export audit logs to a file"""
        return self.audit_logger.export_logs(file_path, format)
        
    def get_audit_logs(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit logs, optionally filtered by event type"""
        if event_type:
            return self.audit_logger.get_logs_by_type(event_type)
        return self.audit_logger.get_all_logs()
