from typing import Dict, Any, List, Optional, Callable, Tuple, Set
import time
import asyncio
import heapq
from collections import deque
from .task_scheduler import PriorityTaskScheduler

class ExecutionTimingGuarantee:
    """
    Provides execution timing guarantees for real-time control tasks.
    Monitors execution times, detects deadline misses, and provides statistics.
    """
    def __init__(self, deadline_ms: float, jitter_tolerance_ms: float = 1.0):
        self.deadline_ms = deadline_ms
        self.jitter_tolerance_ms = jitter_tolerance_ms
        self.execution_times = deque(maxlen=100)  # Store last 100 execution times
        self.deadline_misses = 0
        self.total_executions = 0
        self.max_execution_time = 0.0
        self.min_execution_time = float('inf')
        self.last_start_time = 0.0
        self.last_end_time = 0.0
        
    def start_execution(self):
        """Mark the start of task execution"""
        self.last_start_time = time.time() * 1000  # Convert to ms
        
    def end_execution(self) -> Dict[str, Any]:
        """
        Mark the end of task execution and calculate metrics
        
        Returns:
            Execution statistics
        """
        self.last_end_time = time.time() * 1000  # Convert to ms
        execution_time = self.last_end_time - self.last_start_time
        
        # Update statistics
        self.execution_times.append(execution_time)
        self.total_executions += 1
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.min_execution_time = min(self.min_execution_time, execution_time)
        
        # Check for deadline miss
        deadline_missed = execution_time > self.deadline_ms
        if deadline_missed:
            self.deadline_misses += 1
            
        return {
            'execution_time_ms': execution_time,
            'deadline_missed': deadline_missed,
            'deadline_ms': self.deadline_ms,
            'jitter_ms': self._calculate_jitter()
        }
        
    def _calculate_jitter(self) -> float:
        """Calculate timing jitter from recent executions"""
        if len(self.execution_times) < 2:
            return 0.0
            
        # Jitter is the variation in execution times
        return max(self.execution_times) - min(self.execution_times)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive timing statistics"""
        if not self.execution_times:
            return {
                'deadline_ms': self.deadline_ms,
                'deadline_misses': 0,
                'total_executions': 0,
                'miss_rate': 0.0,
                'avg_execution_time_ms': 0.0,
                'max_execution_time_ms': 0.0,
                'min_execution_time_ms': 0.0,
                'jitter_ms': 0.0
            }
            
        avg_time = sum(self.execution_times) / len(self.execution_times)
        
        return {
            'deadline_ms': self.deadline_ms,
            'deadline_misses': self.deadline_misses,
            'total_executions': self.total_executions,
            'miss_rate': self.deadline_misses / max(1, self.total_executions),
            'avg_execution_time_ms': avg_time,
            'max_execution_time_ms': self.max_execution_time,
            'min_execution_time_ms': self.min_execution_time,
            'jitter_ms': self._calculate_jitter(),
            'jitter_tolerance_exceeded': self._calculate_jitter() > self.jitter_tolerance_ms
        }

# Add this import at the top of the file
from .stability_analysis import ControlSystemStabilityMonitor

class EnhancedRealTimeControlSystem:
    """
    Enhanced real-time control system with guaranteed execution timing.
    Provides deterministic scheduling, execution time monitoring, and fault tolerance.
    Supports task prioritization, deadline enforcement, and execution statistics.
    """
    def __init__(self, cycle_time_ms: float = 10.0, rollback_depth: int = 10):
        self.cycle_time_ms = cycle_time_ms
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_priorities: Dict[str, int] = {}
        self.state: Dict[str, Any] = {}
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self._state_history = deque(maxlen=rollback_depth)
        self.last_cycle_time = 0.0
        self.cycle_count = 0
        self.timing_violations = 0
        
        # Initialize the priority task scheduler
        self.task_scheduler = PriorityTaskScheduler(preemptive=True, resource_aware=True)
        
        # Add rate monitor
        self.rate_monitor = LoopRateMonitor(target_rate_hz=1000.0/cycle_time_ms)
        
        # Add stability monitor
        self.stability_monitor = ControlSystemStabilityMonitor()
    
    def register_task(self, name: str, task_fn: Callable[[], Any], priority: int = 5, 
                     deadline_ms: Optional[float] = None, resources: Optional[Set[str]] = None) -> None:
        """
        Register a control task with timing guarantees
        
        Args:
            name: Task name
            task_fn: Task function to execute
            priority: Task priority (higher = more important)
            deadline_ms: Execution deadline in milliseconds (defaults to cycle time)
            resources: Set of resource names required by this task
        """
        deadline = deadline_ms if deadline_ms is not None else self.cycle_time_ms
        
        # Register with the task scheduler
        self.task_scheduler.register_task(
            name=name,
            function=task_fn,
            priority=priority,
            deadline_ms=deadline,
            resources=resources
        )
        
        # Keep backward compatibility
        self.tasks[name] = {
            'function': task_fn,
            'timing': ExecutionTimingGuarantee(deadline),
            'last_result': None,
            'error_count': 0
        }
        self.task_priorities[name] = priority
        
    def _push_state_history(self):
        """Save current state for potential rollback"""
        self._state_history.append(self.state.copy())
        
    def rollback(self) -> bool:
        """
        Rollback to previous state
        
        Returns:
            True if rollback successful, False otherwise
        """
        if not self._state_history:
            return False
            
        self.state = self._state_history.pop()
        return True
        
    async def execute_cycle(self) -> Dict[str, Any]:
        """
        Execute a single control cycle with timing guarantees
        
        Returns:
            Cycle execution statistics
        """
        # Start rate monitoring for this cycle
        self.rate_monitor.start_cycle()
        
        cycle_start = time.time() * 1000  # Convert to ms
        self._push_state_history()
        
        # Schedule all tasks for this cycle
        for name in self.tasks:
            self.task_scheduler.schedule_task(name)
        
        # Run the scheduler for one cycle duration
        await self.task_scheduler.run(max_runtime_s=self.cycle_time_ms/1000.0)
        
        # Process any pending commands
        await self._process_pending_commands()
        
        # Update stability monitor with control signals
        for name, value in self.state.items():
            if isinstance(value, (int, float)):
                self.stability_monitor.update_signal(name, value)
        
        # Check for stability alerts
        stability_alerts = self.stability_monitor.get_alerts()
        if stability_alerts:
            # Log alerts or take corrective action
            print(f"Stability alerts detected: {len(stability_alerts)}")
        
        # Calculate cycle statistics
        cycle_end = time.time() * 1000
        cycle_duration = cycle_end - cycle_start
        cycle_overrun = cycle_duration > self.cycle_time_ms
        
        if cycle_overrun:
            self.timing_violations += 1
            
        self.last_cycle_time = cycle_duration
        self.cycle_count += 1
        
        # Get task statistics from scheduler
        task_results = self.task_scheduler.get_task_statistics()
        
        # End rate monitoring and get rate statistics
        rate_stats = self.rate_monitor.end_cycle()
        
        return {
            'cycle_duration_ms': cycle_duration,
            'cycle_overrun': cycle_overrun,
            'timing_violations': self.timing_violations,
            'task_results': task_results,
            'rate_stats': rate_stats
        }
    
    async def run(self, duration_s: Optional[float] = None):
        """
        Run the control system for a specified duration
        
        Args:
            duration_s: Duration in seconds (None = run indefinitely)
        """
        self.running = True
        start_time = time.time()
        
        while self.running:
            if duration_s is not None and time.time() - start_time > duration_s:
                break
                
            # Execute cycle and collect statistics
            cycle_stats = await self.execute_cycle()
            
            # Use rate monitor to sleep until next cycle
            # This provides adaptive timing based on system load
            await self.rate_monitor.sleep_until_next_cycle()
    
    async def _process_pending_commands(self):
        """Process any commands in the queue without blocking"""
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                self._apply_command(command)
            except asyncio.QueueEmpty:
                break
    
    def _apply_command(self, command: Dict[str, Any]):
        """Apply a command to the system state"""
        # Implement command application logic
        for k, v in command.items():
            self.state[k] = v
    
    async def enqueue_command(self, command: Dict[str, Any]):
        """Add a command to the execution queue"""
        await self.command_queue.put(command)
    
    def stop(self):
        """Stop the control system"""
        self.running = False
    
    def get_task_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get timing statistics for all tasks"""
        return {name: task['timing'].get_statistics() for name, task in self.tasks.items()}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        scheduler_stats = self.task_scheduler.get_scheduler_statistics()
        rate_stats = self.rate_monitor.get_statistics()
        
        # Add stability information
        stability_report = self.stability_monitor.get_stability_report()
        
        return {
            'cycle_time_ms': self.cycle_time_ms,
            'last_cycle_time_ms': self.last_cycle_time,
            'cycle_count': self.cycle_count,
            'timing_violations': self.timing_violations,
            'violation_rate': self.timing_violations / max(1, self.cycle_count),
            'task_count': len(self.tasks),
            'state_size': len(self.state),
            'scheduler_stats': scheduler_stats,
            'rate_stats': rate_stats
        }

class LoopRateMonitor:
    """
    Monitors and adjusts control loop execution rates to maintain timing consistency.
    Provides adaptive rate adjustment based on system load and performance metrics.
    """
    def __init__(self, target_rate_hz: float, adjustment_threshold: float = 0.1, 
                 max_adjustment_factor: float = 0.5, window_size: int = 20):
        self.target_rate_hz = target_rate_hz
        self.target_period_s = 1.0 / target_rate_hz
        self.adjustment_threshold = adjustment_threshold  # Fractional threshold to trigger adjustment
        self.max_adjustment_factor = max_adjustment_factor  # Maximum adjustment per cycle
        self.execution_times = deque(maxlen=window_size)
        self.cycle_periods = deque(maxlen=window_size)
        self.adjusted_periods = deque(maxlen=window_size)
        self.last_cycle_time = time.perf_counter()
        self.current_period = self.target_period_s
        self.adjustment_count = 0
        
    def start_cycle(self):
        """Mark the beginning of a control cycle"""
        now = time.perf_counter()
        if self.last_cycle_time:
            actual_period = now - self.last_cycle_time
            self.cycle_periods.append(actual_period)
        self.last_cycle_time = now
        
    def end_cycle(self) -> Dict[str, Any]:
        """
        Mark the end of a control cycle and calculate metrics
        
        Returns:
            Cycle timing statistics
        """
        execution_time = time.perf_counter() - self.last_cycle_time
        self.execution_times.append(execution_time)
        
        # Calculate next cycle period adjustment
        adjusted_period = self._adjust_period(execution_time)
        self.adjusted_periods.append(adjusted_period)
        
        return {
            'execution_time_s': execution_time,
            'target_period_s': self.target_period_s,
            'adjusted_period_s': adjusted_period,
            'rate_hz': 1.0 / adjusted_period if adjusted_period > 0 else 0.0,
            'load_factor': execution_time / self.target_period_s
        }
    
    def _adjust_period(self, execution_time: float) -> float:
        """
        Adaptively adjust the cycle period based on execution time
        
        Args:
            execution_time: Time taken to execute the current cycle
            
        Returns:
            Adjusted period for the next cycle
        """
        # If execution time exceeds target period, we need to slow down
        if execution_time > self.target_period_s:
            # Calculate how much to extend the period, but limit the adjustment
            extension = min(
                execution_time - self.target_period_s,
                self.target_period_s * self.max_adjustment_factor
            )
            self.current_period = self.target_period_s + extension
            self.adjustment_count += 1
        # If execution is significantly faster than target, we can speed up
        elif self.current_period > self.target_period_s and \
             execution_time < (1.0 - self.adjustment_threshold) * self.target_period_s:
            # Gradually move back toward target period
            reduction = min(
                self.current_period - self.target_period_s,
                self.target_period_s * self.max_adjustment_factor
            )
            self.current_period = self.current_period - (reduction * 0.5)  # More conservative when speeding up
            self.adjustment_count += 1
            
        return self.current_period
    
    async def sleep_until_next_cycle(self) -> float:
        """
        Sleep until the next cycle should begin
        
        Returns:
            Actual sleep time in seconds
        """
        execution_time = time.perf_counter() - self.last_cycle_time
        sleep_time = max(0.0, self.current_period - execution_time)
        
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
            return sleep_time
        return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rate monitoring statistics"""
        if not self.execution_times:
            return {
                'target_rate_hz': self.target_rate_hz,
                'current_rate_hz': self.target_rate_hz,
                'avg_execution_time_s': 0.0,
                'avg_period_s': 0.0,
                'jitter_s': 0.0,
                'load_factor': 0.0,
                'adjustment_count': 0
            }
            
        avg_execution = sum(self.execution_times) / len(self.execution_times)
        
        if self.cycle_periods:
            avg_period = sum(self.cycle_periods) / len(self.cycle_periods)
            period_jitter = max(self.cycle_periods) - min(self.cycle_periods)
        else:
            avg_period = self.target_period_s
            period_jitter = 0.0
            
        return {
            'target_rate_hz': self.target_rate_hz,
            'current_rate_hz': 1.0 / self.current_period,
            'avg_execution_time_s': avg_execution,
            'avg_period_s': avg_period,
            'jitter_s': period_jitter,
            'load_factor': avg_execution / self.target_period_s,
            'adjustment_count': self.adjustment_count,
            'overloaded': avg_execution > self.target_period_s
        }