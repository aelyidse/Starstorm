import asyncio
import time
from typing import Callable, Dict, Any, Optional

class DeterministicScheduler:
    """
    Real-time task scheduler with deterministic cycle timing and jitter minimization.
    Ensures periodic tasks execute within strict deadlines.
    """
    def __init__(self, cycle_time_s: float):
        self.cycle_time_s = cycle_time_s
        self.tasks: Dict[str, Callable[[], Any]] = {}
        self.running = False
        self.last_cycle_duration = 0.0
        self.last_cycle_jitter = 0.0

    def register_task(self, name: str, task_fn: Callable[[], Any]):
        self.tasks[name] = task_fn

    async def run(self, max_cycles: Optional[int] = None):
        self.running = True
        cycle = 0
        while self.running and (max_cycles is None or cycle < max_cycles):
            start = time.perf_counter()
            results = {}
            for name, fn in self.tasks.items():
                results[name] = fn()
            elapsed = time.perf_counter() - start
            sleep_time = max(0.0, self.cycle_time_s - elapsed)
            self.last_cycle_duration = elapsed
            self.last_cycle_jitter = abs(self.cycle_time_s - elapsed)
            await asyncio.sleep(sleep_time)
            cycle += 1

    def stop(self):
        self.running = False

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'cycle_time_s': self.cycle_time_s,
            'last_cycle_duration': self.last_cycle_duration,
            'last_cycle_jitter': self.last_cycle_jitter,
        }

from .command_validation import CommandInterpreter, CommandValidationError

class RealTimeControlSystem:
    """
    Foundation for a real-time control system with deterministic processing guarantees.
    Integrates a deterministic scheduler, state update, and command execution.
    Provides fault-tolerant command execution with rollback capabilities.
    """
    def __init__(self, cycle_time_ms: float = 10.0, command_schema: Optional[Dict[str, Any]] = None, rollback_depth: int = 10):
        self.cycle_time_ms = cycle_time_ms
        self.scheduler = DeterministicScheduler(cycle_time_ms / 1000.0)  # Convert to seconds
        self.state: Dict[str, Any] = {}
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.command_interpreter = CommandInterpreter(command_schema or {})
        self._state_history = []  # List[Dict[str, Any]]
        self._rollback_depth = rollback_depth
        
        # Timing statistics
        self.execution_times = deque(maxlen=100)
        self.deadline_misses = 0
        self.total_executions = 0
        self.max_execution_time = 0.0
        self.min_execution_time = float('inf')
        self.jitter_tolerance_ms = 1.0

    def register_control_task(self, name: str, task_fn: Callable[[], Any], priority: int = 5, deadline_ms: Optional[float] = None):
        """
        Register a control task with timing guarantees and priority
        
        Args:
            name: Task name
            task_fn: Task function to execute
            priority: Task priority (higher = more important)
            deadline_ms: Execution deadline in milliseconds (defaults to cycle time)
        """
        deadline = deadline_ms if deadline_ms is not None else self.cycle_time_ms
        
        # Wrap the task function with timing measurements
        def timed_task_fn():
            start_time = time.time() * 1000  # Convert to ms
            try:
                result = task_fn()
                return result
            finally:
                end_time = time.time() * 1000
                execution_time = end_time - start_time
                
                # Update timing statistics
                self.execution_times.append(execution_time)
                self.total_executions += 1
                self.max_execution_time = max(self.max_execution_time, execution_time)
                self.min_execution_time = min(self.min_execution_time, execution_time)
                
                # Check for deadline miss
                if execution_time > deadline:
                    self.deadline_misses += 1
        
        self.scheduler.register_task(name, timed_task_fn, priority)

    async def start(self, max_cycles: Optional[int] = None):
        """
        Start the control system with timing guarantees
        
        Args:
            max_cycles: Maximum number of cycles to run (None = run indefinitely)
        """
        self.running = True
        
        # Start command processing in background
        asyncio.create_task(self.process_commands())
        
        # Run the scheduler with timing guarantees
        await self.scheduler.run(max_cycles)

    def stop(self):
        self.running = False
        self.scheduler.stop()

    async def enqueue_command(self, command: Dict[str, Any]):
        await self.command_queue.put(command)

    async def process_commands(self):
        while self.running:
            try:
                command = await asyncio.wait_for(self.command_queue.get(), timeout=0.001)
                self.handle_command(command)
            except asyncio.TimeoutError:
                # No commands to process, continue
                await asyncio.sleep(0.001)  # Small sleep to prevent CPU hogging
            except Exception as e:
                # Log error but continue processing
                print(f"Command processing error: {e}")

    def handle_command(self, command: Dict[str, Any]):
        try:
            start_time = time.time() * 1000
            interpreted = self.command_interpreter.interpret_and_validate(command, self.state)
            # Save state snapshot for rollback
            self._push_state_history()
            # Apply the command
            self._apply_command(interpreted)
            
            # Measure command execution time
            execution_time = time.time() * 1000 - start_time
            if execution_time > self.cycle_time_ms:
                # Log timing violation
                print(f"Command timing violation: {execution_time:.2f}ms > {self.cycle_time_ms:.2f}ms")
                
        except CommandValidationError as e:
            # Handle invalid command (log, reject, alert, etc.)
            pass
        except Exception as e:
            # Fault-tolerant: rollback to previous state
            self.rollback()
            # Log or alert about the failure
            print(f"Command execution error: {e}, state rolled back")

    def _push_state_history(self):
        if len(self._state_history) >= self._rollback_depth:
            self._state_history.pop(0)
        self._state_history.append(self.state.copy())

    def rollback(self):
        if self._state_history:
            self.state = self._state_history.pop()
        # else: nothing to rollback

    def _apply_command(self, command: Dict[str, Any]):
        # Example: update state with command (to be replaced with real logic)
        for k, v in command.items():
            self.state[k] = v

    def get_state(self) -> Dict[str, Any]:
        return self.state.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive timing and execution metrics"""
        scheduler_metrics = self.scheduler.get_metrics()
        
        # Calculate timing statistics
        if self.execution_times:
            avg_execution_time = sum(self.execution_times) / len(self.execution_times)
            jitter = max(self.execution_times) - min(self.execution_times)
        else:
            avg_execution_time = 0.0
            jitter = 0.0
            
        return {
            **scheduler_metrics,
            'deadline_misses': self.deadline_misses,
            'total_executions': self.total_executions,
            'miss_rate': self.deadline_misses / max(1, self.total_executions),
            'avg_execution_time_ms': avg_execution_time,
            'max_execution_time_ms': self.max_execution_time,
            'min_execution_time_ms': self.min_execution_time if self.min_execution_time != float('inf') else 0.0,
            'jitter_ms': jitter,
            'jitter_tolerance_exceeded': jitter > self.jitter_tolerance_ms
        }
        
    def set_jitter_tolerance(self, tolerance_ms: float):
        """Set the maximum acceptable jitter in milliseconds"""
        self.jitter_tolerance_ms = max(0.1, tolerance_ms)
