from typing import Dict, Any, List, Optional, Callable, Tuple, Set
import time
import heapq
from collections import deque
import asyncio

class Task:
    """
    Represents a schedulable task with priority, deadline, and resource requirements.
    """
    def __init__(self, 
                 name: str, 
                 function: Callable[[], Any], 
                 priority: int = 0, 
                 deadline_ms: Optional[float] = None,
                 resources: Optional[Set[str]] = None,
                 estimated_runtime_ms: float = 0.0):
        self.name = name
        self.function = function
        self.priority = priority  # Higher number = higher priority
        self.deadline_ms = deadline_ms
        self.resources = resources or set()
        self.estimated_runtime_ms = estimated_runtime_ms
        self.creation_time = time.time() * 1000  # ms
        
        # Execution statistics
        self.last_execution_time = 0.0
        self.total_executions = 0
        self.total_execution_time = 0.0
        self.deadline_misses = 0
        self.preemption_count = 0
        
    def __lt__(self, other):
        # For priority queue ordering - higher priority first
        if self.priority != other.priority:
            return self.priority > other.priority
        
        # If same priority, use deadline (EDF - Earliest Deadline First)
        if self.deadline_ms is not None and other.deadline_ms is not None:
            return self.deadline_ms < other.deadline_ms
            
        # If no deadline, use creation time (FIFO)
        return self.creation_time < other.creation_time
        
    def execute(self) -> Any:
        """Execute the task and record statistics"""
        start_time = time.time() * 1000
        try:
            result = self.function()
            return result
        finally:
            end_time = time.time() * 1000
            execution_time = end_time - start_time
            
            self.last_execution_time = execution_time
            self.total_executions += 1
            self.total_execution_time += execution_time
            
            # Check for deadline miss
            if self.deadline_ms is not None and execution_time > self.deadline_ms:
                self.deadline_misses += 1
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics"""
        if self.total_executions == 0:
            avg_time = 0.0
        else:
            avg_time = self.total_execution_time / self.total_executions
            
        return {
            'name': self.name,
            'priority': self.priority,
            'deadline_ms': self.deadline_ms,
            'total_executions': self.total_executions,
            'avg_execution_time_ms': avg_time,
            'last_execution_time_ms': self.last_execution_time,
            'deadline_misses': self.deadline_misses,
            'miss_rate': self.deadline_misses / max(1, self.total_executions),
            'preemption_count': self.preemption_count
        }

class PriorityTaskScheduler:
    """
    Priority-based task scheduler with deadline awareness and resource management.
    Supports task preemption, resource allocation, and comprehensive statistics.
    """
    def __init__(self, preemptive: bool = True, resource_aware: bool = True):
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []  # Priority queue
        self.running_task: Optional[Task] = None
        self.preemptive = preemptive
        self.resource_aware = resource_aware
        self.allocated_resources: Set[str] = set()
        self.execution_history: deque = deque(maxlen=100)
        self.running = False
        
        # Scheduler statistics
        self.total_scheduled = 0
        self.total_preemptions = 0
        self.total_resource_conflicts = 0
        self.idle_time_ms = 0.0
        self.busy_time_ms = 0.0
        self.last_cycle_time = 0.0
        
    def register_task(self, 
                      name: str, 
                      function: Callable[[], Any], 
                      priority: int = 0, 
                      deadline_ms: Optional[float] = None,
                      resources: Optional[Set[str]] = None,
                      estimated_runtime_ms: float = 0.0) -> Task:
        """
        Register a task with the scheduler
        
        Args:
            name: Unique task name
            function: Task function to execute
            priority: Task priority (higher = more important)
            deadline_ms: Execution deadline in milliseconds
            resources: Set of resource names required by this task
            estimated_runtime_ms: Estimated runtime in milliseconds
            
        Returns:
            The created Task object
        """
        if name in self.tasks:
            raise ValueError(f"Task '{name}' already registered")
            
        task = Task(name, function, priority, deadline_ms, resources, estimated_runtime_ms)
        self.tasks[name] = task
        return task
        
    def schedule_task(self, name: str) -> bool:
        """
        Schedule a registered task for execution
        
        Args:
            name: Name of the task to schedule
            
        Returns:
            True if task was scheduled, False otherwise
        """
        if name not in self.tasks:
            return False
            
        task = self.tasks[name]
        
        # Check if task's required resources are available
        if self.resource_aware and self.allocated_resources.intersection(task.resources):
            # Resource conflict
            self.total_resource_conflicts += 1
            return False
            
        # Add to priority queue
        heapq.heappush(self.task_queue, task)
        self.total_scheduled += 1
        
        # If preemptive and this task has higher priority than running task
        if (self.preemptive and 
            self.running_task is not None and 
            task.priority > self.running_task.priority):
            self._preempt_current_task()
            
        return True
        
    def _preempt_current_task(self):
        """Preempt the currently running task"""
        if self.running_task is None:
            return
            
        self.running_task.preemption_count += 1
        self.total_preemptions += 1
        
        # Re-add current task to queue
        heapq.heappush(self.task_queue, self.running_task)
        self.running_task = None
        
        # Release allocated resources
        self.allocated_resources = set()
        
    def _allocate_resources(self, task: Task) -> bool:
        """
        Attempt to allocate resources for a task
        
        Returns:
            True if resources were allocated, False if there was a conflict
        """
        if not self.resource_aware:
            return True
            
        # Check for resource conflicts
        if self.allocated_resources.intersection(task.resources):
            self.total_resource_conflicts += 1
            return False
            
        # Allocate resources
        self.allocated_resources.update(task.resources)
        return True
        
    def _release_resources(self, task: Task):
        """Release resources allocated to a task"""
        if not self.resource_aware:
            return
            
        self.allocated_resources.difference_update(task.resources)
        
    async def run(self, max_runtime_s: Optional[float] = None):
        """
        Run the scheduler
        
        Args:
            max_runtime_s: Maximum runtime in seconds (None = run indefinitely)
        """
        self.running = True
        start_time = time.time()
        last_idle_start = None
        
        while self.running:
            if max_runtime_s is not None and time.time() - start_time > max_runtime_s:
                break
                
            cycle_start = time.time() * 1000
                
            if not self.task_queue:
                # No tasks to execute
                if last_idle_start is None:
                    last_idle_start = time.time() * 1000
                    
                # Small sleep to prevent CPU hogging
                await asyncio.sleep(0.001)
                continue
                
            # If we were idle and now have tasks, update idle time
            if last_idle_start is not None:
                self.idle_time_ms += (time.time() * 1000) - last_idle_start
                last_idle_start = None
                
            # Get highest priority task
            task = heapq.heappop(self.task_queue)
            
            # Try to allocate resources
            if not self._allocate_resources(task):
                # Resource conflict, put back in queue
                heapq.heappush(self.task_queue, task)
                await asyncio.sleep(0.001)
                continue
                
            # Execute task
            self.running_task = task
            try:
                result = task.execute()
                self.execution_history.append({
                    'task': task.name,
                    'time': time.time(),
                    'execution_time_ms': task.last_execution_time,
                    'result': result
                })
            finally:
                self._release_resources(task)
                self.running_task = None
                
            # Update busy time
            self.busy_time_ms += task.last_execution_time
            
            # Calculate cycle time
            cycle_end = time.time() * 1000
            self.last_cycle_time = cycle_end - cycle_start
            
        self.running = False
        
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        
    def get_task_statistics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get task execution statistics
        
        Args:
            name: Optional task name (None = get all tasks)
            
        Returns:
            Dictionary of task statistics
        """
        if name is not None:
            if name not in self.tasks:
                raise ValueError(f"Task '{name}' not found")
            return self.tasks[name].get_statistics()
            
        return {name: task.get_statistics() for name, task in self.tasks.items()}
        
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        total_time = self.idle_time_ms + self.busy_time_ms
        utilization = 0.0 if total_time == 0 else (self.busy_time_ms / total_time)
        
        return {
            'total_scheduled': self.total_scheduled,
            'total_preemptions': self.total_preemptions,
            'total_resource_conflicts': self.total_resource_conflicts,
            'idle_time_ms': self.idle_time_ms,
            'busy_time_ms': self.busy_time_ms,
            'utilization': utilization,
            'last_cycle_time_ms': self.last_cycle_time,
            'queue_length': len(self.task_queue),
            'registered_tasks': len(self.tasks)
        }
        
    def clear_queue(self):
        """Clear the task queue"""
        self.task_queue = []