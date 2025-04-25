from typing import Dict, Any, List, Optional, Set, Tuple, Callable
import time
import heapq
from enum import Enum

class ResourceType(Enum):
    """Types of resources that can be allocated during missions"""
    COMPUTATIONAL = "computational"
    POWER = "power"
    BANDWIDTH = "bandwidth"
    STORAGE = "storage"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CUSTOM = "custom"

class ResourcePriority(Enum):
    """Priority levels for resource allocation"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1

class ResourceRequest:
    """Represents a request for resources during mission execution"""
    def __init__(self, 
                 name: str,
                 resource_type: ResourceType,
                 amount: float,
                 priority: ResourcePriority = ResourcePriority.MEDIUM,
                 duration_ms: Optional[float] = None,
                 phase_name: Optional[str] = None):
        self.name = name
        self.resource_type = resource_type
        self.amount = amount
        self.priority = priority
        self.duration_ms = duration_ms
        self.phase_name = phase_name
        self.timestamp = time.time()
        self.id = f"{name}_{self.timestamp}"
        
    def __lt__(self, other):
        # For priority queue - higher priority first
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        # If same priority, use timestamp (FIFO)
        return self.timestamp < other.timestamp

class ResourceAllocation:
    """Represents an allocated resource"""
    def __init__(self, 
                 request: ResourceRequest,
                 allocated_amount: float,
                 start_time: float):
        self.request = request
        self.allocated_amount = allocated_amount
        self.start_time = start_time
        self.end_time: Optional[float] = None
        self.status = "active"
        
    def release(self):
        """Mark this allocation as released"""
        self.end_time = time.time()
        self.status = "released"
        
    def is_expired(self, current_time: float) -> bool:
        """Check if this allocation has expired based on its duration"""
        if not self.request.duration_ms:
            return False
        return (current_time - self.start_time) * 1000 >= self.request.duration_ms

class ResourcePool:
    """Manages a pool of resources of a specific type"""
    def __init__(self, 
                 resource_type: ResourceType,
                 capacity: float,
                 name: str = ""):
        self.resource_type = resource_type
        self.capacity = capacity
        self.name = name or f"{resource_type.value}_pool"
        self.available = capacity
        self.allocations: Dict[str, ResourceAllocation] = {}
        
    def can_allocate(self, amount: float) -> bool:
        """Check if the requested amount can be allocated"""
        return amount <= self.available
        
    def allocate(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """
        Attempt to allocate resources for a request
        
        Returns:
            ResourceAllocation if successful, None otherwise
        """
        if request.amount > self.available:
            return None
            
        # Create allocation
        allocation = ResourceAllocation(
            request=request,
            allocated_amount=request.amount,
            start_time=time.time()
        )
        
        # Update available resources
        self.available -= request.amount
        self.allocations[request.id] = allocation
        
        return allocation
        
    def release(self, allocation_id: str) -> float:
        """
        Release an allocation
        
        Args:
            allocation_id: ID of the allocation to release
            
        Returns:
            Amount of resources released
        """
        if allocation_id not in self.allocations:
            return 0.0
            
        allocation = self.allocations[allocation_id]
        if allocation.status == "released":
            return 0.0
            
        allocation.release()
        self.available += allocation.allocated_amount
        
        return allocation.allocated_amount
        
    def check_expirations(self) -> List[ResourceAllocation]:
        """
        Check for expired allocations and release them
        
        Returns:
            List of expired allocations that were released
        """
        current_time = time.time()
        expired = []
        
        for allocation_id, allocation in list(self.allocations.items()):
            if allocation.status == "active" and allocation.is_expired(current_time):
                self.release(allocation_id)
                expired.append(allocation)
                
        return expired
        
    def get_utilization(self) -> float:
        """Get the current utilization percentage of this resource pool"""
        return (self.capacity - self.available) / self.capacity

class MissionResourceAllocator:
    """
    Manages resource allocation and optimization for mission execution.
    Handles resource requests, allocation, and release with priority-based scheduling.
    """
    def __init__(self):
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.pending_requests: List[ResourceRequest] = []  # Priority queue
        self.allocations: Dict[str, Dict[str, ResourceAllocation]] = {}  # phase_name -> {request_id -> allocation}
        self.allocation_history: List[Dict[str, Any]] = []
        
    def register_resource_pool(self, 
                              resource_type: ResourceType,
                              capacity: float,
                              name: str = "") -> ResourcePool:
        """
        Register a resource pool
        
        Args:
            resource_type: Type of resource
            capacity: Total capacity of the pool
            name: Optional name for the pool
            
        Returns:
            The created ResourcePool
        """
        pool_name = name or f"{resource_type.value}_pool"
        if pool_name in self.resource_pools:
            raise ValueError(f"Resource pool '{pool_name}' already exists")
            
        pool = ResourcePool(resource_type, capacity, pool_name)
        self.resource_pools[pool_name] = pool
        return pool
        
    def request_resource(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """
        Request resources
        
        Args:
            request: Resource request
            
        Returns:
            ResourceAllocation if immediately allocated, None if queued or failed
        """
        # Check if we have a pool for this resource type
        pool_name = f"{request.resource_type.value}_pool"
        if pool_name not in self.resource_pools:
            # Try to find a custom pool with matching name
            matching_pools = [p for p in self.resource_pools.values() 
                             if p.resource_type == request.resource_type]
            if not matching_pools:
                return None
            pool = matching_pools[0]
        else:
            pool = self.resource_pools[pool_name]
            
        # Try to allocate immediately
        if pool.can_allocate(request.amount):
            allocation = pool.allocate(request)
            
            # Record allocation
            if request.phase_name:
                if request.phase_name not in self.allocations:
                    self.allocations[request.phase_name] = {}
                self.allocations[request.phase_name][request.id] = allocation
                
            # Record in history
            self.allocation_history.append({
                "request_id": request.id,
                "resource_type": request.resource_type.value,
                "amount": request.amount,
                "priority": request.priority.value,
                "phase": request.phase_name,
                "status": "allocated",
                "timestamp": time.time()
            })
                
            return allocation
            
        # Queue the request for later
        heapq.heappush(self.pending_requests, request)
        
        # Record in history
        self.allocation_history.append({
            "request_id": request.id,
            "resource_type": request.resource_type.value,
            "amount": request.amount,
            "priority": request.priority.value,
            "phase": request.phase_name,
            "status": "queued",
            "timestamp": time.time()
        })
        
        return None
        
    def release_phase_resources(self, phase_name: str) -> Dict[str, float]:
        """
        Release all resources allocated to a phase
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Dictionary of {resource_type: amount_released}
        """
        if phase_name not in self.allocations:
            return {}
            
        released = {}
        for request_id, allocation in list(self.allocations[phase_name].items()):
            pool_name = f"{allocation.request.resource_type.value}_pool"
            if pool_name in self.resource_pools:
                pool = self.resource_pools[pool_name]
                amount = pool.release(request_id)
                
                resource_type = allocation.request.resource_type.value
                if resource_type in released:
                    released[resource_type] += amount
                else:
                    released[resource_type] = amount
                    
                # Record in history
                self.allocation_history.append({
                    "request_id": request_id,
                    "resource_type": resource_type,
                    "amount": amount,
                    "phase": phase_name,
                    "status": "released",
                    "timestamp": time.time()
                })
                
        # Clear phase allocations
        self.allocations[phase_name] = {}
        
        return released
        
    def process_pending_requests(self) -> int:
        """
        Process pending resource requests
        
        Returns:
            Number of requests processed
        """
        if not self.pending_requests:
            return 0
            
        processed = 0
        remaining = []
        
        # Check for expired allocations first
        for pool in self.resource_pools.values():
            pool.check_expirations()
            
        # Try to allocate pending requests
        while self.pending_requests:
            request = heapq.heappop(self.pending_requests)
            
            # Find appropriate pool
            pool_name = f"{request.resource_type.value}_pool"
            if pool_name not in self.resource_pools:
                # Try to find a custom pool with matching name
                matching_pools = [p for p in self.resource_pools.values() 
                                if p.resource_type == request.resource_type]
                if not matching_pools:
                    continue
                pool = matching_pools[0]
            else:
                pool = self.resource_pools[pool_name]
                
            # Try to allocate
            if pool.can_allocate(request.amount):
                allocation = pool.allocate(request)
                
                # Record allocation
                if request.phase_name:
                    if request.phase_name not in self.allocations:
                        self.allocations[request.phase_name] = {}
                    self.allocations[request.phase_name][request.id] = allocation
                    
                processed += 1
                
                # Record in history
                self.allocation_history.append({
                    "request_id": request.id,
                    "resource_type": request.resource_type.value,
                    "amount": request.amount,
                    "priority": request.priority.value,
                    "phase": request.phase_name,
                    "status": "allocated_from_queue",
                    "timestamp": time.time()
                })
            else:
                # Put back in queue
                remaining.append(request)
                
        # Restore remaining requests to queue
        for request in remaining:
            heapq.heappush(self.pending_requests, request)
            
        return processed
        
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get utilization percentage for each resource pool"""
        return {name: pool.get_utilization() for name, pool in self.resource_pools.items()}
        
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get comprehensive allocation statistics"""
        stats = {
            "total_pools": len(self.resource_pools),
            "pending_requests": len(self.pending_requests),
            "active_allocations": sum(len(allocs) for allocs in self.allocations.values()),
            "utilization": self.get_resource_utilization(),
            "allocation_history_length": len(self.allocation_history)
        }
        
        # Add per-resource-type stats
        resource_type_stats = {}
        for pool in self.resource_pools.values():
            resource_type = pool.resource_type.value
            if resource_type not in resource_type_stats:
                resource_type_stats[resource_type] = {
                    "total_capacity": 0,
                    "available": 0,
                    "allocation_count": 0
                }
                
            resource_type_stats[resource_type]["total_capacity"] += pool.capacity
            resource_type_stats[resource_type]["available"] += pool.available
            resource_type_stats[resource_type]["allocation_count"] += len(pool.allocations)
            
        stats["resource_types"] = resource_type_stats
        
        return stats

class ResourceOptimizer:
    """
    Optimizes resource allocation based on mission requirements and system state.
    Provides recommendations for resource reallocation and efficiency improvements.
    """
    def __init__(self, resource_allocator: MissionResourceAllocator):
        self.resource_allocator = resource_allocator
        self.optimization_history: List[Dict[str, Any]] = []
        
    def analyze_resource_usage(self) -> Dict[str, Any]:
        """
        Analyze current resource usage patterns
        
        Returns:
            Analysis results
        """
        # Get current utilization
        utilization = self.resource_allocator.get_resource_utilization()
        
        # Identify underutilized and overutilized resources
        underutilized = {name: util for name, util in utilization.items() if util < 0.3}
        overutilized = {name: util for name, util in utilization.items() if util > 0.8}
        
        # Analyze allocation history for patterns
        history = self.resource_allocator.allocation_history
        
        # Count allocations by resource type and phase
        resource_type_counts = {}
        phase_counts = {}
        
        for entry in history:
            resource_type = entry["resource_type"]
            if resource_type not in resource_type_counts:
                resource_type_counts[resource_type] = 0
            resource_type_counts[resource_type] += 1
            
            phase = entry.get("phase")
            if phase:
                if phase not in phase_counts:
                    phase_counts[phase] = 0
                phase_counts[phase] += 1
        
        return {
            "utilization": utilization,
            "underutilized_resources": underutilized,
            "overutilized_resources": overutilized,
            "resource_type_allocation_counts": resource_type_counts,
            "phase_allocation_counts": phase_counts
        }
        
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations for resource optimization
        
        Returns:
            List of optimization recommendations
        """
        analysis = self.analyze_resource_usage()
        recommendations = []
        
        # Check for overutilized resources
        for name, util in analysis["overutilized_resources"].items():
            recommendations.append({
                "type": "capacity_increase",
                "resource": name,
                "current_utilization": util,
                "recommendation": f"Increase capacity of {name} resource pool",
                "priority": "high" if util > 0.9 else "medium"
            })
            
        # Check for underutilized resources
        for name, util in analysis["underutilized_resources"].items():
            recommendations.append({
                "type": "capacity_reduction",
                "resource": name,
                "current_utilization": util,
                "recommendation": f"Consider reducing capacity of {name} resource pool",
                "priority": "low"
            })
            
        # Record optimization attempt
        self.optimization_history.append({
            "timestamp": time.time(),
            "analysis": analysis,
            "recommendations": recommendations
        })
            
        return recommendations
        
    def apply_optimization(self, recommendation_id: int) -> bool:
        """
        Apply a specific optimization recommendation
        
        Args:
            recommendation_id: Index of the recommendation to apply
            
        Returns:
            True if applied successfully, False otherwise
        """
        # This would implement the actual optimization logic
        # For now, just record that we attempted to apply it
        if not self.optimization_history:
            return False
            
        latest = self.optimization_history[-1]
        if "recommendations" not in latest or recommendation_id >= len(latest["recommendations"]):
            return False
            
        recommendation = latest["recommendations"][recommendation_id]
        
        # Record application
        if "applied_recommendations" not in latest:
            latest["applied_recommendations"] = []
            
        latest["applied_recommendations"].append({
            "recommendation_id": recommendation_id,
            "timestamp": time.time(),
            "recommendation": recommendation
        })
        
        return True
        
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of optimization analyses and recommendations"""
        return self.optimization_history