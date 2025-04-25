from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
from collections import defaultdict

class CommunicationsModule:
    """
    Manages communications with bandwidth optimization for efficient data transmission.
    Supports prioritization, compression, and adaptive transmission strategies.
    """
    def __init__(self, max_bandwidth: float = 100.0):
        self.max_bandwidth = max_bandwidth  # Maximum bandwidth in Mbps
        self.current_bandwidth = max_bandwidth  # Available bandwidth
        self.transmission_queue: List[Dict[str, Any]] = []
        self.transmission_history: List[Dict[str, Any]] = []
        self.data_priorities: Dict[str, int] = {}  # Data type to priority mapping
        self.compression_ratios: Dict[str, float] = {}  # Data type to compression ratio
        self.channel_quality: float = 1.0  # 1.0 = perfect, 0.0 = no connection
        self.last_optimization_time: Optional[float] = None
        
    def set_data_priority(self, data_type: str, priority: int) -> None:
        """
        Set priority for a data type (higher = more important)
        
        Args:
            data_type: Type of data (e.g., 'telemetry', 'imagery', 'commands')
            priority: Priority level (1-10, higher is more important)
        """
        self.data_priorities[data_type] = max(1, min(10, priority))
        
    def set_compression_ratio(self, data_type: str, ratio: float) -> None:
        """
        Set compression ratio for a data type
        
        Args:
            data_type: Type of data
            ratio: Compression ratio (0.0-1.0, lower means better compression)
        """
        self.compression_ratios[data_type] = max(0.1, min(1.0, ratio))
        
    def update_channel_quality(self, quality: float) -> None:
        """
        Update the communication channel quality
        
        Args:
            quality: Channel quality (0.0-1.0)
        """
        self.channel_quality = max(0.0, min(1.0, quality))
        self.current_bandwidth = self.max_bandwidth * self.channel_quality
        
    def queue_data(self, data: Dict[str, Any], size_mb: float, data_type: str) -> str:
        """
        Queue data for transmission
        
        Args:
            data: Data payload or metadata
            size_mb: Size of data in megabytes
            data_type: Type of data for prioritization
            
        Returns:
            Unique ID for the queued data
        """
        # Generate unique ID
        data_id = f"data_{int(time.time())}_{len(self.transmission_queue)}"
        
        # Get priority or default to lowest
        priority = self.data_priorities.get(data_type, 1)
        
        # Get compression ratio or default to no compression
        compression = self.compression_ratios.get(data_type, 1.0)
        
        # Calculate effective size after compression
        effective_size = size_mb * compression
        
        # Queue the data
        self.transmission_queue.append({
            'id': data_id,
            'data': data,
            'original_size': size_mb,
            'effective_size': effective_size,
            'data_type': data_type,
            'priority': priority,
            'timestamp': time.time(),
            'status': 'queued'
        })
        
        return data_id
        
    def optimize_transmission(self) -> Dict[str, Any]:
        """
        Optimize the transmission queue based on priorities and available bandwidth
        
        Returns:
            Optimization statistics
        """
        self.last_optimization_time = time.time()
        
        if not self.transmission_queue:
            return {'status': 'empty_queue', 'optimized_items': 0}
            
        # Sort queue by priority (highest first)
        self.transmission_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        # Calculate total size and optimize based on available bandwidth
        total_original_size = sum(item['original_size'] for item in self.transmission_queue)
        total_effective_size = sum(item['effective_size'] for item in self.transmission_queue)
        
        # Compression statistics
        compression_savings = total_original_size - total_effective_size
        compression_ratio = total_effective_size / total_original_size if total_original_size > 0 else 1.0
        
        # Estimate transmission time based on current bandwidth
        transmission_time = total_effective_size * 8 / self.current_bandwidth if self.current_bandwidth > 0 else float('inf')
        
        return {
            'status': 'optimized',
            'queue_length': len(self.transmission_queue),
            'total_original_size_mb': total_original_size,
            'total_effective_size_mb': total_effective_size,
            'compression_savings_mb': compression_savings,
            'compression_ratio': compression_ratio,
            'estimated_transmission_time_s': transmission_time,
            'current_bandwidth_mbps': self.current_bandwidth
        }
        
    def transmit(self, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Transmit data from the queue based on available bandwidth
        
        Args:
            max_items: Maximum number of items to transmit (None = no limit)
            
        Returns:
            List of transmitted items
        """
        if not self.transmission_queue:
            return []
            
        # Optimize before transmission
        self.optimize_transmission()
        
        transmitted = []
        remaining_bandwidth = self.current_bandwidth  # Mbps
        items_to_process = self.transmission_queue[:max_items] if max_items else self.transmission_queue
        
        for item in items_to_process:
            # Calculate transmission time for this item
            size_bits = item['effective_size'] * 8  # Convert MB to Mb
            transmission_time = size_bits / remaining_bandwidth if remaining_bandwidth > 0 else float('inf')
            
            # Update item with transmission details
            item['transmission_time'] = transmission_time
            item['transmission_start'] = time.time()
            item['status'] = 'transmitted'
            
            # Add to transmitted list
            transmitted.append(item)
            
            # Remove from queue
            self.transmission_queue.remove(item)
            
            # Add to history
            self.transmission_history.append(item)
            
            # Update remaining bandwidth (simplified model)
            remaining_bandwidth -= size_bits / 10  # Assume each item uses bandwidth for 10 seconds
            if remaining_bandwidth <= 0:
                break
                
        # Keep history at a reasonable size
        if len(self.transmission_history) > 1000:
            self.transmission_history = self.transmission_history[-1000:]
            
        return transmitted
        
    def get_queue_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current transmission queue
        
        Returns:
            Queue statistics
        """
        if not self.transmission_queue:
            return {
                'queue_length': 0,
                'total_size_mb': 0,
                'data_types': {},
                'priorities': {}
            }
            
        # Calculate statistics
        total_size = sum(item['effective_size'] for item in self.transmission_queue)
        
        # Count by data type
        data_types = defaultdict(int)
        for item in self.transmission_queue:
            data_types[item['data_type']] += 1
            
        # Count by priority
        priorities = defaultdict(int)
        for item in self.transmission_queue:
            priorities[item['priority']] += 1
            
        return {
            'queue_length': len(self.transmission_queue),
            'total_size_mb': total_size,
            'data_types': dict(data_types),
            'priorities': dict(priorities),
            'channel_quality': self.channel_quality,
            'available_bandwidth_mbps': self.current_bandwidth
        }
        
    def get_transmission_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get transmission history
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of transmission history items
        """
        return self.transmission_history[-limit:]