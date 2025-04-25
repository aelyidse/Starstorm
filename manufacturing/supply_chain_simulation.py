from typing import Dict, Any, List, Optional, Tuple, Set, Callable, Union
import numpy as np
import random
from enum import Enum, auto
import time
from collections import deque

class SupplyChainEventType(Enum):
    """Types of events that can occur in the supply chain"""
    DELIVERY = auto()
    DELAY = auto()
    SHORTAGE = auto()
    QUALITY_ISSUE = auto()
    PRICE_CHANGE = auto()
    DEMAND_CHANGE = auto()

class SupplierReliability(Enum):
    """Reliability levels for suppliers"""
    LOW = 0.7
    MEDIUM = 0.85
    HIGH = 0.95
    EXCELLENT = 0.99

class SupplyChainNode:
    """Represents a node in the supply chain network (supplier, factory, warehouse, etc.)"""
    def __init__(self, 
                 node_id: str, 
                 node_type: str,
                 lead_time: float,
                 lead_time_variance: float = 0.2,
                 reliability: float = 0.9):
        self.node_id = node_id
        self.node_type = node_type
        self.lead_time = lead_time  # Base lead time in days
        self.lead_time_variance = lead_time_variance  # Variance as a percentage of lead time
        self.reliability = reliability  # Probability of fulfilling orders without issues
        self.inventory: Dict[str, int] = {}
        self.capacity: Dict[str, int] = {}
        self.connections: List['SupplyChainLink'] = []
        
    def add_inventory(self, item_id: str, quantity: int, capacity: int):
        """Add inventory item with initial quantity and capacity"""
        self.inventory[item_id] = quantity
        self.capacity[item_id] = capacity
        
    def add_connection(self, link: 'SupplyChainLink'):
        """Add a connection to another node"""
        self.connections.append(link)
        
    def process_order(self, item_id: str, quantity: int) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Process an order for an item
        
        Returns:
            Tuple of (fulfilled quantity, list of events)
        """
        events = []
        
        # Check if we have the item
        if item_id not in self.inventory:
            events.append({
                'type': SupplyChainEventType.SHORTAGE,
                'item_id': item_id,
                'node_id': self.node_id,
                'timestamp': time.time(),
                'description': f"Item {item_id} not stocked at {self.node_id}"
            })
            return 0, events
            
        # Check if we have enough quantity
        available = self.inventory[item_id]
        fulfilled = min(available, quantity)
        
        # Update inventory
        self.inventory[item_id] -= fulfilled
        
        # Generate events
        if fulfilled < quantity:
            events.append({
                'type': SupplyChainEventType.SHORTAGE,
                'item_id': item_id,
                'node_id': self.node_id,
                'quantity_short': quantity - fulfilled,
                'timestamp': time.time(),
                'description': f"Partial fulfillment of {item_id} at {self.node_id}"
            })
            
        # Random reliability check
        if random.random() > self.reliability:
            # Generate a random issue
            issue_type = random.choice([
                SupplyChainEventType.DELAY,
                SupplyChainEventType.QUALITY_ISSUE
            ])
            
            events.append({
                'type': issue_type,
                'item_id': item_id,
                'node_id': self.node_id,
                'quantity': fulfilled,
                'timestamp': time.time(),
                'description': f"{issue_type.name} for {item_id} at {self.node_id}"
            })
            
        return fulfilled, events
        
    def calculate_delivery_time(self) -> float:
        """Calculate actual delivery time with variance"""
        variance = random.uniform(-self.lead_time_variance, self.lead_time_variance)
        return max(0.1, self.lead_time * (1 + variance))


class SupplyChainLink:
    """Represents a connection between two nodes in the supply chain"""
    def __init__(self, 
                 source_node: SupplyChainNode, 
                 target_node: SupplyChainNode,
                 transport_time: float,
                 transport_reliability: float = 0.9,
                 transport_cost: float = 1.0):
        self.source = source_node
        self.target = target_node
        self.transport_time = transport_time  # Time in days
        self.transport_reliability = transport_reliability
        self.transport_cost = transport_cost  # Cost per unit
        self.active = True
        
    def transfer(self, item_id: str, quantity: int) -> Dict[str, Any]:
        """
        Transfer items from source to target
        
        Returns:
            Transfer result with events
        """
        # Process order at source
        fulfilled, events = self.source.process_order(item_id, quantity)
        
        if fulfilled == 0:
            return {
                'success': False,
                'quantity_fulfilled': 0,
                'events': events
            }
            
        # Calculate delivery time
        base_time = self.source.calculate_delivery_time() + self.transport_time
        
        # Check for transport issues
        if random.random() > self.transport_reliability:
            # Transport delay
            delay_factor = random.uniform(0.2, 1.0)
            base_time *= (1 + delay_factor)
            
            events.append({
                'type': SupplyChainEventType.DELAY,
                'item_id': item_id,
                'source': self.source.node_id,
                'target': self.target.node_id,
                'delay_time': base_time * delay_factor,
                'timestamp': time.time(),
                'description': f"Transport delay from {self.source.node_id} to {self.target.node_id}"
            })
            
        # Schedule delivery to target
        delivery_event = {
            'type': SupplyChainEventType.DELIVERY,
            'item_id': item_id,
            'quantity': fulfilled,
            'source': self.source.node_id,
            'target': self.target.node_id,
            'delivery_time': base_time,
            'timestamp': time.time(),
            'cost': self.transport_cost * fulfilled
        }
        
        events.append(delivery_event)
        
        return {
            'success': True,
            'quantity_fulfilled': fulfilled,
            'delivery_time': base_time,
            'events': events
        }


class SupplyChainSimulator:
    """
    Simulates supply chain operations, disruptions, and performance.
    Models inventory flow, lead times, and supply chain risks.
    """
    def __init__(self):
        self.nodes: Dict[str, SupplyChainNode] = {}
        self.links: List[SupplyChainLink] = []
        self.events: List[Dict[str, Any]] = []
        self.current_time = 0.0  # Simulation time in days
        self.event_queue = deque()
        
    def add_node(self, node: SupplyChainNode):
        """Add a node to the supply chain"""
        self.nodes[node.node_id] = node
        
    def add_link(self, source_id: str, target_id: str, 
                transport_time: float, reliability: float = 0.9, cost: float = 1.0):
        """Add a link between two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Nodes {source_id} or {target_id} not found")
            
        source = self.nodes[source_id]
        target = self.nodes[target_id]
        
        link = SupplyChainLink(source, target, transport_time, reliability, cost)
        self.links.append(link)
        source.add_connection(link)
        
    def place_order(self, source_id: str, target_id: str, item_id: str, quantity: int) -> Dict[str, Any]:
        """Place an order from source to target"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Nodes {source_id} or {target_id} not found")
            
        # Find the link
        link = None
        for l in self.links:
            if l.source.node_id == source_id and l.target.node_id == target_id:
                link = l
                break
                
        if link is None:
            raise ValueError(f"No link found between {source_id} and {target_id}")
            
        # Process the transfer
        result = link.transfer(item_id, quantity)
        
        # Record events
        self.events.extend(result['events'])
        
        # Schedule future delivery
        if result['success'] and 'delivery_time' in result:
            delivery_time = self.current_time + result['delivery_time']
            self.event_queue.append((delivery_time, {
                'type': 'delivery',
                'item_id': item_id,
                'quantity': result['quantity_fulfilled'],
                'target_id': target_id
            }))
            
        return result
        
    def simulate(self, days: float, time_step: float = 0.25) -> Dict[str, Any]:
        """
        Run the simulation for a specified number of days
        
        Args:
            days: Number of days to simulate
            time_step: Simulation time step in days
            
        Returns:
            Simulation results
        """
        end_time = self.current_time + days
        
        while self.current_time < end_time:
            # Process any events scheduled for this time
            while self.event_queue and self.event_queue[0][0] <= self.current_time:
                event_time, event = self.event_queue.popleft()
                
                if event['type'] == 'delivery':
                    # Process delivery
                    target = self.nodes[event['target_id']]
                    item_id = event['item_id']
                    quantity = event['quantity']
                    
                    # Add to target inventory
                    if item_id in target.inventory:
                        target.inventory[item_id] += quantity
                    else:
                        target.inventory[item_id] = quantity
                        
                    # Cap at capacity
                    if item_id in target.capacity:
                        target.inventory[item_id] = min(target.inventory[item_id], target.capacity[item_id])
            
            # Advance time
            self.current_time += time_step
            
        return {
            'events': self.events,
            'current_time': self.current_time,
            'node_status': {node_id: {
                'inventory': node.inventory.copy(),
                'capacity': node.capacity.copy()
            } for node_id, node in self.nodes.items()}
        }
        
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze supply chain performance metrics
        
        Returns:
            Performance metrics
        """
        if not self.events:
            return {'status': 'No events to analyze'}
            
        # Calculate metrics
        total_orders = sum(1 for e in self.events if e['type'] == SupplyChainEventType.DELIVERY)
        total_delays = sum(1 for e in self.events if e['type'] == SupplyChainEventType.DELAY)
        total_shortages = sum(1 for e in self.events if e['type'] == SupplyChainEventType.SHORTAGE)
        total_quality_issues = sum(1 for e in self.events if e['type'] == SupplyChainEventType.QUALITY_ISSUE)
        
        # Calculate on-time delivery rate
        if total_orders > 0:
            on_time_rate = 1.0 - (total_delays / total_orders)
        else:
            on_time_rate = 1.0
            
        # Calculate fill rate
        total_requested = 0
        total_fulfilled = 0
        
        for event in self.events:
            if event['type'] == SupplyChainEventType.DELIVERY:
                total_fulfilled += event['quantity']
                
            elif event['type'] == SupplyChainEventType.SHORTAGE:
                if 'quantity_short' in event:
                    total_requested += event['quantity_short']
                    
        if total_requested + total_fulfilled > 0:
            fill_rate = total_fulfilled / (total_requested + total_fulfilled)
        else:
            fill_rate = 1.0
            
        return {
            'on_time_delivery_rate': on_time_rate,
            'fill_rate': fill_rate,
            'quality_issue_rate': total_quality_issues / max(1, total_orders),
            'total_orders': total_orders,
            'total_delays': total_delays,
            'total_shortages': total_shortages,
            'total_quality_issues': total_quality_issues
        }
        
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify bottlenecks in the supply chain
        
        Returns:
            List of bottleneck nodes and links
        """
        bottlenecks = []
        
        # Count events by node
        node_events = {}
        for event in self.events:
            if 'node_id' in event:
                node_id = event['node_id']
                if node_id not in node_events:
                    node_events[node_id] = {'shortages': 0, 'delays': 0, 'quality_issues': 0}
                    
                if event['type'] == SupplyChainEventType.SHORTAGE:
                    node_events[node_id]['shortages'] += 1
                elif event['type'] == SupplyChainEventType.DELAY:
                    node_events[node_id]['delays'] += 1
                elif event['type'] == SupplyChainEventType.QUALITY_ISSUE:
                    node_events[node_id]['quality_issues'] += 1
        
        # Identify problematic nodes
        for node_id, counts in node_events.items():
            total_issues = counts['shortages'] + counts['delays'] + counts['quality_issues']
            if total_issues > 3:  # Arbitrary threshold
                bottlenecks.append({
                    'type': 'node',
                    'node_id': node_id,
                    'issues': counts,
                    'total_issues': total_issues
                })
                
        return sorted(bottlenecks, key=lambda x: x['total_issues'], reverse=True)