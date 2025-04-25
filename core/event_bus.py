import asyncio
import logging
import time
from typing import Dict, List, Set, Callable, Optional, Any, Tuple, Awaitable
from .events import Event, EventFilter, EventPriority

class EventSubscription:
    """
    Represents a subscription to events on the event bus.
    """
    def __init__(self, 
                 subscriber_id: str,
                 callback: Callable[[Event], Awaitable[None]],
                 filter: Optional[EventFilter] = None):
        self.subscriber_id = subscriber_id
        self.callback = callback
        self.filter = filter or EventFilter()
        self.subscription_id = f"{subscriber_id}_{id(self)}"
        self.created_at = time.time()
        self.last_invoked_at: Optional[float] = None
        self.invocation_count = 0
    
    async def notify(self, event: Event) -> bool:
        """
        Notify this subscriber of an event.
        Returns True if the event was handled, False otherwise.
        """
        if not self.filter.matches(event):
            return False
        
        try:
            await self.callback(event)
            self.last_invoked_at = time.time()
            self.invocation_count += 1
            return True
        except Exception as e:
            logging.error(f"Error in event handler for {self.subscriber_id}: {str(e)}")
            return False

class EventBus:
    """
    Central event bus for the component system.
    Handles event distribution based on subscriptions and filters.
    """
    def __init__(self):
        self._subscriptions: List[EventSubscription] = []
        self._running = False
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._logger = logging.getLogger("core.event_bus")
        self._event_processors: Dict[EventPriority, asyncio.Task] = {}
        self._metrics: Dict[str, Any] = {
            "events_processed": 0,
            "events_dropped": 0,
            "events_by_type": {},
            "events_by_priority": {},
            "processing_time": 0.0,
        }
    
    def subscribe(self, 
                  subscriber_id: str, 
                  callback: Callable[[Event], Awaitable[None]],
                  filter: Optional[EventFilter] = None) -> str:
        """
        Subscribe to events matching the given filter.
        Returns a subscription ID that can be used to unsubscribe.
        """
        subscription = EventSubscription(subscriber_id, callback, filter)
        self._subscriptions.append(subscription)
        self._logger.debug(f"New subscription from {subscriber_id}")
        return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        Returns True if the subscription was found and removed, False otherwise.
        """
        for i, subscription in enumerate(self._subscriptions):
            if subscription.subscription_id == subscription_id:
                self._subscriptions.pop(i)
                self._logger.debug(f"Removed subscription {subscription_id}")
                return True
        return False
    
    def unsubscribe_all(self, subscriber_id: str) -> int:
        """
        Unsubscribe all subscriptions for a subscriber.
        Returns the number of subscriptions removed.
        """
        original_count = len(self._subscriptions)
        self._subscriptions = [s for s in self._subscriptions if s.subscriber_id != subscriber_id]
        removed = original_count - len(self._subscriptions)
        self._logger.debug(f"Removed {removed} subscriptions for {subscriber_id}")
        return removed
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus.
        The event will be queued and processed asynchronously.
        """
        await self._event_queue.put(event)
        
        # Update metrics
        event_type = event.event_type()
        if event_type not in self._metrics["events_by_type"]:
            self._metrics["events_by_type"][event_type] = 0
        self._metrics["events_by_type"][event_type] += 1
        
        priority_name = event.priority.name
        if priority_name not in self._metrics["events_by_priority"]:
            self._metrics["events_by_priority"][priority_name] = 0
        self._metrics["events_by_priority"][priority_name] += 1
    
    async def publish_immediate(self, event: Event) -> int:
        """
        Publish an event immediately, bypassing the queue.
        Returns the number of subscribers that received the event.
        """
        start_time = time.time()
        count = 0
        
        for subscription in self._subscriptions:
            if await subscription.notify(event):
                count += 1
        
        self._metrics["events_processed"] += 1
        self._metrics["processing_time"] += (time.time() - start_time)
        
        return count
    
    async def _process_events(self, priority: EventPriority) -> None:
        """Process events from the queue with the given priority."""
        self._logger.info(f"Starting event processor for priority {priority.name}")
        
        while self._running:
            try:
                # Get the next event from the queue
                event = await self._event_queue.get()
                
                # Skip if not matching our priority
                if event.priority != priority:
                    await self._event_queue.put(event)  # Put it back
                    await asyncio.sleep(0.01)  # Small delay to avoid busy loop
                    continue
                
                # Process the event
                start_time = time.time()
                delivered = False
                
                for subscription in self._subscriptions:
                    if await subscription.notify(event):
                        delivered = True
                
                if not delivered:
                    self._metrics["events_dropped"] += 1
                
                self._metrics["events_processed"] += 1
                self._metrics["processing_time"] += (time.time() - start_time)
                
                # Mark as done
                self._event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error processing event: {str(e)}")
    
    async def run(self) -> None:
        """Start the event bus."""
        if self._running:
            return
        
        self._running = True
        self._logger.info("Starting event bus")
        
        # Create processors for each priority level
        for priority in EventPriority:
            self._event_processors[priority] = asyncio.create_task(
                self._process_events(priority)
            )
    
    def stop(self) -> None:
        """Stop the event bus."""
        if not self._running:
            return
        
        self._logger.info("Stopping event bus")
        self._running = False
        
        # Cancel all processors
        for task in self._event_processors.values():
            task.cancel()
        
        self._event_processors.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the event bus."""
        return {
            **self._metrics,
            "queue_size": self._event_queue.qsize(),
            "subscription_count": len(self._subscriptions),
            "is_running": self._running,
        }
    
    def get_subscriptions(self) -> List[Dict[str, Any]]:
        """Get information about all subscriptions."""
        return [
            {
                "subscriber_id": sub.subscriber_id,
                "subscription_id": sub.subscription_id,
                "created_at": sub.created_at,
                "last_invoked_at": sub.last_invoked_at,
                "invocation_count": sub.invocation_count,
            }
            for sub in self._subscriptions
        ]
