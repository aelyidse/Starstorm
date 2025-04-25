import heapq
from typing import List, Dict, Any, Tuple, Optional
import time

class CommunicationScheduleManager:
    """
    Manages autonomous scheduling of communication sessions, windows, and priorities.
    Supports dynamic rescheduling based on link state, mission events, and system constraints.
    """
    def __init__(self):
        self.schedule: List[Tuple[float, Dict[str, Any]]] = []  # Min-heap: (start_time, session_dict)
        self.current_time = time.time()
        self.active_session: Optional[Dict[str, Any]] = None

    def add_session(self, start_time: float, duration: float, mode: str, priority: int = 10, params: Optional[Dict[str, Any]] = None):
        session = {
            'start_time': start_time,
            'end_time': start_time + duration,
            'mode': mode,
            'priority': priority,
            'params': params or {}
        }
        heapq.heappush(self.schedule, (start_time, session))

    def get_next_session(self) -> Optional[Dict[str, Any]]:
        if not self.schedule:
            return None
        return self.schedule[0][1]

    def advance_time(self, new_time: float):
        self.current_time = new_time
        # Remove expired sessions
        while self.schedule and self.schedule[0][1]['end_time'] <= self.current_time:
            heapq.heappop(self.schedule)
        # Activate session if within window
        if self.schedule and self.schedule[0][1]['start_time'] <= self.current_time < self.schedule[0][1]['end_time']:
            self.active_session = self.schedule[0][1]
        else:
            self.active_session = None

    def reschedule(self, session_idx: int, new_start: float, new_duration: float):
        # Remove and re-add session with new timing
        sessions = [heapq.heappop(self.schedule)[1] for _ in range(len(self.schedule))]
        session = sessions.pop(session_idx)
        session['start_time'] = new_start
        session['end_time'] = new_start + new_duration
        for s in sessions:
            heapq.heappush(self.schedule, (s['start_time'], s))
        heapq.heappush(self.schedule, (session['start_time'], session))

    def get_active_session(self) -> Optional[Dict[str, Any]]:
        return self.active_session

    def get_schedule(self) -> List[Dict[str, Any]]:
        return [s[1] for s in sorted(self.schedule)]
