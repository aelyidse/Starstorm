from typing import Dict, Any, List, Callable, Optional
import time

class MissionScenarioExecutor:
    """
    Executes end-to-end mission scenarios, coordinating all phases and subsystems.
    Supports scenario definition, phase progression, event injection, and full mission logging.
    """
    def __init__(self, phases: Optional[List[str]] = None, phase_funcs: Optional[Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = None):
        self.phases = phases or ['prelaunch', 'launch', 'ascent', 'orbit', 'reentry', 'landing', 'recovery']
        self.phase_funcs = phase_funcs or {}
        self.current_phase = self.phases[0]
        self.state: Dict[str, Any] = {}
        self.log: List[Dict[str, Any]] = []
        self.running = False

    def set_phase_func(self, phase: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.phase_funcs[phase] = func

    def initialize(self, initial_state: Dict[str, Any]):
        self.state = initial_state.copy()
        self.current_phase = self.phases[0]
        self.log = []
        self.running = False

    def run(self, event_schedule: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.running = True
        for phase in self.phases:
            self.current_phase = phase
            # Inject events for this phase if any
            events = event_schedule[phase] if event_schedule and phase in event_schedule else []
            for event in events:
                self.state.update(event)
            # Run phase logic if defined
            if phase in self.phase_funcs:
                self.state = self.phase_funcs[phase](self.state)
            self.log.append({'phase': phase, 'state': self.state.copy(), 'events': events})
        self.running = False
        return self.log

    def get_log(self) -> List[Dict[str, Any]]:
        return self.log

    def get_current_phase(self) -> str:
        return self.current_phase
