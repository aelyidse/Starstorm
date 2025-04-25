import math
from typing import Dict, Any, Optional, List, Tuple, Callable, Set
import concurrent.futures
import threading
import time
import numpy as np
import os
import json
import pickle
from datetime import datetime
from utils.random_generator import DeterministicRandomGenerator

class IntegratedSystemSimulator:
    """
    Runs integrated system simulations for validation, training, and scenario analysis.
    Supports stepwise simulation, subsystem stubbing, and event injection.
    Features multi-threading capabilities for parallel subsystem processing.
    Includes time dilation controls for accelerated/decelerated simulation.
    """
    def __init__(self, subsystems: Optional[Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = None, 
                 max_workers: int = None, use_threading: bool = True, seed: Optional[int] = None):
        self.subsystems = subsystems or {}
        self.simulation_log: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {}
        self.time = 0.0
        self.running = False
        self.max_workers = max_workers  # None means use default (CPU count)
        self.use_threading = use_threading
        self.state_lock = threading.RLock()  # For thread-safe state updates
        
        # Time dilation controls
        self.time_dilation_factor = 1.0  # Default: real-time (1.0)
        self.last_real_time = None
        self.real_time_mode = False
        
        # Deterministic random number generation
        self.random_generator = DeterministicRandomGenerator(seed)
        
        # Checkpoint manager
        self.checkpoint_manager = SimulationCheckpointManager(self)

    def add_subsystem(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.subsystems[name] = func

    def initialize(self, initial_state: Dict[str, Any]):
        self.state = initial_state.copy()
        self.time = 0.0
        self.simulation_log = []

    def set_seed(self, seed: int):
        """
        Set the random seed for deterministic simulation.
        
        Args:
            seed: Integer seed value
        """
        self.random_generator.reset(seed)
        # Update the seed in the state for logging
        with self.state_lock:
            self.state['random_seed'] = seed

    def get_random_generator(self) -> DeterministicRandomGenerator:
        """
        Get the deterministic random generator instance.
        
        Returns:
            The random generator instance
        """
        return self.random_generator

    def _process_subsystem(self, name_func_tuple):
        """Process a single subsystem with thread safety"""
        name, func = name_func_tuple
        # Create a copy of the state to avoid race conditions
        with self.state_lock:
            local_state = self.state.copy()
        
        # Process the subsystem
        result = func(local_state)
        
        # Update the global state with the results
        with self.state_lock:
            self.state.update(result)
        
        return name, result

    def set_time_dilation(self, factor: float):
        """
        Set the time dilation factor for simulation speed control.
        
        Args:
            factor: Time dilation factor (1.0 = real-time, >1.0 = faster, <1.0 = slower)
        """
        if factor <= 0:
            raise ValueError("Time dilation factor must be positive")
        self.time_dilation_factor = factor
    
    def get_time_dilation(self) -> float:
        """Return the current time dilation factor"""
        return self.time_dilation_factor
    
    def enable_real_time_mode(self, enabled: bool = True):
        """
        Enable or disable real-time mode with time dilation.
        In real-time mode, the simulation will use wall clock time 
        multiplied by the dilation factor.
        
        Args:
            enabled: Whether to enable real-time mode
        """
        self.real_time_mode = enabled
        self.last_real_time = time.time() if enabled else None
    
    def step(self, dt: float = 1.0, events: Optional[List[Dict[str, Any]]] = None):
        # Calculate actual dt if in real-time mode
        if self.real_time_mode:
            current_time = time.time()
            if self.last_real_time is not None:
                # Calculate elapsed wall clock time and apply dilation
                elapsed = current_time - self.last_real_time
                dt = elapsed * self.time_dilation_factor
            self.last_real_time = current_time
        
        # Inject events if any
        if events:
            with self.state_lock:
                for event in events:
                    self.state.update(event)
        
        # Step each subsystem (either sequentially or in parallel)
        if self.use_threading and len(self.subsystems) > 1:
            # Parallel execution
            executor_class = concurrent.futures.ThreadPoolExecutor if self.use_threading else concurrent.futures.ProcessPoolExecutor
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all subsystems for processing
                futures = [executor.submit(self._process_subsystem, (name, func)) 
                          for name, func in self.subsystems.items()]
                
                # Wait for all to complete
                concurrent.futures.wait(futures)
        else:
            # Sequential execution
            for name, func in self.subsystems.items():
                self._process_subsystem((name, func))
        
        self.time += dt
        with self.state_lock:
            log_entry = {'time': self.time, 'state': self.state.copy()}
            # Include random generator state in the log for reproducibility
            log_entry['random_state'] = self.random_generator.get_state()
        self.simulation_log.append(log_entry)
        
        # Check if we should create an automatic checkpoint
        self.checkpoint_manager.check_auto_checkpoint()
        
        return log_entry

    def run(self, steps: int = 10, dt: float = 1.0, 
            event_schedule: Optional[Dict[int, List[Dict[str, Any]]]] = None,
            parallel: bool = None, real_time: bool = False, seed: Optional[int] = None):
        """
        Run the simulation for a specified number of steps
        
        Args:
            steps: Number of simulation steps to run
            dt: Time delta for each step (ignored in real-time mode)
            event_schedule: Dictionary mapping step numbers to events
            parallel: Override the default threading setting
            real_time: Whether to run in real-time mode with time dilation
            seed: Optional random seed to use for this run
        
        Returns:
            Complete simulation log
        """
        self.running = True
        
        # Allow overriding the threading setting for this run
        original_threading = self.use_threading
        original_real_time = self.real_time_mode
        
        # Set seed if provided
        if seed is not None:
            self.set_seed(seed)
        
        if parallel is not None:
            self.use_threading = parallel
        
        # Set real-time mode if requested
        if real_time:
            self.enable_real_time_mode(True)
            
        try:
            for i in range(steps):
                events = event_schedule.get(i, []) if event_schedule else None
                self.step(dt, events)
        finally:
            # Restore original settings
            self.use_threading = original_threading
            self.enable_real_time_mode(original_real_time)
            self.running = False
            
        return self.simulation_log

    def get_log(self) -> List[Dict[str, Any]]:
        return self.simulation_log
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics about the simulation"""
        return {
            'subsystem_count': len(self.subsystems),
            'steps_completed': len(self.simulation_log),
            'simulation_time': self.time,
            'threading_enabled': self.use_threading,
            'max_workers': self.max_workers,
            'time_dilation_factor': self.time_dilation_factor,
            'real_time_mode': self.real_time_mode,
            'random_seed': self.random_generator.seed,
            'random_call_count': self.random_generator.get_call_count()
        }
    
    def create_checkpoint(self, checkpoint_id: Optional[str] = None) -> str:
        """
        Create a checkpoint of the current simulation state.
        
        Args:
            checkpoint_id: Optional identifier for the checkpoint
            
        Returns:
            The checkpoint ID
        """
        return self.checkpoint_manager.create_checkpoint(checkpoint_id)
    
    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Load a simulation state from a checkpoint.
        
        Args:
            checkpoint_id: Identifier of the checkpoint to load
            
        Returns:
            True if checkpoint was successfully loaded, False otherwise
        """
        return self.checkpoint_manager.load_checkpoint(checkpoint_id)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata
        """
        return self.checkpoint_manager.list_checkpoints()
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Identifier of the checkpoint to delete
            
        Returns:
            True if checkpoint was successfully deleted, False otherwise
        """
        return self.checkpoint_manager.delete_checkpoint(checkpoint_id)
    
    def enable_auto_checkpointing(self, interval: int = 10, max_checkpoints: int = 5) -> None:
        """
        Enable automatic checkpointing at regular intervals.
        
        Args:
            interval: Number of steps between checkpoints
            max_checkpoints: Maximum number of auto-checkpoints to keep
        """
        self.checkpoint_manager.enable_auto_checkpointing(interval, max_checkpoints)
    
    def disable_auto_checkpointing(self) -> None:
        """
        Disable automatic checkpointing.
        """
        self.checkpoint_manager.disable_auto_checkpointing()


class SimulationCheckpointManager:
    """
    Manages simulation checkpoints for saving and resuming simulation state.
    Supports manual and automatic checkpointing, checkpoint listing, and cleanup.
    """
    def __init__(self, simulator: 'IntegratedSystemSimulator', checkpoint_dir: str = 'checkpoints'):
        self.simulator = simulator
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.auto_checkpoint_enabled = False
        self.auto_checkpoint_interval = 10
        self.auto_checkpoint_counter = 0
        self.max_auto_checkpoints = 5
        self.auto_checkpoint_prefix = 'auto_'
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load existing checkpoint metadata
        self._load_checkpoint_metadata()
    
    def _load_checkpoint_metadata(self) -> None:
        """Load metadata for existing checkpoints."""
        metadata_path = os.path.join(self.checkpoint_dir, 'checkpoint_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.checkpoints = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.checkpoints = {}
    
    def _save_checkpoint_metadata(self) -> None:
        """Save checkpoint metadata to disk."""
        metadata_path = os.path.join(self.checkpoint_dir, 'checkpoint_metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.checkpoints, f)
        except IOError:
            print(f"Warning: Failed to save checkpoint metadata to {metadata_path}")
    
    def create_checkpoint(self, checkpoint_id: Optional[str] = None) -> str:
        """
        Create a checkpoint of the current simulation state.
        
        Args:
            checkpoint_id: Optional identifier for the checkpoint
            
        Returns:
            The checkpoint ID
        """
        # Generate a checkpoint ID if not provided
        if checkpoint_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_id = f"checkpoint_{timestamp}"
        
        # Get the current simulation state
        with self.simulator.state_lock:
            state_copy = self.simulator.state.copy()
            simulation_time = self.simulator.time
            random_state = self.simulator.random_generator.get_state()
            simulation_log = self.simulator.simulation_log.copy()
        
        # Create the checkpoint data
        checkpoint_data = {
            'state': state_copy,
            'time': simulation_time,
            'random_state': random_state,
            'simulation_log': simulation_log,
            'time_dilation_factor': self.simulator.time_dilation_factor,
            'real_time_mode': self.simulator.real_time_mode
        }
        
        # Save the checkpoint to disk
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.pickle")
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except IOError as e:
            print(f"Error saving checkpoint: {e}")
            return ""
        
        # Update checkpoint metadata
        self.checkpoints[checkpoint_id] = {
            'id': checkpoint_id,
            'created_at': datetime.now().isoformat(),
            'simulation_time': simulation_time,
            'path': checkpoint_path,
            'auto_generated': checkpoint_id.startswith(self.auto_checkpoint_prefix)
        }
        
        # Save updated metadata
        self._save_checkpoint_metadata()
        
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Load a simulation state from a checkpoint.
        
        Args:
            checkpoint_id: Identifier of the checkpoint to load
            
        Returns:
            True if checkpoint was successfully loaded, False otherwise
        """
        if checkpoint_id not in self.checkpoints:
            print(f"Checkpoint '{checkpoint_id}' not found")
            return False
        
        checkpoint_path = self.checkpoints[checkpoint_id]['path']
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found at {checkpoint_path}")
            return False
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        except (pickle.PickleError, IOError) as e:
            print(f"Error loading checkpoint: {e}")
            return False
        
        # Restore simulation state
        with self.simulator.state_lock:
            self.simulator.state = checkpoint_data['state']
            self.simulator.time = checkpoint_data['time']
            self.simulator.random_generator.set_state(checkpoint_data['random_state'])
            self.simulator.simulation_log = checkpoint_data['simulation_log']
            self.simulator.time_dilation_factor = checkpoint_data.get('time_dilation_factor', 1.0)
            self.simulator.enable_real_time_mode(checkpoint_data.get('real_time_mode', False))
        
        return True
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata
        """
        return list(self.checkpoints.values())
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Identifier of the checkpoint to delete
            
        Returns:
            True if checkpoint was successfully deleted, False otherwise
        """
        if checkpoint_id not in self.checkpoints:
            return False
        
        checkpoint_path = self.checkpoints[checkpoint_id]['path']
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except OSError:
                print(f"Warning: Failed to delete checkpoint file {checkpoint_path}")
        
        # Remove from metadata
        del self.checkpoints[checkpoint_id]
        self._save_checkpoint_metadata()
        
        return True
    
    def enable_auto_checkpointing(self, interval: int = 10, max_checkpoints: int = 5) -> None:
        """
        Enable automatic checkpointing at regular intervals.
        
        Args:
            interval: Number of steps between checkpoints
            max_checkpoints: Maximum number of auto-checkpoints to keep
        """
        self.auto_checkpoint_enabled = True
        self.auto_checkpoint_interval = interval
        self.max_auto_checkpoints = max_checkpoints
        self.auto_checkpoint_counter = 0
    
    def disable_auto_checkpointing(self) -> None:
        """
        Disable automatic checkpointing.
        """
        self.auto_checkpoint_enabled = False
    
    def check_auto_checkpoint(self) -> Optional[str]:
        """
        Check if an automatic checkpoint should be created and create it if needed.
        
        Returns:
            Checkpoint ID if a checkpoint was created, None otherwise
        """
        if not self.auto_checkpoint_enabled:
            return None
        
        self.auto_checkpoint_counter += 1
        if self.auto_checkpoint_counter >= self.auto_checkpoint_interval:
            self.auto_checkpoint_counter = 0
            
            # Create auto checkpoint
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_id = f"{self.auto_checkpoint_prefix}{timestamp}"
            
            # Manage maximum number of auto checkpoints
            self._cleanup_auto_checkpoints()
            
            return self.create_checkpoint(checkpoint_id)
        
        return None
    
    def _cleanup_auto_checkpoints(self) -> None:
        """Clean up old auto-checkpoints if we exceed the maximum number."""
        auto_checkpoints = [
            (cp_id, cp_data) 
            for cp_id, cp_data in self.checkpoints.items() 
            if cp_data.get('auto_generated', False)
        ]
        
        # Sort by creation time (oldest first)
        auto_checkpoints.sort(key=lambda x: x[1]['created_at'])
        
        # Delete oldest checkpoints if we have too many
        while len(auto_checkpoints) >= self.max_auto_checkpoints:
            oldest_id, _ = auto_checkpoints.pop(0)
            self.delete_checkpoint(oldest_id)


class SystemFailureSimulator:
    """
    Comprehensive failure simulator for all spacecraft subsystems.
    Supports stochastic and deterministic fault injection, detection, and recovery
    across propulsion, power, communications, navigation, thermal, and other subsystems.
    """
    def __init__(self, failure_rates: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize the system failure simulator.
        
        Args:
            failure_rates: Dictionary mapping subsystems to their failure modes and rates
                           Format: {'subsystem': {'failure_mode': rate_per_second, ...}, ...}
        """
        self.failure_rates = failure_rates or {
            'propulsion': {},
            'power': {},
            'comms': {},
            'navigation': {},
            'thermal': {},
            'payload': {},
            'structure': {},
            'command': {}
        }
        self.active_failures: Dict[str, Set[str]] = {
            subsystem: set() for subsystem in self.failure_rates.keys()
        }
        
    def inject_random_failures(self, dt: float):
        """
        Probabilistically inject failures based on failure rates and time step.
        
        Args:
            dt: Time step in seconds
        """
        for subsystem, failure_modes in self.failure_rates.items():
            for mode, rate in failure_modes.items():
                if np.random.random() < 1.0 - pow(1.0 - rate, dt):
                    self.active_failures[subsystem].add(mode)
    
    def inject_failure(self, subsystem: str, mode: str):
        """
        Deterministically inject a specific failure.
        
        Args:
            subsystem: The subsystem to affect
            mode: The failure mode to inject
        """
        if subsystem in self.active_failures:
            self.active_failures[subsystem].add(mode)
    
    def clear_failure(self, subsystem: str, mode: str):
        """
        Clear a specific failure.
        
        Args:
            subsystem: The subsystem to affect
            mode: The failure mode to clear
        """
        if subsystem in self.active_failures:
            self.active_failures[subsystem].discard(mode)
    
    def clear_subsystem(self, subsystem: str):
        """
        Clear all failures for a specific subsystem.
        
        Args:
            subsystem: The subsystem to clear
        """
        if subsystem in self.active_failures:
            self.active_failures[subsystem].clear()
    
    def clear_all(self):
        """Clear all failures across all subsystems."""
        for failures in self.active_failures.values():
            failures.clear()
    
    def get_failures(self, subsystem: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        Get active failures, optionally filtered by subsystem.
        
        Args:
            subsystem: Optional subsystem to filter by
            
        Returns:
            Dictionary of active failures by subsystem
        """
        if subsystem:
            return {subsystem: self.active_failures.get(subsystem, set())}
        return {k: v.copy() for k, v in self.active_failures.items()}
    
    def apply_failures(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all active failures to the system state.
        
        Args:
            system_state: Current system state
            
        Returns:
            Modified system state with failures applied
        """
        state = system_state.copy()
        
        # Apply propulsion failures
        if 'propulsion' in self.active_failures:
            state = self._apply_propulsion_failures(state)
        
        # Apply power failures
        if 'power' in self.active_failures:
            state = self._apply_power_failures(state)
        
        # Apply communications failures
        if 'comms' in self.active_failures:
            state = self._apply_comms_failures(state)
        
        # Apply navigation failures
        if 'navigation' in self.active_failures:
            state = self._apply_navigation_failures(state)
        
        # Apply thermal failures
        if 'thermal' in self.active_failures:
            state = self._apply_thermal_failures(state)
        
        # Apply payload failures
        if 'payload' in self.active_failures:
            state = self._apply_payload_failures(state)
        
        # Apply structure failures
        if 'structure' in self.active_failures:
            state = self._apply_structure_failures(state)
        
        # Apply command failures
        if 'command' in self.active_failures:
            state = self._apply_command_failures(state)
        
        return state
    
    def _apply_propulsion_failures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply propulsion subsystem failures to state."""
        for mode in self.active_failures['propulsion']:
            if mode == 'engine_shutdown':
                state['propulsion'] = state.get('propulsion', {})
                state['propulsion']['thrust_N'] = 0.0
                state['propulsion']['engine_running'] = False
            elif mode == 'fuel_leak':
                state['propulsion'] = state.get('propulsion', {})
                leak_rate = np.random.uniform(0.01, 0.1)
                state['propulsion']['fuel_leak_kg'] = state['propulsion'].get('fuel_leak_kg', 0.0) + leak_rate
                state['propulsion']['fuel_remaining_kg'] = max(0, state['propulsion'].get('fuel_remaining_kg', 100.0) - leak_rate)
            elif mode == 'thrust_vector_failure':
                state['propulsion'] = state.get('propulsion', {})
                state['propulsion']['gimbal_deg'] = (0.0, 0.0)
                state['propulsion']['vectoring_active'] = False
            elif mode == 'throttle_stuck':
                state['propulsion'] = state.get('propulsion', {})
                state['propulsion']['throttle_stuck'] = True
                state['propulsion']['throttle_position'] = state['propulsion'].get('throttle_position', 0.5)
            elif mode == 'ignition_failure':
                state['propulsion'] = state.get('propulsion', {})
                state['propulsion']['ignition_failure'] = True
                state['propulsion']['engine_running'] = False
        return state
    
    def _apply_power_failures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply power subsystem failures to state."""
        for mode in self.active_failures['power']:
            if mode == 'solar_panel_degradation':
                state['power'] = state.get('power', {})
                state['power']['solar_efficiency'] = state['power'].get('solar_efficiency', 1.0) * 0.8
            elif mode == 'battery_failure':
                state['power'] = state.get('power', {})
                state['power']['battery_capacity'] = 0.0
            elif mode == 'power_bus_failure':
                state['power'] = state.get('power', {})
                state['power']['bus_voltage'] = 0.0
                # Propagate power failure to dependent systems
                self._propagate_power_failure(state)
            elif mode == 'voltage_regulator_failure':
                state['power'] = state.get('power', {})
                state['power']['regulated_voltage'] = np.random.uniform(0.5, 1.5) * state['power'].get('nominal_voltage', 5.0)
        return state
    
    def _apply_comms_failures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply communications subsystem failures to state."""
        for mode in self.active_failures['comms']:
            if mode == 'antenna_failure':
                state['comms'] = state.get('comms', {})
                state['comms']['signal_strength'] = 0.0
                state['comms']['transmitting'] = False
            elif mode == 'transceiver_failure':
                state['comms'] = state.get('comms', {})
                state['comms']['receiving'] = False
                state['comms']['transmitting'] = False
            elif mode == 'encryption_failure':
                state['comms'] = state.get('comms', {})
                state['comms']['secure_channel'] = False
            elif mode == 'data_corruption':
                state['comms'] = state.get('comms', {})
                state['comms']['packet_error_rate'] = np.random.uniform(0.3, 0.7)
        return state
    
    def _apply_navigation_failures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply navigation subsystem failures to state."""
        for mode in self.active_failures['navigation']:
            if mode == 'gyro_drift':
                state['navigation'] = state.get('navigation', {})
                drift = np.random.normal(0, 0.1, 3)
                state['navigation']['attitude_error_rad'] = drift
            elif mode == 'star_tracker_failure':
                state['navigation'] = state.get('navigation', {})
                state['navigation']['star_tracker_operational'] = False
            elif mode == 'gps_failure':
                state['navigation'] = state.get('navigation', {})
                state['navigation']['position_error_m'] = np.random.uniform(100, 1000)
            elif mode == 'imu_failure':
                state['navigation'] = state.get('navigation', {})
                state['navigation']['imu_operational'] = False
        return state
    
    def _apply_thermal_failures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply thermal subsystem failures to state."""
        for mode in self.active_failures['thermal']:
            if mode == 'heater_failure':
                state['thermal'] = state.get('thermal', {})
                state['thermal']['heater_operational'] = False
                # Simulate temperature drop
                state['thermal']['temperature_K'] = max(
                    state['thermal'].get('temperature_K', 293.0) - 20.0,
                    state['thermal'].get('min_safe_temp_K', 233.0) - 10.0
                )
            elif mode == 'radiator_failure':
                state['thermal'] = state.get('thermal', {})
                state['thermal']['radiator_operational'] = False
                # Simulate temperature increase
                state['thermal']['temperature_K'] = min(
                    state['thermal'].get('temperature_K', 293.0) + 30.0,
                    state['thermal'].get('max_safe_temp_K', 323.0) + 15.0
                )
            elif mode == 'sensor_failure':
                state['thermal'] = state.get('thermal', {})
                state['thermal']['temperature_sensor_operational'] = False
                state['thermal']['temperature_K'] = np.nan
        return state
    
    def _apply_payload_failures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply payload subsystem failures to state."""
        for mode in self.active_failures['payload']:
            if mode == 'instrument_failure':
                state['payload'] = state.get('payload', {})
                state['payload']['instrument_operational'] = False
            elif mode == 'data_storage_failure':
                state['payload'] = state.get('payload', {})
                state['payload']['storage_available_bytes'] = 0
            elif mode == 'calibration_drift':
                state['payload'] = state.get('payload', {})
                state['payload']['calibration_error'] = np.random.uniform(0.1, 0.5)
        return state
    
    def _apply_structure_failures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply structural subsystem failures to state."""
        for mode in self.active_failures['structure']:
            if mode == 'micrometeoroid_impact':
                state['structure'] = state.get('structure', {})
                state['structure']['hull_integrity'] = max(0.0, state['structure'].get('hull_integrity', 1.0) - np.random.uniform(0.05, 0.2))
            elif mode == 'thermal_stress':
                state['structure'] = state.get('structure', {})
                state['structure']['structural_integrity'] = max(0.0, state['structure'].get('structural_integrity', 1.0) - np.random.uniform(0.01, 0.1))
            elif mode == 'joint_failure':
                state['structure'] = state.get('structure', {})
                state['structure']['joint_integrity'] = max(0.0, state['structure'].get('joint_integrity', 1.0) - np.random.uniform(0.1, 0.3))
        return state
    
    def _apply_command_failures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply command and data handling failures to state."""
        for mode in self.active_failures['command']:
            if mode == 'processor_reset':
                state['command'] = state.get('command', {})
                state['command']['processor_uptime_s'] = 0.0
            elif mode == 'memory_corruption':
                state['command'] = state.get('command', {})
                state['command']['memory_errors'] = state['command'].get('memory_errors', 0) + np.random.randint(1, 10)
            elif mode == 'software_bug':
                state['command'] = state.get('command', {})
                state['command']['software_error'] = True
                state['command']['error_code'] = np.random.randint(1, 100)
        return state
    
    def _propagate_power_failure(self, state: Dict[str, Any]) -> None:
        """Propagate effects of power failure to dependent systems."""
        # This is a simplified model - in a real system, you'd have a more
        # detailed power dependency graph
        
        # Set critical systems to unpowered state
        for subsystem in ['propulsion', 'comms', 'navigation', 'payload']:
            if subsystem in state:
                state[subsystem]['powered'] = False
    
    def detect_failures(self, system_state: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
        """
        Detect failures based on system state.
        
        Args:
            system_state: Current system state
            
        Returns:
            Dictionary mapping subsystems to detected failure modes
        """
        detections = {subsystem: {} for subsystem in self.failure_rates.keys()}
        
        # Detect propulsion failures
        if 'propulsion' in system_state:
            prop_state = system_state['propulsion']
            if prop_state.get('thrust_N', 1.0) == 0.0 and prop_state.get('engine_running', True):
                detections['propulsion']['engine_shutdown'] = True
            if prop_state.get('fuel_leak_kg', 0.0) > 0.0:
                detections['propulsion']['fuel_leak'] = True
            if not prop_state.get('vectoring_active', True):
                detections['propulsion']['thrust_vector_failure'] = True
            if prop_state.get('throttle_stuck', False):
                detections['propulsion']['throttle_stuck'] = True
        
        # Detect power failures
        if 'power' in system_state:
            power_state = system_state['power']
            if power_state.get('solar_efficiency', 1.0) < 0.9:
                detections['power']['solar_panel_degradation'] = True
            if power_state.get('battery_capacity', 1.0) <= 0.0:
                detections['power']['battery_failure'] = True
            if power_state.get('bus_voltage', 5.0) <= 0.1:
                detections['power']['power_bus_failure'] = True
        
        # Add detection logic for other subsystems as needed
        
        return detections
