import numpy as np
from typing import Dict, Any, List, Optional

class TargetTrack:
    """
    Represents a tracked target with state, history, and confidence.
    """
    def __init__(self, track_id: int, init_state: Dict[str, Any], sensor: str):
        self.track_id = track_id
        self.state = init_state  # e.g., {'pos': np.array, 'vel': np.array}
        self.sensor = sensor
        self.history = [init_state]
        self.confidence = 1.0
        self.last_update = 0.0

    def update(self, new_state: Dict[str, Any], sensor: str, confidence: float, timestamp: float):
        self.state = new_state
        self.sensor = sensor
        self.confidence = confidence
        self.last_update = timestamp
        self.history.append(new_state)

class TargetTrackerWithHandoff:
    """
    Tracks targets and manages sensor handoff for persistent surveillance.
    Supports state prediction, update, and dynamic handoff between sensors.
    """
    def __init__(self, sensors: List[str]):
        self.sensors = sensors
        self.tracks: Dict[int, TargetTrack] = {}
        self.next_id = 0

    def initiate_track(self, init_state: Dict[str, Any], sensor: str, timestamp: float) -> int:
        track = TargetTrack(self.next_id, init_state, sensor)
        track.last_update = timestamp
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track.track_id

    def update_track(self, track_id: int, new_state: Dict[str, Any], sensor: str, confidence: float, timestamp: float):
        if track_id in self.tracks:
            self.tracks[track_id].update(new_state, sensor, confidence, timestamp)

    def predict_state(self, track_id: int, dt: float) -> Optional[Dict[str, Any]]:
        # Simple constant-velocity prediction
        track = self.tracks.get(track_id)
        if not track or 'pos' not in track.state or 'vel' not in track.state:
            return None
        pred_pos = np.array(track.state['pos']) + dt * np.array(track.state['vel'])
        return {'pos': pred_pos, 'vel': track.state['vel']}

    def handoff(self, track_id: int, new_sensor: str):
        # Assigns a new sensor to the track (e.g., when target leaves FOV)
        if track_id in self.tracks and new_sensor in self.sensors:
            self.tracks[track_id].sensor = new_sensor
            self.tracks[track_id].confidence *= 0.9  # Slightly reduce confidence on handoff

    def get_tracks(self) -> List[TargetTrack]:
        return list(self.tracks.values())

    def get_track_states(self) -> List[Dict[str, Any]]:
        return [{
            'track_id': t.track_id,
            'state': t.state,
            'sensor': t.sensor,
            'confidence': t.confidence,
            'last_update': t.last_update
        } for t in self.tracks.values()]
