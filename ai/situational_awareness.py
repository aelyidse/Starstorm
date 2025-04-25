from typing import Dict, Any, List, Optional
import numpy as np

class SituationalAwarenessModel:
    """
    Models situational awareness by fusing sensor inputs to maintain an internal state of the environment, threats, and opportunities.
    Supports multi-sensor fusion, entity detection, and context summarization.
    """
    def __init__(self):
        self.sensor_history: List[Dict[str, Any]] = []
        self.entities: List[Dict[str, Any]] = []
        self.last_context: Optional[Dict[str, Any]] = None

    def ingest(self, sensor_data: Dict[str, Any]):
        self.sensor_history.append(sensor_data)
        self.update_entities(sensor_data)
        self.last_context = self.summarize_context()
        return self.last_context

    def update_entities(self, sensor_data: Dict[str, Any]):
        # Simple entity extraction: look for keys like 'detected_objects', 'targets', etc.
        if 'detected_objects' in sensor_data:
            for obj in sensor_data['detected_objects']:
                self.entities.append(obj)
        if 'targets' in sensor_data:
            for tgt in sensor_data['targets']:
                self.entities.append(tgt)

    def summarize_context(self) -> Dict[str, Any]:
        # Summarize current awareness: number of entities, threats, and key sensor states
        n_entities = len(self.entities)
        n_threats = sum(1 for e in self.entities if e.get('type') == 'threat')
        summary = {
            'n_entities': n_entities,
            'n_threats': n_threats,
            'recent_entities': self.entities[-5:],
            'last_sensor': self.sensor_history[-1] if self.sensor_history else None
        }
        return summary

    def get_current_context(self) -> Optional[Dict[str, Any]]:
        return self.last_context

    def get_entity_history(self) -> List[Dict[str, Any]]:
        return self.entities
