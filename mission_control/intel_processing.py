import numpy as np
from typing import Dict, Any, List, Optional

class IntelligenceExtractor:
    """
    Processes raw and fused sensor data to extract actionable intelligence.
    Supports feature extraction, event detection, and summary generation for multi-modal inputs.
    """
    def __init__(self):
        pass

    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Example: extract statistical and geometric features from imagery or tracks
        features = {}
        if 'image' in data:
            img = data['image']
            features['mean_intensity'] = float(np.mean(img))
            features['max_intensity'] = float(np.max(img))
            features['std_intensity'] = float(np.std(img))
        if 'tracks' in data:
            # Extract movement, velocity, clustering
            positions = np.array([t['pos'] for t in data['tracks'] if 'pos' in t])
            if len(positions) > 1:
                features['track_centroid'] = positions.mean(axis=0).tolist()
                features['track_spread'] = float(np.linalg.norm(positions - positions.mean(axis=0)))
        return features

    def detect_events(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Example: detect anomalies or significant events
        events = []
        if 'max_intensity' in features and features['max_intensity'] > 250:
            events.append({'type': 'bright_spot', 'value': features['max_intensity']})
        if 'track_spread' in features and features['track_spread'] > 100:
            events.append({'type': 'dispersed_targets', 'value': features['track_spread']})
        return events

    def summarize(self, features: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
        # Generate a human-readable summary
        summary = []
        for k, v in features.items():
            summary.append(f"{k}: {v}")
        for e in events:
            summary.append(f"Event: {e['type']} (value={e['value']})")
        return "; ".join(summary)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        features = self.extract_features(data)
        events = self.detect_events(features)
        summary = self.summarize(features, events)
        return {'features': features, 'events': events, 'summary': summary}
