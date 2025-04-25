from typing import Dict, Any, List, Optional
import numpy as np

class PredictiveDisplay:
    """
    Provides predictive displays for operator decision support.
    Visualizes future states, warnings, and recommended actions based on current telemetry and predictive models.
    """
    def __init__(self):
        self.prediction_history: List[Dict[str, Any]] = []
        self.last_prediction: Optional[Dict[str, Any]] = None

    def predict_future_state(self, current_state: Dict[str, Any], model_func, horizon: float = 60.0, dt: float = 5.0) -> List[Dict[str, Any]]:
        """
        model_func: function(current_state, dt) -> next_state
        horizon: prediction time horizon (seconds)
        dt: timestep for prediction
        """
        states = [current_state.copy()]
        state = current_state.copy()
        for _ in range(int(horizon // dt)):
            state = model_func(state, dt)
            states.append(state.copy())
        self.last_prediction = {'states': states, 'horizon': horizon, 'dt': dt}
        self.prediction_history.append(self.last_prediction)
        return states

    def generate_display_data(self, predicted_states: List[Dict[str, Any]], alert_func = None) -> Dict[str, Any]:
        # Summarize key trends, warnings, and recommendations
        display = {'future_states': predicted_states}
        if alert_func:
            alerts = [alert_func(s) for s in predicted_states]
            display['alerts'] = [a for a in alerts if a]
        return display

    def get_last_prediction(self) -> Optional[Dict[str, Any]]:
        return self.last_prediction

    def get_prediction_history(self) -> List[Dict[str, Any]]:
        return self.prediction_history
