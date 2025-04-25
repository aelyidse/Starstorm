from typing import Dict, Any, List, Optional
import numpy as np

class SystemPerformanceLearner:
    """
    Implements learning algorithms to improve system performance over time.
    Supports reinforcement learning, experience replay, and parameter adaptation.
    """
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.experience_buffer: List[Dict[str, Any]] = []
        self.policy_params: Dict[str, float] = {}  # e.g., action weights or thresholds
        self.performance_history: List[float] = []

    def record_experience(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any]):
        self.experience_buffer.append({'state': state, 'action': action, 'reward': reward, 'next_state': next_state})

    def update_policy(self):
        # Simple Q-learning-like update for demonstration
        if not self.experience_buffer:
            return
        for exp in self.experience_buffer:
            action = exp['action']
            reward = exp['reward']
            # Update policy parameter (e.g., increase weight for rewarding actions)
            old_val = self.policy_params.get(action, 0.0)
            self.policy_params[action] = old_val + self.learning_rate * (reward - old_val)
        self.experience_buffer.clear()

    def adapt_parameters(self, metrics: Dict[str, float]):
        # Adapt system parameters based on performance metrics
        for k, v in metrics.items():
            old = self.policy_params.get(k, 1.0)
            self.policy_params[k] = old + self.learning_rate * (v - old)
        self.performance_history.append(np.mean(list(metrics.values())))

    def get_policy(self) -> Dict[str, float]:
        return self.policy_params

    def get_performance_history(self) -> List[float]:
        return self.performance_history
