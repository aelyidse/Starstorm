from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class SpikingNeuron:
    """Represents a spiking neuron in a neuromorphic architecture."""
    membrane_potential: float = 0.0
    threshold: float = 1.0
    reset_potential: float = 0.0
    leak_rate: float = 0.1
    spike_history: List[float] = None
    
    def __post_init__(self):
        if self.spike_history is None:
            self.spike_history = []

    def integrate(self, input_current: float, dt: float = 0.001) -> bool:
        """Integrate input current and determine if neuron spikes."""
        # Leaky integrate-and-fire model
        self.membrane_potential += (input_current - self.leak_rate * self.membrane_potential) * dt
        if self.membrane_potential >= self.threshold:
            self.spike_history.append(1.0)
            self.membrane_potential = self.reset_potential
            return True
        self.spike_history.append(0.0)
        return False

    def get_spike_train(self) -> List[float]:
        """Return the spike history."""
        return self.spike_history[-100:]  # Last 100 timesteps for memory efficiency

@dataclass
class Synapse:
    """Represents a synaptic connection with plasticity."""
    weight: float = 0.1
    delay: int = 1  # Timesteps
    plasticity_rate: float = 0.01  # For STDP
    pre_neuron: Optional[SpikingNeuron] = None
    post_neuron: Optional[SpikingNeuron] = None

    def update_weight(self, pre_spike: bool, post_spike: bool) -> None:
        """Update synaptic weight based on spike-timing-dependent plasticity (STDP)."""
        if pre_spike and post_spike:
            # If pre fires before post, strengthen connection (causal)
            self.weight += self.plasticity_rate * (1.0 if len(self.pre_neuron.spike_history) > len(self.post_neuron.spike_history) else -1.0)
        self.weight = max(0.01, min(1.0, self.weight))  # Bound weight

class NeuromorphicLayer:
    """A layer of spiking neurons with synaptic connections."""
    
    def __init__(self, num_neurons: int, threshold: float = 1.0, leak_rate: float = 0.1):
        self.neurons = [SpikingNeuron(threshold=threshold, leak_rate=leak_rate) for _ in range(num_neurons)]
        self.synapses: List[Synapse] = []

    def connect(self, pre_layer: 'NeuromorphicLayer', connectivity: float = 0.5) -> None:
        """Connect this layer to a previous layer with given connectivity probability."""
        for pre_n in pre_layer.neurons:
            for post_n in self.neurons:
                if np.random.random() < connectivity:
                    synapse = Synapse(weight=np.random.uniform(0.05, 0.2), 
                                    delay=np.random.randint(1, 5),
                                    pre_neuron=pre_n,
                                    post_neuron=post_n)
                    self.synapses.append(synapse)

    def process(self, input_currents: List[float], dt: float = 0.001) -> List[bool]:
        """Process input currents through the layer and return spike outputs."""
        spikes = []
        for i, neuron in enumerate(self.neurons):
            current = input_currents[i] if i < len(input_currents) else 0.0
            # Add synaptic inputs
            for synapse in self.synapses:
                if synapse.post_neuron == neuron and len(synapse.pre_neuron.spike_history) > synapse.delay:
                    if synapse.pre_neuron.spike_history[-synapse.delay-1] > 0:
                        current += synapse.weight
            spikes.append(neuron.integrate(current, dt))
        # Update synaptic weights based on STDP
        for synapse in self.synapses:
            pre_spike = synapse.pre_neuron.spike_history[-1] > 0 if synapse.pre_neuron else False
            post_spike = synapse.post_neuron.spike_history[-1] > 0 if synapse.post_neuron else False
            synapse.update_weight(pre_spike, post_spike)
        return spikes

class NeuromorphicNetwork:
    """A neuromorphic network architecture simulating brain-like computation for AI."""
    
    def __init__(self, layer_sizes: List[int], threshold: float = 1.0, leak_rate: float = 0.1):
        self.layers = [NeuromorphicLayer(size, threshold, leak_rate) for size in layer_sizes]
        self.connect_layers(connectivity=0.5)
        self.energy_consumption: float = 0.0  # Simulated energy in arbitrary units

    def connect_layers(self, connectivity: float = 0.5) -> None:
        """Connect consecutive layers in the network."""
        for i in range(len(self.layers) - 1):
            self.layers[i+1].connect(self.layers[i], connectivity)

    def forward(self, input_spikes: List[float], timesteps: int = 100, dt: float = 0.001) -> List[float]:
        """Propagate input spikes through the network over multiple timesteps."""
        activity_history = []
        for t in range(timesteps):
            current_layer_input = input_spikes if t < len(input_spikes) else [0.0] * len(self.layers[0].neurons)
            for layer in self.layers:
                spikes = layer.process(current_layer_input, dt)
                current_layer_input = [1.0 if s else 0.0 for s in spikes]
                self.energy_consumption += len(spikes) * 0.001  # Simulated energy cost per spike
            activity_history.append(sum(current_layer_input))
        # Return output layer activity as result
        return [n.spike_history[-1] for n in self.layers[-1].neurons]

    def train(self, input_patterns: List[List[float]], target_outputs: List[List[float]], 
              epochs: int = 10, timesteps: int = 100) -> float:
        """Train the network using unsupervised STDP learning on input patterns."""
        total_error = 0.0
        for _ in range(epochs):
            for pattern, target in zip(input_patterns, target_outputs):
                output = self.forward(pattern, timesteps)
                total_error += sum((o - t) ** 2 for o, t in zip(output, target)) if len(output) == len(target) else 0
        return total_error / (epochs * len(input_patterns)) if input_patterns else 0.0

    def get_network_activity(self) -> Dict[str, float]:
        """Return metrics about network activity and energy consumption."""
        total_spikes = sum(len([s for s in neuron.spike_history if s > 0]) 
                          for layer in self.layers for neuron in layer.neurons)
        total_neurons = sum(len(layer.neurons) for layer in self.layers)
        return {
            "total_spikes": total_spikes,
            "average_activity": total_spikes / total_neurons if total_neurons > 0 else 0.0,
            "energy_consumption": self.energy_consumption
        }

class NeuromorphicDecisionMaker:
    """Uses neuromorphic network for autonomous decision making in space combat scenarios."""
    
    def __init__(self, input_size: int = 10, hidden_sizes: List[int] = None, output_size: int = 5):
        sizes = [input_size] + (hidden_sizes or [20, 10]) + [output_size]
        self.network = NeuromorphicNetwork(layer_sizes=sizes)
        self.threat_memory: List[List[float]] = []
        self.response_memory: List[List[float]] = []

    def encode_sensory_input(self, sensor_data: Dict[str, float]) -> List[float]:
        """Encode sensor data into spike patterns for neuromorphic input."""
        # Simplified encoding: convert sensor values to spike rates
        spike_rates = []
        for value in sensor_data.values():
            rate = min(1.0, max(0.0, value / 1e-3))  # Normalize to reasonable range
            spike_rates.extend([rate] * 2)  # Duplicate for robustness
        return spike_rates[:len(self.network.layers[0].neurons)]

    def decode_decision(self, output_spikes: List[float]) -> Dict[str, float]:
        """Decode output spikes into actionable decisions."""
        # Simplified decoding: map output neurons to decision categories
        decisions = {
            "threat_level": sum(output_spikes[:2]) / 2.0 if len(output_spikes) > 2 else 0.0,
            "evade_priority": output_spikes[2] if len(output_spikes) > 3 else 0.0,
            "engage_priority": output_spikes[3] if len(output_spikes) > 4 else 0.0,
            "stealth_mode": output_spikes[4] if len(output_spikes) > 5 else 0.0
        }
        return decisions

    def learn_from_experience(self, sensor_data: Dict[str, float], outcome: float) -> None:
        """Learn from experience using feedback."""
        input_pattern = self.encode_sensory_input(sensor_data)
        target = [outcome] * len(self.network.layers[-1].neurons)  # Simplified target
        self.threat_memory.append(input_pattern)
        self.response_memory.append(target)
        if len(self.threat_memory) > 10:  # Limit memory size
            self.threat_memory.pop(0)
            self.response_memory.pop(0)
        self.network.train(self.threat_memory, self.response_memory, epochs=1)

    def make_decision(self, sensor_data: Dict[str, float], timesteps: int = 100) -> Dict[str, float]:
        """Process sensor data through neuromorphic network to make a decision."""
        input_pattern = self.encode_sensory_input(sensor_data)
        output_spikes = self.network.forward(input_pattern, timesteps)
        return self.decode_decision(output_spikes)

    def get_processing_metrics(self) -> Dict[str, float]:
        """Get metrics about neuromorphic processing."""
        activity = self.network.get_network_activity()
        return {
            "computational_efficiency": activity["total_spikes"] / activity["energy_consumption"] if activity["energy_consumption"] > 0 else 0.0,
            "memory_patterns": len(self.threat_memory),
            "energy_consumption": activity["energy_consumption"]
        }

# Example usage
if __name__ == "__main__":
    # Create a neuromorphic decision maker
    decision_maker = NeuromorphicDecisionMaker(input_size=10, hidden_sizes=[20, 10], output_size=5)
    
    # Simulate sensor data for a threat scenario
    sensor_data = {
        "magnetic_field": 5e-6,
        "em_field": 1e-3,
        "gravitational_anomaly": 1e-5,
        "image_threat_score": 0.8
    }
    
    # Make a decision based on sensor input
    decision = decision_maker.make_decision(sensor_data)
    print("Decision Output:")
    for key, value in decision.items():
        print(f"  {key}: {value:.2f}")
    
    # Learn from the experience (positive outcome)
    decision_maker.learn_from_experience(sensor_data, outcome=0.9)
    
    # Check processing metrics
    metrics = decision_maker.get_processing_metrics()
    print("\nProcessing Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    # Simulate another decision after learning
    sensor_data.update({"image_threat_score": 0.9})  # Increased threat
    decision_after_learning = decision_maker.make_decision(sensor_data)
    print("\nDecision After Learning:")
    for key, value in decision_after_learning.items():
        print(f"  {key}: {value:.2f}")
