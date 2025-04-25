from typing import Dict, Any, List, Optional, Tuple, Set, Callable
import numpy as np
import random
import time
from enum import Enum
from manufacturing.digital_twin import ComponentDigitalTwin, ComponentState
from manufacturing.digital_twin_factory import DigitalTwinFactory

class AssemblyConstraintType(Enum):
    """Types of assembly constraints between components"""
    PRECEDENCE = "precedence"  # Component A must be assembled before component B
    GEOMETRIC = "geometric"    # Components have geometric constraints
    TOOLING = "tooling"        # Components require specific tooling
    ACCESSIBILITY = "accessibility"  # Component requires access path
    STABILITY = "stability"    # Assembly must remain stable during process

class AssemblyOperation:
    """Represents a single assembly operation in the manufacturing process"""
    def __init__(self, 
                 operation_id: str,
                 component_id: str,
                 duration: float,
                 required_tools: Optional[List[str]] = None,
                 required_skills: Optional[List[str]] = None):
        self.operation_id = operation_id
        self.component_id = component_id
        self.duration = duration  # Time in minutes
        self.required_tools = required_tools or []
        self.required_skills = required_skills or []
        self.predecessors: Set[str] = set()  # Operations that must occur before this one
        self.successors: Set[str] = set()    # Operations that must occur after this one
        
    def add_predecessor(self, operation_id: str) -> None:
        """Add a predecessor operation"""
        self.predecessors.add(operation_id)
        
    def add_successor(self, operation_id: str) -> None:
        """Add a successor operation"""
        self.successors.add(operation_id)
        
    def __repr__(self) -> str:
        return f"AssemblyOperation({self.operation_id}, component={self.component_id}, duration={self.duration})"


class AssemblySequenceOptimizer:
    """
    Optimizes assembly sequences based on constraints, time, cost, and quality.
    Supports multiple optimization algorithms and constraint types.
    """
    def __init__(self, digital_twin_factory: Optional[DigitalTwinFactory] = None):
        self.digital_twin_factory = digital_twin_factory
        self.operations: Dict[str, AssemblyOperation] = {}
        self.constraints: List[Dict[str, Any]] = []
        self.optimization_results: Dict[str, Any] = {}
        
    def add_operation(self, operation: AssemblyOperation) -> None:
        """Add an assembly operation to the sequence"""
        self.operations[operation.operation_id] = operation
        
    def add_constraint(self, 
                      constraint_type: AssemblyConstraintType,
                      from_operation: str,
                      to_operation: str,
                      weight: float = 1.0,
                      description: Optional[str] = None) -> None:
        """
        Add a constraint between operations
        
        Args:
            constraint_type: Type of constraint
            from_operation: Source operation ID
            to_operation: Target operation ID
            weight: Constraint importance weight
            description: Optional description of the constraint
        """
        if from_operation not in self.operations or to_operation not in self.operations:
            raise ValueError(f"Operations {from_operation} or {to_operation} not found")
            
        constraint = {
            'type': constraint_type,
            'from': from_operation,
            'to': to_operation,
            'weight': weight,
            'description': description
        }
        
        self.constraints.append(constraint)
        
        # Update operation dependencies for precedence constraints
        if constraint_type == AssemblyConstraintType.PRECEDENCE:
            self.operations[from_operation].add_successor(to_operation)
            self.operations[to_operation].add_predecessor(from_operation)
    
    def generate_valid_sequences(self) -> List[List[str]]:
        """
        Generate all valid assembly sequences based on constraints
        
        Returns:
            List of valid operation sequences (each a list of operation IDs)
        """
        # Build dependency graph
        graph = {op_id: set() for op_id in self.operations}
        for constraint in self.constraints:
            if constraint['type'] == AssemblyConstraintType.PRECEDENCE:
                graph[constraint['from']].add(constraint['to'])
        
        # Find operations with no predecessors
        def get_roots():
            roots = []
            for op_id, op in self.operations.items():
                if not op.predecessors:
                    roots.append(op_id)
            return roots
        
        # Generate sequences using backtracking
        valid_sequences = []
        
        def backtrack(sequence, remaining):
            if not remaining:
                valid_sequences.append(sequence.copy())
                return
                
            # Find operations that can be added next
            candidates = []
            for op_id in remaining:
                if all(pred not in remaining for pred in self.operations[op_id].predecessors):
                    candidates.append(op_id)
                    
            for op_id in candidates:
                sequence.append(op_id)
                remaining.remove(op_id)
                backtrack(sequence, remaining)
                remaining.add(op_id)
                sequence.pop()
        
        backtrack([], set(self.operations.keys()))
        return valid_sequences
    
    def evaluate_sequence(self, sequence: List[str]) -> Dict[str, float]:
        """
        Evaluate a sequence based on time, cost, and other metrics
        
        Args:
            sequence: List of operation IDs in execution order
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Check if sequence is valid
        if not self._is_valid_sequence(sequence):
            return {'valid': False, 'total_time': float('inf'), 'tool_changes': float('inf')}
            
        # Calculate total time
        total_time = sum(self.operations[op_id].duration for op_id in sequence)
        
        # Calculate tool changes
        tool_changes = 0
        current_tools = set()
        for op_id in sequence:
            op = self.operations[op_id]
            required_tools = set(op.required_tools)
            if required_tools and required_tools != current_tools:
                tool_changes += 1
                current_tools = required_tools
        
        # Calculate workstation transfers (simplified)
        workstation_transfers = 0
        
        return {
            'valid': True,
            'total_time': total_time,
            'tool_changes': tool_changes,
            'workstation_transfers': workstation_transfers
        }
    
    def _is_valid_sequence(self, sequence: List[str]) -> bool:
        """Check if a sequence satisfies all precedence constraints"""
        # Check if all operations are included
        if set(sequence) != set(self.operations.keys()):
            return False
            
        # Check precedence constraints
        completed = set()
        for op_id in sequence:
            op = self.operations[op_id]
            # Check if all predecessors have been completed
            if not op.predecessors.issubset(completed):
                return False
            completed.add(op_id)
            
        return True
    
    def optimize_sequence(self, 
                         method: str = "genetic",
                         objective: str = "time",
                         population_size: int = 100,
                         generations: int = 50) -> Dict[str, Any]:
        """
        Optimize assembly sequence using specified algorithm
        
        Args:
            method: Optimization method ("genetic", "simulated_annealing", "exhaustive")
            objective: Optimization objective ("time", "cost", "balanced")
            population_size: Population size for genetic algorithm
            generations: Number of generations for genetic algorithm
            
        Returns:
            Optimization results including best sequence
        """
        if method == "exhaustive":
            return self._optimize_exhaustive(objective)
        elif method == "genetic":
            return self._optimize_genetic(objective, population_size, generations)
        elif method == "simulated_annealing":
            return self._optimize_simulated_annealing(objective)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_exhaustive(self, objective: str) -> Dict[str, Any]:
        """Exhaustive search for optimal sequence (only practical for small problems)"""
        valid_sequences = self.generate_valid_sequences()
        
        best_sequence = None
        best_score = float('inf')
        
        for sequence in valid_sequences:
            evaluation = self.evaluate_sequence(sequence)
            
            # Calculate score based on objective
            if objective == "time":
                score = evaluation['total_time']
            elif objective == "balanced":
                score = evaluation['total_time'] + 5 * evaluation['tool_changes']
            else:
                score = evaluation['total_time']  # Default to time
                
            if score < best_score:
                best_score = score
                best_sequence = sequence
        
        result = {
            'method': 'exhaustive',
            'objective': objective,
            'best_sequence': best_sequence,
            'best_evaluation': self.evaluate_sequence(best_sequence) if best_sequence else None,
            'sequences_evaluated': len(valid_sequences)
        }
        
        self.optimization_results = result
        return result
    
    def _optimize_genetic(self, objective: str, population_size: int, generations: int) -> Dict[str, Any]:
        """Genetic algorithm for sequence optimization"""
        # Generate initial population
        population = self._generate_initial_population(population_size)
        
        best_sequence = None
        best_fitness = float('inf')
        fitness_history = []
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for sequence in population:
                evaluation = self.evaluate_sequence(sequence)
                
                # Calculate fitness based on objective
                if objective == "time":
                    fitness = evaluation['total_time']
                elif objective == "balanced":
                    fitness = evaluation['total_time'] + 5 * evaluation['tool_changes']
                else:
                    fitness = evaluation['total_time']  # Default to time
                    
                fitness_scores.append(fitness)
                
                # Update best solution
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_sequence = sequence.copy()
            
            # Record best fitness for this generation
            fitness_history.append(min(fitness_scores))
            
            # Create next generation
            next_population = []
            
            # Elitism: keep best solutions
            elite_count = max(1, population_size // 10)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elite_count]
            for i in elite_indices:
                next_population.append(population[i])
            
            # Fill rest with crossover and mutation
            while len(next_population) < population_size:
                # Selection
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
                
                # Crossover
                if random.random() < 0.8:  # 80% crossover rate
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                    
                # Mutation
                if random.random() < 0.2:  # 20% mutation rate
                    child = self._mutate(child)
                    
                # Add to next generation if valid
                if self._is_valid_sequence(child):
                    next_population.append(child)
            
            # Update population
            population = next_population
        
        result = {
            'method': 'genetic',
            'objective': objective,
            'population_size': population_size,
            'generations': generations,
            'best_sequence': best_sequence,
            'best_evaluation': self.evaluate_sequence(best_sequence) if best_sequence else None,
            'fitness_history': fitness_history
        }
        
        self.optimization_results = result
        return result
    
    def _generate_initial_population(self, size: int) -> List[List[str]]:
        """Generate initial population of valid sequences"""
        population = []
        
        # Start with topologically sorted sequence as seed
        topo_sequence = self._topological_sort()
        population.append(topo_sequence)
        
        # Generate more sequences by random swaps that preserve constraints
        while len(population) < size:
            # Take a random sequence from current population
            base_sequence = random.choice(population).copy()
            
            # Apply random swaps
            for _ in range(3):  # Apply multiple swaps
                i, j = random.sample(range(len(base_sequence)), 2)
                # Swap if it doesn't violate constraints
                sequence_copy = base_sequence.copy()
                sequence_copy[i], sequence_copy[j] = sequence_copy[j], sequence_copy[i]
                if self._is_valid_sequence(sequence_copy):
                    base_sequence = sequence_copy
            
            # Add to population if not already present
            if base_sequence not in population:
                population.append(base_sequence)
        
        return population
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort to get a valid sequence"""
        # Build adjacency list
        graph = {op_id: set() for op_id in self.operations}
        for op_id, op in self.operations.items():
            for succ in op.successors:
                graph[op_id].add(succ)
        
        # Find operations with no predecessors
        def get_roots():
            roots = []
            for op_id, op in self.operations.items():
                if not op.predecessors:
                    roots.append(op_id)
            return roots
        
        # Perform topological sort
        result = []
        visited = set()
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                visit(neighbor)
            result.append(node)
        
        for root in get_roots():
            visit(root)
            
        # Reverse to get correct order
        return result[::-1]
    
    def _selection(self, population: List[List[str]], fitness_scores: List[float]) -> List[str]:
        """Tournament selection"""
        # Select random candidates
        candidates = random.sample(range(len(population)), 3)
        # Return the best one
        best_idx = min(candidates, key=lambda i: fitness_scores[i])
        return population[best_idx]
    
    def _crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """Order crossover that preserves precedence constraints"""
        # Try to create valid child through crossover
        for _ in range(5):  # Try a few times
            # Select crossover points
            length = len(parent1)
            point1, point2 = sorted(random.sample(range(length + 1), 2))
            
            # Create child using segment from parent1
            segment = parent1[point1:point2]
            remaining = [op for op in parent2 if op not in segment]
            
            child = remaining[:point1] + segment + remaining[point1:]
            
            if self._is_valid_sequence(child):
                return child
        
        # If no valid child found, return parent1
        return parent1
    
    def _mutate(self, sequence: List[str]) -> List[str]:
        """Mutate by swapping operations if constraints allow"""
        for _ in range(5):  # Try a few times
            i, j = random.sample(range(len(sequence)), 2)
            mutated = sequence.copy()
            mutated[i], mutated[j] = mutated[j], mutated[i]
            
            if self._is_valid_sequence(mutated):
                return mutated
                
        return sequence  # Return original if no valid mutation found
    
    def _optimize_simulated_annealing(self, objective: str) -> Dict[str, Any]:
        """Simulated annealing for sequence optimization"""
        # Start with a valid sequence
        current_sequence = self._topological_sort()
        current_evaluation = self.evaluate_sequence(current_sequence)
        
        # Calculate initial score
        if objective == "time":
            current_score = current_evaluation['total_time']
        elif objective == "balanced":
            current_score = current_evaluation['total_time'] + 5 * current_evaluation['tool_changes']
        else:
            current_score = current_evaluation['total_time']
            
        best_sequence = current_sequence.copy()
        best_score = current_score
        
        # Simulated annealing parameters
        temperature = 100.0
        cooling_rate = 0.95
        iterations = 1000
        
        score_history = []
        
        for iteration in range(iterations):
            # Generate neighbor by swapping two operations
            neighbor = current_sequence.copy()
            
            # Try to find valid neighbor
            valid_neighbor = False
            for _ in range(10):  # Try multiple times
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor_candidate = neighbor.copy()
                neighbor_candidate[i], neighbor_candidate[j] = neighbor_candidate[j], neighbor_candidate[i]
                
                if self._is_valid_sequence(neighbor_candidate):
                    neighbor = neighbor_candidate
                    valid_neighbor = True
                    break
            
            if not valid_neighbor:
                continue
                
            # Evaluate neighbor
            neighbor_evaluation = self.evaluate_sequence(neighbor)
            
            # Calculate neighbor score
            if objective == "time":
                neighbor_score = neighbor_evaluation['total_time']
            elif objective == "balanced":
                neighbor_score = neighbor_evaluation['total_time'] + 5 * neighbor_evaluation['tool_changes']
            else:
                neighbor_score = neighbor_evaluation['total_time']
                
            # Decide whether to accept neighbor
            delta = neighbor_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_sequence = neighbor
                current_score = neighbor_score
                
                # Update best solution if improved
                if current_score < best_score:
                    best_sequence = current_sequence.copy()
                    best_score = current_score
            
            # Record best score for this iteration
            score_history.append(best_score)
            
            # Cool down
            temperature *= cooling_rate
        
        result = {
            'method': 'simulated_annealing',
            'objective': objective,
            'iterations': iterations,
            'best_sequence': best_sequence,
            'best_evaluation': self.evaluate_sequence(best_sequence),
            'score_history': score_history
        }
        
        self.optimization_results = result
        return result
    
    def visualize_sequence(self, sequence: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate visualization data for an assembly sequence
        
        Args:
            sequence: Sequence to visualize (uses best sequence from optimization if None)
            
        Returns:
            Visualization data
        """
        if sequence is None:
            if not self.optimization_results or 'best_sequence' not in self.optimization_results:
                raise ValueError("No optimized sequence available")
            sequence = self.optimization_results['best_sequence']
            
        # Generate timeline data
        timeline = []
        current_time = 0
        
        for op_id in sequence:
            op = self.operations[op_id]
            timeline.append({
                'operation_id': op_id,
                'component_id': op.component_id,
                'start_time': current_time,
                'end_time': current_time + op.duration,
                'duration': op.duration,
                'tools': op.required_tools
            })
            current_time += op.duration
            
        # Generate dependency graph data
        dependencies = []
        for constraint in self.constraints:
            if constraint['type'] == AssemblyConstraintType.PRECEDENCE:
                dependencies.append({
                    'from': constraint['from'],
                    'to': constraint['to'],
                    'type': constraint['type'].value
                })
                
        return {
            'sequence': sequence,
            'timeline': timeline,
            'dependencies': dependencies,
            'total_time': sum(op.duration for op_id, op in self.operations.items())
        }
    
    def create_digital_twins_from_sequence(self, 
                                         sequence: Optional[List[str]] = None) -> Dict[str, ComponentDigitalTwin]:
        """
        Create digital twins for components in assembly sequence
        
        Args:
            sequence: Assembly sequence (uses best sequence from optimization if None)
            
        Returns:
            Dictionary of component digital twins
        """
        if not self.digital_twin_factory:
            raise ValueError("Digital twin factory not provided")
            
        if sequence is None:
            if not self.optimization_results or 'best_sequence' not in self.optimization_results:
                raise ValueError("No optimized sequence available")
            sequence = self.optimization_results['best_sequence']
            
        # Create digital twins for each component
        twins = {}
        for op_id in sequence:
            op = self.operations[op_id]
            component_id = op.component_id
            
            if component_id not in twins:
                # Create digital twin for this component
                twin = self.digital_twin_factory.create_twin(
                    name=component_id,
                    component_type="assembly_component",
                    initial_state=ComponentState.MANUFACTURING
                )
                twins[component_id] = twin
                
        return twins