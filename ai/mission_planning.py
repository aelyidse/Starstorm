from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import time

class DecisionNode:
    """
    Represents a node in a decision tree for mission planning.
    """
    def __init__(self, attribute: str, threshold: float, left=None, right=None, value=None):
        self.attribute = attribute  # The attribute to test
        self.threshold = threshold  # The threshold for the test
        self.left = left            # Left subtree (if attribute < threshold)
        self.right = right          # Right subtree (if attribute >= threshold)
        self.value = value          # For leaf nodes, the decision value

class MissionPlanningModule:
    """
    Provides mission planning capabilities using decision tree evaluation.
    Integrates with tactical analysis to generate optimal mission plans.
    """
    def __init__(self, tactical_analyzer=None, decision_maker=None):
        self.tactical_analyzer = tactical_analyzer
        self.decision_maker = decision_maker
        self.decision_tree = None
        self.mission_history: List[Dict[str, Any]] = []
        self.current_mission: Optional[Dict[str, Any]] = None
        self.mission_status: str = "idle"
        
    def build_decision_tree(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Build a decision tree from training data for mission planning.
        
        Args:
            training_data: List of scenarios with attributes and outcomes
        """
        self.decision_tree = self._build_tree(training_data)
        
    def _build_tree(self, data: List[Dict[str, Any]], depth: int = 0, max_depth: int = 10) -> Optional[DecisionNode]:
        """
        Recursively build a decision tree.
        
        Args:
            data: Training data
            depth: Current depth in the tree
            max_depth: Maximum tree depth
            
        Returns:
            Root node of the decision tree
        """
        # Stop conditions
        if depth >= max_depth or len(data) < 2:
            # Create a leaf node with the most common outcome
            outcomes = [d.get('outcome', {}) for d in data]
            if not outcomes:
                return None
            # Simplified: just use the first outcome as the leaf value
            return DecisionNode(attribute=None, threshold=None, value=outcomes[0])
            
        # Find the best attribute and threshold to split on
        best_attribute, best_threshold = self._find_best_split(data)
        
        if best_attribute is None:
            # No good split found
            outcomes = [d.get('outcome', {}) for d in data]
            return DecisionNode(attribute=None, threshold=None, value=outcomes[0] if outcomes else None)
            
        # Split the data
        left_data = [d for d in data if d.get(best_attribute, 0) < best_threshold]
        right_data = [d for d in data if d.get(best_attribute, 0) >= best_threshold]
        
        # Create the node and its children
        left_child = self._build_tree(left_data, depth + 1, max_depth)
        right_child = self._build_tree(right_data, depth + 1, max_depth)
        
        return DecisionNode(
            attribute=best_attribute,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
        
    def _find_best_split(self, data: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
        """
        Find the best attribute and threshold to split the data.
        
        Args:
            data: Training data
            
        Returns:
            Tuple of (best_attribute, best_threshold)
        """
        if not data:
            return None, None
            
        # Get all attributes except 'outcome'
        attributes = set()
        for d in data:
            attributes.update(d.keys())
        attributes.discard('outcome')
        
        best_gain = -float('inf')
        best_attribute = None
        best_threshold = None
        
        # For each attribute, find the best threshold
        for attribute in attributes:
            values = [d.get(attribute, 0) for d in data if attribute in d]
            if not values:
                continue
                
            # Try different thresholds (simplified)
            thresholds = sorted(set(values))
            if len(thresholds) <= 1:
                continue
                
            # Use midpoints between values as potential thresholds
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2
                
                # Calculate information gain for this split
                gain = self._calculate_gain(data, attribute, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = attribute
                    best_threshold = threshold
                    
        return best_attribute, best_threshold
        
    def _calculate_gain(self, data: List[Dict[str, Any]], attribute: str, threshold: float) -> float:
        """
        Calculate information gain for a split.
        
        Args:
            data: Training data
            attribute: Attribute to split on
            threshold: Threshold for the split
            
        Returns:
            Information gain
        """
        # Simplified implementation - in a real system, would use entropy or Gini impurity
        left_data = [d for d in data if d.get(attribute, 0) < threshold]
        right_data = [d for d in data if d.get(attribute, 0) >= threshold]
        
        if not left_data or not right_data:
            return 0.0
            
        # Simple metric: balance of the split (closer to 50/50 is better)
        balance = min(len(left_data), len(right_data)) / len(data)
        return balance
        
    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a scenario using the decision tree.
        
        Args:
            scenario: Current tactical situation
            
        Returns:
            Recommended mission plan
        """
        if not self.decision_tree:
            return {"error": "Decision tree not built"}
            
        return self._traverse_tree(self.decision_tree, scenario)
        
    def _traverse_tree(self, node: DecisionNode, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traverse the decision tree to evaluate a scenario.
        
        Args:
            node: Current node in the tree
            scenario: Scenario to evaluate
            
        Returns:
            Decision at the leaf node
        """
        # If it's a leaf node, return its value
        if node.value is not None:
            return node.value
            
        # Get the attribute value from the scenario
        value = scenario.get(node.attribute, 0)
        
        # Traverse left or right based on the threshold
        if value < node.threshold:
            if node.left:
                return self._traverse_tree(node.left, scenario)
        else:
            if node.right:
                return self._traverse_tree(node.right, scenario)
                
        # If we reach here, something went wrong
        return {"error": "Unable to make a decision"}
        
    def plan_mission(self, tactical_situation: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan a mission based on the current tactical situation and constraints.
        
        Args:
            tactical_situation: Current tactical situation from the analyzer
            constraints: Mission constraints and requirements
            
        Returns:
            Mission plan with objectives, waypoints, and contingencies
        """
        # If we have a decision tree, use it
        if self.decision_tree:
            base_plan = self.evaluate_scenario(tactical_situation)
        else:
            # Fallback planning without a decision tree
            base_plan = self._default_planning(tactical_situation)
            
        # Refine the plan with constraints
        refined_plan = self._refine_plan(base_plan, constraints)
        
        # Generate options for multi-objective decision making
        if self.decision_maker:
            options = self._generate_plan_options(refined_plan, constraints)
            selected_plan = self.decision_maker.select(options, method='weighted_sum', constraints=constraints)
        else:
            selected_plan = refined_plan
            
        # Record the mission
        mission = {
            'plan': selected_plan,
            'tactical_situation': tactical_situation,
            'constraints': constraints,
            'timestamp': time.time(),
            'status': 'planned'
        }
        
        self.mission_history.append(mission)
        self.current_mission = mission
        self.mission_status = "planned"
        
        return selected_plan
        
    def _default_planning(self, tactical_situation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default planning method when no decision tree is available.
        
        Args:
            tactical_situation: Current tactical situation
            
        Returns:
            Basic mission plan
        """
        # Extract key information from tactical situation
        threats = tactical_situation.get('threats', [])
        
        # Generate objectives based on threats
        objectives = []
        for threat in threats:
            if threat.get('priority', 0) > 0.7:
                objectives.append({
                    'type': 'neutralize',
                    'target': threat,
                    'priority': threat.get('priority', 0)
                })
            elif threat.get('priority', 0) > 0.3:
                objectives.append({
                    'type': 'monitor',
                    'target': threat,
                    'priority': threat.get('priority', 0)
                })
                
        # Generate waypoints (simplified)
        waypoints = []
        for obj in objectives:
            if 'target' in obj and 'details' in obj['target']:
                target_details = obj['target']['details']
                if 'position' in target_details:
                    waypoints.append({
                        'position': target_details['position'],
                        'objective': obj['type'],
                        'priority': obj['priority']
                    })
                    
        return {
            'objectives': objectives,
            'waypoints': waypoints,
            'contingencies': []
        }
        
    def _refine_plan(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine a mission plan based on constraints.
        
        Args:
            plan: Base mission plan
            constraints: Mission constraints
            
        Returns:
            Refined mission plan
        """
        refined = plan.copy()
        
        # Apply time constraints
        if 'max_duration' in constraints:
            # Prioritize objectives to fit within time constraint
            objectives = sorted(refined.get('objectives', []), key=lambda x: x.get('priority', 0), reverse=True)
            refined['objectives'] = objectives[:constraints.get('max_objective_count', len(objectives))]
            
        # Apply resource constraints
        if 'available_resources' in constraints:
            # Adjust plan based on available resources
            resources = constraints['available_resources']
            if 'fuel' in resources and resources['fuel'] < 100:
                # Reduce mission scope for low fuel
                waypoints = sorted(refined.get('waypoints', []), key=lambda x: x.get('priority', 0), reverse=True)
                refined['waypoints'] = waypoints[:len(waypoints)//2]
                
        # Add contingencies
        refined['contingencies'] = self._generate_contingencies(refined, constraints)
        
        return refined
        
    def _generate_contingencies(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate contingency plans.
        
        Args:
            plan: Mission plan
            constraints: Mission constraints
            
        Returns:
            List of contingency plans
        """
        contingencies = []
        
        # Add abort contingency
        contingencies.append({
            'trigger': 'fuel_below_threshold',
            'threshold': constraints.get('min_fuel', 20),
            'action': 'abort_mission',
            'waypoints': [{'type': 'return_to_base'}]
        })
        
        # Add evasion contingency for high threats
        contingencies.append({
            'trigger': 'new_high_threat_detected',
            'threshold': 0.8,
            'action': 'evasive_maneuvers',
            'waypoints': [{'type': 'evasion_pattern'}]
        })
        
        return contingencies
        
    def _generate_plan_options(self, base_plan: Dict[str, Any], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multiple plan options for decision making.
        
        Args:
            base_plan: Base mission plan
            constraints: Mission constraints
            
        Returns:
            List of plan options with metrics
        """
        options = []
        
        # Option 1: Base plan
        options.append({
            'plan': base_plan,
            'metrics': {
                'effectiveness': 0.8,
                'risk': 0.5,
                'resource_usage': 0.6
            }
        })
        
        # Option 2: High-risk, high-reward plan
        high_risk_plan = base_plan.copy()
        high_risk_plan['objectives'] = [obj for obj in base_plan.get('objectives', []) 
                                       if obj.get('priority', 0) > 0.5]
        options.append({
            'plan': high_risk_plan,
            'metrics': {
                'effectiveness': 0.9,
                'risk': 0.8,
                'resource_usage': 0.7
            }
        })
        
        # Option 3: Conservative plan
        conservative_plan = base_plan.copy()
        conservative_plan['objectives'] = [obj for obj in base_plan.get('objectives', []) 
                                         if obj.get('type') != 'neutralize']
        options.append({
            'plan': conservative_plan,
            'metrics': {
                'effectiveness': 0.6,
                'risk': 0.3,
                'resource_usage': 0.4
            }
        })
        
        return options
        
    def update_mission_status(self, status: str, progress: Dict[str, Any]) -> None:
        """
        Update the status of the current mission.
        
        Args:
            status: New mission status
            progress: Mission progress details
        """
        if self.current_mission:
            self.current_mission['status'] = status
            self.current_mission['progress'] = progress
            self.mission_status = status
            
    def get_current_mission(self) -> Optional[Dict[str, Any]]:
        """
        Get the current mission.
        
        Returns:
            Current mission or None if no mission is active
        """
        return self.current_mission
        
    def get_mission_history(self) -> List[Dict[str, Any]]:
        """
        Get the mission history.
        
        Returns:
            List of past missions
        """
        return self.mission_history