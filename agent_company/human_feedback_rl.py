#!/usr/bin/env python3
"""
Human Feedback Reinforcement Learning for Neuron Tracing
=======================================================
Implements human feedback collection, uncertainty-based intervention,
and human-in-the-loop callbacks for continuous improvement of the tracing agent.
"""

import json
import logging
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading
import queue
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of human feedback."""
    ACCEPT = "accept"
    REJECT = "reject"
    CORRECT = "correct"
    UNCERTAIN = "uncertain"
    SKIP = "skip"

class AutonomyLevel(Enum):
    """Autonomy levels for the tracing agent."""
    FULLY_AUTONOMOUS = "fully_autonomous"  # No human intervention
    SEMI_AUTONOMOUS = "semi_autonomous"    # Human intervention on high uncertainty
    MANUAL_REVIEW = "manual_review"        # Human review for all traces

@dataclass
class TracingFeedback:
    """Structured feedback for a tracing decision."""
    timestamp: str
    region_name: str
    neuron_id: int
    decision_type: str  # "trace_continuation", "branch_detection", "synapse_detection"
    agent_decision: Any
    human_feedback: FeedbackType
    human_correction: Optional[Any] = None
    uncertainty_score: float = 0.0
    confidence_score: float = 0.0
    reasoning: str = ""
    processing_time: float = 0.0

@dataclass
class InterventionPoint:
    """Point where human intervention is requested."""
    timestamp: str
    region_name: str
    neuron_id: int
    decision_type: str
    current_state: Any
    agent_suggestion: Any
    uncertainty_score: float
    confidence_score: float
    context: Dict[str, Any]

class FeedbackCollector:
    """Collect and manage human feedback for reinforcement learning."""
    
    def __init__(self, feedback_dir: str = "feedback_data"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        # Feedback storage
        self.feedback_buffer = deque(maxlen=10000)  # Keep last 10k feedbacks
        self.feedback_stats = defaultdict(int)
        
        # Feedback file
        self.feedback_file = self.feedback_dir / "feedback_log.jsonl"
        self.feedback_lock = threading.Lock()
        
        # Load existing feedback
        self._load_existing_feedback()
        
        logger.info(f"Feedback collector initialized: {self.feedback_dir}")
    
    def _load_existing_feedback(self):
        """Load existing feedback from file."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            feedback_data = json.loads(line)
                            feedback = TracingFeedback(**feedback_data)
                            self.feedback_buffer.append(feedback)
                            self.feedback_stats[feedback.human_feedback.value] += 1
                
                logger.info(f"Loaded {len(self.feedback_buffer)} existing feedback entries")
            except Exception as e:
                logger.error(f"Error loading existing feedback: {e}")
    
    def add_feedback(self, feedback: TracingFeedback) -> bool:
        """Add new feedback to the collection."""
        try:
            with self.feedback_lock:
                # Add to buffer
                self.feedback_buffer.append(feedback)
                
                # Update stats
                self.feedback_stats[feedback.human_feedback.value] += 1
                
                # Save to file
                with open(self.feedback_file, 'a') as f:
                    f.write(json.dumps(asdict(feedback)) + '\n')
                
                logger.info(f"Feedback added: {feedback.human_feedback.value} for neuron {feedback.neuron_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        total_feedback = len(self.feedback_buffer)
        
        stats = {
            'total_feedback': total_feedback,
            'feedback_distribution': dict(self.feedback_stats),
            'recent_feedback': []
        }
        
        # Add recent feedback
        for feedback in list(self.feedback_buffer)[-10:]:
            stats['recent_feedback'].append({
                'timestamp': feedback.timestamp,
                'neuron_id': feedback.neuron_id,
                'feedback_type': feedback.human_feedback.value,
                'uncertainty': feedback.uncertainty_score
            })
        
        return stats
    
    def get_training_data(self, min_feedback_count: int = 100) -> Optional[Dict[str, Any]]:
        """Get feedback data for model training."""
        if len(self.feedback_buffer) < min_feedback_count:
            logger.info(f"Insufficient feedback for training: {len(self.feedback_buffer)} < {min_feedback_count}")
            return None
        
        # Convert feedback to training format
        training_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'corrections': []
        }
        
        for feedback in self.feedback_buffer:
            # Convert feedback to reward
            reward = self._feedback_to_reward(feedback)
            
            training_data['states'].append(feedback.agent_decision)
            training_data['actions'].append(feedback.decision_type)
            training_data['rewards'].append(reward)
            
            if feedback.human_correction is not None:
                training_data['corrections'].append(feedback.human_correction)
        
        logger.info(f"Prepared training data: {len(training_data['states'])} samples")
        return training_data
    
    def _feedback_to_reward(self, feedback: TracingFeedback) -> float:
        """Convert feedback to reward signal."""
        reward_map = {
            FeedbackType.ACCEPT: 1.0,
            FeedbackType.REJECT: -1.0,
            FeedbackType.CORRECT: 0.5,  # Partial reward for corrections
            FeedbackType.UNCERTAIN: 0.0,
            FeedbackType.SKIP: 0.0
        }
        
        base_reward = reward_map[feedback.human_feedback]
        
        # Adjust reward based on uncertainty (higher uncertainty = lower reward)
        uncertainty_penalty = feedback.uncertainty_score * 0.5
        
        return base_reward - uncertainty_penalty

class UncertaintyEstimator:
    """Estimate uncertainty in tracing decisions."""
    
    def __init__(self, uncertainty_threshold: float = 0.7):
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_history = deque(maxlen=1000)
        
        logger.info(f"Uncertainty estimator initialized with threshold: {uncertainty_threshold}")
    
    def estimate_uncertainty(self, model_output: torch.Tensor, 
                           confidence_scores: Optional[torch.Tensor] = None) -> float:
        """Estimate uncertainty from model output."""
        try:
            # Method 1: Entropy-based uncertainty
            if model_output.dim() > 1:
                # Multi-class case
                probs = torch.softmax(model_output, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                uncertainty = torch.mean(entropy).item()
            else:
                # Binary case
                probs = torch.sigmoid(model_output)
                entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
                uncertainty = torch.mean(entropy).item()
            
            # Method 2: Confidence-based uncertainty
            if confidence_scores is not None:
                confidence_uncertainty = 1.0 - torch.mean(confidence_scores).item()
                uncertainty = (uncertainty + confidence_uncertainty) / 2
            
            # Normalize to [0, 1]
            uncertainty = min(1.0, max(0.0, uncertainty))
            
            # Store in history
            self.uncertainty_history.append(uncertainty)
            
            return uncertainty
            
        except Exception as e:
            logger.error(f"Error estimating uncertainty: {e}")
            return 0.5  # Default uncertainty
    
    def should_intervene(self, uncertainty: float) -> bool:
        """Determine if human intervention is needed."""
        return uncertainty > self.uncertainty_threshold
    
    def get_uncertainty_stats(self) -> Dict[str, float]:
        """Get uncertainty statistics."""
        if not self.uncertainty_history:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        
        uncertainties = list(self.uncertainty_history)
        return {
            'mean': np.mean(uncertainties),
            'std': np.std(uncertainties),
            'max': np.max(uncertainties),
            'min': np.min(uncertainties)
        }

class HumanInTheLoopCallback:
    """Callback for human-in-the-loop intervention."""
    
    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.SEMI_AUTONOMOUS):
        self.autonomy_level = autonomy_level
        self.intervention_queue = queue.Queue()
        self.callback_registry = {}
        
        # Register default callbacks
        self._register_default_callbacks()
        
        logger.info(f"Human-in-the-loop callback initialized with autonomy level: {autonomy_level.value}")
    
    def _register_default_callbacks(self):
        """Register default callback functions."""
        self.register_callback("trace_continuation", self._default_trace_callback)
        self.register_callback("branch_detection", self._default_branch_callback)
        self.register_callback("synapse_detection", self._default_synapse_callback)
    
    def register_callback(self, decision_type: str, callback: Callable):
        """Register a callback for a specific decision type."""
        self.callback_registry[decision_type] = callback
        logger.info(f"Registered callback for: {decision_type}")
    
    def request_intervention(self, intervention_point: InterventionPoint) -> Any:
        """Request human intervention for a decision."""
        if self.autonomy_level == AutonomyLevel.FULLY_AUTONOMOUS:
            logger.info("Fully autonomous mode - using agent suggestion")
            return intervention_point.agent_suggestion
        
        # Create intervention request
        intervention_request = {
            'timestamp': intervention_point.timestamp,
            'region_name': intervention_point.region_name,
            'neuron_id': intervention_point.neuron_id,
            'decision_type': intervention_point.decision_type,
            'agent_suggestion': intervention_point.agent_suggestion,
            'uncertainty_score': intervention_point.uncertainty_score,
            'confidence_score': intervention_point.confidence_score,
            'context': intervention_point.context
        }
        
        # Get appropriate callback
        callback = self.callback_registry.get(intervention_point.decision_type, 
                                            self._default_callback)
        
        # Request human input
        try:
            human_decision = callback(intervention_request)
            logger.info(f"Human intervention completed for neuron {intervention_point.neuron_id}")
            return human_decision
        except Exception as e:
            logger.error(f"Error in human intervention: {e}")
            return intervention_point.agent_suggestion
    
    def _default_callback(self, request: Dict[str, Any]) -> Any:
        """Default callback for unregistered decision types."""
        print(f"\n🤖 Human Intervention Required")
        print(f"Neuron ID: {request['neuron_id']}")
        print(f"Decision Type: {request['decision_type']}")
        print(f"Uncertainty: {request['uncertainty_score']:.3f}")
        print(f"Agent Suggestion: {request['agent_suggestion']}")
        
        # Simple CLI interface
        response = input("Accept suggestion? (y/n/c for custom): ").lower().strip()
        
        if response == 'y':
            return request['agent_suggestion']
        elif response == 'n':
            return None
        elif response == 'c':
            custom_input = input("Enter custom decision: ")
            return custom_input
        else:
            return request['agent_suggestion']
    
    def _default_trace_callback(self, request: Dict[str, Any]) -> Any:
        """Default callback for trace continuation decisions."""
        print(f"\n🧠 Trace Continuation Decision")
        print(f"Neuron {request['neuron_id']} - Uncertainty: {request['uncertainty_score']:.3f}")
        print(f"Agent suggests: {request['agent_suggestion']}")
        
        response = input("Continue trace? (y/n/s for skip): ").lower().strip()
        
        if response == 'y':
            return request['agent_suggestion']
        elif response == 'n':
            return None
        elif response == 's':
            return "skip"
        else:
            return request['agent_suggestion']
    
    def _default_branch_callback(self, request: Dict[str, Any]) -> Any:
        """Default callback for branch detection decisions."""
        print(f"\n🌿 Branch Detection Decision")
        print(f"Neuron {request['neuron_id']} - Uncertainty: {request['uncertainty_score']:.3f}")
        print(f"Agent suggests: {request['agent_suggestion']}")
        
        response = input("Accept branch? (y/n/m for manual): ").lower().strip()
        
        if response == 'y':
            return request['agent_suggestion']
        elif response == 'n':
            return None
        elif response == 'm':
            manual_input = input("Enter manual branch coordinates: ")
            return manual_input
        else:
            return request['agent_suggestion']
    
    def _default_synapse_callback(self, request: Dict[str, Any]) -> Any:
        """Default callback for synapse detection decisions."""
        print(f"\n⚡ Synapse Detection Decision")
        print(f"Neuron {request['neuron_id']} - Uncertainty: {request['uncertainty_score']:.3f}")
        print(f"Agent suggests: {request['agent_suggestion']}")
        
        response = input("Accept synapse? (y/n/t for type): ").lower().strip()
        
        if response == 'y':
            return request['agent_suggestion']
        elif response == 'n':
            return None
        elif response == 't':
            synapse_type = input("Enter synapse type (excitatory/inhibitory): ")
            return {'type': synapse_type, 'coordinates': request['agent_suggestion']}
        else:
            return request['agent_suggestion']

class HumanFeedbackRLTracer:
    """Enhanced tracer with human feedback reinforcement learning."""
    
    def __init__(self, 
                 feedback_collector: FeedbackCollector,
                 uncertainty_estimator: UncertaintyEstimator,
                 human_callback: HumanInTheLoopCallback,
                 autonomy_level: AutonomyLevel = AutonomyLevel.SEMI_AUTONOMOUS):
        
        self.feedback_collector = feedback_collector
        self.uncertainty_estimator = uncertainty_estimator
        self.human_callback = human_callback
        self.autonomy_level = autonomy_level
        
        # Tracing state
        self.current_region = None
        self.current_neuron_id = 0
        self.tracing_history = []
        
        logger.info(f"Human Feedback RL Tracer initialized with autonomy level: {autonomy_level.value}")
    
    def trace_with_feedback(self, region_name: str, volume: np.ndarray, 
                          model_output: torch.Tensor) -> Dict[str, Any]:
        """Trace neurons with human feedback integration."""
        self.current_region = region_name
        start_time = time.time()
        
        logger.info(f"Starting tracing with feedback for region: {region_name}")
        
        # Initialize tracing
        traced_neurons = {}
        intervention_count = 0
        
        # Process each potential neuron
        for neuron_id in range(1, int(torch.max(model_output)) + 1):
            self.current_neuron_id = neuron_id
            
            logger.info(f"Processing neuron {neuron_id}")
            
            # Trace individual neuron with feedback
            neuron_result = self._trace_neuron_with_feedback(neuron_id, volume, model_output)
            
            if neuron_result:
                traced_neurons[neuron_id] = neuron_result
                intervention_count += neuron_result.get('interventions', 0)
        
        processing_time = time.time() - start_time
        
        # Generate summary
        summary = {
            'region_name': region_name,
            'total_neurons': len(traced_neurons),
            'intervention_count': intervention_count,
            'processing_time': processing_time,
            'autonomy_level': self.autonomy_level.value,
            'feedback_stats': self.feedback_collector.get_feedback_stats()
        }
        
        logger.info(f"Tracing completed: {len(traced_neurons)} neurons, {intervention_count} interventions")
        
        return {
            'traced_neurons': traced_neurons,
            'summary': summary,
            'tracing_history': self.tracing_history
        }
    
    def _trace_neuron_with_feedback(self, neuron_id: int, volume: np.ndarray, 
                                  model_output: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Trace a single neuron with feedback integration."""
        neuron_start_time = time.time()
        interventions = []
        
        try:
            # Extract neuron mask
            neuron_mask = (model_output == neuron_id).float()
            
            # Initial trace decision
            trace_decision = self._make_trace_decision(neuron_mask, volume, "trace_continuation")
            
            if trace_decision is None:
                return None
            
            # Trace the neuron
            trace_coordinates = []
            current_point = trace_decision['start_point']
            
            while current_point is not None:
                # Add point to trace
                trace_coordinates.append(current_point)
                
                # Make next decision
                next_decision = self._make_trace_decision(neuron_mask, volume, "trace_continuation", 
                                                        current_point)
                
                if next_decision is None or next_decision.get('action') == 'stop':
                    break
                
                current_point = next_decision['next_point']
            
            # Check for branches
            branches = self._detect_branches_with_feedback(trace_coordinates, neuron_mask, volume)
            
            # Check for synapses
            synapses = self._detect_synapses_with_feedback(trace_coordinates, volume)
            
            processing_time = time.time() - neuron_start_time
            
            return {
                'neuron_id': neuron_id,
                'coordinates': trace_coordinates,
                'branches': branches,
                'synapses': synapses,
                'interventions': len(interventions),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error tracing neuron {neuron_id}: {e}")
            return None
    
    def _make_trace_decision(self, neuron_mask: torch.Tensor, volume: np.ndarray, 
                           decision_type: str, current_point: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        """Make a tracing decision with potential human intervention."""
        decision_start_time = time.time()
        
        # Generate agent decision
        agent_decision = self._generate_agent_decision(neuron_mask, volume, decision_type, current_point)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator.estimate_uncertainty(
            agent_decision.get('model_output', torch.tensor([0.5])),
            agent_decision.get('confidence', torch.tensor([0.5]))
        )
        
        # Check if intervention is needed
        if self.uncertainty_estimator.should_intervene(uncertainty):
            # Create intervention point
            intervention_point = InterventionPoint(
                timestamp=datetime.now().isoformat(),
                region_name=self.current_region,
                neuron_id=self.current_neuron_id,
                decision_type=decision_type,
                current_state=current_point,
                agent_suggestion=agent_decision,
                uncertainty_score=uncertainty,
                confidence_score=agent_decision.get('confidence', 0.5),
                context={'volume_shape': volume.shape, 'mask_shape': neuron_mask.shape}
            )
            
            # Request human intervention
            human_decision = self.human_callback.request_intervention(intervention_point)
            
            # Record feedback
            feedback = TracingFeedback(
                timestamp=datetime.now().isoformat(),
                region_name=self.current_region,
                neuron_id=self.current_neuron_id,
                decision_type=decision_type,
                agent_decision=agent_decision,
                human_feedback=FeedbackType.CORRECT if human_decision != agent_decision else FeedbackType.ACCEPT,
                human_correction=human_decision if human_decision != agent_decision else None,
                uncertainty_score=uncertainty,
                confidence_score=agent_decision.get('confidence', 0.5),
                processing_time=time.time() - decision_start_time
            )
            
            self.feedback_collector.add_feedback(feedback)
            
            return human_decision
        else:
            # Use agent decision directly
            feedback = TracingFeedback(
                timestamp=datetime.now().isoformat(),
                region_name=self.current_region,
                neuron_id=self.current_neuron_id,
                decision_type=decision_type,
                agent_decision=agent_decision,
                human_feedback=FeedbackType.ACCEPT,
                uncertainty_score=uncertainty,
                confidence_score=agent_decision.get('confidence', 0.5),
                processing_time=time.time() - decision_start_time
            )
            
            self.feedback_collector.add_feedback(feedback)
            
            return agent_decision
    
    def _generate_agent_decision(self, neuron_mask: torch.Tensor, volume: np.ndarray, 
                               decision_type: str, current_point: Optional[Tuple]) -> Dict[str, Any]:
        """Generate agent decision for tracing."""
        # Simplified decision logic - in practice, this would use your trained model
        if decision_type == "trace_continuation":
            if current_point is None:
                # Find starting point
                start_point = self._find_starting_point(neuron_mask)
                return {
                    'action': 'start',
                    'start_point': start_point,
                    'confidence': 0.8,
                    'model_output': torch.tensor([0.8])
                }
            else:
                # Find next point
                next_point = self._find_next_point(current_point, neuron_mask)
                if next_point is None:
                    return {
                        'action': 'stop',
                        'confidence': 0.9,
                        'model_output': torch.tensor([0.9])
                    }
                else:
                    return {
                        'action': 'continue',
                        'next_point': next_point,
                        'confidence': 0.7,
                        'model_output': torch.tensor([0.7])
                    }
        
        return {
            'action': 'unknown',
            'confidence': 0.5,
            'model_output': torch.tensor([0.5])
        }
    
    def _find_starting_point(self, neuron_mask: torch.Tensor) -> Tuple[int, int, int]:
        """Find starting point for neuron tracing."""
        # Find the center of mass
        coords = torch.nonzero(neuron_mask)
        if len(coords) > 0:
            center = torch.mean(coords.float(), dim=0)
            return tuple(center.long().tolist())
        return (0, 0, 0)
    
    def _find_next_point(self, current_point: Tuple, neuron_mask: torch.Tensor) -> Optional[Tuple[int, int, int]]:
        """Find next point in the trace."""
        # Simple next point detection - in practice, use your trained model
        z, y, x = current_point
        
        # Check neighboring points
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == dy == dx == 0:
                        continue
                    
                    new_z, new_y, new_x = z + dz, y + dy, x + dx
                    
                    if (0 <= new_z < neuron_mask.shape[0] and 
                        0 <= new_y < neuron_mask.shape[1] and 
                        0 <= new_x < neuron_mask.shape[2]):
                        
                        if neuron_mask[new_z, new_y, new_x] > 0:
                            return (new_z, new_y, new_x)
        
        return None
    
    def _detect_branches_with_feedback(self, trace_coordinates: List[Tuple], 
                                     neuron_mask: torch.Tensor, volume: np.ndarray) -> List[Dict]:
        """Detect branches with human feedback."""
        branches = []
        
        for i, point in enumerate(trace_coordinates):
            # Check for branching at this point
            branch_decision = self._make_trace_decision(neuron_mask, volume, "branch_detection", point)
            
            if branch_decision and branch_decision.get('action') == 'branch':
                branches.append({
                    'point': point,
                    'branch_coordinates': branch_decision.get('branch_coordinates', []),
                    'confidence': branch_decision.get('confidence', 0.5)
                })
        
        return branches
    
    def _detect_synapses_with_feedback(self, trace_coordinates: List[Tuple], 
                                     volume: np.ndarray) -> List[Dict]:
        """Detect synapses with human feedback."""
        synapses = []
        
        for point in trace_coordinates:
            # Check for synapse at this point
            synapse_decision = self._make_trace_decision(volume, volume, "synapse_detection", point)
            
            if synapse_decision and synapse_decision.get('action') == 'synapse':
                synapses.append({
                    'point': point,
                    'synapse_type': synapse_decision.get('synapse_type', 'unknown'),
                    'confidence': synapse_decision.get('confidence', 0.5)
                })
        
        return synapses
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback and learning progress."""
        feedback_stats = self.feedback_collector.get_feedback_stats()
        uncertainty_stats = self.uncertainty_estimator.get_uncertainty_stats()
        
        return {
            'feedback_stats': feedback_stats,
            'uncertainty_stats': uncertainty_stats,
            'autonomy_level': self.autonomy_level.value,
            'total_traced_neurons': len(self.tracing_history)
        }

def main():
    """Demo of human feedback RL tracing system."""
    print("Human Feedback RL Tracing System Demo")
    print("=" * 50)
    
    # Initialize components
    feedback_collector = FeedbackCollector()
    uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.6)
    human_callback = HumanInTheLoopCallback(AutonomyLevel.SEMI_AUTONOMOUS)
    
    # Initialize tracer
    tracer = HumanFeedbackRLTracer(
        feedback_collector=feedback_collector,
        uncertainty_estimator=uncertainty_estimator,
        human_callback=human_callback,
        autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS
    )
    
    print("System initialized successfully!")
    print(f"Autonomy level: {tracer.autonomy_level.value}")
    print(f"Uncertainty threshold: {uncertainty_estimator.uncertainty_threshold}")
    
    # Demo with synthetic data
    print("\nRunning demo with synthetic data...")
    
    # Create synthetic volume and model output
    volume = np.random.randint(0, 255, (64, 64, 64))
    model_output = torch.randint(0, 5, (64, 64, 64))
    
    # Run tracing with feedback
    results = tracer.trace_with_feedback("demo_region", volume, model_output)
    
    # Print results
    print(f"\nTracing Results:")
    print(f"  Neurons traced: {results['summary']['total_neurons']}")
    print(f"  Human interventions: {results['summary']['intervention_count']}")
    print(f"  Processing time: {results['summary']['processing_time']:.2f}s")
    
    # Print feedback summary
    feedback_summary = tracer.get_feedback_summary()
    print(f"\nFeedback Summary:")
    print(f"  Total feedback: {feedback_summary['feedback_stats']['total_feedback']}")
    print(f"  Feedback distribution: {feedback_summary['feedback_stats']['feedback_distribution']}")
    print(f"  Average uncertainty: {feedback_summary['uncertainty_stats']['mean']:.3f}")
    
    print("\nDemo completed! Check 'feedback_data/' for detailed logs.")

if __name__ == "__main__":
    main() 