#!/usr/bin/env python3
"""
Integrate Feedback System with Existing Tracing Pipeline
=======================================================
Shows how to add human feedback RL to your existing neuron tracing pipeline.
"""

import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime

# Import the feedback system
from human_feedback_rl import (
    FeedbackCollector,
    UncertaintyEstimator, 
    HumanInTheLoopCallback,
    AutonomyLevel,
    FeedbackType,
    TracingFeedback
)

def integrate_with_existing_pipeline():
    """Show how to integrate feedback system with existing pipeline."""
    print("Integrating Feedback System with Tracing Pipeline")
    print("=" * 60)
    
    # 1. Initialize feedback components
    print("1. Initializing feedback components...")
    
    feedback_collector = FeedbackCollector("production_feedback")
    uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.7)
    human_callback = HumanInTheLoopCallback(AutonomyLevel.SEMI_AUTONOMOUS)
    
    print("✅ Feedback components initialized")
    
    # 2. Example: Integrate with your existing segmentation
    print("\n2. Example: Enhanced segmentation with feedback...")
    
    def enhanced_segment_with_feedback(volume, model_output, region_name="unknown"):
        """Enhanced segmentation that collects feedback on uncertain regions."""
        
        # Your existing segmentation logic here
        segmentation = model_output.clone()
        
        # Add uncertainty estimation
        uncertainty_map = torch.zeros_like(model_output, dtype=torch.float32)
        
        # Simulate uncertainty estimation for each neuron
        for neuron_id in range(1, int(torch.max(model_output)) + 1):
            neuron_mask = (model_output == neuron_id)
            
            # Estimate uncertainty (in practice, use your model's uncertainty)
            uncertainty = uncertainty_estimator.estimate_uncertainty(
                torch.tensor([0.6 + 0.3 * np.random.random()])  # Simulated uncertainty
            )
            
            uncertainty_map[neuron_mask] = uncertainty
            
            # Check if human intervention is needed
            if uncertainty_estimator.should_intervene(uncertainty):
                print(f"🤖 Human intervention needed for neuron {neuron_id} (uncertainty: {uncertainty:.3f})")
                
                # Create intervention request
                intervention_data = {
                    'neuron_id': neuron_id,
                    'uncertainty': uncertainty,
                    'region_name': region_name,
                    'volume_shape': volume.shape,
                    'neuron_size': torch.sum(neuron_mask).item()
                }
                
                # In real usage, this would prompt for human input
                human_decision = human_callback._default_callback(intervention_data)
                
                # Record feedback
                feedback = TracingFeedback(
                    timestamp=datetime.now().isoformat(),
                    region_name=region_name,
                    neuron_id=neuron_id,
                    decision_type="segmentation",
                    agent_decision={"segmentation": neuron_mask.numpy().tolist()},
                    human_feedback=FeedbackType.ACCEPT if human_decision else FeedbackType.REJECT,
                    human_correction=human_decision if human_decision else None,
                    uncertainty_score=uncertainty,
                    confidence_score=1.0 - uncertainty,
                    reasoning="Uncertainty-based intervention"
                )
                
                feedback_collector.add_feedback(feedback)
        
        return segmentation, uncertainty_map
    
    # 3. Example: Enhanced tracing with feedback
    print("\n3. Example: Enhanced tracing with feedback...")
    
    def enhanced_trace_with_feedback(segmentation, volume, region_name="unknown"):
        """Enhanced tracing that collects feedback on trace decisions."""
        
        traced_neurons = {}
        
        for neuron_id in range(1, int(torch.max(segmentation)) + 1):
            neuron_mask = (segmentation == neuron_id)
            
            # Your existing tracing logic here
            trace_coordinates = [(z.item(), y.item(), x.item()) 
                               for z, y, x in torch.nonzero(neuron_mask)[:10]]  # First 10 points
            
            # Check for uncertain trace decisions
            for i, point in enumerate(trace_coordinates):
                # Simulate uncertainty in trace continuation
                if i > 0 and np.random.random() < 0.3:  # 30% chance of uncertainty
                    uncertainty = uncertainty_estimator.estimate_uncertainty(
                        torch.tensor([0.5 + 0.4 * np.random.random()])
                    )
                    
                    if uncertainty_estimator.should_intervene(uncertainty):
                        print(f"🤖 Trace decision intervention for neuron {neuron_id} at point {i}")
                        
                        # Record feedback for trace decision
                        feedback = TracingFeedback(
                            timestamp=datetime.now().isoformat(),
                            region_name=region_name,
                            neuron_id=neuron_id,
                            decision_type="trace_continuation",
                            agent_decision={"next_point": point},
                            human_feedback=FeedbackType.ACCEPT,
                            uncertainty_score=uncertainty,
                            confidence_score=1.0 - uncertainty,
                            reasoning="Trace continuation decision"
                        )
                        
                        feedback_collector.add_feedback(feedback)
            
            traced_neurons[neuron_id] = {
                'coordinates': trace_coordinates,
                'length': len(trace_coordinates)
            }
        
        return traced_neurons
    
    # 4. Example: Enhanced branch detection with feedback
    print("\n4. Example: Enhanced branch detection with feedback...")
    
    def enhanced_branch_detection_with_feedback(traced_neurons, volume, region_name="unknown"):
        """Enhanced branch detection that collects feedback."""
        
        for neuron_id, neuron_data in traced_neurons.items():
            coordinates = neuron_data['coordinates']
            
            # Simulate branch detection
            if len(coordinates) > 5:
                # Check for potential branches
                uncertainty = uncertainty_estimator.estimate_uncertainty(
                    torch.tensor([0.6 + 0.3 * np.random.random()])
                )
                
                if uncertainty_estimator.should_intervene(uncertainty):
                    print(f"🤖 Branch detection intervention for neuron {neuron_id}")
                    
                    # Record feedback for branch detection
                    feedback = TracingFeedback(
                        timestamp=datetime.now().isoformat(),
                        region_name=region_name,
                        neuron_id=neuron_id,
                        decision_type="branch_detection",
                        agent_decision={"potential_branches": coordinates[5:8]},
                        human_feedback=FeedbackType.CORRECT,
                        human_correction={"confirmed_branches": coordinates[5:7]},
                        uncertainty_score=uncertainty,
                        confidence_score=1.0 - uncertainty,
                        reasoning="Branch detection decision"
                    )
                    
                    feedback_collector.add_feedback(feedback)
    
    # 5. Run the enhanced pipeline
    print("\n5. Running enhanced pipeline...")
    
    # Create sample data
    volume = np.random.randint(0, 255, (64, 64, 64), dtype=np.uint8)
    model_output = torch.randint(0, 4, (64, 64, 64))
    
    # Run enhanced pipeline
    segmentation, uncertainty_map = enhanced_segment_with_feedback(volume, model_output, "test_region")
    traced_neurons = enhanced_trace_with_feedback(segmentation, volume, "test_region")
    enhanced_branch_detection_with_feedback(traced_neurons, volume, "test_region")
    
    # 6. Get feedback summary
    print("\n6. Feedback Summary:")
    feedback_stats = feedback_collector.get_feedback_stats()
    uncertainty_stats = uncertainty_estimator.get_uncertainty_stats()
    
    print(f"Total feedback collected: {feedback_stats['total_feedback']}")
    print(f"Feedback distribution: {feedback_stats['feedback_distribution']}")
    print(f"Average uncertainty: {uncertainty_stats['mean']:.3f}")
    
    return {
        'feedback_collector': feedback_collector,
        'uncertainty_estimator': uncertainty_estimator,
        'human_callback': human_callback,
        'feedback_stats': feedback_stats,
        'uncertainty_stats': uncertainty_stats
    }

def show_integration_examples():
    """Show practical integration examples."""
    print("\n" + "="*60)
    print("INTEGRATION EXAMPLES")
    print("="*60)
    
    # Example 1: Add to your existing production_pipeline.py
    print("\n1. Add to production_pipeline.py:")
    print("""
# At the top of your file:
from human_feedback_rl import (
    FeedbackCollector, UncertaintyEstimator, HumanInTheLoopCallback,
    AutonomyLevel, FeedbackType, TracingFeedback
)

# Initialize in your main function:
feedback_collector = FeedbackCollector("h01_feedback")
uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.7)
human_callback = HumanInTheLoopCallback(AutonomyLevel.SEMI_AUTONOMOUS)

# Add to your processing loop:
for region_name, volume in regions.items():
    # Your existing processing...
    segmentation = segment_volume(volume)
    
    # Add uncertainty check
    for neuron_id in range(1, max_neuron_id + 1):
        uncertainty = uncertainty_estimator.estimate_uncertainty(model_output)
        if uncertainty_estimator.should_intervene(uncertainty):
            # Request human intervention
            human_decision = human_callback.request_intervention(...)
            # Record feedback
            feedback_collector.add_feedback(...)
    """)
    
    # Example 2: Add to your batch processor
    print("\n2. Add to production_batch_processor.py:")
    print("""
# In your batch processing function:
def process_batch_with_feedback(batch_data):
    feedback_collector = FeedbackCollector("batch_feedback")
    
    for item in batch_data:
        # Your existing processing...
        result = process_single_item(item)
        
        # Add feedback collection
        if result['uncertainty'] > threshold:
            feedback = TracingFeedback(...)
            feedback_collector.add_feedback(feedback)
    
    # Get training data
    training_data = feedback_collector.get_training_data(min_feedback_count=100)
    if training_data:
        update_model(training_data)
    """)
    
    # Example 3: Custom callback for your specific needs
    print("\n3. Custom callback example:")
    print("""
# Create custom callback for your specific interface:
def custom_trace_callback(request):
    # Your custom UI or interface
    print(f"Neuron {request['neuron_id']} needs review")
    print(f"Uncertainty: {request['uncertainty_score']:.3f}")
    
    # Your custom input method
    decision = your_custom_interface.get_decision()
    return decision

# Register the callback:
human_callback.register_callback("trace_continuation", custom_trace_callback)
    """)

def main():
    """Run the integration demonstration."""
    print("Feedback System Integration Demo")
    print("=" * 80)
    
    try:
        # Run integration demo
        results = integrate_with_existing_pipeline()
        
        # Show examples
        show_integration_examples()
        
        print("\n" + "="*80)
        print("INTEGRATION DEMO COMPLETED!")
        print("="*80)
        
        print("\nKey Integration Points:")
        print("✅ Segmentation with uncertainty estimation")
        print("✅ Tracing with human intervention")
        print("✅ Branch detection with feedback")
        print("✅ Feedback collection and logging")
        print("✅ Training data preparation")
        
        print("\nNext Steps for Your Pipeline:")
        print("1. Add feedback components to your main pipeline")
        print("2. Integrate uncertainty estimation with your model")
        print("3. Add human intervention points at key decision moments")
        print("4. Set up feedback collection in your batch processor")
        print("5. Create custom callbacks for your specific interface")
        print("6. Monitor and analyze feedback patterns")
        
    except Exception as e:
        print(f"Error in integration demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 