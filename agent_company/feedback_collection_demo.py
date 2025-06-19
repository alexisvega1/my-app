#!/usr/bin/env python3
"""
Feedback Collection and Human-in-the-Loop Demo
=============================================
Demonstrates the feedback collection system and uncertainty-based intervention
with practical examples for neuron tracing.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime

# Import our feedback system
from human_feedback_rl import (
    FeedbackCollector, 
    UncertaintyEstimator, 
    HumanInTheLoopCallback,
    HumanFeedbackRLTracer,
    AutonomyLevel,
    FeedbackType,
    TracingFeedback,
    InterventionPoint
)

def create_demo_data():
    """Create realistic demo data for tracing."""
    print("Creating demo data...")
    
    # Create a volume with synthetic neurons
    volume_shape = (128, 128, 128)
    volume = np.random.randint(0, 255, volume_shape, dtype=np.uint8)
    
    # Create model output with 3 neurons
    model_output = np.zeros(volume_shape, dtype=np.int32)
    
    # Neuron 1: Simple linear trace
    for i in range(20, 80):
        model_output[20:30, i, 20:30] = 1
    
    # Neuron 2: Branched trace
    for i in range(30, 70):
        model_output[50:60, i, 50:60] = 2
    for i in range(40, 60):
        model_output[60:70, 60, i] = 2
    
    # Neuron 3: Complex trace with uncertainty
    for i in range(40, 90):
        model_output[80:90, i, 80:90] = 3
        if 60 <= i <= 70:  # Add uncertainty region
            model_output[85:95, i, 85:95] = 3
    
    return volume, torch.from_numpy(model_output)

def demo_feedback_collection():
    """Demonstrate feedback collection system."""
    print("\n" + "="*60)
    print("FEEDBACK COLLECTION DEMO")
    print("="*60)
    
    # Initialize feedback collector
    feedback_collector = FeedbackCollector("demo_feedback")
    
    # Create some example feedback
    feedback_examples = [
        TracingFeedback(
            timestamp=datetime.now().isoformat(),
            region_name="demo_region",
            neuron_id=1,
            decision_type="trace_continuation",
            agent_decision={"action": "continue", "next_point": (25, 45, 25)},
            human_feedback=FeedbackType.ACCEPT,
            uncertainty_score=0.2,
            confidence_score=0.8,
            reasoning="Clear continuation path",
            processing_time=0.15
        ),
        TracingFeedback(
            timestamp=datetime.now().isoformat(),
            region_name="demo_region",
            neuron_id=2,
            decision_type="branch_detection",
            agent_decision={"action": "branch", "branch_coordinates": [(55, 60, 55)]},
            human_feedback=FeedbackType.CORRECT,
            human_correction={"action": "branch", "branch_coordinates": [(55, 60, 55), (55, 60, 56)]},
            uncertainty_score=0.7,
            confidence_score=0.6,
            reasoning="Agent missed secondary branch",
            processing_time=0.25
        ),
        TracingFeedback(
            timestamp=datetime.now().isoformat(),
            region_name="demo_region",
            neuron_id=3,
            decision_type="synapse_detection",
            agent_decision={"action": "synapse", "synapse_type": "excitatory"},
            human_feedback=FeedbackType.REJECT,
            uncertainty_score=0.8,
            confidence_score=0.4,
            reasoning="False positive - no synaptic density",
            processing_time=0.18
        )
    ]
    
    # Add feedback
    for feedback in feedback_examples:
        success = feedback_collector.add_feedback(feedback)
        print(f"Added feedback: {feedback.human_feedback.value} for neuron {feedback.neuron_id} - {'Success' if success else 'Failed'}")
    
    # Get feedback statistics
    stats = feedback_collector.get_feedback_stats()
    print(f"\nFeedback Statistics:")
    print(f"  Total feedback: {stats['total_feedback']}")
    print(f"  Distribution: {stats['feedback_distribution']}")
    
    # Show recent feedback
    print(f"\nRecent Feedback:")
    for feedback in stats['recent_feedback']:
        print(f"  Neuron {feedback['neuron_id']}: {feedback['feedback_type']} (uncertainty: {feedback['uncertainty']:.2f})")

def demo_uncertainty_estimation():
    """Demonstrate uncertainty estimation."""
    print("\n" + "="*60)
    print("UNCERTAINTY ESTIMATION DEMO")
    print("="*60)
    
    # Initialize uncertainty estimator
    uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.6)
    
    # Create example model outputs with different uncertainty levels
    examples = [
        ("High confidence", torch.tensor([[0.9, 0.1]])),  # Low uncertainty
        ("Medium confidence", torch.tensor([[0.6, 0.4]])),  # Medium uncertainty
        ("Low confidence", torch.tensor([[0.51, 0.49]])),  # High uncertainty
        ("Binary high confidence", torch.tensor([0.95])),  # Binary, low uncertainty
        ("Binary low confidence", torch.tensor([0.52])),  # Binary, high uncertainty
    ]
    
    for name, model_output in examples:
        uncertainty = uncertainty_estimator.estimate_uncertainty(model_output)
        should_intervene = uncertainty_estimator.should_intervene(uncertainty)
        
        print(f"{name}:")
        print(f"  Uncertainty: {uncertainty:.3f}")
        print(f"  Should intervene: {should_intervene}")
        print()
    
    # Get uncertainty statistics
    stats = uncertainty_estimator.get_uncertainty_stats()
    print(f"Uncertainty Statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")

def demo_human_callback():
    """Demonstrate human-in-the-loop callback system."""
    print("\n" + "="*60)
    print("HUMAN-IN-THE-LOOP CALLBACK DEMO")
    print("="*60)
    
    # Initialize callback system
    human_callback = HumanInTheLoopCallback(AutonomyLevel.SEMI_AUTONOMOUS)
    
    # Create example intervention points
    intervention_examples = [
        InterventionPoint(
            timestamp=datetime.now().isoformat(),
            region_name="demo_region",
            neuron_id=1,
            decision_type="trace_continuation",
            current_state=(25, 45, 25),
            agent_suggestion={"action": "continue", "next_point": (25, 46, 25)},
            uncertainty_score=0.75,
            confidence_score=0.6,
            context={"volume_shape": (128, 128, 128)}
        ),
        InterventionPoint(
            timestamp=datetime.now().isoformat(),
            region_name="demo_region",
            neuron_id=2,
            decision_type="branch_detection",
            current_state=(55, 60, 55),
            agent_suggestion={"action": "branch", "branch_coordinates": [(55, 60, 56)]},
            uncertainty_score=0.8,
            confidence_score=0.5,
            context={"volume_shape": (128, 128, 128)}
        ),
        InterventionPoint(
            timestamp=datetime.now().isoformat(),
            region_name="demo_region",
            neuron_id=3,
            decision_type="synapse_detection",
            current_state=(85, 65, 85),
            agent_suggestion={"action": "synapse", "synapse_type": "excitatory"},
            uncertainty_score=0.9,
            confidence_score=0.3,
            context={"volume_shape": (128, 128, 128)}
        )
    ]
    
    print("Simulating human interventions...")
    print("(In real usage, these would prompt for actual human input)")
    
    for i, intervention in enumerate(intervention_examples, 1):
        print(f"\nIntervention {i}:")
        print(f"  Neuron {intervention.neuron_id} - {intervention.decision_type}")
        print(f"  Uncertainty: {intervention.uncertainty_score:.3f}")
        print(f"  Agent suggestion: {intervention.agent_suggestion}")
        
        # Simulate human decision (in real usage, this would wait for input)
        if intervention.decision_type == "trace_continuation":
            human_decision = intervention.agent_suggestion  # Accept
        elif intervention.decision_type == "branch_detection":
            human_decision = {"action": "branch", "branch_coordinates": [(55, 60, 56), (55, 60, 57)]}  # Modify
        else:  # synapse_detection
            human_decision = None  # Reject
        
        print(f"  Human decision: {human_decision}")

def demo_full_tracing_pipeline():
    """Demonstrate the full tracing pipeline with feedback."""
    print("\n" + "="*60)
    print("FULL TRACING PIPELINE DEMO")
    print("="*60)
    
    # Create demo data
    volume, model_output = create_demo_data()
    
    # Initialize all components
    feedback_collector = FeedbackCollector("demo_pipeline_feedback")
    uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.6)
    human_callback = HumanInTheLoopCallback(AutonomyLevel.SEMI_AUTONOMOUS)
    
    # Initialize tracer
    tracer = HumanFeedbackRLTracer(
        feedback_collector=feedback_collector,
        uncertainty_estimator=uncertainty_estimator,
        human_callback=human_callback,
        autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS
    )
    
    print("Running tracing pipeline with feedback...")
    print("(This will simulate the full pipeline without actual human input)")
    
    # Run tracing
    start_time = time.time()
    results = tracer.trace_with_feedback("demo_region", volume, model_output)
    processing_time = time.time() - start_time
    
    # Print results
    print(f"\nTracing Results:")
    print(f"  Neurons traced: {results['summary']['total_neurons']}")
    print(f"  Human interventions: {results['summary']['intervention_count']}")
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Autonomy level: {results['summary']['autonomy_level']}")
    
    # Print feedback summary
    feedback_summary = tracer.get_feedback_summary()
    print(f"\nFeedback Summary:")
    print(f"  Total feedback: {feedback_summary['feedback_stats']['total_feedback']}")
    print(f"  Feedback distribution: {feedback_summary['feedback_stats']['feedback_distribution']}")
    print(f"  Average uncertainty: {feedback_summary['uncertainty_stats']['mean']:.3f}")
    
    # Show traced neurons
    print(f"\nTraced Neurons:")
    for neuron_id, neuron_data in results['traced_neurons'].items():
        print(f"  Neuron {neuron_id}:")
        print(f"    Coordinates: {len(neuron_data['coordinates'])} points")
        print(f"    Branches: {len(neuron_data['branches'])}")
        print(f"    Synapses: {len(neuron_data['synapses'])}")
        print(f"    Interventions: {neuron_data['interventions']}")

def create_visualization_report():
    """Create a visualization report of the feedback system."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATION REPORT")
    print("="*60)
    
    # Load feedback data
    feedback_collector = FeedbackCollector("demo_feedback")
    stats = feedback_collector.get_feedback_stats()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Human Feedback RL System - Demo Report', fontsize=16)
    
    # 1. Feedback distribution
    if stats['feedback_distribution']:
        feedback_types = list(stats['feedback_distribution'].keys())
        feedback_counts = list(stats['feedback_distribution'].values())
        
        axes[0, 0].bar(feedback_types, feedback_counts, color=['green', 'red', 'blue', 'orange', 'gray'])
        axes[0, 0].set_title('Feedback Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Uncertainty over time (simulated)
    uncertainty_estimator = UncertaintyEstimator()
    uncertainties = [0.3, 0.5, 0.7, 0.4, 0.8, 0.6, 0.2, 0.9, 0.5, 0.3]
    timestamps = range(len(uncertainties))
    
    axes[0, 1].plot(timestamps, uncertainties, 'b-o', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=uncertainty_estimator.uncertainty_threshold, color='r', linestyle='--', 
                       label=f'Threshold ({uncertainty_estimator.uncertainty_threshold})')
    axes[0, 1].set_title('Uncertainty Over Time')
    axes[0, 1].set_ylabel('Uncertainty Score')
    axes[0, 1].set_xlabel('Decision Number')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Intervention frequency by decision type
    decision_types = ['trace_continuation', 'branch_detection', 'synapse_detection']
    intervention_counts = [15, 8, 12]  # Simulated data
    
    axes[1, 0].bar(decision_types, intervention_counts, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1, 0].set_title('Interventions by Decision Type')
    axes[1, 0].set_ylabel('Intervention Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Learning progress (simulated)
    epochs = range(1, 11)
    accuracy = [0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.87]
    uncertainty = [0.8, 0.75, 0.7, 0.68, 0.65, 0.62, 0.6, 0.58, 0.55, 0.52]
    
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, accuracy, 'g-o', linewidth=2, markersize=6, label='Accuracy')
    line2 = ax2.plot(epochs, uncertainty, 'r-s', linewidth=2, markersize=6, label='Uncertainty')
    
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Accuracy', color='g')
    ax2.set_ylabel('Uncertainty', color='r')
    ax1.set_title('Learning Progress')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    # Save the report
    report_path = Path("feedback_system_report.png")
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"Visualization report saved to: {report_path}")
    
    plt.show()

def main():
    """Run the complete feedback collection demo."""
    print("Human Feedback RL System - Complete Demo")
    print("=" * 80)
    
    try:
        # Run all demos
        demo_feedback_collection()
        demo_uncertainty_estimation()
        demo_human_callback()
        demo_full_tracing_pipeline()
        create_visualization_report()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Features Demonstrated:")
        print("✅ Feedback collection and logging")
        print("✅ Uncertainty estimation and intervention")
        print("✅ Human-in-the-loop callbacks")
        print("✅ Full tracing pipeline integration")
        print("✅ Visualization and reporting")
        
        print("\nNext Steps:")
        print("1. Integrate with your actual tracing pipeline")
        print("2. Customize callback interfaces for your use case")
        print("3. Adjust uncertainty thresholds based on your data")
        print("4. Deploy with real human annotators")
        print("5. Monitor learning progress over time")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 