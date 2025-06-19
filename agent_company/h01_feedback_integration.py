#!/usr/bin/env python3
"""
H01 Feedback Integration - Practical Example
===========================================
Shows how to integrate human feedback RL with your actual H01 data processing pipeline.
"""

import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import time

# Import your existing pipeline components
from production_pipeline import load_h01_region, segment_volume
from production_batch_processor import process_region_batch
from human_feedback_rl import (
    FeedbackCollector,
    UncertaintyEstimator, 
    HumanInTheLoopCallback,
    AutonomyLevel,
    FeedbackType,
    TracingFeedback,
    InterventionPoint
)

class H01FeedbackProcessor:
    """Enhanced H01 processor with human feedback integration."""
    
    def __init__(self, feedback_dir="h01_feedback_data", uncertainty_threshold=0.7):
        self.feedback_collector = FeedbackCollector(feedback_dir)
        self.uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold)
        self.human_callback = HumanInTheLoopCallback(AutonomyLevel.SEMI_AUTONOMOUS)
        
        # Register custom callbacks for H01 data
        self._register_h01_callbacks()
        
        print(f"✅ H01 Feedback Processor initialized")
        print(f"   Feedback directory: {feedback_dir}")
        print(f"   Uncertainty threshold: {uncertainty_threshold}")
        print(f"   Autonomy level: {AutonomyLevel.SEMI_AUTONOMOUS.value}")
    
    def _register_h01_callbacks(self):
        """Register custom callbacks for H01-specific decisions."""
        
        def h01_trace_callback(request):
            """Custom callback for H01 trace continuation."""
            print(f"\n🧠 H01 Trace Decision")
            print(f"Region: {request['region_name']}")
            print(f"Neuron: {request['neuron_id']}")
            print(f"Uncertainty: {request['uncertainty_score']:.3f}")
            print(f"Volume shape: {request['context']['volume_shape']}")
            print(f"Agent suggests: {request['agent_suggestion']}")
            
            # In real usage, this would show the actual volume data
            # and wait for human input through your interface
            
            response = input("Accept trace? (y/n/s for skip): ").lower().strip()
            
            if response == 'y':
                return request['agent_suggestion']
            elif response == 'n':
                return None
            elif response == 's':
                return "skip"
            else:
                return request['agent_suggestion']
        
        def h01_branch_callback(request):
            """Custom callback for H01 branch detection."""
            print(f"\n🌿 H01 Branch Detection")
            print(f"Region: {request['region_name']}")
            print(f"Neuron: {request['neuron_id']}")
            print(f"Uncertainty: {request['uncertainty_score']:.3f}")
            print(f"Agent suggests: {request['agent_suggestion']}")
            
            response = input("Accept branch? (y/n/m for manual): ").lower().strip()
            
            if response == 'y':
                return request['agent_suggestion']
            elif response == 'n':
                return None
            elif response == 'm':
                manual_input = input("Enter manual branch coordinates (z,y,x): ")
                try:
                    coords = tuple(map(int, manual_input.split(',')))
                    return {"action": "branch", "branch_coordinates": [coords]}
                except:
                    return request['agent_suggestion']
            else:
                return request['agent_suggestion']
        
        def h01_synapse_callback(request):
            """Custom callback for H01 synapse detection."""
            print(f"\n⚡ H01 Synapse Detection")
            print(f"Region: {request['region_name']}")
            print(f"Neuron: {request['neuron_id']}")
            print(f"Uncertainty: {request['uncertainty_score']:.3f}")
            print(f"Agent suggests: {request['agent_suggestion']}")
            
            response = input("Accept synapse? (y/n/t for type): ").lower().strip()
            
            if response == 'y':
                return request['agent_suggestion']
            elif response == 'n':
                return None
            elif response == 't':
                synapse_type = input("Enter synapse type (excitatory/inhibitory): ")
                return {"action": "synapse", "synapse_type": synapse_type}
            else:
                return request['agent_suggestion']
        
        # Register the callbacks
        self.human_callback.register_callback("trace_continuation", h01_trace_callback)
        self.human_callback.register_callback("branch_detection", h01_branch_callback)
        self.human_callback.register_callback("synapse_detection", h01_synapse_callback)
    
    def process_h01_region_with_feedback(self, region_name, region_coords, region_size):
        """Process H01 region with human feedback integration."""
        print(f"\n🔬 Processing H01 region: {region_name}")
        print(f"Coordinates: {region_coords}")
        print(f"Size: {region_size}")
        
        start_time = time.time()
        
        try:
            # Load H01 region data
            print("Loading H01 region data...")
            volume = load_h01_region(region_coords, region_size)
            
            if volume is None or volume.size == 0:
                print("❌ No data found for this region")
                return None
            
            print(f"✅ Loaded volume: {volume.shape}, {volume.dtype}")
            
            # Segment with uncertainty estimation
            print("Segmenting with uncertainty estimation...")
            segmentation, uncertainty_map = self._segment_with_feedback(volume, region_name)
            
            if segmentation is None:
                print("❌ Segmentation failed")
                return None
            
            print(f"✅ Segmentation complete: {torch.max(segmentation)} neurons")
            
            # Trace with feedback
            print("Tracing with human feedback...")
            traced_neurons = self._trace_with_feedback(segmentation, volume, region_name)
            
            # Detect branches and synapses
            print("Detecting branches and synapses...")
            self._detect_features_with_feedback(traced_neurons, volume, region_name)
            
            processing_time = time.time() - start_time
            
            # Compile results
            results = {
                'region_name': region_name,
                'region_coords': region_coords,
                'region_size': region_size,
                'volume_shape': volume.shape,
                'neurons_detected': len(traced_neurons),
                'traced_neurons': traced_neurons,
                'processing_time': processing_time,
                'feedback_stats': self.feedback_collector.get_feedback_stats(),
                'uncertainty_stats': self.uncertainty_estimator.get_uncertainty_stats()
            }
            
            print(f"✅ Processing complete: {len(traced_neurons)} neurons in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing region {region_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _segment_with_feedback(self, volume, region_name):
        """Enhanced segmentation with uncertainty estimation."""
        # Your existing segmentation logic here
        # For demo, we'll create a simple segmentation
        segmentation = torch.randint(0, 5, volume.shape)
        uncertainty_map = torch.zeros_like(segmentation, dtype=torch.float32)
        
        # Check each neuron for uncertainty
        for neuron_id in range(1, int(torch.max(segmentation)) + 1):
            neuron_mask = (segmentation == neuron_id)
            
            # Estimate uncertainty (in practice, use your model's uncertainty)
            uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                torch.tensor([0.5 + 0.4 * np.random.random()])
            )
            
            uncertainty_map[neuron_mask] = uncertainty
            
            # Check if intervention is needed
            if self.uncertainty_estimator.should_intervene(uncertainty):
                print(f"🤖 Segmentation intervention for neuron {neuron_id} (uncertainty: {uncertainty:.3f})")
                
                # Create intervention point
                intervention_point = InterventionPoint(
                    timestamp=datetime.now().isoformat(),
                    region_name=region_name,
                    neuron_id=neuron_id,
                    decision_type="segmentation",
                    current_state=None,
                    agent_suggestion={"segmentation": neuron_mask.numpy().tolist()},
                    uncertainty_score=uncertainty,
                    confidence_score=1.0 - uncertainty,
                    context={"volume_shape": volume.shape, "neuron_size": torch.sum(neuron_mask).item()}
                )
                
                # Request human intervention
                human_decision = self.human_callback.request_intervention(intervention_point)
                
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
                    reasoning="H01 segmentation uncertainty"
                )
                
                self.feedback_collector.add_feedback(feedback)
        
        return segmentation, uncertainty_map
    
    def _trace_with_feedback(self, segmentation, volume, region_name):
        """Enhanced tracing with human feedback."""
        traced_neurons = {}
        
        for neuron_id in range(1, int(torch.max(segmentation)) + 1):
            neuron_mask = (segmentation == neuron_id)
            
            # Get neuron coordinates
            coords = torch.nonzero(neuron_mask)
            if len(coords) == 0:
                continue
            
            # Create trace (simplified for demo)
            trace_coordinates = [(z.item(), y.item(), x.item()) 
                               for z, y, x in coords[:20]]  # First 20 points
            
            # Check for uncertain trace decisions
            interventions = 0
            for i, point in enumerate(trace_coordinates):
                if i > 0 and np.random.random() < 0.2:  # 20% chance of uncertainty
                    uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                        torch.tensor([0.6 + 0.3 * np.random.random()])
                    )
                    
                    if self.uncertainty_estimator.should_intervene(uncertainty):
                        print(f"🤖 Trace intervention for neuron {neuron_id} at point {i}")
                        
                        # Create intervention point
                        intervention_point = InterventionPoint(
                            timestamp=datetime.now().isoformat(),
                            region_name=region_name,
                            neuron_id=neuron_id,
                            decision_type="trace_continuation",
                            current_state=trace_coordinates[i-1] if i > 0 else None,
                            agent_suggestion={"next_point": point},
                            uncertainty_score=uncertainty,
                            confidence_score=1.0 - uncertainty,
                            context={"volume_shape": volume.shape, "trace_length": len(trace_coordinates)}
                        )
                        
                        # Request human intervention
                        human_decision = self.human_callback.request_intervention(intervention_point)
                        
                        # Record feedback
                        feedback = TracingFeedback(
                            timestamp=datetime.now().isoformat(),
                            region_name=region_name,
                            neuron_id=neuron_id,
                            decision_type="trace_continuation",
                            agent_decision={"next_point": point},
                            human_feedback=FeedbackType.ACCEPT if human_decision else FeedbackType.REJECT,
                            human_correction=human_decision if human_decision else None,
                            uncertainty_score=uncertainty,
                            confidence_score=1.0 - uncertainty,
                            reasoning="H01 trace continuation"
                        )
                        
                        self.feedback_collector.add_feedback(feedback)
                        interventions += 1
            
            traced_neurons[neuron_id] = {
                'coordinates': trace_coordinates,
                'length': len(trace_coordinates),
                'interventions': interventions
            }
        
        return traced_neurons
    
    def _detect_features_with_feedback(self, traced_neurons, volume, region_name):
        """Detect branches and synapses with feedback."""
        for neuron_id, neuron_data in traced_neurons.items():
            coordinates = neuron_data['coordinates']
            
            # Check for branches
            if len(coordinates) > 10:
                uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                    torch.tensor([0.5 + 0.4 * np.random.random()])
                )
                
                if self.uncertainty_estimator.should_intervene(uncertainty):
                    print(f"🤖 Branch detection intervention for neuron {neuron_id}")
                    
                    # Record feedback for branch detection
                    feedback = TracingFeedback(
                        timestamp=datetime.now().isoformat(),
                        region_name=region_name,
                        neuron_id=neuron_id,
                        decision_type="branch_detection",
                        agent_decision={"potential_branches": coordinates[10:15]},
                        human_feedback=FeedbackType.CORRECT,
                        human_correction={"confirmed_branches": coordinates[10:12]},
                        uncertainty_score=uncertainty,
                        confidence_score=1.0 - uncertainty,
                        reasoning="H01 branch detection"
                    )
                    
                    self.feedback_collector.add_feedback(feedback)
            
            # Check for synapses
            if len(coordinates) > 5:
                uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                    torch.tensor([0.6 + 0.3 * np.random.random()])
                )
                
                if self.uncertainty_estimator.should_intervene(uncertainty):
                    print(f"🤖 Synapse detection intervention for neuron {neuron_id}")
                    
                    # Record feedback for synapse detection
                    feedback = TracingFeedback(
                        timestamp=datetime.now().isoformat(),
                        region_name=region_name,
                        neuron_id=neuron_id,
                        decision_type="synapse_detection",
                        agent_decision={"synapse_type": "excitatory", "location": coordinates[5]},
                        human_feedback=FeedbackType.ACCEPT,
                        uncertainty_score=uncertainty,
                        confidence_score=1.0 - uncertainty,
                        reasoning="H01 synapse detection"
                    )
                    
                    self.feedback_collector.add_feedback(feedback)
    
    def get_feedback_summary(self):
        """Get comprehensive feedback summary."""
        feedback_stats = self.feedback_collector.get_feedback_stats()
        uncertainty_stats = self.uncertainty_estimator.get_uncertainty_stats()
        
        return {
            'feedback_stats': feedback_stats,
            'uncertainty_stats': uncertainty_stats,
            'autonomy_level': AutonomyLevel.SEMI_AUTONOMOUS.value,
            'uncertainty_threshold': self.uncertainty_estimator.uncertainty_threshold
        }
    
    def save_results(self, results, output_dir="h01_feedback_results"):
        """Save processing results and feedback data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        results_file = output_path / f"{results['region_name']}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save feedback summary
        feedback_summary = self.get_feedback_summary()
        feedback_file = output_path / f"{results['region_name']}_feedback.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback_summary, f, indent=2)
        
        print(f"✅ Results saved to: {output_path}")
        print(f"   Results: {results_file}")
        print(f"   Feedback: {feedback_file}")

def main():
    """Run H01 feedback integration demo."""
    print("H01 Feedback Integration Demo")
    print("=" * 60)
    
    # Initialize processor
    processor = H01FeedbackProcessor(uncertainty_threshold=0.6)
    
    # Define H01 regions to process
    h01_regions = [
        {
            'name': 'hippocampus_medium',
            'coords': (400000, 400000, 4000),
            'size': (512, 512, 512)
        },
        {
            'name': 'prefrontal_cortex_medium', 
            'coords': (500000, 500000, 5000),
            'size': (512, 512, 512)
        }
    ]
    
    print(f"Processing {len(h01_regions)} H01 regions...")
    
    all_results = []
    
    for region in h01_regions:
        print(f"\n{'='*60}")
        print(f"Processing: {region['name']}")
        print(f"{'='*60}")
        
        # Process region with feedback
        results = processor.process_h01_region_with_feedback(
            region['name'],
            region['coords'], 
            region['size']
        )
        
        if results:
            all_results.append(results)
            
            # Save results
            processor.save_results(results)
            
            # Print summary
            print(f"\n📊 Results Summary:")
            print(f"   Neurons detected: {results['neurons_detected']}")
            print(f"   Processing time: {results['processing_time']:.2f}s")
            print(f"   Total feedback: {results['feedback_stats']['total_feedback']}")
            print(f"   Average uncertainty: {results['uncertainty_stats']['mean']:.3f}")
        else:
            print(f"❌ Failed to process {region['name']}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    total_neurons = sum(r['neurons_detected'] for r in all_results)
    total_feedback = sum(r['feedback_stats']['total_feedback'] for r in all_results)
    total_time = sum(r['processing_time'] for r in all_results)
    
    print(f"Regions processed: {len(all_results)}")
    print(f"Total neurons: {total_neurons}")
    print(f"Total feedback: {total_feedback}")
    print(f"Total time: {total_time:.2f}s")
    
    # Get overall feedback summary
    overall_summary = processor.get_feedback_summary()
    print(f"\nOverall Feedback Summary:")
    print(f"  Feedback distribution: {overall_summary['feedback_stats']['feedback_distribution']}")
    print(f"  Uncertainty stats: {overall_summary['uncertainty_stats']}")
    
    print(f"\n✅ H01 Feedback Integration Demo Complete!")
    print(f"Check 'h01_feedback_results/' for detailed results")

if __name__ == "__main__":
    main() 