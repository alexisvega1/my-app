#!/usr/bin/env python3
"""
Enhanced H01 Pipeline with Human Feedback Integration
====================================================
Integrates human feedback RL with your existing H01 processing pipeline.
"""

import numpy as np
import torch
from pathlib import Path
import json
import time
from datetime import datetime
import logging

# Import our feedback system
from human_feedback_rl import (
    FeedbackCollector,
    UncertaintyEstimator,
    HumanInTheLoopCallback,
    AutonomyLevel,
    FeedbackType,
    TracingFeedback,
    InterventionPoint
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedH01Processor:
    """Enhanced H01 processor with human feedback integration."""
    
    def __init__(self, 
                 feedback_dir="enhanced_h01_feedback",
                 uncertainty_threshold=0.7,
                 autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS,
                 enable_web_interface=False):
        
        self.feedback_dir = feedback_dir
        self.uncertainty_threshold = uncertainty_threshold
        self.autonomy_level = autonomy_level
        self.enable_web_interface = enable_web_interface
        
        # Initialize feedback components
        self.feedback_collector = FeedbackCollector(feedback_dir)
        self.uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold)
        
        if enable_web_interface:
            # Use web interface callback
            from web_feedback_interface import WebFeedbackCallback
            self.human_callback = WebFeedbackCallback(self.feedback_collector, self.uncertainty_estimator)
        else:
            # Use CLI callback
            self.human_callback = HumanInTheLoopCallback(autonomy_level)
        
        # Register custom callbacks for H01 data
        self._register_h01_callbacks()
        
        # Processing statistics
        self.stats = {
            'regions_processed': 0,
            'total_neurons': 0,
            'total_interventions': 0,
            'total_processing_time': 0,
            'feedback_collected': 0
        }
        
        logger.info(f"✅ Enhanced H01 Processor initialized")
        logger.info(f"   Feedback directory: {feedback_dir}")
        logger.info(f"   Uncertainty threshold: {uncertainty_threshold}")
        logger.info(f"   Autonomy level: {autonomy_level.value}")
        logger.info(f"   Web interface: {enable_web_interface}")
    
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
    
    def process_h01_region(self, region_name, region_coords, region_size):
        """Process H01 region with enhanced feedback integration."""
        logger.info(f"🔬 Processing H01 region: {region_name}")
        logger.info(f"   Coordinates: {region_coords}")
        logger.info(f"   Size: {region_size}")
        
        start_time = time.time()
        region_interventions = 0
        
        try:
            # Load H01 region data (mock for demo)
            logger.info("Loading H01 region data...")
            volume = np.random.randint(0, 255, region_size, dtype=np.uint8)
            
            if volume is None or volume.size == 0:
                logger.warning("❌ No data found for this region")
                return None
            
            logger.info(f"✅ Loaded volume: {volume.shape}, {volume.dtype}")
            
            # Enhanced segmentation with uncertainty estimation
            logger.info("Segmenting with uncertainty estimation...")
            segmentation, uncertainty_map, segmentation_interventions = self._enhanced_segmentation(
                volume, region_name
            )
            
            if segmentation is None:
                logger.error("❌ Segmentation failed")
                return None
            
            region_interventions += segmentation_interventions
            logger.info(f"✅ Segmentation complete: {torch.max(segmentation)} neurons")
            
            # Enhanced tracing with feedback
            logger.info("Tracing with human feedback...")
            traced_neurons, tracing_interventions = self._enhanced_tracing(
                segmentation, volume, region_name
            )
            
            region_interventions += tracing_interventions
            
            # Enhanced feature detection
            logger.info("Detecting branches and synapses...")
            feature_interventions = self._enhanced_feature_detection(
                traced_neurons, volume, region_name
            )
            
            region_interventions += feature_interventions
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['regions_processed'] += 1
            self.stats['total_neurons'] += len(traced_neurons)
            self.stats['total_interventions'] += region_interventions
            self.stats['total_processing_time'] += processing_time
            
            # Compile results
            results = {
                'region_name': region_name,
                'region_coords': region_coords,
                'region_size': region_size,
                'volume_shape': volume.shape,
                'neurons_detected': len(traced_neurons),
                'traced_neurons': traced_neurons,
                'processing_time': processing_time,
                'interventions': region_interventions,
                'feedback_stats': self.feedback_collector.get_feedback_stats(),
                'uncertainty_stats': self.uncertainty_estimator.get_uncertainty_stats()
            }
            
            logger.info(f"✅ Processing complete: {len(traced_neurons)} neurons in {processing_time:.2f}s")
            logger.info(f"   Interventions: {region_interventions}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing region {region_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _enhanced_segmentation(self, volume, region_name):
        """Enhanced segmentation with uncertainty estimation and feedback."""
        # Create segmentation with uncertainty
        segmentation = torch.randint(0, 5, volume.shape)
        uncertainty_map = torch.zeros_like(segmentation, dtype=torch.float32)
        
        interventions = 0
        
        # Check each neuron for uncertainty
        for neuron_id in range(1, int(torch.max(segmentation)) + 1):
            neuron_mask = (segmentation == neuron_id)
            
            # Estimate uncertainty (in practice, use your model's uncertainty)
            # Here we simulate uncertainty based on neuron size and position
            neuron_size = torch.sum(neuron_mask).item()
            center_z, center_y, center_x = [s//2 for s in volume.shape]
            
            # Simulate uncertainty: larger neurons and edge neurons are more uncertain
            size_factor = min(1.0, neuron_size / 1000)  # Normalize by size
            edge_factor = 0.5 if (center_z < 10 or center_z > volume.shape[0]-10) else 0.0
            
            uncertainty = 0.3 + 0.4 * size_factor + edge_factor + 0.2 * np.random.random()
            uncertainty = min(1.0, uncertainty)
            
            uncertainty_map[neuron_mask] = uncertainty
            
            # Check if intervention is needed
            if self.uncertainty_estimator.should_intervene(uncertainty):
                logger.info(f"🤖 Segmentation intervention for neuron {neuron_id} (uncertainty: {uncertainty:.3f})")
                
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
                    context={
                        "volume_shape": volume.shape, 
                        "neuron_size": neuron_size,
                        "center": (center_z, center_y, center_x)
                    }
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
                interventions += 1
        
        return segmentation, uncertainty_map, interventions
    
    def _enhanced_tracing(self, segmentation, volume, region_name):
        """Enhanced tracing with human feedback integration."""
        traced_neurons = {}
        interventions = 0
        
        for neuron_id in range(1, int(torch.max(segmentation)) + 1):
            neuron_mask = (segmentation == neuron_id)
            
            # Get neuron coordinates
            coords = torch.nonzero(neuron_mask)
            if len(coords) == 0:
                continue
            
            # Create trace (simplified for demo)
            # In practice, use your actual tracing algorithm
            trace_coordinates = [(z.item(), y.item(), x.item()) 
                               for z, y, x in coords[:20]]  # First 20 points
            
            # Check for uncertain trace decisions
            neuron_interventions = 0
            for i, point in enumerate(trace_coordinates):
                # Simulate uncertainty in trace continuation
                # Points near edges or with complex geometry are more uncertain
                z, y, x = point
                edge_factor = 0.3 if (z < 5 or z > volume.shape[0]-5 or 
                                     y < 5 or y > volume.shape[1]-5 or 
                                     x < 5 or x > volume.shape[2]-5) else 0.0
                
                complexity_factor = 0.2 if i > 10 else 0.0  # Later points are more complex
                
                uncertainty = 0.4 + edge_factor + complexity_factor + 0.2 * np.random.random()
                uncertainty = min(1.0, uncertainty)
                
                if self.uncertainty_estimator.should_intervene(uncertainty):
                    logger.info(f"🤖 Trace intervention for neuron {neuron_id} at point {i}")
                    
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
                        context={
                            "volume_shape": volume.shape, 
                            "trace_length": len(trace_coordinates),
                            "point_index": i,
                            "coordinates": point
                        }
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
                    neuron_interventions += 1
            
            traced_neurons[neuron_id] = {
                'coordinates': trace_coordinates,
                'length': len(trace_coordinates),
                'interventions': neuron_interventions
            }
            
            interventions += neuron_interventions
        
        return traced_neurons, interventions
    
    def _enhanced_feature_detection(self, traced_neurons, volume, region_name):
        """Enhanced feature detection with feedback."""
        interventions = 0
        
        for neuron_id, neuron_data in traced_neurons.items():
            coordinates = neuron_data['coordinates']
            
            # Check for branches (neurons with longer traces)
            if len(coordinates) > 10:
                # Simulate branch detection uncertainty
                uncertainty = 0.5 + 0.3 * np.random.random()
                
                if self.uncertainty_estimator.should_intervene(uncertainty):
                    logger.info(f"🤖 Branch detection intervention for neuron {neuron_id}")
                    
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
                    interventions += 1
            
            # Check for synapses (neurons with medium traces)
            if len(coordinates) > 5:
                # Simulate synapse detection uncertainty
                uncertainty = 0.6 + 0.2 * np.random.random()
                
                if self.uncertainty_estimator.should_intervene(uncertainty):
                    logger.info(f"🤖 Synapse detection intervention for neuron {neuron_id}")
                    
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
                    interventions += 1
        
        return interventions
    
    def process_batch(self, regions):
        """Process a batch of H01 regions with feedback integration."""
        logger.info(f"🚀 Processing batch of {len(regions)} regions...")
        
        results = []
        
        for region in regions:
            region_name = region['name']
            region_coords = region['coords']
            region_size = region['size']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {region_name}")
            logger.info(f"{'='*60}")
            
            # Process region
            result = self.process_h01_region(region_name, region_coords, region_size)
            
            if result:
                results.append(result)
                
                # Save results
                self.save_results(result)
                
                # Print summary
                logger.info(f"\n📊 Results Summary:")
                logger.info(f"   Neurons detected: {result['neurons_detected']}")
                logger.info(f"   Processing time: {result['processing_time']:.2f}s")
                logger.info(f"   Interventions: {result['interventions']}")
                logger.info(f"   Total feedback: {result['feedback_stats']['total_feedback']}")
            else:
                logger.error(f"❌ Failed to process {region_name}")
        
        return results
    
    def save_results(self, results, output_dir="enhanced_h01_results"):
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
        
        logger.info(f"✅ Results saved to: {output_path}")
    
    def get_feedback_summary(self):
        """Get comprehensive feedback summary."""
        feedback_stats = self.feedback_collector.get_feedback_stats()
        uncertainty_stats = self.uncertainty_estimator.get_uncertainty_stats()
        
        return {
            'feedback_stats': feedback_stats,
            'uncertainty_stats': uncertainty_stats,
            'processing_stats': self.stats,
            'autonomy_level': self.autonomy_level.value,
            'uncertainty_threshold': self.uncertainty_estimator.uncertainty_threshold
        }
    
    def get_training_data(self, min_feedback_count=100):
        """Get feedback data for model training."""
        return self.feedback_collector.get_training_data(min_feedback_count)
    
    def print_summary(self):
        """Print comprehensive summary."""
        logger.info(f"\n{'='*80}")
        logger.info("ENHANCED H01 PIPELINE SUMMARY")
        logger.info(f"{'='*80}")
        
        logger.info(f"Regions processed: {self.stats['regions_processed']}")
        logger.info(f"Total neurons: {self.stats['total_neurons']}")
        logger.info(f"Total interventions: {self.stats['total_interventions']}")
        logger.info(f"Total processing time: {self.stats['total_processing_time']:.2f}s")
        
        # Get overall feedback summary
        overall_summary = self.get_feedback_summary()
        logger.info(f"\nOverall Feedback Summary:")
        logger.info(f"  Feedback distribution: {overall_summary['feedback_stats']['feedback_distribution']}")
        logger.info(f"  Uncertainty stats: {overall_summary['uncertainty_stats']}")
        
        # Check if we have enough data for training
        training_data = self.get_training_data()
        if training_data:
            logger.info(f"✅ Training data available: {len(training_data['states'])} samples")
        else:
            logger.info(f"⚠️  Insufficient training data (need at least 100 feedback samples)")

def main():
    """Run the enhanced H01 pipeline demo."""
    logger.info("Enhanced H01 Pipeline with Human Feedback")
    logger.info("=" * 60)
    
    # Define H01 regions to process
    h01_regions = [
        {
            'name': 'hippocampus_medium',
            'coords': (400000, 400000, 4000),
            'size': (256, 256, 256)  # Smaller for demo
        },
        {
            'name': 'prefrontal_cortex_medium', 
            'coords': (500000, 500000, 5000),
            'size': (256, 256, 256)  # Smaller for demo
        }
    ]
    
    # Initialize processor with different configurations
    logger.info("Initializing enhanced processor...")
    
    # Option 1: CLI interface (default)
    processor = EnhancedH01Processor(
        feedback_dir="enhanced_h01_feedback",
        uncertainty_threshold=0.6,  # Lower threshold for more interventions
        autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS,
        enable_web_interface=False
    )
    
    logger.info(f"Processing {len(h01_regions)} H01 regions...")
    
    # Process batch
    results = processor.process_batch(h01_regions)
    
    # Print final summary
    processor.print_summary()
    
    logger.info(f"\n✅ Enhanced H01 Pipeline Demo Complete!")
    logger.info(f"Check 'enhanced_h01_results/' for detailed results")
    logger.info(f"Check '{processor.feedback_dir}/' for feedback data")

if __name__ == "__main__":
    main() 