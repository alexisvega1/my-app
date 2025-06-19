#!/usr/bin/env python3
"""
Simplified Tracing Agent Demo
============================
Demonstrates the tracing agent system without TensorFlow dependencies.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, Any
import threading

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our components
from proofreading import UncertaintyTriggeredProofreader, ProofreadingResult
from continual_learning import LoRAContinualLearner, TrainingResult
from telemetry import start_metrics_server, get_telemetry, record_processing_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleTracingAgentDemo:
    """Simplified demo class for the tracing agent system."""
    
    def __init__(self):
        """Initialize the demo system."""
        self.proofreader = None
        self.lora_learner = None
        self.telemetry = None
        
        # Demo data
        self.demo_volume = None
        self.demo_labels = None
        
        logger.info("Simple Tracing Agent Demo initialized")
    
    def setup_components(self):
        """Set up all tracing agent components."""
        logger.info("Setting up tracing agent components...")
        
        # Initialize proofreader
        self.proofreader = UncertaintyTriggeredProofreader(
            uncertainty_threshold=0.6,
            enable_firestore=False  # Use stub for demo
        )
        
        # Initialize LoRA learner
        lora_config = {
            'rank': 4,
            'alpha': 8.0,
            'dropout': 0.1
        }
        training_config = {
            'learning_rate': 1e-3,
            'batch_size': 2,
            'num_epochs': 2
        }
        self.lora_learner = LoRAContinualLearner(
            lora_config=lora_config,
            training_config=training_config
        )
        
        # Initialize telemetry
        self.telemetry = get_telemetry()
        
        logger.info("All components initialized successfully")
    
    def generate_demo_data(self):
        """Generate demo volume and labels."""
        logger.info("Generating demo data...")
        
        # Create a simple 3D volume with some structure
        volume_shape = (32, 32, 32)
        self.demo_volume = np.random.random(volume_shape).astype(np.float32)
        
        # Add some structured regions (simulating neural structures)
        # Create a few "neurons" as connected components
        for i in range(3):
            center = np.random.randint(5, 27, 3)
            radius = np.random.randint(2, 5)
            
            # Create spherical region
            x, y, z = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
            mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
            self.demo_volume[mask] += 0.3
        
        # Normalize to [0, 1]
        self.demo_volume = np.clip(self.demo_volume, 0, 1)
        
        # Create corresponding labels (ground truth)
        self.demo_labels = (self.demo_volume > 0.6).astype(np.float32)
        
        logger.info(f"Generated demo data: volume shape {self.demo_volume.shape}")
    
    def demo_segmentation_stub(self):
        """Demonstrate stub segmentation (without FFN-v2)."""
        logger.info("=== Stub Segmentation Demo ===")
        
        start_time = time.time()
        
        # Create a simple segmentation
        seg_volume = self.demo_volume[8:24, 8:24, 8:24]
        
        # Generate random segmentation
        segmentation = np.random.random(seg_volume.shape) > 0.5
        segmentation = segmentation.astype(np.uint32)
        
        # Generate uncertainty map
        uncertainty_map = np.random.random(seg_volume.shape).astype(np.float32)
        
        # Calculate confidence
        confidence_score = 1.0 - np.mean(uncertainty_map)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        self.telemetry.record_segmentation_request("stub", "success")
        self.telemetry.record_segmentation_accuracy("stub", confidence_score)
        record_processing_time("stub", "segmentation", processing_time)
        
        logger.info(f"Stub segmentation completed:")
        logger.info(f"  - Confidence: {confidence_score:.3f}")
        logger.info(f"  - Processing time: {processing_time:.3f}s")
        logger.info(f"  - Output shape: {segmentation.shape}")
        
        return {
            'segmentation': segmentation,
            'uncertainty_map': uncertainty_map,
            'confidence_score': confidence_score,
            'processing_time': processing_time
        }
    
    def demo_proofreading(self, segmentation_result: Dict[str, Any]) -> ProofreadingResult:
        """Demonstrate uncertainty-triggered proofreading."""
        logger.info("=== Proofreading Demo ===")
        
        start_time = time.time()
        
        # Use the segmentation result
        segmentation = segmentation_result['segmentation']
        uncertainty_map = segmentation_result['uncertainty_map']
        
        # Perform proofreading
        result = self.proofreader.proofread(segmentation, uncertainty_map, {
            'source': 'stub',
            'confidence': segmentation_result['confidence_score']
        })
        
        processing_time = time.time() - start_time
        
        # Record metrics
        self.telemetry.record_proofreading_request(
            result.metadata['proofreading_triggered'], 
            'success'
        )
        record_processing_time("proofreader", "proofreading", processing_time)
        
        logger.info(f"Proofreading completed:")
        logger.info(f"  - Triggered: {result.metadata['proofreading_triggered']}")
        logger.info(f"  - Corrections made: {len(result.corrections_made)}")
        logger.info(f"  - Confidence improvement: {result.confidence_improvement:.3f}")
        logger.info(f"  - Processing time: {result.processing_time:.3f}s")
        
        if result.corrections_made:
            logger.info("  - Corrections:")
            for correction in result.corrections_made:
                logger.info(f"    * {correction['rule']}: {correction}")
        
        return result
    
    def demo_training(self):
        """Demonstrate LoRA continual learning."""
        logger.info("=== LoRA Training Demo ===")
        
        start_time = time.time()
        
        # Create training data from our demo volume
        # Use small patches for training
        patch_size = 8
        num_patches = 5
        
        training_data = []
        training_labels = []
        
        for _ in range(num_patches):
            # Random patch location
            x = np.random.randint(0, self.demo_volume.shape[0] - patch_size)
            y = np.random.randint(0, self.demo_volume.shape[1] - patch_size)
            z = np.random.randint(0, self.demo_volume.shape[2] - patch_size)
            
            # Extract patch
            patch = self.demo_volume[x:x+patch_size, y:y+patch_size, z:z+patch_size]
            label_patch = self.demo_labels[x:x+patch_size, y:y+patch_size, z:z+patch_size]
            
            # Flatten for training
            flattened_patch = patch.flatten()
            
            # Pad or truncate to match model input size (100 for stub model)
            if len(flattened_patch) < 100:
                # Pad with zeros
                padded_patch = np.zeros(100)
                padded_patch[:len(flattened_patch)] = flattened_patch
                training_data.append(padded_patch)
            else:
                # Truncate
                training_data.append(flattened_patch[:100])
            
            training_labels.append(np.mean(label_patch))  # Average label
        
        training_data = np.array(training_data)
        training_labels = np.array(training_labels)
        
        # Perform training
        result = self.lora_learner.update_model(training_data, training_labels)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        self.telemetry.record_training_request("lora", "success")
        record_processing_time("lora", "training", processing_time)
        
        logger.info(f"Training completed:")
        logger.info(f"  - Loss: {result.loss:.4f}")
        logger.info(f"  - Accuracy: {result.accuracy:.4f}")
        logger.info(f"  - Learning rate: {result.learning_rate:.6f}")
        logger.info(f"  - Epoch: {result.epoch}")
        logger.info(f"  - Training time: {result.training_time:.3f}s")
        
        return result
    
    def demo_checkpoint_management(self):
        """Demonstrate checkpoint management."""
        logger.info("=== Checkpoint Management Demo ===")
        
        # Save a checkpoint
        checkpoint_path = "demo_checkpoint.pt"
        success = self.lora_learner.save_checkpoint(checkpoint_path, {
            'demo': True,
            'timestamp': time.time()
        })
        
        if success:
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # List checkpoints
            checkpoints = self.lora_learner.list_checkpoints()
            logger.info(f"Available checkpoints: {len(checkpoints)}")
            
            for cp in checkpoints:
                logger.info(f"  - {cp.path}: epoch {cp.epoch}, loss {cp.loss:.4f}")
            
            # Update metrics
            self.telemetry.record_model_checkpoints("lora", len(checkpoints))
        else:
            logger.warning("Failed to save checkpoint")
    
    def demo_telemetry(self):
        """Demonstrate telemetry and monitoring."""
        logger.info("=== Telemetry Demo ===")
        
        # Record some additional metrics
        self.telemetry.record_active_agents("stub", 1)
        self.telemetry.record_active_agents("proofreader", 1)
        self.telemetry.record_active_agents("lora", 1)
        
        # Record some requests
        self.telemetry.record_request("POST", "/segment", 200, 1.5, 1024)
        self.telemetry.record_request("POST", "/proofread", 200, 0.8, 512)
        self.telemetry.record_request("POST", "/train", 200, 5.2, 2048)
        
        # Get performance summary
        summary = self.telemetry.get_performance_summary(hours=1)
        logger.info("Performance summary (last hour):")
        for key, value in summary.items():
            logger.info(f"  - {key}: {value}")
        
        # Export metrics
        metrics_json = self.telemetry.export_metrics('json')
        logger.info("Metrics exported (JSON format)")
    
    def run_full_demo(self):
        """Run the complete tracing agent demo."""
        logger.info("🚀 Starting Simple Tracing Agent Demo")
        logger.info("=" * 50)
        
        try:
            # Setup components
            self.setup_components()
            
            # Generate demo data
            self.generate_demo_data()
            
            # Run segmentation demo
            seg_result = self.demo_segmentation_stub()
            
            # Run proofreading demo
            proof_result = self.demo_proofreading(seg_result)
            
            # Run training demo
            train_result = self.demo_training()
            
            # Run checkpoint management demo
            self.demo_checkpoint_management()
            
            # Run telemetry demo
            self.demo_telemetry()
            
            # Get component statistics
            self.print_statistics()
            
            logger.info("=" * 50)
            logger.info("✅ Simple Tracing Agent Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    def print_statistics(self):
        """Print statistics from all components."""
        logger.info("=== Component Statistics ===")
        
        # Proofreader statistics
        if self.proofreader:
            proof_stats = self.proofreader.get_statistics()
            logger.info("Proofreader:")
            for key, value in proof_stats.items():
                if key != 'proofreading_rules':  # Skip rules for brevity
                    logger.info(f"  - {key}: {value}")
        
        # LoRA learner statistics
        if self.lora_learner:
            lora_stats = self.lora_learner.get_statistics()
            logger.info("LoRA Learner:")
            for key, value in lora_stats.items():
                if key not in ['lora_config', 'training_config']:  # Skip configs for brevity
                    logger.info(f"  - {key}: {value}")
        
        # Telemetry statistics
        if self.telemetry:
            telemetry_stats = self.telemetry.get_statistics()
            logger.info("Telemetry:")
            for key, value in telemetry_stats.items():
                if key not in ['metrics']:  # Skip metrics dict for brevity
                    logger.info(f"  - {key}: {value}")
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up demo resources...")
        
        if self.proofreader:
            self.proofreader.cleanup()
        
        if self.lora_learner:
            self.lora_learner.cleanup()
        
        if self.telemetry:
            self.telemetry.cleanup()
        
        logger.info("Demo cleanup completed")

def main():
    """Main demo function."""
    # Start metrics server in background
    logger.info("Starting metrics server...")
    start_metrics_server(8000)
    logger.info("Prometheus metrics available at http://localhost:8000/metrics")
    
    # Create and run demo
    demo = SimpleTracingAgentDemo()
    
    try:
        demo.run_full_demo()
    finally:
        demo.cleanup()

if __name__ == "__main__":
    main() 