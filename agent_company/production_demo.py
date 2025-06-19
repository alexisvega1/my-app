#!/usr/bin/env python3
"""
Production Demo for Advanced Agentic Tracer
==========================================
Demonstrates the next-generation FFN-v2, advanced proofreading, and continual learning
systems working together for petabyte-scale connectomics processing.
"""

import os
import sys
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json
import tempfile
import shutil

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our advanced components
from segmenters.ffn_v2_advanced import AdvancedFFNv2Plugin, AdvancedSegmentationResult
from proofreading_advanced import AdvancedProofreader, ProofreadingResult
from continual_learning_advanced import AdvancedContinualLearner, TrainingResult
from telemetry import TelemetrySystem
from memory import Memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionAgenticTracer:
    """Production-ready agentic tracer for large-scale connectomics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'ffn_config': {
                'model_config': {
                    'input_channels': 1,
                    'hidden_channels': [32, 64, 128, 256],
                    'output_channels': 1,
                    'use_attention': True,
                    'use_residual': True,
                    'dropout_rate': 0.1
                },
                'distributed_config': {
                    'num_processes': 4,
                    'num_threads': 8,
                    'chunk_size': (64, 64, 64),
                    'batch_size': 4
                }
            },
            'proofreading_config': {
                'error_detection': {
                    'use_topology': True,
                    'use_morphology': True,
                    'use_consistency': True,
                    'use_boundary': True,
                    'use_connectivity': True,
                    'min_component_size': 100,
                    'connectivity_threshold': 5
                },
                'error_correction': {
                    'use_morphological': True,
                    'use_topological': True,
                    'use_interpolation': True,
                    'use_smoothing': True,
                    'use_reconstruction': True,
                    'morphology_kernel_size': 3,
                    'smoothing_sigma': 1.0
                }
            },
            'continual_learning_config': {
                'model_config': {
                    'rank': 16,
                    'alpha': 32.0,
                    'dropout': 0.1,
                    'adaptation_strategy': 'low_rank'
                },
                'training_config': {
                    'batch_size': 32,
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-5,
                    'num_epochs': 5,
                    'validation_split': 0.2,
                    'early_stopping_patience': 3
                }
            },
            'telemetry_config': {
                'host': 'localhost',
                'port': 8000,
                'metrics_interval': 5
            }
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        # Initialize components
        self.ffn_plugin = None
        self.proofreader = None
        self.continual_learner = None
        self.telemetry_server = None
        self.memory_manager = None
        
        # Performance tracking
        self.stats = {
            'volumes_processed': 0,
            'total_processing_time': 0.0,
            'total_errors_corrected': 0,
            'total_training_sessions': 0,
            'start_time': time.time()
        }
        
        logger.info("Production Agentic Tracer initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing production components...")
            
            # Initialize FFN-v2 plugin
            self.ffn_plugin = AdvancedFFNv2Plugin(self.config['ffn_config'])
            
            # Load the trained model
            model_path = 'quick_ffn_v2_model.pt'
            if os.path.exists(model_path):
                success = self.ffn_plugin.load_model(model_path)
                if success:
                    logger.info("✓ FFN-v2 plugin initialized with trained model")
                else:
                    logger.warning("Failed to load trained model, using stub")
            else:
                logger.warning(f"Model file {model_path} not found, using stub")
            
            # Initialize proofreader
            self.proofreader = AdvancedProofreader(self.config['proofreading_config'])
            logger.info("✓ Advanced proofreader initialized")
            
            # Initialize continual learner
            self.continual_learner = AdvancedContinualLearner(self.config['continual_learning_config'])
            logger.info("✓ Advanced continual learner initialized")
            
            # Initialize telemetry server
            self.telemetry_server = TelemetrySystem(
                port=self.config['telemetry_config']['port'],
                enable_prometheus=True,
                enable_system_metrics=True
            )
            self.telemetry_server.start_metrics_server()
            self.telemetry_server.start_system_monitoring()
            logger.info("✓ Telemetry server started")
            
            # Initialize memory manager
            self.memory_manager = MemoryUsageTracker()
            logger.info("✓ Memory manager initialized")
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def process_volume_pipeline(self, 
                              volume_path: str,
                              output_dir: str,
                              seed_point: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """Process a volume through the complete pipeline."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting volume processing pipeline: {volume_path}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: FFN-v2 Segmentation
            logger.info("Step 1: FFN-v2 Segmentation")
            segmentation_output = os.path.join(output_dir, "segmentation")
            segmentation_result = self.ffn_plugin.segment(
                volume_path=volume_path,
                output_path=segmentation_output,
                seed_point=seed_point
            )
            
            # Step 2: Advanced Proofreading
            logger.info("Step 2: Advanced Proofreading")
            proofreading_result = self.proofreader.proofread(
                segmentation=segmentation_result.segmentation,
                uncertainty_map=segmentation_result.uncertainty_map,
                metadata={
                    'volume_path': volume_path,
                    'segmentation_confidence': segmentation_result.confidence_score
                }
            )
            
            # Step 3: Save proofread results
            proofreading_output = os.path.join(output_dir, "proofread")
            self.proofreader.save_results(proofreading_result, proofreading_output)
            
            # Step 4: Prepare training data for continual learning
            logger.info("Step 3: Preparing training data")
            training_data = self._prepare_training_data(
                segmentation_result, proofreading_result, output_dir
            )
            
            # Step 5: Continual Learning (if training data available)
            training_result = None
            if training_data:
                logger.info("Step 4: Continual Learning")
                training_result = self.continual_learner.train_on_new_data(
                    new_data_paths=training_data
                )
                
                # Save updated model
                model_output = os.path.join(output_dir, "updated_model.pt")
                self.continual_learner.save_adapters(model_output)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['volumes_processed'] += 1
            
            # Compile results
            results = {
                'volume_path': volume_path,
                'output_dir': output_dir,
                'processing_time': processing_time,
                'segmentation': {
                    'confidence_score': segmentation_result.confidence_score,
                    'processing_time': segmentation_result.processing_time,
                    'quality_metrics': segmentation_result.quality_metrics,
                    'memory_usage': segmentation_result.memory_usage
                },
                'proofreading': {
                    'errors_detected': np.sum(proofreading_result.error_map),
                    'processing_time': proofreading_result.processing_time,
                    'quality_metrics': proofreading_result.quality_metrics,
                    'memory_usage': proofreading_result.memory_usage
                },
                'continual_learning': {
                    'training_result': training_result.__dict__ if training_result else None,
                    'training_data_available': bool(training_data)
                },
                'pipeline_stats': {
                    'total_errors_corrected': np.sum(proofreading_result.error_map),
                    'volume_change_ratio': proofreading_result.quality_metrics.get('volume_change_ratio', 0.0),
                    'overall_confidence': np.mean(proofreading_result.confidence_scores)
                }
            }
            
            # Update telemetry
            self._update_telemetry(results)
            
            logger.info(f"Volume processing completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Volume processing failed: {e}")
            raise
    
    def _prepare_training_data(self, 
                             segmentation_result: AdvancedSegmentationResult,
                             proofreading_result: ProofreadingResult,
                             output_dir: str) -> List[str]:
        """Prepare training data for continual learning."""
        try:
            # Create synthetic training data based on corrections
            training_data_paths = []
            
            # Generate training samples from corrected regions
            corrected_regions = proofreading_result.error_map
            if np.any(corrected_regions):
                # Create training samples
                sample_size = (32, 32, 32)
                num_samples = min(100, np.sum(corrected_regions) // 1000)
                
                for i in range(num_samples):
                    # Find a corrected region
                    corrected_indices = np.where(corrected_regions)
                    if len(corrected_indices[0]) == 0:
                        break
                    
                    # Random sample from corrected regions
                    idx = np.random.randint(len(corrected_indices[0]))
                    z, y, x = corrected_indices[0][idx], corrected_indices[1][idx], corrected_indices[2][idx]
                    
                    # Extract sample
                    z_start = max(0, z - sample_size[0] // 2)
                    y_start = max(0, y - sample_size[1] // 2)
                    x_start = max(0, x - sample_size[2] // 2)
                    
                    z_end = min(segmentation_result.segmentation.shape[0], z_start + sample_size[0])
                    y_end = min(segmentation_result.segmentation.shape[1], y_start + sample_size[1])
                    x_end = min(segmentation_result.segmentation.shape[2], x_start + sample_size[2])
                    
                    # Extract input and target
                    input_sample = segmentation_result.segmentation[z_start:z_end, y_start:y_end, x_start:x_end]
                    target_sample = proofreading_result.corrected_segmentation[z_start:z_end, y_start:y_end, x_start:x_end]
                    
                    # Save sample
                    sample_path = os.path.join(output_dir, f"training_sample_{i}")
                    np.save(f"{sample_path}_input.npy", input_sample)
                    np.save(f"{sample_path}_target.npy", target_sample)
                    training_data_paths.append(sample_path)
            
            logger.info(f"Prepared {len(training_data_paths)} training samples")
            return training_data_paths
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return []
    
    def _update_telemetry(self, results: Dict[str, Any]):
        """Update telemetry metrics."""
        try:
            if self.telemetry_server:
                # Update processing metrics
                self.telemetry_server.record_processing_time('pipeline', 'volume_processing', results['processing_time'])
                self.telemetry_server.record_segmentation_request('ffn_v2_advanced', 'success')
                self.telemetry_server.record_segmentation_accuracy('ffn_v2_advanced', results['segmentation']['confidence_score'])
                self.telemetry_server.record_proofreading_request(True, 'success')
                
                # Update system metrics
                if self.memory_manager:
                    memory_usage = self.memory_manager.get_memory_usage()
                    self.telemetry_server.record_memory_usage(memory_usage.get('total_bytes', 0))
                
        except Exception as e:
            logger.error(f"Failed to update telemetry: {e}")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            **self.stats,
            'uptime': time.time() - self.stats['start_time'],
            'ffn_stats': self.ffn_plugin.get_statistics() if self.ffn_plugin else {},
            'proofreader_stats': self.proofreader.get_statistics() if self.proofreader else {},
            'continual_learner_stats': self.continual_learner.get_statistics() if self.continual_learner else {},
            'memory_stats': self.memory_manager.get_memory_usage() if self.memory_manager else {},
            'telemetry_url': f"http://{self.config['telemetry_config']['host']}:{self.config['telemetry_config']['port']}/metrics"
        }
        
        return stats
    
    def cleanup(self):
        """Clean up all components."""
        logger.info("Cleaning up production components...")
        
        if self.ffn_plugin:
            self.ffn_plugin.cleanup()
        
        if self.telemetry_server:
            self.telemetry_server.stop_system_monitoring()
            self.telemetry_server.cleanup()
        
        logger.info("Cleanup completed")

def create_synthetic_volume(shape: Tuple[int, int, int] = (256, 256, 256)) -> str:
    """Create a synthetic volume for testing."""
    temp_dir = tempfile.mkdtemp()
    volume_path = os.path.join(temp_dir, "synthetic_volume")
    
    # Create synthetic EM data
    volume = np.random.random(shape).astype(np.float32)
    
    # Add some structure
    for i in range(10):
        center = np.random.randint(0, min(shape), 3)
        radius = np.random.randint(10, 30)
        
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        mask = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2 <= radius**2
        volume[mask] += 0.5
    
    # Save volume
    np.save(f"{volume_path}.npy", volume)
    
    logger.info(f"Created synthetic volume: {volume_path}.npy")
    return f"{volume_path}.npy"

# Simple memory usage tracker
class MemoryUsageTracker:
    """Simple memory usage tracker for production demo."""
    
    def __init__(self):
        self.memory = Memory()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'total_bytes': memory_info.rss,
                'total_gb': memory_info.rss / (1024**3),
                'percent': process.memory_percent()
            }
        except ImportError:
            return {
                'total_bytes': 0,
                'total_gb': 0,
                'percent': 0
            }

def main():
    """Main production demo."""
    logger.info("Starting Production Agentic Tracer Demo")
    logger.info("=" * 50)
    
    # Initialize the production tracer
    tracer = ProductionAgenticTracer()
    
    try:
        # Initialize components
        if not tracer.initialize_components():
            logger.error("Failed to initialize components")
            return
        
        # Create synthetic test volume
        logger.info("Creating synthetic test volume...")
        volume_path = create_synthetic_volume((128, 128, 128))
        
        # Process volume through pipeline
        logger.info("Processing volume through production pipeline...")
        output_dir = "production_output"
        
        results = tracer.process_volume_pipeline(
            volume_path=volume_path,
            output_dir=output_dir
        )
        
        # Display results
        logger.info("Processing Results:")
        logger.info("-" * 30)
        logger.info(f"Volume processed: {results['volume_path']}")
        logger.info(f"Processing time: {results['processing_time']:.2f}s")
        logger.info(f"Segmentation confidence: {results['segmentation']['confidence_score']:.3f}")
        logger.info(f"Errors detected: {results['proofreading']['errors_detected']}")
        logger.info(f"Volume change ratio: {results['pipeline_stats']['volume_change_ratio']:.3f}")
        logger.info(f"Overall confidence: {results['pipeline_stats']['overall_confidence']:.3f}")
        
        # Display system statistics
        logger.info("\nSystem Statistics:")
        logger.info("-" * 30)
        stats = tracer.get_system_statistics()
        logger.info(f"Volumes processed: {stats['volumes_processed']}")
        logger.info(f"Total processing time: {stats['total_processing_time']:.2f}s")
        logger.info(f"System uptime: {stats['uptime']:.2f}s")
        logger.info(f"Telemetry available at: {stats['telemetry_url']}")
        
        # Save results
        results_file = os.path.join(output_dir, "processing_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {results_file}")
        logger.info("Production demo completed successfully!")
        
        # Keep telemetry server running for monitoring
        logger.info("Telemetry server running. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
    except Exception as e:
        logger.error(f"Production demo failed: {e}")
        raise
    
    finally:
        # Cleanup
        tracer.cleanup()
        
        # Clean up temporary files
        if 'volume_path' in locals():
            temp_dir = os.path.dirname(volume_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        logger.info("Production demo cleanup completed")

if __name__ == "__main__":
    main() 