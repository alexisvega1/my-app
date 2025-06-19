#!/usr/bin/env python3
"""
Production Pipeline for Agentic Tracer
======================================
End-to-end production pipeline with configuration management,
data loading, processing, and monitoring.
"""

import os
import sys
import logging
import time
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader, create_data_loader
from segmenters.ffn_v2_advanced import AdvancedFFNv2Plugin
from proofreading_advanced import AdvancedProofreader
from continual_learning import LoRAContinualLearner
from telemetry import TelemetrySystem

logger = logging.getLogger(__name__)

class ProductionPipeline:
    """Production pipeline for large-scale connectomics processing."""
    
    def __init__(self, config_path: str):
        """Initialize the production pipeline."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.data_loader = None
        self.ffn_plugin = None
        self.proofreader = None
        self.continual_learner = None
        self.telemetry = None
        
        # Pipeline state
        self.is_initialized = False
        self.processing_stats = {
            'volumes_processed': 0,
            'total_processing_time': 0.0,
            'errors_encountered': 0
        }
        
        logger.info("Production pipeline initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def initialize(self) -> bool:
        """Initialize all pipeline components."""
        try:
            logger.info("Initializing production pipeline components...")
            
            # Initialize data loader
            self.data_loader = create_data_loader(self.config_path)
            if not self.data_loader.validate_data_source():
                logger.error("Data source validation failed")
                return False
            
            # Initialize FFN-v2 plugin
            ffn_config = self.config.get('model', {}).get('ffn_v2', {})
            self.ffn_plugin = AdvancedFFNv2Plugin(ffn_config)
            
            model_path = ffn_config.get('model_path', 'quick_ffn_v2_model.pt')
            if not self.ffn_plugin.load_model(model_path):
                logger.error("Failed to load FFN-v2 model")
                return False
            
            # Initialize proofreader
            proofreading_config = self.config.get('proofreading', {})
            self.proofreader = AdvancedProofreader(proofreading_config)
            
            # Initialize continual learner
            cl_config = self.config.get('model', {}).get('continual_learning', {})
            if cl_config.get('enabled', True):
                model_path = cl_config.get('model_path', 'quick_ffn_v2_model.pt')
                lora_config = cl_config.get('lora_config', {})
                training_config = cl_config.get('training_config', {})
                self.continual_learner = LoRAContinualLearner(
                    model_path=model_path,
                    lora_config=lora_config,
                    training_config=training_config
                )
            
            # Initialize telemetry
            telemetry_config = self.config.get('telemetry', {})
            if telemetry_config.get('prometheus', {}).get('enabled', True):
                self.telemetry = TelemetrySystem(
                    port=telemetry_config['prometheus'].get('port', 8000),
                    enable_prometheus=True,
                    enable_system_metrics=telemetry_config.get('system_monitoring', {}).get('enabled', True),
                    metrics_prefix="agentic_tracer"
                )
                self.telemetry.start_metrics_server()
                if telemetry_config.get('system_monitoring', {}).get('enabled', True):
                    self.telemetry.start_system_monitoring()
            
            self.is_initialized = True
            logger.info("✓ All pipeline components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def process_volume(self, volume_id: str, output_path: str) -> bool:
        """Process a single volume through the complete pipeline."""
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return False
        
        start_time = time.time()
        success = False
        
        try:
            logger.info(f"Starting processing of volume {volume_id}")
            
            # Get volume information
            volume_info = self.data_loader.get_volume_info()
            logger.info(f"Volume info: {volume_info}")
            
            # Step 1: Segmentation
            logger.info("Step 1: Performing segmentation...")
            segmentation_result = self.ffn_plugin.segment(
                volume_path=volume_id,  # In production, this would be a real path
                output_path=f"{output_path}_segmentation"
            )
            
            if segmentation_result is None:
                logger.error("Segmentation failed")
                return False
            
            # Step 2: Proofreading
            logger.info("Step 2: Performing proofreading...")
            proofreading_result = self.proofreader.proofread(
                segmentation=segmentation_result.segmentation,
                uncertainty_map=segmentation_result.uncertainty_map,
                metadata={
                    'volume_id': volume_id,
                    'volume_shape': volume_info['shape'],
                    'segmentation_confidence': segmentation_result.confidence_score
                }
            )
            
            # Step 3: Continual Learning (if enabled)
            if self.continual_learner is not None:
                logger.info("Step 3: Performing continual learning...")
                learning_result = self.continual_learner.adapt(
                    segmentation=proofreading_result.corrected_segmentation,
                    uncertainty_map=segmentation_result.uncertainty_map,
                    metadata={
                        'volume_id': volume_id,
                        'proofreading_quality': proofreading_result.quality_metrics
                    }
                )
                
                if learning_result:
                    logger.info("✓ Continual learning completed successfully")
            
            # Step 4: Save results
            logger.info("Step 4: Saving results...")
            self._save_results(
                volume_id=volume_id,
                segmentation_result=segmentation_result,
                proofreading_result=proofreading_result,
                output_path=output_path
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['volumes_processed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            
            # Update telemetry
            if self.telemetry:
                self.telemetry.update_metrics({
                    'volumes_processed': self.processing_stats['volumes_processed'],
                    'processing_time': processing_time,
                    'segmentation_confidence': segmentation_result.confidence_score,
                    'proofreading_quality': proofreading_result.quality_metrics.get('overall_quality', 0.0)
                })
            
            success = True
            logger.info(f"✓ Volume {volume_id} processed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            self.processing_stats['errors_encountered'] += 1
            logger.error(f"Failed to process volume {volume_id}: {e}")
            success = False
        
        return success
    
    def _save_results(self, 
                     volume_id: str,
                     segmentation_result,
                     proofreading_result,
                     output_path: str):
        """Save all pipeline results."""
        try:
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Save segmentation results
            segmentation_output = f"{output_path}/segmentation"
            self.ffn_plugin.processor._save_results(
                {
                    'segmentation': segmentation_result.segmentation,
                    'uncertainty': segmentation_result.uncertainty_map
                },
                segmentation_output
            )
            
            # Save proofreading results
            proofreading_output = f"{output_path}/proofreading"
            self.proofreader.save_results(proofreading_result, proofreading_output)
            
            # Save metadata
            metadata = {
                'volume_id': volume_id,
                'processing_timestamp': time.time(),
                'segmentation_confidence': segmentation_result.confidence_score,
                'proofreading_quality': proofreading_result.quality_metrics,
                'processing_stats': self.processing_stats
            }
            
            metadata_path = f"{output_path}/metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def run_batch_processing(self, volume_list: list, output_base_path: str) -> Dict[str, Any]:
        """Process multiple volumes in batch."""
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return {'success': False, 'error': 'Pipeline not initialized'}
        
        results = {
            'total_volumes': len(volume_list),
            'successful_volumes': 0,
            'failed_volumes': 0,
            'processing_times': [],
            'errors': []
        }
        
        logger.info(f"Starting batch processing of {len(volume_list)} volumes")
        
        for i, volume_id in enumerate(volume_list):
            logger.info(f"Processing volume {i+1}/{len(volume_list)}: {volume_id}")
            
            output_path = f"{output_base_path}/{volume_id}"
            start_time = time.time()
            
            success = self.process_volume(volume_id, output_path)
            
            processing_time = time.time() - start_time
            results['processing_times'].append(processing_time)
            
            if success:
                results['successful_volumes'] += 1
            else:
                results['failed_volumes'] += 1
                results['errors'].append(f"Volume {volume_id} failed")
        
        # Calculate summary statistics
        results['total_processing_time'] = sum(results['processing_times'])
        results['average_processing_time'] = results['total_processing_time'] / len(volume_list)
        results['success_rate'] = results['successful_volumes'] / len(volume_list)
        
        logger.info(f"Batch processing completed: {results['successful_volumes']}/{len(volume_list)} successful")
        logger.info(f"Total time: {results['total_processing_time']:.2f}s, Average: {results['average_processing_time']:.2f}s")
        
        return results
    
    def cleanup(self):
        """Clean up pipeline resources."""
        logger.info("Cleaning up pipeline resources...")
        
        if self.ffn_plugin and self.ffn_plugin.processor:
            self.ffn_plugin.processor.cleanup()
        
        if self.telemetry:
            self.telemetry.cleanup()
        
        logger.info("Pipeline cleanup completed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            'processing_stats': self.processing_stats,
            'is_initialized': self.is_initialized,
            'components': {
                'data_loader': self.data_loader is not None,
                'ffn_plugin': self.ffn_plugin is not None,
                'proofreader': self.proofreader is not None,
                'continual_learner': self.continual_learner is not None,
                'telemetry': self.telemetry is not None
            }
        }
        
        if self.ffn_plugin and self.ffn_plugin.processor:
            stats['ffn_stats'] = self.ffn_plugin.processor.get_statistics()
        
        if self.proofreader:
            stats['proofreader_stats'] = self.proofreader.get_statistics()
        
        return stats

def main():
    """Main entry point for the production pipeline."""
    parser = argparse.ArgumentParser(description="Agentic Tracer Production Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--output", required=True, help="Output base path")
    parser.add_argument("--volumes", nargs="+", help="List of volume IDs to process")
    parser.add_argument("--single-volume", help="Process a single volume")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and initialize pipeline
    pipeline = ProductionPipeline(args.config)
    
    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return 1
    
    try:
        if args.single_volume:
            # Process single volume
            success = pipeline.process_volume(args.single_volume, args.output)
            if not success:
                logger.error("Single volume processing failed")
                return 1
        elif args.volumes:
            # Process multiple volumes
            results = pipeline.run_batch_processing(args.volumes, args.output)
            if results['successful_volumes'] == 0:
                logger.error("All volumes failed to process")
                return 1
        else:
            logger.error("Must specify either --single-volume or --volumes")
            return 1
        
        # Print final statistics
        stats = pipeline.get_statistics()
        logger.info(f"Pipeline statistics: {stats}")
        
    finally:
        pipeline.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 