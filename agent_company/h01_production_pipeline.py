#!/usr/bin/env python3
"""
H01 Production Pipeline for Agentic Tracer
==========================================
End-to-end production pipeline for processing H01 connectomics data.
Based on https://h01-release.storage.googleapis.com/data.html
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

from h01_data_loader import H01DataLoader, create_h01_data_loader
from segmenters.ffn_v2_advanced import AdvancedFFNv2Plugin
from proofreading_advanced import AdvancedProofreader
from continual_learning_advanced import AdvancedContinualLearner
from telemetry import TelemetrySystem

logger = logging.getLogger(__name__)

class H01ProductionPipeline:
    """Production pipeline for H01 connectomics data processing."""
    
    def __init__(self, config_path: str):
        """Initialize the H01 production pipeline."""
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
            'errors_encountered': 0,
            'h01_regions_processed': []
        }
        
        logger.info("H01 production pipeline initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"H01 configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load H01 configuration: {e}")
            raise
    
    def initialize(self) -> bool:
        """Initialize all pipeline components for H01 processing."""
        try:
            logger.info("Initializing H01 production pipeline components...")
            
            # Initialize H01 data loader (real)
            self.data_loader = create_h01_data_loader(self.config_path)
            if not self.data_loader.validate_data_source():
                logger.error("H01 data source validation failed")
                return False
            
            # Get H01 data statistics
            stats = self.data_loader.get_data_statistics()
            logger.info(f"H01 data statistics: {stats}")
            
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
                # Prepare config for advanced continual learner
                advanced_cl_config = {
                    'model_config': cl_config.get('lora_config', {}),
                    'training_config': cl_config.get('training_config', {}),
                    'memory_config': cl_config.get('memory_config', {}),
                    'distributed_config': cl_config.get('distributed_config', {})
                }
                self.continual_learner = AdvancedContinualLearner(advanced_cl_config)
            
            # Initialize telemetry
            telemetry_config = self.config.get('telemetry', {})
            if telemetry_config.get('prometheus', {}).get('enabled', True):
                self.telemetry = TelemetrySystem(
                    port=telemetry_config['prometheus'].get('port', 8000),
                    enable_prometheus=True,
                    enable_system_metrics=telemetry_config.get('system_monitoring', {}).get('enabled', True),
                    metrics_prefix="h01_agentic_tracer"
                )
                self.telemetry.start_metrics_server()
                if telemetry_config.get('system_monitoring', {}).get('enabled', True):
                    self.telemetry.start_system_monitoring()
            
            self.is_initialized = True
            logger.info("✓ All H01 pipeline components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize H01 pipeline: {e}")
            return False
    
    def process_h01_region(self, region_name: str, output_path: str) -> bool:
        """Process a specific H01 region through the complete pipeline."""
        if not self.is_initialized:
            logger.error("H01 pipeline not initialized")
            return False
        
        start_time = time.time()
        success = False
        
        try:
            logger.info(f"Starting processing of H01 region: {region_name}")
            
            # Get region information
            region_info = self.data_loader.get_region(region_name)
            logger.info(f"H01 region info: {region_info}")
            
            # Get volume information
            volume_info = self.data_loader.get_volume_info()
            logger.info(f"H01 volume info: {volume_info}")
            
            # Step 1: Segmentation
            logger.info("Step 1: Performing H01 segmentation...")
            segmentation_result = self.ffn_plugin.segment(
                volume_path=region_name,  # Use region name as identifier
                output_path=f"{output_path}_segmentation"
            )
            
            if segmentation_result is None:
                logger.error("H01 segmentation failed")
                return False
            
            # Step 2: Proofreading
            logger.info("Step 2: Performing H01 proofreading...")
            proofreading_result = self.proofreader.proofread(
                segmentation=segmentation_result.segmentation,
                uncertainty_map=segmentation_result.uncertainty_map,
                metadata={
                    'region_name': region_name,
                    'region_bounds': region_info['bounds'],
                    'volume_shape': volume_info['shape'],
                    'voxel_size': volume_info['voxel_size'],
                    'segmentation_confidence': segmentation_result.confidence_score,
                    'h01_dataset': True
                }
            )
            
            # Step 3: Continual Learning (if enabled)
            if self.continual_learner is not None:
                logger.info("Step 3: Performing H01 continual learning...")
                learning_result = self.continual_learner.train(
                    segmentation=proofreading_result.corrected_segmentation,
                    uncertainty_map=segmentation_result.uncertainty_map,
                    metadata={
                        'region_name': region_name,
                        'proofreading_quality': proofreading_result.quality_metrics,
                        'h01_dataset': True
                    }
                )
                
                if learning_result:
                    logger.info("✓ H01 continual learning completed successfully")
            
            # Step 4: Save results
            logger.info("Step 4: Saving H01 results...")
            self._save_h01_results(
                region_name=region_name,
                region_info=region_info,
                segmentation_result=segmentation_result,
                proofreading_result=proofreading_result,
                output_path=output_path
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['volumes_processed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            self.processing_stats['h01_regions_processed'].append(region_name)
            
            # Update telemetry
            if self.telemetry:
                self.telemetry.update_metrics({
                    'volumes_processed': self.processing_stats['volumes_processed'],
                    'processing_time': processing_time,
                    'segmentation_confidence': segmentation_result.confidence_score,
                    'proofreading_quality': proofreading_result.quality_metrics.get('overall_quality', 0.0),
                    'h01_region_size_gb': region_info['size_gb']
                })
            
            success = True
            logger.info(f"✓ H01 region {region_name} processed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            self.processing_stats['errors_encountered'] += 1
            logger.error(f"Failed to process H01 region {region_name}: {e}")
            success = False
        
        return success
    
    def _save_h01_results(self, 
                         region_name: str,
                         region_info: Dict[str, Any],
                         segmentation_result,
                         proofreading_result,
                         output_path: str):
        """Save all H01 pipeline results."""
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
            
            # Save H01-specific metadata
            metadata = {
                'region_name': region_name,
                'region_info': region_info,
                'processing_timestamp': time.time(),
                'segmentation_confidence': segmentation_result.confidence_score,
                'proofreading_quality': proofreading_result.quality_metrics,
                'processing_stats': self.processing_stats,
                'h01_dataset': True,
                'voxel_size': [4, 4, 33],  # H01 specific
                'resolution': '4nm'
            }
            
            metadata_path = f"{output_path}/h01_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"H01 results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save H01 results: {e}")
            raise
    
    def process_all_h01_regions(self, output_base_path: str) -> Dict[str, Any]:
        """Process all available H01 regions."""
        if not self.is_initialized:
            logger.error("H01 pipeline not initialized")
            return {'success': False, 'error': 'Pipeline not initialized'}
        
        # Get all available regions
        available_regions = self.data_loader.list_available_regions()
        logger.info(f"Found {len(available_regions)} H01 regions to process")
        
        results = {
            'total_regions': len(available_regions),
            'successful_regions': 0,
            'failed_regions': 0,
            'processing_times': [],
            'errors': [],
            'region_results': {}
        }
        
        for region_info in available_regions:
            region_name = region_info['name']
            logger.info(f"Processing H01 region {region_name}: {region_info['description']}")
            
            output_path = f"{output_base_path}/{region_name}"
            start_time = time.time()
            
            success = self.process_h01_region(region_name, output_path)
            
            processing_time = time.time() - start_time
            results['processing_times'].append(processing_time)
            results['region_results'][region_name] = {
                'success': success,
                'processing_time': processing_time,
                'region_info': region_info
            }
            
            if success:
                results['successful_regions'] += 1
            else:
                results['failed_regions'] += 1
                results['errors'].append(f"Region {region_name} failed")
        
        # Calculate summary statistics
        results['total_processing_time'] = sum(results['processing_times'])
        results['average_processing_time'] = results['total_processing_time'] / len(available_regions)
        results['success_rate'] = results['successful_regions'] / len(available_regions)
        
        logger.info(f"H01 batch processing completed: {results['successful_regions']}/{len(available_regions)} successful")
        logger.info(f"Total time: {results['total_processing_time']:.2f}s, Average: {results['average_processing_time']:.2f}s")
        
        return results
    
    def cleanup(self):
        """Clean up H01 pipeline resources."""
        logger.info("Cleaning up H01 pipeline resources...")
        
        if self.ffn_plugin and self.ffn_plugin.processor:
            self.ffn_plugin.processor.cleanup()
        
        if self.telemetry:
            self.telemetry.cleanup()
        
        logger.info("H01 pipeline cleanup completed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get H01 pipeline statistics."""
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
        
        if self.data_loader:
            stats['h01_data_stats'] = self.data_loader.get_data_statistics()
        
        if self.ffn_plugin and self.ffn_plugin.processor:
            stats['ffn_stats'] = self.ffn_plugin.processor.get_statistics()
        
        if self.proofreader:
            stats['proofreader_stats'] = self.proofreader.get_statistics()
        
        return stats

def main():
    """Main entry point for the H01 production pipeline."""
    parser = argparse.ArgumentParser(description="H01 Agentic Tracer Production Pipeline")
    parser.add_argument("--config", required=True, help="Path to H01 configuration file")
    parser.add_argument("--output", required=True, help="Output base path")
    parser.add_argument("--region", help="Process a specific H01 region")
    parser.add_argument("--all-regions", action="store_true", help="Process all available H01 regions")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and initialize H01 pipeline
    pipeline = H01ProductionPipeline(args.config)
    
    if not pipeline.initialize():
        logger.error("Failed to initialize H01 pipeline")
        return 1
    
    try:
        if args.region:
            # Process single region
            success = pipeline.process_h01_region(args.region, args.output)
            if not success:
                logger.error("Single H01 region processing failed")
                return 1
        elif args.all_regions:
            # Process all regions
            results = pipeline.process_all_h01_regions(args.output)
            if results['successful_regions'] == 0:
                logger.error("All H01 regions failed to process")
                return 1
        else:
            logger.error("Must specify either --region or --all-regions")
            return 1
        
        # Print final statistics
        stats = pipeline.get_statistics()
        logger.info(f"H01 pipeline statistics: {stats}")
        
    finally:
        pipeline.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 