#!/usr/bin/env python3
"""
Enhanced Connectomics Pipeline
==============================
Production-ready pipeline integrating enhanced floodfilling algorithms,
improved FFN-v2 models, and advanced features for large-scale connectomics.
"""

import os
import sys
import logging
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
import yaml

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced components
from enhanced_floodfill_algorithm import (
    EnhancedFloodFillAlgorithm, 
    FloodFillConfig, 
    FloodFillResult
)
from enhanced_production_ffn_v2 import (
    EnhancedProductionFFNv2Model, 
    ModelConfig, 
    create_model_from_config
)

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the enhanced connectomics pipeline."""
    # Data configuration
    input_volume_path: str
    output_dir: str
    seed_points: List[Tuple[int, int, int]]
    
    # Model configuration
    model_config: ModelConfig
    
    # Flood-filling configuration
    floodfill_config: FloodFillConfig
    
    # Processing configuration
    batch_size: int = 1
    num_workers: int = 4
    use_gpu: bool = True
    mixed_precision: bool = True
    
    # Quality control
    min_segment_size: int = 100
    max_segments_per_volume: int = 1000
    quality_threshold: float = 0.7
    
    # Output configuration
    save_intermediate: bool = True
    save_visualizations: bool = True
    output_format: str = "h5"  # "h5", "npy", "zarr"
    
    # Monitoring
    enable_monitoring: bool = True
    log_interval: int = 100
    save_checkpoints: bool = True

class EnhancedConnectomicsPipeline:
    """Enhanced connectomics pipeline with advanced features."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.model = None
        self.floodfill_algorithm = None
        self.results = []
        
        # Performance tracking
        self.stats = {
            'total_processing_time': 0.0,
            'segments_processed': 0,
            'successful_segments': 0,
            'failed_segments': 0,
            'total_voxels_segmented': 0
        }
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        logger.info(f"Enhanced Connectomics Pipeline initialized on device: {self.device}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the enhanced FFN-v2 model."""
        try:
            logger.info("Loading enhanced FFN-v2 model...")
            
            # Create model from configuration
            self.model = EnhancedProductionFFNv2Model(self.config.model_config)
            self.model.to(self.device)
            
            # Load pretrained weights if provided
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Model weights loaded from {model_path}")
            else:
                logger.info("No pretrained weights provided - using random initialization")
            
            self.model.eval()
            
            # Initialize flood-filling algorithm
            self.floodfill_algorithm = EnhancedFloodFillAlgorithm(self.config.floodfill_config)
            
            logger.info("Model and flood-filling algorithm loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_volume(self, volume_path: str) -> Optional[np.ndarray]:
        """Load volume data with memory optimization."""
        try:
            logger.info(f"Loading volume from {volume_path}")
            
            # Check file format and load accordingly
            if volume_path.endswith('.npy'):
                volume = np.load(volume_path, mmap_mode='r')
            elif volume_path.endswith('.h5') or volume_path.endswith('.hdf5'):
                import h5py
                with h5py.File(volume_path, 'r') as f:
                    # Assume first dataset is the volume
                    dataset_name = list(f.keys())[0]
                    volume = f[dataset_name][:]
            elif volume_path.endswith('.zarr'):
                import zarr
                volume = zarr.open(volume_path, mode='r')[:]
            else:
                # Try to load as numpy array
                volume = np.load(volume_path)
            
            logger.info(f"Volume loaded with shape: {volume.shape}, dtype: {volume.dtype}")
            return volume
            
        except Exception as e:
            logger.error(f"Failed to load volume: {e}")
            return None
    
    def process_segment(self, volume: np.ndarray, seed_point: Tuple[int, int, int]) -> Optional[FloodFillResult]:
        """Process a single segment using enhanced flood-filling."""
        try:
            logger.info(f"Processing segment from seed point: {seed_point}")
            
            # Run enhanced flood-filling
            result = self.floodfill_algorithm.flood_fill(volume, self.model, seed_point)
            
            # Quality check
            if result.quality_metrics['total_voxels'] < self.config.min_segment_size:
                logger.warning(f"Segment too small ({result.quality_metrics['total_voxels']} voxels)")
                return None
            
            if result.quality_metrics['mean_uncertainty'] > (1.0 - self.config.quality_threshold):
                logger.warning(f"Segment quality too low (uncertainty: {result.quality_metrics['mean_uncertainty']:.3f})")
                return None
            
            logger.info(f"Segment processed successfully: {result.quality_metrics['total_voxels']} voxels, "
                       f"quality: {1.0 - result.quality_metrics['mean_uncertainty']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process segment from {seed_point}: {e}")
            return None
    
    def process_volume(self) -> bool:
        """Process the entire volume with all seed points."""
        start_time = time.time()
        
        try:
            # Load volume
            volume = self.load_volume(self.config.input_volume_path)
            if volume is None:
                return False
            
            # Process each seed point
            for i, seed_point in enumerate(self.config.seed_points):
                if self.stats['successful_segments'] >= self.config.max_segments_per_volume:
                    logger.info("Maximum number of segments reached")
                    break
                
                logger.info(f"Processing segment {i+1}/{len(self.config.seed_points)}")
                
                # Process segment
                result = self.process_segment(volume, seed_point)
                
                if result is not None:
                    self.results.append(result)
                    self.stats['successful_segments'] += 1
                    self.stats['total_voxels_segmented'] += result.quality_metrics['total_voxels']
                    
                    # Save individual segment
                    self._save_segment(result, i, seed_point)
                else:
                    self.stats['failed_segments'] += 1
                
                self.stats['segments_processed'] += 1
                
                # Log progress
                if (i + 1) % self.config.log_interval == 0:
                    self._log_progress(i + 1, len(self.config.seed_points))
            
            # Save combined results
            self._save_combined_results(volume)
            
            # Update final statistics
            self.stats['total_processing_time'] = time.time() - start_time
            
            logger.info("Volume processing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Volume processing failed: {e}")
            return False
    
    def _save_segment(self, result: FloodFillResult, segment_id: int, seed_point: Tuple[int, int, int]):
        """Save individual segment results."""
        segment_dir = os.path.join(self.config.output_dir, f"segment_{segment_id:04d}")
        os.makedirs(segment_dir, exist_ok=True)
        
        # Save segmentation
        if self.config.output_format == "npy":
            np.save(os.path.join(segment_dir, "segmentation.npy"), result.segmentation)
            np.save(os.path.join(segment_dir, "uncertainty.npy"), result.uncertainty_map)
            np.save(os.path.join(segment_dir, "confidence.npy"), result.confidence_scores)
        elif self.config.output_format == "h5":
            import h5py
            with h5py.File(os.path.join(segment_dir, "segment.h5"), 'w') as f:
                f.create_dataset('segmentation', data=result.segmentation)
                f.create_dataset('uncertainty', data=result.uncertainty_map)
                f.create_dataset('confidence', data=result.confidence_scores)
                f.attrs['seed_point'] = seed_point
                f.attrs['processing_time'] = result.processing_time
                f.attrs['iterations'] = result.iterations
                f.attrs['quality_metrics'] = json.dumps(result.quality_metrics)
        
        # Save metadata
        metadata = {
            'segment_id': segment_id,
            'seed_point': seed_point,
            'processing_time': result.processing_time,
            'iterations': result.iterations,
            'quality_metrics': result.quality_metrics,
            'memory_usage': result.memory_usage,
            'metadata': result.metadata
        }
        
        with open(os.path.join(segment_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save visualization if enabled
        if self.config.save_visualizations:
            self._save_visualization(result, segment_dir)
    
    def _save_combined_results(self, volume: np.ndarray):
        """Save combined results from all segments."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        logger.info("Saving combined results...")
        
        # Combine all segmentations
        combined_segmentation = np.zeros_like(volume, dtype=np.uint32)
        combined_uncertainty = np.zeros_like(volume, dtype=np.float32)
        combined_confidence = np.zeros_like(volume, dtype=np.float32)
        
        for i, result in enumerate(self.results):
            # Assign unique label to each segment
            segment_label = i + 1
            combined_segmentation[result.segmentation > 0] = segment_label
            
            # Combine uncertainty and confidence (take maximum)
            combined_uncertainty = np.maximum(combined_uncertainty, result.uncertainty_map)
            combined_confidence = np.maximum(combined_confidence, result.confidence_scores)
        
        # Save combined results
        if self.config.output_format == "npy":
            np.save(os.path.join(self.config.output_dir, "combined_segmentation.npy"), combined_segmentation)
            np.save(os.path.join(self.config.output_dir, "combined_uncertainty.npy"), combined_uncertainty)
            np.save(os.path.join(self.config.output_dir, "combined_confidence.npy"), combined_confidence)
        elif self.config.output_format == "h5":
            import h5py
            with h5py.File(os.path.join(self.config.output_dir, "combined_results.h5"), 'w') as f:
                f.create_dataset('segmentation', data=combined_segmentation)
                f.create_dataset('uncertainty', data=combined_uncertainty)
                f.create_dataset('confidence', data=combined_confidence)
                f.attrs['num_segments'] = len(self.results)
                f.attrs['total_voxels'] = np.sum(combined_segmentation > 0)
        
        # Save pipeline statistics
        pipeline_stats = {
            'pipeline_stats': self.stats,
            'segment_stats': [
                {
                    'segment_id': i,
                    'voxels': result.quality_metrics['total_voxels'],
                    'quality': 1.0 - result.quality_metrics['mean_uncertainty'],
                    'processing_time': result.processing_time
                }
                for i, result in enumerate(self.results)
            ]
        }
        
        with open(os.path.join(self.config.output_dir, "pipeline_stats.json"), 'w') as f:
            json.dump(pipeline_stats, f, indent=2)
    
    def _save_visualization(self, result: FloodFillResult, output_dir: str):
        """Save visualization of segment results."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import ImageGrid
            
            # Create visualization
            fig = plt.figure(figsize=(15, 5))
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.1)
            
            # Find center slice
            center_z = result.segmentation.shape[0] // 2
            center_y = result.segmentation.shape[1] // 2
            center_x = result.segmentation.shape[2] // 2
            
            # Plot segmentation
            grid[0].imshow(result.segmentation[center_z, :, :], cmap='gray')
            grid[0].set_title('Segmentation')
            grid[0].axis('off')
            
            # Plot uncertainty
            grid[1].imshow(result.uncertainty_map[center_z, :, :], cmap='hot')
            grid[1].set_title('Uncertainty')
            grid[1].axis('off')
            
            # Plot confidence
            grid[2].imshow(result.confidence_scores[center_z, :, :], cmap='viridis')
            grid[2].set_title('Confidence')
            grid[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualization.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available - skipping visualization")
        except Exception as e:
            logger.warning(f"Failed to create visualization: {e}")
    
    def _log_progress(self, current: int, total: int):
        """Log processing progress."""
        progress = (current / total) * 100
        logger.info(f"Progress: {current}/{total} ({progress:.1f}%) - "
                   f"Successful: {self.stats['successful_segments']}, "
                   f"Failed: {self.stats['failed_segments']}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            **self.stats,
            'average_processing_time_per_segment': (
                self.stats['total_processing_time'] / self.stats['segments_processed']
                if self.stats['segments_processed'] > 0 else 0.0
            ),
            'success_rate': (
                self.stats['successful_segments'] / self.stats['segments_processed']
                if self.stats['segments_processed'] > 0 else 0.0
            ),
            'average_voxels_per_segment': (
                self.stats['total_voxels_segmented'] / self.stats['successful_segments']
                if self.stats['successful_segments'] > 0 else 0
            ),
            'floodfill_stats': self.floodfill_algorithm.get_statistics() if self.floodfill_algorithm else {}
        }

def create_config_from_yaml(config_path: str) -> PipelineConfig:
    """Create pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Parse configurations
    model_config = ModelConfig(**config_dict.get('model', {}))
    floodfill_config = FloodFillConfig(**config_dict.get('floodfill', {}))
    
    # Create pipeline config
    pipeline_config = PipelineConfig(
        input_volume_path=config_dict['input_volume_path'],
        output_dir=config_dict['output_dir'],
        seed_points=config_dict['seed_points'],
        model_config=model_config,
        floodfill_config=floodfill_config,
        **{k: v for k, v in config_dict.items() 
           if k not in ['model', 'floodfill', 'input_volume_path', 'output_dir', 'seed_points']}
    )
    
    return pipeline_config

def main():
    """Main entry point for the enhanced connectomics pipeline."""
    parser = argparse.ArgumentParser(description="Enhanced Connectomics Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--model_path", type=str, help="Path to pretrained model weights")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        config = create_config_from_yaml(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config.output_dir = args.output_dir
        
        # Create pipeline
        pipeline = EnhancedConnectomicsPipeline(config)
        
        # Load model
        if not pipeline.load_model(args.model_path):
            logger.error("Failed to load model")
            return 1
        
        # Process volume
        if not pipeline.process_volume():
            logger.error("Failed to process volume")
            return 1
        
        # Print final statistics
        stats = pipeline.get_statistics()
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final statistics: {json.dumps(stats, indent=2)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 