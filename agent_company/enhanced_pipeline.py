"""
Enhanced Connectomics Pipeline
==============================

A modular, production-ready pipeline for connectomics analysis with:
- Centralized configuration management
- Advanced data loading with caching
- State-of-the-art training with monitoring
- Comprehensive error handling
- Performance optimization
"""

import torch
import torch.nn as nn
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Import our enhanced modules
from config import load_config, PipelineConfig
from data_loader import H01DataLoader, create_data_loader
from training import AdvancedTrainer, create_trainer
from ffn_v2_mathematical_model import MathematicalFFNv2

# SAM 2 imports (if available)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    warnings.warn("SAM2 not available. Refinement will be disabled.")

logger = logging.getLogger(__name__)

class EnhancedConnectomicsPipeline:
    """
    Enhanced connectomics pipeline with modular design and production features.
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "development"):
        """
        Initialize the enhanced pipeline.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (development, production, colab)
        """
        # Load configuration
        self.config = load_config(config_path, environment)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.data_loader = None
        self.model = None
        self.trainer = None
        self.sam_predictor = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Enhanced pipeline initialized on device: {self.device}")
        logger.info(f"Environment: {environment}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Logging is already configured in the config module
        logger.info("Logging setup complete")
    
    def setup_data_loader(self) -> bool:
        """
        Setup the data loader with error handling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up data loader...")
            
            # Initialize H01 data loader
            self.data_loader = H01DataLoader(self.config)
            
            # Test data loading
            test_coords = self.data_loader.get_random_valid_coords(tuple(self.config.data.chunk_size))
            test_data = self.data_loader.load_chunk(test_coords, tuple(self.config.data.chunk_size))
            
            logger.info(f"Data loader test successful. Loaded chunk of shape: {test_data.shape}")
            logger.info(f"Cache stats: {self.data_loader.get_cache_stats()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup data loader: {e}")
            return False
    
    def setup_model(self) -> bool:
        """
        Setup the neural network model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up model...")
            
            # Create model
            self.model = MathematicalFFNv2(
                input_channels=self.config.model.input_channels,
                output_channels=self.config.model.output_channels,
                hidden_channels=self.config.model.hidden_channels,
                depth=self.config.model.depth
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model created successfully")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            return False
    
    def setup_trainer(self) -> bool:
        """
        Setup the trainer with advanced features.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up trainer...")
            
            if self.model is None:
                raise RuntimeError("Model must be setup before trainer")
            
            # Create trainer
            self.trainer = create_trainer(self.model, self.config, self.device)
            
            logger.info("Trainer setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            return False
    
    def setup_sam_refinement(self) -> bool:
        """
        Setup SAM2 for refinement (if available).
        
        Returns:
            True if successful, False otherwise
        """
        if not SAM2_AVAILABLE:
            logger.warning("SAM2 not available, skipping refinement setup")
            return False
        
        try:
            logger.info("Setting up SAM2 refinement...")
            
            # Load SAM2 model
            sam2_model = build_sam2(
                encoder_patch_embed_dim=96,
                encoder_num_heads=3,
                encoder_window_size=7,
                encoder_depth=2,
                encoder_global_attn_indexes=[],
                checkpoint="checkpoints/sam2_hiera_t.pt"
            )
            
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            self.sam_predictor.set_image(None)  # Will be set during inference
            
            logger.info("SAM2 refinement setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup SAM2 refinement: {e}")
            return False
    
    def train_model(self, train_samples: int = 1000, val_samples: int = 200) -> bool:
        """
        Train the model with comprehensive monitoring.
        
        Args:
            train_samples: Number of training samples
            val_samples: Number of validation samples
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting model training...")
            
            if self.trainer is None:
                raise RuntimeError("Trainer must be setup before training")
            
            # Create data loaders
            train_loader = create_data_loader(self.config, "dataset")
            val_loader = create_data_loader(self.config, "dataset")  # For validation
            
            # Train the model
            self.trainer.train(train_loader, val_loader)
            
            # Plot training history
            history_plot_path = Path(self.config.monitoring.checkpoint_dir) / "training_history.png"
            self.trainer.plot_training_history(str(history_plot_path))
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return False
    
    def run_inference(self, region_coords: Optional[tuple] = None, 
                     region_size: Optional[tuple] = None) -> Optional[torch.Tensor]:
        """
        Run inference on a region.
        
        Args:
            region_coords: Coordinates of the region (x, y, z)
            region_size: Size of the region (dx, dy, dz)
            
        Returns:
            Segmentation result or None if failed
        """
        try:
            logger.info("Running inference...")
            
            if self.model is None:
                raise RuntimeError("Model must be setup before inference")
            
            if self.data_loader is None:
                raise RuntimeError("Data loader must be setup before inference")
            
            # Get region coordinates if not provided
            if region_coords is None or region_size is None:
                region_size = tuple(self.config.data.chunk_size)
                region_coords = self.data_loader.get_random_valid_coords(region_size)
            
            # Load data
            data = self.data_loader.load_chunk(region_coords, region_size)
            data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)
            
            # Normalize data
            data_tensor = (data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min() + 1e-8)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(data_tensor)
            
            logger.info(f"Inference completed. Output shape: {output.shape}")
            return output
            
        except Exception as e:
            logger.error(f"Failed to run inference: {e}")
            return None
    
    def refine_with_sam(self, image: torch.Tensor, 
                       segmentation: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Refine segmentation using SAM2.
        
        Args:
            image: Input image tensor
            segmentation: Initial segmentation
            
        Returns:
            Refined segmentation or None if failed
        """
        if not SAM2_AVAILABLE or self.sam_predictor is None:
            logger.warning("SAM2 not available for refinement")
            return segmentation
        
        try:
            logger.info("Running SAM2 refinement...")
            
            # Convert tensors to numpy for SAM2
            image_np = image.squeeze().cpu().numpy()
            seg_np = segmentation.squeeze().cpu().numpy()
            
            # Setup SAM2 predictor
            self.sam_predictor.set_image(image_np)
            
            # Get refinement points from segmentation
            # This is a simplified approach - in practice you'd use more sophisticated point selection
            points = self._extract_points_from_segmentation(seg_np)
            
            # Run SAM2 refinement
            refined_masks = []
            for point in points:
                mask = self.sam_predictor.predict(point_coords=point, point_labels=[1])
                refined_masks.append(mask)
            
            # Combine refined masks
            refined_seg = self._combine_masks(refined_masks, seg_np.shape)
            
            logger.info("SAM2 refinement completed")
            return torch.from_numpy(refined_seg).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Failed to refine with SAM2: {e}")
            return segmentation
    
    def _extract_points_from_segmentation(self, segmentation: np.ndarray) -> list:
        """Extract points from segmentation for SAM2 refinement."""
        # Simplified point extraction - in practice you'd use more sophisticated methods
        points = []
        
        # Find centroids of connected components
        from scipy import ndimage
        labeled, num_features = ndimage.label(segmentation > 0.5)
        
        for i in range(1, num_features + 1):
            component = (labeled == i)
            centroid = ndimage.center_of_mass(component)
            if centroid[0] is not None:  # Check if centroid is valid
                points.append([int(centroid[1]), int(centroid[0])])  # SAM2 expects [x, y]
        
        return points[:10]  # Limit to 10 points for efficiency
    
    def _combine_masks(self, masks: list, shape: tuple) -> np.ndarray:
        """Combine multiple SAM2 masks into a single segmentation."""
        combined = np.zeros(shape, dtype=bool)
        
        for mask in masks:
            if mask is not None:
                combined = combined | mask
        
        return combined.astype(np.float32)
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete pipeline from setup to inference.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting complete pipeline execution...")
            
            # Setup phase
            if not self.setup_data_loader():
                logger.error("Failed to setup data loader")
                return False
            
            if not self.setup_model():
                logger.error("Failed to setup model")
                return False
            
            if not self.setup_trainer():
                logger.error("Failed to setup trainer")
                return False
            
            # Optional: Setup SAM2 refinement
            self.setup_sam_refinement()
            
            # Training phase
            if not self.train_model():
                logger.error("Failed to train model")
                return False
            
            # Inference phase
            result = self.run_inference()
            if result is None:
                logger.error("Failed to run inference")
                return False
            
            logger.info("Complete pipeline execution successful")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False
    
    def save_pipeline_state(self, filepath: str):
        """Save the complete pipeline state."""
        try:
            state = {
                'config': self.config.to_dict(),
                'model_state': self.model.state_dict() if self.model else None,
                'data_loader_cache_stats': self.data_loader.get_cache_stats() if self.data_loader else None,
                'training_history': self.trainer.training_history if self.trainer else None
            }
            
            torch.save(state, filepath)
            logger.info(f"Pipeline state saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline state: {e}")
    
    def load_pipeline_state(self, filepath: str):
        """Load the complete pipeline state."""
        try:
            state = torch.load(filepath, map_location=self.device)
            
            # Load configuration
            self.config = PipelineConfig.from_dict(state['config'])
            
            # Load model state
            if state['model_state'] and self.model:
                self.model.load_state_dict(state['model_state'])
            
            # Load training history
            if state['training_history'] and self.trainer:
                self.trainer.training_history = state['training_history']
            
            logger.info(f"Pipeline state loaded from: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline state: {e}")


def main():
    """Main function to run the enhanced pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Connectomics Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--environment", type=str, default="development", 
                       choices=["development", "production", "colab"],
                       help="Environment to run in")
    parser.add_argument("--mode", type=str, default="complete",
                       choices=["setup", "train", "inference", "complete"],
                       help="Pipeline mode to run")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = EnhancedConnectomicsPipeline(args.config, args.environment)
    
    # Run based on mode
    if args.mode == "setup":
        success = (pipeline.setup_data_loader() and 
                  pipeline.setup_model() and 
                  pipeline.setup_trainer())
    elif args.mode == "train":
        pipeline.setup_data_loader()
        pipeline.setup_model()
        pipeline.setup_trainer()
        success = pipeline.train_model()
    elif args.mode == "inference":
        pipeline.setup_data_loader()
        pipeline.setup_model()
        result = pipeline.run_inference()
        success = result is not None
    else:  # complete
        success = pipeline.run_complete_pipeline()
    
    if success:
        logger.info("Pipeline execution completed successfully")
    else:
        logger.error("Pipeline execution failed")
        exit(1)


if __name__ == "__main__":
    main() 