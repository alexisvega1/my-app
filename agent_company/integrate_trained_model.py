#!/usr/bin/env python3
"""
Integrate Trained FFN-v2 Model into Tracing Pipeline
===================================================
Integrate the trained FFN-v2 model into the neuron tracing pipeline
and test it on real H01 data.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Import our components
from production_ffn_v2 import ProductionFFNv2Model
from neuron_tracer_3d import NeuronTracer3D
from extract_h01_brain_regions import H01RegionExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainedModelIntegrator:
    """Integrate trained FFN-v2 model into the tracing pipeline."""
    
    def __init__(self, model_path: str = "best_ffn_v2_model.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing trained model integrator")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Device: {self.device}")
    
    def load_trained_model(self) -> bool:
        """Load the trained FFN-v2 model."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Initialize the model architecture
            self.model = ProductionFFNv2Model(
                input_channels=1,
                hidden_channels=[32, 64, 128, 256],
                output_channels=1,
                use_attention=True,
                dropout_rate=0.1
            )
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ Trained model loaded successfully")
            logger.info(f"  - Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"  - Training epoch: {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"  - Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load trained model: {e}")
            return False
    
    def segment_with_trained_model(self, volume: np.ndarray) -> np.ndarray:
        """Segment volume using the trained FFN-v2 model."""
        if self.model is None:
            logger.error("Model not loaded. Call load_trained_model() first.")
            return None
        
        try:
            logger.info(f"Segmenting volume with trained model...")
            logger.info(f"  Input shape: {volume.shape}")
            
            # Preprocess volume
            processed_volume = self._preprocess_volume(volume)
            
            # Run inference
            with torch.no_grad():
                # Process in chunks to handle memory
                segmentation = self._inference_in_chunks(processed_volume)
            
            logger.info(f"✓ Segmentation completed")
            logger.info(f"  Output shape: {segmentation.shape}")
            logger.info(f"  Unique labels: {len(np.unique(segmentation))}")
            
            return segmentation
            
        except Exception as e:
            logger.error(f"✗ Segmentation failed: {e}")
            return None
    
    def _preprocess_volume(self, volume: np.ndarray) -> torch.Tensor:
        """Preprocess volume for model input."""
        # Normalize to [0, 1]
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Add channel dimension and convert to tensor
        volume_tensor = torch.from_numpy(volume_norm).float()
        if len(volume_tensor.shape) == 3:
            volume_tensor = volume_tensor.unsqueeze(0)  # Add batch dimension
        
        return volume_tensor.to(self.device)
    
    def _inference_in_chunks(self, volume_tensor: torch.Tensor, chunk_size: int = 64) -> np.ndarray:
        """Run inference in chunks to handle large volumes."""
        original_shape = volume_tensor.shape
        logger.info(f"Running inference in chunks of size {chunk_size}")
        
        # Initialize output
        if len(original_shape) == 4:  # [B, C, H, W, D]
            output_shape = (original_shape[0], 1, *original_shape[2:])  # 1 output channel
        else:
            output_shape = (1, 1, *original_shape[1:])
        
        output = torch.zeros(output_shape, device=self.device)
        
        # Process in chunks
        for z in range(0, original_shape[-1], chunk_size):
            z_end = min(z + chunk_size, original_shape[-1])
            chunk = volume_tensor[..., z:z_end]
            
            # Run model on chunk
            with torch.no_grad():
                seg, _ = self.model(chunk)
            
            output[..., z:z_end] = seg
        
        # Convert to segmentation (threshold at 0.5)
        segmentation = (output > 0.5).long().squeeze(1)  # Remove channel dim
        return segmentation.cpu().numpy()
    
    def test_on_real_data(self, region_name: str = "prefrontal_cortex", size: str = "medium") -> Dict[str, Any]:
        """Test the trained model on real H01 data."""
        logger.info(f"Testing trained model on real H01 data")
        logger.info(f"Region: {region_name}, Size: {size}")
        
        try:
            # Extract real H01 data
            extractor = H01RegionExtractor()
            volume = extractor.extract_region(region_name, size)
            
            if volume is None:
                logger.error(f"Failed to extract region {region_name}")
                return None
            
            logger.info(f"Extracted volume shape: {volume.shape}")
            
            # Segment with trained model
            segmentation = self.segment_with_trained_model(volume)
            
            if segmentation is None:
                logger.error("Segmentation failed")
                return None
            
            # Run tracing on the segmentation
            logger.info("Running neuron tracing on segmentation...")
            tracer = NeuronTracer3D(segmentation_data=segmentation)
            tracer.analyze_connectivity(distance_threshold=10.0)
            
            # Export results
            results = {
                'region_name': region_name,
                'size': size,
                'volume_shape': list(volume.shape),
                'segmentation_shape': list(segmentation.shape),
                'num_neurons': len(tracer.traced_neurons),
                'num_components': int(np.max(segmentation)),
                'model_info': {
                    'model_path': str(self.model_path),
                    'device': str(self.device),
                    'parameters': sum(p.numel() for p in self.model.parameters())
                }
            }
            
            # Save results
            output_dir = Path("trained_model_test_results")
            output_dir.mkdir(exist_ok=True)
            
            # Save segmentation
            np.save(output_dir / "trained_segmentation.npy", segmentation)
            
            # Save traces
            tracer.export_traces(str(output_dir / "trained_traces.json"))
            
            # Save results summary
            with open(output_dir / "test_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✓ Test completed successfully")
            logger.info(f"  Results saved in: {output_dir}")
            logger.info(f"  Neurons detected: {results['num_neurons']}")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            return None
    
    def integrate_into_batch_processor(self) -> bool:
        """Update the batch processor to use the trained model."""
        try:
            logger.info("Integrating trained model into batch processor...")
            
            # Read the current batch processor
            batch_processor_path = Path("production_batch_processor.py")
            if not batch_processor_path.exists():
                logger.error("Batch processor not found")
                return False
            
            with open(batch_processor_path, 'r') as f:
                content = f.read()
            
            # Add import for trained model
            if "from production_ffn_v2 import ProductionFFNv2Model" not in content:
                import_line = "from production_ffn_v2 import ProductionFFNv2Model\n"
                content = content.replace("# Import our pipeline components", 
                                       "# Import our pipeline components\n" + import_line)
            
            # Update the segmentation creation method
            old_segmentation_method = """    def _create_segmentation(self, data: np.ndarray) -> np.ndarray:
        \"\"\"Create segmentation from raw data.\"\"\"
        # Remove extra dimension if present
        if len(data.shape) == 4:
            data = data.squeeze()
        
        # Use Otsu thresholding
        from skimage import filters
        try:
            threshold = filters.threshold_otsu(data)
        except:
            threshold = np.percentile(data[data > 0], 60)
        
        binary = (data > threshold).astype(np.uint8)
        
        # Label connected components
        from skimage import measure
        labeled = measure.label(binary)
        
        return labeled"""
            
            new_segmentation_method = """    def _create_segmentation(self, data: np.ndarray) -> np.ndarray:
        \"\"\"Create segmentation using trained FFN-v2 model.\"\"\"
        # Remove extra dimension if present
        if len(data.shape) == 4:
            data = data.squeeze()
        
        # Use trained model if available
        if hasattr(self, 'trained_model') and self.trained_model is not None:
            logger.info("Using trained FFN-v2 model for segmentation")
            return self.trained_model.segment_with_trained_model(data)
        else:
            # Fallback to traditional method
            logger.info("Using traditional segmentation method")
            from skimage import filters
            try:
                threshold = filters.threshold_otsu(data)
            except:
                threshold = np.percentile(data[data > 0], 60)
            
            binary = (data > threshold).astype(np.uint8)
            
            # Label connected components
            from skimage import measure
            labeled = measure.label(binary)
            
            return labeled"""
            
            content = content.replace(old_segmentation_method, new_segmentation_method)
            
            # Add model initialization to __init__
            init_pattern = "        # Initialize components"
            model_init = """        # Initialize trained model
        self.trained_model = None
        if Path("best_ffn_v2_model.pt").exists():
            try:
                from production_ffn_v2 import ProductionFFNv2Model
                self.trained_model = ProductionFFNv2Model(input_channels=1, num_classes=32)
                checkpoint = torch.load("best_ffn_v2_model.pt", map_location='cpu')
                self.trained_model.load_state_dict(checkpoint['model_state_dict'])
                self.trained_model.eval()
                logger.info("✓ Trained FFN-v2 model loaded")
            except Exception as e:
                logger.warning(f"Failed to load trained model: {e}")
        
        # Initialize components"""
            
            content = content.replace(init_pattern, model_init)
            
            # Save updated batch processor
            backup_path = batch_processor_path.with_suffix('.py.backup')
            batch_processor_path.rename(backup_path)
            
            with open(batch_processor_path, 'w') as f:
                f.write(content)
            
            logger.info(f"✓ Batch processor updated")
            logger.info(f"  Backup saved as: {backup_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to integrate into batch processor: {e}")
            return False

def main():
    """Main function for model integration."""
    print("Integrate Trained FFN-v2 Model")
    print("=" * 40)
    
    # Initialize integrator
    integrator = TrainedModelIntegrator()
    
    # Load trained model
    if not integrator.load_trained_model():
        print("Failed to load trained model. Exiting.")
        return
    
    # Test on real data
    print("\nTesting on real H01 data...")
    results = integrator.test_on_real_data()
    
    if results:
        print(f"\n✓ Test Results:")
        print(f"  Region: {results['region_name']}")
        print(f"  Volume shape: {results['volume_shape']}")
        print(f"  Neurons detected: {results['num_neurons']}")
        print(f"  Components: {results['num_components']}")
    
    # Integrate into batch processor
    print("\nIntegrating into batch processor...")
    if integrator.integrate_into_batch_processor():
        print("✓ Integration completed!")
        print("\nNext steps:")
        print("1. Run: python3 production_batch_processor.py")
        print("2. The batch processor will now use your trained model")
        print("3. Monitor progress with: python3 real_time_monitor.py")
    else:
        print("✗ Integration failed")

if __name__ == "__main__":
    main() 