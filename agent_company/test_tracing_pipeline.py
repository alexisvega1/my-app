#!/usr/bin/env python3
"""
Test Neuron Tracing Pipeline on Synthetic Data
==============================================
Tests the full pipeline: segmentation -> tracing -> visualization -> export
"""

import numpy as np
import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from neuron_tracer_3d import NeuronTracer3D
from visualization import H01Visualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tracing_pipeline(input_file, output_dir="tracing_test_results"):
    """Test the full neuron tracing pipeline on a synthetic dataset."""
    
    print(f"\n{'='*60}")
    print(f"Testing Neuron Tracing Pipeline")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the synthetic data
    print(f"\n1. Loading data from {input_file}...")
    try:
        volume = np.load(input_file)
        print(f"   ✓ Loaded volume with shape: {volume.shape}")
        print(f"   ✓ Data range: {volume.min()} to {volume.max()}")
        print(f"   ✓ Non-zero voxels: {np.count_nonzero(volume)} / {volume.size}")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        return False
    
    # Convert volume to segmentation (threshold to create binary segmentation)
    print(f"\n2. Converting volume to segmentation...")
    try:
        # Threshold the volume to create a binary segmentation
        threshold = np.percentile(volume[volume > 0], 50)  # Use median of non-zero values
        segmentation = (volume > threshold).astype(np.uint8)
        
        # Label connected components
        from skimage import measure
        labeled_segmentation = measure.label(segmentation)
        
        print(f"   ✓ Created segmentation with {np.max(labeled_segmentation)} components")
        print(f"   ✓ Segmentation shape: {labeled_segmentation.shape}")
        
    except Exception as e:
        print(f"   ✗ Failed to create segmentation: {e}")
        return False
    
    # Initialize tracer
    print(f"\n3. Initializing 3D neuron tracer...")
    try:
        tracer = NeuronTracer3D(segmentation_data=labeled_segmentation)
        print(f"   ✓ Tracer initialized successfully")
        print(f"   ✓ Found {len(tracer.traced_neurons)} neurons")
    except Exception as e:
        print(f"   ✗ Failed to initialize tracer: {e}")
        return False
    
    # Analyze connectivity
    print(f"\n4. Analyzing connectivity...")
    try:
        tracer.analyze_connectivity(distance_threshold=10.0)
        print(f"   ✓ Connectivity analysis completed")
    except Exception as e:
        print(f"   ✗ Connectivity analysis failed: {e}")
        return False
    
    # Export traces
    print(f"\n5. Exporting traces...")
    try:
        # Export as JSON
        json_filename = os.path.join(output_dir, "traces.json")
        tracer.export_traces(json_filename)
        print(f"   ✓ Exported traces.json")
        
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return False
    
    # Save data for visualization
    print(f"\n6. Saving data for visualization...")
    try:
        # Save the segmentation
        seg_filename = os.path.join(output_dir, "segmentation.npy")
        np.save(seg_filename, labeled_segmentation)
        print(f"   ✓ Saved segmentation.npy")
        
        # Save the original volume
        vol_filename = os.path.join(output_dir, "volume.npy")
        np.save(vol_filename, volume)
        print(f"   ✓ Saved volume.npy")
        
        # Create metadata
        metadata = {
            "volume_shape": [int(x) for x in volume.shape],
            "segmentation_shape": [int(x) for x in labeled_segmentation.shape],
            "num_neurons": len(tracer.traced_neurons),
            "neuron_info": [
                {
                    "id": int(neuron_id),
                    "volume": int(neuron.volume),
                    "confidence": float(neuron.confidence),
                    "connections": int(len(neuron.connectivity))
                }
                for neuron_id, neuron in tracer.traced_neurons.items()
            ]
        }
        
        import json
        meta_filename = os.path.join(output_dir, "metadata.json")
        with open(meta_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Saved metadata.json")
        
    except Exception as e:
        print(f"   ✗ Data saving failed: {e}")
        return False
    
    # Test visualization
    print(f"\n7. Testing visualization...")
    try:
        # Initialize visualizer
        viz = H01Visualizer(output_dir)
        print(f"   ✓ Visualizer initialized")
        
        # Get available datasets
        datasets = viz.get_available_datasets()
        print(f"   ✓ Available datasets: {datasets}")
        
        # Create a simple visualization
        if "segmentation" in datasets:
            viz.create_2d_slice_viewer("segmentation", 
                                     save_path=os.path.join(output_dir, "slice_view.png"))
            print(f"   ✓ Created slice viewer")
        
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
        return False
    
    # Create comprehensive visualization
    print(f"\n8. Creating comprehensive visualization...")
    try:
        tracer.create_comprehensive_visualization(output_dir)
        print(f"   ✓ Created comprehensive visualization")
        
    except Exception as e:
        print(f"   ✗ Comprehensive visualization failed: {e}")
        return False
    
    print(f"\n{'='*60}")
    print(f"✓ Pipeline test completed successfully!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")
    
    return True

def main():
    """Test the pipeline on all synthetic datasets."""
    
    # Test datasets
    test_files = [
        "test_simple_neurons.npy",
        "test_complex_neurons.npy", 
        "test_dense_network.npy",
        "test_sparse_neurons.npy"
    ]
    
    results = {}
    
    for test_file in test_files:
        if os.path.exists(test_file):
            output_dir = f"tracing_test_{test_file.replace('.npy', '')}"
            success = test_tracing_pipeline(test_file, output_dir)
            results[test_file] = success
        else:
            print(f"⚠ Test file not found: {test_file}")
            results[test_file] = False
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PIPELINE TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_file, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_file}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The tracing pipeline is working correctly.")
    else:
        print("⚠ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 