#!/usr/bin/env python3
"""
Test Real H01 Data Processing
=============================
Test the neuron tracing pipeline on real H01 data extracted from the cloud.
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

def test_real_h01_data(input_file, output_dir="h01_real_data_results"):
    """Test the pipeline on real H01 data."""
    
    print(f"\n{'='*60}")
    print(f"Testing Real H01 Data Processing")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the real H01 data
    print(f"\n1. Loading real H01 data from {input_file}...")
    try:
        volume = np.load(input_file)
        print(f"   ✓ Loaded volume with shape: {volume.shape}")
        print(f"   ✓ Data range: {volume.min()} to {volume.max()}")
        print(f"   ✓ Non-zero voxels: {np.count_nonzero(volume)} / {volume.size}")
        print(f"   ✓ Data density: {(np.count_nonzero(volume) / volume.size) * 100:.2f}%")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        return False
    
    # Convert volume to segmentation
    print(f"\n2. Creating segmentation from real H01 data...")
    try:
        # Use a more sophisticated thresholding for real EM data
        # Remove the 4th dimension if present (CloudVolume sometimes adds it)
        if len(volume.shape) == 4:
            volume = volume.squeeze()
            print(f"   ✓ Removed extra dimension, new shape: {volume.shape}")
        
        # Calculate threshold using Otsu method or percentile
        from skimage import filters
        try:
            threshold = filters.threshold_otsu(volume)
            print(f"   ✓ Otsu threshold: {threshold}")
        except:
            # Fallback to percentile
            threshold = np.percentile(volume[volume > 0], 60)
            print(f"   ✓ Percentile threshold: {threshold}")
        
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
        
        # Print some neuron statistics
        if tracer.traced_neurons:
            volumes = [neuron.volume for neuron in tracer.traced_neurons.values()]
            print(f"   ✓ Neuron volumes: min={min(volumes)}, max={max(volumes)}, avg={np.mean(volumes):.1f}")
        
    except Exception as e:
        print(f"   ✗ Failed to initialize tracer: {e}")
        return False
    
    # Analyze connectivity
    print(f"\n4. Analyzing connectivity...")
    try:
        tracer.analyze_connectivity(distance_threshold=10.0)
        print(f"   ✓ Connectivity analysis completed")
        
        # Count connections
        total_connections = sum(len(neuron.connectivity) for neuron in tracer.traced_neurons.values())
        print(f"   ✓ Total connections: {total_connections}")
        
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
            "data_source": "H01_real_data",
            "volume_shape": list(volume.shape),
            "volume_size_mb": volume.nbytes / (1024 * 1024),
            "segmentation_shape": list(labeled_segmentation.shape),
            "num_components": int(np.max(labeled_segmentation)),
            "threshold_used": float(threshold),
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
    print(f"✓ Real H01 data processing completed successfully!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")
    
    return True

def main():
    """Test the pipeline on real H01 data."""
    
    # Look for real H01 data files
    h01_files = [
        "h01_data_region_400000_400000_4000.npy",  # The best region we found
        "h01_prefrontal_cortex_medium.npy",
        "h01_hippocampus_medium.npy", 
        "h01_visual_cortex_medium.npy"
    ]
    
    results = {}
    
    for h01_file in h01_files:
        if os.path.exists(h01_file):
            print(f"\n{'='*60}")
            print(f"Processing: {h01_file}")
            print(f"{'='*60}")
            
            output_dir = f"h01_real_{h01_file.replace('.npy', '')}"
            success = test_real_h01_data(h01_file, output_dir)
            results[h01_file] = success
        else:
            print(f"⚠ H01 file not found: {h01_file}")
            results[h01_file] = False
    
    # Summary
    print(f"\n{'='*60}")
    print(f"REAL H01 DATA PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    for h01_file, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {h01_file}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed > 0:
        print("🎉 Successfully processed real H01 data!")
        print("This confirms the pipeline works with actual connectomics data.")
    else:
        print("⚠ No real H01 data was successfully processed.")

if __name__ == "__main__":
    main() 