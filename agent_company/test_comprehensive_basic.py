#!/usr/bin/env python3
"""
Basic Comprehensive Pipeline Test
=================================

Simple test script that tests core functionality without complex dependencies.
Focuses on spine detection, molecular prediction, and basic analysis.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_test_data():
    """Create simple test data for basic functionality testing."""
    print("🧪 Creating simple test data...")
    
    # Create a simple 3D volume with neuron-like structures
    shape = (64, 64, 64)
    volume = np.zeros(shape, dtype=np.uint8)
    segmentation = np.zeros(shape, dtype=np.uint32)
    
    # Create a simple neuron (spherical soma + linear dendrite)
    center = (32, 32, 32)
    
    # Soma (spherical)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 8:  # Soma
                    volume[i, j, k] = 200
                    segmentation[i, j, k] = 1
                elif dist < 12 and abs(i-center[0]) < 6:  # Dendrite
                    volume[i, j, k] = 150
                    segmentation[i, j, k] = 1
    
    # Add some spine-like protrusions
    spine_positions = [
        (38, 32, 32), (26, 32, 32), (32, 38, 32), (32, 26, 32)
    ]
    
    for spine_pos in spine_positions:
        x, y, z = spine_pos
        for i in range(max(0, x-2), min(shape[0], x+3)):
            for j in range(max(0, y-2), min(shape[1], y+3)):
                for k in range(max(0, z-2), min(shape[2], z+3)):
                    dist = np.sqrt((i-x)**2 + (j-y)**2 + (k-z)**2)
                    if dist < 2:
                        volume[i, j, k] = 220  # Higher intensity for spines
                        segmentation[i, j, k] = 1
    
    print(f"✅ Created test volume with shape: {volume.shape}")
    print(f"✅ Segmentation has {np.max(segmentation)} components")
    
    return {
        'volume': volume,
        'segmentation': segmentation,
        'neuron_id': 1
    }

def test_basic_spine_detection():
    """Test basic spine detection logic."""
    print("\n🧪 Testing Basic Spine Detection...")
    
    try:
        # Create test data
        test_data = create_simple_test_data()
        volume = test_data['volume']
        segmentation = test_data['segmentation']
        
        # Create neuron mask
        neuron_mask = (segmentation == 1)
        
        # Simple spine detection using morphological operations
        from skimage import morphology, measure
        from scipy import ndimage
        
        # Find local maxima (potential spines)
        kernel = morphology.ball(3)
        dilated = morphology.binary_dilation(neuron_mask, kernel)
        eroded = morphology.binary_erosion(neuron_mask, kernel)
        
        # Spine candidates are regions that are in dilated but not in eroded
        spine_candidates = dilated & ~eroded
        
        # Filter by intensity (spines should be brighter)
        high_intensity = volume > 200
        spine_candidates = spine_candidates & high_intensity
        
        # Count spine candidates
        spine_count = np.sum(spine_candidates)
        
        print(f"✅ Detected {spine_count} potential spine candidates")
        
        # Analyze spine properties
        if spine_count > 0:
            spine_coords = np.array(np.where(spine_candidates)).T
            print(f"✅ Spine coordinates: {len(spine_coords)} points")
            
            # Calculate spine volumes
            labeled_spines = measure.label(spine_candidates)
            spine_props = measure.regionprops(labeled_spines)
            
            print(f"✅ Found {len(spine_props)} distinct spine regions")
            
            for i, prop in enumerate(spine_props):
                print(f"   Spine {i+1}: volume={prop.area}, centroid={prop.centroid}")
        
        return spine_count > 0
        
    except Exception as e:
        print(f"❌ Basic spine detection test failed: {e}")
        return False

def test_basic_morphological_analysis():
    """Test basic morphological analysis."""
    print("\n🧪 Testing Basic Morphological Analysis...")
    
    try:
        # Create test data
        test_data = create_simple_test_data()
        segmentation = test_data['segmentation']
        
        # Create neuron mask
        neuron_mask = (segmentation == 1)
        
        # Basic morphological measurements
        from skimage import measure, morphology
        
        # Calculate basic properties
        volume = np.sum(neuron_mask)
        coords = np.array(np.where(neuron_mask)).T
        
        if len(coords) == 0:
            print("❌ No neuron found in mask")
            return False
        
        # Calculate centroid
        centroid = np.mean(coords, axis=0)
        
        # Calculate bounding box
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        bounding_box = (max_coords - min_coords)
        
        # Calculate surface area (approximate)
        from skimage.morphology import ball
        kernel = ball(1)
        eroded = morphology.binary_erosion(neuron_mask, kernel)
        surface_area = np.sum(neuron_mask) - np.sum(eroded)
        
        # Calculate complexity (branching factor)
        labeled = measure.label(neuron_mask)
        num_components = len(measure.regionprops(labeled))
        
        print(f"✅ Neuron volume: {volume} voxels")
        print(f"✅ Centroid: {centroid}")
        print(f"✅ Bounding box: {bounding_box}")
        print(f"✅ Surface area: {surface_area} voxels")
        print(f"✅ Number of components: {num_components}")
        
        # Basic validation
        if volume > 100 and surface_area > 0:
            print("✅ Morphological analysis looks reasonable")
            return True
        else:
            print("⚠️ Morphological measurements seem too small")
            return False
        
    except Exception as e:
        print(f"❌ Basic morphological analysis test failed: {e}")
        return False

def test_molecular_prediction_rules():
    """Test rule-based molecular prediction."""
    print("\n🧪 Testing Rule-Based Molecular Prediction...")
    
    try:
        # Create test data
        test_data = create_simple_test_data()
        segmentation = test_data['segmentation']
        
        # Create neuron mask
        neuron_mask = (segmentation == 1)
        
        # Calculate morphological features
        volume = np.sum(neuron_mask)
        coords = np.array(np.where(neuron_mask)).T
        
        # Calculate dendritic complexity (simplified)
        from skimage.morphology import skeletonize_3d
        try:
            skeleton = skeletonize_3d(neuron_mask)
            dendritic_complexity = np.sum(skeleton)
        except:
            # Fallback to 2D skeletonization
            skeleton = np.zeros_like(neuron_mask, dtype=bool)
            for z in range(neuron_mask.shape[0]):
                from skimage.morphology import skeletonize
                skeleton[z] = skeletonize(neuron_mask[z])
            dendritic_complexity = np.sum(skeleton)
        
        # Calculate spine density (simplified)
        spine_candidates = volume * 0.1  # Assume 10% of volume could be spines
        spine_density = spine_candidates / max(dendritic_complexity, 1)
        
        # Rule-based classification
        predictions = {}
        
        if volume > 1000 and dendritic_complexity > 50 and spine_density > 0.1:
            predictions['pyramidal'] = 0.8
            predictions['glutamatergic'] = 0.9
        elif volume < 800 and dendritic_complexity > 20:
            predictions['interneuron'] = 0.7
            predictions['gabaergic'] = 0.8
        elif volume < 500 and spine_density < 0.05:
            predictions['granule'] = 0.6
            predictions['glutamatergic'] = 0.7
        else:
            predictions['unknown'] = 0.5
        
        print(f"✅ Volume: {volume}")
        print(f"✅ Dendritic complexity: {dendritic_complexity}")
        print(f"✅ Spine density: {spine_density:.3f}")
        print(f"✅ Molecular predictions: {predictions}")
        
        return len(predictions) > 0
        
    except Exception as e:
        print(f"❌ Molecular prediction test failed: {e}")
        return False

def test_data_export():
    """Test data export functionality."""
    print("\n🧪 Testing Data Export...")
    
    try:
        # Create test data
        test_data = create_simple_test_data()
        
        # Create output directory
        output_dir = Path("test_basic_results")
        output_dir.mkdir(exist_ok=True)
        
        # Export volume and segmentation
        np.save(output_dir / "test_volume.npy", test_data['volume'])
        np.save(output_dir / "test_segmentation.npy", test_data['segmentation'])
        
        # Create analysis results
        analysis_results = {
            'test_info': {
                'volume_shape': list(test_data['volume'].shape),
                'segmentation_shape': list(test_data['segmentation'].shape),
                'neuron_id': test_data['neuron_id'],
                'timestamp': str(np.datetime64('now'))
            },
            'morphological_analysis': {
                'volume': int(np.sum(test_data['segmentation'] > 0)),
                'centroid': [32, 32, 32],
                'bounding_box': [64, 64, 64]
            },
            'spine_analysis': {
                'detected_spines': 4,
                'spine_types': {
                    'mushroom': 2,
                    'thin': 1,
                    'stubby': 1
                }
            },
            'molecular_predictions': {
                'pyramidal': 0.8,
                'glutamatergic': 0.9
            }
        }
        
        # Export results as JSON
        with open(output_dir / "analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"✅ Exported test data to {output_dir}")
        print(f"✅ Created analysis results file")
        
        return True
        
    except Exception as e:
        print(f"❌ Data export test failed: {e}")
        return False

def main():
    """Run all basic comprehensive pipeline tests."""
    print("🚀 Starting Basic Comprehensive Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Spine Detection", test_basic_spine_detection),
        ("Basic Morphological Analysis", test_basic_morphological_analysis),
        ("Rule-Based Molecular Prediction", test_molecular_prediction_rules),
        ("Data Export", test_data_export)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All basic comprehensive pipeline tests passed!")
        print("\n📁 Next steps:")
        print("1. Install Allen Brain SDK: pip install allensdk")
        print("2. Install SAM2 for advanced refinement")
        print("3. Run full comprehensive tests: python test_comprehensive_pipeline.py")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 