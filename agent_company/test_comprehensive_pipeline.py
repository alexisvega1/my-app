#!/usr/bin/env python3
"""
Test Comprehensive Pipeline
===========================

Test script for the enhanced comprehensive pipeline including:
- Spine detection and classification
- Molecular identity prediction
- Allen Brain SDK integration
- Complete neuron analysis workflow
"""

import numpy as np
import torch
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import pipeline components
try:
    from comprehensive_neuron_analyzer import ComprehensiveNeuronAnalyzer, SpineDetector, MolecularIdentityPredictor
    COMPREHENSIVE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Comprehensive analyzer not available: {e}")
    COMPREHENSIVE_AVAILABLE = False

# Import Allen Brain SDK
try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    from allensdk.api.queries.cell_types_api import CellTypesApi
    ALLEN_SDK_AVAILABLE = True
except ImportError:
    ALLEN_SDK_AVAILABLE = False
    print("Warning: Allen Brain SDK not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_neuron_data(shape: tuple = (128, 128, 128)) -> Dict[str, np.ndarray]:
    """
    Create realistic test neuron data with spines and dendrites.
    
    Args:
        shape: Shape of the test volume
        
    Returns:
        Dictionary containing test data
    """
    print("🧪 Creating realistic test neuron data...")
    
    # Create base volume
    volume = np.zeros(shape, dtype=np.uint8)
    segmentation = np.zeros(shape, dtype=np.uint32)
    
    # Create multiple neurons with different morphologies
    neurons = []
    
    # Neuron 1: Pyramidal-like neuron with spines
    center1 = (64, 64, 64)
    neuron1_mask = create_pyramidal_neuron(center1, volume, segmentation, neuron_id=1)
    neurons.append({'id': 1, 'type': 'pyramidal', 'mask': neuron1_mask})
    
    # Neuron 2: Interneuron-like neuron
    center2 = (32, 32, 32)
    neuron2_mask = create_interneuron(center2, volume, segmentation, neuron_id=2)
    neurons.append({'id': 2, 'type': 'interneuron', 'mask': neuron2_mask})
    
    # Neuron 3: Granule-like neuron
    center3 = (96, 96, 96)
    neuron3_mask = create_granule_neuron(center3, volume, segmentation, neuron_id=3)
    neurons.append({'id': 3, 'type': 'granule', 'mask': neuron3_mask})
    
    print(f"✅ Created {len(neurons)} test neurons")
    return {
        'volume': volume,
        'segmentation': segmentation,
        'neurons': neurons
    }

def create_pyramidal_neuron(center: tuple, volume: np.ndarray, segmentation: np.ndarray, neuron_id: int) -> np.ndarray:
    """Create a pyramidal-like neuron with dendritic spines."""
    x, y, z = center
    
    # Create soma (large spherical region)
    for i in range(max(0, x-8), min(volume.shape[0], x+9)):
        for j in range(max(0, y-8), min(volume.shape[1], y+9)):
            for k in range(max(0, z-8), min(volume.shape[2], z+9)):
                dist = np.sqrt((i-x)**2 + (j-y)**2 + (k-z)**2)
                if dist < 8:
                    volume[i, j, k] = 200
                    segmentation[i, j, k] = neuron_id
    
    # Create apical dendrite (main trunk)
    for i in range(z+8, min(volume.shape[2], z+40)):
        for j in range(max(0, y-3), min(volume.shape[1], y+4)):
            for k in range(max(0, x-3), min(volume.shape[2], x+4)):
                dist = np.sqrt((j-y)**2 + (k-x)**2)
                if dist < 3:
                    volume[i, j, k] = 180
                    segmentation[i, j, k] = neuron_id
    
    # Create basal dendrites
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.radians(angle)
        for length in range(20, 35):
            dx = int(length * np.cos(rad))
            dy = int(length * np.sin(rad))
            new_x, new_y = x + dx, y + dy
            
            if (0 <= new_x < volume.shape[0] and 0 <= new_y < volume.shape[1] and 0 <= z < volume.shape[2]):
                # Create dendrite branch
                for i in range(max(0, new_x-2), min(volume.shape[0], new_x+3)):
                    for j in range(max(0, new_y-2), min(volume.shape[1], new_y+3)):
                        for k in range(max(0, z-2), min(volume.shape[2], z+3)):
                            dist = np.sqrt((i-new_x)**2 + (j-new_y)**2 + (k-z)**2)
                            if dist < 2:
                                volume[i, j, k] = 160
                                segmentation[i, j, k] = neuron_id
                
                # Add spines to dendrites
                if length > 15:  # Add spines to distal parts
                    for spine_idx in range(3):
                        spine_angle = np.radians(angle + np.random.uniform(-30, 30))
                        spine_length = np.random.randint(3, 8)
                        spine_dx = int(spine_length * np.cos(spine_angle))
                        spine_dy = int(spine_length * np.sin(spine_angle))
                        spine_x, spine_y = new_x + spine_dx, new_y + spine_dy
                        
                        if (0 <= spine_x < volume.shape[0] and 0 <= spine_y < volume.shape[1] and 0 <= z < volume.shape[2]):
                            # Create spine head
                            for i in range(max(0, spine_x-2), min(volume.shape[0], spine_x+3)):
                                for j in range(max(0, spine_y-2), min(volume.shape[1], spine_y+3)):
                                    for k in range(max(0, z-2), min(volume.shape[2], z+3)):
                                        dist = np.sqrt((i-spine_x)**2 + (j-spine_y)**2 + (k-z)**2)
                                        if dist < 2:
                                            volume[i, j, k] = 220  # Higher intensity for spines
                                            segmentation[i, j, k] = neuron_id
    
    return segmentation == neuron_id

def create_interneuron(center: tuple, volume: np.ndarray, segmentation: np.ndarray, neuron_id: int) -> np.ndarray:
    """Create an interneuron-like neuron with fewer spines."""
    x, y, z = center
    
    # Create smaller soma
    for i in range(max(0, x-5), min(volume.shape[0], x+6)):
        for j in range(max(0, y-5), min(volume.shape[1], y+6)):
            for k in range(max(0, z-5), min(volume.shape[2], z+6)):
                dist = np.sqrt((i-x)**2 + (j-y)**2 + (k-z)**2)
                if dist < 5:
                    volume[i, j, k] = 180
                    segmentation[i, j, k] = neuron_id
    
    # Create dendrites (fewer and shorter)
    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        for length in range(15, 25):
            dx = int(length * np.cos(rad))
            dy = int(length * np.sin(rad))
            new_x, new_y = x + dx, y + dy
            
            if (0 <= new_x < volume.shape[0] and 0 <= new_y < volume.shape[1] and 0 <= z < volume.shape[2]):
                for i in range(max(0, new_x-2), min(volume.shape[0], new_x+3)):
                    for j in range(max(0, new_y-2), min(volume.shape[1], new_y+3)):
                        for k in range(max(0, z-2), min(volume.shape[2], z+3)):
                            dist = np.sqrt((i-new_x)**2 + (j-new_y)**2 + (k-z)**2)
                            if dist < 2:
                                volume[i, j, k] = 150
                                segmentation[i, j, k] = neuron_id
    
    return segmentation == neuron_id

def create_granule_neuron(center: tuple, volume: np.ndarray, segmentation: np.ndarray, neuron_id: int) -> np.ndarray:
    """Create a granule-like neuron with minimal dendritic arborization."""
    x, y, z = center
    
    # Create very small soma
    for i in range(max(0, x-3), min(volume.shape[0], x+4)):
        for j in range(max(0, y-3), min(volume.shape[1], y+4)):
            for k in range(max(0, z-3), min(volume.shape[2], z+4)):
                dist = np.sqrt((i-x)**2 + (j-y)**2 + (k-z)**2)
                if dist < 3:
                    volume[i, j, k] = 160
                    segmentation[i, j, k] = neuron_id
    
    # Create minimal dendrites
    for angle in [0, 180]:
        rad = np.radians(angle)
        for length in range(8, 12):
            dx = int(length * np.cos(rad))
            dy = int(length * np.sin(rad))
            new_x, new_y = x + dx, y + dy
            
            if (0 <= new_x < volume.shape[0] and 0 <= new_y < volume.shape[1] and 0 <= z < volume.shape[2]):
                for i in range(max(0, new_x-1), min(volume.shape[0], new_x+2)):
                    for j in range(max(0, new_y-1), min(volume.shape[1], new_y+2)):
                        for k in range(max(0, z-1), min(volume.shape[2], z+2)):
                            dist = np.sqrt((i-new_x)**2 + (j-new_y)**2 + (k-z)**2)
                            if dist < 1:
                                volume[i, j, k] = 140
                                segmentation[i, j, k] = neuron_id
    
    return segmentation == neuron_id

def test_spine_detection():
    """Test spine detection functionality."""
    print("\n🧪 Testing Spine Detection...")
    
    if not COMPREHENSIVE_AVAILABLE:
        print("❌ Comprehensive analyzer not available")
        return False
    
    try:
        # Create test data
        test_data = create_test_neuron_data()
        
        # Create spine detector
        config = {
            'min_spine_volume': 30,
            'max_spine_volume': 1500,
            'spine_detection_threshold': 0.6
        }
        
        spine_detector = SpineDetector(config)
        
        # Test spine detection on pyramidal neuron
        pyramidal_mask = test_data['segmentation'] == 1
        
        # Create skeleton for spine detection
        from skimage.morphology import skeletonize_3d
        try:
            skeleton = skeletonize_3d(pyramidal_mask)
        except:
            # Fallback to 2D skeletonization
            skeleton = np.zeros_like(pyramidal_mask, dtype=bool)
            for z in range(pyramidal_mask.shape[0]):
                from skimage.morphology import skeletonize
                skeleton[z] = skeletonize(pyramidal_mask[z])
        
        # Detect spines
        spines = spine_detector.detect_spines(pyramidal_mask, skeleton)
        
        print(f"✅ Detected {len(spines)} spines")
        
        # Analyze spine types
        spine_types = {}
        for spine in spines:
            spine_types[spine.spine_type] = spine_types.get(spine.spine_type, 0) + 1
        
        print(f"✅ Spine type distribution: {spine_types}")
        
        return len(spines) > 0
        
    except Exception as e:
        print(f"❌ Spine detection test failed: {e}")
        return False

def test_molecular_prediction():
    """Test molecular identity prediction."""
    print("\n🧪 Testing Molecular Identity Prediction...")
    
    if not COMPREHENSIVE_AVAILABLE:
        print("❌ Comprehensive analyzer not available")
        return False
    
    try:
        # Create molecular predictor
        config = {
            'use_allen_brain_sdk': ALLEN_SDK_AVAILABLE
        }
        
        molecular_predictor = MolecularIdentityPredictor(config)
        
        # Create test training data
        training_data = [
            {
                'soma_volume': 1500,
                'dendritic_length': 2000,
                'dendritic_complexity': 0.8,
                'branching_factor': 15,
                'spine_density': 0.2,
                'spine_types': {'mushroom': 10, 'thin': 15, 'stubby': 5},
                'molecular_type': 'pyramidal'
            },
            {
                'soma_volume': 600,
                'dendritic_length': 800,
                'dendritic_complexity': 0.4,
                'branching_factor': 8,
                'spine_density': 0.05,
                'spine_types': {'thin': 3, 'stubby': 1},
                'molecular_type': 'interneuron'
            },
            {
                'soma_volume': 300,
                'dendritic_length': 400,
                'dendritic_complexity': 0.2,
                'branching_factor': 3,
                'spine_density': 0.02,
                'spine_types': {'stubby': 1},
                'molecular_type': 'granule'
            }
        ]
        
        # Train classifier
        molecular_predictor.train_classifier(training_data)
        
        # Test prediction
        from comprehensive_neuron_analyzer import NeuronMorphology
        
        test_morphology = NeuronMorphology(
            neuron_id=1,
            soma_volume=1200,
            soma_position=np.array([64, 64, 64]),
            dendritic_length=1800,
            dendritic_complexity=0.7,
            branching_factor=12,
            spine_density=0.15,
            spine_types={'mushroom': 8, 'thin': 12, 'stubby': 4},
            morphological_type='unknown',
            molecular_markers={},
            confidence=0.8
        )
        
        predictions = molecular_predictor.predict_molecular_identity(test_morphology)
        
        print(f"✅ Molecular predictions: {predictions}")
        
        # Check if predictions are reasonable
        if 'pyramidal' in predictions and predictions['pyramidal'] > 0.5:
            print("✅ Pyramidal prediction looks reasonable")
            return True
        else:
            print("⚠️ Predictions may need adjustment")
            return False
        
    except Exception as e:
        print(f"❌ Molecular prediction test failed: {e}")
        return False

def test_allen_brain_integration():
    """Test Allen Brain SDK integration."""
    print("\n🧪 Testing Allen Brain SDK Integration...")
    
    if not ALLEN_SDK_AVAILABLE:
        print("⚠️ Allen Brain SDK not available, skipping test")
        return True
    
    try:
        # Test basic Allen Brain SDK functionality
        cell_types_api = CellTypesApi()
        
        # Get some basic cell data
        cells = cell_types_api.list_cells()
        
        if cells and len(cells) > 0:
            print(f"✅ Successfully connected to Allen Brain SDK")
            print(f"✅ Retrieved {len(cells)} cell records")
            return True
        else:
            print("⚠️ No cell data retrieved from Allen Brain SDK")
            return False
        
    except Exception as e:
        print(f"❌ Allen Brain SDK test failed: {e}")
        return False

def test_comprehensive_analysis():
    """Test complete comprehensive analysis workflow."""
    print("\n🧪 Testing Comprehensive Analysis Workflow...")
    
    if not COMPREHENSIVE_AVAILABLE:
        print("❌ Comprehensive analyzer not available")
        return False
    
    try:
        # Create test data
        test_data = create_test_neuron_data()
        
        # Create comprehensive analyzer
        config = {
            'spine_detection': {
                'min_spine_volume': 30,
                'max_spine_volume': 1500,
                'spine_detection_threshold': 0.6
            },
            'molecular_prediction': {
                'use_allen_brain_sdk': ALLEN_SDK_AVAILABLE
            }
        }
        
        analyzer = ComprehensiveNeuronAnalyzer(config)
        
        # Analyze each neuron
        results = []
        for neuron_info in test_data['neurons']:
            neuron_id = neuron_info['id']
            neuron_mask = test_data['segmentation'] == neuron_id
            
            print(f"  Analyzing neuron {neuron_id} ({neuron_info['type']})...")
            
            analysis = analyzer.analyze_neuron(neuron_mask, neuron_id)
            results.append(analysis)
            
            print(f"    Spines: {len(analysis['spines'])}")
            print(f"    Synapses: {len(analysis['synapses'])}")
            print(f"    Confidence: {analysis['analysis_confidence']:.2f}")
        
        # Save results
        output_dir = Path("test_comprehensive_results")
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / "comprehensive_analysis_test.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Comprehensive analysis completed")
        print(f"✅ Results saved to {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive analysis test failed: {e}")
        return False

def main():
    """Run all comprehensive pipeline tests."""
    print("🚀 Starting Comprehensive Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Spine Detection", test_spine_detection),
        ("Molecular Prediction", test_molecular_prediction),
        ("Allen Brain Integration", test_allen_brain_integration),
        ("Comprehensive Analysis", test_comprehensive_analysis)
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
        print("🎉 All comprehensive pipeline tests passed!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 