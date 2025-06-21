#!/usr/bin/env python3
"""
H01 Data Access Verification Script
==================================
Test the updated H01 data loader with multiple access methods.
Based on Google FFN patterns and H01 release data.
"""

import os
import sys
import yaml
import logging
import time
import numpy as np
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from h01_data_loader import H01DataLoader, test_h01_connection

def setup_logging():
    """Setup logging for the verification script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('h01_verification.log')
        ]
    )

def test_dependencies():
    """Test if all required dependencies are available."""
    print("🔍 Testing H01 data access dependencies...")
    
    # Test basic connection
    connection_ok = test_h01_connection()
    if not connection_ok:
        print("❌ Basic connection test failed")
        return False
    
    print("✅ Basic connection test passed")
    return True

def test_data_loader_initialization():
    """Test H01 data loader initialization."""
    print("\n🔍 Testing H01 data loader initialization...")
    
    try:
        # Load configuration
        config_path = "h01_config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize data loader
        data_loader = H01DataLoader(config)
        
        print(f"✅ Data loader initialized successfully")
        print(f"   Connection method: {data_loader.connection_method}")
        print(f"   Cache directory: {data_loader.cache_dir}")
        
        return data_loader
        
    except Exception as e:
        print(f"❌ Data loader initialization failed: {e}")
        return None

def test_volume_info(data_loader):
    """Test volume information retrieval."""
    print("\n🔍 Testing volume information retrieval...")
    
    try:
        info = data_loader.get_volume_info()
        
        print("✅ Volume information retrieved successfully:")
        print(f"   Shape: {info['shape']}")
        print(f"   Dtype: {info['dtype']}")
        print(f"   Voxel size: {info['voxel_size']}")
        print(f"   Bounds: {info['bounds']}")
        print(f"   Connection type: {info['connection_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Volume information retrieval failed: {e}")
        return False

def test_chunk_loading(data_loader):
    """Test chunk loading functionality."""
    print("\n🔍 Testing chunk loading...")
    
    try:
        # Test coordinates and chunk size
        test_coords = (1000, 1000, 1000)
        test_size = (32, 32, 32)
        
        print(f"   Loading chunk at {test_coords} with size {test_size}...")
        start_time = time.time()
        
        chunk = data_loader.load_chunk(test_coords, test_size)
        load_time = time.time() - start_time
        
        print(f"✅ Chunk loaded successfully:")
        print(f"   Shape: {chunk.shape}")
        print(f"   Dtype: {chunk.dtype}")
        print(f"   Load time: {load_time:.2f}s")
        print(f"   Memory usage: {chunk.nbytes / 1024 / 1024:.2f} MB")
        
        # Test statistics
        stats = chunk.flatten()
        print(f"   Min value: {stats.min()}")
        print(f"   Max value: {stats.max()}")
        print(f"   Mean value: {stats.mean():.2f}")
        print(f"   Std value: {stats.std():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunk loading failed: {e}")
        return False

def test_region_access(data_loader):
    """Test region access functionality."""
    print("\n🔍 Testing region access...")
    
    try:
        # List available regions
        regions = data_loader.list_available_regions()
        print(f"✅ Found {len(regions)} available regions:")
        
        for region in regions:
            print(f"   - {region['name']}: {region['description']}")
            print(f"     Bounds: {region['bounds']}")
            print(f"     Size: {region['size_gb']:.3f} GB")
        
        # Test specific region
        if regions:
            test_region = regions[0]['name']
            region_info = data_loader.get_region(test_region)
            print(f"✅ Successfully accessed region: {test_region}")
        
        return True
        
    except Exception as e:
        print(f"❌ Region access failed: {e}")
        return False

def test_data_statistics(data_loader):
    """Test data statistics functionality."""
    print("\n🔍 Testing data statistics...")
    
    try:
        stats = data_loader.get_data_statistics()
        
        print("✅ Data statistics retrieved successfully:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value}")
            elif isinstance(value, list):
                print(f"   {key}: {value[:3]}...")  # Show first 3 elements
            else:
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data statistics failed: {e}")
        return False

def test_robustness(data_loader):
    """Test robustness features."""
    print("\n🔍 Testing robustness features...")
    
    try:
        # Test with different chunk sizes
        chunk_sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
        test_coords = (2000, 2000, 2000)
        
        for chunk_size in chunk_sizes:
            try:
                chunk = data_loader.load_chunk(test_coords, chunk_size)
                print(f"✅ Successfully loaded chunk size {chunk_size}")
            except Exception as e:
                print(f"❌ Failed to load chunk size {chunk_size}: {e}")
        
        # Test boundary conditions
        try:
            # Test near boundary
            boundary_coords = (9990, 9990, 9990)
            boundary_chunk = data_loader.load_chunk(boundary_coords, (10, 10, 10))
            print(f"✅ Successfully loaded boundary chunk")
        except Exception as e:
            print(f"❌ Boundary chunk loading failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Robustness testing failed: {e}")
        return False

def main():
    """Main verification function."""
    print("🚀 H01 Data Access Verification")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Dependency test failed. Please install required packages:")
        print("   pip install gcsfs cloudvolume pyyaml numpy")
        return False
    
    # Test data loader initialization
    data_loader = test_data_loader_initialization()
    if data_loader is None:
        return False
    
    # Run all tests
    tests = [
        ("Volume Info", test_volume_info),
        ("Chunk Loading", test_chunk_loading),
        ("Region Access", test_region_access),
        ("Data Statistics", test_data_statistics),
        ("Robustness", test_robustness)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func(data_loader):
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! H01 data access is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 