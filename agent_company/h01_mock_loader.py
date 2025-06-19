#!/usr/bin/env python3
"""
Mock H01 Dataset Data Loader for Testing
========================================
Simulates H01 dataset structure for testing without requiring Google Cloud authentication.
"""

import os
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class MockH01DataLoader:
    """Mock H01 data loader for testing without authentication."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_source_config = config.get('data_source', {})
        self.cloudvolume_config = self.data_source_config.get('cloudvolume', {})
        
        # Mock volume properties based on H01
        self.bounds = self.cloudvolume_config.get('bounds', [[0, 0, 0], [1000, 1000, 1000]])
        self.voxel_size = [4, 4, 33]  # 4nm x 4nm x 33nm as per H01
        self.dtype = np.uint8
        
        # Calculate mock volume shape
        start, end = self.bounds
        self.volume_shape = [end[i] - start[i] for i in range(3)]
        
        # Cache directory
        cache_config = self.config.get('data_access', {}).get('caching', {})
        if cache_config.get('enabled', True):
            self.cache_dir = cache_config.get('cache_dir', './h01_mock_cache')
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Mock H01 cache directory: {self.cache_dir}")
        else:
            self.cache_dir = None
        
        logger.info(f"Mock H01 data loader initialized with bounds: {self.bounds}")
        logger.info(f"Mock volume shape: {self.volume_shape}")
    
    def get_volume_info(self) -> Dict[str, Any]:
        """Get information about the mock H01 volume."""
        start, end = self.bounds
        
        return {
            'cloud_path': self.cloudvolume_config.get('cloud_path', 'gs://h01-release/data/20210601/4nm_raw'),
            'shape': self.volume_shape,
            'dtype': str(self.dtype),
            'voxel_size': self.voxel_size,
            'mip': 0,
            'bounds': self.bounds,
            'region_shape': [end[i] - start[i] for i in range(3)],
            'region_size_gb': self._calculate_region_size(),
            'cache_enabled': self.cache_dir is not None,
            'cache_dir': self.cache_dir,
            'mock': True
        }
    
    def _calculate_region_size(self) -> float:
        """Calculate the size of the current region in GB."""
        start, end = self.bounds
        region_shape = [end[i] - start[i] for i in range(3)]
        voxel_size = self.dtype().itemsize
        total_bytes = np.prod(region_shape) * voxel_size
        return total_bytes / (1024**3)
    
    def load_chunk(self, 
                  coordinates: Tuple[int, int, int],
                  chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load a mock chunk from the H01 volume."""
        try:
            z, y, x = coordinates
            z_end = min(z + chunk_size[0], self.volume_shape[0])
            y_end = min(y + chunk_size[1], self.volume_shape[1])
            x_end = min(x + chunk_size[2], self.volume_shape[2])
            
            # Generate mock data that simulates H01 EM data
            # H01 data has characteristic features like:
            # - Neuronal membranes (dark lines)
            # - Synapses (dense regions)
            # - Mitochondria (oval structures)
            # - Vesicles (small circular structures)
            
            chunk_shape = [z_end - z, y_end - y, x_end - x]
            chunk = self._generate_mock_h01_data(chunk_shape)
            
            # Pad if necessary
            if chunk.shape != chunk_size:
                padded_chunk = np.zeros(chunk_size, dtype=self.dtype)
                padded_chunk[:z_end-z, :y_end-y, :x_end-x] = chunk
                return padded_chunk
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to load mock chunk at {coordinates}: {e}")
            raise
    
    def _generate_mock_h01_data(self, shape: List[int]) -> np.ndarray:
        """Generate mock H01-like EM data."""
        # Start with noise
        data = np.random.normal(128, 30, shape).astype(self.dtype)
        
        # Add neuronal membranes (dark lines)
        for i in range(0, shape[0], 20):
            data[i:i+2, :, :] = np.clip(data[i:i+2, :, :] - 50, 0, 255)
        
        for j in range(0, shape[1], 15):
            data[:, j:j+2, :] = np.clip(data[:, j:j+2, :] - 50, 0, 255)
        
        for k in range(0, shape[2], 25):
            data[:, :, k:k+2] = np.clip(data[:, :, k:k+2] - 50, 0, 255)
        
        # Add synapses (dense regions)
        for _ in range(shape[0] // 50):
            center_z = np.random.randint(0, shape[0])
            center_y = np.random.randint(0, shape[1])
            center_x = np.random.randint(0, shape[2])
            radius = np.random.randint(5, 15)
            
            z_range = slice(max(0, center_z-radius), min(shape[0], center_z+radius))
            y_range = slice(max(0, center_y-radius), min(shape[1], center_y+radius))
            x_range = slice(max(0, center_x-radius), min(shape[2], center_x+radius))
            
            data[z_range, y_range, x_range] = np.clip(
                data[z_range, y_range, x_range] + 30, 0, 255
            )
        
        # Add mitochondria (oval structures)
        for _ in range(shape[0] // 100):
            center_z = np.random.randint(0, shape[0])
            center_y = np.random.randint(0, shape[1])
            center_x = np.random.randint(0, shape[2])
            
            # Create oval shape
            for dz in range(-10, 11):
                for dy in range(-8, 9):
                    for dx in range(-8, 9):
                        if (dz**2/100 + dy**2/64 + dx**2/64) <= 1:
                            z, y, x = center_z + dz, center_y + dy, center_x + dx
                            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                                data[z, y, x] = np.clip(data[z, y, x] + 20, 0, 255)
        
        return data
    
    def get_region(self, region_name: str) -> Dict[str, Any]:
        """Get a predefined region from the mock H01 dataset."""
        regions = self.config.get('h01_regions', {})
        
        if region_name in regions:
            region_config = regions[region_name]
            bounds = region_config.get('bounds', self.bounds)
            description = region_config.get('description', '')
            
            return {
                'name': region_name,
                'bounds': bounds,
                'description': description,
                'size_gb': self._calculate_region_size_for_bounds(bounds),
                'mock': True
            }
        else:
            raise ValueError(f"Region '{region_name}' not found in configuration")
    
    def _calculate_region_size_for_bounds(self, bounds: List[List[int]]) -> float:
        """Calculate size for given bounds."""
        start, end = bounds
        region_shape = [end[i] - start[i] for i in range(3)]
        voxel_size = self.dtype().itemsize
        total_bytes = np.prod(region_shape) * voxel_size
        return total_bytes / (1024**3)
    
    def list_available_regions(self) -> List[Dict[str, Any]]:
        """List all available regions in the mock H01 configuration."""
        regions = self.config.get('h01_regions', {})
        available_regions = []
        
        for name, config in regions.items():
            if isinstance(config, dict) and 'bounds' in config:
                region_info = self.get_region(name)
                available_regions.append(region_info)
        
        return available_regions
    
    def validate_data_source(self) -> bool:
        """Validate that the mock H01 data source is accessible."""
        try:
            info = self.get_volume_info()
            logger.info(f"Mock H01 data source validation successful: {info}")
            return True
        except Exception as e:
            logger.error(f"Mock H01 data source validation failed: {e}")
            return False
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the mock H01 data."""
        try:
            info = self.get_volume_info()
            regions = self.list_available_regions()
            
            return {
                'volume_info': info,
                'available_regions': regions,
                'total_regions': len(regions),
                'cache_status': {
                    'enabled': self.cache_dir is not None,
                    'directory': self.cache_dir,
                    'exists': os.path.exists(self.cache_dir) if self.cache_dir else False
                },
                'connection_status': 'connected (mock)',
                'mock': True
            }
        except Exception as e:
            logger.error(f"Failed to get data statistics: {e}")
            return {'error': str(e)}

def create_mock_h01_data_loader(config_path: str) -> MockH01DataLoader:
    """Create a mock H01 data loader from configuration file."""
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Mock H01 configuration loaded from {config_path}")
        return MockH01DataLoader(config)
    except Exception as e:
        logger.error(f"Failed to load mock H01 configuration from {config_path}: {e}")
        raise

def test_mock_h01_connection():
    """Test function to verify mock H01 data access."""
    try:
        # Test with a small region
        test_config = {
            'data_source': {
                'cloudvolume': {
                    'cloud_path': 'gs://h01-release/data/20210601/4nm_raw',
                    'mip': 0,
                    'bounds': [[0, 0, 0], [100, 100, 100]]  # Small test region
                }
            },
            'data_access': {
                'caching': {
                    'enabled': True,
                    'cache_dir': './h01_mock_test_cache'
                }
            },
            'h01_regions': {
                'test_region': {
                    'bounds': [[0, 0, 0], [100, 100, 100]],
                    'description': 'Small test region'
                }
            }
        }
        
        loader = MockH01DataLoader(test_config)
        
        if loader.validate_data_source():
            info = loader.get_volume_info()
            logger.info(f"Mock H01 connection test successful: {info}")
            
            # Test loading a chunk
            chunk = loader.load_chunk((0, 0, 0), (32, 32, 32))
            logger.info(f"Successfully loaded mock chunk with shape: {chunk.shape}")
            
            return True
        else:
            logger.error("Mock H01 connection test failed")
            return False
            
    except Exception as e:
        logger.error(f"Mock H01 connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the mock H01 data loader
    logging.basicConfig(level=logging.INFO)
    success = test_mock_h01_connection()
    print(f"Mock H01 connection test: {'SUCCESS' if success else 'FAILED'}") 