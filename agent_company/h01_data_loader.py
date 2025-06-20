#!/usr/bin/env python3
"""
H01 Dataset Data Loader for Agentic Tracer
==========================================
Specialized data loader for the H01 connectomics dataset.
Based on https://h01-release.storage.googleapis.com/data.html
"""

import os
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# CloudVolume for H01 data access
try:
    from cloudvolume import CloudVolume
    CLOUDVOLUME_AVAILABLE = True
    logger.info("CloudVolume available for H01 data access")
except ImportError:
    CLOUDVOLUME_AVAILABLE = False
    logger.error("CloudVolume not available - H01 data access disabled")

class H01DataLoader:
    """Specialized data loader for the H01 connectomics dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_source_config = config.get('data_source', {})
        self.cloudvolume_config = self.data_source_config.get('cloudvolume', {})
        
        # Initialize CloudVolume connection
        self.volume = None
        self.bounds = None
        self.cache_dir = None
        
        # Validate and initialize
        self._validate_config()
        self._initialize_cloudvolume()
        
        logger.info("H01 data loader initialized")
    
    def _validate_config(self):
        """Validate the H01 configuration."""
        if not CLOUDVOLUME_AVAILABLE:
            raise RuntimeError("CloudVolume not available - required for H01 data access")
        
        if 'cloud_path' not in self.cloudvolume_config:
            raise ValueError("H01 configuration requires 'cloud_path'")
        
        # Set bounds
        self.bounds = self.cloudvolume_config.get('bounds', [[0, 0, 0], [1000, 1000, 1000]])
        logger.info(f"H01 bounds: {self.bounds}")
        
        # Set cache directory
        cache_config = self.config.get('data_access', {}).get('caching', {})
        if cache_config.get('enabled', True):
            self.cache_dir = cache_config.get('cache_dir', './h01_cache')
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"H01 cache directory: {self.cache_dir}")
    
    def _initialize_cloudvolume(self):
        """Initialize CloudVolume connection to H01 data."""
        try:
            cloud_path = self.cloudvolume_config['cloud_path']
            mip = self.cloudvolume_config.get('mip', 0)
            
            # Initialize CloudVolume with caching
            cv_config = {
                'mip': mip,
                'cache': self.cache_dir if self.cache_dir else False,
                'compress': 'gzip',
                'progress': True
            }
            
            self.volume = CloudVolume(cloud_path, **cv_config)
            
            logger.info(f"Connected to H01 data: {cloud_path}")
            logger.info(f"Volume shape: {self.volume.shape}")
            logger.info(f"Volume dtype: {self.volume.dtype}")
            
            # Get voxel size - handle different CloudVolume versions
            if hasattr(self.volume.meta, 'resolution'):
                self.voxel_size = self.volume.meta.resolution
            elif hasattr(self.volume, 'resolution'):
                 self.voxel_size = self.volume.resolution
            else:
                # Default H01 voxel size: 4nm x 4nm x 33nm
                self.voxel_size = [4, 4, 33]
                logger.warning("Could not get voxel size from CloudVolume, using default H01 values")
            
            logger.info(f"Voxel size: {self.voxel_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CloudVolume: {e}")
            raise
    
    def get_volume_info(self) -> Dict[str, Any]:
        """Get volume information."""
        return {
            'shape': self.volume.shape,
            'dtype': str(self.volume.dtype),
            'voxel_size': self.voxel_size,
            'bounds': self.bounds
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data loader statistics for robustness testing."""
        return {
            'connection_status': 'connected' if self.volume is not None else 'disconnected',
            'available_regions': [
                {
                    'name': 'test_region',
                    'bounds': self.bounds,
                    'size_gb': self._calculate_region_size(self.bounds)
                }
            ],
            'volume_shape': self.volume.shape if self.volume is not None else None,
            'voxel_size': self.voxel_size,
            'cache_directory': self.cache_dir
        }
    
    def _calculate_region_size(self, bounds: List[List[int]]) -> float:
        """Calculate size for given bounds."""
        start, end = bounds
        region_shape = [end[i] - start[i] for i in range(3)]
        voxel_size = self.volume.dtype.itemsize
        total_bytes = np.prod(region_shape) * voxel_size
        return total_bytes / (1024**3)
    
    def load_chunk(self, 
                  coordinates: Tuple[int, int, int],
                  chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load a chunk from the H01 volume."""
        if not self.volume:
            raise RuntimeError("CloudVolume not initialized")
        
        try:
            z, y, x = coordinates
            z_end = min(z + chunk_size[0], self.volume.shape[0])
            y_end = min(y + chunk_size[1], self.volume.shape[1])
            x_end = min(x + chunk_size[2], self.volume.shape[2])
            
            # Load chunk from CloudVolume
            chunk = self.volume[z:z_end, y:y_end, x:x_end]
            
            # The H01 dataset can be 4D (z,y,x,channel), so we squeeze it to 3D
            if chunk.ndim == 4:
                chunk = np.squeeze(chunk, axis=3)

            # Pad if necessary to ensure consistent chunk sizes
            if chunk.shape != chunk_size:
                padded_chunk = np.zeros(chunk_size, dtype=self.volume.dtype)
                # Use the actual chunk shape for slicing to prevent errors at the boundary
                padded_chunk[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2]] = chunk
                return padded_chunk
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to load chunk at {coordinates}: {e}")
            raise
    
    def get_region(self, region_name: str) -> Dict[str, Any]:
        """Get a predefined region from the H01 dataset."""
        regions = self.config.get('h01_regions', {})
        
        if region_name in regions:
            region_config = regions[region_name]
            bounds = region_config.get('bounds', self.bounds)
            description = region_config.get('description', '')
            
            return {
                'name': region_name,
                'bounds': bounds,
                'description': description,
                'size_gb': self._calculate_region_size(bounds)
            }
        else:
            raise ValueError(f"Region '{region_name}' not found in configuration")
    
    def list_available_regions(self) -> List[Dict[str, Any]]:
        """List all available regions in the H01 configuration."""
        regions = self.config.get('h01_regions', {})
        available_regions = []
        
        for name, config in regions.items():
            if isinstance(config, dict) and 'bounds' in config:
                region_info = self.get_region(name)
                available_regions.append(region_info)
        
        return available_regions
    
    def validate_data_source(self) -> bool:
        """Validate that the H01 data source is accessible."""
        try:
            info = self.get_volume_info()
            logger.info(f"H01 data source validation successful: {info}")
            return True
        except Exception as e:
            logger.error(f"H01 data source validation failed: {e}")
            return False
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the H01 data."""
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
                'connection_status': 'connected' if self.volume else 'disconnected'
            }
        except Exception as e:
            logger.error(f"Failed to get data statistics: {e}")
            return {'error': str(e)}

def create_h01_data_loader(config_path: str) -> H01DataLoader:
    """Create an H01 data loader from configuration file."""
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"H01 configuration loaded from {config_path}")
        return H01DataLoader(config)
    except Exception as e:
        logger.error(f"Failed to load H01 configuration from {config_path}: {e}")
        raise

def test_h01_connection():
    """Test function to verify H01 data access."""
    try:
        # Test with a small region
        test_config = {
            'data_source': {
                'cloudvolume': {
                    'cloud_path': 'gs://h01-release/data/20210601/4nm_raw',
                    'mip': 0,
                    'bounds': [[0, 0, 0], [100, 100, 100]]  # Very small test region
                }
            },
            'data_access': {
                'caching': {
                    'enabled': True,
                    'cache_dir': './h01_test_cache'
                }
            }
        }
        
        loader = H01DataLoader(test_config)
        
        if loader.validate_data_source():
            info = loader.get_volume_info()
            logger.info(f"H01 connection test successful: {info}")
            return True
        else:
            logger.error("H01 connection test failed")
            return False
            
    except Exception as e:
        logger.error(f"H01 connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the H01 data loader
    logging.basicConfig(level=logging.INFO)
    success = test_h01_connection()
    print(f"H01 connection test: {'SUCCESS' if success else 'FAILED'}") 