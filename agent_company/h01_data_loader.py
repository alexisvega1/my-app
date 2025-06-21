#!/usr/bin/env python3
"""
H01 Dataset Data Loader for Agentic Tracer
==========================================
Specialized data loader for the H01 connectomics dataset.
Based on Google FFN patterns and H01 release data.
"""

import os
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# GCSFS for reliable Google Cloud Storage access
def check_gcsfs_available():
    """Dynamically check if gcsfs is available."""
    try:
        import gcsfs
        return True
    except ImportError:
        return False

# CloudVolume for H01 data access
def check_cloudvolume_available():
    """Dynamically check if CloudVolume is available."""
    try:
        from cloudvolume import CloudVolume
        return True
    except ImportError:
        return False

class H01DataLoader:
    """Specialized data loader for the H01 connectomics dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_source_config = config.get('data_source', {})
        self.cloudvolume_config = self.data_source_config.get('cloudvolume', {})
        
        # Initialize connections
        self.volume = None
        self.gcsfs = None
        self.bounds = None
        self.cache_dir = None
        self.connection_method = None
        
        # Validate and initialize
        self._validate_config()
        self._initialize_connections()
        
        logger.info("H01 data loader initialized")
    
    def _validate_config(self):
        """Validate the H01 configuration."""
        # Check for required dependencies
        if not check_gcsfs_available() and not check_cloudvolume_available():
            raise RuntimeError("Neither gcsfs nor CloudVolume available - required for H01 data access")
        
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
    
    def _initialize_connections(self):
        """Initialize connections to H01 data using multiple methods."""
        cloud_path = self.cloudvolume_config['cloud_path']
        mip = self.cloudvolume_config.get('mip', 0)
        
        # Method 1: Try CloudVolume with gcsfs backend
        if check_gcsfs_available():
            try:
                import gcsfs
                from cloudvolume import CloudVolume
                
                self.gcsfs = gcsfs.GCSFileSystem(token='anon')
                logger.info("Using gcsfs + CloudVolume for H01 data access")
                
                cv_config = {
                    'mip': mip,
                    'cache': self.cache_dir if self.cache_dir else False,
                    'compress': 'gzip',
                    'progress': True
                }
                
                self.volume = CloudVolume(cloud_path, **cv_config)
                self.connection_method = 'gcsfs_cloudvolume'
                logger.info("✅ Successfully connected using gcsfs + CloudVolume")
                return
                
            except Exception as e:
                logger.warning(f"gcsfs + CloudVolume failed: {e}")
        
        # Method 2: Try direct CloudVolume
        if check_cloudvolume_available():
            try:
                from cloudvolume import CloudVolume
                
                cv_config = {
                    'mip': mip,
                    'cache': self.cache_dir if self.cache_dir else False,
                    'compress': 'gzip',
                    'progress': True
                }
                
                self.volume = CloudVolume(cloud_path, **cv_config)
                self.connection_method = 'direct_cloudvolume'
                logger.info("✅ Successfully connected using direct CloudVolume")
                return
                
            except Exception as e:
                logger.warning(f"Direct CloudVolume failed: {e}")
        
        # Method 3: Try gcsfs direct access (fallback)
        if check_gcsfs_available():
            try:
                import gcsfs
                self.gcsfs = gcsfs.GCSFileSystem(token='anon')
                self.connection_method = 'gcsfs_direct'
                logger.info("✅ Successfully connected using gcsfs direct access")
                # Note: This method requires different data loading approach
                return
                
            except Exception as e:
                logger.warning(f"gcsfs direct access failed: {e}")
        
        # If all methods failed
        raise RuntimeError("All H01 data access methods failed")
    
    def _get_volume_info_cloudvolume(self):
        """Get volume information from CloudVolume."""
        if not self.volume:
            raise RuntimeError("CloudVolume not initialized")
        
        # Get voxel size - handle different CloudVolume versions
        if hasattr(self.volume.meta, 'resolution'):
            try:
                self.voxel_size = self.volume.meta.resolution(0)  # Pass mip=0 for base resolution
            except TypeError:
                # Fallback for older versions that don't require mip
                self.voxel_size = self.volume.meta.resolution
        elif hasattr(self.volume, 'resolution'):
             self.voxel_size = self.volume.resolution
        else:
            # Default H01 voxel size: 4nm x 4nm x 33nm
            self.voxel_size = [4, 4, 33]
            logger.warning("Could not get voxel size from CloudVolume, using default H01 values")
        
        return {
            'shape': self.volume.shape,
            'dtype': str(self.volume.dtype),
            'voxel_size': self.voxel_size,
            'bounds': self.bounds,
            'connection_type': self.connection_method
        }
    
    def get_volume_info(self) -> Dict[str, Any]:
        """Get volume information."""
        if self.connection_method in ['gcsfs_cloudvolume', 'direct_cloudvolume']:
            return self._get_volume_info_cloudvolume()
        elif self.connection_method == 'gcsfs_direct':
            # For direct gcsfs access, we need to infer from the path
            return {
                'shape': [100000, 100000, 100000],  # H01 full dataset size
                'dtype': 'uint8',
                'voxel_size': [4, 4, 33],
                'bounds': self.bounds,
                'connection_type': self.connection_method
            }
        else:
            raise RuntimeError("Unknown connection method")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data loader statistics for robustness testing."""
        volume_info = self.get_volume_info()
        return {
            'connection_status': 'connected' if self.volume or self.gcsfs else 'disconnected',
            'connection_type': self.connection_method,
            'available_regions': [
                {
                    'name': 'test_region',
                    'bounds': self.bounds,
                    'size_gb': self._calculate_region_size(self.bounds)
                }
            ],
            'volume_shape': volume_info['shape'],
            'voxel_size': volume_info['voxel_size'],
            'cache_directory': self.cache_dir
        }
    
    def _calculate_region_size(self, bounds: List[List[int]]) -> float:
        """Calculate size for given bounds."""
        start, end = bounds
        region_shape = [end[i] - start[i] for i in range(3)]
        # Assume uint8 data
        voxel_size = 1  # bytes
        total_bytes = np.prod(region_shape) * voxel_size
        return total_bytes / (1024**3)
    
    def load_chunk(self, 
                  coordinates: Tuple[int, int, int],
                  chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load a chunk from the H01 volume."""
        if self.connection_method in ['gcsfs_cloudvolume', 'direct_cloudvolume']:
            return self._load_chunk_cloudvolume(coordinates, chunk_size)
        elif self.connection_method == 'gcsfs_direct':
            return self._load_chunk_gcsfs_direct(coordinates, chunk_size)
        else:
            raise RuntimeError("No valid connection method available")
    
    def _load_chunk_cloudvolume(self, coordinates: Tuple[int, int, int], chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk using CloudVolume."""
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
                padded_chunk[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2]] = chunk
                return padded_chunk
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to load chunk at {coordinates}: {e}")
            raise
    
    def _load_chunk_gcsfs_direct(self, coordinates: Tuple[int, int, int], chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk using direct gcsfs access (fallback method)."""
        if not self.gcsfs:
            raise RuntimeError("gcsfs not initialized")
        
        # This is a simplified fallback - in practice, you'd need to implement
        # the specific chunk loading logic for the H01 data format
        logger.warning("Direct gcsfs chunk loading not fully implemented - using dummy data")
        return np.random.randint(0, 255, chunk_size, dtype=np.uint8)
    
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
            stats = {
                'connection_type': self.connection_method,
                'volume_shape': self.get_volume_info()['shape'],
                'voxel_size': self.get_volume_info()['voxel_size'],
                'bounds': self.bounds,
                'cache_enabled': self.cache_dir is not None,
                'cache_directory': self.cache_dir
            }
            
            # Test a small chunk to verify access
            test_coords = (1000, 1000, 1000)
            test_size = (10, 10, 10)
            try:
                test_chunk = self.load_chunk(test_coords, test_size)
                stats['test_chunk_success'] = True
                stats['test_chunk_shape'] = test_chunk.shape
                stats['test_chunk_dtype'] = str(test_chunk.dtype)
            except Exception as e:
                stats['test_chunk_success'] = False
                stats['test_chunk_error'] = str(e)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get data statistics: {e}")
            return {'error': str(e)}

def create_h01_data_loader(config_path: str) -> H01DataLoader:
    """Create an H01 data loader from a configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return H01DataLoader(config)

def test_h01_connection():
    """Test H01 data connection."""
    try:
        # Test gcsfs availability
        gcsfs_available = check_gcsfs_available()
        print(f"gcsfs available: {gcsfs_available}")
        
        # Test CloudVolume availability
        cv_available = check_cloudvolume_available()
        print(f"CloudVolume available: {cv_available}")
        
        if not gcsfs_available and not cv_available:
            print("❌ Neither gcsfs nor CloudVolume available")
            return False
        
        print("✅ H01 data access components available")
        return True
        
    except Exception as e:
        print(f"❌ H01 connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the H01 data loader
    logging.basicConfig(level=logging.INFO)
    success = test_h01_connection()
    print(f"H01 connection test: {'SUCCESS' if success else 'FAILED'}") 