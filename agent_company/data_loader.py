#!/usr/bin/env python3
"""
Production Data Loader for Agentic Tracer
=========================================
Handles multiple data source types with proper error handling and validation.
"""

import os
import logging
import yaml
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import glob

logger = logging.getLogger(__name__)

# Optional imports for different data formats
try:
    import zarr
    import numcodecs
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    logger.warning("Zarr not available - zarr data sources disabled")

try:
    from cloudvolume import CloudVolume
    CLOUDVOLUME_AVAILABLE = True
except ImportError:
    CLOUDVOLUME_AVAILABLE = False
    logger.warning("CloudVolume not available - cloud data sources disabled")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logger.warning("h5py not available - HDF5 data sources disabled")

class DataSourceError(Exception):
    """Custom exception for data source errors."""
    pass

class DataLoader:
    """Production data loader for multiple data source types."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_source_config = config.get('data_source', {})
        self.source_type = self.data_source_config.get('type', 'numpy')
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Data loader initialized for {self.source_type} data source")
    
    def _validate_config(self):
        """Validate the data source configuration."""
        if self.source_type == 'numpy':
            if 'numpy' not in self.data_source_config:
                raise DataSourceError("Numpy data source requires 'numpy' configuration section")
            
            numpy_config = self.data_source_config['numpy']
            if 'base_path' not in numpy_config:
                raise DataSourceError("Numpy data source requires 'base_path'")
            
            if not os.path.exists(numpy_config['base_path']):
                logger.warning(f"Numpy base path does not exist: {numpy_config['base_path']}")
        
        elif self.source_type == 'zarr':
            if not ZARR_AVAILABLE:
                raise DataSourceError("Zarr data source requires zarr package")
            
            if 'zarr' not in self.data_source_config:
                raise DataSourceError("Zarr data source requires 'zarr' configuration section")
            
            zarr_config = self.data_source_config['zarr']
            if 'store_path' not in zarr_config:
                raise DataSourceError("Zarr data source requires 'store_path'")
        
        elif self.source_type == 'cloudvolume':
            if not CLOUDVOLUME_AVAILABLE:
                raise DataSourceError("CloudVolume data source requires cloudvolume package")
            
            if 'cloudvolume' not in self.data_source_config:
                raise DataSourceError("CloudVolume data source requires 'cloudvolume' configuration section")
            
            cv_config = self.data_source_config['cloudvolume']
            if 'cloud_path' not in cv_config:
                raise DataSourceError("CloudVolume data source requires 'cloud_path'")
        
        elif self.source_type == 'hdf5':
            if not H5PY_AVAILABLE:
                raise DataSourceError("HDF5 data source requires h5py package")
            
            if 'hdf5' not in self.data_source_config:
                raise DataSourceError("HDF5 data source requires 'hdf5' configuration section")
            
            h5_config = self.data_source_config['hdf5']
            if 'file_path' not in h5_config:
                raise DataSourceError("HDF5 data source requires 'file_path'")
        
        else:
            raise DataSourceError(f"Unsupported data source type: {self.source_type}")
    
    def get_volume_info(self) -> Dict[str, Any]:
        """Get information about the volume (shape, dtype, etc.)."""
        try:
            if self.source_type == 'numpy':
                return self._get_numpy_volume_info()
            elif self.source_type == 'zarr':
                return self._get_zarr_volume_info()
            elif self.source_type == 'cloudvolume':
                return self._get_cloudvolume_volume_info()
            elif self.source_type == 'hdf5':
                return self._get_hdf5_volume_info()
            else:
                raise DataSourceError(f"Unsupported data source type: {self.source_type}")
        except Exception as e:
            logger.error(f"Failed to get volume info: {e}")
            raise DataSourceError(f"Failed to get volume info: {e}")
    
    def _get_numpy_volume_info(self) -> Dict[str, Any]:
        """Get volume info for numpy data source."""
        numpy_config = self.data_source_config['numpy']
        base_path = numpy_config['base_path']
        file_pattern = numpy_config.get('file_pattern', '*.npy')
        
        # Find numpy files
        pattern = os.path.join(base_path, file_pattern)
        files = glob.glob(pattern)
        
        if not files:
            raise DataSourceError(f"No files found matching pattern: {pattern}")
        
        # Load first file to get info
        first_file = files[0]
        try:
            data = np.load(first_file, mmap_mode='r')
            return {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'num_files': len(files),
                'file_size_gb': os.path.getsize(first_file) / (1024**3),
                'total_size_gb': sum(os.path.getsize(f) for f in files) / (1024**3),
                'files': files
            }
        except Exception as e:
            raise DataSourceError(f"Failed to load numpy file {first_file}: {e}")
    
    def _get_zarr_volume_info(self) -> Dict[str, Any]:
        """Get volume info for zarr data source."""
        zarr_config = self.data_source_config['zarr']
        store_path = zarr_config['store_path']
        dataset_name = zarr_config.get('dataset_name', 'data')
        
        try:
            store = zarr.open(store_path, mode='r')
            dataset = store[dataset_name]
            
            return {
                'shape': dataset.shape,
                'dtype': str(dataset.dtype),
                'chunks': dataset.chunks,
                'compressor': str(dataset.compressor),
                'size_gb': dataset.nbytes / (1024**3)
            }
        except Exception as e:
            raise DataSourceError(f"Failed to get zarr volume info: {e}")
    
    def _get_cloudvolume_volume_info(self) -> Dict[str, Any]:
        """Get volume info for cloudvolume data source."""
        cv_config = self.data_source_config['cloudvolume']
        cloud_path = cv_config['cloud_path']
        
        try:
            vol = CloudVolume(cloud_path)
            return {
                'shape': vol.shape,
                'dtype': str(vol.dtype),
                'voxel_size': vol.voxel_size,
                'mip': vol.mip,
                'size_gb': np.prod(vol.shape) * vol.dtype.itemsize / (1024**3)
            }
        except Exception as e:
            raise DataSourceError(f"Failed to get cloudvolume info: {e}")
    
    def _get_hdf5_volume_info(self) -> Dict[str, Any]:
        """Get volume info for HDF5 data source."""
        h5_config = self.data_source_config['hdf5']
        file_path = h5_config['file_path']
        dataset_name = h5_config.get('dataset_name', '/data')
        
        try:
            with h5py.File(file_path, 'r') as f:
                dataset = f[dataset_name]
                return {
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype),
                    'chunks': dataset.chunks,
                    'compression': dataset.compression,
                    'size_gb': dataset.nbytes / (1024**3)
                }
        except Exception as e:
            raise DataSourceError(f"Failed to get HDF5 volume info: {e}")
    
    def load_chunk(self, 
                  coordinates: Tuple[int, int, int],
                  chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load a chunk from the data source."""
        try:
            if self.source_type == 'numpy':
                return self._load_numpy_chunk(coordinates, chunk_size)
            elif self.source_type == 'zarr':
                return self._load_zarr_chunk(coordinates, chunk_size)
            elif self.source_type == 'cloudvolume':
                return self._load_cloudvolume_chunk(coordinates, chunk_size)
            elif self.source_type == 'hdf5':
                return self._load_hdf5_chunk(coordinates, chunk_size)
            else:
                raise DataSourceError(f"Unsupported data source type: {self.source_type}")
        except Exception as e:
            logger.error(f"Failed to load chunk at {coordinates}: {e}")
            raise DataSourceError(f"Failed to load chunk at {coordinates}: {e}")
    
    def _load_numpy_chunk(self, 
                         coordinates: Tuple[int, int, int],
                         chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk from numpy file."""
        numpy_config = self.data_source_config['numpy']
        base_path = numpy_config['base_path']
        
        # For now, assume single large numpy file
        # In production, you might have multiple files or a specific file selection strategy
        pattern = os.path.join(base_path, numpy_config.get('file_pattern', '*.npy'))
        files = glob.glob(pattern)
        
        if not files:
            raise DataSourceError(f"No numpy files found in {base_path}")
        
        # Load from first file (simplified - in production you'd have a file selection strategy)
        file_path = files[0]
        data = np.load(file_path, mmap_mode='r')
        
        z, y, x = coordinates
        z_end = min(z + chunk_size[0], data.shape[0])
        y_end = min(y + chunk_size[1], data.shape[1])
        x_end = min(x + chunk_size[2], data.shape[2])
        
        chunk = data[z:z_end, y:y_end, x:x_end]
        
        # Pad if necessary
        if chunk.shape != chunk_size:
            padded_chunk = np.zeros(chunk_size, dtype=data.dtype)
            padded_chunk[:z_end-z, :y_end-y, :x_end-x] = chunk
            return padded_chunk
        
        return chunk
    
    def _load_zarr_chunk(self,
                        coordinates: Tuple[int, int, int],
                        chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk from zarr store."""
        zarr_config = self.data_source_config['zarr']
        store_path = zarr_config['store_path']
        dataset_name = zarr_config.get('dataset_name', 'data')
        
        store = zarr.open(store_path, mode='r')
        dataset = store[dataset_name]
        
        z, y, x = coordinates
        z_end = min(z + chunk_size[0], dataset.shape[0])
        y_end = min(y + chunk_size[1], dataset.shape[1])
        x_end = min(x + chunk_size[2], dataset.shape[2])
        
        chunk = dataset[z:z_end, y:y_end, x:x_end]
        
        # Pad if necessary
        if chunk.shape != chunk_size:
            padded_chunk = np.zeros(chunk_size, dtype=dataset.dtype)
            padded_chunk[:z_end-z, :y_end-y, :x_end-x] = chunk
            return padded_chunk
        
        return chunk
    
    def _load_cloudvolume_chunk(self,
                               coordinates: Tuple[int, int, int],
                               chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk from cloudvolume."""
        cv_config = self.data_source_config['cloudvolume']
        cloud_path = cv_config['cloud_path']
        
        vol = CloudVolume(cloud_path)
        
        z, y, x = coordinates
        z_end = min(z + chunk_size[0], vol.shape[0])
        y_end = min(y + chunk_size[1], vol.shape[1])
        x_end = min(x + chunk_size[2], vol.shape[2])
        
        chunk = vol[z:z_end, y:y_end, x:x_end]
        
        # Pad if necessary
        if chunk.shape != chunk_size:
            padded_chunk = np.zeros(chunk_size, dtype=vol.dtype)
            padded_chunk[:z_end-z, :y_end-y, :x_end-x] = chunk
            return padded_chunk
        
        return chunk
    
    def _load_hdf5_chunk(self,
                        coordinates: Tuple[int, int, int],
                        chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk from HDF5 file."""
        h5_config = self.data_source_config['hdf5']
        file_path = h5_config['file_path']
        dataset_name = h5_config.get('dataset_name', '/data')
        
        with h5py.File(file_path, 'r') as f:
            dataset = f[dataset_name]
            
            z, y, x = coordinates
            z_end = min(z + chunk_size[0], dataset.shape[0])
            y_end = min(y + chunk_size[1], dataset.shape[1])
            x_end = min(x + chunk_size[2], dataset.shape[2])
            
            chunk = dataset[z:z_end, y:y_end, x:x_end]
            
            # Pad if necessary
            if chunk.shape != chunk_size:
                padded_chunk = np.zeros(chunk_size, dtype=dataset.dtype)
                padded_chunk[:z_end-z, :y_end-y, :x_end-x] = chunk
                return padded_chunk
            
            return chunk
    
    def validate_data_source(self) -> bool:
        """Validate that the data source is accessible and properly configured."""
        try:
            info = self.get_volume_info()
            logger.info(f"Data source validation successful: {info}")
            return True
        except Exception as e:
            logger.error(f"Data source validation failed: {e}")
            return False

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise

def create_data_loader(config_path: str) -> DataLoader:
    """Create a data loader from configuration file."""
    config = load_config(config_path)
    return DataLoader(config) 