#!/usr/bin/env python3
"""
Enhanced data loader for connectomics pipeline with improved error handling,
memory management, and caching capabilities.
"""

import os
import logging
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import glob
import time
import pickle
import hashlib
from functools import lru_cache
import warnings

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
    warnings.warn("CloudVolume not available. Data loading will be limited.")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logger.warning("h5py not available - HDF5 data sources disabled")

class DataSourceError(Exception):
    """Custom exception for data source errors."""
    pass

class H01DataLoader:
    """
    Enhanced H01 data loader with improved error handling and caching.
    """
    
    def __init__(self, config, cache_dir: str = "h01_cache"):
        """
        Initialize the H01 data loader.
        
        Args:
            config: Configuration object with data settings
            cache_dir: Directory for caching data
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.volume = None
        self.volume_size = None
        self.resolution = None
        
        self._initialize_volume()
        self._setup_cache()
    
    def _initialize_volume(self):
        """Initialize the CloudVolume connection with error handling."""
        if not CLOUDVOLUME_AVAILABLE:
            raise ImportError("CloudVolume is required but not available")
        
        try:
            logger.info(f"Initializing CloudVolume with path: {self.config.data.volume_path}")
            self.volume = CloudVolume(
                self.config.data.volume_path,
                mip=self.config.data.mip,
                cache=str(self.cache_dir),
                parallel=True,
                progress=False
            )
            
            # Get volume information
            self.volume_size = self.volume.mip_volume_size(self.config.data.mip)
            self.resolution = self.volume.mip_resolution(self.config.data.mip)
            
            logger.info(f"Volume initialized successfully")
            logger.info(f"Volume size: {self.volume_size}")
            logger.info(f"Resolution: {self.resolution}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CloudVolume: {e}")
            raise
    
    def _setup_cache(self):
        """Setup caching system."""
        self.cache_metadata_file = self.cache_dir / "cache_metadata.pkl"
        self.cache_metadata = self._load_cache_metadata()
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.cache_metadata_file, 'wb') as f:
                pickle.dump(self.cache_metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, coords: Tuple[int, int, int], size: Tuple[int, int, int]) -> str:
        """Generate a cache key for the given coordinates and size."""
        key_data = f"{coords}_{size}_{self.config.data.mip}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.npy"
    
    def load_chunk(self, coords: Tuple[int, int, int], size: Tuple[int, int, int], 
                   use_cache: bool = True) -> np.ndarray:
        """
        Load a chunk of data from the volume.
        
        Args:
            coords: (x, y, z) coordinates
            size: (dx, dy, dz) size of the chunk
            use_cache: Whether to use caching
            
        Returns:
            Loaded data as numpy array
        """
        cache_key = self._get_cache_key(coords, size)
        cache_path = self._get_cache_path(cache_key)
        
        # Try to load from cache first
        if use_cache and cache_path.exists():
            try:
                data = np.load(cache_path)
                logger.debug(f"Loaded chunk from cache: {coords}, {size}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Load from volume
        try:
            x, y, z = coords
            dx, dy, dz = size
            
            # Validate coordinates
            if not self._validate_coordinates(coords, size):
                raise ValueError(f"Invalid coordinates: {coords}, {size}")
            
            start_time = time.time()
            data = self.volume[x:x+dx, y:y+dy, z:z+dz].squeeze()
            load_time = time.time() - start_time
            
            logger.debug(f"Loaded chunk from volume: {coords}, {size} in {load_time:.2f}s")
            
            # Cache the data
            if use_cache:
                try:
                    np.save(cache_path, data)
                    self.cache_metadata[cache_key] = {
                        'coords': coords,
                        'size': size,
                        'timestamp': time.time(),
                        'load_time': load_time
                    }
                    self._save_cache_metadata()
                except Exception as e:
                    logger.warning(f"Failed to cache data: {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load chunk {coords}, {size}: {e}")
            raise
    
    def _validate_coordinates(self, coords: Tuple[int, int, int], 
                            size: Tuple[int, int, int]) -> bool:
        """Validate that coordinates and size are within volume bounds."""
        if self.volume_size is None:
            return False
        
        x, y, z = coords
        dx, dy, dz = size
        
        # Check if coordinates are non-negative
        if x < 0 or y < 0 or z < 0:
            return False
        
        # Check if chunk fits within volume
        if (x + dx > self.volume_size[0] or 
            y + dy > self.volume_size[1] or 
            z + dz > self.volume_size[2]):
            return False
        
        return True
    
    def get_random_valid_coords(self, chunk_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get random valid coordinates for a given chunk size."""
        if self.volume_size is None:
            raise RuntimeError("Volume not initialized")
        
        max_x = self.volume_size[0] - chunk_size[0]
        max_y = self.volume_size[1] - chunk_size[1]
        max_z = self.volume_size[2] - chunk_size[2]
        
        if max_x < 0 or max_y < 0 or max_z < 0:
            raise ValueError(f"Chunk size {chunk_size} is larger than volume size {self.volume_size}")
        
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        z = np.random.randint(0, max_z)
        
        return (x, y, z)
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink()
            self.cache_metadata.clear()
            self._save_cache_metadata()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.npy"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'num_cached_chunks': len(cache_files),
            'total_cache_size_mb': total_size / (1024 * 1024),
            'cache_metadata_entries': len(self.cache_metadata)
        }


class H01Dataset(Dataset):
    """
    PyTorch Dataset for H01 data with augmentation and preprocessing.
    """
    
    def __init__(self, data_loader: H01DataLoader, config, 
                 samples_per_epoch: int = 1000, augment: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_loader: H01DataLoader instance
            config: Configuration object
            samples_per_epoch: Number of samples per epoch
            augment: Whether to apply data augmentation
        """
        self.data_loader = data_loader
        self.config = config
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        
        # Pre-generate valid coordinates for faster access
        self.valid_coords = self._generate_valid_coordinates()
        
        logger.info(f"Dataset initialized with {len(self.valid_coords)} valid coordinates")
    
    def _generate_valid_coordinates(self) -> List[Tuple[int, int, int]]:
        """Generate a list of valid coordinates for the dataset."""
        coords = []
        chunk_size = tuple(self.config.data.chunk_size)
        
        for _ in range(self.samples_per_epoch):
            try:
                coord = self.data_loader.get_random_valid_coords(chunk_size)
                coords.append(coord)
            except Exception as e:
                logger.warning(f"Failed to generate coordinate: {e}")
                continue
        
        return coords
    
    def __len__(self) -> int:
        return self.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self.valid_coords):
            # Regenerate coordinates if needed
            self.valid_coords = self._generate_valid_coordinates()
            idx = idx % len(self.valid_coords)
        
        coords = self.valid_coords[idx]
        chunk_size = tuple(self.config.data.chunk_size)
        
        try:
            # Load data
            data = self.data_loader.load_chunk(coords, chunk_size)
            
            # Convert to tensor
            data_tensor = torch.from_numpy(data).float()
            
            # Add channel dimension if needed
            if data_tensor.dim() == 3:
                data_tensor = data_tensor.unsqueeze(0)  # Add channel dimension
            
            # Normalize data
            data_tensor = self._normalize_data(data_tensor)
            
            # Apply augmentation if enabled
            if self.augment:
                data_tensor = self._apply_augmentation(data_tensor)
            
            # Create dummy target (for training, you'd load actual segmentation)
            target = self._create_target(data_tensor)
            
            return data_tensor, target
            
        except Exception as e:
            logger.error(f"Failed to load sample {idx}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros((1,) + tuple(chunk_size)), torch.zeros((3,) + tuple(chunk_size))
    
    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize the data to [0, 1] range."""
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
        return data
    
    def _apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation."""
        # Random rotation
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            data = torch.rot90(data, k, dims=[-2, -1])
        
        # Random flip
        if np.random.random() > 0.5:
            data = torch.flip(data, dims=[-1])
        
        # Add noise
        if np.random.random() > 0.7:
            noise = torch.randn_like(data) * 0.1
            data = data + noise
            data = torch.clamp(data, 0, 1)
        
        return data
    
    def _create_target(self, data: torch.Tensor) -> torch.Tensor:
        """Create a dummy target tensor for training."""
        # This is a placeholder - in real usage, you'd load actual segmentation data
        target_shape = (3,) + data.shape[1:]  # 3 channels for segmentation
        target = torch.zeros(target_shape)
        
        # Create some dummy segmentation based on data intensity
        intensity = data.mean(dim=0, keepdim=True)
        target[0] = (intensity > 0.5).float()  # Background
        target[1] = (intensity > 0.3).float()  # Foreground
        target[2] = (intensity > 0.7).float()  # High intensity
        
        return target


class H01IterableDataset(IterableDataset):
    """
    Iterable dataset for streaming data from H01.
    """
    
    def __init__(self, data_loader: H01DataLoader, config, samples_per_epoch: int = 1000):
        self.data_loader = data_loader
        self.config = config
        self.samples_per_epoch = samples_per_epoch
    
    def __iter__(self):
        """Iterate over the dataset."""
        for _ in range(self.samples_per_epoch):
            try:
                coords = self.data_loader.get_random_valid_coords(tuple(self.config.data.chunk_size))
                data = self.data_loader.load_chunk(coords, tuple(self.config.data.chunk_size))
                
                # Convert to tensor and normalize
                data_tensor = torch.from_numpy(data).float()
                if data_tensor.dim() == 3:
                    data_tensor = data_tensor.unsqueeze(0)
                
                data_tensor = (data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min() + 1e-8)
                
                # Create dummy target
                target = torch.zeros((3,) + data_tensor.shape[1:])
                
                yield data_tensor, target
                
            except Exception as e:
                logger.warning(f"Failed to load sample: {e}")
                continue


def create_data_loader(config, dataset_type: str = "dataset") -> DataLoader:
    """
    Create a data loader with the specified configuration.
    
    Args:
        config: Configuration object
        dataset_type: Type of dataset ("dataset" or "iterable")
    
    Returns:
        DataLoader instance
    """
    # Initialize H01 data loader
    h01_loader = H01DataLoader(config)
    
    # Create dataset
    if dataset_type == "dataset":
        dataset = H01Dataset(h01_loader, config)
    elif dataset_type == "iterable":
        dataset = H01IterableDataset(h01_loader, config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=(dataset_type == "dataset"),
        num_workers=config.data.num_workers,
        pin_memory=True,
        prefetch_factor=config.data.prefetch_factor
    )
    
    return loader

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
    return create_data_loader(config) 