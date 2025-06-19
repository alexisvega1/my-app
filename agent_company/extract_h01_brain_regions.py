#!/usr/bin/env python3
"""
Extract Specific Brain Regions from H01 Dataset
==============================================
Extracts predefined brain regions from the H01 connectomics dataset.
Based on anatomical coordinates and H01 dataset structure.
"""

import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from cloudvolume import CloudVolume
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class H01RegionExtractor:
    """Extract specific brain regions from H01 dataset."""
    
    def __init__(self, cache_dir: str = "./h01_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # H01 dataset path (cloud or local)
        self.cloud_path = "gs://h01-release/data/20210601/4nm_raw"
        
        # Initialize CloudVolume
        self.volume = None
        self._initialize_cloudvolume()
        
        # Define brain regions with approximate coordinates
        # These are based on H01 dataset structure and anatomical regions
        self.brain_regions = {
            "prefrontal_cortex": {
                "description": "Prefrontal cortex - executive function and decision making",
                "coordinates": {
                    "small": ([50000, 300000, 1000], [50256, 300256, 1064]),  # 256x256x64
                    "medium": ([50000, 300000, 1000], [50512, 300512, 1128]),  # 512x512x128
                    "large": ([50000, 300000, 1000], [51024, 301024, 1256])   # 1024x1024x256
                }
            },
            "hippocampus": {
                "description": "Hippocampus - memory formation and spatial navigation",
                "coordinates": {
                    "small": ([200000, 200000, 2000], [200256, 200256, 2064]),
                    "medium": ([200000, 200000, 2000], [200512, 200512, 2128]),
                    "large": ([200000, 200000, 2000], [201024, 201024, 2256])
                }
            },
            "visual_cortex": {
                "description": "Primary visual cortex - visual processing",
                "coordinates": {
                    "small": ([400000, 100000, 1500], [400256, 100256, 1564]),
                    "medium": ([400000, 100000, 1500], [400512, 100512, 1628]),
                    "large": ([400000, 100000, 1500], [401024, 101024, 1756])
                }
            },
            "motor_cortex": {
                "description": "Primary motor cortex - movement control",
                "coordinates": {
                    "small": ([300000, 250000, 1200], [300256, 250256, 1264]),
                    "medium": ([300000, 250000, 1200], [300512, 250512, 1328]),
                    "large": ([300000, 250000, 1200], [301024, 251024, 1456])
                }
            },
            "thalamus": {
                "description": "Thalamus - sensory relay and consciousness",
                "coordinates": {
                    "small": ([250000, 250000, 1800], [250256, 250256, 1864]),
                    "medium": ([250000, 250000, 1800], [250512, 250512, 1928]),
                    "large": ([250000, 250000, 1800], [251024, 251024, 2056])
                }
            },
            "cerebellum": {
                "description": "Cerebellum - motor coordination and balance",
                "coordinates": {
                    "small": ([150000, 400000, 800], [150256, 400256, 864]),
                    "medium": ([150000, 400000, 800], [150512, 400512, 928]),
                    "large": ([150000, 400000, 800], [151024, 401024, 1056])
                }
            }
        }
    
    def _initialize_cloudvolume(self):
        """Initialize CloudVolume connection."""
        try:
            logger.info("Initializing CloudVolume connection...")
            
            # Try cloud path first
            try:
                self.volume = CloudVolume(
                    self.cloud_path,
                    mip=0,  # Highest resolution
                    cache=str(self.cache_dir),
                    progress=True,
                    use_https=True
                )
                logger.info(f"Connected to cloud H01 data: {self.cloud_path}")
                logger.info(f"Volume shape: {self.volume.shape}")
                logger.info(f"Volume dtype: {self.volume.dtype}")
                
            except Exception as cloud_error:
                logger.warning(f"Cloud connection failed: {cloud_error}")
                
                # Try local cache
                local_path = f"file://{self.cache_dir}/gs/h01-release/data/20210601/4nm_raw"
                try:
                    self.volume = CloudVolume(
                        local_path,
                        mip=0,
                        cache=str(self.cache_dir),
                        progress=True
                    )
                    logger.info(f"Connected to local H01 cache: {local_path}")
                    logger.info(f"Volume shape: {self.volume.shape}")
                    
                except Exception as local_error:
                    logger.error(f"Both cloud and local connections failed:")
                    logger.error(f"Cloud error: {cloud_error}")
                    logger.error(f"Local error: {local_error}")
                    raise RuntimeError("Cannot connect to H01 data source")
                    
        except Exception as e:
            logger.error(f"Failed to initialize CloudVolume: {e}")
            raise
    
    def get_volume_info(self) -> Dict:
        """Get information about the H01 volume."""
        if not self.volume:
            raise RuntimeError("CloudVolume not initialized")
        
        return {
            "shape": self.volume.shape,
            "dtype": str(self.volume.dtype),
            "voxel_size": getattr(self.volume, 'voxel_size', [4, 4, 33]),
            "mip": getattr(self.volume, 'mip', 0)
        }
    
    def extract_region(self, region_name: str, size: str = "medium") -> Optional[np.ndarray]:
        """Extract a specific brain region."""
        if region_name not in self.brain_regions:
            raise ValueError(f"Unknown region: {region_name}")
        
        if size not in ["small", "medium", "large"]:
            raise ValueError(f"Invalid size: {size}. Use 'small', 'medium', or 'large'")
        
        region_info = self.brain_regions[region_name]
        coordinates = region_info["coordinates"][size]
        
        # Handle both tuple and dict formats
        if isinstance(coordinates, tuple):
            start, stop = coordinates
        elif isinstance(coordinates, dict):
            start = coordinates["start"]
            stop = coordinates["stop"]
        else:
            raise ValueError(f"Invalid coordinate format for {region_name}")
        
        logger.info(f"Extracting {region_name} ({size}) from {start} to {stop}...")
        
        try:
            # Extract the region
            region_data = self.volume[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            
            logger.info(f"✓ Successfully extracted {region_name} with shape: {region_data.shape}")
            logger.info(f"  Data range: {region_data.min()} to {region_data.max()}")
            logger.info(f"  Non-zero voxels: {np.count_nonzero(region_data)} / {region_data.size}")
            
            return region_data
            
        except Exception as e:
            logger.error(f"✗ Failed to extract {region_name}: {e}")
            return None
    
    def extract_multiple_regions(self, regions: List[str], size: str = "medium") -> Dict[str, np.ndarray]:
        """Extract multiple brain regions."""
        results = {}
        
        for region_name in regions:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing region: {region_name}")
            logger.info(f"{'='*50}")
            
            data = self.extract_region(region_name, size)
            if data is not None:
                results[region_name] = data
                
                # Save immediately
                filename = f"h01_{region_name}_{size}.npy"
                np.save(filename, data)
                logger.info(f"✓ Saved {filename}")
            
            # Add delay between extractions to avoid overwhelming the server
            time.sleep(2)
        
        return results
    
    def extract_all_regions(self, size: str = "medium") -> Dict[str, np.ndarray]:
        """Extract all available brain regions."""
        regions = list(self.brain_regions.keys())
        return self.extract_multiple_regions(regions, size)
    
    def get_region_info(self, region_name: str) -> Dict:
        """Get information about a specific region."""
        if region_name not in self.brain_regions:
            raise ValueError(f"Unknown region: {region_name}")
        
        region_info = self.brain_regions[region_name].copy()
        
        # Calculate sizes for each region
        for size, (start, stop) in region_info["coordinates"].items():
            shape = [stop[i] - start[i] for i in range(3)]
            size_mb = np.prod(shape) * 1 / (1024 * 1024)  # Assuming uint8
            region_info["coordinates"][size] = {
                "start": start,
                "stop": stop,
                "shape": shape,
                "size_mb": size_mb
            }
        
        return region_info
    
    def list_available_regions(self) -> List[Dict]:
        """List all available regions with their information."""
        regions = []
        
        for name, info in self.brain_regions.items():
            region_info = self.get_region_info(name)
            regions.append({
                "name": name,
                "description": region_info["description"],
                "sizes": list(region_info["coordinates"].keys()),
                "coordinates": region_info["coordinates"]
            })
        
        return regions

def main():
    """Main function to extract brain regions."""
    print("H01 Brain Region Extractor")
    print("=" * 50)
    
    # Initialize extractor
    extractor = H01RegionExtractor()
    
    # Get volume info
    try:
        volume_info = extractor.get_volume_info()
        print(f"✓ Connected to H01 dataset")
        print(f"  Volume shape: {volume_info['shape']}")
        print(f"  Voxel size: {volume_info['voxel_size']}")
    except Exception as e:
        print(f"✗ Failed to connect to H01 dataset: {e}")
        return
    
    # List available regions
    print(f"\nAvailable brain regions:")
    regions = extractor.list_available_regions()
    for region in regions:
        print(f"  - {region['name']}: {region['description']}")
        print(f"    Sizes: {', '.join(region['sizes'])}")
    
    # Extract specific regions (you can modify this list)
    target_regions = ["prefrontal_cortex", "hippocampus", "visual_cortex"]
    size = "medium"  # Options: "small", "medium", "large"
    
    print(f"\nExtracting regions: {target_regions}")
    print(f"Size: {size}")
    
    try:
        results = extractor.extract_multiple_regions(target_regions, size)
        
        print(f"\n{'='*50}")
        print(f"EXTRACTION SUMMARY")
        print(f"{'='*50}")
        
        for region_name, data in results.items():
            print(f"✓ {region_name}: {data.shape} - {data.nbytes / (1024*1024):.1f} MB")
        
        print(f"\nTotal extracted: {len(results)} regions")
        total_size = sum(data.nbytes for data in results.values()) / (1024*1024)
        print(f"Total size: {total_size:.1f} MB")
        
    except Exception as e:
        print(f"✗ Extraction failed: {e}")

if __name__ == "__main__":
    main() 