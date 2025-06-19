#!/usr/bin/env python3
"""
Find H01 Data Regions
=====================
Script to find regions in the H01 dataset that contain actual data.
Samples different coordinates to locate non-zero regions.
"""

import numpy as np
import logging
from cloudvolume import CloudVolume
import time
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class H01DataFinder:
    """Find regions in H01 dataset that contain actual data."""
    
    def __init__(self):
        self.cloud_path = "gs://h01-release/data/20210601/4nm_raw"
        self.volume = None
        self._initialize_cloudvolume()
    
    def _initialize_cloudvolume(self):
        """Initialize CloudVolume connection."""
        try:
            logger.info("Initializing CloudVolume connection...")
            self.volume = CloudVolume(
                self.cloud_path,
                mip=0,
                cache="./h01_cache",
                progress=True,
                use_https=True
            )
            logger.info(f"Connected to H01 data: {self.cloud_path}")
            logger.info(f"Volume shape: {self.volume.shape}")
        except Exception as e:
            logger.error(f"Failed to initialize CloudVolume: {e}")
            raise
    
    def sample_region(self, start: List[int], size: Tuple[int, int, int] = (64, 64, 32)) -> Dict:
        """Sample a region to check if it contains data."""
        try:
            stop = [start[i] + size[i] for i in range(3)]
            logger.info(f"Sampling region from {start} to {stop}")
            
            # Extract the region
            region = self.volume[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            
            # Calculate statistics
            non_zero_count = np.count_nonzero(region)
            total_voxels = region.size
            non_zero_percent = (non_zero_count / total_voxels) * 100
            data_range = (int(region.min()), int(region.max()))
            
            result = {
                'start': start,
                'stop': stop,
                'shape': region.shape,
                'non_zero_count': non_zero_count,
                'total_voxels': total_voxels,
                'non_zero_percent': non_zero_percent,
                'data_range': data_range,
                'has_data': non_zero_count > 0
            }
            
            logger.info(f"  Non-zero voxels: {non_zero_count}/{total_voxels} ({non_zero_percent:.2f}%)")
            logger.info(f"  Data range: {data_range}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to sample region {start}: {e}")
            return {
                'start': start,
                'error': str(e),
                'has_data': False
            }
    
    def find_data_regions(self, sample_size: Tuple[int, int, int] = (64, 64, 32)) -> List[Dict]:
        """Find regions that contain actual data."""
        logger.info("Searching for regions with data...")
        
        # Define sampling strategy - try different regions
        sample_coordinates = [
            # Start with smaller regions near the beginning
            [0, 0, 0],
            [1000, 1000, 100],
            [2000, 2000, 200],
            [5000, 5000, 500],
            [10000, 10000, 1000],
            [20000, 20000, 1500],
            [50000, 50000, 2000],
            [100000, 100000, 2500],
            [200000, 200000, 3000],
            [300000, 300000, 3500],
            [400000, 400000, 4000],
            [500000, 500000, 4500],
            
            # Try some random coordinates
            [150000, 250000, 1200],
            [250000, 150000, 1800],
            [350000, 350000, 2200],
            [450000, 450000, 2800],
            
            # Try coordinates from the original regions
            [50000, 300000, 1000],
            [200000, 200000, 2000],
            [400000, 100000, 1500],
            [300000, 250000, 1200],
            [250000, 250000, 1800],
            [150000, 400000, 800],
        ]
        
        results = []
        data_regions = []
        
        for i, coords in enumerate(sample_coordinates):
            logger.info(f"\nSample {i+1}/{len(sample_coordinates)}: {coords}")
            
            result = self.sample_region(coords, sample_size)
            results.append(result)
            
            if result.get('has_data', False):
                data_regions.append(result)
                logger.info(f"  ✓ Found data region!")
            
            # Add delay to avoid overwhelming the server
            time.sleep(1)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"SEARCH SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total samples: {len(results)}")
        logger.info(f"Regions with data: {len(data_regions)}")
        
        if data_regions:
            logger.info(f"\nRegions with data:")
            for i, region in enumerate(data_regions):
                logger.info(f"  {i+1}. Start: {region['start']}")
                logger.info(f"     Non-zero: {region['non_zero_count']}/{region['total_voxels']} ({region['non_zero_percent']:.2f}%)")
                logger.info(f"     Range: {region['data_range']}")
        
        return data_regions
    
    def extract_data_region(self, start: List[int], size: Tuple[int, int, int] = (256, 256, 128)) -> np.ndarray:
        """Extract a region that contains data."""
        logger.info(f"Extracting data region from {start} with size {size}")
        
        stop = [start[i] + size[i] for i in range(3)]
        region = self.volume[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
        
        logger.info(f"Extracted region with shape: {region.shape}")
        logger.info(f"Non-zero voxels: {np.count_nonzero(region)}/{region.size}")
        logger.info(f"Data range: {region.min()} to {region.max()}")
        
        return region

def main():
    """Main function to find data regions."""
    print("H01 Data Region Finder")
    print("=" * 50)
    
    # Initialize finder
    finder = H01DataFinder()
    
    # Find regions with data
    data_regions = finder.find_data_regions()
    
    if data_regions:
        print(f"\n✓ Found {len(data_regions)} regions with data!")
        
        # Extract the best region (highest non-zero percentage)
        best_region = max(data_regions, key=lambda x: x['non_zero_percent'])
        
        print(f"\nExtracting best region:")
        print(f"  Start: {best_region['start']}")
        print(f"  Non-zero: {best_region['non_zero_percent']:.2f}%")
        
        # Extract a larger region around this point
        start_coords = best_region['start']
        extracted_data = finder.extract_data_region(start_coords, (256, 256, 128))
        
        # Save the extracted data
        filename = f"h01_data_region_{start_coords[0]}_{start_coords[1]}_{start_coords[2]}.npy"
        np.save(filename, extracted_data)
        print(f"✓ Saved extracted data: {filename}")
        
    else:
        print("✗ No regions with data found. The coordinates may need adjustment.")

if __name__ == "__main__":
    main() 