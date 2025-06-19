from cloudvolume import CloudVolume
import numpy as np
import os

# Path to your local H01 dataset
vol_path = 'file:///Users/alexisvega/my-app/agent_company/h01_cache/gs/h01-release/data/20210601/4nm_raw'

# Use highest resolution (mip=0) and enable local cache
volume = CloudVolume(vol_path, mip=0, cache="./h01_cache", progress=True)

print(f"Volume shape: {volume.shape}")
print(f"Volume dtype: {volume.dtype}")

# Define small, traceable regions from the beginning of the dataset
# These are much smaller and should be more likely to be available
regions = {
    'test_region_1': ([0, 0, 0], [128, 128, 64]),      # Small test region
    'test_region_2': ([128, 128, 64], [256, 256, 128]), # Slightly larger region
    'test_region_3': ([0, 0, 64], [256, 256, 128]),    # Overlapping region for testing
}

for name, (start, stop) in regions.items():
    print(f'\nExtracting {name} from {start} to {stop}...')
    try:
        subvol = volume[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
        np.save(f'h01_{name}.npy', subvol)
        print(f'✓ Saved h01_{name}.npy with shape: {subvol.shape}')
        print(f'  Data range: {subvol.min():.3f} to {subvol.max():.3f}')
        print(f'  Non-zero voxels: {np.count_nonzero(subvol)} / {subvol.size}')
        
        # Check if data is traceable (has some structure)
        if np.count_nonzero(subvol) > 0:
            print(f'  ✓ Data appears traceable (contains non-zero values)')
        else:
            print(f'  ⚠ Data is all zeros - may not be traceable')
            
    except Exception as e:
        print(f'✗ Failed to extract {name}: {e}')
        continue

print(f'\nExtraction complete! Check the generated .npy files.') 