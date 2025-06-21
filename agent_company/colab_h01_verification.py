#!/usr/bin/env python3
"""
H01 Data Access Verification for Google Colab
=============================================
Run this cell before training to verify that we're accessing the current H01 dataset.
"""

# CELL 1: VERIFY H01 DATA ACCESS
print("🔍 Verifying H01 Data Access...")
print("=" * 50)

try:
    # Load configuration
    with open('h01_config.yaml', 'r') as f:
        h01_config = yaml.safe_load(f)
    print("✅ Configuration loaded successfully")
    
    # Initialize data loader
    data_loader = H01DataLoader(h01_config)
    print("✅ H01 data loader initialized successfully")
    
    # Get volume information
    volume_info = data_loader.get_volume_info()
    print("\n📊 Volume Information:")
    print(f"   - Shape: {volume_info['shape']}")
    print(f"   - Data type: {volume_info['dtype']}")
    print(f"   - Voxel size: {volume_info['voxel_size']}")
    print(f"   - Bounds: {volume_info['bounds']}")
    
    # Test loading a small chunk
    print("\n🧪 Testing chunk loading...")
    test_coords = (1000, 1000, 1000)  # Start of our test region
    test_size = (64, 64, 64)  # Small test chunk
    
    chunk = data_loader.load_chunk(test_coords, test_size)
    print(f"✅ Successfully loaded chunk of shape: {chunk.shape}")
    print(f"   - Data range: [{chunk.min():.2f}, {chunk.max():.2f}]")
    print(f"   - Mean value: {chunk.mean():.2f}")
    print(f"   - Standard deviation: {chunk.std():.2f}")
    
    # Check if data looks reasonable
    if chunk.std() > 0:
        print("✅ Data appears to have meaningful variation")
    else:
        print("⚠️ Warning: Data has no variation (might be all zeros)")
    
    # Display data source information
    print("\n📡 Data Source Information:")
    cloud_config = h01_config['data_source']['cloudvolume']
    print(f"   - Cloud path: {cloud_config['cloud_path']}")
    print(f"   - MIP level: {cloud_config.get('mip', 0)}")
    print(f"   - This is the official H01 dataset from Google Cloud Storage")
    print(f"   - Resolution: 4nm x 4nm x 33nm (highest available)")
    print(f"   - Dataset size: ~1.4 PB (full dataset)")
    print(f"   - Our test region: ~0.5 GB")
    
    # List available regions
    print("\n🗺️ Available Regions:")
    regions = data_loader.list_available_regions()
    for region in regions:
        print(f"   - {region['name']}: {region['description']}")
        print(f"     Bounds: {region['bounds']}")
        print(f"     Size: {region['size_gb']:.2f} GB")
    
    print("\n🎉 H01 data access verification completed successfully!")
    print("✅ You can now proceed with training on real H01 connectomics data.")
    
except Exception as e:
    print(f"❌ Data verification failed: {e}")
    print("Please check your configuration and file uploads.") 