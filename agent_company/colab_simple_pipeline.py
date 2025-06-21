# ============================================================================
# Simplified Connectomics Pipeline for Colab
# ============================================================================
# This is a simplified version that can be run step by step
# ============================================================================

# Step 1: Environment Setup
print("Step 1: Setting up environment...")

# Install dependencies
get_ipython().system('pip install cloud-volume==11.0.0 gcsfs -q')
get_ipython().system('pip install hydra-core omegaconf timm iopath fvcore -q')
get_ipython().system('pip install opencv-python matplotlib -q')
get_ipython().system('pip install monai>=0.9.1 -q')
get_ipython().system('pip install scipy>=1.5 scikit-learn>=0.23.1 scikit-image>=0.17.2 -q')
get_ipython().system('pip install Cython>=0.29.22 yacs>=0.1.8 h5py>=2.10.0 -q')
get_ipython().system('pip install gputil>=1.4.0 imageio>=2.9.0 tensorboard>=2.2.2 -q')
get_ipython().system('pip install einops>=0.3.0 tqdm>=4.58.0 -q')


# Clone repositories
print("Cloning repositories...")
get_ipython().system('rm -rf optimizers pytorch_connectomics sam2')
get_ipython().system('git clone https://github.com/facebookresearch/optimizers.git')
get_ipython().system('git clone https://github.com/zudi-lin/pytorch_connectomics.git')
get_ipython().system('git clone https://github.com/facebookresearch/sam2.git')

# Add to path
import sys
sys.path.extend(['optimizers', 'pytorch_connectomics', 'sam2'])

# Download SAM 2 checkpoint
get_ipython().system('mkdir -p checkpoints')
get_ipython().system('curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -o checkpoints/sam2_hiera_t.pt')

print("✓ Environment setup complete!")

# Step 2: Test Imports
print("\nStep 2: Testing imports...")

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    print("✓ Basic imports")
except Exception as e:
    print(f"✗ Basic imports: {e}")

try:
    from sam2.build_sam import build_sam2
    print("✓ SAM 2 imports")
except Exception as e:
    print(f"✗ SAM 2 imports: {e}")

try:
    from optimizers.distributed_shampoo import DistributedShampoo
    print("✓ Optimizer imports")
except Exception as e:
    print(f"✗ Optimizer imports: {e}")

try:
    from pytorch_connectomics.connectomics.config import get_cfg_defaults
    print("✓ PyTC imports")
except Exception as e:
    print(f"✗ PyTC imports: {e}")

try:
    from h01_data_loader import H01DataLoader
    from ffn_v2_mathematical_model import MathematicalFFNv2
    print("✓ Local imports")
except Exception as e:
    print(f"✗ Local imports: {e}")

print("✓ Import test complete!")

# Step 3: Test SAM 2 Setup
print("\nStep 3: Testing SAM 2 setup...")

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if config file exists
    import os
    config_path = "sam2/configs/sam2.1/sam2.1_hiera_t"
    if os.path.exists(config_path):
        print(f"✓ Config file found: {config_path}")
    else:
        print(f"✗ Config file not found: {config_path}")
        # List available configs
        config_dir = "sam2/configs/sam2.1/"
        if os.path.exists(config_dir):
            print("Available configs:")
            for f in os.listdir(config_dir):
                print(f"  - {f}")
    
    # Check if checkpoint exists
    ckpt_path = "checkpoints/sam2_hiera_t.pt"
    if os.path.exists(ckpt_path):
        print(f"✓ Checkpoint found: {ckpt_path}")
    else:
        print(f"✗ Checkpoint not found: {ckpt_path}")
    
except Exception as e:
    print(f"✗ SAM 2 setup test failed: {e}")

print("✓ SAM 2 test complete!")

# Step 4: Test Data Loader
print("\nStep 4: Testing data loader...")

try:
    # Simple config for testing
    test_config = {
        'dataset': 'h01',
        'volume_path': 'graphene://h01-release/',
        'mip': 1,
        'chunk_size': [64, 64, 64]
    }
    
    data_loader = H01DataLoader(test_config)
    print("✓ Data loader created successfully")
    
    # Try to load a small test chunk
    test_coords = [400000, 400000, 4000]
    test_size = [64, 64, 64]
    
    try:
        test_data = data_loader.load_chunk(test_coords, test_size)
        print(f"✓ Test data loaded: {test_data.shape}")
    except Exception as e:
        print(f"⚠ Test data loading failed (this is expected if no H01 access): {e}")
    
except Exception as e:
    print(f"✗ Data loader test failed: {e}")

print("✓ Data loader test complete!")

print("\n" + "="*60)
print("SIMPLIFIED PIPELINE SETUP COMPLETE")
print("="*60)
print("If all tests passed, you can now run the complete pipeline.")
print("If any tests failed, check the error messages above.") 