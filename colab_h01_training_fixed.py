#!/usr/bin/env python3
"""
H01 FFN-v2 Training in Google Colab
===================================
This script trains the MathematicalFFNv2 model on real H01 connectomics data,
leveraging the H01DataLoader and a YAML configuration.
"""

# ============================================================================
# SETUP AND INSTALLATION
# ============================================================================

print("🚀 Setting up H01 Training Environment...")
!pip install torch torchvision matplotlib numpy scipy tqdm cloud-volume PyYAML -q

# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# Import project-specific modules
# Ensure these files are uploaded: h01_data_loader.py, ffn_v2_mathematical_model.py
try:
    from h01_data_loader import H01DataLoader
    from ffn_v2_mathematical_model import MathematicalFFNv2
    import yaml
    print("✅ Custom modules imported successfully.")
except ImportError as e:
    print(f"❌ Error importing custom modules: {e}")
    print("Please ensure 'h01_data_loader.py' and 'ffn_v2_mathematical_model.py' are uploaded.")
    # Exit or handle the error appropriately if modules are missing
    # For this Colab example, we'll continue but training will fail later
    H01DataLoader = None # Define dummy classes to prevent NameError later
    MathematicalFFNv2 = None
    yaml = None

# ============================================================================
# GPU AND PYTORCH CONFIGURATION
# ============================================================================

print("\n🔍 Configuring GPU and PyTorch...")
if torch.cuda.is_available():
    print("✅ GPU available. Setting device to 'cuda'.")
    device = torch.device('cuda')
    # Optimizations for A100 GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    print("⚠️ GPU not available. Using CPU.")
    device = torch.device('cpu')

print(f"   - Using device: {device}")
print(f"   - PyTorch version: {torch.__version__}")

# ============================================================================
# H01 DATASET CLASS FOR PYTORCH
# ============================================================================

class H01Dataset(Dataset):
    """PyTorch Dataset wrapper for the H01DataLoader."""
    def __init__(self, data_loader: H01DataLoader, region_name: str, samples: int = 100, chunk_size: tuple = (64, 64, 64)):
        if data_loader is None:
            raise ValueError("H01DataLoader is not initialized. Please ensure h01_data_loader.py was imported correctly.")
        self.loader = data_loader
        self.region = self.loader.get_region(region_name)
        self.bounds = self.region['bounds'] # Bounds from config (likely non-voxel)
        self.samples = samples
        self.chunk_size = chunk_size

        # Get volume info from the loader to perform coordinate transformation
        if self.loader.volume is None:
             raise RuntimeError("CloudVolume not initialized in H01DataLoader.")

        # --- Start: Debugging and robust access for voxel info ---
        print(f"\nDEBUG: Type of self.loader.volume: {type(self.loader.volume)}")
        print(f"DEBUG: Content of self.loader.volume: {self.loader.volume}")

        self.voxel_offset = None
        self.voxel_size = None

        # Attempt to get voxel_offset and voxel_size from the volume object
        # Prioritize accessing attributes directly from the volume object as that's where they should be
        if hasattr(self.loader.volume, 'voxel_offset'):
             self.voxel_offset = np.array(self.loader.volume.voxel_offset)
             print(f"DEBUG: Accessed voxel_offset from volume: {self.voxel_offset}")
        elif hasattr(self.loader, 'voxel_offset'): # Fallback to loader attribute if stored there
             self.voxel_offset = np.array(self.loader.voxel_offset)
             print(f"DEBUG: Accessed voxel_offset from loader: {self.voxel_offset}")
        else:
             print("DEBUG: voxel_offset attribute not found on volume or loader. Using default [0, 0, 0]")
             self.voxel_offset = np.array([0, 0, 0]) # Default to origin if not found


        # Attempt to get voxel_size from the volume object
        if hasattr(self.loader.volume, 'voxel_size'):
             self.voxel_size = np.array(self.loader.volume.voxel_size)
             print(f"DEBUG: Accessed voxel_size from volume: {self.voxel_size}")
        elif hasattr(self.loader.volume.meta, 'resolution') and callable(self.loader.volume.meta.resolution): # Check if it's a callable method
             # Call the method to get the actual resolution values
             resolution_values = self.loader.volume.meta.resolution(0) # Pass mip=0 for base resolution
             self.voxel_size = np.array(resolution_values)
             print(f"DEBUG: Accessed voxel_size by calling volume.meta.resolution(0): {resolution_values}")
        elif hasattr(self.loader, 'voxel_size'): # Fallback to loader attribute if stored there
             self.voxel_size = np.array(self.loader.voxel_size)
             print(f"DEBUG: Accessed voxel_size from loader: {self.voxel_size}")
        else:
             print("DEBUG: voxel_size attribute not found on volume, volume.meta (as callable), or loader. Using default [4, 4, 33]")
             self.voxel_size = np.array([4, 4, 33]) # Default to H01 standard if not found


        # Check if we successfully got required info for transformation
        if self.voxel_offset is None or self.voxel_size is None or len(self.voxel_offset) != 3 or len(self.voxel_size) != 3:
             # This check is updated to also verify the length is 3
             raise AttributeError(f"Could not retrieve valid 3D voxel_offset ({self.voxel_offset}) or voxel_size ({self.voxel_size}) for transformation.")
        # --- End: Debugging and robust access for voxel info ---


        # Convert config bounds to voxel bounds
        self.voxel_bounds = self._config_bounds_to_voxel_bounds(self.bounds)

        self.coordinates = self._generate_coordinates()
        print(f"✅ H01 Dataset initialized for region '{region_name}' with {samples} samples.")
        print(f"   - Config bounds: {self.bounds}")
        print(f"   - Voxel bounds for sampling: {self.voxel_bounds}")
        print(f"   - Voxel offset used: {self.voxel_offset}")
        print(f"   - Voxel size used: {self.voxel_size}")


    def _config_bounds_to_voxel_bounds(self, config_bounds):
        """Converts bounds from config coordinate system to voxel coordinates."""
        # Assuming bounds are in physical units (e.g., nanometers) and
        # cloud-volume's voxel_offset and voxel_size are also in corresponding units.
        # Formula: voxel_coord = (config_coord - voxel_offset) / voxel_size

        start_config = np.array(config_bounds[0])
        end_config = np.array(config_bounds[1])
        voxel_offset = np.array(self.voxel_offset)
        voxel_size = np.array(self.voxel_size)

        # --- Start: Debugging shapes before calculation ---
        print(f"DEBUG: Shapes before voxel bounds calculation:")
        print(f"  start_config.shape: {start_config.shape}")
        print(f"  end_config.shape: {end_config.shape}")
        print(f"  voxel_offset.shape: {voxel_offset.shape}")
        print(f"  voxel_size.shape: {voxel_size.shape}")
        # --- End: Debugging shapes before calculation ---

        # Ensure voxel_size is a 3-element array
        if voxel_size.shape != (3,):
             print(f"⚠️ Warning: voxel_size has unexpected shape {voxel_size.shape}. Attempting to reshape to (3,).")
             # Attempt to reshape or slice to get the first 3 elements
             if voxel_size.size >= 3:
                  voxel_size = voxel_size[:3]
                  # Ensure it's a 1D array of shape (3,)
                  if voxel_size.ndim > 1:
                       voxel_size = voxel_size.flatten()[:3]
                  voxel_size = np.array(voxel_size) # Re-create to ensure shape is (3,)
                  print(f"   Reshaped voxel_size. New shape: {voxel_size.shape}")
             else:
                  raise ValueError(f"voxel_size has too few elements ({voxel_size.size}) to reshape to (3,).")


        # Calculate voxel start and end coordinates
        # Ensure division handles potential zero voxel size if necessary, though unlikely for H01
        # The error is likely happening on the line below:
        voxel_start = np.floor((start_config - voxel_offset) / voxel_size).astype(int)
        voxel_end = np.ceil((end_config - voxel_offset) / voxel_size).astype(int)

        # Ensure voxel bounds are within the actual volume shape
        volume_shape = np.array(self.loader.volume.shape)
        voxel_start = np.maximum(voxel_start, [0, 0, 0])
        voxel_end = np.minimum(voxel_end, volume_shape)


        # Ensure end bound is at least start bound + chunk size for valid sampling
        # This might require adjusting the region bounds in the config if they are too small
        min_end_bound = voxel_start + np.array(self.chunk_size)
        # Check each dimension separately for clarity and correct adjustment
        adjusted_voxel_end = np.copy(voxel_end)
        for i in range(3):
             if voxel_end[i] < voxel_start[i] + self.chunk_size[i]:
                  print(f"⚠️ Warning: Voxel bounds end {voxel_end[i]} for dimension {i} is smaller than start {voxel_start[i]} + chunk size {self.chunk_size[i]}. Adjusting end bound.")
                  adjusted_voxel_end[i] = voxel_start[i] + self.chunk_size[i]
                  # Ensure adjustment doesn't exceed volume shape
                  adjusted_voxel_end[i] = min(adjusted_voxel_end[i], volume_shape[i])

        if np.any(adjusted_voxel_end != voxel_end):
             print(f"   Adjusted voxel bounds end for sampling: {adjusted_voxel_end.tolist()}")
        voxel_end = adjusted_voxel_end


        return [voxel_start.tolist(), voxel_end.tolist()]


    def _generate_coordinates(self):
        z_min, y_min, x_min = self.voxel_bounds[0]
        z_max, y_max, x_max = self.voxel_bounds[1]

        # Correctly calculate the maximum possible starting coordinate for each dimension
        # such that the chunk size does not exceed the max bound.
        max_z_start = z_max - self.chunk_size[0]
        max_y_start = y_max - self.chunk_size[1]
        max_x_start = x_max - self.chunk_size[2]

        coords = []
        for _ in range(self.samples):
            # Generate random coordinates within the valid starting range
            # Ensure the range is valid (max_dim_start >= min_dim)
            z = np.random.randint(z_min, max_z_start + 1) if max_z_start >= z_min else z_min
            y = np.random.randint(y_min, max_y_start + 1) if max_y_start >= y_min else y_min
            x = np.random.randint(x_min, max_x_start + 1) if max_x_start >= x_min else x_min


            # Ensure generated coordinates are within the derived voxel bounds
            # Clipping is redundant if randint range is calculated correctly, but kept as a safeguard
            z = np.clip(z, z_min, max_z_start)
            y = np.clip(y, y_min, max_y_start)
            x = np.clip(x, x_min, max_x_start)


            coords.append((int(z), int(y), int(x))) # Ensure coordinates are integers
        return coords

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        coords = self.coordinates[idx]
        try:
            data_chunk = self.loader.load_chunk(coords, self.chunk_size)
        except Exception as e:
            print(f"❌ Error loading chunk at coordinates {coords} with size {self.chunk_size}: {e}")
            # Return dummy data or raise a more specific error if necessary
            # For now, returning zeros to allow the script to continue
            return torch.zeros(1, *self.chunk_size), torch.zeros(1, *self.chunk_size)


        input_tensor = torch.from_numpy(data_chunk.astype(np.float32)).unsqueeze(0)

        # Basic normalization (can be enhanced)
        mean = input_tensor.mean()
        std = input_tensor.std()
        if std > 0:
            input_tensor = (input_tensor - mean) / std
        else:
             # Handle cases with zero standard deviation (e.g., all zeros in the chunk)
            input_tensor = input_tensor - mean


        # Simplified target generation for this quick start.
        # In a real scenario, targets would come from ground truth segmentation.
        target_tensor = torch.randint(0, 2, input_tensor.shape, dtype=torch.float32)

        return input_tensor, target_tensor

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(model, device, train_loader, epochs=5):
    """A simplified training loop."""
    if model is None:
        print("❌ Model not initialized due to import errors. Skipping training.")
        return None

    model.to(device)
    # Using AdamW as in the comprehensive script for consistency and better performance
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Using a combined loss function as in the comprehensive script
    loss_fn = MathematicalLossFunction(alpha=0.5, beta=0.5)


    print("\n🚀 Starting H01 Training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets) # Use the combined loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")

    print("🎉 Training finished.")
    return model

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n🧠 Starting H01 FFN-v2 Training script.")

    config_path = 'h01_config.yaml'
    train_region = 'test_region_1'
    num_samples = 50
    batch_size = 4
    epochs = 3
    chunk_size = (64, 64, 64) # Z, Y, X

    if not os.path.exists(config_path):
        print(f"❌ Config file not found at {config_path}")
        return

    print(f"\n1. Loading H01 data loader with config: {config_path}")
    try:
        if yaml: # Check if yaml was imported successfully
            with open(config_path, 'r') as f:
                h01_config = yaml.safe_load(f)
            if H01DataLoader: # Check if H01DataLoader was imported successfully
                data_loader = H01DataLoader(h01_config)
                print("✅ H01 data loader created successfully.")
            else:
                print("❌ H01DataLoader class not available due to import error.")
                return
        else:
            print("❌ PyYAML not imported successfully. Cannot load config.")
            return
    except Exception as e:
        print(f"❌ Failed to load H01 data loader or config: {e}")
        return

    print(f"\n2. Creating PyTorch dataset for region '{train_region}'...")
    try:
        h01_dataset = H01Dataset(data_loader, train_region, samples=num_samples, chunk_size=chunk_size)
        train_loader = DataLoader(h01_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        print("✅ DataLoader ready.")
    except Exception as e:
        print(f"❌ Failed to create H01 Dataset or DataLoader: {e}")
        return


    print("\n3. Initializing MathematicalFFNv2 model...")
    if MathematicalFFNv2: # Check if MathematicalFFNv2 was imported successfully
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=1,
            hidden_channels=64,
            depth=3
        )
        print("✅ Model initialized.")
    else:
        print("❌ MathematicalFFNv2 class not available due to import error.")
        return


    trained_model = train(model, device, train_loader, epochs=epochs)

    if trained_model:
        model_save_path = 'h01_trained_model.pt'
        print(f"\n5. Saving trained model to {model_save_path}...")
        try:
            torch.save(trained_model.state_dict(), model_save_path)
            print("✅ Model saved.")
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
    else:
        print("\n5. Skipping model saving as training failed.")


    print("\n🏁 H01 training script finished.")

if __name__ == "__main__":
    # Define MathematicalLossFunction here as it's used in the train function
    class MathematicalLossFunction:
        """
        Advanced loss function combining multiple mathematical insights
        Based on optimization theory and probability theory
        """

        def __init__(self, alpha: float = 0.5, beta: float = 0.5):
            self.alpha = alpha  # BCE weight
            self.beta = beta    # Dice weight
            self.bce_loss = nn.BCELoss()

        def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Dice loss for segmentation optimization"""
            smooth = 1e-6

            pred_flat = pred.view(-1)
            target_flat = target.view(-1)

            intersection = (pred_flat * target_flat).sum()
            dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

            return 1 - dice_coeff

        def focal_loss(self, pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
            """Focal loss for handling class imbalance"""
            bce_loss = self.bce_loss(pred, target)
            pt = torch.exp(-bce_loss)
            focal_loss = (1 - pt) ** gamma * bce_loss
            return focal_loss

        def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Combined loss function with mathematical weighting"""
            bce = self.bce_loss(pred, target)
            dice = self.dice_loss(pred, target)
            focal = self.focal_loss(pred, target)

            # Mathematical combination based on optimization theory
            combined_loss = self.alpha * bce + self.beta * dice + 0.1 * focal

            return combined_loss

    main() 