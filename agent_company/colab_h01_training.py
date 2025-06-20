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
warnings.filterwarnings('ignore')

# Import project-specific modules
from h01_data_loader import H01DataLoader
from ffn_v2_mathematical_model import MathematicalFFNv2
import yaml

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
        self.loader = data_loader
        self.region = self.loader.get_region(region_name)
        self.bounds = self.region['bounds']
        self.samples = samples
        self.chunk_size = chunk_size
        
        # Pre-generate random coordinates for fetching chunks
        self.coordinates = self._generate_coordinates()
        print(f"✅ H01 Dataset initialized for region '{region_name}' with {samples} samples.")

    def _generate_coordinates(self):
        z_min, y_min, x_min = self.bounds[0]
        z_max, y_max, x_max = self.bounds[1]
        
        # Ensure we don't sample outside the bounds when considering chunk size
        z_range = z_max - z_min - self.chunk_size[0]
        y_range = y_max - y_min - self.chunk_size[1]
        x_range = x_max - x_min - self.chunk_size[2]

        coords = []
        for _ in range(self.samples):
            z = np.random.randint(0, z_range) + z_min
            y = np.random.randint(0, y_range) + y_min
            x = np.random.randint(0, x_range) + x_min
            coords.append((z, y, x))
        return coords

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        coords = self.coordinates[idx]
        # Load a chunk of data (this will be our input)
        data_chunk = self.loader.load_chunk(coords, self.chunk_size)
        
        # Normalize and convert to tensor
        input_tensor = torch.from_numpy(data_chunk.astype(np.float32)).unsqueeze(0)
        input_tensor = (input_tensor - input_tensor.mean()) / input_tensor.std()

        # For this example, we'll create a dummy target.
        # In a real scenario, you would load a corresponding segmentation label chunk.
        target_tensor = torch.randint(0, 2, input_tensor.shape, dtype=torch.float32)
        
        return input_tensor, target_tensor

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(model, device, train_loader, epochs=5):
    """A simplified training loop."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() # Numerically stable version of BCE
    
    print("\n🚀 Starting H01 Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 10 == 0: # Print progress every 10 batches
                 print(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} complete. Average Loss: {epoch_loss / len(train_loader):.4f}")

    print("🎉 Training finished.")
    return model

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n🧠 Starting H01 FFN-v2 Training script.")
    
    # --- Configuration ---
    config_path = 'h01_config.yaml'
    train_region = 'test_region_1'
    num_samples = 50
    batch_size = 4
    epochs = 3
    chunk_size = (64, 64, 64)

    # 1. Load H01 Data Loader
    print(f"\n1. Loading H01 data loader with config: {config_path}")
    try:
        with open(config_path, 'r') as f:
            h01_config = yaml.safe_load(f)
        data_loader = H01DataLoader(h01_config)
        print("✅ H01 data loader created successfully.")
    except Exception as e:
        print(f"❌ Failed to load H01 data loader: {e}")
        return

    # 2. Create PyTorch Dataset and DataLoader
    print(f"\n2. Creating PyTorch dataset for region '{train_region}'...")
    h01_dataset = H01Dataset(data_loader, train_region, samples=num_samples, chunk_size=chunk_size)
    train_loader = DataLoader(h01_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print("✅ DataLoader ready.")

    # 3. Initialize Model
    print("\n3. Initializing MathematicalFFNv2 model...")
    model = MathematicalFFNv2(
        input_channels=1,
        output_channels=1,
        hidden_channels=64,
        depth=3
    )
    print("✅ Model initialized.")

    # 4. Run Training
    trained_model = train(model, device, train_loader, epochs=epochs)

    # 5. Save the trained model
    model_save_path = 'h01_trained_model.pt'
    print(f"\n5. Saving trained model to {model_save_path}...")
    torch.save(trained_model.state_dict(), model_save_path)
    print("✅ Model saved.")

    print("\n🏁 H01 training script finished successfully!")
    print(f"   To use this model, load '{model_save_path}' in your production pipeline.")

if __name__ == "__main__":
    # This check ensures the script runs when executed, e.g., in Colab
    main() 