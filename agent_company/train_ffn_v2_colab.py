# ================================================================
# FFN-v2 Training on Google Colab GPU
# ================================================================
# Usage:
#   1. Open https://colab.research.google.com
#   2. Switch Runtime → Change runtime type → GPU (T4 / A100)
#   3. Paste this entire cell and run.
#   4. The notebook will clone your repo, install deps, and kick off training.
#      Checkpoint & history are written to /content/outputs .
# ================================================================

import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime

# ================================================================
# Setup and Installation
# ================================================================

def install_dependencies():
    """Install required packages for Colab GPU training"""
    print("🛠️  Installing dependencies...")
    
    # Install CUDA-enabled PyTorch
    subprocess.run([
        "pip", "-q", "install", 
        "torch==2.2.*+cu121", "torchvision", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], check=True)
    
    # Install other dependencies
    packages = [
        "cloud-volume", "tensorstore", "dask[array]", 
        "prometheus-client", "peft", "accelerate", 
        "matplotlib", "scikit-image", "tqdm", "wandb"
    ]
    
    for package in packages:
        subprocess.run(["pip", "-q", "install", package], check=True)
    
    print("✅ Dependencies installed successfully")

def clone_repository():
    """Clone the repository from GitHub"""
    print("📥 Cloning repository...")
    
    # Update this URL to your actual repository
    repo_url = "https://github.com/alexisvega1/my-app.git"
    
    if not os.path.exists("/content/agent_company"):
        subprocess.run([
            "git", "clone", "-q", repo_url, "/content/agent_company"
        ], check=True)
    
    # Add to Python path
    sys.path.insert(0, "/content/agent_company")
    
    print("✅ Repository cloned successfully")

# ================================================================
# FFN-v2 Model Definition
# ================================================================

class FFNv2Model(nn.Module):
    """FFN-v2 model optimized for 3D neuron segmentation"""
    
    def __init__(self, input_channels=1, num_classes=2, base_channels=32):
        super(FFNv2Model, self).__init__()
        
        # Encoder path
        self.encoder1 = self._make_layer(input_channels, base_channels, 3)
        self.encoder2 = self._make_layer(base_channels, base_channels * 2, 3)
        self.encoder3 = self._make_layer(base_channels * 2, base_channels * 4, 3)
        self.encoder4 = self._make_layer(base_channels * 4, base_channels * 8, 3)
        
        # Decoder path with skip connections
        self.decoder4 = self._make_layer(base_channels * 8, base_channels * 4, 3)
        self.decoder3 = self._make_layer(base_channels * 4, base_channels * 2, 3)
        self.decoder2 = self._make_layer(base_channels * 2, base_channels, 3)
        self.decoder1 = self._make_layer(base_channels, num_classes, 3, final=True)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels, kernel_size, final=False):
        """Create a convolutional layer block"""
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if not final:
            layers.extend([
                nn.Conv3d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Decoder with skip connections
        dec4 = self.decoder4(self.upsample(enc4))
        dec3 = self.decoder3(self.upsample(dec4 + enc3))
        dec2 = self.decoder2(self.upsample(dec3 + enc2))
        dec1 = self.decoder1(self.upsample(dec2 + enc1))
        
        return dec1

# ================================================================
# Data Generation and Loading
# ================================================================

def generate_synthetic_data(batch_size=8, volume_size=64):
    """Generate synthetic 3D data for training"""
    print(f"🎲 Generating synthetic data: {batch_size} batches of {volume_size}³ volumes")
    
    # Generate random 3D volumes
    volumes = np.random.rand(batch_size, 1, volume_size, volume_size, volume_size).astype(np.float32)
    
    # Generate synthetic labels (simple threshold-based segmentation)
    labels = (volumes > 0.5).astype(np.float32)
    
    # Add some noise and structure to make it more realistic
    for i in range(batch_size):
        # Add some structured noise
        noise = np.random.normal(0, 0.1, (volume_size, volume_size, volume_size))
        volumes[i, 0] += noise
        
        # Create some connected components
        from scipy import ndimage
        labels[i, 0] = ndimage.binary_fill_holes(labels[i, 0])
        labels[i, 0] = ndimage.binary_opening(labels[i, 0])
    
    return torch.from_numpy(volumes), torch.from_numpy(labels)

class SyntheticDataset:
    """Synthetic dataset for training"""
    
    def __init__(self, num_samples=1000, volume_size=64):
        self.num_samples = num_samples
        self.volume_size = volume_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate a single sample
        volume, label = generate_synthetic_data(1, self.volume_size)
        return volume.squeeze(0), label.squeeze(0)

# ================================================================
# Training Functions
# ================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, (volumes, labels) in enumerate(dataloader):
        volumes, labels = volumes.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress update
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for volumes, labels in dataloader:
            volumes, labels = volumes.to(device), labels.to(device)
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

# ================================================================
# Main Training Function
# ================================================================

def train_ffn_v2_colab(epochs=20, batch_size=8, volume_size=64, learning_rate=0.001):
    """Main training function for Colab"""
    
    # Setup
    print("🚀 Starting FFN-v2 training on Colab GPU...")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    output_dir = Path("/content/outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = FFNv2Model(input_channels=1, num_classes=2, base_channels=32)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Data loaders
    train_dataset = SyntheticDataset(num_samples=100, volume_size=volume_size)
    val_dataset = SyntheticDataset(num_samples=20, volume_size=volume_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training history
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    print(f"📊 Training for {epochs} epochs with batch size {batch_size}")
    print(f"📦 Dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"✅ Epoch {epoch + 1}/{epochs}")
        print(f"   Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"   Epoch Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss,
                output_dir / "best_ffn_v2_model.pt"
            )
            print(f"💾 Saved best model (loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss,
                output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            )
    
    total_time = time.time() - start_time
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_time': total_time,
        'epochs': epochs,
        'batch_size': batch_size,
        'volume_size': volume_size,
        'device': str(device)
    }
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'o-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📁 Outputs saved to: {output_dir}")
    print(f"💾 Best model: {output_dir}/best_ffn_v2_model.pt")
    print(f"📊 Training history: {output_dir}/training_history.json")
    
    return model, history

# ================================================================
# Colab Integration Functions
# ================================================================

def setup_colab_environment():
    """Setup the complete Colab environment"""
    print("🔧 Setting up Colab environment...")
    
    # Install dependencies
    install_dependencies()
    
    # Clone repository
    clone_repository()
    
    print("✅ Colab environment setup complete!")

def download_results():
    """Download training results from Colab"""
    try:
        from google.colab import files
        
        output_dir = Path("/content/outputs")
        if output_dir.exists():
            # Download the best model
            if (output_dir / "best_ffn_v2_model.pt").exists():
                files.download(str(output_dir / "best_ffn_v2_model.pt"))
            
            # Download training history
            if (output_dir / "training_history.json").exists():
                files.download(str(output_dir / "training_history.json"))
            
            # Download training curves
            if (output_dir / "training_curves.png").exists():
                files.download(str(output_dir / "training_curves.png"))
            
            print("📥 Downloads initiated!")
        else:
            print("❌ No outputs directory found. Run training first.")
    except ImportError:
        print("⚠️  Not running in Colab. Use manual download.")

# ================================================================
# Main Execution
# ================================================================

if __name__ == "__main__":
    # Setup environment
    setup_colab_environment()
    
    # Train the model
    model, history = train_ffn_v2_colab(
        epochs=20,
        batch_size=8,
        volume_size=64,
        learning_rate=0.001
    )
    
    # Download results
    download_results()
    
    print("\n🎯 Training script completed successfully!")
    print("📋 Next steps:")
    print("   1. Download the model checkpoint")
    print("   2. Use the model for inference on your H01 data")
    print("   3. Integrate with your human feedback RL system") 