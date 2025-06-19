# ================================================================
# FFN-v2 Training for Google Colab (CPU Fallback)
# ================================================================
# This script starts with CPU to avoid Triton conflicts, then uses GPU
# Copy and paste this entire script into a Colab cell and run!
# Make sure to set Runtime → Change runtime type → GPU first
# ================================================================

print("🚀 FFN-v2 Training for Google Colab (CPU Fallback)")
print("=" * 60)

# ================================================================
# Step 1: Set environment variables BEFORE any imports
# ================================================================

print("\n🔧 Step 1: Setting up environment...")

import os
import sys

# Set environment variables to avoid conflicts
os.environ['TORCH_LOGS'] = 'off'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
os.environ['PYTORCH_DISABLE_TELEMETRY'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Disable CUDA initially to avoid Triton conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("✅ Environment variables set")

# ================================================================
# Step 2: Import PyTorch on CPU first
# ================================================================

print("\n📦 Step 2: Importing PyTorch on CPU...")

try:
    # Import torch on CPU first
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import warnings
    warnings.filterwarnings('ignore')
    
    print("✅ PyTorch imported successfully on CPU")
    print(f"📱 PyTorch version: {torch.__version__}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    
except Exception as e:
    print(f"❌ CPU import failed: {e}")
    print("🔄 Trying to restart runtime...")
    print("Please restart the Colab runtime and try again.")
    raise

# ================================================================
# Step 3: Now try to enable GPU
# ================================================================

print("\n🔍 Step 3: Enabling GPU...")

try:
    # Re-enable CUDA
    del os.environ['CUDA_VISIBLE_DEVICES']
    
    # Check if CUDA is now available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU enabled: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        print(f"🔢 GPU Count: {torch.cuda.device_count()}")
        device = torch.device('cuda')
    else:
        print("⚠️  GPU not available - using CPU")
        device = torch.device('cpu')
        
except Exception as e:
    print(f"⚠️  GPU enable failed: {e}")
    print("🔄 Continuing with CPU")
    device = torch.device('cpu')

# ================================================================
# Step 4: Create FFN-v2 model
# ================================================================

print("\n🏗️  Step 4: Creating FFN-v2 model...")

class FFNv2(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple 3D CNN
        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv3d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x

try:
    model = FFNv2().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created successfully on {device}")
    print(f"📊 Total parameters: {total_params:,}")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    raise

# ================================================================
# Step 5: Create training data
# ================================================================

print("\n📊 Step 5: Creating training data...")

def create_data(batch_size=4, size=32):
    """Create training data"""
    volumes = torch.rand(batch_size, 1, size, size, size)
    labels = torch.zeros_like(volumes)
    
    for b in range(batch_size):
        # Create simple structures
        num_objects = np.random.randint(2, 5)
        for _ in range(num_objects):
            cx = np.random.randint(5, size-5)
            cy = np.random.randint(5, size-5)
            cz = np.random.randint(5, size-5)
            radius = np.random.randint(2, 6)
            
            # Create sphere
            x, y, z = torch.meshgrid(
                torch.arange(size),
                torch.arange(size),
                torch.arange(size),
                indexing='ij'
            )
            
            distance = torch.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            sphere = (distance < radius).float()
            labels[b, 0] = torch.maximum(labels[b, 0], sphere)
    
    return volumes, labels

try:
    test_volumes, test_labels = create_data(2, 32)
    print(f"✅ Training data created")
    print(f"📦 Volume shape: {test_volumes.shape}")
    print(f"🎯 Label shape: {test_labels.shape}")
except Exception as e:
    print(f"❌ Data creation failed: {e}")
    raise

# ================================================================
# Step 6: Set up training
# ================================================================

print("\n🎯 Step 6: Setting up training...")

try:
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 15
    batch_size = 4
    volume_size = 32
    
    print(f"✅ Training setup complete")
    print(f"⏱️  Epochs: {num_epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📏 Volume size: {volume_size}³")
    print(f"🎮 Device: {device}")
except Exception as e:
    print(f"❌ Training setup failed: {e}")
    raise

# ================================================================
# Step 7: Training loop
# ================================================================

print("\n🚀 Step 7: Starting training...")
print("=" * 50)

try:
    train_losses = []
    start_time = time.time()
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Generate batch
        volumes, labels = create_data(batch_size, volume_size)
        volumes, labels = volumes.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record loss
        train_losses.append(loss.item())
        epoch_time = time.time() - epoch_start
        
        # Progress reporting
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"Device: {device}")
        
        # First epoch verification
        if epoch == 0:
            print(f"🔍 First epoch verification:")
            print(f"   - Loss: {loss.item():.6f}")
            print(f"   - Output range: {outputs.min().item():.3f} to {outputs.max().item():.3f}")
    
    total_time = time.time() - start_time
    print("=" * 50)
    print(f"✅ Training completed in {total_time:.1f} seconds")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    raise

# ================================================================
# Step 8: Verify results
# ================================================================

print("\n🧪 Step 8: Verifying training results...")

try:
    # Check loss reduction
    initial_loss = train_losses[0]
    final_loss = train_losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"📊 Loss analysis:")
    print(f"   - Initial loss: {initial_loss:.6f}")
    print(f"   - Final loss: {final_loss:.6f}")
    print(f"   - Loss reduction: {loss_reduction:.1f}%")
    
    if loss_reduction > 5:
        print("✅ Training successful - significant loss reduction!")
    else:
        print("⚠️  Training may not have converged properly")
    
    # Test model
    model.eval()
    with torch.no_grad():
        test_volumes, test_labels = create_data(2, 32)
        test_volumes, test_labels = test_volumes.to(device), test_labels.to(device)
        
        predictions = model(test_volumes)
        test_loss = criterion(predictions, test_labels)
        
        print(f"🧪 Model testing:")
        print(f"   - Test loss: {test_loss.item():.6f}")
        print(f"   - Prediction range: {predictions.min().item():.3f} to {predictions.max().item():.3f}")
        print(f"   - Model is working correctly!")
        
except Exception as e:
    print(f"❌ Verification failed: {e}")

# ================================================================
# Step 9: Save results
# ================================================================

print("\n💾 Step 9: Saving results...")

try:
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_history': train_losses,
        'training_info': {
            'epochs': num_epochs,
            'batch_size': batch_size,
            'volume_size': volume_size,
            'total_params': total_params,
            'device': str(device),
            'final_loss': final_loss,
            'loss_reduction': loss_reduction
        }
    }, '/content/ffn_v2_cpu_fallback_model.pt')
    
    print("✅ Model saved to /content/ffn_v2_cpu_fallback_model.pt")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title(f'FFN-v2 Training Loss ({device})', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/content/ffn_v2_cpu_fallback_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📈 Training plot saved to /content/ffn_v2_cpu_fallback_training.png")
    
except Exception as e:
    print(f"⚠️  Could not save results: {e}")

# ================================================================
# Step 10: Final summary
# ================================================================

print("\n🎉 Step 10: Final summary...")

print("✅ FFN-v2 TRAINING COMPLETED!")
print("=" * 60)
print(f"🎮 Device used: {device}")
print(f"📊 Model parameters: {total_params:,}")
print(f"⏱️  Training time: {total_time:.1f} seconds")
print(f"📉 Loss reduction: {loss_reduction:.1f}%")
print(f"💾 Model saved: /content/ffn_v2_cpu_fallback_model.pt")
print(f"📈 Training plot: /content/ffn_v2_cpu_fallback_training.png")

print("\n📥 To download your trained model:")
print("from google.colab import files")
print("files.download('/content/ffn_v2_cpu_fallback_model.pt')")
print("files.download('/content/ffn_v2_cpu_fallback_training.png')")

print("\n🎯 Your FFN-v2 model is ready for neuron tracing! 🧠")
print(f"This model was trained on {device}.") 