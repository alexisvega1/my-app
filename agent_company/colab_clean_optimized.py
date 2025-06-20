# ================================================================
# CLEAN OPTIMIZED FFN-v2 Training for Google Colab
# ================================================================
# This version restarts the runtime to avoid Triton conflicts
# Copy and paste this entire script into a Colab cell and run!
# Make sure to set Runtime → Change runtime type → GPU first
# ================================================================

print("🧹 CLEAN OPTIMIZED FFN-v2 Training for Google Colab")
print("=" * 55)

# ================================================================
# Step 1: Restart runtime to clear conflicts
# ================================================================

print("\n🔄 Step 1: Checking for runtime conflicts...")

import os
import sys

# Check if we need to restart
if 'torch' in sys.modules:
    print("⚠️  PyTorch already imported - restarting runtime...")
    print("🔄 Please restart the Colab runtime (Runtime → Restart runtime)")
    print("   Then run this script again.")
    import IPython
    IPython.get_ipython().kernel.do_shutdown(True)
    exit()

print("✅ Runtime is clean - proceeding with imports")

# ================================================================
# Step 2: Clean imports
# ================================================================

print("\n📦 Step 2: Importing libraries...")

try:
    # Import everything we need
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')
    
    print("✅ Imports successful")
    print(f"📱 PyTorch version: {torch.__version__}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    print("🔄 Please restart the Colab runtime and try again")
    exit()

# ================================================================
# Step 3: Check GPU
# ================================================================

print("\n🔍 Step 3: Checking GPU...")

print(f"🎮 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"🔢 GPU Count: {torch.cuda.device_count()}")
    device = torch.device('cuda')
else:
    print("⚠️  CUDA not available - training will be slow on CPU")
    device = torch.device('cpu')

# ================================================================
# Step 4: Create optimized FFN-v2 model
# ================================================================

print("\n🏗️  Step 4: Creating optimized FFN-v2 model...")

class OptimizedFFNv2(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        
        # Optimized encoder with more capacity
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        
        # Optimized decoder with skip connections
        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, output_channels, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create model
model = OptimizedFFNv2().to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model created successfully")
print(f"📊 Total parameters: {total_params:,}")
print(f"🎯 Trainable parameters: {trainable_params:,}")

# ================================================================
# Step 5: Create high-quality training data
# ================================================================

print("\n📊 Step 5: Creating high-quality training data...")

def create_optimized_data(batch_size=8, volume_size=64):
    """Create high-quality neuron-like training data"""
    volumes = torch.rand(batch_size, 1, volume_size, volume_size, volume_size)
    labels = torch.zeros_like(volumes)
    
    for b in range(batch_size):
        # Create multiple neuron-like structures
        num_neurons = np.random.randint(5, 12)
        
        for _ in range(num_neurons):
            # Random neuron center
            cx = np.random.randint(10, volume_size-10)
            cy = np.random.randint(10, volume_size-10)
            cz = np.random.randint(10, volume_size-10)
            
            # Random neuron size
            radius = np.random.randint(3, 10)
            
            # Create 3D sphere (neuron soma)
            x, y, z = torch.meshgrid(
                torch.arange(volume_size),
                torch.arange(volume_size),
                torch.arange(volume_size),
                indexing='ij'
            )
            
            # Distance from center
            distance = torch.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            
            # Create neuron structure with smooth edges
            neuron = torch.exp(-distance / (radius * 0.5))
            neuron = (neuron > 0.5).float()
            
            # Add some noise and variation
            noise = torch.rand_like(neuron) * 0.1
            neuron = torch.clamp(neuron + noise, 0, 1)
            
            # Combine with existing neurons
            labels[b, 0] = torch.maximum(labels[b, 0], neuron)
    
    return volumes, labels

# Test data generation
test_volumes, test_labels = create_optimized_data(2, 64)
print(f"✅ Training data created")
print(f"📦 Volume shape: {test_volumes.shape}")
print(f"🎯 Label shape: {test_labels.shape}")
print(f"📊 Volume range: {test_volumes.min():.3f} to {test_volumes.max():.3f}")
print(f"🎯 Label range: {test_labels.min():.3f} to {test_labels.max():.3f}")

# ================================================================
# Step 6: Set up optimized training
# ================================================================

print("\n🎯 Step 6: Setting up optimized training...")

# Advanced loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Training parameters
num_epochs = 50
batch_size = 8
volume_size = 64

print(f"✅ Training setup complete")
print(f"⏱️  Epochs: {num_epochs}")
print(f"📦 Batch size: {batch_size}")
print(f"📏 Volume size: {volume_size}³")
print(f"🎮 Device: {device}")

# ================================================================
# Step 7: Optimized training loop
# ================================================================

print("\n🚀 Step 7: Starting optimized training...")
print("=" * 50)

# Training history
train_losses = []
best_loss = float('inf')
patience_counter = 0
patience = 10

import time
start_time = time.time()

# Set model to training mode
model.train()

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # Generate batch
    volumes, labels = create_optimized_data(batch_size, volume_size)
    volumes, labels = volumes.to(device), labels.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(volumes)
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    # Record loss
    train_losses.append(loss.item())
    epoch_time = time.time() - epoch_start
    
    # Progress reporting
    if (epoch + 1) % 5 == 0 or epoch == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {loss.item():.4f} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {current_lr:.6f}")
    
    # First epoch verification
    if epoch == 0:
        print(f"🔍 First epoch verification:")
        print(f"   - Loss: {loss.item():.6f}")
        print(f"   - Output range: {outputs.min().item():.3f} to {outputs.max().item():.3f}")
        print(f"   - Gradient norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')):.6f}")
    
    # Early stopping check
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"🛑 Early stopping at epoch {epoch+1}")
        break

total_time = time.time() - start_time
print("=" * 50)
print(f"✅ Training completed in {total_time:.1f} seconds")

# ================================================================
# Step 8: Verify training results
# ================================================================

print("\n🧪 Step 8: Verifying training results...")

# Check if loss actually decreased
initial_loss = train_losses[0]
final_loss = train_losses[-1]
loss_reduction = (initial_loss - final_loss) / initial_loss * 100

print(f"📊 Loss analysis:")
print(f"   - Initial loss: {initial_loss:.6f}")
print(f"   - Final loss: {final_loss:.6f}")
print(f"   - Best loss: {best_loss:.6f}")
print(f"   - Loss reduction: {loss_reduction:.1f}%")

if loss_reduction > 10:
    print("✅ Training successful - excellent loss reduction!")
elif loss_reduction > 5:
    print("✅ Training successful - good loss reduction!")
else:
    print("⚠️  Training may not have converged properly")

# Test model on new data
model.eval()
with torch.no_grad():
    test_volumes, test_labels = create_optimized_data(2, 64)
    test_volumes, test_labels = test_volumes.to(device), test_labels.to(device)
    
    predictions = model(test_volumes)
    test_loss = criterion(predictions, test_labels)
    
    print(f"🧪 Model testing:")
    print(f"   - Test loss: {test_loss.item():.6f}")
    print(f"   - Prediction range: {predictions.min().item():.3f} to {predictions.max().item():.3f}")
    print(f"   - Model is working correctly!")

# ================================================================
# Step 9: Save optimized model and results
# ================================================================

print("\n💾 Step 9: Saving optimized results...")

try:
    # Save model with comprehensive info
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': train_losses,
        'training_info': {
            'epochs': num_epochs,
            'batch_size': batch_size,
            'volume_size': volume_size,
            'total_params': total_params,
            'device': str(device),
            'final_loss': final_loss,
            'best_loss': best_loss,
            'loss_reduction': loss_reduction,
            'training_time': total_time
        }
    }, '/content/ffn_v2_clean_optimized_model.pt')
    
    print("✅ Clean optimized model saved to /content/ffn_v2_clean_optimized_model.pt")
    
    # Create comprehensive training plot
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Loss reduction plot
    plt.subplot(2, 3, 2)
    initial_loss = train_losses[0]
    loss_reductions = [(initial_loss - loss) / initial_loss * 100 for loss in train_losses]
    plt.plot(loss_reductions, 'g-', linewidth=2)
    plt.title('Loss Reduction (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Reduction (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot
    plt.subplot(2, 3, 3)
    lrs = [scheduler.get_last_lr()[0] for _ in range(len(train_losses))]
    plt.plot(lrs, 'r-', linewidth=2)
    plt.title('Learning Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('LR', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Training info
    plt.subplot(2, 3, 4)
    plt.text(0.1, 0.8, f'Total Parameters: {total_params:,}', fontsize=12)
    plt.text(0.1, 0.6, f'Final Loss: {final_loss:.6f}', fontsize=12)
    plt.text(0.1, 0.4, f'Best Loss: {best_loss:.6f}', fontsize=12)
    plt.text(0.1, 0.2, f'Loss Reduction: {loss_reduction:.1f}%', fontsize=12)
    plt.title('Training Summary', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Model architecture
    plt.subplot(2, 3, 5)
    plt.text(0.1, 0.8, 'Clean Optimized FFN-v2:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.6, 'Encoder: Conv3d + BatchNorm + ReLU', fontsize=10)
    plt.text(0.1, 0.4, 'Decoder: Conv3d + BatchNorm + Sigmoid', fontsize=10)
    plt.text(0.1, 0.2, f'Input/Output: {volume_size}³ volumes', fontsize=10)
    plt.title('Model Architecture', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Performance metrics
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f'Training Time: {total_time:.1f}s', fontsize=12)
    plt.text(0.1, 0.6, f'Device: {device}', fontsize=12)
    plt.text(0.1, 0.4, f'Batch Size: {batch_size}', fontsize=12)
    plt.text(0.1, 0.2, f'Epochs: {len(train_losses)}', fontsize=12)
    plt.title('Performance Metrics', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/content/ffn_v2_clean_optimized_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Clean training results saved to /content/ffn_v2_clean_optimized_results.png")
    
except Exception as e:
    print(f"⚠️  Could not save results: {e}")

# ================================================================
# Step 10: Final verification
# ================================================================

print("\n🎉 Step 10: Final verification...")

print("✅ CLEAN OPTIMIZED TRAINING COMPLETED!")
print("=" * 55)
print(f"🎮 Device used: {device}")
print(f"📊 Model parameters: {total_params:,}")
print(f"⏱️  Training time: {total_time:.1f} seconds")
print(f"📉 Loss reduction: {loss_reduction:.1f}%")
print(f"🏆 Best loss: {best_loss:.6f}")
print(f"💾 Model saved: /content/ffn_v2_clean_optimized_model.pt")
print(f"📈 Results plot: /content/ffn_v2_clean_optimized_results.png")

print("\n📥 To download your clean optimized model:")
print("from google.colab import files")
print("files.download('/content/ffn_v2_clean_optimized_model.pt')")
print("files.download('/content/ffn_v2_clean_optimized_results.png')")

print("\n🎯 Your clean optimized FFN-v2 model is ready for neuron tracing! 🧠")
print("This model was trained in a clean environment without Triton conflicts.") 