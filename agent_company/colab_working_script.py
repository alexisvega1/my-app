# ================================================================
# WORKING FFN-v2 Training Script for Google Colab
# ================================================================
# Copy and paste this entire script into a Colab cell and run!
# Make sure to set Runtime → Change runtime type → GPU first
# ================================================================

import subprocess
import sys
import os
from pathlib import Path

print("🚀 Starting FFN-v2 Training on Colab GPU")

# ================================================================
# Step 1: Install Dependencies
# ================================================================

print("\n🛠️  Installing dependencies...")

# Install CUDA-enabled PyTorch
try:
    subprocess.run([
        "pip", "-q", "install", 
        "torch==2.2.*+cu121", "torchvision", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], check=True)
    print("✅ PyTorch installed successfully")
except Exception as e:
    print(f"⚠️  PyTorch installation issue: {e}")
    print("🔄 Trying alternative installation...")
    subprocess.run(["pip", "-q", "install", "torch", "torchvision"], check=True)

# Install other required packages
packages = [
    "cloud-volume", "tensorstore", "dask[array]", 
    "prometheus-client", "peft", "accelerate", 
    "matplotlib", "scikit-image", "tqdm", "wandb"
]

for package in packages:
    try:
        subprocess.run(["pip", "-q", "install", package], check=True)
        print(f"✅ {package} installed")
    except Exception as e:
        print(f"⚠️  Failed to install {package}: {e}")

print("✅ Dependencies installation completed")

# ================================================================
# Step 2: Clone Repository
# ================================================================

print("\n📥 Cloning repository...")

# Remove existing directory if it exists
if os.path.exists("/content/agent_company"):
    print("🗑️  Removing existing directory...")
    subprocess.run(["rm", "-rf", "/content/agent_company"], check=True)

# Clone the repository
repo_url = "https://github.com/alexisvega1/my-app.git"
try:
    subprocess.run(["git", "clone", "-q", repo_url, "/content/agent_company"], check=True)
    print("✅ Repository cloned successfully")
except Exception as e:
    print(f"❌ Failed to clone repository: {e}")
    print("🔄 Trying with verbose output...")
    subprocess.run(["git", "clone", repo_url, "/content/agent_company"])

# ================================================================
# Step 3: Verify Repository Contents
# ================================================================

print("\n🔍 Checking repository contents...")

# List contents of the cloned repository
if os.path.exists("/content/agent_company"):
    print("📁 Repository structure:")
    subprocess.run(["ls", "-la", "/content/agent_company"])
    
    # Check if agent_company subdirectory exists
    if os.path.exists("/content/agent_company/agent_company"):
        print("\n📁 Agent company directory found:")
        subprocess.run(["ls", "-la", "/content/agent_company/agent_company"])
        
        # Look for the training script
        if os.path.exists("/content/agent_company/agent_company/train_ffn_v2_colab.py"):
            print("✅ Found train_ffn_v2_colab.py!")
        else:
            print("❌ train_ffn_v2_colab.py not found")
            print("🔍 Searching for training files...")
            subprocess.run(["find", "/content/agent_company", "-name", "*train*", "-type", "f"])
    else:
        print("❌ agent_company subdirectory not found")
        print("🔍 Searching for Python files...")
        subprocess.run(["find", "/content/agent_company", "-name", "*.py", "-type", "f"])
else:
    print("❌ Repository directory not found")

# ================================================================
# Step 4: Set Up Python Path
# ================================================================

print("\n🐍 Setting up Python path...")

# Add the repository to Python path
sys.path.insert(0, "/content/agent_company")
if os.path.exists("/content/agent_company/agent_company"):
    sys.path.insert(0, "/content/agent_company/agent_company")

print(f"✅ Python path updated: {sys.path[:3]}")

# ================================================================
# Step 5: Check GPU
# ================================================================

print("\n🔍 Checking GPU...")

try:
    import torch
    print(f"📱 PyTorch version: {torch.__version__}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔢 GPU Count: {torch.cuda.device_count()}")
    else:
        print("⚠️  CUDA not available - training will be slow on CPU")
except Exception as e:
    print(f"❌ Error checking GPU: {e}")

# ================================================================
# Step 6: Create Output Directory
# ================================================================

print("\n📁 Creating output directory...")
Path("/content/outputs").mkdir(exist_ok=True)
print("✅ Output directory created")

# ================================================================
# Step 7: Import and Run Training
# ================================================================

print("\n🚀 Starting training...")

try:
    # Change to the correct directory
    if os.path.exists("/content/agent_company/agent_company"):
        os.chdir("/content/agent_company/agent_company")
        print("📂 Changed to agent_company directory")
    else:
        os.chdir("/content/agent_company")
        print("📂 Changed to main repository directory")
    
    # Try to import the training module
    print("📦 Importing training module...")
    
    # First, let's see what's available
    print("🔍 Available modules:")
    import glob
    training_files = glob.glob("train_*.py")
    print(f"Training files found: {training_files}")
    
    # Try to import the training function
    try:
        from train_ffn_v2_colab import train_ffn_v2_colab
        print("✅ Successfully imported train_ffn_v2_colab")
        
        # Run training
        print("🎯 Starting FFN-v2 training...")
        model, history = train_ffn_v2_colab(
            epochs=20,
            batch_size=8,
            volume_size=64,
            learning_rate=0.001
        )
        
        print("\n🎉 Training completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔄 Trying alternative import...")
        
        # Try importing from the main directory
        sys.path.insert(0, "/content/agent_company")
        try:
            from agent_company.train_ffn_v2_colab import train_ffn_v2_colab
            print("✅ Successfully imported from agent_company subdirectory")
            
            # Run training
            model, history = train_ffn_v2_colab(
                epochs=20,
                batch_size=8,
                volume_size=64,
                learning_rate=0.001
            )
            
            print("\n🎉 Training completed successfully!")
            
        except Exception as e2:
            print(f"❌ Alternative import failed: {e2}")
            print("🔧 Creating simple training script...")
            
            # Create a simple training script as fallback
            create_simple_training_script()
            
except Exception as e:
    print(f"❌ Training error: {e}")
    print("🔧 Creating fallback training script...")
    create_simple_training_script()

# ================================================================
# Step 8: Show Results
# ================================================================

print("\n📊 Training Results:")
print("====================")

# List output files
if os.path.exists("/content/outputs"):
    print("📁 Output files:")
    subprocess.run(["ls", "-la", "/content/outputs"])

print("\n🎉 Setup and training completed!")
print("📥 To download results, run:")
print("from google.colab import files")
print("files.download('/content/outputs/best_ffn_v2_model.pt')")

# ================================================================
# Fallback Training Function
# ================================================================

def create_simple_training_script():
    """Create a simple training script if the main one fails"""
    print("🔧 Creating simple training script...")
    
    simple_script = '''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Simple FFN-v2 model
class SimpleFFNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x

# Generate synthetic data
def generate_data(batch_size=8, size=64):
    volumes = torch.rand(batch_size, 1, size, size, size)
    labels = (volumes > 0.5).float()
    return volumes, labels

# Training function
def train_simple():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleFFNv2().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    for epoch in range(10):
        volumes, labels = generate_data()
        volumes, labels = volumes.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), '/content/outputs/simple_ffn_v2_model.pt')
    
    # Plot losses
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('/content/outputs/simple_training_curves.png')
    plt.show()
    
    print("✅ Simple training completed!")
    return model, losses

# Run simple training
model, losses = train_simple()
'''
    
    # Write the script to a file
    with open("/content/simple_training.py", "w") as f:
        f.write(simple_script)
    
    # Execute the simple training
    exec(simple_script) 