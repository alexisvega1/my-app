#!/bin/bash
# ================================================================
# FFN-v2 Training on Google Colab GPU - Bash Script
# ================================================================
# Usage:
#   1. Open https://colab.research.google.com
#   2. Switch Runtime → Change runtime type → GPU (T4 / A100)
#   3. Create a new cell and paste this entire script
#   4. Run the cell (Ctrl+Enter)
# ================================================================

set -e  # Exit on any error

echo "🚀 Starting FFN-v2 training setup on Colab..."

# ================================================================
# Step 1: Install Dependencies
# ================================================================

echo "🛠️  Installing dependencies..."

# Install CUDA-enabled PyTorch
pip -q install torch==2.2.*+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other required packages
pip -q install cloud-volume tensorstore dask[array] prometheus-client peft accelerate matplotlib scikit-image tqdm wandb

echo "✅ Dependencies installed successfully"

# ================================================================
# Step 2: Clone Repository
# ================================================================

echo "📥 Cloning repository..."

# Update this URL to your actual repository
REPO_URL="https://github.com/alexisvega1/my-app.git"

if [ ! -d "/content/agent_company" ]; then
    git clone -q "$REPO_URL" /content/agent_company
fi

cd /content/agent_company

echo "✅ Repository cloned successfully"

# ================================================================
# Step 3: Check GPU and Environment
# ================================================================

echo "🔍 Checking environment..."

# Check if CUDA is available
python -c "
import torch
print(f'📱 PyTorch version: {torch.__version__}')
print(f'🎮 CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎮 GPU: {torch.cuda.get_device_name(0)}')
    print(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'🔢 GPU Count: {torch.cuda.device_count()}')
"

# ================================================================
# Step 4: Create Output Directory
# ================================================================

echo "📁 Creating output directory..."
mkdir -p /content/outputs

# ================================================================
# Step 5: Launch Training
# ================================================================

echo "🚀 Launching FFN-v2 training..."

# Run the training script
python train_ffn_v2_colab.py

echo "✅ Training completed!"

# ================================================================
# Step 6: Show Results
# ================================================================

echo "📊 Training Results:"
echo "===================="

# List output files
echo "📁 Output files:"
ls -la /content/outputs/

# Show training history if available
if [ -f "/content/outputs/training_history.json" ]; then
    echo ""
    echo "📈 Training Summary:"
    python -c "
import json
with open('/content/outputs/training_history.json', 'r') as f:
    history = json.load(f)
print(f'⏱️  Total time: {history[\"total_time\"]/60:.1f} minutes')
print(f'📊 Final train loss: {history[\"train_losses\"][-1]:.4f}')
print(f'📊 Final val loss: {history[\"val_losses\"][-1]:.4f}')
print(f'🎯 Best val loss: {min(history[\"val_losses\"]):.4f}')
"
fi

echo ""
echo "🎉 Setup and training completed successfully!"
echo "📋 Next steps:"
echo "   1. Download the model checkpoint from /content/outputs/"
echo "   2. Use the model for inference on your H01 data"
echo "   3. Integrate with your human feedback RL system"

# ================================================================
# Step 7: Download Instructions
# ================================================================

echo ""
echo "📥 To download results, run this in a new cell:"
echo "from google.colab import files"
echo "files.download('/content/outputs/best_ffn_v2_model.pt')"
echo "files.download('/content/outputs/training_history.json')"
echo "files.download('/content/outputs/training_curves.png')" 