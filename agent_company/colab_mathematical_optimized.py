#!/usr/bin/env python3
"""
Colab-Ready Mathematical FFN-v2 Training Script
Enhanced with optimization insights from mathematical textbooks
"""

# Install required packages
!pip install torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Tuple, List, Dict
import math

# Disable CUDA initially to avoid Triton conflicts
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import PyTorch on CPU first
import torch
torch.backends.cudnn.enabled = False

# Now re-enable GPU if available
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("✅ GPU available and enabled")
else:
    print("⚠️ Using CPU (GPU not available)")

class MathematicalFFNv2(nn.Module):
    """
    Enhanced FFN-v2 with mathematical optimization insights
    Based on matrix analysis and optimization theory
    """
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1, 
                 hidden_channels: int = 64, depth: int = 3):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        
        # Mathematical weight initialization based on matrix analysis
        self._init_weights_mathematically()
        
        # Core FFN-v2 architecture with mathematical enhancements
        self.layers = nn.ModuleList()
        
        # Input layer with mathematical normalization
        self.layers.append(nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, 1),
            nn.BatchNorm3d(hidden_channels),  # Internal covariate shift reduction
            nn.ReLU(inplace=True)
        ))
        
        # Hidden layers with residual connections and mathematical optimizations
        for i in range(depth - 1):
            layer = nn.Sequential(
                nn.Conv3d(hidden_channels, hidden_channels, 1),
                nn.BatchNorm3d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_channels, hidden_channels, 1),
                nn.BatchNorm3d(hidden_channels)
            )
            self.layers.append(layer)
        
        # Output layer with mathematical activation
        self.output_layer = nn.Sequential(
            nn.Conv3d(hidden_channels, output_channels, 1),
            nn.Sigmoid()  # Bounded output for segmentation
        )
        
        # Mathematical regularization components
        self.dropout = nn.Dropout3d(0.1)  # Stochastic regularization
        
    def _init_weights_mathematically(self):
        """Mathematical weight initialization based on matrix analysis"""
        def init_weights(m):
            if isinstance(m, nn.Conv3d):
                # Xavier/Glorot initialization for optimal gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mathematical optimizations"""
        
        # Input normalization for numerical stability
        x = self._normalize_input(x)
        
        # Apply layers with residual connections
        for i, layer in enumerate(self.layers[:-1]):
            identity = x
            x = layer(x)
            
            # Residual connection for gradient flow (mathematical insight)
            if x.shape == identity.shape:
                x = x + identity
            
            # Stochastic regularization
            x = self.dropout(x)
        
        # Final hidden layer
        x = self.layers[-1](x)
        
        # Output with mathematical constraints
        output = self.output_layer(x)
        
        return output
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Mathematical input normalization"""
        # Ensure input is in valid range [0, 1]
        x = torch.clamp(x, 0, 1)
        return x

class MathematicalOptimizer:
    """
    Advanced optimizer incorporating mathematical insights
    Based on optimization theory and matrix analysis
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        self.model = model
        
        # AdamW optimizer with mathematical weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,  # L2 regularization
            betas=(0.9, 0.999),  # Momentum parameters
            eps=1e-8  # Numerical stability
        )
        
        # Cosine annealing scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Initial restart period
            T_mult=2,  # Period multiplier
            eta_min=1e-6  # Minimum learning rate
        )
        
        # Gradient clipping for stability
        self.max_grad_norm = 1.0
        
    def step(self, loss: torch.Tensor):
        """Optimization step with mathematical enhancements"""
        # Backward pass
        loss.backward()
        
        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimization step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Learning rate scheduling
        self.scheduler.step()
        
        return self.scheduler.get_last_lr()[0]

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

class MathematicalDataGenerator:
    """
    Advanced data generator with mathematical insights
    Based on probability theory and stochastic processes
    """
    
    def __init__(self, volume_size: Tuple[int, int, int] = (64, 64, 64)):
        self.volume_size = volume_size
        
    def generate_neuron_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate realistic neuron data with mathematical properties"""
        
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            # Generate input volume with mathematical noise
            input_vol = self._generate_input_volume()
            
            # Generate target segmentation with mathematical constraints
            target_vol = self._generate_target_volume()
            
            inputs.append(input_vol)
            targets.append(target_vol)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def _generate_input_volume(self) -> torch.Tensor:
        """Generate input volume with mathematical properties"""
        # Base volume with Gaussian noise
        volume = torch.randn(self.volume_size) * 0.1
        
        # Add mathematical structures (simulated neurons)
        num_neurons = np.random.randint(3, 8)
        
        for _ in range(num_neurons):
            # Random neuron position
            center = (
                np.random.randint(10, self.volume_size[0] - 10),
                np.random.randint(10, self.volume_size[1] - 10),
                np.random.randint(10, self.volume_size[2] - 10)
            )
            
            # Generate neuron structure with mathematical properties
            neuron_intensity = np.random.uniform(0.3, 0.8)
            neuron_radius = np.random.uniform(2, 5)
            
            # Create spherical neuron with Gaussian falloff
            for i in range(self.volume_size[0]):
                for j in range(self.volume_size[1]):
                    for k in range(self.volume_size[2]):
                        dist = math.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                        if dist < neuron_radius:
                            intensity = neuron_intensity * math.exp(-dist**2 / (2 * neuron_radius**2))
                            volume[i, j, k] += intensity
        
        # Normalize to [0, 1]
        volume = torch.clamp(volume, 0, 1)
        return volume.unsqueeze(0)  # Add channel dimension
    
    def _generate_target_volume(self) -> torch.Tensor:
        """Generate target segmentation with mathematical constraints"""
        # Create binary segmentation
        target = torch.zeros(self.volume_size)
        
        # Add mathematical structures
        num_segments = np.random.randint(2, 6)
        
        for _ in range(num_segments):
            # Random segment position
            center = (
                np.random.randint(10, self.volume_size[0] - 10),
                np.random.randint(10, self.volume_size[1] - 10),
                np.random.randint(10, self.volume_size[2] - 10)
            )
            
            # Generate segment with mathematical properties
            segment_radius = np.random.uniform(1, 3)
            
            # Create spherical segment
            for i in range(self.volume_size[0]):
                for j in range(self.volume_size[1]):
                    for k in range(self.volume_size[2]):
                        dist = math.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                        if dist < segment_radius:
                            target[i, j, k] = 1.0
        
        return target.unsqueeze(0)  # Add channel dimension

class MathematicalTrainer:
    """
    Advanced trainer with mathematical monitoring and optimization
    Based on convergence analysis and optimization theory
    """
    
    def __init__(self, model: nn.Module, optimizer: MathematicalOptimizer, 
                 loss_fn: MathematicalLossFunction, device: str = 'cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Mathematical monitoring
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Early stopping with mathematical criteria
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch with mathematical optimizations"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.loss_fn(outputs, targets)
            
            # Optimization step
            lr = self.optimizer.step(loss)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Mathematical progress monitoring
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.6f}, LR = {lr:.2e}")
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validation with mathematical analysis"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 100) -> Dict[str, List[float]]:
        """Complete training with mathematical monitoring"""
        
        print("🚀 Starting Mathematical FFN-v2 Training")
        print("=" * 50)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Mathematical monitoring
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(self.optimizer.scheduler.get_last_lr()[0])
            
            epoch_time = time.time() - start_time
            
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
            print(f"   Train Loss: {train_loss:.6f}")
            print(f"   Val Loss: {val_loss:.6f}")
            print(f"   Learning Rate: {self.learning_rates[-1]:.2e}")
            print(f"   Time: {epoch_time:.2f}s")
            
            # Early stopping with mathematical criteria
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 'best_mathematical_ffn_v2.pth')
                print("   ✅ New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"   🛑 Early stopping after {epoch + 1} epochs")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
    
    def plot_training_curves(self, save_path: str = 'mathematical_training_curves.png'):
        """Plot mathematical training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Mathematical Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate curve
        ax2.plot(self.learning_rates, label='Learning Rate', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Mathematical Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Main training function
def main():
    """Main training function with mathematical optimizations"""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Mathematical data generation
    print("\n📊 Generating mathematical training data...")
    data_generator = MathematicalDataGenerator(volume_size=(32, 32, 32))
    
    # Generate training data
    train_inputs, train_targets = data_generator.generate_neuron_data(100)
    val_inputs, val_targets = data_generator.generate_neuron_data(20)
    
    # Create data loaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Mathematical model initialization
    print("\n🧠 Initializing Mathematical FFN-v2...")
    model = MathematicalFFNv2(
        input_channels=1,
        output_channels=1,
        hidden_channels=32,
        depth=3
    )
    
    # Mathematical optimizer
    optimizer = MathematicalOptimizer(model, learning_rate=1e-3)
    
    # Mathematical loss function
    loss_fn = MathematicalLossFunction(alpha=0.6, beta=0.4)
    
    # Mathematical trainer
    trainer = MathematicalTrainer(model, optimizer, loss_fn, device)
    
    # Training with mathematical monitoring
    print("\n🚀 Starting Mathematical Training...")
    training_history = trainer.train(train_loader, val_loader, num_epochs=50)
    
    # Plot mathematical results
    print("\n📈 Plotting Mathematical Training Curves...")
    trainer.plot_training_curves()
    
    # Mathematical model evaluation
    print("\n🔍 Mathematical Model Evaluation...")
    model.eval()
    
    with torch.no_grad():
        test_input = train_inputs[0:1].to(device)
        test_output = model(test_input)
        
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {test_output.shape}")
        print(f"   Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    print("\n✅ Mathematical FFN-v2 Training Complete!")
    print("📁 Best model saved as: best_mathematical_ffn_v2.pth")
    print("📊 Training curves saved as: mathematical_training_curves.png")
    
    # Download instructions
    print("\n📥 To download the trained model:")
    print("   from google.colab import files")
    print("   files.download('best_mathematical_ffn_v2.pth')")
    print("   files.download('mathematical_training_curves.png')")

# Run the training
if __name__ == "__main__":
    main() 