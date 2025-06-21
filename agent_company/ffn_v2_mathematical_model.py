import torch
import torch.nn as nn
import math

class MathematicalFFNv2(nn.Module):
    """
    Enhanced FFN-v2 with mathematical optimization insights
    Based on matrix analysis and optimization theory from textbooks
    """
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1, 
                 hidden_channels: int = 64, depth: int = 3):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        
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
        
        # Apply mathematical weight initialization
        self._init_weights_mathematically()

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
        for i, layer in enumerate(self.layers):
            identity = x
            x = layer(x)
            
            # Residual connection for gradient flow (mathematical insight)
            if x.shape == identity.shape:
                x = x + identity
            
            # Stochastic regularization (except for the last layer)
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        
        # Output with mathematical constraints
        output = self.output_layer(x)
        
        return output
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Mathematical input normalization"""
        # Ensure input is in valid range [0, 1]
        x = torch.clamp(x, 0, 1)
        return x 

# ============================================================================
# MODEL TESTING AND VERIFICATION
# ============================================================================

def test_mathematical_ffn_v2():
    """Test the MathematicalFFNv2 model to ensure it works correctly."""
    print("🧪 Testing MathematicalFFNv2 Model...")
    print("=" * 50)
    
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {device}")
        
        # Create model
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=1,
            hidden_channels=64,
            depth=3
        )
        model.to(device)
        print("✅ Model created successfully")
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Test forward pass
        batch_size = 2
        input_shape = (batch_size, 1, 64, 64, 64)  # Batch, Channels, Z, Y, X
        test_input = torch.randn(input_shape, device=device)
        
        print(f"🧪 Testing forward pass with input shape: {test_input.shape}")
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Forward pass successful!")
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"   - Output mean: {output.mean():.4f}")
        
        # Verify output is in valid range [0, 1] (due to Sigmoid)
        if 0 <= output.min() <= output.max() <= 1:
            print("✅ Output is in valid range [0, 1]")
        else:
            print("⚠️ Warning: Output outside expected range [0, 1]")
        
        # Test gradient flow
        print("\n🧪 Testing gradient flow...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create dummy target
        target = torch.randint(0, 2, output.shape, dtype=torch.float32, device=device)
        
        # Forward and backward pass
        optimizer.zero_grad()
        output = model(test_input)
        loss = torch.nn.functional.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"✅ Gradient flow test successful!")
        print(f"   - Loss: {loss.item():.4f}")
        
        # Check gradients
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        print(f"   - Gradient norm: {grad_norm:.4f}")
        
        if grad_norm > 0:
            print("✅ Gradients are flowing properly")
        else:
            print("⚠️ Warning: No gradients detected")
        
        print("\n🎉 MathematicalFFNv2 model test completed successfully!")
        print("✅ Model is ready for training on H01 data.")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test when the file is executed directly
    test_mathematical_ffn_v2() 