import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    """
    A 3D U-Net architecture, which is state-of-the-art for volumetric segmentation.
    This model uses an encoder-decoder structure with skip connections to capture
    both context and localization information.
    """

    def __init__(self, in_channels, out_channels, n_levels=4, initial_features=32):
        super(UNet3D, self).__init__()

        features = initial_features
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # Encoder path (downsampling)
        for i in range(n_levels):
            self.encoder_layers.append(
                ConvBlock(in_channels if i == 0 else features, features, features)
            )
            features *= 2

        # Bottleneck
        self.bottleneck = ConvBlock(features // 2, features, features)

        # Decoder path (upsampling)
        for i in range(n_levels):
            self.decoder_layers.append(
                UpConvBlock(features, features // 2, features // 2)
            )
            features //= 2

        # Final convolution
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for level, layer in enumerate(self.encoder_layers):
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool3d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections.reverse()
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, skip_connections[i])

        # Final output
        x = self.final_conv(x)
        return torch.sigmoid(x)


class ConvBlock(nn.Module):
    """A double convolution block."""
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConvBlock(nn.Module):
    """An upsampling block followed by a convolution block."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels, out_channels)

    def forward(self, x, skip_x):
        x = self.up(x)
        
        # Handle padding differences if necessary
        if x.shape != skip_x.shape:
            # This can happen if input size is not a power of 2
            # We crop the skip connection to match the upsampled feature map
            diff_z = skip_x.size()[2] - x.size()[2]
            diff_y = skip_x.size()[3] - x.size()[3]
            diff_x = skip_x.size()[4] - x.size()[4]
            skip_x = skip_x[:, :, diff_z // 2 : diff_z // 2 + x.size()[2],
                            diff_y // 2 : diff_y // 2 + x.size()[3],
                            diff_x // 2 : diff_x // 2 + x.size()[4]]

        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)

# For backward compatibility, we can wrap the U-Net in the original class name
class MathematicalFFNv2(UNet3D):
    def __init__(self, input_channels: int = 1, output_channels: int = 3, 
                 hidden_channels: int = 32, depth: int = 4):
        # Map old parameters to new U-Net parameters
        super().__init__(
            in_channels=input_channels, 
            out_channels=output_channels,
            n_levels=depth,
            initial_features=hidden_channels
        )

# ============================================================================
# MODEL TESTING AND VERIFICATION
# ============================================================================

def test_unet3d_model():
    """Test the UNet3D model to ensure it works correctly."""
    print("🧪 Testing UNet3D Model...")
    print("=" * 50)
    
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {device}")
        
        # Create model
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=3,
            hidden_channels=32,
            depth=4
        )
        model.to(device)
        print("✅ Model created successfully")
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Test forward pass
        batch_size = 1
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
        
        # Verify output shape
        assert output.shape == (1, 3, 64, 64, 64)
        print("✅ Output shape is correct.")
        
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
        
        print("\n🎉 UNet3D model test completed successfully!")
        print("✅ Model is ready for training on H01 data.")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test when the file is executed directly
    test_unet3d_model() 