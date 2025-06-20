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
        identity = self.layers[0](x)
        x = identity

        for i, layer in enumerate(self.layers[1:]):
            identity = x
            x = layer(x)
            
            # Residual connection for gradient flow (mathematical insight)
            if x.shape == identity.shape:
                x = x + identity
            
            # Stochastic regularization
            x = self.dropout(x)
        
        # Output with mathematical constraints
        output = self.output_layer(x)
        
        return output
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Mathematical input normalization"""
        # Ensure input is in valid range [0, 1]
        x = torch.clamp(x, 0, 1)
        return x 