#!/usr/bin/env python3
"""
Enhanced Production FFN-v2 Model
================================
Advanced implementation with improved architecture, performance optimizations,
and production-ready features for large-scale connectomics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, List, Dict, Any, Optional
import math
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the enhanced FFN-v2 model."""
    # Architecture parameters
    input_channels: int = 1
    hidden_channels: List[int] = None  # Will be set to [32, 64, 128, 256, 512] if None
    output_channels: int = 1
    use_attention: bool = True
    use_residual: bool = True
    use_batch_norm: bool = True
    use_dropout: bool = True
    dropout_rate: float = 0.1
    
    # Advanced features
    use_skip_connections: bool = True
    use_deep_supervision: bool = True
    use_uncertainty_estimation: bool = True
    use_auxiliary_losses: bool = True
    
    # Performance optimizations
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    use_amp: bool = True
    
    # Memory optimizations
    use_memory_efficient_attention: bool = True
    chunk_size: int = 1024
    
    def __post_init__(self):
        if self.hidden_channels is None:
            self.hidden_channels = [32, 64, 128, 256, 512]

class ResidualBlock3D(nn.Module):
    """3D Residual block with optional batch normalization and dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, use_batch_norm: bool = True, 
                 use_dropout: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        # Main path
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=not use_batch_norm)
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if use_dropout:
            layers.append(nn.Dropout3d(dropout_rate))
        
        layers.extend([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=not use_batch_norm)
        ])
        
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        
        self.main_path = nn.Sequential(*layers)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            skip_layers = [nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                                   stride=stride, bias=not use_batch_norm)]
            if use_batch_norm:
                skip_layers.append(nn.BatchNorm3d(out_channels))
            self.skip_path = nn.Sequential(*skip_layers)
        else:
            self.skip_path = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_path(x)
        out = self.main_path(x)
        return F.relu(out + residual, inplace=True)

class AttentionBlock3D(nn.Module):
    """3D Self-attention block for spatial feature modeling."""
    
    def __init__(self, channels: int, num_heads: int = 8, 
                 use_memory_efficient: bool = True):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.use_memory_efficient = use_memory_efficient
        
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        # Linear projections
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 4, channels)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Reshape for attention: (B, D*H*W, C)
        x_flat = x.view(B, C, -1).transpose(1, 2)
        
        # Self-attention
        residual = x_flat
        x_flat = self.norm1(x_flat)
        
        if self.use_memory_efficient and D * H * W > 1024:
            # Use chunked attention for memory efficiency
            attn_out = self._chunked_attention(x_flat)
        else:
            attn_out = self._standard_attention(x_flat)
        
        x_flat = residual + self.dropout(attn_out)
        
        # Feed-forward
        residual = x_flat
        x_flat = self.norm2(x_flat)
        ffn_out = self.ffn(x_flat)
        x_flat = residual + self.dropout(ffn_out)
        
        # Reshape back: (B, C, D, H, W)
        return x_flat.transpose(1, 2).view(B, C, D, H, W)
    
    def _standard_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard multi-head attention."""
        B, N, C = x.shape
        
        # Linear projections
        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, C)
        
        return self.output_proj(attn_out)
    
    def _chunked_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient chunked attention."""
        B, N, C = x.shape
        chunk_size = min(1024, N)
        
        output = torch.zeros_like(x)
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            chunk_x = x[:, i:end_i]
            
            # Compute attention for this chunk
            q = self.query(chunk_x).view(B, end_i - i, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
            attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, end_i - i, C)
            
            output[:, i:end_i] = self.output_proj(attn_out)
        
        return output

class EncoderBlock(nn.Module):
    """Enhanced encoder block with residual connections and attention."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 config: ModelConfig, use_attention: bool = False):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock3D(in_channels, out_channels, stride=1, 
                           use_batch_norm=config.use_batch_norm,
                           use_dropout=config.use_dropout,
                           dropout_rate=config.dropout_rate),
            ResidualBlock3D(out_channels, out_channels, stride=1,
                           use_batch_norm=config.use_batch_norm,
                           use_dropout=config.use_dropout,
                           dropout_rate=config.dropout_rate)
        )
        
        # Downsampling
        self.downsample = nn.MaxPool3d(2)
        
        # Attention (optional)
        if use_attention and config.use_attention:
            self.attention = AttentionBlock3D(out_channels, 
                                            use_memory_efficient=config.use_memory_efficient_attention)
        else:
            self.attention = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Residual blocks
        x = self.residual_blocks(x)
        
        # Save feature for skip connection
        skip_features = x
        
        # Attention (if enabled)
        if self.attention is not None:
            x = self.attention(x)
        
        # Downsampling
        x = self.downsample(x)
        
        return x, skip_features

class DecoderBlock(nn.Module):
    """Enhanced decoder block with skip connections and upsampling."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 skip_channels: int, config: ModelConfig):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, 
                                          kernel_size=2, stride=2)
        
        # Skip connection processing
        if config.use_skip_connections and skip_channels > 0:
            self.skip_conv = nn.Conv3d(skip_channels, out_channels, 
                                     kernel_size=1, bias=False)
            self.combine_conv = nn.Conv3d(out_channels * 2, out_channels, 
                                        kernel_size=3, padding=1)
        else:
            self.skip_conv = None
            self.combine_conv = None
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock3D(out_channels, out_channels, stride=1,
                           use_batch_norm=config.use_batch_norm,
                           use_dropout=config.use_dropout,
                           dropout_rate=config.dropout_rate),
            ResidualBlock3D(out_channels, out_channels, stride=1,
                           use_batch_norm=config.use_batch_norm,
                           use_dropout=config.use_dropout,
                           dropout_rate=config.dropout_rate)
        )
    
    def forward(self, x: torch.Tensor, skip_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Upsampling
        x = self.upsample(x)
        
        # Skip connection
        if self.skip_conv is not None and skip_features is not None:
            skip_processed = self.skip_conv(skip_features)
            x = torch.cat([x, skip_processed], dim=1)
            x = self.combine_conv(x)
        
        # Residual blocks
        x = self.residual_blocks(x)
        
        return x

class EnhancedProductionFFNv2Model(nn.Module):
    """Enhanced production-quality FFN-v2 model with advanced features."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Output heads
        self.segmentation_head = nn.Conv3d(config.hidden_channels[0], 
                                          config.output_channels, 1)
        
        if config.use_uncertainty_estimation:
            self.uncertainty_head = nn.Sequential(
                nn.Conv3d(config.hidden_channels[-1], config.hidden_channels[-1] // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(config.hidden_channels[-1] // 2, 1, 1),
                nn.Sigmoid()
            )
        
        # Deep supervision heads (if enabled)
        if config.use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(channels, config.output_channels, 1)
                for channels in config.hidden_channels[:-1]  # Skip bottleneck
            ])
        
        # Auxiliary heads (if enabled)
        if config.use_auxiliary_losses:
            self.auxiliary_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(config.hidden_channels[-1], 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(64, 1, 1),
                    nn.Sigmoid()
                )
            ])
        
        # Initialize weights
        self._initialize_weights()
        
        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing:
            self.gradient_checkpointing_enable()
        
        logger.info(f"Enhanced FFN-v2 model initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def _build_encoder(self) -> nn.ModuleList:
        """Build enhanced encoder with attention."""
        encoder = nn.ModuleList()
        
        in_channels = self.config.input_channels
        for i, hidden_dim in enumerate(self.config.hidden_channels):
            # Use attention in deeper layers
            use_attention = (i >= len(self.config.hidden_channels) // 2)
            
            block = EncoderBlock(in_channels, hidden_dim, self.config, use_attention)
            encoder.append(block)
            in_channels = hidden_dim
        
        return encoder
    
    def _build_decoder(self) -> nn.ModuleList:
        """Build enhanced decoder with skip connections."""
        decoder = nn.ModuleList()
        
        # Reverse channels for decoder
        decoder_channels = list(reversed(self.config.hidden_channels))
        
        for i, hidden_dim in enumerate(decoder_channels):
            if i == 0:
                # First decoder block - input from bottleneck
                in_channels = self.config.hidden_channels[-1]
                skip_channels = 0  # No skip connection for bottleneck
            else:
                # Subsequent blocks with skip connections
                in_channels = decoder_channels[i-1]
                skip_channels = self.config.hidden_channels[-(i+1)]
            
            block = DecoderBlock(in_channels, hidden_dim, skip_channels, self.config)
            decoder.append(block)
        
        return decoder
    
    def _initialize_weights(self):
        """Initialize model weights using improved initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multiple outputs for different tasks."""
        outputs = {}
        
        # Encoder path with skip connections
        skip_features = []
        for encoder_block in self.encoder:
            x, skip_feat = encoder_block(x)
            skip_features.append(skip_feat)
        
        # Bottleneck features
        bottleneck_features = x
        
        # Decoder path with skip connections
        deep_supervision_outputs = []
        for i, decoder_block in enumerate(self.decoder):
            if self.config.use_skip_connections and i < len(skip_features):
                skip_idx = len(skip_features) - 1 - i
                x = decoder_block(x, skip_features[skip_idx])
            else:
                x = decoder_block(x)
            
            # Deep supervision
            if self.config.use_deep_supervision and i < len(self.deep_supervision_heads):
                deep_out = self.deep_supervision_heads[i](x)
                deep_supervision_outputs.append(deep_out)
        
        # Main segmentation output
        segmentation = self.segmentation_head(x)
        segmentation = torch.sigmoid(segmentation)
        outputs['segmentation'] = segmentation
        
        # Uncertainty estimation
        if self.config.use_uncertainty_estimation:
            uncertainty = self.uncertainty_head(bottleneck_features)
            outputs['uncertainty'] = uncertainty
        
        # Deep supervision outputs
        if self.config.use_deep_supervision and deep_supervision_outputs:
            outputs['deep_supervision'] = deep_supervision_outputs
        
        # Auxiliary outputs
        if self.config.use_auxiliary_losses:
            auxiliary_outputs = []
            for aux_head in self.auxiliary_heads:
                aux_out = aux_head(bottleneck_features)
                auxiliary_outputs.append(aux_out)
            outputs['auxiliary'] = auxiliary_outputs
        
        return outputs
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract feature maps from all encoder layers for analysis."""
        feature_maps = []
        
        for encoder_block in self.encoder:
            x, skip_feat = encoder_block(x)
            feature_maps.append(skip_feat)
        
        return feature_maps
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute comprehensive loss including auxiliary losses."""
        losses = {}
        
        # Main segmentation loss
        if 'segmentation' in outputs and 'segmentation' in targets:
            seg_loss = F.binary_cross_entropy(outputs['segmentation'], targets['segmentation'])
            losses['segmentation'] = seg_loss
        
        # Uncertainty loss (if using uncertainty-aware training)
        if 'uncertainty' in outputs and 'segmentation' in targets:
            # Uncertainty should be high where predictions are uncertain
            pred_confidence = torch.abs(outputs['segmentation'] - 0.5) * 2  # 0 to 1
            uncertainty_target = 1.0 - pred_confidence
            unc_loss = F.mse_loss(outputs['uncertainty'], uncertainty_target)
            losses['uncertainty'] = unc_loss
        
        # Deep supervision losses
        if 'deep_supervision' in outputs and 'segmentation' in targets:
            deep_losses = []
            for deep_out in outputs['deep_supervision']:
                # Resize deep supervision output to match target
                resized_deep = F.interpolate(deep_out, size=targets['segmentation'].shape[2:],
                                           mode='trilinear', align_corners=False)
                deep_loss = F.binary_cross_entropy(resized_deep, targets['segmentation'])
                deep_losses.append(deep_loss)
            
            losses['deep_supervision'] = sum(deep_losses) / len(deep_losses)
        
        # Auxiliary losses
        if 'auxiliary' in outputs:
            aux_losses = []
            for aux_out in outputs['auxiliary']:
                # Resize auxiliary output to match target
                resized_aux = F.interpolate(aux_out, size=targets['segmentation'].shape[2:],
                                          mode='trilinear', align_corners=False)
                aux_loss = F.binary_cross_entropy(resized_aux, targets['segmentation'])
                aux_losses.append(aux_loss)
            
            losses['auxiliary'] = sum(aux_losses) / len(aux_losses)
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses

# Utility functions for model management
def create_model_from_config(config_dict: Dict[str, Any]) -> EnhancedProductionFFNv2Model:
    """Create model from configuration dictionary."""
    config = ModelConfig(**config_dict)
    return EnhancedProductionFFNv2Model(config)

def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and non-trainable parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    
    return {
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'total': total_params
    }

# Example usage and testing
def test_enhanced_model():
    """Test the enhanced FFN-v2 model."""
    # Create configuration
    config = ModelConfig(
        input_channels=1,
        hidden_channels=[32, 64, 128, 256],
        output_channels=1,
        use_attention=True,
        use_residual=True,
        use_skip_connections=True,
        use_deep_supervision=True,
        use_uncertainty_estimation=True
    )
    
    # Create model
    model = EnhancedProductionFFNv2Model(config)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 64, 64, 64)
    
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} outputs")
        else:
            print(f"  {key}: {value.shape}")
    
    # Model statistics
    param_counts = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print(f"\nModel statistics:")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Model size: {model_size:.2f} MB")
    
    return model, outputs

if __name__ == "__main__":
    test_enhanced_model() 