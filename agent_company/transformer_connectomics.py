#!/usr/bin/env python3
"""
Transformer-Based Connectomics Architecture
==========================================
Advanced transformer architectures for automated tracing and proofreading
in connectomics, incorporating state-of-the-art approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TransformerConfig:
    """Configuration for transformer-based connectomics models."""
    # Architecture parameters
    input_channels: int = 1
    hidden_dim: int = 256
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    
    # Patch and embedding parameters
    patch_size: Tuple[int, int, int] = (4, 4, 4)
    embed_dim: int = 256
    
    # Window attention (for Swin Transformer)
    window_size: Tuple[int, int, int] = (8, 8, 8)
    shift_size: Tuple[int, int, int] = (4, 4, 4)
    
    # Positional encoding
    use_absolute_pos: bool = True
    use_relative_pos: bool = True
    pos_embed_type: str = "learned"  # "learned", "sinusoidal", "none"
    
    # Specialized features
    use_3d_attention: bool = True
    use_cross_attention: bool = True
    use_hierarchical_features: bool = True
    
    # Output heads
    num_classes: int = 1
    use_uncertainty: bool = True
    use_auxiliary_heads: bool = True

class PositionalEncoding3D(nn.Module):
    """3D positional encoding for transformer models."""
    
    def __init__(self, embed_dim: int, max_positions: Tuple[int, int, int] = (512, 512, 512)):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        
        # Create positional encodings for each dimension
        pos_encodings = []
        for dim, max_pos in enumerate(max_positions):
            pos_enc = torch.zeros(max_pos, embed_dim)
            position = torch.arange(0, max_pos).unsqueeze(1).float()
            
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                               -(math.log(10000.0) / embed_dim))
            
            pos_enc[:, 0::2] = torch.sin(position * div_term)
            pos_enc[:, 1::2] = torch.cos(position * div_term)
            pos_encodings.append(pos_enc)
        
        self.register_buffer('pos_enc_x', pos_encodings[0])
        self.register_buffer('pos_enc_y', pos_encodings[1])
        self.register_buffer('pos_enc_z', pos_encodings[2])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add 3D positional encoding to input tensor."""
        B, C, D, H, W = x.shape
        
        # Get positional encodings for current dimensions
        pos_x = self.pos_enc_x[:W].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pos_y = self.pos_enc_y[:H].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        pos_z = self.pos_enc_z[:D].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # Combine positional encodings
        pos_encoding = pos_x + pos_y + pos_z
        
        # Add to input
        return x + pos_encoding

class MultiHeadAttention3D(nn.Module):
    """3D Multi-head attention with specialized features for connectomics."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 use_3d_attention: bool = True, use_cross_attention: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_3d_attention = use_3d_attention
        self.use_cross_attention = use_cross_attention
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Cross-attention (for proofreading)
        if use_cross_attention:
            self.cross_q_proj = nn.Linear(embed_dim, embed_dim)
            self.cross_k_proj = nn.Linear(embed_dim, embed_dim)
            self.cross_v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 3D-specific attention
        if use_3d_attention:
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional cross-attention."""
        B, N, C = x.shape
        
        # Self-attention
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_weights = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, C)
        attn_output = self.out_proj(attn_output)
        
        # Cross-attention if context is provided
        if self.use_cross_attention and context is not None:
            cross_q = self.cross_q_proj(attn_output)
            cross_k = self.cross_k_proj(context)
            cross_v = self.cross_v_proj(context)
            
            cross_attn_weights = torch.softmax(
                cross_q @ cross_k.transpose(-2, -1) * self.scale, dim=-1
            )
            cross_attn_weights = self.dropout(cross_attn_weights)
            
            cross_output = (cross_attn_weights @ cross_v)
            attn_output = attn_output + cross_output
        
        return attn_output

class TransformerBlock(nn.Module):
    """Transformer block with layer normalization and residual connections."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 use_3d_attention: bool = True, use_cross_attention: bool = False):
        super().__init__()
        
        # Attention
        self.attention = MultiHeadAttention3D(
            embed_dim, num_heads, attention_dropout, use_3d_attention, use_cross_attention
        )
        self.attention_norm = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Attention with residual
        attn_out = self.attention(x, context, mask)
        x = self.attention_norm(x + attn_out)
        
        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.mlp_norm(x + mlp_out)
        
        return x

class VisionTransformer3D(nn.Module):
    """3D Vision Transformer for connectomics data."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            config.input_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        
        # Positional encoding
        if config.use_absolute_pos:
            if config.pos_embed_type == "learned":
                self.pos_embed = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            elif config.pos_embed_type == "sinusoidal":
                self.pos_embed = PositionalEncoding3D(config.embed_dim)
            else:
                self.pos_embed = None
        else:
            self.pos_embed = None
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim, config.num_heads, config.mlp_ratio,
                config.dropout_rate, config.attention_dropout,
                config.use_3d_attention, config.use_cross_attention
            )
            for _ in range(config.num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Output heads
        self.segmentation_head = nn.Linear(config.embed_dim, config.num_classes)
        
        if config.use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(config.embed_dim // 2, 1),
                nn.Sigmoid()
            )
        
        if config.use_auxiliary_heads:
            self.auxiliary_heads = nn.ModuleList([
                nn.Linear(config.embed_dim, config.num_classes)
                for _ in range(config.num_layers // 3)  # Auxiliary heads at 1/3 intervals
            ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with optional context for proofreading."""
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        
        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        # Add positional encoding
        if self.pos_embed is not None:
            if isinstance(self.pos_embed, nn.Parameter):
                x = x + self.pos_embed
            else:
                x = self.pos_embed(x)
        
        # Process context if provided (for proofreading)
        context_embed = None
        if context is not None:
            context_embed = self.patch_embed(context)
            context_embed = context_embed.flatten(2).transpose(1, 2)
            if self.pos_embed is not None:
                if isinstance(self.pos_embed, nn.Parameter):
                    context_embed = context_embed + self.pos_embed
                else:
                    context_embed = self.pos_embed(context_embed)
        
        # Transformer blocks
        auxiliary_outputs = []
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context_embed)
            
            # Collect auxiliary outputs
            if self.config.use_auxiliary_heads and (i + 1) % 3 == 0:
                aux_idx = (i + 1) // 3 - 1
                if aux_idx < len(self.auxiliary_heads):
                    aux_out = self.auxiliary_heads[aux_idx](self.norm(x))
                    auxiliary_outputs.append(aux_out)
        
        # Final normalization
        x = self.norm(x)
        
        # Output heads
        outputs = {}
        
        # Main segmentation output
        seg_out = self.segmentation_head(x)
        seg_out = seg_out.transpose(1, 2).view(B, self.config.num_classes, D//self.config.patch_size[0], 
                                               H//self.config.patch_size[1], W//self.config.patch_size[2])
        outputs['segmentation'] = torch.sigmoid(seg_out)
        
        # Uncertainty output
        if self.config.use_uncertainty:
            unc_out = self.uncertainty_head(x)
            unc_out = unc_out.transpose(1, 2).view(B, 1, D//self.config.patch_size[0], 
                                                   H//self.config.patch_size[1], W//self.config.patch_size[2])
            outputs['uncertainty'] = unc_out
        
        # Auxiliary outputs
        if auxiliary_outputs:
            outputs['auxiliary'] = auxiliary_outputs
        
        return outputs

class SwinTransformer3D(nn.Module):
    """3D Swin Transformer with hierarchical features for connectomics."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            config.input_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        
        # Swin transformer stages
        self.stages = nn.ModuleList()
        current_dim = config.embed_dim
        
        for stage_idx in range(4):  # 4 stages
            stage = nn.ModuleList()
            
            # Window attention blocks
            for block_idx in range(2):  # 2 blocks per stage
                block = SwinTransformerBlock3D(
                    current_dim, config.num_heads, config.window_size,
                    shift_size=config.shift_size if block_idx % 2 == 1 else (0, 0, 0),
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout_rate,
                    attention_dropout=config.attention_dropout
                )
                stage.append(block)
            
            self.stages.append(stage)
            
            # Downsample between stages
            if stage_idx < 3:
                downsample = PatchMerging3D(current_dim, current_dim * 2)
                stage.append(downsample)
                current_dim *= 2
        
        # Output heads
        self.segmentation_head = nn.Conv3d(current_dim, config.num_classes, 1)
        
        if config.use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv3d(current_dim, current_dim // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(current_dim // 2, 1, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with hierarchical feature extraction."""
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Swin transformer stages
        features = []
        for stage in self.stages:
            for block in stage[:-1]:  # All but last (downsample)
                x = block(x)
            features.append(x)
            
            if len(stage) > 1:  # Has downsample
                x = stage[-1](x)  # Downsample
        
        # Final features
        features.append(x)
        
        # Output heads
        outputs = {}
        
        # Segmentation output
        seg_out = self.segmentation_head(x)
        outputs['segmentation'] = torch.sigmoid(seg_out)
        
        # Uncertainty output
        if self.config.use_uncertainty:
            unc_out = self.uncertainty_head(x)
            outputs['uncertainty'] = unc_out
        
        # Multi-scale features for hierarchical processing
        outputs['features'] = features
        
        return outputs

class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer block with window attention."""
    
    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int, int],
                 shift_size: Tuple[int, int, int], mlp_ratio: float = 4.0,
                 dropout: float = 0.1, attention_dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        
        # Window attention
        self.attention = WindowAttention3D(
            dim, window_size, num_heads, attention_dropout
        )
        self.attention_norm = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(dim)
        
        # Shift window attention
        if any(s > 0 for s in shift_size):
            self.shift_attention = WindowAttention3D(
                dim, window_size, num_heads, attention_dropout
            )
            self.shift_attention_norm = nn.LayerNorm(dim)
        else:
            self.shift_attention = None
            self.shift_attention_norm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with window attention."""
        B, C, D, H, W = x.shape
        
        # Reshape to windows
        x_windows = self._window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        
        # Window attention
        attn_windows = self.attention(x_windows)
        attn_windows = self.attention_norm(x_windows + attn_windows)
        
        # Shift window attention
        if self.shift_attention is not None:
            x_shift = self._window_partition_shift(x, self.window_size, self.shift_size)
            x_shift = x_shift.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
            
            shift_attn_windows = self.shift_attention(x_shift)
            shift_attn_windows = self.shift_attention_norm(x_shift + shift_attn_windows)
            
            # Merge window and shift attention
            attn_windows = torch.cat([attn_windows, shift_attn_windows], dim=0)
        
        # Merge windows back
        x = self._window_reverse(attn_windows, self.window_size, D, H, W)
        
        # MLP
        x_flat = x.view(B, -1, C)
        mlp_out = self.mlp(x_flat)
        x_flat = self.mlp_norm(x_flat + mlp_out)
        x = x_flat.view(B, C, D, H, W)
        
        return x
    
    def _window_partition(self, x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
        """Partition input into windows."""
        B, C, D, H, W = x.shape
        x = x.view(B, C, D // window_size[0], window_size[0],
                   H // window_size[1], window_size[1],
                   W // window_size[2], window_size[2])
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.view(-1, C, window_size[0], window_size[1], window_size[2])
        return x
    
    def _window_partition_shift(self, x: torch.Tensor, window_size: Tuple[int, int, int],
                               shift_size: Tuple[int, int, int]) -> torch.Tensor:
        """Partition input into shifted windows."""
        B, C, D, H, W = x.shape
        
        # Apply shift
        x = torch.roll(x, shifts=shift_size, dims=(2, 3, 4))
        
        # Partition into windows
        return self._window_partition(x, window_size)
    
    def _window_reverse(self, windows: torch.Tensor, window_size: Tuple[int, int, int],
                       D: int, H: int, W: int) -> torch.Tensor:
        """Reverse window partition."""
        B = windows.shape[0] // (D // window_size[0] * H // window_size[1] * W // window_size[2])
        C = windows.shape[-1]
        
        x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                        window_size[0], window_size[1], window_size[2], C)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, D, H, W)
        return x

class WindowAttention3D(nn.Module):
    """3D Window-based multi-head attention."""
    
    def __init__(self, dim: int, window_size: Tuple[int, int, int], num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        
        # Get relative position index
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with relative position bias."""
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class PatchMerging3D(nn.Module):
    """3D Patch merging for hierarchical feature extraction."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge patches by concatenating neighboring patches."""
        B, C, D, H, W = x.shape
        
        # Reshape to merge patches
        x = x.view(B, C, D // 2, 2, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.view(B, D // 2 * H // 2 * W // 2, 4 * C)
        
        # Reduce dimension
        x = self.norm(x)
        x = self.reduction(x)
        
        # Reshape back to 3D
        x = x.view(B, -1, D // 2, H // 2, W // 2)
        
        return x

class TransformerConnectomicsModel(nn.Module):
    """Main transformer-based model for connectomics tracing and proofreading."""
    
    def __init__(self, config: TransformerConfig, model_type: str = "vit"):
        super().__init__()
        self.config = config
        self.model_type = model_type
        
        # Choose transformer architecture
        if model_type == "vit":
            self.transformer = VisionTransformer3D(config)
        elif model_type == "swin":
            self.transformer = SwinTransformer3D(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Upsampling layers for full resolution output
        self.upsample_layers = nn.ModuleList()
        current_size = config.patch_size[0]
        
        while current_size < 32:  # Upsample to reasonable resolution
            self.upsample_layers.append(
                nn.ConvTranspose3d(config.embed_dim, config.embed_dim, 2, stride=2)
            )
            current_size *= 2
        
        # Final output head
        self.final_head = nn.Conv3d(config.embed_dim, config.num_classes, 1)
        
        if config.use_uncertainty:
            self.final_uncertainty_head = nn.Conv3d(config.embed_dim, 1, 1)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with optional context for proofreading."""
        # Transformer forward pass
        if self.model_type == "vit":
            outputs = self.transformer(x, context)
        else:
            outputs = self.transformer(x)
        
        # Upsample features to full resolution
        if 'features' in outputs:
            features = outputs['features'][-1]  # Use last stage features
        else:
            # For ViT, use the main output
            features = outputs['segmentation']
        
        # Upsample
        for upsample_layer in self.upsample_layers:
            features = upsample_layer(features)
        
        # Final outputs
        final_outputs = {}
        final_outputs['segmentation'] = torch.sigmoid(self.final_head(features))
        
        if self.config.use_uncertainty:
            final_outputs['uncertainty'] = torch.sigmoid(self.final_uncertainty_head(features))
        
        # Add intermediate outputs
        final_outputs.update(outputs)
        
        return final_outputs

# Utility functions for transformer-based connectomics
def create_transformer_model(config_dict: Dict[str, Any], model_type: str = "vit") -> TransformerConnectomicsModel:
    """Create transformer model from configuration."""
    config = TransformerConfig(**config_dict)
    return TransformerConnectomicsModel(config, model_type)

def get_model_complexity(model: nn.Module, input_shape: Tuple[int, int, int, int, int]) -> Dict[str, Any]:
    """Calculate model complexity and memory usage."""
    import torch.profiler
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage
    input_tensor = torch.randn(input_shape)
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Get memory usage from profiler
    memory_usage = 0
    for event in prof.events():
        if hasattr(event, 'cuda_memory_usage'):
            memory_usage = max(memory_usage, event.cuda_memory_usage)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'estimated_memory_mb': memory_usage / 1024 / 1024 if memory_usage > 0 else 0,
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    }

# Example usage and testing
def test_transformer_models():
    """Test different transformer architectures."""
    # Configuration
    config_dict = {
        'input_channels': 1,
        'embed_dim': 256,
        'num_layers': 12,
        'num_heads': 8,
        'patch_size': (4, 4, 4),
        'use_uncertainty': True,
        'use_auxiliary_heads': True
    }
    
    # Test Vision Transformer
    print("Testing Vision Transformer...")
    vit_model = create_transformer_model(config_dict, "vit")
    
    # Test Swin Transformer
    print("Testing Swin Transformer...")
    swin_model = create_transformer_model(config_dict, "swin")
    
    # Test input
    input_tensor = torch.randn(2, 1, 64, 64, 64)
    
    # Forward pass
    with torch.no_grad():
        vit_outputs = vit_model(input_tensor)
        swin_outputs = swin_model(input_tensor)
    
    print("Vision Transformer outputs:")
    for key, value in vit_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {len(value)} items")
    
    print("\nSwin Transformer outputs:")
    for key, value in swin_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {len(value)} items")
    
    # Model complexity
    vit_complexity = get_model_complexity(vit_model, (1, 1, 64, 64, 64))
    swin_complexity = get_model_complexity(swin_model, (1, 1, 64, 64, 64))
    
    print(f"\nVision Transformer complexity: {vit_complexity}")
    print(f"Swin Transformer complexity: {swin_complexity}")
    
    return vit_model, swin_model, vit_outputs, swin_outputs

if __name__ == "__main__":
    test_transformer_models() 