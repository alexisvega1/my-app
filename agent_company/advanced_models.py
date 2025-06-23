"""
Advanced Model Architectures for Connectomics Research
=====================================================

Implements cutting-edge model architectures including attention mechanisms,
transformer-based models, and state-of-the-art connectomics models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block for 3D connectomics data.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, DHW, C)
        
        qkv = self.qkv(x_flat).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x_attn = self.proj(x_attn)
        x_attn = x_attn.transpose(1, 2).reshape(B, C, D, H, W)
        
        return x_attn

class TransformerFFN(nn.Module):
    """
    Transformer-based Flood-Filling Network with attention mechanisms.
    """
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1, 
                 hidden_channels: int = 64, depth: int = 4, num_heads: int = 8):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        
        # Initial convolution
        self.input_conv = nn.Conv3d(input_channels, hidden_channels, 3, padding=1)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.Sequential(
                AttentionBlock(hidden_channels, num_heads),
                nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
                nn.BatchNorm3d(hidden_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(depth)
        ])
        
        # Output convolution
        self.output_conv = nn.Conv3d(hidden_channels, output_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        
        for block in self.transformer_blocks:
            x = x + block(x)  # Residual connection
            
        x = self.output_conv(x)
        return torch.sigmoid(x)

class SwinTransformer3D(nn.Module):
    """
    3D Swin Transformer for connectomics segmentation.
    """
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1,
                 embed_dim: int = 96, depths: List[int] = [2, 2, 6, 2],
                 num_heads: List[int] = [3, 6, 12, 24], window_size: int = 7):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(input_channels, embed_dim, 4, stride=4)
        
        # Swin transformer stages
        self.stages = nn.ModuleList()
        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            stage = nn.ModuleList([
                SwinTransformerBlock3D(
                    embed_dim * (2 ** i),
                    num_head,
                    window_size,
                    shift_size=0 if j % 2 == 0 else window_size // 2
                ) for j in range(depth)
            ])
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                self.stages.append(PatchMerging3D(embed_dim * (2 ** i)))
        
        # Output head
        self.output_head = nn.Sequential(
            nn.ConvTranspose3d(embed_dim * (2 ** (len(depths) - 1)), embed_dim, 4, stride=4),
            nn.Conv3d(embed_dim, output_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        
        for stage in self.stages:
            if isinstance(stage, nn.ModuleList):
                for block in stage:
                    x = block(x)
            else:
                x = stage(x)
        
        x = self.output_head(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    """
    3D Swin Transformer block with windowed attention.
    """
    
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # Reshape for attention
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)
        
        # Self-attention
        shortcut = x_flat
        x_flat = self.norm1(x_flat)
        x_flat = self.attn(x_flat, D, H, W)
        x_flat = shortcut + x_flat
        
        # MLP
        shortcut = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x_flat = shortcut + x_flat
        
        # Reshape back
        x = x_flat.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return x

class WindowAttention3D(nn.Module):
    """
    3D windowed multi-head self-attention.
    """
    
    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        
        # Create windows
        x_windows = self.create_windows(x, D, H, W)
        
        # Apply attention to each window
        attn_windows = []
        for window in x_windows:
            qkv = self.qkv(window).reshape(-1, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            
            window_attn = (attn @ v).transpose(1, 2).reshape(-1, C)
            window_attn = self.proj(window_attn)
            attn_windows.append(window_attn)
        
        # Merge windows back
        x = self.merge_windows(attn_windows, D, H, W)
        return x
    
    def create_windows(self, x: torch.Tensor, D: int, H: int, W: int) -> List[torch.Tensor]:
        """Create windows from input tensor."""
        # Simplified window creation - in practice, you'd implement proper windowing
        window_size = self.window_size
        windows = []
        
        for d in range(0, D, window_size):
            for h in range(0, H, window_size):
                for w in range(0, W, window_size):
                    d_end = min(d + window_size, D)
                    h_end = min(h + window_size, H)
                    w_end = min(w + window_size, W)
                    
                    window = x[:, d:h_end, h:h_end, w:w_end, :].reshape(-1, self.dim)
                    windows.append(window)
        
        return windows
    
    def merge_windows(self, windows: List[torch.Tensor], D: int, H: int, W: int) -> torch.Tensor:
        """Merge windows back to original tensor shape."""
        # Simplified merging - in practice, you'd implement proper merging
        return torch.cat(windows, dim=0)

class PatchMerging3D(nn.Module):
    """
    3D patch merging layer for Swin Transformer.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # Reshape for merging
        x = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)
        
        # Merge patches (simplified)
        x = self.norm(x)
        x = self.reduction(x)
        
        # Reshape back
        x = x.reshape(B, D // 2, H // 2, W // 2, 2 * C).permute(0, 4, 1, 2, 3)
        return x

class MLP(nn.Module):
    """
    Multi-layer perceptron for transformer blocks.
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class HybridConnectomicsModel(nn.Module):
    """
    Hybrid model combining CNN and transformer architectures.
    """
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1,
                 cnn_channels: int = 64, transformer_dim: int = 256):
        super().__init__()
        
        # CNN backbone
        self.cnn_backbone = nn.Sequential(
            nn.Conv3d(input_channels, cnn_channels, 3, padding=1),
            nn.BatchNorm3d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(cnn_channels, cnn_channels, 3, padding=1),
            nn.BatchNorm3d(cnn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Transformer head
        self.transformer = TransformerFFN(
            input_channels=cnn_channels,
            output_channels=transformer_dim,
            hidden_channels=transformer_dim,
            depth=2
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Conv3d(transformer_dim, cnn_channels, 3, padding=1),
            nn.BatchNorm3d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(cnn_channels, output_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)
        
        # Transformer processing
        transformer_features = self.transformer(cnn_features)
        
        # Output generation
        output = self.output_head(transformer_features)
        return output

def create_advanced_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create advanced model architectures.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific parameters
        
    Returns:
        PyTorch model
    """
    models = {
        'transformer_ffn': TransformerFFN,
        'swin_transformer': SwinTransformer3D,
        'hybrid': HybridConnectomicsModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs) 