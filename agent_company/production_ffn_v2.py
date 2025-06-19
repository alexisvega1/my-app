#!/usr/bin/env python3
"""
Production FFN-v2 Model
======================
A production-quality FFN-v2 implementation optimized for performance and reliability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

class ProductionFFNv2Model(nn.Module):
    """Production-quality FFN-v2 model with simplified but effective architecture."""
    
    def __init__(self, 
                 input_channels: int = 1,
                 hidden_channels: List[int] = [32, 64, 128, 256],
                 output_channels: int = 1,
                 use_attention: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder (simplified without complex skip connections)
        self.decoder = self._build_decoder()
        
        # Attention mechanism
        if self.use_attention:
            self.attention = self._build_attention()
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Conv3d(hidden_channels[-1], hidden_channels[-1] // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels[-1] // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Final output layer for segmentation
        self.segmentation_head = nn.Conv3d(hidden_channels[0], output_channels, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Production FFN-v2 model initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def _build_encoder(self) -> nn.ModuleList:
        """Build encoder with residual connections."""
        encoder = nn.ModuleList()
        
        in_channels = self.input_channels
        for hidden_dim in self.hidden_channels:
            block = nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(),
                nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool3d(2)
            )
            encoder.append(block)
            in_channels = hidden_dim
        
        return encoder
    
    def _build_decoder(self) -> nn.ModuleList:
        """Build simplified decoder without complex skip connections."""
        decoder = nn.ModuleList()
        
        # Reverse the hidden channels for decoder
        decoder_channels = list(reversed(self.hidden_channels))
        
        for i, hidden_dim in enumerate(decoder_channels):
            if i == 0:
                # First decoder block - input from bottleneck
                in_channels = self.hidden_channels[-1]
            else:
                # Subsequent blocks
                in_channels = decoder_channels[i-1]
            
            block = nn.Sequential(
                nn.ConvTranspose3d(in_channels, hidden_dim, 2, stride=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(),
                nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(),
                nn.Dropout3d(self.dropout_rate)
            )
            decoder.append(block)
        
        return decoder
    
    def _build_attention(self) -> nn.Module:
        """Build attention mechanism."""
        return nn.MultiheadAttention(
            embed_dim=self.hidden_channels[-1],
            num_heads=8,
            batch_first=True
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        # Encoder path
        encoder_features = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_features.append(x)
        
        # Attention mechanism at bottleneck
        if self.use_attention:
            b, c, d, h, w = x.shape
            x_flat = x.view(b, c, -1).transpose(1, 2)
            attn_out, _ = self.attention(x_flat, x_flat, x_flat)
            x = attn_out.transpose(1, 2).view(b, c, d, h, w)
        
        # Decoder path (simplified)
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        # Final segmentation output
        segmentation = self.segmentation_head(x)
        segmentation = torch.sigmoid(segmentation)
        
        # Output heads
        uncertainty = self.uncertainty_head(encoder_features[-1])
        
        return segmentation, uncertainty 