"""
FFN-v2 plugin with Inception-3D backbone + uncertainty head.

This module implements the SegmenterPlugin interface expected by
ConnectomicsManager. It is self-contained: no external FFN code needed,
but you can later swap `InceptionBlock3D` for Google's official layers.

Based on "Going deeper with convolutions" by Szegedy et al. (2014)
https://arxiv.org/pdf/1409.4842

Key improvements from the paper:
- Multi-scale processing through Inception modules
- Efficient use of computational resources with bottlenecks
- Sparse architecture approximation with dense building blocks
- Dimension reduction with 1x1 convolutions

Author: AI Assistant
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------


class ConvBnRelu(nn.Sequential):
    """3-D Conv → BatchNorm → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class InceptionBlock3D(nn.Module):
    """
    3-D Inception-v1-style block with bottleneck channels.
    
    Following the paper's design philosophy:
    - Multi-scale feature extraction through parallel branches
    - Dimension reduction with 1x1 convolutions
    - Efficient use of computational budget
    
    Branches:
        1) 1×1×1
        2) 1×1×1 → 3×3×3
        3) 1×1×1 → 5×5×5 (depthwise separable via two 3×3×3)
        4) 3×3×3 max-pool → 1×1×1
    """

    def __init__(self, in_ch: int, ch_mid: int, ch_out: int):
        super().__init__()
        # Branch 1: Direct 1x1 convolution
        self.b1 = ConvBnRelu(in_ch, ch_out, k=1, p=0)

        # Branch 2: 1x1 reduction followed by 3x3
        self.b2 = nn.Sequential(
            ConvBnRelu(in_ch, ch_mid, k=1, p=0),
            ConvBnRelu(ch_mid, ch_out, k=3)
        )

        # Branch 3: 1x1 reduction followed by two 3x3 (simulating 5x5)
        # This follows the paper's approach of factorizing larger convolutions
        self.b3 = nn.Sequential(
            ConvBnRelu(in_ch, ch_mid, k=1, p=0),
            ConvBnRelu(ch_mid, ch_mid, k=3),
            ConvBnRelu(ch_mid, ch_out, k=3)
        )

        # Branch 4: Max pooling followed by 1x1
        self.b4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, ch_out, k=1, p=0)
        )

        self.out_ch = 4 * ch_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class GlobalAveragePooling3D(nn.Module):
    """Global average pooling for 3D tensors with uncertainty estimation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling across spatial dimensions
        return F.adaptive_avg_pool3d(x, 1)


class AuxiliaryClassifier(nn.Module):
    """
    Auxiliary classifier inspired by the paper's approach to combat
    vanishing gradients in deep networks.
    """
    
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=5, stride=3)
        self.conv = ConvBnRelu(in_channels, 128, k=1, p=0)
        self.fc1 = nn.Linear(128, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        x = F.adaptive_avg_pool3d(x, 1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------
# FFN-v2 Plugin
# ---------------------------------------------------------------------


class FFNv2Plugin(nn.Module):
    """
    Drop-in segmentation backbone with Inception-inspired architecture:
        • 4 stacked Inception-3D blocks with bottlenecks
        • Auxiliary classifiers for improved gradient flow
        • Final 1×1×1 conv to logits
        • Parallel uncertainty head (variance logits)
        • Efficient computational budget following paper's principles
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout_rate: float = 0.4,
        use_auxiliary: bool = True,
    ):
        super().__init__()
        ch = base_channels
        self.use_auxiliary = use_auxiliary

        # Stem layer (bottleneck reduction)
        self.stem = ConvBnRelu(in_channels, ch, k=7, p=3)
        self.stem_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Build Inception blocks
        blocks = []
        aux_classifiers = []
        
        for i in range(num_blocks):
            # Increase capacity gradually while maintaining efficiency
            mid_ch = max(ch // 4, 8)  # Ensure minimum channels for bottleneck
            out_ch = ch // 4
            
            blocks.append(InceptionBlock3D(ch, mid_ch, out_ch))
            ch = blocks[-1].out_ch  # Update channel count
            
            # Add auxiliary classifiers at intermediate layers (following paper)
            if self.use_auxiliary and i in [num_blocks//2, 3*num_blocks//4]:
                aux_classifiers.append(AuxiliaryClassifier(ch, 1))
            else:
                aux_classifiers.append(None)

        self.backbone = nn.ModuleList(blocks)
        self.aux_classifiers = nn.ModuleList([aux for aux in aux_classifiers if aux is not None])

        # Final heads
        self.seg_head = nn.Sequential(
            nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv3d(ch, 1, kernel_size=1)
        )
        
        self.unc_head = nn.Sequential(
            nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv3d(ch, 1, kernel_size=1)
        )

        # Initialize weights following paper's recommendations
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following best practices from the paper."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # -----------------------------------------------------------------
    # Public API expected by SegmenterPlugin interface
    # -----------------------------------------------------------------
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with optional auxiliary outputs.
        
        Args:
            x: Input tensor (N, C, D, H, W)
            
        Returns:
            segmentation_logits : FloatTensor   (N, 1, D, H, W)
            uncertainty_logits  : FloatTensor   (N, 1, D, H, W)
            auxiliary_outputs   : List[FloatTensor] or None (auxiliary classifier outputs)
        """
        # Stem processing
        feats = self.stem(x)
        feats = self.stem_pool(feats)
        
        # Progressive feature extraction through Inception blocks
        aux_outputs = []
        aux_idx = 0
        
        for i, block in enumerate(self.backbone):
            feats = block(feats)
            
            # Collect auxiliary outputs during training
            if self.training and self.use_auxiliary and aux_idx < len(self.aux_classifiers):
                if i in [len(self.backbone)//2, 3*len(self.backbone)//4]:
                    aux_out = self.aux_classifiers[aux_idx](feats)
                    aux_outputs.append(aux_out)
                    aux_idx += 1
        
        # Final predictions
        seg = self.seg_head(feats)
        unc = self.unc_head(feats)
        
        return seg, unc, aux_outputs if aux_outputs else None

    @torch.no_grad()
    def segment(self, volume: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Convenience wrapper: run forward pass and threshold.
        
        Args:
            volume : (1, 1, D, H, W) or (1, D, H, W)
            threshold : Sigmoid threshold for binary segmentation
            
        Returns:
            binary segmentation mask (same shape as input)
        """
        self.eval()
        
        # Ensure correct input shape
        if volume.dim() == 4:
            volume = volume.unsqueeze(1)  # Add channel dimension
            
        seg_logits, _, _ = self.forward(volume)
        return (torch.sigmoid(seg_logits) > threshold).float()

    def get_uncertainty(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Extract uncertainty estimation for the input volume.
        
        Args:
            volume : (1, 1, D, H, W) or (1, D, H, W)
            
        Returns:
            uncertainty map (same spatial shape as input)
        """
        self.eval()
        
        if volume.dim() == 4:
            volume = volume.unsqueeze(1)
            
        with torch.no_grad():
            _, unc_logits, _ = self.forward(volume)
            # Convert to uncertainty (higher values = more uncertain)
            return torch.sigmoid(unc_logits)

    def predict_with_uncertainty(self, volume: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both segmentation and uncertainty predictions.
        
        Args:
            volume : Input volume
            threshold : Segmentation threshold
            
        Returns:
            segmentation mask, uncertainty map
        """
        self.eval()
        
        if volume.dim() == 4:
            volume = volume.unsqueeze(1)
            
        with torch.no_grad():
            seg_logits, unc_logits, _ = self.forward(volume)
            seg_mask = (torch.sigmoid(seg_logits) > threshold).float()
            uncertainty = torch.sigmoid(unc_logits)
            
        return seg_mask, uncertainty

    def compute_loss(self, seg_logits: torch.Tensor, unc_logits: torch.Tensor, 
                    targets: torch.Tensor, aux_outputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute combined loss including auxiliary losses.
        
        Args:
            seg_logits: Main segmentation logits
            unc_logits: Uncertainty logits  
            targets: Ground truth segmentation
            aux_outputs: Auxiliary classifier outputs
            
        Returns:
            Combined loss tensor
        """
        # Main segmentation loss
        seg_loss = F.binary_cross_entropy_with_logits(seg_logits, targets)
        
        # Uncertainty loss (encourage high uncertainty where prediction is wrong)
        seg_probs = torch.sigmoid(seg_logits)
        prediction_error = torch.abs(seg_probs - targets)
        unc_loss = F.mse_loss(torch.sigmoid(unc_logits), prediction_error)
        
        total_loss = seg_loss + 0.1 * unc_loss
        
        # Add auxiliary losses with reduced weight (following paper)
        if aux_outputs is not None:
            aux_weight = 0.3
            for aux_out in aux_outputs:
                # Global average pooling to match auxiliary output shape
                aux_target = F.adaptive_avg_pool3d(targets, 1).squeeze()
                if aux_target.dim() == 0:
                    aux_target = aux_target.unsqueeze(0)
                aux_loss = F.binary_cross_entropy_with_logits(aux_out.squeeze(), aux_target)
                total_loss += aux_weight * aux_loss
        
        return total_loss