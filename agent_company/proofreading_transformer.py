#!/usr/bin/env python3
"""
Proofreading Transformer for Connectomics
========================================
Specialized transformer architecture for learning from human corrections
and improving automated tracing quality through interactive proofreading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from transformer_connectomics import TransformerConfig, MultiHeadAttention3D

logger = logging.getLogger(__name__)

@dataclass
class ProofreadingConfig:
    """Configuration for proofreading transformer."""
    # Architecture parameters
    input_channels: int = 1
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    
    # Proofreading-specific parameters
    use_correction_history: bool = True
    max_correction_history: int = 10
    correction_embed_dim: int = 64
    use_attention_guidance: bool = True
    use_uncertainty_guidance: bool = True
    
    # Interactive features
    use_human_feedback: bool = True
    feedback_embed_dim: int = 128
    use_confidence_estimation: bool = True
    
    # Output configuration
    num_classes: int = 1
    use_uncertainty: bool = True
    use_confidence: bool = True

class CorrectionHistoryEncoder(nn.Module):
    """Encodes correction history for learning from past human edits."""
    
    def __init__(self, embed_dim: int, correction_embed_dim: int, max_history: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.correction_embed_dim = correction_embed_dim
        self.max_history = max_history
        
        # Correction type embedding (add, remove, modify, etc.)
        self.correction_type_embed = nn.Embedding(5, correction_embed_dim)  # 5 correction types
        
        # Position embedding for corrections
        self.position_embed = nn.Embedding(max_history, correction_embed_dim)
        
        # Correction encoder
        self.correction_encoder = nn.Sequential(
            nn.Linear(correction_embed_dim * 3, embed_dim),  # type + position + intensity
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Attention over correction history
        self.history_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )
    
    def forward(self, correction_history: List[Dict[str, Any]]) -> torch.Tensor:
        """Encode correction history into embeddings."""
        if not correction_history:
            return torch.zeros(1, self.embed_dim, device=next(self.parameters()).device)
        
        # Limit history length
        history = correction_history[-self.max_history:]
        
        # Encode each correction
        correction_embeddings = []
        for i, correction in enumerate(history):
            # Correction type (0: add, 1: remove, 2: modify, 3: split, 4: merge)
            correction_type = correction.get('type', 0)
            type_embed = self.correction_type_embed(torch.tensor(correction_type))
            
            # Position embedding
            pos_embed = self.position_embed(torch.tensor(i))
            
            # Correction intensity (confidence of the correction)
            intensity = torch.tensor(correction.get('intensity', 1.0))
            
            # Combine embeddings
            combined = torch.cat([type_embed, pos_embed, intensity.unsqueeze(0)])
            correction_embed = self.correction_encoder(combined)
            correction_embeddings.append(correction_embed)
        
        # Stack corrections
        if correction_embeddings:
            history_embeddings = torch.stack(correction_embeddings)
            
            # Apply attention over history
            attended_history, _ = self.history_attention(
                history_embeddings.unsqueeze(0),
                history_embeddings.unsqueeze(0),
                history_embeddings.unsqueeze(0)
            )
            
            # Global history representation
            global_history = attended_history.mean(dim=1)
            return global_history
        else:
            return torch.zeros(1, self.embed_dim, device=next(self.parameters()).device)

class HumanFeedbackEncoder(nn.Module):
    """Encodes human feedback for interactive learning."""
    
    def __init__(self, embed_dim: int, feedback_embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.feedback_embed_dim = feedback_embed_dim
        
        # Feedback type embedding (approval, rejection, suggestion, etc.)
        self.feedback_type_embed = nn.Embedding(4, feedback_embed_dim)  # 4 feedback types
        
        # Feedback encoder
        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_embed_dim + 1, embed_dim),  # type + confidence
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Cross-attention for feedback integration
        self.feedback_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )
    
    def forward(self, feedback: Dict[str, Any], features: torch.Tensor) -> torch.Tensor:
        """Encode human feedback and integrate with features."""
        # Feedback type (0: approval, 1: rejection, 2: suggestion, 3: correction)
        feedback_type = feedback.get('type', 0)
        type_embed = self.feedback_type_embed(torch.tensor(feedback_type))
        
        # Feedback confidence
        confidence = torch.tensor(feedback.get('confidence', 1.0))
        
        # Combine feedback information
        feedback_info = torch.cat([type_embed, confidence.unsqueeze(0)])
        feedback_embed = self.feedback_encoder(feedback_info)
        
        # Apply cross-attention to integrate feedback with features
        B, N, C = features.shape
        feedback_embed = feedback_embed.expand(B, N, C)
        
        attended_features, _ = self.feedback_attention(
            features, feedback_embed, feedback_embed
        )
        
        return attended_features

class ProofreadingTransformerBlock(nn.Module):
    """Transformer block specialized for proofreading with correction awareness."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, use_correction_awareness: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_correction_awareness = use_correction_awareness
        
        # Self-attention
        self.self_attention = MultiHeadAttention3D(
            embed_dim, num_heads, dropout, use_cross_attention=True
        )
        self.self_attention_norm = nn.LayerNorm(embed_dim)
        
        # Correction-aware attention (if enabled)
        if use_correction_awareness:
            self.correction_attention = MultiHeadAttention3D(
                embed_dim, num_heads, dropout, use_cross_attention=True
            )
            self.correction_attention_norm = nn.LayerNorm(embed_dim)
        
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
    
    def forward(self, x: torch.Tensor, correction_context: Optional[torch.Tensor] = None,
                feedback_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with correction and feedback awareness."""
        # Self-attention
        attn_out = self.self_attention(x)
        x = self.self_attention_norm(x + attn_out)
        
        # Correction-aware attention
        if self.use_correction_awareness and correction_context is not None:
            correction_out = self.correction_attention(x, correction_context)
            x = self.correction_attention_norm(x + correction_out)
        
        # Feedback integration
        if feedback_context is not None:
            x = x + feedback_context
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.mlp_norm(x + mlp_out)
        
        return x

class ProofreadingTransformer(nn.Module):
    """Main proofreading transformer for learning from human corrections."""
    
    def __init__(self, config: ProofreadingConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            config.input_channels, config.embed_dim,
            kernel_size=4, stride=4
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Correction history encoder
        if config.use_correction_history:
            self.correction_encoder = CorrectionHistoryEncoder(
                config.embed_dim, config.correction_embed_dim, config.max_correction_history
            )
        
        # Human feedback encoder
        if config.use_human_feedback:
            self.feedback_encoder = HumanFeedbackEncoder(
                config.embed_dim, config.feedback_embed_dim
            )
        
        # Proofreading transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ProofreadingTransformerBlock(
                config.embed_dim, config.num_heads, config.mlp_ratio,
                config.dropout_rate, config.use_correction_history
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
        
        if config.use_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(config.embed_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Attention guidance (for highlighting areas needing attention)
        if config.use_attention_guidance:
            self.attention_guidance = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(config.embed_dim // 2, 1),
                nn.Sigmoid()
            )
        
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
    
    def forward(self, x: torch.Tensor, correction_history: Optional[List[Dict[str, Any]]] = None,
                human_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with correction history and human feedback."""
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Encode correction history
        correction_context = None
        if self.config.use_correction_history and correction_history:
            correction_context = self.correction_encoder(correction_history)
            correction_context = correction_context.expand(B, x.shape[1], -1)
        
        # Encode human feedback
        feedback_context = None
        if self.config.use_human_feedback and human_feedback:
            feedback_context = self.feedback_encoder(human_feedback, x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, correction_context, feedback_context)
        
        # Final normalization
        x = self.norm(x)
        
        # Output heads
        outputs = {}
        
        # Segmentation output
        seg_out = self.segmentation_head(x)
        seg_out = seg_out.transpose(1, 2).view(B, self.config.num_classes, D//4, H//4, W//4)
        outputs['segmentation'] = torch.sigmoid(seg_out)
        
        # Uncertainty output
        if self.config.use_uncertainty:
            unc_out = self.uncertainty_head(x)
            unc_out = unc_out.transpose(1, 2).view(B, 1, D//4, H//4, W//4)
            outputs['uncertainty'] = unc_out
        
        # Confidence output
        if self.config.use_confidence:
            conf_out = self.confidence_head(x)
            conf_out = conf_out.transpose(1, 2).view(B, 1, D//4, H//4, W//4)
            outputs['confidence'] = conf_out
        
        # Attention guidance
        if self.config.use_attention_guidance:
            attn_guidance = self.attention_guidance(x)
            attn_guidance = attn_guidance.transpose(1, 2).view(B, 1, D//4, H//4, W//4)
            outputs['attention_guidance'] = attn_guidance
        
        return outputs

class InteractiveProofreadingSystem(nn.Module):
    """Complete interactive proofreading system."""
    
    def __init__(self, config: ProofreadingConfig):
        super().__init__()
        self.config = config
        
        # Main proofreading transformer
        self.proofreading_transformer = ProofreadingTransformer(config)
        
        # Upsampling layers for full resolution
        self.upsample_layers = nn.ModuleList()
        current_size = 4  # Patch size
        
        while current_size < 32:  # Upsample to reasonable resolution
            self.upsample_layers.append(
                nn.ConvTranspose3d(config.embed_dim, config.embed_dim, 2, stride=2)
            )
            current_size *= 2
        
        # Final output heads
        self.final_segmentation_head = nn.Conv3d(config.embed_dim, config.num_classes, 1)
        
        if config.use_uncertainty:
            self.final_uncertainty_head = nn.Conv3d(config.embed_dim, 1, 1)
        
        if config.use_confidence:
            self.final_confidence_head = nn.Conv3d(config.embed_dim, 1, 1)
        
        # Correction memory (for storing correction history)
        self.correction_memory = []
        self.max_memory_size = 1000
    
    def add_correction(self, correction: Dict[str, Any]):
        """Add a correction to the memory."""
        self.correction_memory.append(correction)
        
        # Limit memory size
        if len(self.correction_memory) > self.max_memory_size:
            self.correction_memory = self.correction_memory[-self.max_memory_size:]
    
    def get_recent_corrections(self, num_corrections: int = 10) -> List[Dict[str, Any]]:
        """Get recent corrections for context."""
        return self.correction_memory[-num_corrections:] if self.correction_memory else []
    
    def forward(self, x: torch.Tensor, correction_history: Optional[List[Dict[str, Any]]] = None,
                human_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with interactive features."""
        # Get recent corrections if not provided
        if correction_history is None:
            correction_history = self.get_recent_corrections()
        
        # Proofreading transformer forward pass
        outputs = self.proofreading_transformer(x, correction_history, human_feedback)
        
        # Upsample features to full resolution
        features = outputs['segmentation']  # Use segmentation as base features
        
        # Upsample
        for upsample_layer in self.upsample_layers:
            features = upsample_layer(features)
        
        # Final outputs
        final_outputs = {}
        final_outputs['segmentation'] = torch.sigmoid(self.final_segmentation_head(features))
        
        if self.config.use_uncertainty:
            final_outputs['uncertainty'] = torch.sigmoid(self.final_uncertainty_head(features))
        
        if self.config.use_confidence:
            final_outputs['confidence'] = torch.sigmoid(self.final_confidence_head(features))
        
        # Add intermediate outputs
        final_outputs.update(outputs)
        
        return final_outputs
    
    def learn_from_correction(self, input_data: torch.Tensor, correction: Dict[str, Any],
                             target: torch.Tensor, learning_rate: float = 1e-4):
        """Learn from a human correction."""
        # Add correction to memory
        self.add_correction(correction)
        
        # Forward pass with correction context
        outputs = self.forward(input_data, [correction])
        
        # Compute loss
        loss = F.binary_cross_entropy(outputs['segmentation'], target)
        
        # Add uncertainty loss if available
        if 'uncertainty' in outputs and 'uncertainty' in correction:
            uncertainty_target = torch.tensor(correction['uncertainty'])
            uncertainty_loss = F.mse_loss(outputs['uncertainty'], uncertainty_target)
            loss = loss + 0.1 * uncertainty_loss
        
        # Backward pass (simplified - in practice, you'd use an optimizer)
        loss.backward()
        
        # Update weights (simplified)
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
        
        return loss.item()

# Utility functions for proofreading
def create_proofreading_model(config_dict: Dict[str, Any]) -> InteractiveProofreadingSystem:
    """Create proofreading model from configuration."""
    config = ProofreadingConfig(**config_dict)
    return InteractiveProofreadingSystem(config)

def simulate_human_correction(segmentation: torch.Tensor, correction_type: int,
                            position: Tuple[int, int, int], intensity: float = 1.0) -> Dict[str, Any]:
    """Simulate a human correction for testing."""
    return {
        'type': correction_type,  # 0: add, 1: remove, 2: modify, 3: split, 4: merge
        'position': position,
        'intensity': intensity,
        'timestamp': torch.tensor(time.time()),
        'uncertainty': torch.tensor(0.1)  # Low uncertainty for human corrections
    }

def simulate_human_feedback(feedback_type: int, confidence: float = 1.0) -> Dict[str, Any]:
    """Simulate human feedback for testing."""
    return {
        'type': feedback_type,  # 0: approval, 1: rejection, 2: suggestion, 3: correction
        'confidence': confidence,
        'timestamp': torch.tensor(time.time())
    }

# Example usage and testing
def test_proofreading_system():
    """Test the proofreading system with simulated corrections."""
    import time
    
    # Configuration
    config_dict = {
        'input_channels': 1,
        'embed_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'use_correction_history': True,
        'use_human_feedback': True,
        'use_uncertainty': True,
        'use_confidence': True
    }
    
    # Create model
    model = create_proofreading_model(config_dict)
    
    # Test input
    input_tensor = torch.randn(1, 1, 64, 64, 64)
    
    # Initial prediction
    print("Initial prediction...")
    initial_outputs = model(input_tensor)
    print(f"Initial segmentation shape: {initial_outputs['segmentation'].shape}")
    
    # Simulate human corrections
    corrections = [
        simulate_human_correction(initial_outputs['segmentation'], 0, (32, 32, 32), 1.0),  # Add
        simulate_human_correction(initial_outputs['segmentation'], 1, (16, 16, 16), 0.8),  # Remove
        simulate_human_correction(initial_outputs['segmentation'], 2, (48, 48, 48), 0.9),  # Modify
    ]
    
    # Add corrections to model
    for correction in corrections:
        model.add_correction(correction)
    
    # Prediction with correction history
    print("\nPrediction with correction history...")
    corrected_outputs = model(input_tensor, corrections)
    print(f"Corrected segmentation shape: {corrected_outputs['segmentation'].shape}")
    
    # Simulate human feedback
    feedback = simulate_human_feedback(0, 0.9)  # Approval with high confidence
    
    # Prediction with feedback
    print("\nPrediction with human feedback...")
    feedback_outputs = model(input_tensor, corrections, feedback)
    print(f"Feedback segmentation shape: {feedback_outputs['segmentation'].shape}")
    
    # Test learning from correction
    print("\nLearning from correction...")
    target = torch.randn_like(feedback_outputs['segmentation'])
    loss = model.learn_from_correction(input_tensor, corrections[0], target)
    print(f"Learning loss: {loss:.4f}")
    
    return model, initial_outputs, corrected_outputs, feedback_outputs

if __name__ == "__main__":
    test_proofreading_system() 