#!/usr/bin/env python3
"""
Quick FFN-v2 Training for Demo
=============================
Quick training script for immediate demonstration of the production pipeline.
"""

import os
import sys
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple, List, Dict, Any
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_ffn_v2 import ProductionFFNv2Model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuickConnectomicsDataset(Dataset):
    """Quick synthetic connectomics dataset for demo training."""
    
    def __init__(self, num_samples: int = 100, volume_size: Tuple[int, int, int] = (32, 32, 32)):
        self.num_samples = num_samples
        self.volume_size = volume_size
        logger.info(f"Creating quick dataset with {num_samples} samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate simple synthetic data."""
        # Simple synthetic EM volume
        em_volume = np.random.random(self.volume_size).astype(np.float32)
        
        # Simple segmentation (threshold-based)
        segmentation = (em_volume > 0.5).astype(np.float32)
        
        # Add channel dimension
        em_volume = em_volume[None, ...]
        segmentation = segmentation[None, ...]
        
        return torch.FloatTensor(em_volume), torch.FloatTensor(segmentation)

def quick_train():
    """Quick training for demo."""
    logger.info("Starting Quick FFN-v2 Training for Demo")
    
    # Smaller model config for quick training
    model_config = {
        'input_channels': 1,
        'hidden_channels': [16, 32, 64],  # Smaller model
        'output_channels': 1,
        'use_attention': True,
        'dropout_rate': 0.1
    }
    
    training_config = {
        'batch_size': 4,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 5,  # Quick training
        'early_stopping_patience': 3
    }
    
    # Create datasets
    train_dataset = QuickConnectomicsDataset(num_samples=50, volume_size=(32, 32, 32))
    val_dataset = QuickConnectomicsDataset(num_samples=10, volume_size=(32, 32, 32))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
    
    # Initialize model
    model = ProductionFFNv2Model(**model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=training_config['learning_rate'])
    
    logger.info(f"Quick training on {device}")
    
    # Quick training loop
    for epoch in range(training_config['num_epochs']):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            segmentation, uncertainty = model(inputs)
            
            # Loss
            loss = F.binary_cross_entropy_with_logits(segmentation, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Save model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'training_config': training_config
    }
    
    model_path = 'quick_ffn_v2_model.pt'
    torch.save(checkpoint, model_path)
    logger.info(f"Quick model saved: {model_path}")
    
    return model_path

if __name__ == "__main__":
    quick_train() 