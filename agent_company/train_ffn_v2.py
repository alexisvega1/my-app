#!/usr/bin/env python3
"""
FFN-v2 Model Training from Scratch
=================================
Train a real FFN-v2 model from scratch using synthetic connectomics data.
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
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import json
import tempfile
import shutil

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_ffn_v2 import ProductionFFNv2Model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyntheticConnectomicsDataset(Dataset):
    """Synthetic connectomics dataset for training FFN-v2."""
    
    def __init__(self, 
                 num_samples: int = 1000,
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 noise_level: float = 0.1,
                 complexity: float = 0.5):
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.noise_level = noise_level
        self.complexity = complexity
        
        logger.info(f"Creating synthetic dataset with {num_samples} samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic EM volume and ground truth segmentation."""
        # Generate synthetic EM volume
        em_volume = self._generate_em_volume()
        
        # Generate ground truth segmentation
        segmentation = self._generate_segmentation(em_volume)
        
        # Add noise to EM volume
        em_volume += np.random.normal(0, self.noise_level, em_volume.shape)
        em_volume = np.clip(em_volume, 0, 1)
        
        # Add channel dimension
        em_volume = em_volume[None, ...]  # shape: (1, D, H, W)
        segmentation = segmentation[None, ...]  # shape: (1, D, H, W)
        
        return torch.FloatTensor(em_volume), torch.FloatTensor(segmentation)
    
    def _generate_em_volume(self) -> np.ndarray:
        """Generate synthetic EM volume with realistic structures."""
        volume = np.zeros(self.volume_size, dtype=np.float32)
        
        # Add membrane-like structures
        num_membranes = int(5 + 10 * self.complexity)
        for _ in range(num_membranes):
            # Random membrane
            center = np.random.randint(0, min(self.volume_size), 3)
            radius = np.random.randint(5, 15)
            thickness = np.random.randint(1, 3)
            
            # Create membrane
            z, y, x = np.ogrid[:self.volume_size[0], :self.volume_size[1], :self.volume_size[2]]
            distance = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
            
            # Membrane is a thin shell
            membrane = (distance >= radius - thickness) & (distance <= radius + thickness)
            volume[membrane] += 0.8
        
        # Add synaptic vesicles
        num_vesicles = int(20 + 30 * self.complexity)
        for _ in range(num_vesicles):
            center = np.random.randint(0, min(self.volume_size), 3)
            radius = np.random.randint(2, 6)
            
            z, y, x = np.ogrid[:self.volume_size[0], :self.volume_size[1], :self.volume_size[2]]
            distance = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
            
            vesicle = distance <= radius
            volume[vesicle] += 0.6
        
        # Add microtubules (linear structures)
        num_microtubules = int(3 + 5 * self.complexity)
        for _ in range(num_microtubules):
            start = np.random.randint(0, min(self.volume_size), 3)
            end = np.random.randint(0, min(self.volume_size), 3)
            radius = np.random.randint(1, 3)
            
            # Create linear microtubule
            t = np.linspace(0, 1, 50)
            for ti in t:
                point = start + ti * (end - start)
                point = point.astype(int)
                
                if (0 <= point[0] < self.volume_size[0] and 
                    0 <= point[1] < self.volume_size[1] and 
                    0 <= point[2] < self.volume_size[2]):
                    
                    z, y, x = np.ogrid[:self.volume_size[0], :self.volume_size[1], :self.volume_size[2]]
                    distance = np.sqrt((z - point[0])**2 + (y - point[1])**2 + (x - point[2])**2)
                    
                    microtubule = distance <= radius
                    volume[microtubule] += 0.7
        
        # Normalize
        volume = np.clip(volume, 0, 1)
        return volume
    
    def _generate_segmentation(self, em_volume: np.ndarray) -> np.ndarray:
        """Generate ground truth segmentation from EM volume."""
        # Simple thresholding-based segmentation
        # In practice, this would be manual annotation
        segmentation = np.zeros_like(em_volume)
        
        # Segment membranes (high intensity regions)
        membrane_threshold = 0.6
        membranes = em_volume > membrane_threshold
        
        # Segment vesicles (medium intensity, spherical)
        vesicle_threshold = 0.4
        vesicles = (em_volume > vesicle_threshold) & (em_volume <= membrane_threshold)
        
        # Segment microtubules (linear structures)
        microtubule_threshold = 0.5
        microtubules = (em_volume > microtubule_threshold) & (em_volume <= membrane_threshold)
        
        # Combine all structures
        segmentation = membranes | vesicles | microtubules
        
        # Clean up small components
        from scipy import ndimage
        labeled, num_components = ndimage.label(segmentation)
        for i in range(1, num_components + 1):
            component_size = np.sum(labeled == i)
            if component_size < 10:  # Remove small components
                segmentation[labeled == i] = False
        
        return segmentation.astype(np.float32)

class FFNv2Trainer:
    """Trainer for FFN-v2 model."""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any]):
        self.model_config = model_config
        self.training_config = training_config
        
        # Initialize model
        self.model = ProductionFFNv2Model(**model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=training_config['num_epochs']
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        logger.info(f"FFN-v2 trainer initialized on {self.device}")
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int) -> Dict[str, Any]:
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_accuracy = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self._validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint('best_ffn_v2_model.pt', epoch, val_loss)
            else:
                patience_counter += 1
            
            if patience_counter >= self.training_config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            segmentation, uncertainty = self.model(inputs)
            
            # Calculate loss (combine segmentation and uncertainty)
            seg_loss = F.binary_cross_entropy_with_logits(segmentation, targets)
            unc_loss = F.mse_loss(uncertainty, torch.zeros_like(uncertainty))  # Encourage low uncertainty
            total_batch_loss = seg_loss + 0.1 * unc_loss
            
            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            pred_seg = torch.sigmoid(segmentation) > 0.5
            accuracy = (pred_seg == targets).float().mean().item()
            
            total_loss += total_batch_loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}: Loss={total_batch_loss.item():.4f}, Acc={accuracy:.4f}")
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                segmentation, uncertainty = self.model(inputs)
                
                # Calculate loss
                seg_loss = F.binary_cross_entropy_with_logits(segmentation, targets)
                unc_loss = F.mse_loss(uncertainty, torch.zeros_like(uncertainty))
                total_batch_loss = seg_loss + 0.1 * unc_loss
                
                # Calculate accuracy
                pred_seg = torch.sigmoid(segmentation) > 0.5
                accuracy = (pred_seg == targets).float().mean().item()
                
                total_loss += total_batch_loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def _save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'history': self.history
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> bool:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.history = checkpoint.get('history', self.history)
            
            logger.info(f"Checkpoint loaded: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_accuracy'], label='Train Accuracy')
        ax2.plot(self.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        lr_history = [group['lr'] for group in self.optimizer.param_groups]
        ax3.plot(lr_history)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Loss vs Accuracy
        ax4.scatter(self.history['train_loss'], self.history['train_accuracy'], 
                   alpha=0.6, label='Train')
        ax4.scatter(self.history['val_loss'], self.history['val_accuracy'], 
                   alpha=0.6, label='Val')
        ax4.set_title('Loss vs Accuracy')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved: {save_path}")
        
        plt.show()

def main():
    """Main training function."""
    logger.info("Starting FFN-v2 Model Training")
    logger.info("=" * 50)
    
    # Configuration
    model_config = {
        'input_channels': 1,
        'hidden_channels': [32, 64, 128, 256],
        'output_channels': 1,
        'use_attention': True,
        'dropout_rate': 0.1
    }
    
    training_config = {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 20,
        'early_stopping_patience': 5
    }
    
    # Create datasets
    logger.info("Creating synthetic datasets...")
    train_dataset = SyntheticConnectomicsDataset(
        num_samples=500,
        volume_size=(64, 64, 64),
        noise_level=0.1,
        complexity=0.7
    )
    
    val_dataset = SyntheticConnectomicsDataset(
        num_samples=100,
        volume_size=(64, 64, 64),
        noise_level=0.1,
        complexity=0.7
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize trainer
    trainer = FFNv2Trainer(model_config, training_config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader, training_config['num_epochs'])
    
    # Plot training history
    trainer.plot_training_history('ffn_v2_training_history.png')
    
    # Save final model
    final_model_path = 'ffn_v2_final_model.pt'
    trainer._save_checkpoint(final_model_path, training_config['num_epochs'], history['val_loss'][-1])
    
    logger.info(f"Training completed! Final model saved: {final_model_path}")
    logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
    logger.info(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")

if __name__ == "__main__":
    main() 