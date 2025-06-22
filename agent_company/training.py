"""
Enhanced training module for the connectomics pipeline with advanced optimization,
monitoring, and error handling capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import warnings

logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """
    Advanced trainer with comprehensive monitoring, optimization, and error handling.
    """
    
    def __init__(self, model: nn.Module, config, device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            config: Configuration object
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_monitoring()
        self._setup_checkpointing()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        optimizer_name = self.config.optimization.optimizer.lower()
        
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_name == "shampoo":
            # Note: This requires the optimizers package
            try:
                from optimizers.distributed_shampoo import DistributedShampoo
                self.optimizer = DistributedShampoo(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay
                )
            except ImportError:
                logger.warning("Shampoo optimizer not available, falling back to Adam")
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay
                )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        logger.info(f"Optimizer: {optimizer_name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_name = self.config.optimization.scheduler.lower()
        
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=1e-6
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler_patience,
                gamma=self.config.training.scheduler_factor
            )
        elif scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        logger.info(f"Scheduler: {scheduler_name}")
    
    def _setup_loss_function(self):
        """Setup loss function."""
        self.criterion = DiceBCELoss(
            dice_weight=self.config.loss.dice_weight,
            bce_weight=self.config.loss.bce_weight,
            smooth=self.config.loss.smooth
        )
        logger.info("Loss function: DiceBCELoss")
    
    def _setup_monitoring(self):
        """Setup monitoring and logging."""
        if self.config.monitoring.tensorboard_dir:
            self.writer = SummaryWriter(self.config.monitoring.tensorboard_dir)
        else:
            self.writer = None
        
        # Setup gradient scaler for mixed precision
        self.scaler = GradScaler() if self.config.optimization.use_amp else None
        
        logger.info("Monitoring setup complete")
    
    def _setup_checkpointing(self):
        """Setup checkpoint directory."""
        self.checkpoint_dir = Path(self.config.monitoring.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            try:
                # Move data to device
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                if self.config.optimization.use_amp and self.scaler:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.training.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip_val
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.training.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip_val
                        )
                    
                    self.optimizer.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
                
                # Log to tensorboard
                if self.writer and batch_idx % 10 == 0:
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), 
                                         self.current_epoch * num_batches + batch_idx)
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    # Move data to device
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    if self.config.optimization.use_amp and self.scaler:
                        with autocast():
                            output = self.model(data)
                            loss = self.criterion(output, target)
                    else:
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Train the model for the specified number of epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        logger.info(f"Starting training for {self.config.training.epochs} epochs")
        
        for epoch in range(self.config.training.epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate if validation loader is provided
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
            else:
                self.scheduler.step()
            
            # Record metrics
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['val_loss'].append(val_metrics.get('val_loss', float('inf')))
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['epoch_times'].append(time.time() - epoch_start_time)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Train/EpochLoss', train_metrics['train_loss'], epoch)
                if val_metrics:
                    self.writer.add_scalar('Val/EpochLoss', val_metrics['val_loss'], epoch)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Train/EpochTime', time.time() - epoch_start_time, epoch)
            
            # Log to console
            logger.info(f"Epoch {epoch + 1}/{self.config.training.epochs}: "
                       f"Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Val Loss: {val_metrics.get('val_loss', 'N/A'):.4f}, "
                       f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            if self.config.monitoring.save_checkpoints and (epoch + 1) % self.config.monitoring.save_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Early stopping
            if val_metrics and self._should_stop_early(val_metrics['val_loss']):
                logger.info("Early stopping triggered")
                break
            
            self.current_epoch += 1
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        self.save_training_history()
        
        logger.info("Training completed")
    
    def _should_stop_early(self, val_loss: float) -> bool:
        """Check if training should stop early."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint("best_model.pt")
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.training.early_stopping_patience
    
    def save_checkpoint(self, filename: str):
        """Save a checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        if any(loss != float('inf') for loss in self.training_history['val_loss']):
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.training_history['learning_rate'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Epoch times
        axes[1, 0].plot(self.training_history['epoch_times'])
        axes[1, 0].set_title('Epoch Training Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)
        
        # Loss distribution
        axes[1, 1].hist(self.training_history['train_loss'], bins=20, alpha=0.7, label='Train')
        if any(loss != float('inf') for loss in self.training_history['val_loss']):
            axes[1, 1].hist([l for l in self.training_history['val_loss'] if l != float('inf')], 
                           bins=20, alpha=0.7, label='Val')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved: {save_path}")
        else:
            plt.show()


class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy loss for segmentation.
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.
        
        Args:
            inputs: Model predictions (B, C, D, H, W)
            targets: Ground truth targets (B, C, D, H, W)
            
        Returns:
            Combined loss value
        """
        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Binary Cross Entropy
        bce_loss = nn.functional.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        
        # Dice Loss
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        
        # Combined loss
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss


def create_trainer(model: nn.Module, config, device: torch.device) -> AdvancedTrainer:
    """
    Create a trainer instance with the given configuration.
    
    Args:
        model: Neural network model
        config: Configuration object
        device: Device to train on
        
    Returns:
        AdvancedTrainer instance
    """
    return AdvancedTrainer(model, config, device) 