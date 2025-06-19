#!/usr/bin/env python3
"""
Training script for FFN-v2 with Inception-3D backbone.

This script implements training strategies inspired by the Google Inception paper:
- Efficient use of computational resources
- Multi-scale feature learning
- Auxiliary loss functions for better gradient flow
- Uncertainty-aware training

Usage:
    python train_ffn_v2.py --config config.yaml
    python train_ffn_v2.py --model ffn_v2_inception_lite --epochs 100

Based on "Going deeper with convolutions" by Szegedy et al. (2014)
https://arxiv.org/pdf/1409.4842
"""

import argparse
import logging
import os
import time
from typing import Dict, Any, Tuple, Optional
import yaml
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    from tqdm import tqdm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch torchvision")

from tool_registry import get_model, list_available_models, ModelConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolumeDataset(Dataset):
    """
    Example dataset for 3D volume segmentation.
    
    In practice, this would load your specific connectomics or medical imaging data.
    Following the paper's principles of efficient data loading and processing.
    """
    
    def __init__(self, data_dir: str, split: str = "train", volume_size: Tuple[int, int, int] = (64, 64, 64)):
        self.data_dir = Path(data_dir)
        self.split = split
        self.volume_size = volume_size
        
        # In practice, load your file list here
        self.samples = self._load_sample_list()
        
    def _load_sample_list(self):
        """Load list of available samples. Placeholder implementation."""
        # This would typically read from a file or scan directory
        # For now, create dummy data for demonstration
        return [f"sample_{i:04d}" for i in range(100 if self.split == "train" else 20)]
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        """
        Load a volume and its segmentation mask.
        
        Returns:
            volume: (1, D, H, W) normalized volume
            mask: (1, D, H, W) binary segmentation mask
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        # Placeholder: generate synthetic data for demonstration
        # In practice, load from HDF5, ZARR, or other formats
        volume = torch.randn(1, *self.volume_size)
        mask = (torch.rand(1, *self.volume_size) > 0.7).float()
        
        # Normalize volume to [0, 1] range
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        return volume, mask


class TrainingConfig:
    """Configuration class following the paper's efficient training principles."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        # Model configuration
        self.model_name = config_dict.get("model_name", "ffn_v2_inception")
        self.model_config = config_dict.get("model_config", {})
        
        # Training configuration
        self.batch_size = config_dict.get("batch_size", 4)
        self.learning_rate = config_dict.get("learning_rate", 1e-3)
        self.num_epochs = config_dict.get("num_epochs", 100)
        self.weight_decay = config_dict.get("weight_decay", 1e-4)
        
        # Data configuration
        self.data_dir = config_dict.get("data_dir", "./data")
        self.volume_size = tuple(config_dict.get("volume_size", [64, 64, 64]))
        
        # Training strategies from the paper
        self.use_auxiliary_loss = config_dict.get("use_auxiliary_loss", True)
        self.auxiliary_weight = config_dict.get("auxiliary_weight", 0.3)
        self.uncertainty_weight = config_dict.get("uncertainty_weight", 0.1)
        
        # Optimization settings
        self.lr_schedule = config_dict.get("lr_schedule", "cosine")
        self.warmup_epochs = config_dict.get("warmup_epochs", 5)
        
        # Output settings
        self.output_dir = config_dict.get("output_dir", "./outputs")
        self.save_frequency = config_dict.get("save_frequency", 10)
        self.log_frequency = config_dict.get("log_frequency", 100)


class FFNTrainer:
    """
    Trainer class implementing the training strategies from the Inception paper.
    
    Key features:
    - Efficient computational budget management
    - Auxiliary loss functions for better gradient flow
    - Uncertainty-aware training
    - Multi-scale learning through the Inception architecture
    """
    
    def __init__(self, config: TrainingConfig):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for training")
            
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Initialize logging
        self.writer = SummaryWriter(os.path.join(config.output_dir, "logs"))
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _create_model(self):
        """Create and initialize the model."""
        model = get_model(self.config.model_name, **self.config.model_config)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
        
    def _create_optimizer(self):
        """Create optimizer following paper's recommendations."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_schedule == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.num_epochs
            )
        elif self.config.lr_schedule == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            
    def _create_data_loaders(self):
        """Create training and validation data loaders."""
        train_dataset = VolumeDataset(
            self.config.data_dir, 
            split="train",
            volume_size=self.config.volume_size
        )
        val_dataset = VolumeDataset(
            self.config.data_dir,
            split="val", 
            volume_size=self.config.volume_size
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        seg_loss_total = 0.0
        unc_loss_total = 0.0
        aux_loss_total = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (volumes, masks) in enumerate(pbar):
            volumes = volumes.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            seg_logits, unc_logits, aux_outputs = self.model(volumes)
            
            # Compute losses
            loss = self.model.compute_loss(seg_logits, unc_logits, masks, aux_outputs)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Log detailed metrics
            if batch_idx % self.config.log_frequency == 0:
                self.writer.add_scalar("train/total_loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/learning_rate", 
                                     self.optimizer.param_groups[0]['lr'], self.global_step)
                
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            self.global_step += 1
            
        return {
            'total_loss': total_loss / len(self.train_loader),
            'seg_loss': seg_loss_total / len(self.train_loader),
            'unc_loss': unc_loss_total / len(self.train_loader),
            'aux_loss': aux_loss_total / len(self.train_loader),
        }
        
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for volumes, masks in tqdm(self.val_loader, desc="Validating"):
                volumes = volumes.to(self.device)
                masks = masks.to(self.device)
                
                seg_logits, unc_logits, aux_outputs = self.model(volumes)
                loss = self.model.compute_loss(seg_logits, unc_logits, masks, aux_outputs)
                
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        return {'total_loss': avg_loss}
        
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint_epoch_{self.epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
            
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            self.writer.add_scalar("epoch/train_loss", train_metrics['total_loss'], epoch)
            self.writer.add_scalar("epoch/val_loss", val_metrics['total_loss'], epoch)
            
            # Check if best model
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                
            # Save checkpoint
            if epoch % self.config.save_frequency == 0 or is_best:
                self.save_checkpoint(is_best)
                
            logger.info(
                f"Epoch {epoch:3d}: "
                f"Train Loss = {train_metrics['total_loss']:.4f}, "
                f"Val Loss = {val_metrics['total_loss']:.4f}"
                f"{' (Best!)' if is_best else ''}"
            )
            
        logger.info("Training completed!")
        self.writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train FFN-v2 with Inception backbone")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, default="ffn_v2_inception", 
                       help="Model name from registry")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name, desc in list_available_models().items():
            print(f"  {name}: {desc}")
        return
        
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Install with: pip install torch torchvision")
        return
        
    # Load configuration
    if args.config:
        config_dict = load_config(args.config)
    else:
        # Create default configuration
        config_dict = {
            "model_name": args.model,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "output_dir": args.output_dir,
        }
        
    config = TrainingConfig(config_dict)
    
    # Initialize trainer and start training
    trainer = FFNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()