#!/usr/bin/env python3
"""
Optimized FFN-v2 Training for MacBook
=====================================
Memory-efficient training with MacBook-specific optimizations.
"""

import os
import sys
import time
import logging
import gc
import psutil
from typing import Dict, Any, Optional
import numpy as np

# Import training components
from production_ffn_v2 import ProductionFFNv2Model
from data_loader import SyntheticDataLoader

logger = logging.getLogger(__name__)

class MacBookOptimizedTrainer:
    """Memory-efficient trainer optimized for MacBook."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cpu'  # MacBook optimization: use CPU efficiently
        
        # Memory management
        self.memory_limit_gb = 4.0  # Conservative limit for MacBook
        self.gc_frequency = 10  # Garbage collect every N batches
        
        # Performance tracking
        self.training_stats = {
            'epochs_completed': 0,
            'total_time': 0.0,
            'memory_usage': [],
            'learning_curves': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        }
        
        logger.info("MacBook-optimized trainer initialized")
    
    def check_memory_usage(self):
        """Monitor memory usage and trigger cleanup if needed."""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024**3)
        
        self.training_stats['memory_usage'].append({
            'timestamp': time.time(),
            'memory_gb': memory_gb
        })
        
        if memory_gb > self.memory_limit_gb:
            logger.warning(f"Memory usage high: {memory_gb:.2f}GB, triggering cleanup")
            self.cleanup_memory()
        
        return memory_gb
    
    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        gc.collect()
        
        # Clear PyTorch cache if available
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def create_optimized_model(self) -> ProductionFFNv2Model:
        """Create model with MacBook optimizations."""
        model_config = {
            'input_channels': 1,
            'hidden_channels': [16, 32, 64],  # Smaller for memory efficiency
            'output_channels': 1,
            'use_attention': True,
            'dropout_rate': 0.1,
            'memory_efficient': True,  # Enable memory optimizations
            'gradient_checkpointing': True  # Trade compute for memory
        }
        
        model = ProductionFFNv2Model(model_config)
        logger.info(f"Optimized model created with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def create_optimized_data_loaders(self, train_size: int = 200, val_size: int = 50):
        """Create memory-efficient data loaders."""
        # Smaller datasets for MacBook
        train_loader = SyntheticDataLoader(
            num_samples=train_size,
            volume_shape=(64, 64, 64),  # Smaller volumes
            batch_size=2,  # Smaller batches
            shuffle=True
        )
        
        val_loader = SyntheticDataLoader(
            num_samples=val_size,
            volume_shape=(64, 64, 64),
            batch_size=2,
            shuffle=False
        )
        
        logger.info(f"Created optimized data loaders: train={train_size}, val={val_size}")
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, epoch: int):
        """Memory-efficient training epoch."""
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Memory check
            if batch_idx % self.gc_frequency == 0:
                self.check_memory_usage()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            
            # Calculate loss (simplified for memory efficiency)
            loss = self.calculate_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(outputs, targets)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Progress logging
            if batch_idx % 10 == 0:
                memory_gb = self.check_memory_usage()
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, "
                          f"Acc={accuracy:.4f}, Memory={memory_gb:.2f}GB")
        
        return total_loss / num_batches, total_accuracy / num_batches
    
    def validate_epoch(self, model, val_loader, epoch: int):
        """Memory-efficient validation."""
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                # Memory check
                if batch_idx % self.gc_frequency == 0:
                    self.check_memory_usage()
                
                outputs = model(data)
                loss = self.calculate_loss(outputs, targets)
                accuracy = self.calculate_accuracy(outputs, targets)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        return total_loss / num_batches, total_accuracy / num_batches
    
    def calculate_loss(self, outputs, targets):
        """Calculate loss with memory efficiency."""
        import torch.nn.functional as F
        
        # Ensure shapes match
        if outputs.shape != targets.shape:
            # Reshape if needed
            if len(outputs.shape) == 5 and len(targets.shape) == 5:
                if outputs.shape[1] != targets.shape[1]:
                    # Handle channel dimension mismatch
                    if outputs.shape[1] > targets.shape[1]:
                        outputs = outputs[:, :targets.shape[1], :, :, :]
                    else:
                        targets = targets[:, :outputs.shape[1], :, :, :]
        
        return F.binary_cross_entropy_with_logits(outputs, targets)
    
    def calculate_accuracy(self, outputs, targets):
        """Calculate accuracy with memory efficiency."""
        import torch
        
        # Convert to probabilities
        probs = torch.sigmoid(outputs)
        
        # Binary classification accuracy
        predictions = (probs > 0.5).float()
        accuracy = (predictions == targets).float().mean()
        
        return accuracy.item()
    
    def train(self, num_epochs: int = 20):
        """Main training loop with MacBook optimizations."""
        logger.info("Starting MacBook-optimized training...")
        start_time = time.time()
        
        # Create optimized components
        model = self.create_optimized_model()
        train_loader, val_loader = self.create_optimized_data_loaders()
        
        # Optimizer with learning rate scheduling
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, epoch + 1)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader, epoch + 1)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Update statistics
            self.training_stats['learning_curves']['train_loss'].append(train_loss)
            self.training_stats['learning_curves']['val_loss'].append(val_loss)
            self.training_stats['learning_curves']['train_acc'].append(train_acc)
            self.training_stats['learning_curves']['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            memory_gb = self.check_memory_usage()
            
            # Log progress
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
                       f"Time={epoch_time:.1f}s, Memory={memory_gb:.2f}GB")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(model, optimizer, epoch + 1, val_loss, 'best_ffn_v2_model.pt')
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(model, optimizer, epoch + 1, val_loss, f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Memory cleanup
            self.cleanup_memory()
        
        # Final statistics
        total_time = time.time() - start_time
        self.training_stats['total_time'] = total_time
        self.training_stats['epochs_completed'] = num_epochs
        
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.save_training_stats()
        
        return model
    
    def save_checkpoint(self, model, optimizer, epoch, val_loss, filename):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'training_stats': self.training_stats
        }
        
        import torch
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def save_training_stats(self):
        """Save training statistics."""
        import json
        
        stats_file = 'macbook_training_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        logger.info(f"Training statistics saved to {stats_file}")

def main():
    """Main entry point for MacBook-optimized training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MacBook-Optimized FFN-v2 Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--memory-limit", type=float, default=4.0, help="Memory limit in GB")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # MacBook-specific configuration
    config = {
        'memory_limit_gb': args.memory_limit,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'optimization': {
            'use_mixed_precision': True,
            'gradient_checkpointing': True,
            'memory_efficient': True
        }
    }
    
    # Set environment variables for optimization
    os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(psutil.cpu_count())
    
    # Start training
    trainer = MacBookOptimizedTrainer(config)
    model = trainer.train(args.epochs)
    
    logger.info("MacBook-optimized training completed successfully!")

if __name__ == "__main__":
    main() 