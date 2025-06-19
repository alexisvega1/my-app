#!/usr/bin/env python3
"""
Advanced Continual Learning System for Production Connectomics
============================================================
Next-generation continual learning with sophisticated training strategies,
memory management, and production monitoring for large-scale model updates.
"""

import os
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import json
import hashlib
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)

# Production-grade imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torch.optim import Adam, AdamW, SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
    logger.info("PyTorch available for advanced continual learning")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using stub implementation")

try:
    import dask.array as da
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask not available - distributed processing disabled")

try:
    import zarr
    import numcodecs
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    logger.warning("Zarr not available - chunked storage disabled")

@dataclass
class TrainingResult:
    """Advanced training result with comprehensive metadata."""
    model_state: Dict[str, Any]
    training_loss: float
    validation_loss: float
    learning_rate: float
    epoch: int
    training_time: float
    memory_usage: Dict[str, float]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    metadata: Dict[str, Any]

class ExperienceReplayBuffer:
    """Advanced experience replay buffer with priority sampling."""
    
    def __init__(self, 
                 max_size: int = 10000,
                 priority_alpha: float = 0.6,
                 priority_beta: float = 0.4):
        self.max_size = max_size
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.position = 0
        self.size = 0
        
        logger.info(f"Experience replay buffer initialized with max_size={max_size}")
    
    def add(self, experience: Dict[str, Any], priority: float = 1.0):
        """Add experience to buffer with priority."""
        if self.size < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority ** self.priority_alpha)
            self.size += 1
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.priority_alpha
            self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], List[int], List[float]]:
        """Sample experiences with priority weighting."""
        if self.size == 0:
            return [], [], []
        
        # Calculate sampling probabilities
        priorities = np.array(list(self.priorities)[:self.size])
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance weights
        weights = (self.size * probabilities[indices]) ** (-self.priority_beta)
        weights /= np.max(weights)  # Normalize
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices.tolist(), weights.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if idx < self.size:
                self.priorities[idx] = priority ** self.priority_alpha
    
    def get_size(self) -> int:
        return self.size
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.position = 0
        self.size = 0

class AdvancedLoRAModule(nn.Module):
    """Advanced LoRA module with multiple adaptation strategies."""
    
    def __init__(self, 
                 base_model: nn.Module,
                 rank: int = 16,
                 alpha: float = 32.0,
                 dropout: float = 0.1,
                 adaptation_strategy: str = 'low_rank'):
        super().__init__()
        
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.adaptation_strategy = adaptation_strategy
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Initialize LoRA adapters
        self.lora_adapters = nn.ModuleDict()
        self._initialize_lora_adapters()
        
        logger.info(f"Advanced LoRA module initialized with rank={rank}")
    
    def _initialize_lora_adapters(self):
        """Initialize LoRA adapters for different layers."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Create LoRA adapter for linear layers
                adapter = self._create_lora_adapter(module)
                self.lora_adapters[name] = adapter
            elif isinstance(module, nn.Conv3d):
                # Create LoRA adapter for conv layers
                adapter = self._create_conv_lora_adapter(module)
                self.lora_adapters[name] = adapter
    
    def _create_lora_adapter(self, linear_layer: nn.Linear) -> nn.Module:
        """Create LoRA adapter for linear layer."""
        return nn.Sequential(
            nn.Linear(linear_layer.in_features, self.rank, bias=False),
            nn.Dropout(self.dropout),
            nn.Linear(self.rank, linear_layer.out_features, bias=False)
        )
    
    def _create_conv_lora_adapter(self, conv_layer: nn.Conv3d) -> nn.Module:
        """Create LoRA adapter for 3D conv layer."""
        return nn.Sequential(
            nn.Conv3d(conv_layer.in_channels, self.rank, 1, bias=False),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(self.rank, conv_layer.out_channels, 1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Apply base model
        base_output = self.base_model(x)
        
        # Apply LoRA adaptations
        adapted_output = base_output
        for name, adapter in self.lora_adapters.items():
            # Get the corresponding module in base model
            module = dict(self.base_model.named_modules())[name]
            
            if isinstance(module, nn.Linear):
                # Apply LoRA to linear layer
                lora_output = adapter(x.view(x.size(0), -1))
                adapted_output = adapted_output + (self.alpha / self.rank) * lora_output.view_as(adapted_output)
            elif isinstance(module, nn.Conv3d):
                # Apply LoRA to conv layer
                lora_output = adapter(x)
                adapted_output = adapted_output + (self.alpha / self.rank) * lora_output
        
        return adapted_output
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters (only LoRA adapters)."""
        return list(self.lora_adapters.parameters())
    
    def save_adapters(self, path: str):
        """Save LoRA adapters."""
        torch.save({
            'lora_adapters': self.lora_adapters.state_dict(),
            'rank': self.rank,
            'alpha': self.alpha,
            'adaptation_strategy': self.adaptation_strategy
        }, path)
    
    def load_adapters(self, path: str):
        """Load LoRA adapters."""
        checkpoint = torch.load(path)
        self.lora_adapters.load_state_dict(checkpoint['lora_adapters'])

class ContinualLearningDataset(Dataset):
    """Dataset for continual learning with dynamic data loading."""
    
    def __init__(self, 
                 data_paths: List[str],
                 transform=None,
                 cache_size: int = 1000):
        self.data_paths = data_paths
        self.transform = transform
        self.cache_size = cache_size
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_order = deque()
        
        logger.info(f"Continual learning dataset initialized with {len(data_paths)} paths")
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item with caching."""
        if idx in self.cache:
            # Move to end of cache order
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            return self.cache[idx]
        
        # Load data
        data_path = self.data_paths[idx]
        data = self._load_data(data_path)
        
        # Apply transform
        if self.transform:
            data = self.transform(data)
        
        # Cache data
        if len(self.cache) < self.cache_size:
            self.cache[idx] = data
            self.cache_order.append(idx)
        else:
            # Remove oldest item
            oldest_idx = self.cache_order.popleft()
            del self.cache[oldest_idx]
            self.cache[idx] = data
            self.cache_order.append(idx)
        
        return data
    
    def _load_data(self, data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data from path."""
        if ZARR_AVAILABLE and data_path.endswith('.zarr'):
            store = zarr.open(data_path, mode='r')
            input_data = torch.FloatTensor(store['input'][:])
            target_data = torch.FloatTensor(store['target'][:])
        else:
            # Load as numpy arrays
            input_data = torch.FloatTensor(np.load(f"{data_path}_input.npy"))
            target_data = torch.FloatTensor(np.load(f"{data_path}_target.npy"))
        
        return input_data, target_data
    
    def add_data_paths(self, new_paths: List[str]):
        """Add new data paths for continual learning."""
        self.data_paths.extend(new_paths)
        logger.info(f"Added {len(new_paths)} new data paths")

class AdvancedContinualLearner:
    """Production-ready advanced continual learning system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'model_config': {
                'rank': 16,
                'alpha': 32.0,
                'dropout': 0.1,
                'adaptation_strategy': 'low_rank'
            },
            'training_config': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'num_epochs': 10,
                'validation_split': 0.2,
                'early_stopping_patience': 5
            },
            'memory_config': {
                'replay_buffer_size': 10000,
                'priority_alpha': 0.6,
                'priority_beta': 0.4,
                'cache_size': 1000
            },
            'distributed_config': {
                'num_processes': mp.cpu_count(),
                'num_threads': 10
            }
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.replay_buffer = None
        self.dataset = None
        
        # Performance tracking
        self.stats = {
            'training_sessions': 0,
            'total_training_time': 0.0,
            'total_epochs': 0,
            'best_validation_loss': float('inf')
        }
        
        logger.info("Advanced continual learner initialized")
    
    def initialize_model(self, base_model: nn.Module) -> bool:
        """Initialize the continual learning model."""
        try:
            self.model = AdvancedLoRAModule(
                base_model=base_model,
                **self.config['model_config']
            )
            
            # Initialize optimizer
            trainable_params = self.model.get_trainable_parameters()
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config['training_config']['learning_rate'],
                weight_decay=self.config['training_config']['weight_decay']
            )
            
            # Initialize scheduler
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training_config']['num_epochs']
            )
            
            # Initialize replay buffer
            self.replay_buffer = ExperienceReplayBuffer(
                **self.config['memory_config']
            )
            
            logger.info("Model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    def train_on_new_data(self, 
                         new_data_paths: List[str],
                         validation_data_paths: Optional[List[str]] = None) -> TrainingResult:
        """Train on new data with continual learning."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        start_time = time.time()
        
        try:
            # Create dataset
            self.dataset = ContinualLearningDataset(
                new_data_paths,
                cache_size=self.config['memory_config']['cache_size']
            )
            
            # Split data
            if validation_data_paths:
                train_paths = new_data_paths
                val_paths = validation_data_paths
            else:
                split_idx = int(len(new_data_paths) * (1 - self.config['training_config']['validation_split']))
                train_paths = new_data_paths[:split_idx]
                val_paths = new_data_paths[split_idx:]
            
            train_dataset = ContinualLearningDataset(train_paths)
            val_dataset = ContinualLearningDataset(val_paths)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training_config']['batch_size'],
                shuffle=True,
                num_workers=4
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training_config']['batch_size'],
                shuffle=False,
                num_workers=4
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config['training_config']['num_epochs']):
                # Training phase
                train_loss, train_metrics = self._train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_metrics = self._validate_epoch(val_loader)
                
                # Update scheduler
                self.scheduler.step()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint('best_model.pt', epoch, val_loss)
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['training_config']['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Log progress
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
                
                self.stats['total_epochs'] += 1
            
            training_time = time.time() - start_time
            self.stats['total_training_time'] += training_time
            self.stats['training_sessions'] += 1
            
            return TrainingResult(
                model_state=self.model.state_dict(),
                training_loss=train_loss,
                validation_loss=val_loss,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                epoch=epoch,
                training_time=training_time,
                memory_usage=self._get_memory_usage(),
                training_metrics=train_metrics,
                validation_metrics=val_metrics,
                metadata={
                    'new_data_paths': len(new_data_paths),
                    'train_samples': len(train_dataset),
                    'val_samples': len(val_dataset),
                    'replay_buffer_size': self.replay_buffer.get_size()
                }
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        metrics = defaultdict(float)
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update replay buffer
            self.replay_buffer.add({
                'input': inputs.detach().cpu(),
                'target': targets.detach().cpu(),
                'loss': loss.item()
            }, priority=loss.item())
            
            total_loss += loss.item()
            metrics['batch_loss'] += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}: Loss={loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        metrics['avg_loss'] = avg_loss
        
        return avg_loss, dict(metrics)
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                loss = F.binary_cross_entropy_with_logits(outputs, targets)
                
                total_loss += loss.item()
                metrics['batch_loss'] += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        metrics['avg_loss'] = avg_loss
        
        return avg_loss, dict(metrics)
    
    def _save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'stats': self.stats
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> bool:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.stats = checkpoint.get('stats', self.stats)
            
            logger.info(f"Checkpoint loaded: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return {
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**3,
                'cpu_memory': 0.0
            }
        else:
            return {
                'gpu_memory_allocated': 0.0,
                'gpu_memory_cached': 0.0,
                'cpu_memory': 0.0
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            **self.stats,
            'config': self.config,
            'replay_buffer_size': self.replay_buffer.get_size() if self.replay_buffer else 0
        }
    
    def save_adapters(self, path: str):
        """Save LoRA adapters."""
        if self.model:
            self.model.save_adapters(path)
            logger.info(f"Adapters saved: {path}")
    
    def load_adapters(self, path: str):
        """Load LoRA adapters from path."""
        if hasattr(self, 'lora_module'):
            self.lora_module.load_adapters(path)
            logger.info(f"LoRA adapters loaded from {path}")

    def train(self, 
              segmentation: np.ndarray,
              uncertainty_map: np.ndarray,
              metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Train method for production pipeline compatibility.
        Wraps train_on_new_data() for the production pipeline interface.
        """
        try:
            # Create temporary data paths for the input arrays
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save arrays as temporary files
                seg_path = os.path.join(temp_dir, "segmentation.npy")
                unc_path = os.path.join(temp_dir, "uncertainty.npy")
                
                np.save(seg_path, segmentation)
                np.save(unc_path, uncertainty_map)
                
                # Train on the new data
                result = self.train_on_new_data([seg_path, unc_path])
                
                if result:
                    logger.info("Continual learning training completed successfully")
                    return {
                        'training_loss': result.training_loss,
                        'validation_loss': result.validation_loss,
                        'learning_rate': result.learning_rate,
                        'epoch': result.epoch,
                        'training_time': result.training_time,
                        'metadata': metadata
                    }
                else:
                    logger.warning("Continual learning training failed")
                    return None
                    
        except Exception as e:
            logger.error(f"Continual learning training error: {e}")
            return None

    def adapt(self, 
              new_data: Dict[str, Any],
              metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Adapt method for production pipeline compatibility.
        Wraps train_on_new_data() for the production pipeline interface.
        """
        try:
            # Extract data from the new_data dict
            if 'segmentation' in new_data and 'uncertainty_map' in new_data:
                return self.train(
                    segmentation=new_data['segmentation'],
                    uncertainty_map=new_data['uncertainty_map'],
                    metadata=metadata
                )
            else:
                logger.warning("Invalid data format for adaptation")
                return None
                
        except Exception as e:
            logger.error(f"Continual learning adaptation error: {e}")
            return None 