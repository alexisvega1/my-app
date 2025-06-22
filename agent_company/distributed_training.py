"""
Distributed Training Module for Enhanced Connectomics Pipeline
=============================================================

Enables multi-GPU training with PyTorch Distributed Data Parallel (DDP)
for production-scale connectomics analysis.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from config import PipelineConfig
from data_loader import create_data_loader
from training import AdvancedTrainer, create_trainer
from ffn_v2_mathematical_model import MathematicalFFNv2

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """
    Distributed trainer supporting multi-GPU training with DDP.
    """
    
    def __init__(self, config: PipelineConfig, rank: int, world_size: int):
        """
        Initialize distributed trainer.
        
        Args:
            config: Pipeline configuration
            rank: Process rank (0 for main process)
            world_size: Total number of processes
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Setup distributed environment
        self._setup_distributed()
        
        # Initialize components
        self.model = None
        self.trainer = None
        
        logger.info(f"Distributed trainer initialized on rank {rank}/{world_size}")
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        if self.world_size > 1:
            # Initialize process group
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                rank=self.rank,
                world_size=self.world_size
            )
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.rank)
            
            logger.info(f"Distributed environment setup complete on rank {self.rank}")
    
    def setup_model(self) -> bool:
        """Setup model with DDP wrapper."""
        try:
            # Create model
            self.model = MathematicalFFNv2(
                input_channels=self.config.model.input_channels,
                output_channels=self.config.model.output_channels,
                hidden_channels=self.config.model.hidden_channels,
                depth=self.config.model.depth
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Wrap with DDP if using multiple GPUs
            if self.world_size > 1:
                self.model = DDP(
                    self.model,
                    device_ids=[self.rank] if torch.cuda.is_available() else None,
                    output_device=self.rank if torch.cuda.is_available() else None,
                    find_unused_parameters=False
                )
            
            # Print model info on main process
            if self.rank == 0:
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model created with {total_params:,} parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup model on rank {self.rank}: {e}")
            return False
    
    def setup_data_loaders(self):
        """Setup distributed data loaders."""
        try:
            # Create datasets
            train_dataset = create_data_loader(self.config, "dataset")
            val_dataset = create_data_loader(self.config, "dataset")
            
            # Create distributed samplers
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            
            # Create data loaders
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.data.batch_size,
                sampler=train_sampler,
                num_workers=self.config.data.num_workers,
                pin_memory=True,
                prefetch_factor=self.config.data.prefetch_factor
            )
            
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.data.batch_size,
                sampler=val_sampler,
                num_workers=self.config.data.num_workers,
                pin_memory=True,
                prefetch_factor=self.config.data.prefetch_factor
            )
            
            logger.info(f"Data loaders setup complete on rank {self.rank}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup data loaders on rank {self.rank}: {e}")
            return False
    
    def setup_trainer(self) -> bool:
        """Setup trainer for distributed training."""
        try:
            if self.model is None:
                raise RuntimeError("Model must be setup before trainer")
            
            # Create trainer
            self.trainer = create_trainer(self.model, self.config, self.device)
            
            # Adjust batch size for distributed training
            effective_batch_size = self.config.data.batch_size * self.world_size
            if self.rank == 0:
                logger.info(f"Effective batch size: {effective_batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup trainer on rank {self.rank}: {e}")
            return False
    
    def train(self):
        """Run distributed training."""
        try:
            if self.rank == 0:
                logger.info(f"Starting distributed training with {self.world_size} GPUs")
            
            # Train the model
            self.trainer.train(self.train_loader, self.val_loader)
            
            if self.rank == 0:
                logger.info("Distributed training completed")
                
        except Exception as e:
            logger.error(f"Training failed on rank {self.rank}: {e}")
            raise
    
    def cleanup(self):
        """Cleanup distributed environment."""
        if self.world_size > 1:
            dist.destroy_process_group()
            logger.info(f"Distributed environment cleaned up on rank {self.rank}")


def setup_distributed_training(config: PipelineConfig) -> DistributedTrainer:
    """
    Setup distributed training environment.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        DistributedTrainer instance
    """
    # Get environment variables for distributed training
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Update device rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    return DistributedTrainer(config, rank, world_size)


def launch_distributed_training(config_path: Optional[str] = None, 
                              num_gpus: int = 4,
                              environment: str = "production"):
    """
    Launch distributed training using torchrun.
    
    Args:
        config_path: Path to configuration file
        num_gpus: Number of GPUs to use
        environment: Environment name
    """
    from enhanced_pipeline import EnhancedConnectomicsPipeline
    
    # Create pipeline
    pipeline = EnhancedConnectomicsPipeline(config_path, environment)
    
    # Setup distributed training
    distributed_trainer = setup_distributed_training(pipeline.config)
    
    # Setup components
    if not distributed_trainer.setup_model():
        raise RuntimeError("Failed to setup model")
    
    if not distributed_trainer.setup_data_loaders():
        raise RuntimeError("Failed to setup data loaders")
    
    if not distributed_trainer.setup_trainer():
        raise RuntimeError("Failed to setup trainer")
    
    # Train
    distributed_trainer.train()
    
    # Cleanup
    distributed_trainer.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Training for Connectomics Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--environment", type=str, default="production", 
                       choices=["development", "production", "colab"],
                       help="Environment to run in")
    
    args = parser.parse_args()
    
    # Launch distributed training
    launch_distributed_training(args.config, args.num_gpus, args.environment) 