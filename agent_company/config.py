"""
Configuration management for the connectomics pipeline.
Provides centralized configuration with validation and environment-specific settings.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    volume_path: str = "graphene://https://h01-materialization.cr.neuroglancer.org/1.0/h01_c3_flat"
    mip: int = 0
    cache_path: str = "h01_cache"
    chunk_size: List[int] = field(default_factory=lambda: [64, 64, 64])
    batch_size: int = 2
    num_workers: int = 4
    prefetch_factor: int = 2

@dataclass
class ModelConfig:
    """Configuration for the neural network model."""
    input_channels: int = 1
    output_channels: int = 3
    hidden_channels: int = 32
    depth: int = 4
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    activation: str = "relu"

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 20
    gradient_clip_val: float = 1.0
    mixed_precision: bool = True
    accumulate_grad_batches: int = 1

@dataclass
class LossConfig:
    """Configuration for loss functions."""
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    smooth: float = 1e-6

@dataclass
class OptimizationConfig:
    """Configuration for optimization and performance."""
    optimizer: str = "adam"  # "adam", "adamw", "shampoo"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0
    use_amp: bool = True
    num_gpus: int = 1

@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    log_level: str = "INFO"
    log_file: str = "pipeline.log"
    tensorboard_dir: str = "runs"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 10
    validation_frequency: int = 5

@dataclass
class PipelineConfig:
    """Main configuration class that combines all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Environment-specific settings
    environment: str = "development"  # "development", "production", "colab"
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        self._setup_logging()
    
    def _validate(self):
        """Validate configuration values."""
        assert self.data.mip >= 0, "MIP level must be non-negative"
        assert self.model.depth > 0, "Model depth must be positive"
        assert self.training.epochs > 0, "Number of epochs must be positive"
        assert 0 < self.loss.dice_weight < 1, "Dice weight must be between 0 and 1"
        assert self.optimization.num_gpus >= 0, "Number of GPUs must be non-negative"
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.monitoring.log_level.upper())
        
        # Create logs directory if it doesn't exist
        log_dir = Path(self.monitoring.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.monitoring.log_file),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        optimization_config = OptimizationConfig(**config_dict.get('optimization', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
        
        # Extract top-level config
        top_level_config = {k: v for k, v in config_dict.items() 
                           if k not in ['data', 'model', 'training', 'loss', 'optimization', 'monitoring']}
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            loss=loss_config,
            optimization=optimization_config,
            monitoring=monitoring_config,
            **top_level_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'loss': self.loss.__dict__,
            'optimization': self.optimization.__dict__,
            'monitoring': self.monitoring.__dict__,
            'environment': self.environment,
            'seed': self.seed,
            'deterministic': self.deterministic
        }
    
    def save(self, config_path: str):
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides."""
        env_configs = {
            'development': {
                'monitoring': {'log_level': 'DEBUG'},
                'training': {'epochs': 10},
                'data': {'batch_size': 1}
            },
            'production': {
                'monitoring': {'log_level': 'WARNING'},
                'training': {'epochs': 1000},
                'data': {'batch_size': 4},
                'optimization': {'use_amp': True}
            },
            'colab': {
                'monitoring': {'log_level': 'INFO'},
                'training': {'epochs': 50},
                'data': {'batch_size': 2},
                'optimization': {'num_gpus': 1}
            }
        }
        return env_configs.get(self.environment, {})

def load_config(config_path: Optional[str] = None, environment: str = "development") -> PipelineConfig:
    """
    Load configuration with environment-specific overrides.
    
    Args:
        config_path: Path to YAML configuration file
        environment: Environment name (development, production, colab)
    
    Returns:
        PipelineConfig instance
    """
    if config_path and os.path.exists(config_path):
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()
    
    # Apply environment-specific overrides
    config.environment = environment
    env_overrides = config.get_environment_config()
    
    # Apply overrides recursively
    for section, overrides in env_overrides.items():
        section_config = getattr(config, section)
        for key, value in overrides.items():
            setattr(section_config, key, value)
    
    return config

# Default configurations for different environments
DEFAULT_CONFIGS = {
    'development': PipelineConfig(
        environment='development',
        training=TrainingConfig(epochs=10),
        monitoring=MonitoringConfig(log_level='DEBUG')
    ),
    'production': PipelineConfig(
        environment='production',
        training=TrainingConfig(epochs=1000),
        monitoring=MonitoringConfig(log_level='WARNING'),
        optimization=OptimizationConfig(use_amp=True, num_gpus=4)
    ),
    'colab': PipelineConfig(
        environment='colab',
        training=TrainingConfig(epochs=50),
        data=DataConfig(batch_size=2),
        optimization=OptimizationConfig(num_gpus=1)
    )
} 