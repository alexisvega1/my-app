"""
Automated Hyperparameter Optimization for Connectomics Pipeline
==============================================================

Uses Optuna for automated hyperparameter tuning and optimization
of model architectures and training parameters.
"""

import optuna
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Callable
import json
from pathlib import Path
import numpy as np
from datetime import datetime

from config import PipelineConfig, load_config
from enhanced_pipeline import EnhancedConnectomicsPipeline
from advanced_models import create_advanced_model
from training import AdvancedTrainer

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization using Optuna.
    """
    
    def __init__(self, config: PipelineConfig, n_trials: int = 100, 
                 timeout: Optional[int] = None, study_name: str = "connectomics_optimization"):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            config: Base configuration
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            study_name: Name for the Optuna study
        """
        self.config = config
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction="minimize",  # Minimize validation loss
            study_name=study_name,
            storage="sqlite:///optuna_studies.db",
            load_if_exists=True
        )
        
        logger.info(f"Hyperparameter optimization initialized with {n_trials} trials")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss (to be minimized)
        """
        try:
            # Suggest hyperparameters
            hyperparams = self._suggest_hyperparameters(trial)
            
            # Create configuration with suggested hyperparameters
            config = self._create_config_with_hyperparams(hyperparams)
            
            # Create and train model
            validation_loss = self._train_and_evaluate(config, trial)
            
            # Report intermediate values
            trial.report(validation_loss, step=0)
            
            return validation_loss
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('inf')  # Return high loss for failed trials
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the trial."""
        hyperparams = {}
        
        # Model architecture hyperparameters
        hyperparams['model_type'] = trial.suggest_categorical(
            'model_type', 
            ['transformer_ffn', 'swin_transformer', 'hybrid', 'mathematical_ffn']
        )
        
        hyperparams['hidden_channels'] = trial.suggest_categorical(
            'hidden_channels', 
            [32, 64, 128, 256]
        )
        
        hyperparams['depth'] = trial.suggest_int('depth', 2, 6)
        
        if hyperparams['model_type'] in ['transformer_ffn', 'hybrid']:
            hyperparams['num_heads'] = trial.suggest_categorical(
                'num_heads', 
                [4, 8, 16]
            )
        
        # Training hyperparameters
        hyperparams['learning_rate'] = trial.suggest_float(
            'learning_rate', 
            1e-5, 1e-2, 
            log=True
        )
        
        hyperparams['batch_size'] = trial.suggest_categorical(
            'batch_size', 
            [1, 2, 4, 8]
        )
        
        hyperparams['optimizer'] = trial.suggest_categorical(
            'optimizer', 
            ['adam', 'adamw', 'sgd', 'rmsprop']
        )
        
        hyperparams['weight_decay'] = trial.suggest_float(
            'weight_decay', 
            1e-6, 1e-3, 
            log=True
        )
        
        # Scheduler hyperparameters
        hyperparams['scheduler'] = trial.suggest_categorical(
            'scheduler', 
            ['cosine', 'step', 'exponential', 'plateau']
        )
        
        if hyperparams['scheduler'] == 'step':
            hyperparams['step_size'] = trial.suggest_int('step_size', 10, 50)
            hyperparams['gamma'] = trial.suggest_float('gamma', 0.1, 0.9)
        
        # Loss function hyperparameters
        hyperparams['loss_alpha'] = trial.suggest_float('loss_alpha', 0.1, 0.9)
        
        # Data augmentation hyperparameters
        hyperparams['augmentation_strength'] = trial.suggest_float(
            'augmentation_strength', 
            0.0, 1.0
        )
        
        return hyperparams
    
    def _create_config_with_hyperparams(self, hyperparams: Dict[str, Any]) -> PipelineConfig:
        """Create configuration with suggested hyperparameters."""
        # Create a copy of the base config
        config_dict = self.config.to_dict()
        
        # Update with hyperparameters
        config_dict['model']['hidden_channels'] = hyperparams['hidden_channels']
        config_dict['model']['depth'] = hyperparams['depth']
        config_dict['model']['type'] = hyperparams['model_type']
        
        if 'num_heads' in hyperparams:
            config_dict['model']['num_heads'] = hyperparams['num_heads']
        
        config_dict['training']['learning_rate'] = hyperparams['learning_rate']
        config_dict['training']['optimizer'] = hyperparams['optimizer']
        config_dict['training']['weight_decay'] = hyperparams['weight_decay']
        config_dict['training']['scheduler'] = hyperparams['scheduler']
        
        if hyperparams['scheduler'] == 'step':
            config_dict['training']['step_size'] = hyperparams['step_size']
            config_dict['training']['gamma'] = hyperparams['gamma']
        
        config_dict['data']['batch_size'] = hyperparams['batch_size']
        config_dict['data']['augmentation_strength'] = hyperparams['augmentation_strength']
        
        # Create new config
        return PipelineConfig(**config_dict)
    
    def _train_and_evaluate(self, config: PipelineConfig, trial: optuna.Trial) -> float:
        """Train model and evaluate performance."""
        try:
            # Create pipeline with optimized config
            pipeline = EnhancedConnectomicsPipeline(config=config)
            
            # Setup components
            if not pipeline.setup_data_loader():
                raise RuntimeError("Failed to setup data loader")
            
            if not pipeline.setup_model():
                raise RuntimeError("Failed to setup model")
            
            # Train for a limited number of epochs for optimization
            max_epochs = min(10, config.training.epochs)  # Limit epochs for optimization
            
            # Create trainer with early stopping
            trainer = pipeline.trainer
            trainer.config.training.epochs = max_epochs
            trainer.config.training.early_stopping_patience = 3
            
            # Train model
            train_losses, val_losses = [], []
            
            for epoch in range(max_epochs):
                # Train one epoch
                train_loss = trainer.train_epoch(pipeline.train_loader)
                val_loss = trainer.validate_epoch(pipeline.val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Report intermediate value
                trial.report(val_loss, step=epoch)
                
                # Early stopping check
                if trainer.early_stopping(val_loss):
                    break
            
            # Return best validation loss
            best_val_loss = min(val_losses)
            
            # Log trial results
            logger.info(f"Trial {trial.number}: Best val_loss = {best_val_loss:.6f}")
            
            return best_val_loss
            
        except Exception as e:
            logger.error(f"Training failed in trial: {e}")
            return float('inf')
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Best hyperparameters found
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Optimization completed. Best validation loss: {best_value:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save results
        self._save_optimization_results(best_params, best_value)
        
        return best_params
    
    def _save_optimization_results(self, best_params: Dict[str, Any], best_value: float):
        """Save optimization results to file."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'study_name': self.study_name,
            'best_value': best_value,
            'best_params': best_params,
            'n_trials': self.n_trials,
            'timeout': self.timeout
        }
        
        # Save to JSON file
        output_path = Path(f"optimization_results_{self.study_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to: {output_path}")
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history and statistics."""
        trials = self.study.trials
        
        # Extract trial data
        trial_data = []
        for trial in trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'duration': trial.duration.total_seconds()
                })
        
        # Calculate statistics
        values = [t['value'] for t in trial_data]
        
        stats = {
            'n_trials': len(trials),
            'n_completed': len(trial_data),
            'best_value': min(values) if values else None,
            'worst_value': max(values) if values else None,
            'mean_value': np.mean(values) if values else None,
            'std_value': np.std(values) if values else None,
            'optimization_history': trial_data
        }
        
        return stats
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
            ax1.set_title('Optimization History')
            
            # Plot parameter importance
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title('Parameter Importance')
            
            # Plot parallel coordinate
            optuna.visualization.matplotlib.plot_parallel_coordinate(self.study, ax=ax3)
            ax3.set_title('Parallel Coordinate')
            
            # Plot slice
            optuna.visualization.matplotlib.plot_slice(self.study, ax=ax4)
            ax4.set_title('Slice Plot')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plots saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available. Skipping plot generation.")


def run_hyperparameter_optimization(config_path: Optional[str] = None,
                                  n_trials: int = 50,
                                  timeout: Optional[int] = 3600,
                                  study_name: str = "connectomics_optimization") -> Dict[str, Any]:
    """
    Run hyperparameter optimization for the connectomics pipeline.
    
    Args:
        config_path: Path to configuration file
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        study_name: Name for the Optuna study
        
    Returns:
        Best hyperparameters found
    """
    # Load configuration
    config = load_config(config_path, "development")
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        config=config,
        n_trials=n_trials,
        timeout=timeout,
        study_name=study_name
    )
    
    # Run optimization
    best_params = optimizer.optimize()
    
    # Get optimization statistics
    stats = optimizer.get_optimization_history()
    logger.info(f"Optimization statistics: {stats}")
    
    # Plot results
    optimizer.plot_optimization_history(f"optimization_plots_{study_name}.png")
    
    return best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Connectomics Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--study-name", type=str, default="connectomics_optimization", 
                       help="Name for the Optuna study")
    
    args = parser.parse_args()
    
    # Run optimization
    best_params = run_hyperparameter_optimization(
        config_path=args.config,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name
    )
    
    print(f"Best hyperparameters found: {best_params}") 