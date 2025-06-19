"""
Model Registry for Agent Company Framework

Centralized registry for managing different model architectures and plugins.
Follows the principles from the Inception paper for efficient model management.
"""

from typing import Dict, Type, Any, Optional
import inspect


class ModelRegistry:
    """
    Registry for managing model architectures and their configurations.
    
    This allows dynamic model selection and instantiation following
    the efficient resource utilization principles from the Inception paper.
    """
    
    def __init__(self):
        self._models: Dict[str, Type] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        
    def register(self, name: str, model_class: Type, default_config: Optional[Dict[str, Any]] = None):
        """
        Register a model class with optional default configuration.
        
        Args:
            name: Model identifier
            model_class: Model class to register
            default_config: Default parameters for model instantiation
        """
        self._models[name] = model_class
        self._configs[name] = default_config or {}
        
    def get(self, name: str, **kwargs) -> Any:
        """
        Get a model instance by name with optional parameter overrides.
        
        Args:
            name: Registered model name
            **kwargs: Parameters to override default config
            
        Returns:
            Instantiated model
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not registered. Available: {list(self._models.keys())}")
            
        model_class = self._models[name]
        config = self._configs[name].copy()
        config.update(kwargs)
        
        # Filter out parameters not accepted by the model constructor
        sig = inspect.signature(model_class.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        
        return model_class(**valid_params)
        
    def list_models(self) -> Dict[str, str]:
        """
        List all registered models with their descriptions.
        
        Returns:
            Dictionary mapping model names to descriptions
        """
        result = {}
        for name, model_class in self._models.items():
            doc = model_class.__doc__
            if doc:
                # Extract first line of docstring as description
                description = doc.split('\n')[0].strip()
            else:
                description = f"{model_class.__name__} model"
            result[name] = description
        return result
        
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get the default configuration for a model."""
        if name not in self._configs:
            raise ValueError(f"Model '{name}' not registered")
        return self._configs[name].copy()
        
    def update_config(self, name: str, config: Dict[str, Any]):
        """Update the default configuration for a model."""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not registered")
        self._configs[name].update(config)


# Create global registry instance
registry = ModelRegistry()

# Register the FFN-v2 Inception model with efficient default settings
# Following the paper's computational budget constraints
try:
    from .segmenters import FFNv2Plugin
    
    registry.register(
        "ffn_v2_inception",
        FFNv2Plugin,
        {
            "in_channels": 1,
            "base_channels": 32,  # Efficient base channel count
            "num_blocks": 4,      # Balanced depth following paper's principles
            "dropout_rate": 0.4,  # Good regularization
            "use_auxiliary": True  # Enable auxiliary classifiers for better gradients
        }
    )
    
    # Also register a lightweight version for mobile/edge deployment
    registry.register(
        "ffn_v2_inception_lite",
        FFNv2Plugin,
        {
            "in_channels": 1,
            "base_channels": 16,  # Reduced for efficiency
            "num_blocks": 3,      # Shallower network
            "dropout_rate": 0.3,
            "use_auxiliary": False  # Disable auxiliary for speed
        }
    )
    
    # High-capacity version for when computational resources allow
    registry.register(
        "ffn_v2_inception_large", 
        FFNv2Plugin,
        {
            "in_channels": 1,
            "base_channels": 64,  # Higher capacity
            "num_blocks": 6,      # Deeper network
            "dropout_rate": 0.5,
            "use_auxiliary": True
        }
    )
    
except ImportError:
    # Handle case where PyTorch is not available
    pass


def get_model(name: str, **kwargs):
    """
    Convenience function to get a model from the global registry.
    
    Args:
        name: Model name
        **kwargs: Configuration overrides
        
    Returns:
        Model instance
    """
    return registry.get(name, **kwargs)


def list_available_models() -> Dict[str, str]:
    """List all available models in the registry."""
    return registry.list_models()


# Example usage and integration patterns
def create_segmentation_pipeline(model_name: str = "ffn_v2_inception", **model_kwargs):
    """
    Create a complete segmentation pipeline with the specified model.
    
    This demonstrates how to integrate the Inception-based FFN-v2 model
    into a larger system following the paper's efficiency principles.
    """
    model = get_model(model_name, **model_kwargs)
    
    class SegmentationPipeline:
        def __init__(self, model):
            self.model = model
            
        def predict(self, volume, threshold=0.5):
            """Standard prediction."""
            return self.model.segment(volume, threshold)
            
        def predict_with_uncertainty(self, volume, threshold=0.5):
            """Prediction with uncertainty estimation."""
            return self.model.predict_with_uncertainty(volume, threshold)
            
        def get_high_uncertainty_regions(self, volume, uncertainty_threshold=0.7):
            """Identify regions that need human review."""
            _, uncertainty = self.model.predict_with_uncertainty(volume)
            return uncertainty > uncertainty_threshold
    
    return SegmentationPipeline(model)


# Configuration utilities for production deployment
class ModelConfig:
    """Configuration management for different deployment scenarios."""
    
    # Efficient configurations following Inception paper principles
    MOBILE_CONFIG = {
        "base_channels": 16,
        "num_blocks": 3,
        "dropout_rate": 0.2,
        "use_auxiliary": False
    }
    
    STANDARD_CONFIG = {
        "base_channels": 32,
        "num_blocks": 4, 
        "dropout_rate": 0.4,
        "use_auxiliary": True
    }
    
    HIGH_ACCURACY_CONFIG = {
        "base_channels": 64,
        "num_blocks": 6,
        "dropout_rate": 0.5,
        "use_auxiliary": True
    }
    
    @classmethod
    def get_config(cls, deployment_target: str) -> Dict[str, Any]:
        """Get configuration for different deployment targets."""
        configs = {
            "mobile": cls.MOBILE_CONFIG,
            "standard": cls.STANDARD_CONFIG,
            "high_accuracy": cls.HIGH_ACCURACY_CONFIG,
            "edge": cls.MOBILE_CONFIG,  # Same as mobile
            "cloud": cls.HIGH_ACCURACY_CONFIG,  # Same as high accuracy
        }
        
        if deployment_target not in configs:
            raise ValueError(f"Unknown deployment target: {deployment_target}")
            
        return configs[deployment_target].copy()