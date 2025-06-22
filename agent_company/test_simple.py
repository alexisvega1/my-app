#!/usr/bin/env python3
"""
Simple test script for the enhanced pipeline components.
Tests core functionality without external dependencies.
"""

import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_config():
    """Test configuration management."""
    print("🧪 Testing Configuration...")
    
    try:
        from config import PipelineConfig
        
        # Test default configuration
        config = PipelineConfig()
        assert config.environment == "development"
        assert config.data.batch_size == 2
        assert config.model.input_channels == 1
        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_model():
    """Test neural network model."""
    print("🧪 Testing Model...")
    
    try:
        from ffn_v2_mathematical_model import MathematicalFFNv2
        
        # Create model
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=3,
            hidden_channels=16,
            depth=3
        )
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        batch_size = 2
        input_shape = (batch_size, 1, 32, 32, 32)
        test_input = torch.randn(input_shape, device=device)
        
        with torch.no_grad():
            output = model(test_input)
        
        assert output.shape == (batch_size, 3, 32, 32, 32)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        
        print("✅ Model test passed")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_loss_function():
    """Test loss function."""
    print("🧪 Testing Loss Function...")
    
    try:
        from training import DiceBCELoss
        
        criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        
        # Create test data
        batch_size = 2
        shape = (batch_size, 3, 16, 16, 16)
        predictions = torch.sigmoid(torch.randn(shape))
        targets = torch.randint(0, 2, shape, dtype=torch.float32)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        assert torch.isfinite(loss)
        assert loss.item() > 0
        
        print("✅ Loss function test passed")
        return True
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        return False

def test_trainer():
    """Test trainer creation."""
    print("🧪 Testing Trainer...")
    
    try:
        from config import PipelineConfig
        from training import create_trainer
        from ffn_v2_mathematical_model import MathematicalFFNv2
        
        # Create configuration
        config = PipelineConfig()
        config.training.epochs = 2
        config.data.batch_size = 1
        
        # Create model
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=3,
            hidden_channels=16,
            depth=3
        )
        
        # Create trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = create_trainer(model, config, device)
        
        assert trainer is not None
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'criterion')
        
        print("✅ Trainer test passed")
        return True
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        return False

def test_pipeline():
    """Test pipeline creation."""
    print("🧪 Testing Pipeline...")
    
    try:
        from enhanced_pipeline import EnhancedConnectomicsPipeline
        
        # Create pipeline
        pipeline = EnhancedConnectomicsPipeline(environment="development")
        
        assert pipeline is not None
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'device')
        assert pipeline.config.environment == "development"
        
        print("✅ Pipeline test passed")
        return True
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_config_serialization():
    """Test configuration serialization."""
    print("🧪 Testing Configuration Serialization...")
    
    try:
        from config import PipelineConfig
        
        # Create configuration
        config = PipelineConfig()
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'data' in config_dict
        assert 'model' in config_dict
        
        # Test from_dict
        new_config = PipelineConfig.from_dict(config_dict)
        assert new_config.environment == config.environment
        assert new_config.data.batch_size == config.data.batch_size
        
        print("✅ Configuration serialization test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration serialization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Pipeline Tests")
    print("=" * 50)
    
    tests = [
        test_config,
        test_model,
        test_loss_function,
        test_trainer,
        test_pipeline,
        test_config_serialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 