"""
Comprehensive test suite for the enhanced connectomics pipeline.
Tests all major components with proper error handling and validation.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import logging
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import components to test
from config import PipelineConfig, load_config
from data_loader import H01DataLoader, H01Dataset, create_data_loader
from training import AdvancedTrainer, DiceBCELoss, create_trainer
from ffn_v2_mathematical_model import MathematicalFFNv2
from enhanced_pipeline import EnhancedConnectomicsPipeline

# Suppress warnings during testing
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = PipelineConfig()
        
        # Test basic properties
        self.assertEqual(config.environment, "development")
        self.assertEqual(config.seed, 42)
        self.assertFalse(config.deterministic)
        
        # Test nested configurations
        self.assertEqual(config.data.batch_size, 2)
        self.assertEqual(config.model.input_channels, 1)
        self.assertEqual(config.training.epochs, 100)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = PipelineConfig()
        self.assertTrue(hasattr(config, 'data'))
        self.assertTrue(hasattr(config, 'model'))
        
        # Test invalid configuration (should raise assertion error)
        with self.assertRaises(AssertionError):
            config = PipelineConfig()
            config.data.mip = -1  # Invalid MIP level
            config._validate()
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = PipelineConfig()
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('data', config_dict)
        self.assertIn('model', config_dict)
        
        # Test from_dict
        new_config = PipelineConfig.from_dict(config_dict)
        self.assertEqual(new_config.environment, config.environment)
        self.assertEqual(new_config.data.batch_size, config.data.batch_size)
    
    def test_config_save_load(self):
        """Test configuration save and load from file."""
        config = PipelineConfig()
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Save configuration
        config.save(config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Load configuration
        loaded_config = PipelineConfig.from_yaml(config_path)
        self.assertEqual(loaded_config.environment, config.environment)
        self.assertEqual(loaded_config.data.batch_size, config.data.batch_size)


class TestModel(unittest.TestCase):
    """Test neural network model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_model_creation(self):
        """Test model creation and basic properties."""
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=3,
            hidden_channels=32,
            depth=4
        )
        
        # Test model properties
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(model.encoder_layers[0].conv_block[0].in_channels, 1)
        self.assertEqual(model.final_conv.out_channels, 3)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=3,
            hidden_channels=16,  # Smaller for faster testing
            depth=3
        )
        model.to(self.device)
        
        # Create test input
        batch_size = 2
        input_shape = (batch_size, 1, 32, 32, 32)  # Smaller for faster testing
        test_input = torch.randn(input_shape, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            output = model(test_input)
        
        # Test output properties
        self.assertEqual(output.shape, (batch_size, 3, 32, 32, 32))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))  # Sigmoid output
    
    def test_model_parameters(self):
        """Test model parameter counting."""
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=3,
            hidden_channels=32,
            depth=4
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All parameters should be trainable
    
    def test_model_gradient_flow(self):
        """Test model gradient flow."""
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=3,
            hidden_channels=16,
            depth=3
        )
        model.to(self.device)
        
        # Create test input and target
        batch_size = 1
        input_shape = (batch_size, 1, 16, 16, 16)
        test_input = torch.randn(input_shape, device=self.device)
        test_target = torch.randint(0, 2, (batch_size, 3, 16, 16, 16), 
                                  dtype=torch.float32, device=self.device)
        
        # Forward and backward pass
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.functional.binary_cross_entropy
        
        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, test_target)
        loss.backward()
        optimizer.step()
        
        # Check gradients
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        self.assertGreater(grad_norm, 0, "Gradients should flow through the model")


class TestLossFunction(unittest.TestCase):
    """Test loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_dice_bce_loss(self):
        """Test DiceBCELoss function."""
        criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        
        # Create test data
        batch_size = 2
        shape = (batch_size, 3, 16, 16, 16)
        predictions = torch.sigmoid(torch.randn(shape, device=self.device))
        targets = torch.randint(0, 2, shape, dtype=torch.float32, device=self.device)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Test loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
    
    def test_loss_gradient(self):
        """Test loss gradient computation."""
        criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        
        # Create test data
        batch_size = 1
        shape = (batch_size, 3, 8, 8, 8)
        predictions = torch.sigmoid(torch.randn(shape, device=self.device, requires_grad=True))
        targets = torch.randint(0, 2, shape, dtype=torch.float32, device=self.device)
        
        # Compute loss and gradients
        loss = criterion(predictions, targets)
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(predictions.grad)
        self.assertTrue(torch.isfinite(predictions.grad).all())


class TestDataLoader(unittest.TestCase):
    """Test data loader components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock configuration
        self.config = PipelineConfig()
        self.config.data.volume_path = "graphene://https://h01-materialization.cr.neuroglancer.org/1.0/h01_c3_flat"
        self.config.data.mip = 0
        self.config.data.cache_path = self.temp_dir
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_loader_creation(self):
        """Test data loader creation (mock test)."""
        # This test will fail if CloudVolume is not available, which is expected
        # in a test environment without proper setup
        try:
            data_loader = H01DataLoader(self.config)
            self.assertIsInstance(data_loader, H01DataLoader)
        except Exception as e:
            # Expected in test environment
            self.assertIn("CloudVolume", str(e) or "ImportError", str(e))
    
    def test_dataset_creation(self):
        """Test dataset creation with mock data loader."""
        # Create a mock data loader for testing
        class MockDataLoader:
            def get_random_valid_coords(self, chunk_size):
                return (0, 0, 0)
            
            def load_chunk(self, coords, size):
                return np.random.rand(*size)
        
        mock_loader = MockDataLoader()
        
        # Create dataset
        dataset = H01Dataset(mock_loader, self.config, samples_per_epoch=10)
        
        # Test dataset properties
        self.assertEqual(len(dataset), 10)
        
        # Test getting an item
        data, target = dataset[0]
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        self.assertEqual(data.shape[0], 1)  # Channel dimension
        self.assertEqual(target.shape[0], 3)  # 3 output channels


class TestTraining(unittest.TestCase):
    """Test training components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp()
        
        # Create configuration
        self.config = PipelineConfig()
        self.config.monitoring.checkpoint_dir = self.temp_dir
        self.config.training.epochs = 2  # Short training for testing
        self.config.data.batch_size = 1
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=3,
            hidden_channels=16,
            depth=3
        )
        
        trainer = create_trainer(model, self.config, self.device)
        
        # Test trainer properties
        self.assertIsInstance(trainer, AdvancedTrainer)
        self.assertIsInstance(trainer.optimizer, torch.optim.Optimizer)
        self.assertIsInstance(trainer.criterion, DiceBCELoss)
    
    def test_trainer_checkpointing(self):
        """Test trainer checkpointing."""
        model = MathematicalFFNv2(
            input_channels=1,
            output_channels=3,
            hidden_channels=16,
            depth=3
        )
        
        trainer = create_trainer(model, self.config, self.device)
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path)
        
        # Verify checkpoint exists
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Load checkpoint
        new_trainer = create_trainer(model, self.config, self.device)
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Verify state was loaded
        self.assertEqual(new_trainer.current_epoch, trainer.current_epoch)
        self.assertEqual(new_trainer.best_loss, trainer.best_loss)


class TestEnhancedPipeline(unittest.TestCase):
    """Test the enhanced pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = EnhancedConnectomicsPipeline(environment="development")
        
        # Test basic properties
        self.assertIsInstance(pipeline.config, PipelineConfig)
        self.assertIsInstance(pipeline.device, torch.device)
        self.assertEqual(pipeline.config.environment, "development")
    
    def test_pipeline_setup(self):
        """Test pipeline setup (mock test)."""
        pipeline = EnhancedConnectomicsPipeline(environment="development")
        
        # Test model setup
        success = pipeline.setup_model()
        self.assertTrue(success)
        self.assertIsNotNone(pipeline.model)
        
        # Test trainer setup
        success = pipeline.setup_trainer()
        self.assertTrue(success)
        self.assertIsNotNone(pipeline.trainer)
    
    def test_pipeline_state_save_load(self):
        """Test pipeline state save and load."""
        pipeline = EnhancedConnectomicsPipeline(environment="development")
        pipeline.setup_model()
        pipeline.setup_trainer()
        
        # Save state
        state_path = os.path.join(self.temp_dir, "pipeline_state.pt")
        pipeline.save_pipeline_state(state_path)
        
        # Verify state file exists
        self.assertTrue(os.path.exists(state_path))
        
        # Load state
        new_pipeline = EnhancedConnectomicsPipeline(environment="development")
        new_pipeline.load_pipeline_state(state_path)
        
        # Verify state was loaded
        self.assertEqual(new_pipeline.config.environment, pipeline.config.environment)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestConfiguration,
        TestModel,
        TestLossFunction,
        TestDataLoader,
        TestTraining,
        TestEnhancedPipeline
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 