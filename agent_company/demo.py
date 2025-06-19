#!/usr/bin/env python3
"""
Demo script for FFN-v2 Inception model.

This script demonstrates the key features of the Inception-based segmentation model:
- Model instantiation and usage
- Uncertainty-aware predictions
- Multi-scale feature extraction
- Efficient computational budget

Usage:
    python demo.py
"""

import sys
import time
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

from tool_registry import get_model, list_available_models, ModelConfig


def demonstrate_model_variants():
    """Demonstrate different model variants and their characteristics."""
    print("🏗️  Available Model Variants:")
    print("=" * 50)
    
    models = list_available_models()
    for name, description in models.items():
        print(f"  {name}: {description}")
    
    print("\n📊 Model Specifications:")
    print("-" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot instantiate models")
        return
    
    # Compare model sizes
    variants = [
        ("ffn_v2_inception_lite", "Mobile/Edge"),
        ("ffn_v2_inception", "Standard"),
        ("ffn_v2_inception_large", "High Accuracy")
    ]
    
    for model_name, use_case in variants:
        try:
            model = get_model(model_name)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  {model_name:25} | {total_params:8,} params | {use_case}")
        except Exception as e:
            print(f"  {model_name:25} | Error: {e}")


def demonstrate_basic_usage():
    """Demonstrate basic model usage."""
    print("\n🎯 Basic Usage Example:")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot run inference demo")
        return
    
    try:
        # Create model
        model = get_model("ffn_v2_inception")
        model.eval()
        
        print("✓ Model created successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create dummy input
        volume = torch.randn(1, 1, 32, 32, 32)  # Small volume for demo
        print(f"✓ Input volume shape: {volume.shape}")
        
        # Basic segmentation
        start_time = time.time()
        with torch.no_grad():
            segmentation = model.segment(volume, threshold=0.5)
        inference_time = time.time() - start_time
        
        print(f"✓ Segmentation completed in {inference_time:.3f}s")
        print(f"  Output shape: {segmentation.shape}")
        print(f"  Segmented voxels: {segmentation.sum().item():.0f}")
        
    except Exception as e:
        print(f"❌ Error in basic usage: {e}")


def demonstrate_uncertainty_prediction():
    """Demonstrate uncertainty-aware prediction."""
    print("\n🔮 Uncertainty-Aware Prediction:")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot run uncertainty demo")
        return
    
    try:
        model = get_model("ffn_v2_inception")
        model.eval()
        
        # Create input with some structure
        volume = torch.randn(1, 1, 32, 32, 32)
        
        # Get predictions with uncertainty
        start_time = time.time()
        with torch.no_grad():
            seg_mask, uncertainty = model.predict_with_uncertainty(volume)
        inference_time = time.time() - start_time
        
        print(f"✓ Prediction with uncertainty completed in {inference_time:.3f}s")
        print(f"  Segmentation shape: {seg_mask.shape}")
        print(f"  Uncertainty shape: {uncertainty.shape}")
        
        # Analyze uncertainty
        uncertainty_stats = {
            "mean": uncertainty.mean().item(),
            "std": uncertainty.std().item(),
            "min": uncertainty.min().item(),
            "max": uncertainty.max().item()
        }
        
        print("📊 Uncertainty Statistics:")
        for stat, value in uncertainty_stats.items():
            print(f"    {stat:4}: {value:.4f}")
        
        # High uncertainty regions
        high_uncertainty_threshold = 0.7
        high_uncertainty_voxels = (uncertainty > high_uncertainty_threshold).sum().item()
        total_voxels = uncertainty.numel()
        high_uncertainty_pct = 100 * high_uncertainty_voxels / total_voxels
        
        print(f"  High uncertainty regions (>{high_uncertainty_threshold}): {high_uncertainty_pct:.1f}%")
        
    except Exception as e:
        print(f"❌ Error in uncertainty prediction: {e}")


def demonstrate_model_efficiency():
    """Demonstrate computational efficiency of different variants."""
    print("\n⚡ Model Efficiency Comparison:")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot run efficiency demo")
        return
    
    variants = ["ffn_v2_inception_lite", "ffn_v2_inception", "ffn_v2_inception_large"]
    volume = torch.randn(1, 1, 32, 32, 32)  # Standard test volume
    
    print(f"Test volume shape: {volume.shape}")
    print(f"{'Model':25} | {'Params':>8} | {'Time (ms)':>10} | {'Memory (MB)':>12}")
    print("-" * 65)
    
    for variant in variants:
        try:
            model = get_model(variant)
            model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Measure inference time
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):  # Average over multiple runs
                    _ = model.segment(volume)
            avg_time = (time.time() - start_time) / 10 * 1000  # Convert to ms
            
            # Rough memory estimate (parameters + activations)
            param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per float32, convert to MB
            
            print(f"{variant:25} | {total_params:8,} | {avg_time:8.1f} ms | {param_memory:8.1f} MB")
            
        except Exception as e:
            print(f"{variant:25} | Error: {str(e)[:30]}...")


def demonstrate_auxiliary_features():
    """Demonstrate auxiliary features and model internals."""
    print("\n🔧 Advanced Features:")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot demonstrate advanced features")
        return
    
    try:
        # Create model with auxiliary classifiers enabled
        model = get_model("ffn_v2_inception", use_auxiliary=True)
        model.train()  # Enable training mode to see auxiliary outputs
        
        volume = torch.randn(1, 1, 32, 32, 32)
        
        # Forward pass with auxiliary outputs
        with torch.no_grad():
            seg_logits, unc_logits, aux_outputs = model(volume)
        
        print("✓ Forward pass with auxiliary outputs:")
        print(f"  Segmentation logits: {seg_logits.shape}")
        print(f"  Uncertainty logits: {unc_logits.shape}")
        
        if aux_outputs:
            print(f"  Auxiliary outputs: {len(aux_outputs)} classifiers")
            for i, aux_out in enumerate(aux_outputs):
                print(f"    Aux {i+1}: {aux_out.shape}")
        else:
            print("  No auxiliary outputs (not enough blocks or disabled)")
        
        # Demonstrate loss computation
        dummy_target = torch.randint(0, 2, seg_logits.shape).float()
        loss = model.compute_loss(seg_logits, unc_logits, dummy_target, aux_outputs)
        print(f"✓ Combined loss computation: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Error in advanced features: {e}")


def demonstrate_deployment_configs():
    """Demonstrate different deployment configurations."""
    print("\n🚀 Deployment Configurations:")
    print("=" * 50)
    
    deployment_targets = ["mobile", "standard", "high_accuracy", "edge", "cloud"]
    
    for target in deployment_targets:
        try:
            config = ModelConfig.get_config(target)
            print(f"  {target:12}: {config}")
        except Exception as e:
            print(f"  {target:12}: Error - {e}")


def main():
    """Main demonstration function."""
    print("🎉 FFN-v2 Inception Model Demonstration")
    print("Based on 'Going deeper with convolutions' by Szegedy et al.")
    print("https://arxiv.org/pdf/1409.4842")
    print("\n")
    
    # Run all demonstrations
    demonstrate_model_variants()
    demonstrate_basic_usage()
    demonstrate_uncertainty_prediction()
    demonstrate_model_efficiency()
    demonstrate_auxiliary_features()
    demonstrate_deployment_configs()
    
    print("\n✨ Demo completed!")
    print("\nNext steps:")
    print("  1. Install PyTorch: pip install torch torchvision")
    print("  2. Train your own model: python train_ffn_v2.py")
    print("  3. Check the README.md for detailed usage")


if __name__ == "__main__":
    main()