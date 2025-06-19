#!/usr/bin/env python3
"""
Create Traceable Test Data for Neuron Tracing
============================================
Generates synthetic EM data with traceable neuron structures for testing.
"""

import numpy as np
import os
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def create_synthetic_neurons(shape=(256, 256, 128), num_neurons=5):
    """Create synthetic EM data with traceable neuron structures."""
    
    # Initialize volume
    volume = np.zeros(shape, dtype=np.uint8)
    
    # Create multiple neurons with different morphologies
    for i in range(num_neurons):
        # Random neuron parameters
        center = np.random.randint(50, shape[0]-50, 3)
        length = np.random.randint(80, 150)
        radius = np.random.randint(8, 15)
        num_branches = np.random.randint(0, 4)
        
        # Create main axon/dendrite
        neuron = create_linear_neuron(center, length, radius, shape)
        
        # Add branches
        for _ in range(num_branches):
            branch_start = np.random.randint(20, length-20)
            branch_length = np.random.randint(30, 60)
            branch_radius = max(3, radius - np.random.randint(2, 6))
            
            # Random direction for branch
            branch_direction = np.random.randn(3)
            branch_direction = branch_direction / np.linalg.norm(branch_direction)
            
            # Create branch
            branch_center = center + branch_start * np.array([1, 0, 0])  # Along x-axis
            branch = create_linear_neuron(branch_center, branch_length, branch_radius, shape, 
                                        direction=branch_direction)
            neuron = np.logical_or(neuron, branch)
        
        # Add to volume
        volume[neuron] = np.random.randint(180, 255)
    
    # Add some background structure (membranes, organelles)
    add_background_structure(volume)
    
    # Add noise
    noise = np.random.normal(0, 10, volume.shape).astype(np.uint8)
    volume = np.clip(volume + noise, 0, 255).astype(np.uint8)
    
    return volume

def create_linear_neuron(center, length, radius, shape, direction=None):
    """Create a linear neuron segment."""
    if direction is None:
        direction = np.array([1, 0, 0])  # Default along x-axis
    
    # Create line of points
    t = np.linspace(0, length, length)
    points = []
    
    for ti in t:
        point = center + ti * direction
        if (0 <= point[0] < shape[0] and 
            0 <= point[1] < shape[1] and 
            0 <= point[2] < shape[2]):
            points.append(point.astype(int))
    
    if not points:
        return np.zeros(shape, dtype=bool)
    
    points = np.array(points)
    
    # Create mask for the neuron
    mask = np.zeros(shape, dtype=bool)
    
    # For each point, create a sphere
    for point in points:
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt((z - point[0])**2 + (y - point[1])**2 + (x - point[2])**2)
        sphere = dist <= radius
        mask = np.logical_or(mask, sphere)
    
    return mask

def add_background_structure(volume):
    """Add background cellular structures."""
    shape = volume.shape
    
    # Add some membrane-like structures
    for _ in range(10):
        center = np.random.randint(0, min(shape), 3)
        size = np.random.randint(20, 40)
        
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        
        # Create membrane-like structure
        membrane = np.abs(dist - size/2) < 2
        volume[membrane] = np.random.randint(100, 150)
    
    # Add some organelles
    for _ in range(20):
        center = np.random.randint(0, min(shape), 3)
        size = np.random.randint(5, 15)
        
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        
        organelle = dist <= size
        volume[organelle] = np.random.randint(80, 120)

def create_test_regions():
    """Create multiple test regions with different characteristics."""
    
    regions = {
        'simple_neurons': create_synthetic_neurons((128, 128, 64), num_neurons=3),
        'complex_neurons': create_synthetic_neurons((256, 256, 128), num_neurons=8),
        'dense_network': create_synthetic_neurons((192, 192, 96), num_neurons=12),
        'sparse_neurons': create_synthetic_neurons((160, 160, 80), num_neurons=2),
    }
    
    return regions

def save_and_visualize_regions(regions):
    """Save regions and create visualizations."""
    
    for name, volume in regions.items():
        # Save as numpy file
        filename = f'test_{name}.npy'
        np.save(filename, volume)
        print(f"✓ Saved {filename} with shape: {volume.shape}")
        print(f"  Data range: {volume.min()} to {volume.max()}")
        print(f"  Non-zero voxels: {np.count_nonzero(volume)} / {volume.size}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Middle slices
        mid_z = volume.shape[0] // 2
        mid_y = volume.shape[1] // 2
        mid_x = volume.shape[2] // 2
        
        axes[0].imshow(volume[mid_z, :, :], cmap='gray')
        axes[0].set_title(f'{name} - XY slice')
        axes[0].axis('off')
        
        axes[1].imshow(volume[:, mid_y, :], cmap='gray')
        axes[1].set_title(f'{name} - XZ slice')
        axes[1].axis('off')
        
        axes[2].imshow(volume[:, :, mid_x], cmap='gray')
        axes[2].set_title(f'{name} - YZ slice')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'test_{name}_preview.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Created visualization: test_{name}_preview.png")

if __name__ == "__main__":
    print("Creating traceable test data for neuron tracing...")
    
    # Create test regions
    regions = create_test_regions()
    
    # Save and visualize
    save_and_visualize_regions(regions)
    
    print("\n✓ Test data creation complete!")
    print("Generated files:")
    for name in regions.keys():
        print(f"  - test_{name}.npy")
        print(f"  - test_{name}_preview.png")
    
    print("\nThese files can now be used for testing the neuron tracing pipeline.") 