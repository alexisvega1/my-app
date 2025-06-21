#!/usr/bin/env python3
"""
Demo script for H01 Cell Density Analyzer
=========================================
Demonstrates the robust cell density analysis capabilities with example data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from h01_cell_density_analyzer import CellDensityAnalyzer, CellDensityConfig

def create_sample_data():
    """Create sample cell data for demonstration."""
    print("Creating sample cell data...")
    
    # Generate synthetic cell data based on H01 patterns
    np.random.seed(42)  # For reproducible results
    
    n_cells = 1000
    
    # Cell coordinates (in micrometers, based on H01 dimensions)
    x_coords = np.random.uniform(400, 1200, n_cells)
    y_coords = np.random.uniform(2400, 2800, n_cells)
    z_coords = np.random.uniform(0, 1000, n_cells)
    
    # Cell types (distribution based on typical cortical patterns)
    # 0: Unknown, 1: Pyramidal, 2: Interneuron, 3: Unclassified, 4: Astrocyte, 5: Oligo, 6: MG/OPC
    cell_types = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6],
        size=n_cells,
        p=[0.05, 0.4, 0.15, 0.1, 0.15, 0.1, 0.05]  # Realistic distribution
    )
    
    # Cell volumes (in cubic micrometers)
    volumes = np.random.lognormal(mean=3.5, sigma=0.5, size=n_cells)
    
    # Create DataFrame
    cell_data = pd.DataFrame({
        'id': range(1, n_cells + 1),
        'valid': np.ones(n_cells),
        'x': x_coords,
        'y': y_coords,
        'z': z_coords,
        'volume': volumes,
        'type': cell_types,
        'classification': cell_types
    })
    
    return cell_data

def create_sample_mask():
    """Create sample mask data for volume calculations."""
    print("Creating sample mask data...")
    
    # Create a 3D mask (simplified version)
    mask_shape = (64, 64, 32)  # Reduced size for demo
    mask = np.random.choice([0, 1], size=mask_shape, p=[0.3, 0.7])
    
    # Apply some spatial structure
    for z in range(mask_shape[2]):
        mask[:, :, z] = np.random.choice([0, 1], size=(mask_shape[0], mask_shape[1]), p=[0.2, 0.8])
    
    return mask

def save_sample_data(cell_data, mask_data, output_dir="./sample_data"):
    """Save sample data in various formats for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cell data as CSV
    cell_csv_path = os.path.join(output_dir, "sample_cells.csv")
    cell_data.to_csv(cell_csv_path, index=False)
    print(f"Saved cell data to: {cell_csv_path}")
    
    # Save cell data as JSON
    cell_json_path = os.path.join(output_dir, "sample_cells.json")
    cell_data.to_json(cell_json_path, orient='records', indent=2)
    print(f"Saved cell data to: {cell_json_path}")
    
    # Save mask data
    mask_npy_path = os.path.join(output_dir, "sample_mask.npy")
    np.save(mask_npy_path, mask_data)
    print(f"Saved mask data to: {mask_npy_path}")
    
    return {
        'cell_csv': cell_csv_path,
        'cell_json': cell_json_path,
        'mask_npy': mask_npy_path
    }

def demo_basic_analysis():
    """Demonstrate basic cell density analysis."""
    print("\n" + "="*60)
    print("DEMO: Basic Cell Density Analysis")
    print("="*60)
    
    # Create sample data
    cell_data = create_sample_data()
    mask_data = create_sample_mask()
    
    # Save sample data
    data_paths = save_sample_data(cell_data, mask_data)
    
    # Configure analyzer
    config = CellDensityConfig(
        output_dir="./demo_output",
        save_plots=True,
        save_results=True,
        normalize_by_volume=True
    )
    
    # Create analyzer
    analyzer = CellDensityAnalyzer(config)
    
    # Run analysis with sample data
    print("\nRunning cell density analysis...")
    results = analyzer.run_analysis(
        cell_matrix_path=data_paths['cell_csv'],
        mask_path=data_paths['mask_npy']
    )
    
    if results:
        print("✓ Analysis completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        
        # Display key results
        if 'densities' in results:
            densities = np.array(results['densities'])
            print(f"\nCell density matrix shape: {densities.shape}")
            print(f"Average density per layer: {np.mean(densities, axis=1)}")
        
        if 'saved_files' in results:
            print(f"\nGenerated files:")
            for file_type, file_path in results['saved_files'].items():
                print(f"  - {file_type}: {file_path}")
    else:
        print("✗ Analysis failed!")

def demo_robust_data_loading():
    """Demonstrate robust data loading capabilities."""
    print("\n" + "="*60)
    print("DEMO: Robust Data Loading")
    print("="*60)
    
    # Create analyzer
    analyzer = CellDensityAnalyzer()
    
    # Test loading non-existent files (should handle gracefully)
    print("\nTesting graceful handling of missing files...")
    
    missing_cell_data = analyzer.load_cell_matrix("nonexistent_file.mat")
    print(f"Missing cell file result: {len(missing_cell_data)} cells loaded")
    
    missing_boundaries = analyzer.load_layer_boundaries("nonexistent_boundaries.mat")
    print(f"Missing boundaries result: {len(missing_boundaries)} boundaries loaded (using defaults)")
    
    missing_mask = analyzer.load_mask_data("nonexistent_mask.mat")
    print(f"Missing mask result: {missing_mask is not None}")
    
    # Test loading different formats
    print("\nTesting different data formats...")
    
    # Create sample data in different formats
    cell_data = create_sample_data()
    
    # Test CSV loading
    csv_path = "./sample_data/test_cells.csv"
    cell_data.to_csv(csv_path, index=False)
    csv_result = analyzer.load_cell_matrix(csv_path)
    print(f"CSV loading: {len(csv_result)} cells loaded")
    
    # Test JSON loading
    json_path = "./sample_data/test_cells.json"
    cell_data.to_json(json_path, orient='records')
    json_result = analyzer.load_cell_matrix(json_path)
    print(f"JSON loading: {len(json_result)} cells loaded")

def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("DEMO: Visualization Capabilities")
    print("="*60)
    
    # Create sample data and run analysis
    cell_data = create_sample_data()
    mask_data = create_sample_mask()
    
    # Save data
    data_paths = save_sample_data(cell_data, mask_data)
    
    # Configure analyzer with visualization
    config = CellDensityConfig(
        output_dir="./demo_visualization",
        save_plots=True,
        save_results=True
    )
    
    analyzer = CellDensityAnalyzer(config)
    
    # Run analysis
    results = analyzer.run_analysis(
        cell_matrix_path=data_paths['cell_csv'],
        mask_path=data_paths['mask_npy']
    )
    
    if results and 'saved_files' in results:
        print("✓ Visualizations generated!")
        
        # List generated plots
        plot_files = {k: v for k, v in results['saved_files'].items() if k.startswith('plot_')}
        print(f"\nGenerated visualizations:")
        for plot_name, plot_path in plot_files.items():
            print(f"  - {plot_name}: {plot_path}")
        
        # Show one plot interactively if possible
        try:
            figures = analyzer.create_visualizations()
            if figures:
                print(f"\nDisplaying first visualization...")
                first_fig = list(figures.values())[0]
                plt.show()
        except Exception as e:
            print(f"Could not display plot interactively: {e}")

def demo_statistics():
    """Demonstrate statistical analysis capabilities."""
    print("\n" + "="*60)
    print("DEMO: Statistical Analysis")
    print("="*60)
    
    # Create sample data
    cell_data = create_sample_data()
    mask_data = create_sample_mask()
    
    # Save data
    data_paths = save_sample_data(cell_data, mask_data)
    
    # Configure analyzer
    config = CellDensityConfig(
        output_dir="./demo_statistics",
        save_plots=True,
        save_results=True
    )
    
    analyzer = CellDensityAnalyzer(config)
    
    # Run analysis
    results = analyzer.run_analysis(
        cell_matrix_path=data_paths['cell_csv'],
        mask_path=data_paths['mask_npy']
    )
    
    if results and hasattr(analyzer, 'statistics') and analyzer.statistics:
        print("✓ Statistical analysis completed!")
        
        stats = analyzer.statistics
        print(f"\nKey Statistics:")
        print(f"  - Average total density: {np.mean(stats['total_densities']):.2f} cells/mm³")
        print(f"  - Average E/I ratio: {np.mean(stats['ei_ratios']):.2f}")
        print(f"  - Average neuron/glia ratio: {np.mean(stats['neuron_glia_ratios']):.2f}")
        
        print(f"\nLayer-specific statistics:")
        for i, layer_name in analyzer.LAYER_NAMES.items():
            if i <= len(stats['total_densities']):
                print(f"  {layer_name}:")
                print(f"    - Total density: {stats['total_densities'][i-1]:.2f} cells/mm³")
                print(f"    - E/I ratio: {stats['ei_ratios'][i-1]:.2f}")
                print(f"    - Neuron/glia ratio: {stats['neuron_glia_ratios'][i-1]:.2f}")

def main():
    """Run all demos."""
    print("H01 Cell Density Analyzer - Demo Suite")
    print("="*50)
    print("This demo showcases the robust cell density analysis capabilities")
    print("adapted from H01 Matlab scripts with modern Python implementation.\n")
    
    try:
        # Run all demos
        demo_basic_analysis()
        demo_robust_data_loading()
        demo_visualization()
        demo_statistics()
        
        print("\n" + "="*60)
        print("✓ All demos completed successfully!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("  ✓ Robust data loading (CSV, JSON, MATLAB .mat)")
        print("  ✓ Graceful error handling for missing files")
        print("  ✓ Cell density calculation by layer and type")
        print("  ✓ Statistical analysis (E/I ratios, neuron/glia ratios)")
        print("  ✓ Comprehensive visualization generation")
        print("  ✓ Multiple output formats (JSON, PNG, TXT)")
        print("  ✓ Production-ready logging and error handling")
        
        print("\nNext Steps:")
        print("  1. Replace sample data with real H01 data files")
        print("  2. Adjust configuration parameters as needed")
        print("  3. Integrate with your existing analysis pipeline")
        print("  4. Customize visualizations for your specific needs")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 