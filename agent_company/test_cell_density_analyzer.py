#!/usr/bin/env python3
"""
Test script for H01 Cell Density Analyzer
=========================================
Simple test to verify the analyzer works correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from h01_cell_density_analyzer import CellDensityAnalyzer, CellDensityConfig

def test_basic_functionality():
    """Test basic functionality of the analyzer."""
    print("Testing basic functionality...")
    
    # Create sample data
    np.random.seed(42)
    n_cells = 100
    
    cell_data = pd.DataFrame({
        'id': range(1, n_cells + 1),
        'valid': np.ones(n_cells),
        'x': np.random.uniform(400, 1200, n_cells),
        'y': np.random.uniform(2400, 2800, n_cells),
        'z': np.random.uniform(0, 1000, n_cells),
        'volume': np.random.lognormal(mean=3.5, sigma=0.5, size=n_cells),
        'type': np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n_cells),
        'classification': np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n_cells)
    })
    
    # Save sample data
    os.makedirs("./test_data", exist_ok=True)
    cell_csv_path = "./test_data/test_cells.csv"
    cell_data.to_csv(cell_csv_path, index=False)
    
    # Create analyzer
    config = CellDensityConfig(
        output_dir="./test_output",
        save_plots=False,  # Disable plots for faster testing
        save_results=True
    )
    
    analyzer = CellDensityAnalyzer(config)
    
    # Test data loading
    print("Testing data loading...")
    loaded_data = analyzer.load_cell_matrix(cell_csv_path)
    assert len(loaded_data) == n_cells, f"Expected {n_cells} cells, got {len(loaded_data)}"
    print("✓ Data loading works")
    
    # Test layer boundaries
    print("Testing layer boundaries...")
    boundaries = analyzer.load_layer_boundaries()
    assert len(boundaries) > 0, "No layer boundaries loaded"
    # Store the boundaries in the analyzer
    analyzer.layer_boundaries = boundaries
    print("✓ Layer boundaries loaded")
    
    # Test layer assignment
    print("Testing layer assignment...")
    analyzer.cell_data = loaded_data
    layer_assignments = analyzer.assign_cells_to_layers()
    assert len(layer_assignments) == n_cells, "Layer assignment failed"
    print("✓ Layer assignment works")
    
    # Test density calculation
    print("Testing density calculation...")
    results = analyzer.calculate_cell_densities()
    assert 'densities' in results, "Density calculation failed"
    print("✓ Density calculation works")
    
    print("✓ All basic functionality tests passed!")

def test_error_handling():
    """Test error handling capabilities."""
    print("\nTesting error handling...")
    
    analyzer = CellDensityAnalyzer()
    
    # Test loading non-existent files
    print("Testing missing file handling...")
    missing_data = analyzer.load_cell_matrix("nonexistent_file.mat")
    assert len(missing_data) == 0, "Should return empty DataFrame for missing file"
    print("✓ Missing file handling works")
    
    # Test with empty data
    print("Testing empty data handling...")
    analyzer.cell_data = pd.DataFrame()
    results = analyzer.calculate_cell_densities()
    assert not results, "Should return empty results for empty data"
    print("✓ Empty data handling works")
    
    print("✓ All error handling tests passed!")

def main():
    """Run all tests."""
    print("H01 Cell Density Analyzer - Test Suite")
    print("="*40)
    
    try:
        test_basic_functionality()
        test_error_handling()
        
        print("\n" + "="*40)
        print("✓ All tests passed!")
        print("="*40)
        print("\nThe CellDensityAnalyzer is working correctly and ready for use.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 