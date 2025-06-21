#!/usr/bin/env python3
"""
Demo script for H01 Comprehensive Analysis Pipeline
==================================================
Demonstrates the comprehensive H01 analysis capabilities with example data.
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

from h01_comprehensive_analysis import H01ComprehensiveAnalyzer, H01AnalysisConfig

def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data for comprehensive analysis...")
    
    # Create sample directory
    os.makedirs("./sample_data", exist_ok=True)
    
    # Create sample cell data
    np.random.seed(42)
    n_cells = 500
    
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
    
    cell_csv_path = "./sample_data/sample_cells.csv"
    cell_data.to_csv(cell_csv_path, index=False)
    print(f"✓ Created cell data: {cell_csv_path}")
    
    # Create sample synapse data
    n_synapses = 200
    synapse_data = {
        'synapse_pairs': [
            {
                'id': i,
                'pre_segment': np.random.randint(1, 100),
                'post_segment': np.random.randint(1, 100),
                'distance': np.random.uniform(0, 1000),
                'features': {
                    'area': np.random.uniform(0.1, 2.0),
                    'intensity': np.random.uniform(0.5, 1.0),
                    'shape_score': np.random.uniform(0.3, 0.9)
                }
            }
            for i in range(1, n_synapses + 1)
        ]
    }
    
    synapse_json_path = "./sample_data/sample_synapses.json"
    import json
    with open(synapse_json_path, 'w') as f:
        json.dump(synapse_data, f, indent=2)
    print(f"✓ Created synapse data: {synapse_json_path}")
    
    # Create sample skeleton data
    n_skeletons = 50
    skeleton_data = {
        'skeletons': [
            {
                'id': i,
                'segment_id': np.random.randint(1, 100),
                'length': np.random.uniform(100, 1000),
                'branch_points': np.random.randint(1, 10),
                'features': {
                    'straightness': np.random.uniform(0.5, 1.0),
                    'tortuosity': np.random.uniform(1.0, 2.0),
                    'density': np.random.uniform(0.1, 1.0)
                }
            }
            for i in range(1, n_skeletons + 1)
        ]
    }
    
    skeleton_json_path = "./sample_data/sample_skeletons.json"
    with open(skeleton_json_path, 'w') as f:
        json.dump(skeleton_data, f, indent=2)
    print(f"✓ Created skeleton data: {skeleton_json_path}")
    
    # Create sample connection data
    n_connections = 100
    connection_data = {
        'connections': [
            {
                'id': i,
                'pre_cell': np.random.randint(1, 100),
                'post_cell': np.random.randint(1, 100),
                'strength': np.random.uniform(0.1, 1.0),
                'type': np.random.choice(['excitatory', 'inhibitory']),
                'synapse_count': np.random.randint(1, 10)
            }
            for i in range(1, n_connections + 1)
        ]
    }
    
    connection_json_path = "./sample_data/sample_connections.json"
    with open(connection_json_path, 'w') as f:
        json.dump(connection_data, f, indent=2)
    print(f"✓ Created connection data: {connection_json_path}")
    
    return {
        'cell_matrix_path': cell_csv_path,
        'synapse_data_path': synapse_json_path,
        'skeleton_data_path': skeleton_json_path,
        'connection_data_path': connection_json_path
    }

def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis capabilities."""
    print("\n" + "="*70)
    print("DEMO: Comprehensive H01 Analysis Pipeline")
    print("="*70)
    
    # Create sample data
    data_paths = create_sample_data()
    
    # Configure comprehensive analyzer
    config = H01AnalysisConfig(
        cell_matrix_path=data_paths['cell_matrix_path'],
        synapse_data_path=data_paths['synapse_data_path'],
        skeleton_data_path=data_paths['skeleton_data_path'],
        connection_data_path=data_paths['connection_data_path'],
        output_dir="./demo_comprehensive_output",
        save_plots=True,
        save_results=True,
        generate_report=True,
        # Enable all analyses
        run_cell_density=True,
        run_synapse_merge=True,
        run_skeleton_pruning=True,
        run_connection_analysis=True,
        run_statistical_analysis=True
    )
    
    # Create and run analyzer
    print("\nInitializing comprehensive analyzer...")
    analyzer = H01ComprehensiveAnalyzer(config)
    
    print("\nRunning comprehensive analysis...")
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print("✓ Comprehensive analysis completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        
        # Display key results
        print(f"\nAnalyses completed: {len(results)}")
        for analysis_name, analysis_results in results.items():
            if analysis_name != 'saved_files':
                print(f"  - {analysis_name}: {'✓' if analysis_results else '✗'}")
        
        # Display statistics
        if hasattr(analyzer, 'statistics') and analyzer.statistics:
            summary = analyzer.statistics.get('summary', {})
            print(f"\nSummary Statistics:")
            print(f"  - Data Quality Score: {summary.get('data_quality_score', 0):.2f}")
            print(f"  - Overall Confidence: {summary.get('overall_confidence', 0):.2f}")
            print(f"  - Total Analyses: {summary.get('total_analyses_run', 0)}")
        
        # List generated files
        if 'saved_files' in results:
            print(f"\nGenerated files:")
            for file_type, file_path in results['saved_files'].items():
                print(f"  - {file_type}: {file_path}")
    else:
        print("✗ Comprehensive analysis failed!")

def demo_selective_analysis():
    """Demonstrate selective analysis capabilities."""
    print("\n" + "="*70)
    print("DEMO: Selective Analysis (Cell Density Only)")
    print("="*70)
    
    # Create sample data
    data_paths = create_sample_data()
    
    # Configure for selective analysis
    config = H01AnalysisConfig(
        cell_matrix_path=data_paths['cell_matrix_path'],
        output_dir="./demo_selective_output",
        save_plots=True,
        save_results=True,
        # Disable other analyses
        run_cell_density=True,
        run_synapse_merge=False,
        run_skeleton_pruning=False,
        run_connection_analysis=False,
        run_statistical_analysis=True
    )
    
    # Create and run analyzer
    print("\nRunning selective analysis (cell density only)...")
    analyzer = H01ComprehensiveAnalyzer(config)
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print("✓ Selective analysis completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        
        # Show what was analyzed
        print(f"\nAnalyses run: {list(results.keys())}")
    else:
        print("✗ Selective analysis failed!")

def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n" + "="*70)
    print("DEMO: Error Handling and Robustness")
    print("="*70)
    
    # Configure with missing data
    config = H01AnalysisConfig(
        output_dir="./demo_error_handling_output",
        save_plots=False,
        save_results=True,
        # All analyses enabled but no data provided
        run_cell_density=True,
        run_synapse_merge=True,
        run_skeleton_pruning=True,
        run_connection_analysis=True,
        run_statistical_analysis=True
    )
    
    # Create and run analyzer
    print("\nRunning analysis with missing data (testing error handling)...")
    analyzer = H01ComprehensiveAnalyzer(config)
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print("✓ Analysis completed gracefully despite missing data!")
        print(f"Results saved to: {config.output_dir}")
        
        # Show what succeeded/failed
        print(f"\nAnalyses attempted: {len(results)}")
        for analysis_name, analysis_results in results.items():
            if analysis_name != 'saved_files':
                status = "✓" if analysis_results else "✗ (no data)"
                print(f"  - {analysis_name}: {status}")
    else:
        print("✗ Analysis failed completely!")

def main():
    """Run all demos."""
    print("H01 Comprehensive Analysis Pipeline - Demo Suite")
    print("="*60)
    print("This demo showcases the comprehensive H01 analysis capabilities")
    print("with robust error handling and multiple analysis types.\n")
    
    try:
        # Run all demos
        demo_comprehensive_analysis()
        demo_selective_analysis()
        demo_error_handling()
        
        print("\n" + "="*70)
        print("✓ All demos completed successfully!")
        print("="*70)
        print("\nKey Features Demonstrated:")
        print("  ✓ Comprehensive analysis pipeline")
        print("  ✓ Selective analysis capabilities")
        print("  ✓ Robust error handling")
        print("  ✓ Multiple data format support")
        print("  ✓ Statistical analysis and visualization")
        print("  ✓ Production-ready logging and reporting")
        
        print("\nNext Steps:")
        print("  1. Replace sample data with real H01 data files")
        print("  2. Adjust configuration parameters as needed")
        print("  3. Customize analysis components for your specific needs")
        print("  4. Integrate with your existing analysis workflow")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 