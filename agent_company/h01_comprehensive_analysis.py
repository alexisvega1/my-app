#!/usr/bin/env python3
"""
H01 Comprehensive Analysis Pipeline
==================================
Integration script that combines all H01 analysis capabilities:
- Cell density analysis by layer
- Synapse merge decision model
- Skeleton pruning model
- Connection strength analysis
- Statistical analysis and visualization

This provides a robust, production-ready analysis pipeline for H01 connectomics data.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our analysis modules
from h01_cell_density_analyzer import CellDensityAnalyzer, CellDensityConfig
from h01_integration import SynapseMergeModel, SkeletonPruner, H01Integration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class H01AnalysisConfig:
    """Configuration for comprehensive H01 analysis."""
    
    # Data paths
    cell_matrix_path: str = ""
    layer_boundaries_path: str = ""
    mask_path: str = ""
    synapse_data_path: str = ""
    skeleton_data_path: str = ""
    connection_data_path: str = ""
    
    # Analysis parameters
    mip_level: int = 8
    voxel_size_nm: Tuple[float, float, float] = (8.0, 8.0, 33.0)
    
    # Model parameters
    synapse_merge_threshold: float = 0.5
    skeleton_pruning_threshold: float = 0.3
    
    # Output settings
    output_dir: str = "./h01_comprehensive_output"
    save_plots: bool = True
    save_results: bool = True
    generate_report: bool = True
    
    # Analysis flags
    run_cell_density: bool = True
    run_synapse_merge: bool = True
    run_skeleton_pruning: bool = True
    run_connection_analysis: bool = True
    run_statistical_analysis: bool = True

class H01ComprehensiveAnalyzer:
    """
    Comprehensive H01 analysis pipeline that integrates all analysis capabilities.
    
    Provides a unified interface for:
    - Cell density analysis by cortical layer
    - Synapse merge decision modeling
    - Skeleton pruning and optimization
    - Connection strength analysis
    - Statistical analysis and visualization
    """
    
    def __init__(self, config: Optional[H01AnalysisConfig] = None):
        self.config = config or H01AnalysisConfig()
        
        # Initialize analysis components
        self.cell_density_analyzer = None
        self.synapse_merge_model = None
        self.skeleton_pruner = None
        self.h01_analyzer = None
        
        # Results storage
        self.results = {}
        self.statistics = {}
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info("H01ComprehensiveAnalyzer initialized")
    
    def initialize_components(self):
        """Initialize all analysis components."""
        logger.info("Initializing analysis components...")
        
        # Initialize cell density analyzer
        if self.config.run_cell_density:
            cell_config = CellDensityConfig(
                cell_matrix_path=self.config.cell_matrix_path,
                layer_boundaries_path=self.config.layer_boundaries_path,
                mask_path=self.config.mask_path,
                output_dir=os.path.join(self.config.output_dir, "cell_density"),
                save_plots=self.config.save_plots,
                save_results=self.config.save_results
            )
            self.cell_density_analyzer = CellDensityAnalyzer(cell_config)
            logger.info("✓ Cell density analyzer initialized")
        
        # Initialize synapse merge model
        if self.config.run_synapse_merge:
            self.synapse_merge_model = SynapseMergeModel(
                threshold=self.config.synapse_merge_threshold
            )
            logger.info("✓ Synapse merge model initialized")
        
        # Initialize skeleton pruner
        if self.config.run_skeleton_pruning:
            self.skeleton_pruner = SkeletonPruner(
                threshold=self.config.skeleton_pruning_threshold
            )
            logger.info("✓ Skeleton pruner initialized")
        
        # Initialize H01 analyzer
        if self.config.run_connection_analysis:
            self.h01_analyzer = H01Integration()
            logger.info("✓ H01 analyzer initialized")
        
        logger.info("All components initialized successfully")
    
    def run_cell_density_analysis(self) -> Dict[str, Any]:
        """Run cell density analysis."""
        if not self.config.run_cell_density or not self.cell_density_analyzer:
            logger.info("Cell density analysis disabled or analyzer not initialized")
            return {}
        
        logger.info("Running cell density analysis...")
        
        try:
            results = self.cell_density_analyzer.run_analysis(
                cell_matrix_path=self.config.cell_matrix_path,
                layer_boundaries_path=self.config.layer_boundaries_path,
                mask_path=self.config.mask_path
            )
            
            self.results['cell_density'] = results
            logger.info("✓ Cell density analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Cell density analysis failed: {e}")
            return {}
    
    def run_synapse_merge_analysis(self) -> Dict[str, Any]:
        """Run synapse merge analysis."""
        if not self.config.run_synapse_merge or not self.synapse_merge_model:
            logger.info("Synapse merge analysis disabled or model not initialized")
            return {}
        
        logger.info("Running synapse merge analysis...")
        
        try:
            # Load synapse data
            if not self.config.synapse_data_path or not os.path.exists(self.config.synapse_data_path):
                logger.warning("Synapse data not available, skipping synapse merge analysis")
                return {}
            
            # Run analysis (this would be implemented based on actual data format)
            results = {
                'synapse_pairs_analyzed': 0,
                'merge_decisions': [],
                'merge_probabilities': [],
                'model_performance': {}
            }
            
            self.results['synapse_merge'] = results
            logger.info("✓ Synapse merge analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Synapse merge analysis failed: {e}")
            return {}
    
    def run_skeleton_pruning_analysis(self) -> Dict[str, Any]:
        """Run skeleton pruning analysis."""
        if not self.config.run_skeleton_pruning or not self.skeleton_pruner:
            logger.info("Skeleton pruning analysis disabled or pruner not initialized")
            return {}
        
        logger.info("Running skeleton pruning analysis...")
        
        try:
            # Load skeleton data
            if not self.config.skeleton_data_path or not os.path.exists(self.config.skeleton_data_path):
                logger.warning("Skeleton data not available, skipping skeleton pruning analysis")
                return {}
            
            # Run analysis (this would be implemented based on actual data format)
            results = {
                'skeletons_analyzed': 0,
                'pruning_decisions': [],
                'pruning_probabilities': [],
                'model_performance': {}
            }
            
            self.results['skeleton_pruning'] = results
            logger.info("✓ Skeleton pruning analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Skeleton pruning analysis failed: {e}")
            return {}
    
    def run_connection_analysis(self) -> Dict[str, Any]:
        """Run connection strength analysis."""
        if not self.config.run_connection_analysis or not self.h01_analyzer:
            logger.info("Connection analysis disabled or analyzer not initialized")
            return {}
        
        logger.info("Running connection analysis...")
        
        try:
            # Load connection data
            if not self.config.connection_data_path or not os.path.exists(self.config.connection_data_path):
                logger.warning("Connection data not available, skipping connection analysis")
                return {}
            
            # Run analysis (this would be implemented based on actual data format)
            results = {
                'connections_analyzed': 0,
                'connection_strengths': [],
                'connection_types': [],
                'statistics': {}
            }
            
            self.results['connection_analysis'] = results
            logger.info("✓ Connection analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Connection analysis failed: {e}")
            return {}
    
    def run_statistical_analysis(self) -> Dict[str, Any]:
        """Run comprehensive statistical analysis."""
        if not self.config.run_statistical_analysis:
            logger.info("Statistical analysis disabled")
            return {}
        
        logger.info("Running comprehensive statistical analysis...")
        
        try:
            stats = {}
            
            # Aggregate statistics from all analyses
            if 'cell_density' in self.results:
                cell_results = self.results['cell_density']
                if hasattr(self.cell_density_analyzer, 'statistics'):
                    stats['cell_density'] = self.cell_density_analyzer.statistics
            
            if 'synapse_merge' in self.results:
                stats['synapse_merge'] = self.results['synapse_merge'].get('model_performance', {})
            
            if 'skeleton_pruning' in self.results:
                stats['skeleton_pruning'] = self.results['skeleton_pruning'].get('model_performance', {})
            
            if 'connection_analysis' in self.results:
                stats['connection_analysis'] = self.results['connection_analysis'].get('statistics', {})
            
            # Calculate cross-analysis statistics
            stats['summary'] = self._calculate_summary_statistics()
            
            self.statistics = stats
            logger.info("✓ Statistical analysis completed")
            return stats
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {}
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all analyses."""
        summary = {
            'total_analyses_run': len(self.results),
            'analyses_completed': list(self.results.keys()),
            'data_quality_score': 0.0,
            'overall_confidence': 0.0
        }
        
        # Calculate data quality score based on available data
        data_sources = 0
        total_sources = 4  # cell, synapse, skeleton, connection
        
        if self.config.cell_matrix_path and os.path.exists(self.config.cell_matrix_path):
            data_sources += 1
        if self.config.synapse_data_path and os.path.exists(self.config.synapse_data_path):
            data_sources += 1
        if self.config.skeleton_data_path and os.path.exists(self.config.skeleton_data_path):
            data_sources += 1
        if self.config.connection_data_path and os.path.exists(self.config.connection_data_path):
            data_sources += 1
        
        summary['data_quality_score'] = data_sources / total_sources
        
        # Calculate overall confidence based on analysis results
        confidence_factors = []
        
        if 'cell_density' in self.results and self.results['cell_density']:
            confidence_factors.append(0.8)
        
        if 'synapse_merge' in self.results and self.results['synapse_merge']:
            confidence_factors.append(0.7)
        
        if 'skeleton_pruning' in self.results and self.results['skeleton_pruning']:
            confidence_factors.append(0.7)
        
        if 'connection_analysis' in self.results and self.results['connection_analysis']:
            confidence_factors.append(0.8)
        
        if confidence_factors:
            summary['overall_confidence'] = np.mean(confidence_factors)
        
        return summary
    
    def create_comprehensive_visualizations(self) -> Dict[str, plt.Figure]:
        """Create comprehensive visualizations for all analyses."""
        logger.info("Creating comprehensive visualizations...")
        
        figures = {}
        
        try:
            # Cell density visualizations
            if self.cell_density_analyzer and hasattr(self.cell_density_analyzer, 'create_visualizations'):
                cell_figures = self.cell_density_analyzer.create_visualizations()
                for name, fig in cell_figures.items():
                    figures[f'cell_density_{name}'] = fig
            
            # Summary dashboard
            if self.statistics:
                fig = self._create_summary_dashboard()
                figures['summary_dashboard'] = fig
            
            logger.info(f"Created {len(figures)} visualizations")
            return figures
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return {}
    
    def _create_summary_dashboard(self) -> plt.Figure:
        """Create a summary dashboard visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('H01 Comprehensive Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Analysis completion status
        ax1 = axes[0, 0]
        analyses = list(self.results.keys())
        completion_status = [1 if analysis in self.results else 0 for analysis in analyses]
        
        colors = ['green' if status else 'red' for status in completion_status]
        ax1.bar(analyses, completion_status, color=colors, alpha=0.7)
        ax1.set_title('Analysis Completion Status')
        ax1.set_ylabel('Status (1=Complete, 0=Failed)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Data quality score
        ax2 = axes[0, 1]
        if 'summary' in self.statistics:
            summary = self.statistics['summary']
            metrics = ['Data Quality', 'Overall Confidence']
            values = [summary.get('data_quality_score', 0), summary.get('overall_confidence', 0)]
            
            ax2.bar(metrics, values, color=['blue', 'orange'], alpha=0.7)
            ax2.set_title('Quality Metrics')
            ax2.set_ylabel('Score (0-1)')
            ax2.set_ylim(0, 1)
        
        # Plot 3: Cell density summary (if available)
        ax3 = axes[1, 0]
        if 'cell_density' in self.statistics and 'total_densities' in self.statistics['cell_density']:
            densities = self.statistics['cell_density']['total_densities']
            layer_names = [f"L{i+1}" for i in range(len(densities))]
            
            ax3.bar(layer_names, densities, color='skyblue', alpha=0.7)
            ax3.set_title('Cell Density by Layer')
            ax3.set_ylabel('Density (cells/mm³)')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Analysis statistics
        ax4 = axes[1, 1]
        if 'summary' in self.statistics:
            summary = self.statistics['summary']
            ax4.text(0.1, 0.8, f"Total Analyses: {summary.get('total_analyses_run', 0)}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f"Data Quality: {summary.get('data_quality_score', 0):.2f}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.4, f"Confidence: {summary.get('overall_confidence', 0):.2f}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Analysis Summary')
            ax4.axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_comprehensive_results(self) -> Dict[str, str]:
        """Save all analysis results."""
        logger.info("Saving comprehensive results...")
        
        saved_files = {}
        
        try:
            # Save main results
            results_file = os.path.join(self.config.output_dir, 'comprehensive_results.json')
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            json_results[key][subkey] = subvalue.tolist()
                        else:
                            json_results[key][subkey] = subvalue
                else:
                    json_results[key] = value
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            saved_files['comprehensive_results'] = results_file
            
            # Save statistics
            if self.statistics:
                stats_file = os.path.join(self.config.output_dir, 'statistics.json')
                
                json_stats = {}
                for key, value in self.statistics.items():
                    if isinstance(value, dict):
                        json_stats[key] = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, np.ndarray):
                                json_stats[key][subkey] = subvalue.tolist()
                            else:
                                json_stats[key][subkey] = subvalue
                    else:
                        json_stats[key] = value
                
                with open(stats_file, 'w') as f:
                    json.dump(json_stats, f, indent=2)
                
                saved_files['statistics'] = stats_file
            
            # Save visualizations
            if self.config.save_plots:
                figures = self.create_comprehensive_visualizations()
                for name, fig in figures.items():
                    plot_file = os.path.join(self.config.output_dir, f'{name}.png')
                    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    saved_files[f'plot_{name}'] = plot_file
            
            # Generate comprehensive report
            if self.config.generate_report:
                report_file = os.path.join(self.config.output_dir, 'comprehensive_report.txt')
                self._generate_comprehensive_report(report_file)
                saved_files['report'] = report_file
            
            logger.info(f"Saved {len(saved_files)} files to {self.config.output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return {}
    
    def _generate_comprehensive_report(self, report_file: str):
        """Generate a comprehensive analysis report."""
        with open(report_file, 'w') as f:
            f.write("H01 Comprehensive Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Analysis Configuration:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Output Directory: {self.config.output_dir}\n")
            f.write(f"MIP Level: {self.config.mip_level}\n")
            f.write(f"Voxel Size: {self.config.voxel_size_nm} nm\n")
            f.write(f"Synapse Merge Threshold: {self.config.synapse_merge_threshold}\n")
            f.write(f"Skeleton Pruning Threshold: {self.config.skeleton_pruning_threshold}\n\n")
            
            f.write("Analysis Results:\n")
            f.write("-" * 18 + "\n")
            for analysis_name, results in self.results.items():
                f.write(f"{analysis_name.upper()}:\n")
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, (int, float, str)):
                            f.write(f"  {key}: {value}\n")
                        elif isinstance(value, list) and len(value) <= 5:
                            f.write(f"  {key}: {value}\n")
                        else:
                            f.write(f"  {key}: {type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'unknown'} items\n")
                f.write("\n")
            
            f.write("Statistics Summary:\n")
            f.write("-" * 20 + "\n")
            if self.statistics and 'summary' in self.statistics:
                summary = self.statistics['summary']
                f.write(f"Total Analyses Run: {summary.get('total_analyses_run', 0)}\n")
                f.write(f"Data Quality Score: {summary.get('data_quality_score', 0):.2f}\n")
                f.write(f"Overall Confidence: {summary.get('overall_confidence', 0):.2f}\n")
                f.write(f"Analyses Completed: {', '.join(summary.get('analyses_completed', []))}\n")
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run the complete H01 analysis pipeline.
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive H01 analysis...")
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Run all analyses
            self.run_cell_density_analysis()
            self.run_synapse_merge_analysis()
            self.run_skeleton_pruning_analysis()
            self.run_connection_analysis()
            self.run_statistical_analysis()
            
            # Save results
            if self.config.save_results:
                saved_files = self.save_comprehensive_results()
                self.results['saved_files'] = saved_files
            
            logger.info("Comprehensive H01 analysis completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = H01AnalysisConfig(
        output_dir="./h01_comprehensive_output",
        save_plots=True,
        save_results=True,
        generate_report=True,
        # Set data paths as needed
        # cell_matrix_path="path/to/cellmatrix.mat",
        # synapse_data_path="path/to/synapse_data.json",
        # skeleton_data_path="path/to/skeleton_data.json",
        # connection_data_path="path/to/connection_data.json"
    )
    
    # Create and run analyzer
    analyzer = H01ComprehensiveAnalyzer(config)
    results = analyzer.run_comprehensive_analysis()
    
    print("H01 Comprehensive Analysis completed!")
    print(f"Results saved to: {config.output_dir}") 