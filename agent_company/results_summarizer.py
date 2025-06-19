#!/usr/bin/env python3
"""
Comprehensive Results Summarizer for H01 Connectomics
====================================================
Generate detailed reports, visualizations, and exports from batch processing results:
- Aggregated statistics across all regions
- Interactive visualizations
- Export to various formats
- Quality assessment reports
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy import stats

logger = logging.getLogger(__name__)

class ResultsSummarizer:
    """Comprehensive results summarizer for batch processing."""
    
    def __init__(self, batch_output_dir: str = "h01_production_batch"):
        self.batch_dir = Path(batch_output_dir)
        self.results = {}
        self.aggregated_stats = {}
        
        logger.info(f"Results summarizer initialized for: {self.batch_dir}")
    
    def load_all_results(self) -> Dict[str, Any]:
        """Load all results from batch processing."""
        logger.info("Loading all batch processing results...")
        
        all_results = {
            'regions': {},
            'aggregated': {},
            'metadata': {
                'total_regions': 0,
                'successful_regions': 0,
                'failed_regions': 0,
                'processing_start': None,
                'processing_end': None
            }
        }
        
        # Find all region directories
        region_dirs = [d for d in self.batch_dir.iterdir() if d.is_dir() and '_' in d.name]
        
        for region_dir in region_dirs:
            region_name = region_dir.name
            logger.info(f"Processing results for: {region_name}")
            
            region_results = self._load_region_results(region_dir)
            if region_results:
                all_results['regions'][region_name] = region_results
                all_results['metadata']['successful_regions'] += 1
            else:
                all_results['metadata']['failed_regions'] += 1
            
            all_results['metadata']['total_regions'] += 1
        
        # Aggregate results
        all_results['aggregated'] = self._aggregate_results(all_results['regions'])
        
        self.results = all_results
        logger.info(f"Loaded results for {all_results['metadata']['successful_regions']} regions")
        
        return all_results
    
    def _load_region_results(self, region_dir: Path) -> Optional[Dict[str, Any]]:
        """Load results for a single region."""
        try:
            region_results = {
                'region_name': region_dir.name,
                'files': {},
                'analysis': {},
                'statistics': {}
            }
            
            # Load advanced analysis report
            adv_report_file = region_dir / "advanced_analysis_report.json"
            if adv_report_file.exists():
                with open(adv_report_file, 'r') as f:
                    adv_report = json.load(f)
                region_results['analysis'] = adv_report
            
            # Load traces
            traces_file = region_dir / "traces.json"
            if traces_file.exists():
                with open(traces_file, 'r') as f:
                    traces = json.load(f)
                region_results['files']['traces'] = traces
            
            # Load raw data info
            raw_file = region_dir / "raw_data.npy"
            if raw_file.exists():
                raw_data = np.load(raw_file)
                region_results['statistics']['data_shape'] = list(raw_data.shape)
                region_results['statistics']['data_size_mb'] = raw_data.nbytes / (1024 * 1024)
            
            # Load segmentation info
            seg_file = region_dir / "segmentation.npy"
            if seg_file.exists():
                seg_data = np.load(seg_file)
                region_results['statistics']['segmentation_shape'] = list(seg_data.shape)
                region_results['statistics']['num_components'] = int(np.max(seg_data))
            
            return region_results
        
        except Exception as e:
            logger.error(f"Error loading results for {region_dir}: {e}")
            return None
    
    def _aggregate_results(self, regions: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results across all regions."""
        aggregated = {
            'total_neurons': 0,
            'total_synapses': 0,
            'total_circuits': 0,
            'synapse_types': Counter(),
            'neuron_types': Counter(),
            'motif_types': Counter(),
            'region_performance': {},
            'data_statistics': {
                'total_data_size_gb': 0,
                'average_region_size_mb': 0,
                'total_components': 0
            }
        }
        
        for region_name, region_data in regions.items():
            analysis = region_data.get('analysis', {})
            summary = analysis.get('summary', {})
            stats = analysis.get('statistics', {})
            
            # Basic counts
            aggregated['total_neurons'] += summary.get('total_neurons', 0)
            aggregated['total_synapses'] += summary.get('total_synapses', 0)
            aggregated['total_circuits'] += summary.get('total_motifs', 0)
            
            # Type distributions
            synapse_stats = stats.get('synapse_analysis', {})
            type_dist = synapse_stats.get('type_distribution', {})
            for synapse_type, count in type_dist.items():
                aggregated['synapse_types'][synapse_type] += count
            
            morph_stats = stats.get('morphological_analysis', {})
            type_dist = morph_stats.get('type_distribution', {})
            for neuron_type, count in type_dist.items():
                aggregated['neuron_types'][neuron_type] += count
            
            motif_stats = stats.get('motif_analysis', {})
            type_dist = motif_stats.get('statistics', {}).get('type_distribution', {})
            for motif_type, count in type_dist.items():
                aggregated['motif_types'][motif_type] += count
            
            # Data statistics
            region_stats = region_data.get('statistics', {})
            aggregated['data_statistics']['total_data_size_gb'] += region_stats.get('data_size_mb', 0) / 1024
            aggregated['data_statistics']['total_components'] += region_stats.get('num_components', 0)
        
        # Calculate averages
        num_regions = len(regions)
        if num_regions > 0:
            aggregated['data_statistics']['average_region_size_mb'] = (
                aggregated['data_statistics']['total_data_size_gb'] * 1024 / num_regions
            )
        
        return aggregated
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive HTML report."""
        if not self.results:
            self.load_all_results()
        
        html_content = self._create_html_report()
        
        # Save report
        report_file = self.batch_dir / "comprehensive_report.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive report saved: {report_file}")
        return str(report_file)
    
    def _create_html_report(self) -> str:
        """Create comprehensive HTML report."""
        aggregated = self.results['aggregated']
        metadata = self.results['metadata']
        
        # Create visualizations
        self._create_summary_plots()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>H01 Connectomics - Comprehensive Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; }}
                .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; background: #f8f9fa; padding: 15px; margin: 10px; border-radius: 8px; text-align: center; min-width: 150px; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .metric-label {{ color: #666; font-size: 0.9em; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                .region-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .region-table th, .region-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .region-table th {{ background: #f8f9fa; font-weight: bold; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 H01 Connectomics - Comprehensive Results Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="metric">
                    <div class="metric-value">{metadata['successful_regions']}</div>
                    <div class="metric-label">Regions Processed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{aggregated['total_neurons']:,}</div>
                    <div class="metric-label">Total Neurons</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{aggregated['total_synapses']:,}</div>
                    <div class="metric-label">Total Synapses</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{aggregated['total_circuits']:,}</div>
                    <div class="metric-label">Circuit Motifs</div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔬 Analysis Results</h2>
                
                <h3>Synapse Type Distribution</h3>
                <div class="chart">
                    <img src="synapse_types.png" alt="Synapse Types" style="max-width: 100%; height: auto;">
                </div>
                
                <h3>Neuron Type Distribution</h3>
                <div class="chart">
                    <img src="neuron_types.png" alt="Neuron Types" style="max-width: 100%; height: auto;">
                </div>
                
                <h3>Circuit Motif Distribution</h3>
                <div class="chart">
                    <img src="motif_types.png" alt="Circuit Motifs" style="max-width: 100%; height: auto;">
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Performance Statistics</h2>
                <p><strong>Total Data Processed:</strong> {aggregated['data_statistics']['total_data_size_gb']:.2f} GB</p>
                <p><strong>Average Region Size:</strong> {aggregated['data_statistics']['average_region_size_mb']:.1f} MB</p>
                <p><strong>Total Components:</strong> {aggregated['data_statistics']['total_components']:,}</p>
            </div>
            
            <div class="section">
                <h2>🌍 Region Details</h2>
                <table class="region-table">
                    <thead>
                        <tr>
                            <th>Region</th>
                            <th>Neurons</th>
                            <th>Synapses</th>
                            <th>Circuits</th>
                            <th>Data Size (MB)</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for region_name, region_data in self.results['regions'].items():
            analysis = region_data.get('analysis', {})
            summary = analysis.get('summary', {})
            stats = region_data.get('statistics', {})
            
            html_content += f"""
                        <tr>
                            <td>{region_name}</td>
                            <td>{summary.get('total_neurons', 0):,}</td>
                            <td>{summary.get('total_synapses', 0):,}</td>
                            <td>{summary.get('total_motifs', 0):,}</td>
                            <td>{stats.get('data_size_mb', 0):.1f}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📋 Quality Assessment</h2>
                <p><strong>Success Rate:</strong> <span class="success">{metadata['successful_regions']}/{metadata['total_regions']} ({metadata['successful_regions']/metadata['total_regions']*100:.1f}%)</span></p>
                <p><strong>Data Coverage:</strong> <span class="success">Comprehensive analysis across multiple brain regions</span></p>
                <p><strong>Analysis Depth:</strong> <span class="success">Advanced synapse classification, circuit motif detection, and morphological analysis</span></p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _create_summary_plots(self):
        """Create summary plots for the report."""
        aggregated = self.results['aggregated']
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Synapse types
        if aggregated['synapse_types']:
            axes[0].pie(aggregated['synapse_types'].values(), labels=aggregated['synapse_types'].keys(), autopct='%1.1f%%')
            axes[0].set_title('Synapse Type Distribution')
        
        # Neuron types
        if aggregated['neuron_types']:
            axes[1].pie(aggregated['neuron_types'].values(), labels=aggregated['neuron_types'].keys(), autopct='%1.1f%%')
            axes[1].set_title('Neuron Type Distribution')
        
        # Motif types
        if aggregated['motif_types']:
            axes[2].pie(aggregated['motif_types'].values(), labels=aggregated['motif_types'].keys(), autopct='%1.1f%%')
            axes[2].set_title('Circuit Motif Distribution')
        
        plt.tight_layout()
        
        # Save individual plots
        if aggregated['synapse_types']:
            plt.figure(figsize=(8, 6))
            plt.pie(aggregated['synapse_types'].values(), labels=aggregated['synapse_types'].keys(), autopct='%1.1f%%')
            plt.title('Synapse Type Distribution')
            plt.savefig(self.batch_dir / 'synapse_types.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        if aggregated['neuron_types']:
            plt.figure(figsize=(8, 6))
            plt.pie(aggregated['neuron_types'].values(), labels=aggregated['neuron_types'].keys(), autopct='%1.1f%%')
            plt.title('Neuron Type Distribution')
            plt.savefig(self.batch_dir / 'neuron_types.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        if aggregated['motif_types']:
            plt.figure(figsize=(8, 6))
            plt.pie(aggregated['motif_types'].values(), labels=aggregated['motif_types'].keys(), autopct='%1.1f%%')
            plt.title('Circuit Motif Distribution')
            plt.savefig(self.batch_dir / 'motif_types.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def export_to_csv(self) -> str:
        """Export results to CSV format."""
        if not self.results:
            self.load_all_results()
        
        # Create region summary DataFrame
        region_data = []
        for region_name, region_data_dict in self.results['regions'].items():
            analysis = region_data_dict.get('analysis', {})
            summary = analysis.get('summary', {})
            stats = region_data_dict.get('statistics', {})
            
            row = {
                'region_name': region_name,
                'total_neurons': summary.get('total_neurons', 0),
                'total_synapses': summary.get('total_synapses', 0),
                'total_circuits': summary.get('total_motifs', 0),
                'data_size_mb': stats.get('data_size_mb', 0),
                'data_shape': str(stats.get('data_shape', [])),
                'num_components': stats.get('num_components', 0)
            }
            
            # Add type distributions
            synapse_stats = analysis.get('statistics', {}).get('synapse_analysis', {})
            for synapse_type, count in synapse_stats.get('type_distribution', {}).items():
                row[f'synapse_{synapse_type}'] = count
            
            morph_stats = analysis.get('statistics', {}).get('morphological_analysis', {})
            for neuron_type, count in morph_stats.get('type_distribution', {}).items():
                row[f'neuron_{neuron_type}'] = count
            
            region_data.append(row)
        
        df = pd.DataFrame(region_data)
        csv_file = self.batch_dir / "region_summary.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"CSV export saved: {csv_file}")
        return str(csv_file)
    
    def create_interactive_dashboard(self) -> str:
        """Create an interactive Plotly dashboard."""
        if not self.results:
            self.load_all_results()
        
        aggregated = self.results['aggregated']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Synapse Types', 'Neuron Types', 'Circuit Motifs', 'Region Performance'),
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Synapse types pie chart
        if aggregated['synapse_types']:
            fig.add_trace(
                go.Pie(labels=list(aggregated['synapse_types'].keys()), 
                      values=list(aggregated['synapse_types'].values()),
                      name="Synapse Types"),
                row=1, col=1
            )
        
        # Neuron types pie chart
        if aggregated['neuron_types']:
            fig.add_trace(
                go.Pie(labels=list(aggregated['neuron_types'].keys()), 
                      values=list(aggregated['neuron_types'].values()),
                      name="Neuron Types"),
                row=1, col=2
            )
        
        # Circuit motifs bar chart
        if aggregated['motif_types']:
            fig.add_trace(
                go.Bar(x=list(aggregated['motif_types'].keys()), 
                      y=list(aggregated['motif_types'].values()),
                      name="Circuit Motifs"),
                row=2, col=1
            )
        
        # Region performance scatter
        region_names = []
        neuron_counts = []
        synapse_counts = []
        
        for region_name, region_data in self.results['regions'].items():
            analysis = region_data.get('analysis', {})
            summary = analysis.get('summary', {})
            
            region_names.append(region_name)
            neuron_counts.append(summary.get('total_neurons', 0))
            synapse_counts.append(summary.get('total_synapses', 0))
        
        if region_names:
            fig.add_trace(
                go.Scatter(x=neuron_counts, y=synapse_counts, mode='markers+text',
                          text=region_names, textposition="top center",
                          name="Region Performance"),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="H01 Connectomics - Interactive Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save dashboard
        dashboard_file = self.batch_dir / "interactive_dashboard.html"
        fig.write_html(str(dashboard_file))
        
        logger.info(f"Interactive dashboard saved: {dashboard_file}")
        return str(dashboard_file)

def main():
    """Main function for results summarization."""
    print("H01 Connectomics Results Summarizer")
    print("=" * 40)
    
    # Initialize summarizer
    summarizer = ResultsSummarizer()
    
    # Load results
    results = summarizer.load_all_results()
    
    print(f"Loaded results for {results['metadata']['successful_regions']} regions")
    print(f"Total neurons: {results['aggregated']['total_neurons']:,}")
    print(f"Total synapses: {results['aggregated']['total_synapses']:,}")
    print(f"Total circuits: {results['aggregated']['total_circuits']:,}")
    
    # Generate reports
    print("\nGenerating reports...")
    
    # Comprehensive HTML report
    report_file = summarizer.generate_comprehensive_report()
    print(f"✓ Comprehensive report: {report_file}")
    
    # CSV export
    csv_file = summarizer.export_to_csv()
    print(f"✓ CSV export: {csv_file}")
    
    # Interactive dashboard
    dashboard_file = summarizer.create_interactive_dashboard()
    print(f"✓ Interactive dashboard: {dashboard_file}")
    
    print("\nAll reports generated successfully!")
    print("Open the HTML files in your browser to view the results.")

if __name__ == "__main__":
    main() 