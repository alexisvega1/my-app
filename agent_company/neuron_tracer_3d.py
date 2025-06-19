#!/usr/bin/env python3
"""
3D Neuron Tracer for LICONN Replication
Specialized tool for tracing and visualizing neurons in 3D space
Replicates the neuron tracing capabilities shown in the Nature LICONN paper
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import napari
from skimage import measure, morphology, filters
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.spatial import distance
import pandas as pd
import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import colorsys
import skimage.morphology

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NeuronTrace:
    """Represents a traced neuron with its properties."""
    id: int
    coordinates: np.ndarray  # (N, 3) array of voxel coordinates
    volume: int
    surface_area: float
    centroid: np.ndarray
    bounding_box: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    connectivity: List[int]  # IDs of connected neurons
    confidence: float
    uncertainty: float

class NeuronTracer3D:
    """3D neuron tracing and visualization system."""
    
    def __init__(self, segmentation_data: np.ndarray, uncertainty_data: Optional[np.ndarray] = None):
        self.segmentation = segmentation_data
        self.uncertainty = uncertainty_data
        self.traced_neurons: Dict[int, NeuronTrace] = {}
        self.connectivity_matrix = None
        self._extract_neurons()
        
    def _extract_neurons(self):
        """Extract individual neurons from segmentation data."""
        logger.info("Extracting neurons from segmentation data...")
        
        unique_labels = np.unique(self.segmentation)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        logger.info(f"Found {len(unique_labels)} neurons to trace")
        
        for neuron_id in unique_labels:
            # Create binary mask for this neuron
            mask = self.segmentation == neuron_id
            
            # Get coordinates
            coords = np.array(np.where(mask)).T
            
            # Calculate properties
            volume = len(coords)
            centroid = np.mean(coords, axis=0)
            
            # Calculate bounding box
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            bounding_box = (tuple(min_coords), tuple(max_coords))
            
            # Calculate surface area (approximate)
            surface_area = self._calculate_surface_area(mask)
            
            # Calculate confidence and uncertainty
            confidence = 1.0
            uncertainty = 0.0
            if self.uncertainty is not None:
                neuron_uncertainty = self.uncertainty[mask]
                uncertainty = np.mean(neuron_uncertainty)
                confidence = 1.0 - uncertainty
            
            # Create neuron trace
            neuron_trace = NeuronTrace(
                id=neuron_id,
                coordinates=coords,
                volume=volume,
                surface_area=surface_area,
                centroid=centroid,
                bounding_box=bounding_box,
                connectivity=[],
                confidence=confidence,
                uncertainty=uncertainty
            )
            
            self.traced_neurons[neuron_id] = neuron_trace
            
        logger.info(f"Successfully traced {len(self.traced_neurons)} neurons")
        
    def _calculate_surface_area(self, mask: np.ndarray) -> float:
        """Calculate approximate surface area of a neuron."""
        # Use morphological operations to estimate surface area
        eroded = morphology.binary_erosion(mask)
        surface_voxels = np.sum(mask) - np.sum(eroded)
        return float(surface_voxels)
        
    def analyze_connectivity(self, distance_threshold: float = 5.0):
        """Analyze connectivity between neurons based on proximity."""
        logger.info("Analyzing neuron connectivity...")
        
        neuron_ids = list(self.traced_neurons.keys())
        n_neurons = len(neuron_ids)
        
        # Initialize connectivity matrix
        self.connectivity_matrix = np.zeros((n_neurons, n_neurons))
        
        for i, id1 in enumerate(neuron_ids):
            neuron1 = self.traced_neurons[id1]
            
            for j, id2 in enumerate(neuron_ids):
                if i != j:
                    neuron2 = self.traced_neurons[id2]
                    
                    # Calculate minimum distance between neurons
                    min_distance = self._calculate_min_distance(neuron1, neuron2)
                    
                    # Check if neurons are connected (within threshold)
                    if min_distance <= distance_threshold:
                        self.connectivity_matrix[i, j] = 1.0
                        neuron1.connectivity.append(id2)
                        
        logger.info(f"Connectivity analysis complete. Found connections between {np.sum(self.connectivity_matrix > 0)} neuron pairs")
        
    def _calculate_min_distance(self, neuron1: NeuronTrace, neuron2: NeuronTrace) -> float:
        """Calculate minimum distance between two neurons."""
        # Use centroids for efficiency
        distance = np.linalg.norm(neuron1.centroid - neuron2.centroid)
        return distance
        
    def create_3d_tracing_visualization(self, output_path: str = "neuron_tracing_3d.html", 
                                      max_neurons: int = 20):
        """Create interactive 3D visualization of traced neurons."""
        logger.info(f"Creating 3D tracing visualization with up to {max_neurons} neurons...")
        
        # Select neurons to visualize
        neuron_ids = list(self.traced_neurons.keys())
        if len(neuron_ids) > max_neurons:
            # Select neurons with highest confidence
            confidences = [self.traced_neurons[nid].confidence for nid in neuron_ids]
            sorted_indices = np.argsort(confidences)[::-1]
            selected_ids = [neuron_ids[i] for i in sorted_indices[:max_neurons]]
        else:
            selected_ids = neuron_ids
            
        # Create 3D plot
        fig = go.Figure()
        
        # Generate distinct colors
        colors = self._generate_distinct_colors(len(selected_ids))
        
        for i, neuron_id in enumerate(selected_ids):
            neuron = self.traced_neurons[neuron_id]
            
            # Sample coordinates for performance
            coords = neuron.coordinates
            if len(coords) > 1000:
                indices = np.random.choice(len(coords), 1000, replace=False)
                coords = coords[indices]
                
            # Add neuron trace
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1], 
                z=coords[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors[i],
                    opacity=0.8
                ),
                name=f'Neuron {neuron_id}',
                hovertemplate=(
                    f'Neuron {neuron_id}<br>'
                    f'Volume: {neuron.volume}<br>'
                    f'Confidence: {neuron.confidence:.3f}<br>'
                    f'Connections: {len(neuron.connectivity)}<br>'
                    f'X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
                )
            ))
            
            # Add centroid marker
            fig.add_trace(go.Scatter3d(
                x=[neuron.centroid[0]],
                y=[neuron.centroid[1]],
                z=[neuron.centroid[2]],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors[i],
                    symbol='diamond'
                ),
                name=f'Centroid {neuron_id}',
                showlegend=False,
                hovertemplate=f'Centroid {neuron_id}<br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
            ))
            
        # Add connectivity lines
        if self.connectivity_matrix is not None:
            for i, id1 in enumerate(selected_ids):
                for j, id2 in enumerate(selected_ids):
                    if i != j and self.connectivity_matrix[i, j] > 0:
                        neuron1 = self.traced_neurons[id1]
                        neuron2 = self.traced_neurons[id2]
                        
                        fig.add_trace(go.Scatter3d(
                            x=[neuron1.centroid[0], neuron2.centroid[0]],
                            y=[neuron1.centroid[1], neuron2.centroid[1]],
                            z=[neuron1.centroid[2], neuron2.centroid[2]],
                            mode='lines',
                            line=dict(color='gray', width=2, dash='dash'),
                            name=f'Connection {id1}-{id2}',
                            showlegend=False,
                            hovertemplate=f'Connection: {id1} ↔ {id2}<extra></extra>'
                        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'LICONN 3D Neuron Tracing Visualization',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': 'black'}
            },
            scene=dict(
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                zaxis_title='Z (μm)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='data'
            ),
            width=1200,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save interactive plot
        fig.write_html(output_path)
        logger.info(f"3D tracing visualization saved to {output_path}")
        
        return fig
        
    def create_connectivity_analysis(self, output_path: str = "connectivity_analysis.html"):
        """Create connectivity analysis visualization."""
        if self.connectivity_matrix is None:
            logger.warning("Connectivity matrix not available. Run analyze_connectivity() first.")
            return None
            
        logger.info("Creating connectivity analysis visualization...")
        
        neuron_ids = list(self.traced_neurons.keys())
        n_neurons = len(neuron_ids)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Connectivity Matrix', 'Connection Distribution', 
                          'Neuron Properties', 'Network Graph'),
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Connectivity matrix heatmap
        fig.add_trace(
            go.Heatmap(
                z=self.connectivity_matrix,
                x=[f'N{id}' for id in neuron_ids],
                y=[f'N{id}' for id in neuron_ids],
                colorscale='Viridis',
                showscale=True,
                name='Connectivity'
            ),
            row=1, col=1
        )
        
        # 2. Connection distribution
        connection_counts = np.sum(self.connectivity_matrix, axis=1)
        fig.add_trace(
            go.Bar(
                x=[f'N{id}' for id in neuron_ids],
                y=connection_counts,
                name='Connections',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Neuron properties scatter
        volumes = [self.traced_neurons[id].volume for id in neuron_ids]
        confidences = [self.traced_neurons[id].confidence for id in neuron_ids]
        
        fig.add_trace(
            go.Scatter(
                x=volumes,
                y=confidences,
                mode='markers+text',
                text=[f'N{id}' for id in neuron_ids],
                textposition="top center",
                marker=dict(size=10, color='red'),
                name='Properties'
            ),
            row=2, col=1
        )
        
        # 4. Network graph (2D projection)
        # Use PCA to project 3D centroids to 2D
        centroids = np.array([self.traced_neurons[id].centroid for id in neuron_ids])
        
        # Simple 2D projection (X vs Y)
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode='markers+text',
                text=[f'N{id}' for id in neuron_ids],
                textposition="top center",
                marker=dict(size=15, color='green'),
                name='Network'
            ),
            row=2, col=2
        )
        
        # Add connection lines to network graph
        for i, id1 in enumerate(neuron_ids):
            for j, id2 in enumerate(neuron_ids):
                if i != j and self.connectivity_matrix[i, j] > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[centroids[i, 0], centroids[j, 0]],
                            y=[centroids[i, 1], centroids[j, 1]],
                            mode='lines',
                            line=dict(color='gray', width=1),
                            showlegend=False
                        ),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'LICONN Connectivity Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            width=1200,
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Neuron ID", row=1, col=1)
        fig.update_yaxes(title_text="Neuron ID", row=1, col=1)
        fig.update_xaxes(title_text="Neuron ID", row=1, col=2)
        fig.update_yaxes(title_text="Connection Count", row=1, col=2)
        fig.update_xaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        fig.update_xaxes(title_text="X Position", row=2, col=2)
        fig.update_yaxes(title_text="Y Position", row=2, col=2)
        
        # Save plot
        fig.write_html(output_path)
        logger.info(f"Connectivity analysis saved to {output_path}")
        
        return fig
        
    def create_statistical_report(self, output_path: str = "neuron_statistics.html"):
        """Create comprehensive statistical report."""
        logger.info("Creating statistical report...")
        
        if not self.traced_neurons:
            logger.warning("No neurons traced")
            return None
            
        # Calculate statistics
        volumes = [neuron.volume for neuron in self.traced_neurons.values()]
        confidences = [neuron.confidence for neuron in self.traced_neurons.values()]
        uncertainties = [neuron.uncertainty for neuron in self.traced_neurons.values()]
        surface_areas = [neuron.surface_area for neuron in self.traced_neurons.values()]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volume Distribution', 'Confidence Distribution',
                          'Uncertainty vs Volume', 'Surface Area vs Volume'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Volume distribution
        fig.add_trace(
            go.Histogram(x=volumes, nbinsx=20, name='Volume', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Confidence distribution
        fig.add_trace(
            go.Histogram(x=confidences, nbinsx=20, name='Confidence', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # 3. Uncertainty vs Volume
        fig.add_trace(
            go.Scatter(x=volumes, y=uncertainties, mode='markers', 
                      name='Uncertainty', marker_color='red'),
            row=2, col=1
        )
        
        # 4. Surface Area vs Volume
        fig.add_trace(
            go.Scatter(x=volumes, y=surface_areas, mode='markers',
                      name='Surface Area', marker_color='purple'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'LICONN Neuron Statistics',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            width=1200,
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Volume (voxels)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Confidence", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Volume (voxels)", row=2, col=1)
        fig.update_yaxes(title_text="Uncertainty", row=2, col=1)
        fig.update_xaxes(title_text="Volume (voxels)", row=2, col=2)
        fig.update_yaxes(title_text="Surface Area (voxels)", row=2, col=2)
        
        # Save plot
        fig.write_html(output_path)
        logger.info(f"Statistical report saved to {output_path}")
        
        return fig
        
    def _generate_distinct_colors(self, n_colors: int) -> List[str]:
        """Generate n distinct colors."""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            saturation = 0.7
            value = 0.8
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
        return colors
        
    def export_traces(self, output_path: str = "neuron_traces.json"):
        """Export traced neurons to JSON format."""
        logger.info("Exporting neuron traces to JSON...")
        
        def to_py(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (list, tuple)):
                return [to_py(x) for x in obj]
            return obj
        
        traces_data = {}
        for neuron_id, neuron in self.traced_neurons.items():
            traces_data[str(int(neuron_id))] = {
                'id': int(neuron.id),
                'volume': int(neuron.volume),
                'surface_area': float(neuron.surface_area),
                'centroid': to_py(neuron.centroid),
                'bounding_box': to_py(neuron.bounding_box),
                'connectivity': [int(x) for x in neuron.connectivity],
                'confidence': float(neuron.confidence),
                'uncertainty': float(neuron.uncertainty),
                'coordinate_count': int(len(neuron.coordinates))
            }
        with open(output_path, 'w') as f:
            json.dump(traces_data, f, indent=2)
        logger.info(f"Neuron traces exported to {output_path}")
        
    def create_comprehensive_visualization(self, output_dir: str = "neuron_tracing_results"):
        """Create comprehensive visualization package."""
        logger.info("Creating comprehensive visualization package...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Analyze connectivity
        self.analyze_connectivity()
        
        # Create all visualizations
        self.create_3d_tracing_visualization(output_path / "3d_tracing.html")
        self.create_connectivity_analysis(output_path / "connectivity_analysis.html")
        self.create_statistical_report(output_path / "statistics.html")
        
        # Export traces
        self.export_traces(output_path / "neuron_traces.json")
        
        # Create summary report
        self._create_summary_report(output_path)
        
        logger.info(f"Comprehensive visualization package created in {output_path}")
        
    def _create_summary_report(self, output_path: Path):
        """Create summary report."""
        report_content = f"""# LICONN Neuron Tracing Results

## Summary
- **Total Neurons Traced**: {len(self.traced_neurons)}
- **Total Volume**: {sum(n.volume for n in self.traced_neurons.values())} voxels
- **Average Confidence**: {np.mean([n.confidence for n in self.traced_neurons.values()]):.3f}
- **Average Uncertainty**: {np.mean([n.uncertainty for n in self.traced_neurons.values()]):.3f}

## Files Generated
- `3d_tracing.html`: Interactive 3D neuron visualization
- `connectivity_analysis.html`: Connectivity analysis and network graphs
- `statistics.html`: Statistical analysis of neuron properties
- `neuron_traces.json`: Raw neuron trace data

## Key Features
- 3D neuron tracing with confidence scores
- Connectivity analysis between neurons
- Statistical property analysis
- Interactive visualizations
- Export capabilities

---
Generated by LICONN Neuron Tracer 3D
"""
        
        with open(output_path / "README.md", 'w') as f:
            f.write(report_content)

    def create_morphological_traces(self, max_neurons: int = 10, output_dir: str = "neuron_tracing_results"):
        """Extract and visualize 3D skeleton traces for the largest neurons."""
        logger.info(f"Extracting and visualizing skeleton traces for the largest {max_neurons} neurons...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Check for 3D skeletonization
        try:
            from skimage.morphology import skeletonize_3d
            use_3d = True
        except ImportError:
            logger.warning("skeletonize_3d not available, using 2D skeletonization slice-by-slice.")
            from skimage.morphology import skeletonize
            use_3d = False

        # Select largest neurons
        neuron_ids = list(self.traced_neurons.keys())
        volumes = [self.traced_neurons[nid].volume for nid in neuron_ids]
        sorted_ids = [nid for _, nid in sorted(zip(volumes, neuron_ids), reverse=True)]
        selected_ids = sorted_ids[:max_neurons]

        traces = []
        colors = self._generate_distinct_colors(len(selected_ids))

        fig = go.Figure()
        fig_png = plt.figure(figsize=(10, 10))
        ax_png = fig_png.add_subplot(111, projection='3d')

        for i, neuron_id in enumerate(selected_ids):
            mask = (self.segmentation == neuron_id)
            if np.sum(mask) == 0:
                continue
            # Skeletonize
            if use_3d:
                skeleton = skeletonize_3d(mask)
            else:
                skeleton = np.zeros_like(mask, dtype=bool)
                for z in range(mask.shape[2]):
                    skeleton[:, :, z] = skeletonize(mask[:, :, z])
            skel_coords = np.array(np.where(skeleton)).T
            if len(skel_coords) < 2:
                continue
            # For visualization, plot as a 3D line (approximate by sorting by Z)
            skel_coords = skel_coords[np.argsort(skel_coords[:,2])]
            traces.append(skel_coords)
            # Plotly 3D line (use 'rgb(r,g,b)' string)
            fig.add_trace(go.Scatter3d(
                x=skel_coords[:,0], y=skel_coords[:,1], z=skel_coords[:,2],
                mode='lines',
                line=dict(color=colors[i], width=6),
                name=f'Neuron {neuron_id}'
            ))
            # Matplotlib 3D line (convert to 0-1 float tuple)
            rgb = colors[i]
            if rgb.startswith('rgb'):
                rgb_tuple = tuple(int(x)/255 for x in rgb[4:-1].split(','))
            else:
                rgb_tuple = rgb
            ax_png.plot(skel_coords[:,0], skel_coords[:,1], skel_coords[:,2], color=rgb_tuple, linewidth=2, label=f'Neuron {neuron_id}')

        # Plotly layout
        fig.update_layout(
            title='Morphological Traces (Skeletons) of Largest Neurons',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            width=1000, height=900
        )
        fig.write_html(str(output_path / 'morphological_traces.html'))
        logger.info(f"Interactive 3D skeleton traces saved to {output_path / 'morphological_traces.html'}")

        # Matplotlib PNG
        ax_png.set_title('Morphological Traces (Skeletons) of Largest Neurons')
        ax_png.set_xlabel('X')
        ax_png.set_ylabel('Y')
        ax_png.set_zlabel('Z')
        ax_png.legend()
        plt.tight_layout()
        plt.savefig(str(output_path / 'morphological_traces.png'), dpi=300)
        plt.close(fig_png)
        logger.info(f"Publication-quality PNG saved to {output_path / 'morphological_traces.png'}")

def main():
    """Main entry point for neuron tracing."""
    parser = argparse.ArgumentParser(description="3D Neuron Tracer for LICONN Replication")
    parser.add_argument("segmentation_file", help="Path to segmentation .npy file")
    parser.add_argument("--uncertainty", help="Path to uncertainty .npy file (optional)")
    parser.add_argument("--output", default="neuron_tracing_results", help="Output directory")
    parser.add_argument("--max-neurons", type=int, default=20, help="Maximum neurons to visualize")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading segmentation data from {args.segmentation_file}")
    segmentation = np.load(args.segmentation_file)
    
    uncertainty = None
    if args.uncertainty:
        logger.info(f"Loading uncertainty data from {args.uncertainty}")
        uncertainty = np.load(args.uncertainty)
    
    # Create tracer
    tracer = NeuronTracer3D(segmentation, uncertainty)
    
    # Create comprehensive visualization
    tracer.create_comprehensive_visualization(args.output)
    
    logger.info("Neuron tracing completed!")

if __name__ == "__main__":
    main() 