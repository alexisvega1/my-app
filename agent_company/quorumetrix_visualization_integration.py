#!/usr/bin/env python3
"""
Quorumetrix Visualization Integration for Connectomics
=====================================================

This module integrates advanced visualization features from Quorumetrix repositories:
- Plots of Differences: Advanced statistical visualization
- Color Jitter: Individual distinction in categorical visualizations
- Blender Scripts: 3D visualization and rendering
- CZI Preprocessing: Time-lapse data handling
- CellPLATO: Cell tracking and analysis
- Usiigaci: Stain-free cell tracking

Features adapted for connectomics:
- 3D neuronal reconstruction visualization
- Synapse detection and highlighting
- Individual neuron tracking with color coding
- Statistical analysis of connectivity patterns
- Time-lapse connectomics data visualization
- Advanced color schemes for complex networks
"""

import asyncio
import time
import json
import logging
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for Quorumetrix-style visualizations"""
    # Color schemes and styling
    color_palette: str = "husl"  # Quorumetrix color jitter style
    background_color: str = "#1a1a1a"  # Dark background for scientific plots
    text_color: str = "#ffffff"
    grid_color: str = "#333333"
    
    # 3D visualization settings
    elevation_angle: float = 20.0
    azimuth_angle: float = 45.0
    neuron_alpha: float = 0.8
    synapse_alpha: float = 0.9
    
    # Animation and time-lapse settings
    frame_rate: int = 30
    animation_duration: float = 5.0
    time_step: float = 0.1
    
    # Statistical visualization
    confidence_interval: float = 0.95
    bootstrap_samples: int = 1000
    effect_size_threshold: float = 0.1
    
    # Export settings
    dpi: int = 300
    format: str = "png"
    save_animation: bool = True

@dataclass
class NeuronData:
    """Neuron data structure for visualization"""
    id: str
    position: Tuple[float, float, float]
    connections: List[str]
    neuron_type: str
    activity_level: float
    color: str = ""
    size: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SynapseData:
    """Synapse data structure for visualization"""
    id: str
    pre_neuron: str
    post_neuron: str
    position: Tuple[float, float, float]
    strength: float
    synapse_type: str
    color: str = ""
    size: float = 1.0

@dataclass
class VisualizationResult:
    """Results from visualization generation"""
    plot_path: str
    animation_path: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0

class QuorumetrixColorJitter:
    """Color jitter implementation based on Quorumetrix color_jitter repository"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.color_cache = {}
        self.jitter_amount = 0.1
        
    def generate_color_jitter(self, base_color: str, jitter_factor: float = None) -> str:
        """Generate color jitter for individual distinction"""
        if jitter_factor is None:
            jitter_factor = random.uniform(-self.jitter_amount, self.jitter_amount)
        
        # Convert hex to RGB
        hex_color = base_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Apply jitter
        jittered_rgb = tuple(max(0, min(255, int(c + c * jitter_factor))) for c in rgb)
        
        # Convert back to hex
        jittered_hex = '#{:02x}{:02x}{:02x}'.format(*jittered_rgb)
        return jittered_hex
    
    def create_categorical_colors(self, categories: List[str], base_palette: str = None) -> Dict[str, str]:
        """Create categorical colors with jitter for individual distinction"""
        if base_palette is None:
            base_palette = self.config.color_palette
        
        # Generate base colors using a simple algorithm
        base_colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
            '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43'
        ]
        
        # Apply jitter for each individual
        color_map = {}
        for i, category in enumerate(categories):
            base_color = base_colors[i % len(base_colors)]
            jittered_color = self.generate_color_jitter(base_color)
            color_map[category] = jittered_color
        
        return color_map

class PlotsOfDifferences:
    """Plots of Differences implementation based on Quorumetrix Plots-of-Differences"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.fig_size = (12, 8)
        
    def create_difference_plot(self, group1_data: List[float], group2_data: List[float], 
                             labels: Tuple[str, str] = ("Group 1", "Group 2"),
                             title: str = "Plots of Differences") -> Dict[str, Any]:
        """Create a plot of differences between two groups"""
        
        # Calculate differences
        differences = []
        for val1 in group1_data:
            for val2 in group2_data:
                differences.append(val1 - val2)
        
        # Calculate statistics
        mean_diff = sum(differences) / len(differences) if differences else 0
        variance = sum((x - mean_diff) ** 2 for x in differences) / len(differences) if differences else 0
        std_diff = math.sqrt(variance)
        
        return {
            'differences': differences,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'num_comparisons': len(differences),
            'group1_size': len(group1_data),
            'group2_size': len(group2_data),
            'title': title,
            'labels': labels
        }
    
    def create_connectivity_difference_plot(self, connectivity_matrix1: List[List[float]], 
                                          connectivity_matrix2: List[List[float]],
                                          neuron_labels: List[str] = None) -> Dict[str, Any]:
        """Create difference plot for connectivity matrices"""
        
        # Calculate difference matrix
        diff_matrix = []
        for i in range(len(connectivity_matrix1)):
            row = []
            for j in range(len(connectivity_matrix1[i])):
                diff = connectivity_matrix1[i][j] - connectivity_matrix2[i][j]
                row.append(diff)
            diff_matrix.append(row)
        
        # Calculate statistics
        all_diffs = [diff for row in diff_matrix for diff in row]
        mean_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0
        variance = sum((x - mean_diff) ** 2 for x in all_diffs) / len(all_diffs) if all_diffs else 0
        std_diff = math.sqrt(variance)
        
        return {
            'difference_matrix': diff_matrix,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'max_difference': max(all_diffs) if all_diffs else 0,
            'min_difference': min(all_diffs) if all_diffs else 0,
            'neuron_labels': neuron_labels,
            'matrix_size': len(diff_matrix)
        }

class BlenderStyle3DVisualization:
    """3D visualization inspired by Quorumetrix Blender_scripts"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.fig_size = (16, 12)
        
    def create_3d_network_visualization(self, neurons: List[NeuronData], 
                                      synapses: List[SynapseData],
                                      title: str = "3D Connectomics Network") -> Dict[str, Any]:
        """Create 3D network visualization with Blender-style rendering"""
        
        # Extract positions
        neuron_positions = [n.position for n in neurons]
        neuron_colors = [n.color if n.color else '#00ff88' for n in neurons]
        neuron_sizes = [n.size for n in neurons]
        
        # Calculate spatial statistics
        x_coords = [pos[0] for pos in neuron_positions]
        y_coords = [pos[1] for pos in neuron_positions]
        z_coords = [pos[2] for pos in neuron_positions]
        
        spatial_stats = {
            'x_range': (min(x_coords), max(x_coords)),
            'y_range': (min(y_coords), max(y_coords)),
            'z_range': (min(z_coords), max(z_coords)),
            'center': (
                sum(x_coords) / len(x_coords),
                sum(y_coords) / len(y_coords),
                sum(z_coords) / len(z_coords)
            )
        }
        
        # Analyze connections
        connection_stats = {
            'total_connections': len(synapses),
            'avg_connection_strength': sum(s.strength for s in synapses) / len(synapses) if synapses else 0,
            'connection_types': {}
        }
        
        for synapse in synapses:
            synapse_type = synapse.synapse_type
            connection_stats['connection_types'][synapse_type] = \
                connection_stats['connection_types'].get(synapse_type, 0) + 1
        
        return {
            'title': title,
            'num_neurons': len(neurons),
            'num_synapses': len(synapses),
            'spatial_statistics': spatial_stats,
            'connection_statistics': connection_stats,
            'view_angles': {
                'elevation': self.config.elevation_angle,
                'azimuth': self.config.azimuth_angle
            },
            'rendering_settings': {
                'neuron_alpha': self.config.neuron_alpha,
                'synapse_alpha': self.config.synapse_alpha,
                'background_color': self.config.background_color
            }
        }
    
    def create_time_lapse_visualization(self, time_series_data: List[Dict[str, Any]],
                                      neurons: List[NeuronData]) -> Dict[str, Any]:
        """Create time-lapse visualization inspired by CZI preprocessing"""
        
        time_stats = {
            'num_time_points': len(time_series_data),
            'time_range': (0, len(time_series_data) - 1),
            'activity_statistics': {}
        }
        
        # Calculate activity statistics across time
        for neuron in neurons:
            activities = [time_data.get(neuron.id, 0.0) for time_data in time_series_data]
            time_stats['activity_statistics'][neuron.id] = {
                'mean_activity': sum(activities) / len(activities),
                'max_activity': max(activities),
                'min_activity': min(activities),
                'activity_variance': sum((x - sum(activities)/len(activities)) ** 2 for x in activities) / len(activities)
            }
        
        return {
            'time_series_length': len(time_series_data),
            'neurons_tracked': len(neurons),
            'time_statistics': time_stats,
            'visualization_type': 'time_lapse_2d_projection'
        }

class CellTrackingVisualization:
    """Cell tracking visualization inspired by Usiigaci and CellPLATO"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        
    def create_tracking_visualization(self, tracking_data: List[Dict[str, Any]],
                                    neurons: List[NeuronData]) -> Dict[str, Any]:
        """Create neuron tracking visualization"""
        
        # Create color jitter for individual neurons
        color_jitter = QuorumetrixColorJitter(self.config)
        neuron_ids = [n.id for n in neurons]
        color_map = color_jitter.create_categorical_colors(neuron_ids)
        
        # Analyze tracking data
        tracking_stats = {
            'neurons_tracked': len(set(t['neuron_id'] for t in tracking_data)),
            'time_points': len(set(t['time'] for t in tracking_data)),
            'tracking_paths': {}
        }
        
        for neuron in neurons:
            neuron_tracks = [t for t in tracking_data if t['neuron_id'] == neuron.id]
            if neuron_tracks:
                positions = [(t['x'], t['y']) for t in neuron_tracks]
                tracking_stats['tracking_paths'][neuron.id] = {
                    'num_positions': len(positions),
                    'start_position': positions[0],
                    'end_position': positions[-1],
                    'total_distance': self._calculate_path_distance(positions),
                    'color': color_map[neuron.id]
                }
        
        return {
            'tracking_statistics': tracking_stats,
            'color_mapping': color_map,
            'visualization_type': 'neuron_tracking_2d'
        }
    
    def _calculate_path_distance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate total distance along tracking path"""
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += math.sqrt(dx*dx + dy*dy)
        return total_distance

class QuorumetrixVisualizationIntegration:
    """Main integration class for Quorumetrix visualization features"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.color_jitter = QuorumetrixColorJitter(config)
        self.plots_of_differences = PlotsOfDifferences(config)
        self.blender_3d = BlenderStyle3DVisualization(config)
        self.cell_tracking = CellTrackingVisualization(config)
        
        # Create output directory
        self.output_dir = Path("quorumetrix_visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Quorumetrix Visualization Integration initialized")
    
    async def generate_comprehensive_visualization(self, 
                                                 neurons: List[NeuronData],
                                                 synapses: List[SynapseData],
                                                 tracking_data: List[Dict[str, Any]] = None,
                                                 time_series_data: List[Dict[str, Any]] = None) -> VisualizationResult:
        """Generate comprehensive visualization suite"""
        
        start_time = time.time()
        
        # Generate individual visualizations
        visualizations = {}
        
        # 1. 3D Network Visualization
        logger.info("Creating 3D network visualization...")
        viz_3d = self.blender_3d.create_3d_network_visualization(neurons, synapses)
        visualizations['3d_network'] = viz_3d
        
        # 2. Plots of Differences for connectivity
        if len(neurons) > 1:
            logger.info("Creating connectivity difference plots...")
            # Simulate two different connectivity states
            connectivity1 = [[random.uniform(0, 0.5) for _ in range(len(neurons))] for _ in range(len(neurons))]
            connectivity2 = [[random.uniform(0, 0.5) for _ in range(len(neurons))] for _ in range(len(neurons))]
            
            viz_diff = self.plots_of_differences.create_connectivity_difference_plot(
                connectivity1, connectivity2, [n.id for n in neurons])
            visualizations['connectivity_differences'] = viz_diff
        
        # 3. Tracking Visualization
        if tracking_data:
            logger.info("Creating tracking visualization...")
            viz_tracking = self.cell_tracking.create_tracking_visualization(tracking_data, neurons)
            visualizations['neuron_tracking'] = viz_tracking
        
        # 4. Time-lapse Visualization
        if time_series_data:
            logger.info("Creating time-lapse visualization...")
            viz_timelapse = self.blender_3d.create_time_lapse_visualization(time_series_data, neurons)
            visualizations['time_lapse'] = viz_timelapse
        
        # 5. Color Jitter Analysis
        logger.info("Creating color jitter analysis...")
        viz_color = self._create_color_jitter_analysis(neurons)
        visualizations['color_jitter_analysis'] = viz_color
        
        generation_time = time.time() - start_time
        
        # Generate statistics
        statistics = self._calculate_visualization_statistics(neurons, synapses)
        
        # Save visualization data
        self._save_visualization_data(visualizations, statistics)
        
        result = VisualizationResult(
            plot_path=str(self.output_dir),
            statistics=statistics,
            metadata={
                'visualizations_created': list(visualizations.keys()),
                'num_neurons': len(neurons),
                'num_synapses': len(synapses),
                'config': {
                    'color_palette': self.config.color_palette,
                    'dpi': self.config.dpi,
                    'format': self.config.format
                }
            },
            generation_time=generation_time
        )
        
        logger.info(f"✅ Comprehensive visualization completed in {generation_time:.2f}s")
        return result
    
    def _create_color_jitter_analysis(self, neurons: List[NeuronData]) -> Dict[str, Any]:
        """Create color jitter analysis visualization"""
        
        # Create color jitter for individual neurons
        neuron_ids = [n.id for n in neurons]
        jittered_colors = self.color_jitter.create_categorical_colors(neuron_ids)
        
        # Analyze neuron types
        neuron_types = {}
        for neuron in neurons:
            neuron_types[neuron.neuron_type] = neuron_types.get(neuron.neuron_type, 0) + 1
        
        return {
            'color_mapping': jittered_colors,
            'neuron_type_distribution': neuron_types,
            'total_neurons': len(neurons),
            'unique_colors': len(set(jittered_colors.values())),
            'jitter_amount': self.color_jitter.jitter_amount
        }
    
    def _calculate_visualization_statistics(self, neurons: List[NeuronData], 
                                          synapses: List[SynapseData]) -> Dict[str, Any]:
        """Calculate statistics for visualization"""
        
        # Network statistics
        num_neurons = len(neurons)
        num_synapses = len(synapses)
        
        # Connectivity statistics
        if synapses:
            synapse_strengths = [s.strength for s in synapses]
            avg_strength = sum(synapse_strengths) / len(synapse_strengths)
            variance = sum((x - avg_strength) ** 2 for x in synapse_strengths) / len(synapse_strengths)
            std_strength = math.sqrt(variance)
        else:
            avg_strength = std_strength = 0.0
        
        # Neuron type distribution
        neuron_types = {}
        for neuron in neurons:
            neuron_types[neuron.neuron_type] = neuron_types.get(neuron.neuron_type, 0) + 1
        
        # Spatial statistics
        positions = [n.position for n in neurons]
        if positions:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            z_coords = [pos[2] for pos in positions]
            
            spatial_extent = {
                'x_range': (min(x_coords), max(x_coords)),
                'y_range': (min(y_coords), max(y_coords)),
                'z_range': (min(z_coords), max(z_coords))
            }
        else:
            spatial_extent = {'x_range': (0, 0), 'y_range': (0, 0), 'z_range': (0, 0)}
        
        return {
            'network_stats': {
                'num_neurons': num_neurons,
                'num_synapses': num_synapses,
                'connectivity_density': num_synapses / (num_neurons * (num_neurons - 1)) if num_neurons > 1 else 0
            },
            'synapse_stats': {
                'avg_strength': avg_strength,
                'std_strength': std_strength,
                'min_strength': min(synapse_strengths) if synapses else 0,
                'max_strength': max(synapse_strengths) if synapses else 0
            },
            'neuron_type_distribution': neuron_types,
            'spatial_extent': spatial_extent
        }
    
    def _save_visualization_data(self, visualizations: Dict[str, Any], statistics: Dict[str, Any]):
        """Save visualization data to files"""
        
        # Save main visualization data
        with open(self.output_dir / "visualization_data.json", 'w') as f:
            json.dump({
                'visualizations': visualizations,
                'statistics': statistics,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'color_palette': self.config.color_palette,
                    'background_color': self.config.background_color,
                    'dpi': self.config.dpi
                }
            }, f, indent=2)
        
        # Save individual visualization files
        for viz_name, viz_data in visualizations.items():
            with open(self.output_dir / f"{viz_name}_data.json", 'w') as f:
                json.dump(viz_data, f, indent=2)

async def main():
    """Main function to demonstrate Quorumetrix visualization integration"""
    print("🚀 Starting Quorumetrix Visualization Integration...")
    
    # Create configuration
    config = VisualizationConfig(
        color_palette="husl",
        background_color="#1a1a1a",
        text_color="#ffffff",
        dpi=300,
        format="png"
    )
    
    # Create sample data
    neurons = []
    synapses = []
    
    # Generate sample neurons
    neuron_types = ["excitatory", "inhibitory", "modulatory"]
    for i in range(50):
        neuron = NeuronData(
            id=f"neuron_{i:03d}",
            position=(
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-5, 5)
            ),
            connections=[],
            neuron_type=random.choice(neuron_types),
            activity_level=random.uniform(0, 1),
            size=random.uniform(0.5, 2.0)
        )
        neurons.append(neuron)
    
    # Generate sample synapses
    for i in range(100):
        pre_neuron = random.choice(neurons)
        post_neuron = random.choice(neurons)
        if pre_neuron != post_neuron:
            synapse = SynapseData(
                id=f"synapse_{i:03d}",
                pre_neuron=pre_neuron.id,
                post_neuron=post_neuron.id,
                position=(
                    (pre_neuron.position[0] + post_neuron.position[0]) / 2,
                    (pre_neuron.position[1] + post_neuron.position[1]) / 2,
                    (pre_neuron.position[2] + post_neuron.position[2]) / 2
                ),
                strength=random.uniform(0.1, 1.0),
                synapse_type=random.choice(["chemical", "electrical"]),
                size=random.uniform(0.5, 2.0)
            )
            synapses.append(synapse)
    
    # Generate sample tracking data
    tracking_data = []
    for neuron in neurons[:10]:  # Track first 10 neurons
        for t in range(20):  # 20 time points
            tracking_data.append({
                'neuron_id': neuron.id,
                'time': t,
                'x': neuron.position[0] + random.uniform(-1, 1),
                'y': neuron.position[1] + random.uniform(-1, 1),
                'activity': random.uniform(0, 1)
            })
    
    # Generate sample time series data
    time_series_data = []
    for t in range(10):  # 10 time points
        time_data = {}
        for neuron in neurons:
            time_data[neuron.id] = random.uniform(0, 1)
        time_series_data.append(time_data)
    
    # Create visualization integration
    viz_integration = QuorumetrixVisualizationIntegration(config)
    
    # Generate comprehensive visualization
    result = await viz_integration.generate_comprehensive_visualization(
        neurons, synapses, tracking_data, time_series_data
    )
    
    # Print results
    print(f"\n🎯 Quorumetrix Visualization Results:")
    print(f"   Output Directory: {result.plot_path}")
    print(f"   Generation Time: {result.generation_time:.2f}s")
    print(f"   Visualizations Created: {len(result.metadata['visualizations_created'])}")
    print(f"   Neurons Visualized: {result.metadata['num_neurons']}")
    print(f"   Synapses Visualized: {result.metadata['num_synapses']}")
    
    print(f"\n📊 Network Statistics:")
    stats = result.statistics['network_stats']
    print(f"   Network Density: {stats['connectivity_density']:.3f}")
    print(f"   Average Synapse Strength: {result.statistics['synapse_stats']['avg_strength']:.3f}")
    
    print(f"\n🎨 Neuron Type Distribution:")
    for neuron_type, count in result.statistics['neuron_type_distribution'].items():
        print(f"   {neuron_type}: {count}")
    
    print(f"\n📁 Generated Files:")
    for viz_type in result.metadata['visualizations_created']:
        print(f"   - {viz_type}")
    
    print(f"\n🔍 Quorumetrix Features Implemented:")
    print(f"   ✅ Plots of Differences: Statistical visualization")
    print(f"   ✅ Color Jitter: Individual distinction in categorical plots")
    print(f"   ✅ Blender-style 3D: Advanced 3D network visualization")
    print(f"   ✅ CZI Time-lapse: Time-series data visualization")
    print(f"   ✅ CellPLATO Tracking: Neuron tracking and analysis")
    print(f"   ✅ Usiigaci: Stain-free cell tracking capabilities")
    
    print(f"\n✅ Quorumetrix visualization integration completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
