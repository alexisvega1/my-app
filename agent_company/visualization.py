#!/usr/bin/env python3
"""
H01 Tracing Results Visualization
================================
Interactive visualization system for H01 connectomics data and tracing results.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import Slider, Button, RadioButtons
    MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib available for visualization")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - basic visualization disabled")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    logger.info("Plotly available for interactive visualization")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - interactive visualization disabled")

try:
    import napari
    NAPARI_AVAILABLE = True
    logger.info("Napari available for 3D visualization")
except ImportError:
    NAPARI_AVAILABLE = False
    logger.warning("Napari not available - 3D visualization disabled")

class H01Visualizer:
    """Comprehensive H01 tracing results visualizer."""
    
    def __init__(self, results_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing H01 processing results
        """
        self.results_dir = Path(results_dir)
        self.data = {}
        self.metadata = {}
        
        # Load available data
        self._load_results()
        
        logger.info(f"H01 Visualizer initialized with data from {results_dir}")
    
    def _load_results(self):
        """Load all available results from the directory."""
        try:
            # Load segmentation results
            seg_files = list(self.results_dir.glob("*segmentation*.npy"))
            for seg_file in seg_files:
                name = seg_file.stem.replace("_segmentation", "")
                self.data[f"{name}_segmentation"] = np.load(seg_file)
                logger.info(f"Loaded segmentation: {name} - shape: {self.data[f'{name}_segmentation'].shape}")
            
            # Load uncertainty results
            unc_files = list(self.results_dir.glob("*uncertainty*.npy"))
            for unc_file in unc_files:
                name = unc_file.stem.replace("_uncertainty", "")
                self.data[f"{name}_uncertainty"] = np.load(unc_file)
                logger.info(f"Loaded uncertainty: {name} - shape: {self.data[f'{name}_uncertainty'].shape}")
            
            # Load proofreading results
            proof_files = list(self.results_dir.glob("proofread_*.npy"))
            for proof_file in proof_files:
                name = proof_file.stem
                self.data[name] = np.load(proof_file)
                logger.info(f"Loaded proofreading: {name} - shape: {self.data[name].shape}")
            
            # Load metadata
            metadata_files = list(self.results_dir.glob("*.json"))
            for meta_file in metadata_files:
                name = meta_file.stem
                with open(meta_file, 'r') as f:
                    self.metadata[name] = json.load(f)
                logger.info(f"Loaded metadata: {name}")
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(self.data.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        if dataset_name not in self.data:
            return {}
        
        data = self.data[dataset_name]
        return {
            'name': dataset_name,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'std_value': float(np.std(data)),
            'size_mb': data.nbytes / (1024 * 1024)
        }
    
    def create_2d_slice_viewer(self, dataset_name: str, save_path: Optional[str] = None):
        """Create an interactive 2D slice viewer."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for 2D visualization")
            return
        
        if dataset_name not in self.data:
            logger.error(f"Dataset {dataset_name} not found")
            return
        
        data = self.data[dataset_name]
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.9)
        
        # Initial slice (middle of the volume)
        current_slice = data.shape[0] // 2
        img = ax.imshow(data[current_slice], cmap='viridis', aspect='equal')
        ax.set_title(f'{dataset_name} - Slice {current_slice}/{data.shape[0]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Intensity')
        
        # Slider for slice selection
        ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
        slider = Slider(
            ax_slider, 'Slice', 0, data.shape[0] - 1,
            valinit=current_slice, valstep=1
        )
        
        def update_slice(val):
            slice_idx = int(val)
            img.set_array(data[slice_idx])
            ax.set_title(f'{dataset_name} - Slice {slice_idx}/{data.shape[0]}')
            fig.canvas.draw_idle()
        
        slider.on_changed(update_slice)
        
        # Add buttons for different views
        ax_xy = plt.axes([0.8, 0.8, 0.1, 0.05])
        ax_xz = plt.axes([0.8, 0.7, 0.1, 0.05])
        ax_yz = plt.axes([0.8, 0.6, 0.1, 0.05])
        
        btn_xy = Button(ax_xy, 'XY')
        btn_xz = Button(ax_xz, 'XZ')
        btn_yz = Button(ax_yz, 'YZ')
        
        def switch_to_xy(event):
            slider.valmax = data.shape[0] - 1
            slider.set_val(current_slice)
            update_slice(current_slice)
        
        def switch_to_xz(event):
            slider.valmax = data.shape[1] - 1
            slider.set_val(current_slice)
            img.set_array(data[:, current_slice, :])
            ax.set_title(f'{dataset_name} - XZ Slice {current_slice}/{data.shape[1]}')
            fig.canvas.draw_idle()
        
        def switch_to_yz(event):
            slider.valmax = data.shape[2] - 1
            slider.set_val(current_slice)
            img.set_array(data[:, :, current_slice])
            ax.set_title(f'{dataset_name} - YZ Slice {current_slice}/{data.shape[2]}')
            fig.canvas.draw_idle()
        
        btn_xy.on_clicked(switch_to_xy)
        btn_xz.on_clicked(switch_to_xz)
        btn_yz.on_clicked(switch_to_yz)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 2D viewer to {save_path}")
        
        plt.show()
    
    def create_comparison_viewer(self, dataset1: str, dataset2: str, save_path: Optional[str] = None):
        """Create a side-by-side comparison viewer."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for comparison visualization")
            return
        
        if dataset1 not in self.data or dataset2 not in self.data:
            logger.error("One or both datasets not found")
            return
        
        data1 = self.data[dataset1]
        data2 = self.data[dataset2]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(bottom=0.15)
        
        # Initial slice
        current_slice = min(data1.shape[0], data2.shape[0]) // 2
        
        # Plot first dataset
        img1 = ax1.imshow(data1[current_slice], cmap='viridis', aspect='equal')
        ax1.set_title(f'{dataset1} - Slice {current_slice}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(img1, ax=ax1)
        
        # Plot second dataset
        img2 = ax2.imshow(data2[current_slice], cmap='viridis', aspect='equal')
        ax2.set_title(f'{dataset2} - Slice {current_slice}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(img2, ax=ax2)
        
        # Slider for slice selection
        ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
        max_slice = min(data1.shape[0], data2.shape[0]) - 1
        slider = Slider(
            ax_slider, 'Slice', 0, max_slice,
            valinit=current_slice, valstep=1
        )
        
        def update_slice(val):
            slice_idx = int(val)
            img1.set_array(data1[slice_idx])
            img2.set_array(data2[slice_idx])
            ax1.set_title(f'{dataset1} - Slice {slice_idx}')
            ax2.set_title(f'{dataset2} - Slice {slice_idx}')
            fig.canvas.draw_idle()
        
        slider.on_changed(update_slice)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison viewer to {save_path}")
        
        plt.show()
    
    def create_3d_viewer(self, dataset_name: str):
        """Create a 3D viewer using Napari."""
        if not NAPARI_AVAILABLE:
            logger.error("Napari not available for 3D visualization")
            return
        
        if dataset_name not in self.data:
            logger.error(f"Dataset {dataset_name} not found")
            return
        
        data = self.data[dataset_name]
        
        # Create Napari viewer
        viewer = napari.Viewer()
        
        # Add the data as a 3D image layer
        viewer.add_image(data, name=dataset_name)
        
        # Add segmentation layer if it's a segmentation dataset
        if 'segmentation' in dataset_name:
            # Create binary mask for visualization
            binary_mask = (data > 0.5).astype(np.uint8)
            viewer.add_labels(binary_mask, name=f"{dataset_name}_labels")
        
        logger.info(f"Opened 3D viewer for {dataset_name}")
        napari.run()
    
    def create_interactive_plot(self, dataset_name: str, save_path: Optional[str] = None):
        """Create an interactive plot using Plotly."""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for interactive visualization")
            return
        
        if dataset_name not in self.data:
            logger.error(f"Dataset {dataset_name} not found")
            return
        
        data = self.data[dataset_name]
        
        # Create subplots for different views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('XY View', 'XZ View', 'YZ View', '3D Surface'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "surface"}]]
        )
        
        # Middle slices for each view
        mid_z = data.shape[0] // 2
        mid_y = data.shape[1] // 2
        mid_x = data.shape[2] // 2
        
        # XY view (Z slice)
        fig.add_trace(
            go.Heatmap(z=data[mid_z], colorscale='viridis', name='XY'),
            row=1, col=1
        )
        
        # XZ view (Y slice)
        fig.add_trace(
            go.Heatmap(z=data[:, mid_y, :], colorscale='viridis', name='XZ'),
            row=1, col=2
        )
        
        # YZ view (X slice)
        fig.add_trace(
            go.Heatmap(z=data[:, :, mid_x], colorscale='viridis', name='YZ'),
            row=2, col=1
        )
        
        # 3D surface (downsampled for performance)
        if data.shape[0] <= 100:  # Only for smaller datasets
            fig.add_trace(
                go.Surface(z=data, colorscale='viridis', name='3D'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'Interactive Visualization: {dataset_name}',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive plot to {save_path}")
        
        fig.show()
    
    def create_quality_report(self, save_path: Optional[str] = None):
        """Create a comprehensive quality report."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for quality report")
            return
        
        # Find segmentation and uncertainty datasets
        seg_datasets = [k for k in self.data.keys() if 'segmentation' in k]
        unc_datasets = [k for k in self.data.keys() if 'uncertainty' in k]
        
        if not seg_datasets:
            logger.error("No segmentation datasets found for quality report")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('H01 Processing Quality Report', fontsize=16)
        
        # 1. Segmentation confidence distribution
        seg_data = self.data[seg_datasets[0]]
        axes[0, 0].hist(seg_data.flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Segmentation Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Uncertainty distribution
        if unc_datasets:
            unc_data = self.data[unc_datasets[0]]
            axes[0, 1].hist(unc_data.flatten(), bins=50, alpha=0.7, color='red')
            axes[0, 1].set_title('Uncertainty Distribution')
            axes[0, 1].set_xlabel('Uncertainty')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Segmentation volume statistics
        seg_stats = self.get_dataset_info(seg_datasets[0])
        axes[1, 0].bar(['Min', 'Mean', 'Max'], 
                      [seg_stats['min_value'], seg_stats['mean_value'], seg_stats['max_value']],
                      color=['red', 'blue', 'green'])
        axes[1, 0].set_title('Segmentation Statistics')
        axes[1, 0].set_ylabel('Value')
        
        # 4. Dataset sizes
        dataset_names = list(self.data.keys())
        dataset_sizes = [self.get_dataset_info(name)['size_mb'] for name in dataset_names]
        axes[1, 1].bar(range(len(dataset_names)), dataset_sizes)
        axes[1, 1].set_title('Dataset Sizes')
        axes[1, 1].set_xlabel('Dataset')
        axes[1, 1].set_ylabel('Size (MB)')
        axes[1, 1].set_xticks(range(len(dataset_names)))
        axes[1, 1].set_xticklabels(dataset_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved quality report to {save_path}")
        
        plt.show()
    
    def export_visualization_data(self, output_dir: str):
        """Export visualization data in various formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export dataset information
        dataset_info = {}
        for name in self.data.keys():
            dataset_info[name] = self.get_dataset_info(name)
        
        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Export sample slices as images
        if MATPLOTLIB_AVAILABLE:
            for name, data in self.data.items():
                # Save middle slice
                mid_slice = data.shape[0] // 2
                plt.figure(figsize=(10, 8))
                plt.imshow(data[mid_slice], cmap='viridis')
                plt.title(f'{name} - Middle Slice')
                plt.colorbar()
                plt.savefig(output_path / f'{name}_middle_slice.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Exported visualization data to {output_dir}")

def main():
    """Main entry point for visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="H01 Tracing Results Visualizer")
    parser.add_argument("results_dir", help="Directory containing H01 processing results")
    parser.add_argument("--dataset", help="Specific dataset to visualize")
    parser.add_argument("--viewer", choices=['2d', '3d', 'interactive', 'comparison', 'quality'], 
                       default='2d', help="Type of viewer to create")
    parser.add_argument("--dataset2", help="Second dataset for comparison")
    parser.add_argument("--save", help="Path to save visualization")
    parser.add_argument("--export", help="Directory to export visualization data")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = H01Visualizer(args.results_dir)
    
    # Show available datasets
    datasets = visualizer.get_available_datasets()
    print(f"Available datasets: {datasets}")
    
    if not datasets:
        print("No datasets found in the results directory")
        return
    
    # Create visualization based on arguments
    if args.viewer == '2d':
        dataset = args.dataset or datasets[0]
        visualizer.create_2d_slice_viewer(dataset, args.save)
    
    elif args.viewer == '3d':
        dataset = args.dataset or datasets[0]
        visualizer.create_3d_viewer(dataset)
    
    elif args.viewer == 'interactive':
        dataset = args.dataset or datasets[0]
        visualizer.create_interactive_plot(dataset, args.save)
    
    elif args.viewer == 'comparison':
        if not args.dataset2:
            print("Please specify --dataset2 for comparison")
            return
        visualizer.create_comparison_viewer(args.dataset, args.dataset2, args.save)
    
    elif args.viewer == 'quality':
        visualizer.create_quality_report(args.save)
    
    # Export data if requested
    if args.export:
        visualizer.export_visualization_data(args.export)

if __name__ == "__main__":
    main() 