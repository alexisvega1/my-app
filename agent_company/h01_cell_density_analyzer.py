#!/usr/bin/env python3
"""
H01 Cell Density Analyzer for Agentic Tracer
============================================
Adapted from H01 Matlab analysis scripts for cell density computation by layer.
Provides robust cell density analysis with support for multiple data formats.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.spatial.distance import cdist
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CellType:
    """Cell type definitions with colors and names."""
    id: int
    name: str
    color: Tuple[float, float, float]
    category: str  # 'neuron', 'glia', 'unknown'

@dataclass
class LayerBoundary:
    """Cortical layer boundary definition."""
    center_x: float
    center_y: float
    radius: float
    layer_name: str

@dataclass
class CellDensityConfig:
    """Configuration for cell density analysis."""
    # Data paths
    cell_matrix_path: str = ""
    layer_boundaries_path: str = ""
    mask_path: str = ""
    
    # Analysis parameters
    mip_level: int = 8
    voxel_size_nm: Tuple[float, float, float] = (8.0, 8.0, 33.0)
    section_compression_factor: float = 1.0  # Compensation for sectioning
    
    # Cell type mapping
    cell_type_mapping: Dict[int, int] = field(default_factory=dict)
    
    # Output settings
    output_dir: str = "./cell_density_output"
    save_plots: bool = True
    save_results: bool = True
    
    # Processing flags
    use_mask: bool = True
    normalize_by_volume: bool = True
    
    def __post_init__(self):
        # Default cell type mapping based on H01 analysis
        if not self.cell_type_mapping:
            self.cell_type_mapping = {
                8: 7,   # C->G (satellite MGs or OPCs)
                9: 0,   # B->remove (blood vessel cells)
                10: 2,  # S->P (group as excitatory neurons)
                11: 2   # E->P (group as excitatory neurons)
            }

class CellDensityAnalyzer:
    """
    Comprehensive cell density analyzer adapted from H01 Matlab scripts.
    
    Provides robust analysis of cell densities by cortical layer and cell type,
    with support for multiple data formats and comprehensive visualization.
    """
    
    # Cell type definitions based on H01 analysis
    CELL_TYPES = {
        0: CellType(0, "Unknown", (0.3, 0.3, 0.3), "unknown"),
        1: CellType(1, "Pyramidal/Spiny", (1.0, 0.0, 0.0), "neuron"),
        2: CellType(2, "Interneuron", (0.0, 0.7, 1.0), "neuron"),
        3: CellType(3, "Unclassified Neuron", (0.5, 0.5, 0.5), "neuron"),
        4: CellType(4, "Astrocyte", (1.0, 1.0, 0.0), "glia"),
        5: CellType(5, "Oligodendrocyte", (0.1, 0.1, 1.0), "glia"),
        6: CellType(6, "Microglia/OPC", (0.8, 0.0, 0.8), "glia"),
        7: CellType(7, "Satellite MG/OPC", (0.6, 0.3, 0.9), "glia")
    }
    
    # Layer names
    LAYER_NAMES = {
        1: "Layer 1",
        2: "Layer 2", 
        3: "Layer 3",
        4: "Layer 4",
        5: "Layer 5",
        6: "Layer 6",
        7: "White Matter"
    }
    
    def __init__(self, config: Optional[CellDensityConfig] = None):
        self.config = config or CellDensityConfig()
        
        # Data storage
        self.cell_data = None
        self.layer_boundaries = []
        self.mask_data = None
        self.layer_assignments = []
        self.cell_type_assignments = []
        
        # Results storage
        self.density_results = {}
        self.volume_results = {}
        self.statistics = {}
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info("CellDensityAnalyzer initialized")
    
    def load_cell_matrix(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load cell matrix data from various formats.
        
        Args:
            path: Path to cell matrix file
            
        Returns:
            DataFrame with cell data
        """
        path = path or self.config.cell_matrix_path
        
        if not path or not os.path.exists(path):
            logger.warning(f"Cell matrix file not found: {path}")
            return pd.DataFrame()
        
        logger.info(f"Loading cell matrix from: {path}")
        
        try:
            if path.endswith('.mat'):
                # Load MATLAB .mat file
                mat_data = loadmat(path)
                
                # Extract cell matrix (assuming it's called 'cmtx' or similar)
                if 'cmtx' in mat_data:
                    cell_matrix = mat_data['cmtx']
                elif 'cellmatrix' in mat_data:
                    cell_matrix = mat_data['cellmatrix']
                else:
                    # Try to find any 2D array
                    for key, value in mat_data.items():
                        if isinstance(value, np.ndarray) and value.ndim == 2:
                            cell_matrix = value
                            break
                    else:
                        raise ValueError("No suitable cell matrix found in .mat file")
                
                # Convert to DataFrame
                columns = ['id', 'valid', 'x', 'y', 'z', 'volume', 'type', 'classification']
                if cell_matrix.shape[1] >= len(columns):
                    df = pd.DataFrame(cell_matrix[:, :len(columns)], columns=columns)
                else:
                    df = pd.DataFrame(cell_matrix)
                
            elif path.endswith('.csv'):
                # Load CSV file
                df = pd.read_csv(path)
                
            elif path.endswith('.json'):
                # Load JSON file
                with open(path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            # Clean and validate data
            df = self._clean_cell_data(df)
            
            logger.info(f"Loaded {len(df)} cells from {path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load cell matrix from {path}: {e}")
            return pd.DataFrame()
    
    def load_layer_boundaries(self, path: Optional[str] = None) -> List[LayerBoundary]:
        """
        Load layer boundary definitions.
        
        Args:
            path: Path to layer boundaries file
            
        Returns:
            List of LayerBoundary objects
        """
        path = path or self.config.layer_boundaries_path
        
        if not path or not os.path.exists(path):
            logger.warning(f"Layer boundaries file not found: {path}")
            # Use default H01 layer boundaries
            boundaries = self._get_default_layer_boundaries()
            logger.info(f"Using default layer boundaries: {len(boundaries)} boundaries")
            return boundaries
        
        logger.info(f"Loading layer boundaries from: {path}")
        
        try:
            if path.endswith('.mat'):
                # Load MATLAB .mat file
                mat_data = loadmat(path)
                coeff = mat_data.get('coeff', [])
                
            elif path.endswith('.csv'):
                # Load CSV file
                coeff = pd.read_csv(path).values
                
            elif path.endswith('.json'):
                # Load JSON file
                with open(path, 'r') as f:
                    data = json.load(f)
                coeff = np.array(data)
                
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            # Convert to LayerBoundary objects
            boundaries = []
            for i, row in enumerate(coeff):
                if len(row) >= 3:
                    layer_name = self.LAYER_NAMES.get(i + 1, f"Layer {i + 1}")
                    boundary = LayerBoundary(
                        center_x=float(row[0]),
                        center_y=float(row[1]),
                        radius=float(row[2]),
                        layer_name=layer_name
                    )
                    boundaries.append(boundary)
            
            logger.info(f"Loaded {len(boundaries)} layer boundaries")
            return boundaries
            
        except Exception as e:
            logger.error(f"Failed to load layer boundaries from {path}: {e}")
            # Fall back to default boundaries
            boundaries = self._get_default_layer_boundaries()
            logger.info(f"Using default layer boundaries: {len(boundaries)} boundaries")
            return boundaries
    
    def load_mask_data(self, path: Optional[str] = None) -> np.ndarray:
        """
        Load mask data for volume calculations.
        
        Args:
            path: Path to mask file
            
        Returns:
            Mask array
        """
        path = path or self.config.mask_path
        
        if not path or not os.path.exists(path):
            logger.warning(f"Mask file not found: {path}")
            return None
        
        logger.info(f"Loading mask data from: {path}")
        
        try:
            if path.endswith('.mat'):
                # Load MATLAB .mat file
                mat_data = loadmat(path)
                mask = mat_data.get('donemask', mat_data.get('mask', None))
                
            elif path.endswith('.csv'):
                # Load CSV file
                mask = pd.read_csv(path).values
                
            elif path.endswith('.npy'):
                # Load NumPy array
                mask = np.load(path)
                
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            logger.info(f"Loaded mask with shape: {mask.shape}")
            return mask
            
        except Exception as e:
            logger.error(f"Failed to load mask from {path}: {e}")
            return None
    
    def _clean_cell_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate cell data."""
        # Remove invalid cells
        if 'valid' in df.columns:
            df = df[df['valid'] > 0].copy()
        
        # Convert coordinates to micrometers
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            # Apply voxel size scaling
            voxel_size = self.config.voxel_size_nm
            df['x_um'] = df['x'] * voxel_size[0] / 1000
            df['y_um'] = df['y'] * voxel_size[1] / 1000
            df['z_um'] = df['z'] * voxel_size[2] / 1000
        
        # Apply cell type mapping
        if 'type' in df.columns:
            df['mapped_type'] = df['type'].map(self.config.cell_type_mapping).fillna(df['type'])
        
        return df
    
    def _get_default_layer_boundaries(self) -> List[LayerBoundary]:
        """Get default H01 layer boundaries."""
        # Default H01 layer boundaries (from Matlab script)
        coeff = [
            [462.2683543371059, 2805.973374970087, 1416.3922980940845],
            [462.2683543371059, 2805.973374970087, 1731.0819866303507],
            [1149.9313666484356, 2458.417758523231, 1476.134832192077],
            [1149.9313666484356, 2458.417758523231, 1692.4091828120922],
            [1158.1594071389497, 2411.7378929961988, 2135.2674700433477],
            [1158.1594071389497, 2411.7378929961988, 2479.253415996829]
        ]
        
        boundaries = []
        for i, (cx, cy, r) in enumerate(coeff):
            layer_name = self.LAYER_NAMES.get(i + 1, f"Layer {i + 1}")
            boundary = LayerBoundary(
                center_x=cx,
                center_y=cy,
                radius=r,
                layer_name=layer_name
            )
            boundaries.append(boundary)
        
        return boundaries
    
    def assign_cells_to_layers(self, cell_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Assign cells to cortical layers based on geometric boundaries.
        
        Args:
            cell_data: Cell data DataFrame
            
        Returns:
            Array of layer assignments
        """
        if cell_data is None:
            cell_data = self.cell_data
        
        if cell_data.empty:
            logger.warning("No cell data available for layer assignment")
            return np.array([])
        
        # Ensure we have layer boundaries (use defaults if none loaded)
        if not self.layer_boundaries:
            logger.info("No layer boundaries loaded, using defaults")
            self.layer_boundaries = self._get_default_layer_boundaries()
        
        logger.info("Assigning cells to layers...")
        
        # Extract cell coordinates
        if 'x_um' not in cell_data.columns or 'y_um' not in cell_data.columns:
            logger.error("Cell coordinates not available")
            return np.array([])
        
        cell_coords = cell_data[['x_um', 'y_um']].values
        layer_assignments = np.zeros(len(cell_coords), dtype=int)
        
        # Assign each cell to the appropriate layer
        for i, (x, y) in enumerate(cell_coords):
            for j, boundary in enumerate(self.layer_boundaries):
                distance = np.sqrt((x - boundary.center_x)**2 + (y - boundary.center_y)**2)
                if distance <= boundary.radius:
                    layer_assignments[i] = j + 1
                    break
        
        # Count assignments
        unique_layers, counts = np.unique(layer_assignments, return_counts=True)
        for layer_id, count in zip(unique_layers, counts):
            if layer_id > 0:
                layer_name = self.LAYER_NAMES.get(layer_id, f"Layer {layer_id}")
                logger.info(f"Assigned {count} cells to {layer_name}")
        
        return layer_assignments
    
    def calculate_cell_densities(self) -> Dict[str, Any]:
        """
        Calculate cell densities by layer and cell type.
        
        Returns:
            Dictionary with density results
        """
        if self.cell_data is None or self.cell_data.empty:
            logger.error("No cell data available for density calculation")
            return {}
        
        logger.info("Calculating cell densities...")
        
        # Get layer assignments
        layer_assignments = self.assign_cells_to_layers()
        if len(layer_assignments) == 0:
            return {}
        
        # Get cell types
        cell_types = self.cell_data.get('mapped_type', self.cell_data.get('type', np.zeros(len(self.cell_data))))
        
        # Initialize results
        num_layers = len(self.LAYER_NAMES)
        num_cell_types = len(self.CELL_TYPES)
        
        # Count cells by layer and type
        cell_counts = np.zeros((num_layers, num_cell_types))
        
        for i, (layer_id, cell_type) in enumerate(zip(layer_assignments, cell_types)):
            if layer_id > 0 and cell_type < num_cell_types:
                cell_counts[layer_id - 1, int(cell_type)] += 1
        
        # Calculate volumes if mask is available
        layer_volumes = self._calculate_layer_volumes()
        
        # Calculate densities
        if layer_volumes is not None and self.config.normalize_by_volume:
            densities = cell_counts / layer_volumes[:, np.newaxis]
            # Convert to cells per mm³
            densities *= 1e9  # Convert from μm³ to mm³
        else:
            densities = cell_counts
            layer_volumes = np.ones(num_layers)
        
        # Store results
        self.density_results = {
            'cell_counts': cell_counts,
            'densities': densities,
            'layer_volumes': layer_volumes,
            'layer_assignments': layer_assignments,
            'cell_types': cell_types
        }
        
        # Calculate statistics
        self._calculate_statistics()
        
        logger.info("Cell density calculation completed")
        return self.density_results
    
    def _calculate_layer_volumes(self) -> Optional[np.ndarray]:
        """Calculate layer volumes from mask data."""
        if self.mask_data is None:
            logger.warning("No mask data available for volume calculation")
            return None
        
        logger.info("Calculating layer volumes...")
        
        # This is a simplified volume calculation
        # In practice, you would use the actual layer segmentation data
        
        # For now, assume equal volumes or use mask data
        if self.mask_data is not None:
            total_voxels = np.sum(self.mask_data > 0)
            voxel_volume_nm3 = np.prod(self.config.voxel_size_nm)
            total_volume_mm3 = total_voxels * voxel_volume_nm3 / 1e9
            
            # Distribute volume across layers (simplified)
            num_layers = len(self.LAYER_NAMES)
            layer_volumes = np.full(num_layers, total_volume_mm3 / num_layers)
        else:
            layer_volumes = np.ones(len(self.LAYER_NAMES))
        
        return layer_volumes
    
    def _calculate_statistics(self) -> None:
        """Calculate additional statistics."""
        if not self.density_results:
            return
        
        densities = self.density_results['densities']
        cell_counts = self.density_results['cell_counts']
        
        # Calculate total densities per layer
        total_densities = np.sum(densities, axis=1)
        
        # Calculate excitatory vs inhibitory ratios
        excitatory_types = [1, 3]  # Pyramidal, Unclassified neurons
        inhibitory_types = [2]     # Interneurons
        
        excitatory_densities = np.sum(densities[:, excitatory_types], axis=1)
        inhibitory_densities = np.sum(densities[:, inhibitory_types], axis=1)
        
        # Calculate neuron vs glia ratios
        neuron_types = [1, 2, 3]  # All neurons
        glia_types = [4, 5, 6, 7]  # All glia
        
        neuron_densities = np.sum(densities[:, neuron_types], axis=1)
        glia_densities = np.sum(densities[:, glia_types], axis=1)
        
        self.statistics = {
            'total_densities': total_densities,
            'excitatory_densities': excitatory_densities,
            'inhibitory_densities': inhibitory_densities,
            'neuron_densities': neuron_densities,
            'glia_densities': glia_densities,
            'ei_ratios': excitatory_densities / (excitatory_densities + inhibitory_densities),
            'neuron_glia_ratios': neuron_densities / (neuron_densities + glia_densities)
        }
    
    def create_visualizations(self) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualizations of cell density results.
        
        Returns:
            Dictionary of matplotlib figures
        """
        if not self.density_results:
            logger.warning("No density results available for visualization")
            return {}
        
        logger.info("Creating visualizations...")
        
        figures = {}
        densities = self.density_results['densities']
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Stacked bar chart of cell densities by type
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        layer_names = list(self.LAYER_NAMES.values())
        
        # Prepare data for stacked bar chart
        cell_type_names = [self.CELL_TYPES[i].name for i in range(len(self.CELL_TYPES))]
        cell_type_colors = [self.CELL_TYPES[i].color for i in range(len(self.CELL_TYPES))]
        
        bottom = np.zeros(len(layer_names))
        for i in range(len(self.CELL_TYPES)):
            ax1.bar(layer_names, densities[:, i], bottom=bottom, 
                   label=cell_type_names[i], color=cell_type_colors[i])
            bottom += densities[:, i]
        
        ax1.set_xlabel('Cortical Layer')
        ax1.set_ylabel('Cell Density (cells/mm³)')
        ax1.set_title('Cell Densities by Layer and Type')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        figures['cell_densities_by_type'] = fig1
        
        # 2. Excitatory vs Inhibitory ratio
        if 'ei_ratios' in self.statistics:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ei_ratios = self.statistics['ei_ratios']
            
            ax2.bar(layer_names, ei_ratios, color='orange', alpha=0.7)
            ax2.set_xlabel('Cortical Layer')
            ax2.set_ylabel('Excitatory/Inhibitory Ratio')
            ax2.set_title('Excitatory vs Inhibitory Neuron Ratio')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Typical Ratio')
            ax2.legend()
            
            figures['ei_ratio'] = fig2
        
        # 3. Neuron vs Glia ratio
        if 'neuron_glia_ratios' in self.statistics:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            neuron_glia_ratios = self.statistics['neuron_glia_ratios']
            
            ax3.bar(layer_names, neuron_glia_ratios, color='purple', alpha=0.7)
            ax3.set_xlabel('Cortical Layer')
            ax3.set_ylabel('Neuron/Glia Ratio')
            ax3.set_title('Neuron vs Glia Ratio')
            ax3.tick_params(axis='x', rotation=45)
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Typical Ratio')
            ax3.legend()
            
            figures['neuron_glia_ratio'] = fig3
        
        # 4. Total cell density by layer
        if 'total_densities' in self.statistics:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            total_densities = self.statistics['total_densities']
            
            ax4.bar(layer_names, total_densities, color='blue', alpha=0.7)
            ax4.set_xlabel('Cortical Layer')
            ax4.set_ylabel('Total Cell Density (cells/mm³)')
            ax4.set_title('Total Cell Density by Layer')
            ax4.tick_params(axis='x', rotation=45)
            
            figures['total_density'] = fig4
        
        logger.info(f"Created {len(figures)} visualizations")
        return figures
    
    def save_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save analysis results to files.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary of saved file paths
        """
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save density results
        if self.density_results:
            results_file = os.path.join(output_dir, 'cell_density_results.json')
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in self.density_results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            saved_files['density_results'] = results_file
        
        # Save statistics
        if self.statistics:
            stats_file = os.path.join(output_dir, 'cell_density_statistics.json')
            
            json_stats = {}
            for key, value in self.statistics.items():
                if isinstance(value, np.ndarray):
                    json_stats[key] = value.tolist()
                else:
                    json_stats[key] = value
            
            with open(stats_file, 'w') as f:
                json.dump(json_stats, f, indent=2)
            
            saved_files['statistics'] = stats_file
        
        # Save visualizations
        if self.config.save_plots:
            figures = self.create_visualizations()
            for name, fig in figures.items():
                plot_file = os.path.join(output_dir, f'{name}.png')
                fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files[f'plot_{name}'] = plot_file
        
        # Save summary report
        summary_file = os.path.join(output_dir, 'analysis_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("H01 Cell Density Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            if self.cell_data is not None:
                f.write(f"Total cells analyzed: {len(self.cell_data)}\n")
            
            if self.density_results:
                f.write(f"Layers analyzed: {len(self.LAYER_NAMES)}\n")
                f.write(f"Cell types analyzed: {len(self.CELL_TYPES)}\n")
            
            if self.statistics:
                f.write(f"\nKey Statistics:\n")
                f.write(f"Average total density: {np.mean(self.statistics['total_densities']):.2f} cells/mm³\n")
                f.write(f"Average E/I ratio: {np.mean(self.statistics['ei_ratios']):.2f}\n")
                f.write(f"Average neuron/glia ratio: {np.mean(self.statistics['neuron_glia_ratios']):.2f}\n")
        
        saved_files['summary'] = summary_file
        
        logger.info(f"Saved {len(saved_files)} files to {output_dir}")
        return saved_files
    
    def run_analysis(self, 
                    cell_matrix_path: Optional[str] = None,
                    layer_boundaries_path: Optional[str] = None,
                    mask_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete cell density analysis.
        
        Args:
            cell_matrix_path: Path to cell matrix file
            layer_boundaries_path: Path to layer boundaries file
            mask_path: Path to mask file
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting cell density analysis...")
        
        # Load data
        self.cell_data = self.load_cell_matrix(cell_matrix_path)
        self.layer_boundaries = self.load_layer_boundaries(layer_boundaries_path)
        self.mask_data = self.load_mask_data(mask_path)
        
        # Run analysis
        density_results = self.calculate_cell_densities()
        
        # Save results
        if self.config.save_results:
            saved_files = self.save_results()
            density_results['saved_files'] = saved_files
        
        logger.info("Cell density analysis completed")
        return density_results

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create analyzer
    config = CellDensityConfig(
        output_dir="./cell_density_output",
        save_plots=True,
        save_results=True
    )
    
    analyzer = CellDensityAnalyzer(config)
    
    # Run analysis (with example paths)
    # results = analyzer.run_analysis(
    #     cell_matrix_path="path/to/cellmatrix.mat",
    #     layer_boundaries_path="path/to/layer_boundaries.mat",
    #     mask_path="path/to/mask.mat"
    # )
    
    print("CellDensityAnalyzer ready for use with H01 data") 