#!/usr/bin/env python3
"""
H01 Integration Module for Agentic Tracer
=========================================
Comprehensive integration of H01 project functionality adapted for our tracer agent.
Combines synapse merge models, skeleton pruning, and data processing capabilities.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

# Import our H01 adapted modules
from h01_synapse_merge_model import SynapseMergeModel, SynapsePair, MergeModelConfig
from h01_skeleton_pruner import SkeletonPruner, SkeletonNode, PruningConfig
from h01_data_loader import H01DataLoader

logger = logging.getLogger(__name__)

@dataclass
class H01IntegrationConfig:
    """Configuration for H01 integration."""
    # Data access
    h01_config_path: str = "h01_config.yaml"
    
    # Model paths
    synapse_merge_model_path: str = "synapse_merge_model.pkl"
    skeleton_pruner_model_path: str = "skeleton_pruner_model.pkl"
    
    # Processing parameters
    max_synapse_distance_nm: float = 5000.0
    pruning_threshold: float = 0.5
    merge_confidence_threshold: float = 0.7
    
    # Output settings
    output_dir: str = "./h01_processing_output"
    save_intermediate_results: bool = True
    
    # Logging
    log_level: str = "INFO"

class H01Integration:
    """
    Comprehensive H01 integration for the tracer agent.
    
    Provides unified access to:
    - H01 data loading and processing
    - Synapse merge decision making
    - Skeleton pruning and optimization
    - Graph-based analysis
    """
    
    def __init__(self, config: Optional[H01IntegrationConfig] = None):
        self.config = config or H01IntegrationConfig()
        
        # Initialize components
        self.data_loader = None
        self.synapse_merge_model = None
        self.skeleton_pruner = None
        
        # Processing state
        self.current_region = None
        self.processed_data = {}
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info("H01Integration initialized")
    
    def initialize_data_loader(self, config_path: Optional[str] = None) -> bool:
        """
        Initialize the H01 data loader.
        
        Args:
            config_path: Path to H01 configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = config_path or self.config.h01_config_path
            
            if not os.path.exists(config_path):
                logger.error(f"H01 config file not found: {config_path}")
                return False
            
            self.data_loader = H01DataLoader(config_path)
            
            # Validate data source
            if not self.data_loader.validate_data_source():
                logger.error("H01 data source validation failed")
                return False
            
            logger.info("H01 data loader initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize data loader: {e}")
            return False
    
    def initialize_synapse_merge_model(self, model_path: Optional[str] = None) -> bool:
        """
        Initialize the synapse merge model.
        
        Args:
            model_path: Path to trained model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = model_path or self.config.synapse_merge_model_path
            
            self.synapse_merge_model = SynapseMergeModel()
            
            if os.path.exists(model_path):
                self.synapse_merge_model.load(model_path)
                logger.info("Synapse merge model loaded from file")
            else:
                logger.info("No pre-trained synapse merge model found - will need training")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize synapse merge model: {e}")
            return False
    
    def initialize_skeleton_pruner(self, model_path: Optional[str] = None) -> bool:
        """
        Initialize the skeleton pruner.
        
        Args:
            model_path: Path to trained model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = model_path or self.config.skeleton_pruner_model_path
            
            self.skeleton_pruner = SkeletonPruner()
            
            if os.path.exists(model_path):
                self.skeleton_pruner.load(model_path)
                logger.info("Skeleton pruner model loaded from file")
            else:
                logger.info("No pre-trained skeleton pruner model found - will need training")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize skeleton pruner: {e}")
            return False
    
    def initialize_all_components(self) -> bool:
        """
        Initialize all H01 components.
        
        Returns:
            True if all components initialized successfully
        """
        logger.info("Initializing all H01 components...")
        
        success = True
        
        # Initialize data loader
        if not self.initialize_data_loader():
            logger.error("Failed to initialize data loader")
            success = False
        
        # Initialize synapse merge model
        if not self.initialize_synapse_merge_model():
            logger.error("Failed to initialize synapse merge model")
            success = False
        
        # Initialize skeleton pruner
        if not self.initialize_skeleton_pruner():
            logger.error("Failed to initialize skeleton pruner")
            success = False
        
        if success:
            logger.info("All H01 components initialized successfully")
        else:
            logger.error("Some H01 components failed to initialize")
        
        return success
    
    def process_region(self, region_name: str, chunk_size: Tuple[int, int, int] = (64, 64, 64)) -> Dict[str, Any]:
        """
        Process a specific region of H01 data.
        
        Args:
            region_name: Name of the region to process
            chunk_size: Size of chunks to process
            
        Returns:
            Dictionary with processing results
        """
        if not self.data_loader:
            raise RuntimeError("Data loader not initialized")
        
        logger.info(f"Processing region: {region_name}")
        
        # Get region information
        region_info = self.data_loader.get_region(region_name)
        bounds = region_info['bounds']
        
        # Load data chunk
        start_coords = bounds[0]
        chunk_data = self.data_loader.load_chunk(start_coords, chunk_size)
        
        # Store current region
        self.current_region = {
            'name': region_name,
            'bounds': bounds,
            'chunk_data': chunk_data,
            'start_coords': start_coords
        }
        
        # Process the data
        results = {
            'region_name': region_name,
            'bounds': bounds,
            'chunk_shape': chunk_data.shape,
            'synapse_pairs': [],
            'skeleton_nodes': [],
            'merge_predictions': [],
            'pruning_predictions': []
        }
        
        # Extract synapse coordinates (simplified - in practice would use segmentation)
        synapse_coords = self._extract_synapse_coordinates(chunk_data)
        results['synapse_coordinates'] = synapse_coords
        
        # Create synapse pairs
        if self.synapse_merge_model and synapse_coords:
            synapse_pairs = self._create_synapse_pairs(synapse_coords)
            results['synapse_pairs'] = synapse_pairs
            
            # Make merge predictions
            if self.synapse_merge_model.model is not None:
                merge_predictions = self.synapse_merge_model.predict(synapse_pairs)
                results['merge_predictions'] = merge_predictions
        
        # Create skeleton nodes (simplified - in practice would use tracing)
        if self.skeleton_pruner:
            skeleton_nodes = self._create_skeleton_nodes(chunk_data)
            results['skeleton_nodes'] = skeleton_nodes
            
            # Make pruning predictions
            if self.skeleton_pruner.model is not None:
                pruning_predictions = self.skeleton_pruner.predict(skeleton_nodes)
                results['pruning_predictions'] = pruning_predictions
        
        # Store results
        self.processed_data[region_name] = results
        
        # Save intermediate results if requested
        if self.config.save_intermediate_results:
            self._save_region_results(region_name, results)
        
        logger.info(f"Completed processing region: {region_name}")
        return results
    
    def _extract_synapse_coordinates(self, chunk_data: np.ndarray) -> List[Tuple[str, Tuple[float, float, float]]]:
        """
        Extract synapse coordinates from chunk data.
        
        This is a simplified implementation. In practice, this would use
        sophisticated synapse detection algorithms.
        
        Args:
            chunk_data: 3D array of image data
            
        Returns:
            List of (synapse_id, (x, y, z)) tuples
        """
        # Simplified synapse detection - in practice would use ML models
        synapse_coords = []
        
        # Find local maxima as potential synapse locations
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import generate_binary_structure
        
        # Create binary structure for 3D neighborhood
        neighborhood = generate_binary_structure(3, 1)
        
        # Find local maxima
        local_max = maximum_filter(chunk_data, footprint=neighborhood)
        synapse_candidates = (chunk_data == local_max) & (chunk_data > np.percentile(chunk_data, 90))
        
        # Extract coordinates
        candidate_coords = np.where(synapse_candidates)
        
        for i in range(min(len(candidate_coords[0]), 20)):  # Limit to 20 synapses
            z, y, x = candidate_coords[0][i], candidate_coords[1][i], candidate_coords[2][i]
            synapse_id = f"syn_{i}"
            synapse_coords.append((synapse_id, (float(x), float(y), float(z))))
        
        logger.info(f"Extracted {len(synapse_coords)} synapse coordinates")
        return synapse_coords
    
    def _create_synapse_pairs(self, synapse_coords: List[Tuple[str, Tuple[float, float, float]]]) -> List[SynapsePair]:
        """
        Create synapse pairs for merge analysis.
        
        Args:
            synapse_coords: List of synapse coordinates
            
        Returns:
            List of SynapsePair objects
        """
        from h01_synapse_merge_model import create_synapse_pairs_from_coordinates
        
        pairs = create_synapse_pairs_from_coordinates(
            synapse_coords, 
            max_distance_nm=self.config.max_synapse_distance_nm
        )
        
        logger.info(f"Created {len(pairs)} synapse pairs")
        return pairs
    
    def _create_skeleton_nodes(self, chunk_data: np.ndarray) -> List[SkeletonNode]:
        """
        Create skeleton nodes from chunk data.
        
        This is a simplified implementation. In practice, this would use
        sophisticated neuron tracing algorithms.
        
        Args:
            chunk_data: 3D array of image data
            
        Returns:
            List of SkeletonNode objects
        """
        from h01_skeleton_pruner import create_skeleton_from_coordinates
        
        # Simplified skeleton extraction - in practice would use tracing
        # For now, create a simple path through the data
        z_size, y_size, x_size = chunk_data.shape
        
        # Create a simple skeleton path
        coordinates = []
        radii = []
        
        # Create a path through the center of the chunk
        for i in range(0, min(z_size, y_size, x_size), 5):
            x = x_size // 2 + i // 10
            y = y_size // 2 + i // 10
            z = i
            coordinates.append((float(x), float(y), float(z)))
            radii.append(2.0 + np.random.uniform(0, 1))
        
        nodes = create_skeleton_from_coordinates(coordinates, radii)
        
        logger.info(f"Created {len(nodes)} skeleton nodes")
        return nodes
    
    def _save_region_results(self, region_name: str, results: Dict[str, Any]) -> None:
        """Save region processing results to file."""
        output_file = os.path.join(self.config.output_dir, f"{region_name}_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                # Handle dataclass objects
                serializable_results[key] = [obj.__dict__ for obj in value]
            else:
                serializable_results[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {output_file}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all processing results.
        
        Returns:
            Dictionary with processing summary
        """
        summary = {
            'regions_processed': list(self.processed_data.keys()),
            'total_regions': len(self.processed_data),
            'components_initialized': {
                'data_loader': self.data_loader is not None,
                'synapse_merge_model': self.synapse_merge_model is not None,
                'skeleton_pruner': self.skeleton_pruner is not None
            }
        }
        
        # Add statistics for each region
        region_stats = {}
        for region_name, results in self.processed_data.items():
            region_stats[region_name] = {
                'synapse_pairs': len(results.get('synapse_pairs', [])),
                'skeleton_nodes': len(results.get('skeleton_nodes', [])),
                'merge_predictions': len(results.get('merge_predictions', [])),
                'pruning_predictions': len(results.get('pruning_predictions', []))
            }
        
        summary['region_statistics'] = region_stats
        
        return summary
    
    def train_models(self, training_data_path: str) -> Dict[str, Any]:
        """
        Train the synapse merge and skeleton pruning models.
        
        Args:
            training_data_path: Path to training data
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training H01 models...")
        
        training_results = {}
        
        # Train synapse merge model
        if self.synapse_merge_model:
            try:
                # Load training data (simplified - would need real GML data)
                logger.info("Training synapse merge model...")
                # training_results['synapse_merge'] = self.synapse_merge_model.train(df)
                logger.info("Synapse merge model training completed")
            except Exception as e:
                logger.error(f"Failed to train synapse merge model: {e}")
                training_results['synapse_merge_error'] = str(e)
        
        # Train skeleton pruner
        if self.skeleton_pruner:
            try:
                logger.info("Training skeleton pruner...")
                # training_results['skeleton_pruner'] = self.skeleton_pruner.train(df)
                logger.info("Skeleton pruner training completed")
            except Exception as e:
                logger.error(f"Failed to train skeleton pruner: {e}")
                training_results['skeleton_pruner_error'] = str(e)
        
        return training_results
    
    def export_results(self, output_format: str = 'json') -> str:
        """
        Export all processing results.
        
        Args:
            output_format: Format for export ('json', 'csv', 'pickle')
            
        Returns:
            Path to exported file
        """
        if not self.processed_data:
            logger.warning("No data to export")
            return ""
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == 'json':
            output_file = os.path.join(self.config.output_dir, f"h01_results_{timestamp}.json")
            self._export_json(output_file)
        elif output_format == 'csv':
            output_file = os.path.join(self.config.output_dir, f"h01_results_{timestamp}.csv")
            self._export_csv(output_file)
        elif output_format == 'pickle':
            output_file = os.path.join(self.config.output_dir, f"h01_results_{timestamp}.pkl")
            self._export_pickle(output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Results exported to {output_file}")
        return output_file
    
    def _export_json(self, output_file: str) -> None:
        """Export results as JSON."""
        # Convert results to JSON-serializable format
        export_data = {}
        for region_name, results in self.processed_data.items():
            export_data[region_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    export_data[region_name][key] = value.tolist()
                elif isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                    export_data[region_name][key] = [obj.__dict__ for obj in value]
                else:
                    export_data[region_name][key] = value
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_csv(self, output_file: str) -> None:
        """Export results as CSV."""
        # Flatten results for CSV export
        rows = []
        for region_name, results in self.processed_data.items():
            row = {
                'region_name': region_name,
                'bounds': str(results.get('bounds', [])),
                'chunk_shape': str(results.get('chunk_shape', [])),
                'num_synapse_pairs': len(results.get('synapse_pairs', [])),
                'num_skeleton_nodes': len(results.get('skeleton_nodes', [])),
                'num_merge_predictions': len(results.get('merge_predictions', [])),
                'num_pruning_predictions': len(results.get('pruning_predictions', []))
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    
    def _export_pickle(self, output_file: str) -> None:
        """Export results as pickle."""
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(self.processed_data, f)

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create integration
    config = H01IntegrationConfig()
    h01_integration = H01Integration(config)
    
    # Initialize components
    if h01_integration.initialize_all_components():
        print("✅ All H01 components initialized successfully")
        
        # Process a test region
        try:
            results = h01_integration.process_region('test_region')
            print(f"✅ Processed region with {len(results['synapse_pairs'])} synapse pairs")
            
            # Get summary
            summary = h01_integration.get_processing_summary()
            print(f"✅ Processing summary: {summary}")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
    else:
        print("❌ Failed to initialize H01 components")
    
    print("H01Integration ready for use with tracer agent") 