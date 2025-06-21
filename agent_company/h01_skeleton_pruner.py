#!/usr/bin/env python3
"""
H01 Skeleton Pruning Model for Agentic Tracer
=============================================
Adapted from H01 project skeleton pruning functionality.
Provides intelligent skeleton pruning for neuron reconstruction.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict

# ML imports
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean

# Graph imports
try:
    import networkx as nx
    from networkx.readwrite.gml import literal_destringizer as destringizer
    from networkx import read_gml
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available - graph functionality limited")

logger = logging.getLogger(__name__)

@dataclass
class SkeletonNode:
    """Represents a skeleton node with features."""
    node_id: str
    coordinates: Tuple[float, float, float]
    radius: float
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    synapse_distances: List[float] = None
    branch_length: float = 0.0
    stalk_length: float = 0.0
    is_terminal: bool = False
    is_branch_point: bool = False
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.synapse_distances is None:
            self.synapse_distances = []

@dataclass
class PruningConfig:
    """Configuration for skeleton pruning."""
    # Threshold ranges for optimization
    max_stalk_len_range: range = range(16, 31, 4)  # In nm
    lower_min_range: range = range(5, 9, 1)        # In nm  
    higher_min_range: range = range(16, 31, 4)     # In nm
    min_branch_len_range: range = range(0, 10, 2)  # In nm
    
    # Model parameters
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    max_iter: int = 1000
    
    # Feature engineering
    use_synapse_features: bool = True
    use_topology_features: bool = True
    use_morphology_features: bool = True
    
    # Pruning parameters
    nodes_or_lengths: str = 'nodes'  # 'nodes' or 'lengths'
    neurite_type: str = 'axon'       # 'axon' or 'dendrite'
    
    # Model persistence
    model_save_path: str = "skeleton_pruner_model.pkl"
    cache_dir: str = "./pruner_model_cache"

class SkeletonPruner:
    """
    Skeleton pruning model based on H01 project.
    
    Provides intelligent pruning of neuron skeletons based on ML predictions
    and morphological features.
    """
    
    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()
        self.model = None
        self.best_params = {}
        self.feature_names = []
        self.metrics = {}
        
        # Ensure cache directory exists
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        logger.info("SkeletonPruner initialized")
    
    def extract_skeleton_features(self, 
                                 skeleton_nodes: List[SkeletonNode],
                                 synapse_coords: List[Tuple[float, float, float]]) -> pd.DataFrame:
        """
        Extract features from skeleton nodes and synapse data.
        
        Args:
            skeleton_nodes: List of SkeletonNode objects
            synapse_coords: List of synapse coordinates
            
        Returns:
            DataFrame with skeleton features
        """
        logger.info(f"Extracting features from {len(skeleton_nodes)} skeleton nodes")
        
        features = []
        
        for node in skeleton_nodes:
            # Calculate distances to synapses
            synapse_dists = []
            for syn_coord in synapse_coords:
                dist = euclidean(node.coordinates, syn_coord)
                synapse_dists.append(dist)
            
            # Sort distances and get closest synapses
            synapse_dists.sort()
            closest_syn_dist = synapse_dists[0] if synapse_dists else float('inf')
            second_closest_dist = synapse_dists[1] if len(synapse_dists) > 1 else float('inf')
            
            # Topology features
            num_children = len(node.children_ids)
            is_terminal = num_children == 0
            is_branch_point = num_children > 1
            
            # Morphology features
            node_volume = (4/3) * np.pi * (node.radius ** 3)
            
            # Create feature vector
            feature_vector = {
                'node_id': node.node_id,
                'radius': node.radius,
                'volume': node_volume,
                'closest_synapse_dist': closest_syn_dist,
                'second_closest_synapse_dist': second_closest_dist,
                'num_children': num_children,
                'is_terminal': is_terminal,
                'is_branch_point': is_branch_point,
                'branch_length': node.branch_length,
                'stalk_length': node.stalk_length,
                'parent_radius': 0.0,  # Will be filled if parent exists
                'child_radius_sum': 0.0,  # Will be filled if children exist
                'radius_ratio_parent': 1.0,
                'radius_ratio_children': 1.0
            }
            
            features.append(feature_vector)
        
        # Add parent/child radius features
        node_dict = {node.node_id: node for node in skeleton_nodes}
        
        for feature in features:
            node_id = feature['node_id']
            node = node_dict[node_id]
            
            # Parent radius
            if node.parent_id and node.parent_id in node_dict:
                parent = node_dict[node.parent_id]
                feature['parent_radius'] = parent.radius
                feature['radius_ratio_parent'] = node.radius / parent.radius if parent.radius > 0 else 1.0
            
            # Children radius
            if node.children_ids:
                child_radii = [node_dict[child_id].radius for child_id in node.children_ids if child_id in node_dict]
                if child_radii:
                    feature['child_radius_sum'] = sum(child_radii)
                    feature['radius_ratio_children'] = node.radius / (sum(child_radii) / len(child_radii)) if child_radii else 1.0
        
        df = pd.DataFrame(features)
        logger.info(f"Extracted features for {len(df)} nodes")
        
        return df
    
    def extract_features_from_gml(self, 
                                 gml_files: List[str],
                                 seg_list_file: str,
                                 synapse_db_name: str = None) -> pd.DataFrame:
        """
        Extract features from GML files and ground truth data.
        
        Args:
            gml_files: List of GML file paths
            seg_list_file: JSON file with segment IDs
            synapse_db_name: Optional synapse database name
            
        Returns:
            DataFrame with features and labels
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for GML processing")
        
        logger.info(f"Extracting features from {len(gml_files)} GML files")
        
        # Load segment list
        with open(seg_list_file, 'r') as f:
            segs = json.load(f)
        
        all_features = []
        all_labels = []
        
        for gml_file in gml_files:
            try:
                g = read_gml(gml_file)
                temp = destringizer(g.graph['info'])
                
                # Extract skeleton data
                skeleton_data = temp.get('skeleton_data', {})
                verified_synapses = temp.get('verified_synapses', {})
                
                # Process skeleton nodes
                for node_id, node_data in skeleton_data.items():
                    # Extract node features
                    coords = node_data.get('coordinates', [0, 0, 0])
                    radius = node_data.get('radius', 0.0)
                    parent_id = node_data.get('parent_id')
                    children_ids = node_data.get('children_ids', [])
                    
                    # Calculate distances to synapses
                    synapse_dists = []
                    for syn_data in verified_synapses.values():
                        if 'tp_synapses' in syn_data:
                            for syn_coord in syn_data['tp_synapses'].values():
                                if isinstance(syn_coord, dict) and 'coordinates' in syn_coord:
                                    dist = euclidean(coords, syn_coord['coordinates'])
                                    synapse_dists.append(dist)
                    
                    # Create skeleton node
                    skeleton_node = SkeletonNode(
                        node_id=node_id,
                        coordinates=tuple(coords),
                        radius=radius,
                        parent_id=parent_id,
                        children_ids=children_ids,
                        synapse_distances=synapse_dists,
                        is_terminal=len(children_ids) == 0,
                        is_branch_point=len(children_ids) > 1
                    )
                    
                    # Extract features
                    features = self._extract_node_features(skeleton_node, verified_synapses)
                    all_features.append(features)
                    
                    # Get ground truth label (if available)
                    label = temp.get('pruning_labels', {}).get(node_id, 0)
                    all_labels.append(label)
                
            except Exception as e:
                logger.warning(f"Failed to process {gml_file}: {e}")
                continue
        
        # Create DataFrame
        if all_features:
            df = pd.DataFrame(all_features)
            df['label'] = all_labels
            logger.info(f"Extracted features for {len(df)} nodes")
            return df
        else:
            logger.warning("No features extracted")
            return pd.DataFrame()
    
    def _extract_node_features(self, node: SkeletonNode, verified_synapses: Dict) -> Dict[str, Any]:
        """Extract features for a single skeleton node."""
        # Calculate synapse distances
        synapse_dists = []
        for syn_data in verified_synapses.values():
            if 'tp_synapses' in syn_data:
                for syn_coord in syn_data['tp_synapses'].values():
                    if isinstance(syn_coord, dict) and 'coordinates' in syn_coord:
                        dist = euclidean(node.coordinates, syn_coord['coordinates'])
                        synapse_dists.append(dist)
        
        synapse_dists.sort()
        closest_syn_dist = synapse_dists[0] if synapse_dists else float('inf')
        second_closest_dist = synapse_dists[1] if len(synapse_dists) > 1 else float('inf')
        
        # Topology features
        num_children = len(node.children_ids)
        is_terminal = num_children == 0
        is_branch_point = num_children > 1
        
        # Morphology features
        node_volume = (4/3) * np.pi * (node.radius ** 3)
        
        return {
            'node_id': node.node_id,
            'radius': node.radius,
            'volume': node_volume,
            'closest_synapse_dist': closest_syn_dist,
            'second_closest_synapse_dist': second_closest_dist,
            'num_children': num_children,
            'is_terminal': is_terminal,
            'is_branch_point': is_branch_point,
            'branch_length': node.branch_length,
            'stalk_length': node.stalk_length
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training/prediction.
        
        Args:
            df: DataFrame with skeleton features
            
        Returns:
            X: Feature array
            Y: Target array
        """
        # Select numerical features
        feature_columns = [
            'radius', 'volume', 'closest_synapse_dist', 'second_closest_synapse_dist',
            'num_children', 'is_terminal', 'is_branch_point', 'branch_length', 'stalk_length'
        ]
        
        # Add optional features
        if 'parent_radius' in df.columns:
            feature_columns.extend(['parent_radius', 'radius_ratio_parent'])
        if 'child_radius_sum' in df.columns:
            feature_columns.extend(['child_radius_sum', 'radius_ratio_children'])
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].values
        Y = df['label'].values if 'label' in df.columns else np.zeros(len(df))
        
        self.feature_names = available_features
        
        return X, Y
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the skeleton pruning model.
        
        Args:
            df: DataFrame with skeleton features and labels
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training skeleton pruning model")
        
        # Prepare features
        X, Y = self.prepare_features(df)
        
        if len(X) == 0:
            raise ValueError("No features available for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=Y
        )
        
        # Train logistic regression
        self.model = LogisticRegressionCV(
            cv=self.config.cv_folds, 
            random_state=self.config.random_state, 
            max_iter=self.config.max_iter
        ).fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Store metrics
        self.metrics = {
            'train_auc_roc': auc_roc,
            'feature_names': self.feature_names,
            'model_coefficients': dict(zip(self.feature_names, self.model.coef_[0])),
            'intercept': self.model.intercept_[0]
        }
        
        logger.info(f"Training completed. AUC-ROC: {auc_roc:.3f}")
        logger.info(f"Features: {self.feature_names}")
        
        return self.metrics
    
    def predict(self, skeleton_nodes: List[SkeletonNode]) -> List[Dict[str, Any]]:
        """
        Predict pruning decisions for skeleton nodes.
        
        Args:
            skeleton_nodes: List of SkeletonNode objects
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        df = self.extract_skeleton_features(skeleton_nodes, [])
        X, _ = self.prepare_features(df)
        
        if len(X) == 0:
            return []
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        results = []
        for i, node in enumerate(skeleton_nodes):
            results.append({
                'node_id': node.node_id,
                'should_prune': bool(predictions[i]),
                'prune_probability': float(probabilities[i]),
                'confidence': max(probabilities[i], 1 - probabilities[i])
            })
        
        return results
    
    def apply_pruning(self, 
                     skeleton_nodes: List[SkeletonNode],
                     pruning_predictions: List[Dict[str, Any]],
                     threshold: float = 0.5) -> List[SkeletonNode]:
        """
        Apply pruning decisions to skeleton nodes.
        
        Args:
            skeleton_nodes: List of skeleton nodes
            pruning_predictions: Predictions from predict() method
            threshold: Probability threshold for pruning
            
        Returns:
            List of pruned skeleton nodes
        """
        # Create node lookup
        node_dict = {node.node_id: node for node in skeleton_nodes}
        
        # Identify nodes to prune
        nodes_to_prune = set()
        for pred in pruning_predictions:
            if pred['prune_probability'] >= threshold:
                nodes_to_prune.add(pred['node_id'])
        
        # Apply pruning
        pruned_nodes = []
        for node in skeleton_nodes:
            if node.node_id not in nodes_to_prune:
                # Remove pruned children from children_ids
                node.children_ids = [child_id for child_id in node.children_ids 
                                   if child_id not in nodes_to_prune]
                pruned_nodes.append(node)
        
        logger.info(f"Pruned {len(nodes_to_prune)} nodes from {len(skeleton_nodes)} total nodes")
        
        return pruned_nodes
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the trained model."""
        path = path or self.config.model_save_path
        model_data = {
            'model': self.model,
            'config': self.config,
            'metrics': self.metrics,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load a trained model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data.get('config', self.config)
        self.metrics = model_data.get('metrics', {})
        self.feature_names = model_data.get('feature_names', [])
        
        logger.info(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        return {
            'metrics': self.metrics,
            'config': self.config,
            'feature_names': self.feature_names,
            'is_trained': self.model is not None
        }

def create_skeleton_from_coordinates(coordinates: List[Tuple[float, float, float]], 
                                   radii: List[float] = None) -> List[SkeletonNode]:
    """
    Create skeleton nodes from coordinate data.
    
    Args:
        coordinates: List of (x, y, z) coordinates
        radii: List of radii (optional, defaults to 1.0)
        
    Returns:
        List of SkeletonNode objects
    """
    if radii is None:
        radii = [1.0] * len(coordinates)
    
    nodes = []
    for i, (coord, radius) in enumerate(zip(coordinates, radii)):
        node = SkeletonNode(
            node_id=f"node_{i}",
            coordinates=coord,
            radius=radius,
            parent_id=f"node_{i-1}" if i > 0 else None,
            children_ids=[f"node_{i+1}"] if i < len(coordinates) - 1 else []
        )
        nodes.append(node)
    
    return nodes

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create pruner
    config = PruningConfig()
    pruner = SkeletonPruner(config)
    
    # Example: Create synthetic skeleton data
    np.random.seed(42)
    n_nodes = 50
    
    # Generate synthetic skeleton coordinates
    coordinates = []
    for i in range(n_nodes):
        if i == 0:
            coord = (0, 0, 0)
        else:
            prev_coord = coordinates[-1]
            coord = (
                prev_coord[0] + np.random.normal(0, 10),
                prev_coord[1] + np.random.normal(0, 10),
                prev_coord[2] + np.random.normal(0, 10)
            )
        coordinates.append(coord)
    
    # Create skeleton nodes
    skeleton_nodes = create_skeleton_from_coordinates(coordinates)
    
    print(f"Created {len(skeleton_nodes)} skeleton nodes")
    
    # Example: Train model (would need real data with labels)
    # pruner.train(df)
    # pruner.save()
    
    print("SkeletonPruner ready for integration with tracer agent") 