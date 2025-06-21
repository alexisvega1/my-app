#!/usr/bin/env python3
"""
H01 Synapse Merge Model for Agentic Tracer
==========================================
Adapted from H01 project synapse merge decision model.
Provides robust synapse merge/split decisions using ML and rule-based approaches.
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
class SynapsePair:
    """Represents a pair of synapses for merge decision."""
    synapse_1_id: str
    synapse_2_id: str
    synapse_1_coords: Tuple[float, float, float]
    synapse_2_coords: Tuple[float, float, float]
    distance_nm: float
    pre_skel_dist_nm: Optional[float] = None
    post_skel_dist_nm: Optional[float] = None
    pre_skel_dist_normalized: Optional[float] = None
    post_skel_dist_normalized: Optional[float] = None
    true_decision: Optional[str] = None  # 'join' or 'separate'
    
    @property
    def pair_id(self) -> str:
        """Get unique pair identifier."""
        ids = sorted([self.synapse_1_id, self.synapse_2_id])
        return '-'.join(ids)
    
    @property
    def max_skel_dist_normalized(self) -> Optional[float]:
        """Get maximum normalized skeleton distance."""
        if self.pre_skel_dist_normalized is not None and self.post_skel_dist_normalized is not None:
            return max(self.pre_skel_dist_normalized, self.post_skel_dist_normalized)
        return None

@dataclass
class MergeModelConfig:
    """Configuration for synapse merge model."""
    lower_threshold_range: range = range(750, 3000, 50)
    upper_threshold_range: range = range(1000, 5000, 50)
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    max_iter: int = 1000
    
    # Feature engineering
    use_distance_thresholds: bool = True
    use_skeleton_features: bool = True
    use_normalized_features: bool = True
    
    # Model persistence
    model_save_path: str = "synapse_merge_model.pkl"
    cache_dir: str = "./merge_model_cache"

class SynapseMergeModel:
    """
    Synapse merge decision model based on H01 project.
    
    Combines rule-based thresholds with ML predictions for robust merge/split decisions.
    """
    
    def __init__(self, config: Optional[MergeModelConfig] = None):
        self.config = config or MergeModelConfig()
        self.model = None
        self.lower_threshold = None
        self.upper_threshold = None
        self.feature_names = []
        self.metrics = {}
        
        # Ensure cache directory exists
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        logger.info("SynapseMergeModel initialized")
    
    def extract_features_from_gml(self, 
                                 gml_files: List[str], 
                                 seg_list_file: str,
                                 path_data_file: str) -> pd.DataFrame:
        """
        Extract features from GML files and ground truth data.
        
        Args:
            gml_files: List of GML file paths
            seg_list_file: JSON file with segment IDs
            path_data_file: JSON file with path data
            
        Returns:
            DataFrame with features and labels
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for GML processing")
        
        logger.info(f"Extracting features from {len(gml_files)} GML files")
        
        # Load segment list
        with open(seg_list_file, 'r') as f:
            segs = json.load(f)
        
        # Load path data
        with open(path_data_file, 'r') as f:
            path_data = json.load(f)
        
        # Process GML files
        syn_data = {}
        merge_data = []
        
        for gml_file in gml_files:
            try:
                g = read_gml(gml_file)
                temp = destringizer(g.graph['info'])
                
                # Extract verified synapses
                for k in temp.get('verified_synapses', {}).keys():
                    if 'tp_synapses' in temp['verified_synapses'][k]:
                        for k2 in temp['verified_synapses'][k]['tp_synapses'].keys():
                            syn_data[k2] = deepcopy(temp['verified_synapses'][k]['tp_synapses'][k2])
                
                # Extract merge decisions
                merge_data.extend(temp.get('synapse_merge_decisions', []))
                
            except Exception as e:
                logger.warning(f"Failed to process {gml_file}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(columns=[
            'true_condition', 'dist', 'pre_skel_dist', 'pre_skel_dist_n', 
            'post_skel_dist', 'post_skel_dist_n'
        ])
        
        # Process merge decisions
        for x in merge_data:
            x['synapse_ids'].sort()
            combined_id = '-'.join(x['synapse_ids'])
            
            tc = 1 if x['decision'] == 'join' else 0
            assert x['decision'] in ['join', 'separate']
            
            df.loc[combined_id] = [tc, x['distance_nm'], None, None, None, None]
        
        # Add path data
        for x in path_data:
            syn1_id = f"{x['synapse_1'][2]}_{x['synapse_1'][3]}"
            syn2_id = f"{x['synapse_2'][2]}_{x['synapse_2'][3]}"
            
            both_ids = sorted([syn1_id, syn2_id])
            combined_id = '-'.join(both_ids)
            
            if combined_id in df.index:
                pre_dist = euclidean(x['synapse_1'][0], x['synapse_2'][0])
                post_dist = euclidean(x['synapse_1'][1], x['synapse_2'][1])
                
                df.at[combined_id, 'pre_skel_dist'] = x['pre_path_len_nm']
                df.at[combined_id, 'post_skel_dist'] = x['post_path_len_nm']
                
                # Normalized distances
                df.at[combined_id, 'pre_skel_dist_n'] = (
                    x['pre_path_len_nm'] / pre_dist if pre_dist > 0 else 0
                )
                df.at[combined_id, 'post_skel_dist_n'] = (
                    x['post_path_len_nm'] / post_dist if post_dist > 0 else 0
                )
        
        logger.info(f"Extracted {len(df)} synapse pairs")
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                        upper_threshold: float, 
                        lower_threshold: float) -> Tuple[np.ndarray, np.ndarray, List, List]:
        """
        Prepare features for model training/prediction.
        
        Args:
            df: DataFrame with synapse pair data
            upper_threshold: Upper distance threshold
            lower_threshold: Lower distance threshold
            
        Returns:
            X: Feature array for ML model
            Y: Target array
            simple_pred: Rule-based predictions
            simple_true: Corresponding true values
        """
        X = []
        Y = []
        simple_pred = []
        simple_true = []
        
        for idx in df.index:
            dist = df.at[idx, 'dist']
            
            # Apply distance thresholds
            if dist >= upper_threshold:
                simple_pred.append(0)  # Separate
                simple_true.append(df.at[idx, 'true_condition'])
                continue
            
            if dist <= lower_threshold:
                simple_pred.append(1)  # Join
                simple_true.append(df.at[idx, 'true_condition'])
                continue
            
            # Use ML features for ambiguous cases
            pre_skel_dist = df.at[idx, 'pre_skel_dist_n']
            post_skel_dist = df.at[idx, 'post_skel_dist_n']
            
            features = []
            if self.config.use_skeleton_features:
                features.append(max(pre_skel_dist, post_skel_dist))
            
            if self.config.use_normalized_features:
                features.extend([pre_skel_dist, post_skel_dist])
            
            if features:
                X.append(features)
                Y.append(df.at[idx, 'true_condition'])
        
        return np.array(X), np.array(Y), simple_pred, simple_true
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the synapse merge model.
        
        Args:
            df: DataFrame with synapse pair data
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training synapse merge model")
        
        # Split data
        train_df, test_df = train_test_split(
            df, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=df['true_condition']
        )
        
        # Initial training with default thresholds
        X, Y, simple_pred, simple_true = self.prepare_features(
            train_df, 5000, 0
        )
        
        if len(X) == 0:
            raise ValueError("No features available for training")
        
        # Train logistic regression
        self.model = LogisticRegressionCV(
            cv=self.config.cv_folds, 
            random_state=self.config.random_state, 
            max_iter=self.config.max_iter
        ).fit(X, Y)
        
        # Optimize thresholds
        current_best_thresholds = None
        current_best_auc_roc = 0
        
        for upper_threshold in self.config.upper_threshold_range:
            for lower_threshold in self.config.lower_threshold_range:
                if lower_threshold >= upper_threshold:
                    continue
                
                X, Y, simple_pred, simple_true = self.prepare_features(
                    train_df, upper_threshold, lower_threshold
                )
                
                if len(X) == 0:
                    pred_x = []
                else:
                    pred_x = list(self.model.predict(X))
                
                binary_predictions = [int(a) for a in simple_pred + pred_x]
                true_vals = [int(a) for a in simple_true + list(Y)]
                
                auc_roc = roc_auc_score(true_vals, binary_predictions)
                
                if auc_roc > current_best_auc_roc:
                    current_best_auc_roc = auc_roc
                    current_best_thresholds = (lower_threshold, upper_threshold)
        
        # Set best thresholds
        self.lower_threshold, self.upper_threshold = current_best_thresholds
        
        # Evaluate on test set
        test_metrics = self.evaluate(test_df)
        
        # Store metrics
        self.metrics = {
            'train_auc_roc': current_best_auc_roc,
            'lower_threshold': self.lower_threshold,
            'upper_threshold': self.upper_threshold,
            'test_metrics': test_metrics
        }
        
        logger.info(f"Training completed. Best thresholds: {current_best_thresholds}")
        logger.info(f"Train AUC-ROC: {current_best_auc_roc:.3f}")
        
        return self.metrics
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            df: Test DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        X, Y, simple_pred, simple_true = self.prepare_features(
            df, self.upper_threshold, self.lower_threshold
        )
        
        if len(X) == 0:
            pred_x = []
        else:
            pred_x = list(self.model.predict(X))
        
        binary_predictions = [int(a) for a in simple_pred + pred_x]
        true_vals = [int(a) for a in simple_true + list(Y)]
        
        # Calculate accuracies
        sep_accuracy, join_accuracy, joined_incorrectly, separated_incorrectly = \
            self._get_accuracies(true_vals, binary_predictions)
        
        # Calculate metrics
        auc_roc = roc_auc_score(true_vals, binary_predictions)
        
        metrics = {
            'separation_accuracy': sep_accuracy,
            'join_accuracy': join_accuracy,
            'auc_roc': auc_roc,
            'false_merges': joined_incorrectly,
            'false_splits': separated_incorrectly,
            'total_predictions': len(binary_predictions)
        }
        
        logger.info(f"Test AUC-ROC: {auc_roc:.3f}")
        logger.info(f"Separation accuracy: {sep_accuracy}")
        logger.info(f"Join accuracy: {join_accuracy}")
        
        return metrics
    
    def predict(self, synapse_pairs: List[SynapsePair]) -> List[Dict[str, Any]]:
        """
        Predict merge/split decisions for synapse pairs.
        
        Args:
            synapse_pairs: List of SynapsePair objects
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        
        for pair in synapse_pairs:
            # Apply distance thresholds
            if pair.distance_nm >= self.upper_threshold:
                decision = 'separate'
                confidence = 1.0
                method = 'threshold'
            elif pair.distance_nm <= self.lower_threshold:
                decision = 'join'
                confidence = 1.0
                method = 'threshold'
            else:
                # Use ML model
                if pair.max_skel_dist_normalized is not None:
                    features = [[pair.max_skel_dist_normalized]]
                    if self.config.use_normalized_features:
                        features[0].extend([pair.pre_skel_dist_normalized, pair.post_skel_dist_normalized])
                    
                    prob = self.model.predict_proba(features)[0]
                    prediction = self.model.predict(features)[0]
                    
                    decision = 'join' if prediction == 1 else 'separate'
                    confidence = max(prob)
                    method = 'ml'
                else:
                    # Fallback to distance-based decision
                    decision = 'separate' if pair.distance_nm > (self.lower_threshold + self.upper_threshold) / 2 else 'join'
                    confidence = 0.5
                    method = 'fallback'
            
            predictions.append({
                'pair_id': pair.pair_id,
                'decision': decision,
                'confidence': confidence,
                'method': method,
                'distance_nm': pair.distance_nm,
                'max_skel_dist_normalized': pair.max_skel_dist_normalized
            })
        
        return predictions
    
    def _get_accuracies(self, true_vals: List[int], binary_predictions: List[int]) -> Tuple[float, float, int, int]:
        """Calculate separation and join accuracies."""
        c = list(zip(true_vals, binary_predictions))
        
        joined_correctly = len([x for x in c if x[0] == 1 and x[1] == 1])
        joined_incorrectly = len([x for x in c if x[0] == 1 and x[1] == 0])
        separated_incorrectly = len([x for x in c if x[0] == 0 and x[1] == 1])
        separated_correctly = len([x for x in c if x[0] == 0 and x[1] == 0])
        
        join_accuracy = (
            joined_correctly / (joined_correctly + separated_incorrectly)
            if joined_correctly + separated_incorrectly > 0 else 0
        )
        
        sep_accuracy = (
            separated_correctly / (separated_correctly + joined_incorrectly)
            if separated_correctly + joined_incorrectly > 0 else 0
        )
        
        return sep_accuracy, join_accuracy, joined_incorrectly, separated_incorrectly
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the trained model."""
        path = path or self.config.model_save_path
        model_data = {
            'model': self.model,
            'lower_threshold': self.lower_threshold,
            'upper_threshold': self.upper_threshold,
            'config': self.config,
            'metrics': self.metrics
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load a trained model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.lower_threshold = model_data['lower_threshold']
        self.upper_threshold = model_data['upper_threshold']
        self.config = model_data.get('config', self.config)
        self.metrics = model_data.get('metrics', {})
        
        logger.info(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        return {
            'lower_threshold': self.lower_threshold,
            'upper_threshold': self.upper_threshold,
            'metrics': self.metrics,
            'config': self.config,
            'is_trained': self.model is not None
        }

def create_synapse_pairs_from_coordinates(synapse_coords: List[Tuple[str, Tuple[float, float, float]]], 
                                        max_distance_nm: float = 5000) -> List[SynapsePair]:
    """
    Create synapse pairs from coordinate data.
    
    Args:
        synapse_coords: List of (synapse_id, (x, y, z)) tuples
        max_distance_nm: Maximum distance to consider for pairs
        
    Returns:
        List of SynapsePair objects
    """
    pairs = []
    
    for i, (syn1_id, syn1_coords) in enumerate(synapse_coords):
        for j, (syn2_id, syn2_coords) in enumerate(synapse_coords[i+1:], i+1):
            distance = euclidean(syn1_coords, syn2_coords)
            
            if distance <= max_distance_nm:
                pair = SynapsePair(
                    synapse_1_id=syn1_id,
                    synapse_2_id=syn2_id,
                    synapse_1_coords=syn1_coords,
                    synapse_2_coords=syn2_coords,
                    distance_nm=distance
                )
                pairs.append(pair)
    
    return pairs

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    config = MergeModelConfig()
    model = SynapseMergeModel(config)
    
    # Example: Create synthetic data for testing
    np.random.seed(42)
    n_synapses = 100
    
    # Generate synthetic synapse coordinates
    synapse_coords = [
        (f"syn_{i}", (np.random.uniform(0, 1000), np.random.uniform(0, 1000), np.random.uniform(0, 1000)))
        for i in range(n_synapses)
    ]
    
    # Create pairs
    pairs = create_synapse_pairs_from_coordinates(synapse_coords, max_distance_nm=500)
    
    print(f"Created {len(pairs)} synapse pairs")
    
    # Example: Train model (would need real data)
    # model.train(df)
    # model.save()
    
    print("SynapseMergeModel ready for integration with tracer agent") 