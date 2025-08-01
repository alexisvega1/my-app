#!/usr/bin/env python3
"""
Neuronal Function and Molecular Identity Predictor
=================================================
Machine learning system for predicting neuronal function, behavior, and molecular identity
based on morphological features extracted from connectomics data.
"""

import os
import sys
import json
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

@dataclass
class NeuronalFunctionConfig:
    """Configuration for neuronal function prediction system."""
    
    # Feature extraction parameters
    morphological_features: List[str] = None
    connectivity_features: List[str] = None
    electrophysiological_features: List[str] = None
    
    # Model parameters
    use_random_forest: bool = True
    use_gradient_boosting: bool = True
    use_neural_network: bool = True
    use_xgboost: bool = True
    use_lightgbm: bool = True
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5
    
    # Model hyperparameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    
    gb_n_estimators: int = 200
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 10
    
    nn_hidden_layers: Tuple[int, ...] = (256, 128, 64)
    nn_dropout: float = 0.3
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 32
    nn_epochs: int = 100
    
    # Prediction parameters
    confidence_threshold: float = 0.7
    top_k_predictions: int = 3
    
    def __post_init__(self):
        if self.morphological_features is None:
            self.morphological_features = [
                'soma_volume', 'soma_surface_area', 'soma_sphericity',
                'dendritic_length', 'dendritic_complexity', 'dendritic_branching',
                'axonal_length', 'axonal_complexity', 'axonal_branching',
                'spine_density', 'spine_distribution', 'spine_types',
                'total_volume', 'surface_area_ratio', 'elongation_factor',
                'branching_factor', 'tortuosity', 'density_factor'
            ]
        
        if self.connectivity_features is None:
            self.connectivity_features = [
                'incoming_connections', 'outgoing_connections', 'connection_strength',
                'synaptic_density', 'connection_distance', 'connection_pattern',
                'reciprocal_connections', 'feedforward_connections', 'feedback_connections',
                'modularity', 'clustering_coefficient', 'betweenness_centrality',
                'eigenvector_centrality', 'degree_centrality', 'local_efficiency'
            ]
        
        if self.electrophysiological_features is None:
            self.electrophysiological_features = [
                'firing_rate', 'spike_width', 'spike_amplitude', 'spike_threshold',
                'resting_potential', 'input_resistance', 'time_constant',
                'sag_ratio', 'rebound_depolarization', 'accommodation',
                'burst_frequency', 'burst_duration', 'inter_burst_interval',
                'adaptation_ratio', 'regularity_index', 'synchronization_index'
            ]

class MorphologicalFeatureExtractor:
    """Extract comprehensive morphological features from neuronal data."""
    
    def __init__(self, config: NeuronalFunctionConfig):
        self.config = config
        logger.info("Morphological feature extractor initialized")
    
    def extract_features(self, neuron_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract all morphological features from neuron data."""
        
        features = {}
        
        # Basic morphological features
        features.update(self._extract_basic_morphology(neuron_data))
        
        # Dendritic features
        features.update(self._extract_dendritic_features(neuron_data))
        
        # Axonal features
        features.update(self._extract_axonal_features(neuron_data))
        
        # Spine features
        features.update(self._extract_spine_features(neuron_data))
        
        # Geometric features
        features.update(self._extract_geometric_features(neuron_data))
        
        # Complexity features
        features.update(self._extract_complexity_features(neuron_data))
        
        return features
    
    def _extract_basic_morphology(self, neuron_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic morphological features."""
        
        soma_position = neuron_data.get('soma_position', (0, 0, 0))
        soma_volume = self._calculate_soma_volume(neuron_data)
        soma_surface_area = self._calculate_soma_surface_area(neuron_data)
        
        return {
            'soma_volume': soma_volume,
            'soma_surface_area': soma_surface_area,
            'soma_sphericity': self._calculate_sphericity(soma_volume, soma_surface_area),
            'soma_x': soma_position[0],
            'soma_y': soma_position[1],
            'soma_z': soma_position[2]
        }
    
    def _extract_dendritic_features(self, neuron_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract dendritic morphological features."""
        
        dendrites = neuron_data.get('dendrites', [])
        
        if not dendrites:
            return {
                'dendritic_length': 0.0,
                'dendritic_complexity': 0.0,
                'dendritic_branching': 0.0,
                'dendritic_tortuosity': 0.0,
                'dendritic_density': 0.0
            }
        
        total_length = sum(d.get('features', {}).get('path_length', 0) for d in dendrites)
        complexity = len(dendrites)
        branching = sum(self._calculate_branching_points(d) for d in dendrites)
        tortuosity = np.mean([self._calculate_tortuosity(d) for d in dendrites])
        density = total_length / max(complexity, 1)
        
        return {
            'dendritic_length': total_length,
            'dendritic_complexity': complexity,
            'dendritic_branching': branching,
            'dendritic_tortuosity': tortuosity,
            'dendritic_density': density
        }
    
    def _extract_axonal_features(self, neuron_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract axonal morphological features."""
        
        axons = neuron_data.get('axons', [])
        
        if not axons:
            return {
                'axonal_length': 0.0,
                'axonal_complexity': 0.0,
                'axonal_branching': 0.0,
                'axonal_tortuosity': 0.0,
                'axonal_density': 0.0
            }
        
        total_length = sum(a.get('features', {}).get('path_length', 0) for a in axons)
        complexity = len(axons)
        branching = sum(self._calculate_branching_points(a) for a in axons)
        tortuosity = np.mean([self._calculate_tortuosity(a) for a in axons])
        density = total_length / max(complexity, 1)
        
        return {
            'axonal_length': total_length,
            'axonal_complexity': complexity,
            'axonal_branching': branching,
            'axonal_tortuosity': tortuosity,
            'axonal_density': density
        }
    
    def _extract_spine_features(self, neuron_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract spine-related features."""
        
        spines = neuron_data.get('spines', [])
        dendrites = neuron_data.get('dendrites', [])
        
        if not spines:
            return {
                'spine_density': 0.0,
                'spine_distribution': 0.0,
                'mushroom_spine_ratio': 0.0,
                'thin_spine_ratio': 0.0,
                'stubby_spine_ratio': 0.0,
                'filopodia_ratio': 0.0,
                'branched_spine_ratio': 0.0
            }
        
        total_spines = len(spines)
        dendritic_length = sum(d.get('features', {}).get('path_length', 0) for d in dendrites)
        spine_density = total_spines / max(dendritic_length, 1)
        
        # Calculate spine type ratios
        spine_types = [s.get('spine_type', 'unknown') for s in spines]
        type_counts = {}
        for spine_type in ['mushroom', 'thin', 'stubby', 'filopodia', 'branched']:
            type_counts[spine_type] = spine_types.count(spine_type)
        
        total_spines_typed = sum(type_counts.values())
        if total_spines_typed == 0:
            type_ratios = {f'{spine_type}_spine_ratio': 0.0 for spine_type in type_counts}
        else:
            type_ratios = {f'{spine_type}_spine_ratio': count / total_spines_typed 
                          for spine_type, count in type_counts.items()}
        
        return {
            'spine_density': spine_density,
            'spine_distribution': self._calculate_spine_distribution(spines),
            **type_ratios
        }
    
    def _extract_geometric_features(self, neuron_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract geometric features."""
        
        dendrites = neuron_data.get('dendrites', [])
        axons = neuron_data.get('axons', [])
        
        total_volume = self._calculate_total_volume(neuron_data)
        total_surface_area = self._calculate_total_surface_area(neuron_data)
        
        return {
            'total_volume': total_volume,
            'surface_area_ratio': total_surface_area / max(total_volume, 1),
            'elongation_factor': self._calculate_elongation_factor(neuron_data),
            'density_factor': total_volume / max(len(dendrites) + len(axons), 1)
        }
    
    def _extract_complexity_features(self, neuron_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract complexity features."""
        
        dendrites = neuron_data.get('dendrites', [])
        axons = neuron_data.get('axons', [])
        
        return {
            'branching_factor': self._calculate_branching_factor(neuron_data),
            'tortuosity': self._calculate_overall_tortuosity(neuron_data),
            'complexity_index': len(dendrites) + len(axons),
            'connectivity_index': len(neuron_data.get('connectivity', {}).get('synaptic_contacts', []))
        }
    
    # Helper methods for calculations
    def _calculate_soma_volume(self, neuron_data: Dict[str, Any]) -> float:
        """Calculate soma volume."""
        # Simplified calculation - in practice would use actual soma segmentation
        return 1000.0  # Placeholder
    
    def _calculate_soma_surface_area(self, neuron_data: Dict[str, Any]) -> float:
        """Calculate soma surface area."""
        return 500.0  # Placeholder
    
    def _calculate_sphericity(self, volume: float, surface_area: float) -> float:
        """Calculate sphericity."""
        if surface_area == 0:
            return 0.0
        return (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area
    
    def _calculate_branching_points(self, projection: Dict[str, Any]) -> int:
        """Calculate number of branching points in a projection."""
        path = projection.get('path', [])
        if len(path) < 3:
            return 0
        
        branching_points = 0
        for i in range(1, len(path) - 1):
            # Simplified branching detection
            if self._is_branching_point(path, i):
                branching_points += 1
        
        return branching_points
    
    def _is_branching_point(self, path: List[Tuple[int, int, int]], index: int) -> bool:
        """Check if a point is a branching point."""
        # Simplified implementation
        return index % 10 == 0  # Placeholder
    
    def _calculate_tortuosity(self, projection: Dict[str, Any]) -> float:
        """Calculate tortuosity of a projection."""
        path = projection.get('path', [])
        if len(path) < 2:
            return 1.0
        
        # Calculate actual path length vs straight-line distance
        actual_length = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i-1])) 
                           for i in range(1, len(path)))
        straight_distance = np.linalg.norm(np.array(path[-1]) - np.array(path[0]))
        
        return actual_length / max(straight_distance, 1e-8)
    
    def _calculate_spine_distribution(self, spines: List[Dict[str, Any]]) -> float:
        """Calculate spine distribution uniformity."""
        if not spines:
            return 0.0
        
        positions = [s.get('position', (0, 0, 0)) for s in spines]
        distances = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                distances.append(dist)
        
        return np.std(distances) if distances else 0.0
    
    def _calculate_total_volume(self, neuron_data: Dict[str, Any]) -> float:
        """Calculate total neuron volume."""
        return 10000.0  # Placeholder
    
    def _calculate_total_surface_area(self, neuron_data: Dict[str, Any]) -> float:
        """Calculate total neuron surface area."""
        return 5000.0  # Placeholder
    
    def _calculate_elongation_factor(self, neuron_data: Dict[str, Any]) -> float:
        """Calculate elongation factor."""
        dendrites = neuron_data.get('dendrites', [])
        axons = neuron_data.get('axons', [])
        
        total_length = sum(d.get('features', {}).get('path_length', 0) for d in dendrites)
        total_length += sum(a.get('features', {}).get('path_length', 0) for a in axons)
        
        return total_length / max(len(dendrites) + len(axons), 1)
    
    def _calculate_branching_factor(self, neuron_data: Dict[str, Any]) -> float:
        """Calculate overall branching factor."""
        dendrites = neuron_data.get('dendrites', [])
        axons = neuron_data.get('axons', [])
        
        total_branching = sum(self._calculate_branching_points(d) for d in dendrites)
        total_branching += sum(self._calculate_branching_points(a) for a in axons)
        
        return total_branching / max(len(dendrites) + len(axons), 1)
    
    def _calculate_overall_tortuosity(self, neuron_data: Dict[str, Any]) -> float:
        """Calculate overall tortuosity."""
        dendrites = neuron_data.get('dendrites', [])
        axons = neuron_data.get('axons', [])
        
        tortuosities = [self._calculate_tortuosity(d) for d in dendrites]
        tortuosities.extend([self._calculate_tortuosity(a) for a in axons])
        
        return np.mean(tortuosities) if tortuosities else 1.0

class ConnectivityFeatureExtractor:
    """Extract connectivity features from neuronal data."""
    
    def __init__(self, config: NeuronalFunctionConfig):
        self.config = config
        logger.info("Connectivity feature extractor initialized")
    
    def extract_features(self, neuron_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract connectivity features."""
        
        connectivity = neuron_data.get('connectivity', {})
        synaptic_contacts = connectivity.get('synaptic_contacts', [])
        
        features = {
            'incoming_connections': self._count_incoming_connections(synaptic_contacts),
            'outgoing_connections': self._count_outgoing_connections(synaptic_contacts),
            'connection_strength': self._calculate_connection_strength(synaptic_contacts),
            'synaptic_density': len(synaptic_contacts),
            'connection_distance': self._calculate_connection_distance(synaptic_contacts),
            'connection_pattern': self._calculate_connection_pattern(synaptic_contacts),
            'reciprocal_connections': self._count_reciprocal_connections(synaptic_contacts),
            'feedforward_connections': self._count_feedforward_connections(synaptic_contacts),
            'feedback_connections': self._count_feedback_connections(synaptic_contacts),
            'modularity': self._calculate_modularity(synaptic_contacts),
            'clustering_coefficient': self._calculate_clustering_coefficient(synaptic_contacts),
            'betweenness_centrality': self._calculate_betweenness_centrality(synaptic_contacts),
            'eigenvector_centrality': self._calculate_eigenvector_centrality(synaptic_contacts),
            'degree_centrality': self._calculate_degree_centrality(synaptic_contacts),
            'local_efficiency': self._calculate_local_efficiency(synaptic_contacts)
        }
        
        return features
    
    def _count_incoming_connections(self, synaptic_contacts: List[Dict[str, Any]]) -> int:
        """Count incoming connections."""
        return len([c for c in synaptic_contacts if c.get('type') == 'incoming'])
    
    def _count_outgoing_connections(self, synaptic_contacts: List[Dict[str, Any]]) -> int:
        """Count outgoing connections."""
        return len([c for c in synaptic_contacts if c.get('type') == 'outgoing'])
    
    def _calculate_connection_strength(self, synaptic_contacts: List[Dict[str, Any]]) -> float:
        """Calculate average connection strength."""
        if not synaptic_contacts:
            return 0.0
        return np.mean([c.get('strength', 0.0) for c in synaptic_contacts])
    
    def _calculate_connection_distance(self, synaptic_contacts: List[Dict[str, Any]]) -> float:
        """Calculate average connection distance."""
        if not synaptic_contacts:
            return 0.0
        return np.mean([c.get('distance', 0.0) for c in synaptic_contacts])
    
    def _calculate_connection_pattern(self, synaptic_contacts: List[Dict[str, Any]]) -> float:
        """Calculate connection pattern complexity."""
        return len(set(c.get('pattern', 'unknown') for c in synaptic_contacts))
    
    def _count_reciprocal_connections(self, synaptic_contacts: List[Dict[str, Any]]) -> int:
        """Count reciprocal connections."""
        return len([c for c in synaptic_contacts if c.get('reciprocal', False)])
    
    def _count_feedforward_connections(self, synaptic_contacts: List[Dict[str, Any]]) -> int:
        """Count feedforward connections."""
        return len([c for c in synaptic_contacts if c.get('direction') == 'feedforward'])
    
    def _count_feedback_connections(self, synaptic_contacts: List[Dict[str, Any]]) -> int:
        """Count feedback connections."""
        return len([c for c in synaptic_contacts if c.get('direction') == 'feedback'])
    
    def _calculate_modularity(self, synaptic_contacts: List[Dict[str, Any]]) -> float:
        """Calculate modularity."""
        return 0.5  # Placeholder
    
    def _calculate_clustering_coefficient(self, synaptic_contacts: List[Dict[str, Any]]) -> float:
        """Calculate clustering coefficient."""
        return 0.3  # Placeholder
    
    def _calculate_betweenness_centrality(self, synaptic_contacts: List[Dict[str, Any]]) -> float:
        """Calculate betweenness centrality."""
        return 0.2  # Placeholder
    
    def _calculate_eigenvector_centrality(self, synaptic_contacts: List[Dict[str, Any]]) -> float:
        """Calculate eigenvector centrality."""
        return 0.4  # Placeholder
    
    def _calculate_degree_centrality(self, synaptic_contacts: List[Dict[str, Any]]) -> float:
        """Calculate degree centrality."""
        return len(synaptic_contacts) / 100.0  # Normalized
    
    def _calculate_local_efficiency(self, synaptic_contacts: List[Dict[str, Any]]) -> float:
        """Calculate local efficiency."""
        return 0.6  # Placeholder

class NeuronalFunctionPredictor:
    """Main class for predicting neuronal function and molecular identity."""
    
    def __init__(self, config: NeuronalFunctionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize feature extractors
        self.morphological_extractor = MorphologicalFeatureExtractor(config)
        self.connectivity_extractor = ConnectivityFeatureExtractor(config)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Neuron type definitions
        self.neuron_types = {
            'excitatory': ['pyramidal', 'stellate', 'granule', 'chandelier'],
            'inhibitory': ['basket', 'chandelier', 'martinotti', 'bitufted'],
            'modulatory': ['serotonergic', 'dopaminergic', 'noradrenergic', 'cholinergic'],
            'sensory': ['retinal_ganglion', 'cochlear', 'olfactory', 'somatosensory'],
            'motor': ['motor_neuron', 'spinal_motor', 'cranial_motor'],
            'interneuron': ['parvalbumin', 'somatostatin', 'vasoactive_intestinal_peptide']
        }
        
        logger.info(f"Neuronal function predictor initialized on {self.device}")
    
    def train_models(self, training_data: List[Dict[str, Any]], 
                    labels: Dict[str, List[str]]) -> Dict[str, Any]:
        """Train all prediction models."""
        
        logger.info("Training neuronal function prediction models")
        
        # Extract features
        features_df = self._extract_features_batch(training_data)
        
        # Train models for each prediction task
        results = {}
        
        for task_name, task_labels in labels.items():
            logger.info(f"Training model for task: {task_name}")
            
            # Prepare data
            X = features_df.values
            y = task_labels
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            
            # Train models
            task_results = self._train_task_models(
                X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded
            )
            
            # Store results
            results[task_name] = {
                'models': task_results['models'],
                'scaler': scaler,
                'label_encoder': label_encoder,
                'performance': task_results['performance']
            }
            
            # Store for later use
            self.models[task_name] = task_results['models']
            self.scalers[task_name] = scaler
            self.label_encoders[task_name] = label_encoder
        
        return results
    
    def _extract_features_batch(self, neuron_data_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features from a batch of neurons."""
        
        features_list = []
        
        for neuron_data in neuron_data_list:
            # Extract morphological features
            morph_features = self.morphological_extractor.extract_features(neuron_data)
            
            # Extract connectivity features
            conn_features = self.connectivity_extractor.extract_features(neuron_data)
            
            # Combine features
            combined_features = {**morph_features, **conn_features}
            features_list.append(combined_features)
        
        return pd.DataFrame(features_list)
    
    def _train_task_models(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train models for a specific prediction task."""
        
        models = {}
        performance = {}
        
        # Random Forest
        if self.config.use_random_forest:
            rf_model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            rf_model.fit(X_train, y_train)
            rf_score = rf_model.score(X_test, y_test)
            
            models['random_forest'] = rf_model
            performance['random_forest'] = {
                'accuracy': rf_score,
                'feature_importance': rf_model.feature_importances_
            }
        
        # Gradient Boosting
        if self.config.use_gradient_boosting:
            gb_model = GradientBoostingClassifier(
                n_estimators=self.config.gb_n_estimators,
                learning_rate=self.config.gb_learning_rate,
                max_depth=self.config.gb_max_depth,
                random_state=self.config.random_state
            )
            gb_model.fit(X_train, y_train)
            gb_score = gb_model.score(X_test, y_test)
            
            models['gradient_boosting'] = gb_model
            performance['gradient_boosting'] = {
                'accuracy': gb_score,
                'feature_importance': gb_model.feature_importances_
            }
        
        # Neural Network
        if self.config.use_neural_network:
            nn_model = self._create_neural_network(X_train.shape[1], len(np.unique(y_train)))
            nn_performance = self._train_neural_network(nn_model, X_train, y_train, X_test, y_test)
            
            models['neural_network'] = nn_model
            performance['neural_network'] = nn_performance
        
        # XGBoost
        if self.config.use_xgboost:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=10,
                random_state=self.config.random_state
            )
            xgb_model.fit(X_train, y_train)
            xgb_score = xgb_model.score(X_test, y_test)
            
            models['xgboost'] = xgb_model
            performance['xgboost'] = {
                'accuracy': xgb_score,
                'feature_importance': xgb_model.feature_importances_
            }
        
        # LightGBM
        if self.config.use_lightgbm:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=10,
                random_state=self.config.random_state
            )
            lgb_model.fit(X_train, y_train)
            lgb_score = lgb_model.score(X_test, y_test)
            
            models['lightgbm'] = lgb_model
            performance['lightgbm'] = {
                'accuracy': lgb_score,
                'feature_importance': lgb_model.feature_importances_
            }
        
        return {'models': models, 'performance': performance}
    
    def _create_neural_network(self, input_size: int, num_classes: int) -> nn.Module:
        """Create neural network for classification."""
        
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size, hidden_layers, num_classes, dropout):
                super().__init__()
                
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_layers:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.BatchNorm1d(hidden_size)
                    ])
                    prev_size = hidden_size
                
                layers.append(nn.Linear(prev_size, num_classes))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return NeuralNetwork(
            input_size, 
            self.config.nn_hidden_layers, 
            num_classes, 
            self.config.nn_dropout
        ).to(self.device)
    
    def _train_neural_network(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Train neural network."""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.nn_learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(self.config.nn_epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean().item()
        
        return {'accuracy': accuracy}
    
    def predict_neuronal_function(self, neuron_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict neuronal function and molecular identity."""
        
        logger.info("Predicting neuronal function and molecular identity")
        
        # Extract features
        morph_features = self.morphological_extractor.extract_features(neuron_data)
        conn_features = self.connectivity_extractor.extract_features(neuron_data)
        combined_features = {**morph_features, **conn_features}
        
        # Convert to array
        feature_array = np.array(list(combined_features.values())).reshape(1, -1)
        
        predictions = {}
        
        # Make predictions for each task
        for task_name, models in self.models.items():
            task_predictions = {}
            
            # Scale features
            scaler = self.scalers[task_name]
            feature_scaled = scaler.transform(feature_array)
            
            # Get predictions from each model
            for model_name, model in models.items():
                if model_name == 'neural_network':
                    # Neural network prediction
                    model.eval()
                    with torch.no_grad():
                        feature_tensor = torch.FloatTensor(feature_scaled).to(self.device)
                        outputs = model(feature_tensor)
                        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                        predicted_class = np.argmax(probabilities)
                else:
                    # Traditional ML model prediction
                    probabilities = model.predict_proba(feature_scaled)[0]
                    predicted_class = model.predict(feature_scaled)[0]
                
                # Decode class labels
                label_encoder = self.label_encoders[task_name]
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                
                # Get top predictions
                top_indices = np.argsort(probabilities)[::-1][:self.config.top_k_predictions]
                top_predictions = []
                
                for idx in top_indices:
                    label = label_encoder.inverse_transform([idx])[0]
                    probability = probabilities[idx]
                    if probability >= self.config.confidence_threshold:
                        top_predictions.append({
                            'label': label,
                            'probability': float(probability),
                            'confidence': 'high' if probability > 0.8 else 'medium' if probability > 0.6 else 'low'
                        })
                
                task_predictions[model_name] = {
                    'predicted_class': predicted_label,
                    'confidence': float(probabilities[predicted_class]),
                    'top_predictions': top_predictions,
                    'all_probabilities': {label_encoder.inverse_transform([i])[0]: float(probabilities[i]) 
                                        for i in range(len(probabilities))}
                }
            
            # Ensemble prediction
            ensemble_prediction = self._ensemble_predictions(task_predictions)
            task_predictions['ensemble'] = ensemble_prediction
            
            predictions[task_name] = task_predictions
        
        return predictions
    
    def _ensemble_predictions(self, model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble prediction from multiple models."""
        
        # Collect all predictions and probabilities
        all_predictions = []
        all_probabilities = {}
        
        for model_name, prediction in model_predictions.items():
            if model_name != 'ensemble':
                all_predictions.append(prediction['predicted_class'])
                
                # Aggregate probabilities
                for label, prob in prediction['all_probabilities'].items():
                    if label not in all_probabilities:
                        all_probabilities[label] = []
                    all_probabilities[label].append(prob)
        
        # Calculate ensemble probabilities
        ensemble_probabilities = {}
        for label, probs in all_probabilities.items():
            ensemble_probabilities[label] = np.mean(probs)
        
        # Get ensemble prediction
        ensemble_class = max(ensemble_probabilities.items(), key=lambda x: x[1])
        
        # Get top ensemble predictions
        top_indices = np.argsort(list(ensemble_probabilities.values()))[::-1][:self.config.top_k_predictions]
        top_predictions = []
        
        for idx in top_indices:
            label = list(ensemble_probabilities.keys())[idx]
            probability = ensemble_probabilities[label]
            if probability >= self.config.confidence_threshold:
                top_predictions.append({
                    'label': label,
                    'probability': float(probability),
                    'confidence': 'high' if probability > 0.8 else 'medium' if probability > 0.6 else 'low'
                })
        
        return {
            'predicted_class': ensemble_class[0],
            'confidence': float(ensemble_class[1]),
            'top_predictions': top_predictions,
            'all_probabilities': ensemble_probabilities,
            'model_agreement': len(set(all_predictions)) / len(all_predictions) if all_predictions else 0.0
        }
    
    def save_models(self, filepath: str):
        """Save trained models."""
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'config': asdict(self.config)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models."""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.label_encoders = model_data['label_encoders']
        
        logger.info(f"Models loaded from {filepath}")

# Example usage and testing
def test_neuronal_function_predictor():
    """Test the neuronal function prediction system."""
    
    # Configuration
    config = NeuronalFunctionConfig(
        use_random_forest=True,
        use_gradient_boosting=True,
        use_neural_network=True,
        use_xgboost=True,
        use_lightgbm=True,
        confidence_threshold=0.6,
        top_k_predictions=3
    )
    
    # Initialize predictor
    predictor = NeuronalFunctionPredictor(config)
    
    # Create sample training data
    training_data = []
    labels = {
        'neuron_type': [],
        'function': [],
        'molecular_identity': []
    }
    
    # Generate sample data (in practice, this would be real connectomics data)
    for i in range(100):
        # Create sample neuron data
        neuron_data = {
            'soma_position': (i, i, i),
            'dendrites': [{'path': [(i, i, i), (i+1, i+1, i+1)], 'features': {'path_length': 10}}],
            'axons': [{'path': [(i, i, i), (i+2, i+2, i+2)], 'features': {'path_length': 15}}],
            'spines': [{'position': (i, i, i), 'spine_type': 'mushroom'}],
            'connectivity': {'synaptic_contacts': [{'type': 'incoming', 'strength': 0.8}]}
        }
        
        training_data.append(neuron_data)
        
        # Assign sample labels
        labels['neuron_type'].append('pyramidal' if i % 3 == 0 else 'interneuron')
        labels['function'].append('excitatory' if i % 2 == 0 else 'inhibitory')
        labels['molecular_identity'].append('glutamatergic' if i % 2 == 0 else 'gabaergic')
    
    # Train models
    training_results = predictor.train_models(training_data, labels)
    
    # Test prediction
    test_neuron = {
        'soma_position': (50, 50, 50),
        'dendrites': [{'path': [(50, 50, 50), (51, 51, 51)], 'features': {'path_length': 12}}],
        'axons': [{'path': [(50, 50, 50), (52, 52, 52)], 'features': {'path_length': 18}}],
        'spines': [{'position': (50, 50, 50), 'spine_type': 'thin'}],
        'connectivity': {'synaptic_contacts': [{'type': 'outgoing', 'strength': 0.9}]}
    }
    
    predictions = predictor.predict_neuronal_function(test_neuron)
    
    # Print results
    print("Neuronal Function Predictions:")
    for task, task_predictions in predictions.items():
        print(f"\n{task.upper()}:")
        ensemble = task_predictions['ensemble']
        print(f"  Predicted: {ensemble['predicted_class']}")
        print(f"  Confidence: {ensemble['confidence']:.3f}")
        print(f"  Model Agreement: {ensemble['model_agreement']:.3f}")
        print("  Top Predictions:")
        for pred in ensemble['top_predictions']:
            print(f"    {pred['label']}: {pred['probability']:.3f} ({pred['confidence']})")
    
    return predictor, predictions

if __name__ == "__main__":
    predictor, predictions = test_neuronal_function_predictor() 