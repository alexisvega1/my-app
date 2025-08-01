#!/usr/bin/env python3
"""
Neuronal Projection Tracer and Spine Classifier
==============================================
Specialized algorithms for tracing neuronal projections and classifying dendritic spines
for comprehensive neural connectivity analysis.
"""

import os
import sys
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# JAX imports for optimization
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ProjectionTracingConfig:
    """Configuration for neuronal projection tracing."""
    # Tracing parameters
    min_projection_length: int = 10  # Minimum projection length in voxels
    max_projection_width: int = 5    # Maximum projection width in voxels
    connectivity_threshold: float = 0.7  # Threshold for synaptic connections
    direction_smoothing: float = 0.8  # Smoothing factor for direction vectors
    
    # Spine detection parameters
    spine_size_range: Tuple[int, int] = (3, 15)  # Min/max spine size in voxels
    spine_intensity_threshold: float = 0.6  # Intensity threshold for spine detection
    spine_shape_threshold: float = 0.8  # Shape similarity threshold
    
    # Classification parameters
    spine_types: List[str] = None
    classification_confidence_threshold: float = 0.8
    
    # Performance optimization
    use_gpu: bool = True
    batch_size: int = 32
    chunk_size: Tuple[int, int, int] = (512, 512, 512)
    
    def __post_init__(self):
        if self.spine_types is None:
            self.spine_types = [
                "mushroom",      # Large, stable spines
                "thin",          # Long, thin spines
                "stubby",        # Short, wide spines
                "filopodia",     # Long, dynamic spines
                "branched"       # Complex, branched spines
            ]

class NeuronalProjectionTracer:
    """Specialized tracer for neuronal projections and axonal/dendritic processes."""
    
    def __init__(self, config: ProjectionTracingConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Initialize projection models
        self.direction_predictor = self._build_direction_predictor()
        self.connection_detector = self._build_connection_detector()
        
        logger.info(f"Neuronal projection tracer initialized on {self.device}")
    
    def _build_direction_predictor(self) -> nn.Module:
        """Build neural network for predicting projection direction."""
        model = nn.Sequential(
            # 3D convolutional layers for local context
            nn.Conv3d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool3d(1),
            
            # Direction prediction (3D vector)
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # x, y, z direction
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        return model.to(self.device)
    
    def _build_connection_detector(self) -> nn.Module:
        """Build neural network for detecting synaptic connections."""
        model = nn.Sequential(
            # Feature extraction
            nn.Conv3d(2, 32, kernel_size=5, padding=2),  # 2 channels: pre/post synaptic
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool3d(1),
            
            # Connection probability
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        return model.to(self.device)
    
    def trace_projection(self, volume: np.ndarray, start_point: Tuple[int, int, int],
                        projection_type: str = "auto") -> Dict[str, Any]:
        """Trace neuronal projection from start point."""
        
        logger.info(f"Tracing projection from {start_point}")
        
        # Initialize tracing
        path = [start_point]
        visited = set([start_point])
        current_point = np.array(start_point, dtype=np.float32)
        
        # Determine projection type if auto
        if projection_type == "auto":
            projection_type = self._classify_projection_type(volume, start_point)
        
        # Tracing parameters based on projection type
        tracing_params = self._get_tracing_parameters(projection_type)
        
        max_steps = tracing_params['max_steps']
        step_size = tracing_params['step_size']
        direction_weight = tracing_params['direction_weight']
        
        for step in range(max_steps):
            # Get local context around current point
            context = self._extract_local_context(volume, current_point)
            
            # Predict next direction
            direction = self._predict_direction(context)
            
            # Apply projection type constraints
            direction = self._apply_projection_constraints(direction, projection_type)
            
            # Smooth direction with previous direction
            if len(path) > 1:
                prev_direction = np.array(path[-1]) - np.array(path[-2])
                prev_direction = prev_direction / (np.linalg.norm(prev_direction) + 1e-8)
                direction = (direction_weight * direction + 
                           (1 - direction_weight) * prev_direction)
                direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Calculate next point
            next_point = current_point + step_size * direction
            next_point = np.round(next_point).astype(int)
            
            # Check bounds and validity
            if not self._is_valid_point(volume, next_point):
                break
            
            # Check if already visited
            next_point_tuple = tuple(next_point)
            if next_point_tuple in visited:
                break
            
            # Check intensity threshold
            if volume[next_point_tuple] < tracing_params['intensity_threshold']:
                break
            
            # Add to path
            path.append(next_point_tuple)
            visited.add(next_point_tuple)
            current_point = next_point.astype(np.float32)
            
            # Check for termination conditions
            if self._should_terminate(volume, next_point, projection_type):
                break
        
        # Post-process path
        smoothed_path = self._smooth_projection_path(path)
        
        # Extract projection features
        features = self._extract_projection_features(volume, smoothed_path, projection_type)
        
        return {
            'path': smoothed_path,
            'projection_type': projection_type,
            'features': features,
            'length': len(smoothed_path),
            'start_point': start_point,
            'end_point': smoothed_path[-1] if smoothed_path else start_point
        }
    
    def _classify_projection_type(self, volume: np.ndarray, 
                                 point: Tuple[int, int, int]) -> str:
        """Classify projection type based on local morphology."""
        
        # Extract local context
        context = self._extract_local_context(volume, point, size=21)
        
        # Calculate morphological features
        features = {
            'intensity': np.mean(context),
            'variance': np.var(context),
            'skewness': self._calculate_skewness(context),
            'elongation': self._calculate_elongation(context),
            'branching': self._calculate_branching_factor(context)
        }
        
        # Classification logic based on features
        if features['elongation'] > 0.8 and features['intensity'] > 0.7:
            return "axon"
        elif features['branching'] > 0.6:
            return "dendrite"
        elif features['intensity'] > 0.8:
            return "spine"
        else:
            return "unknown"
    
    def _get_tracing_parameters(self, projection_type: str) -> Dict[str, Any]:
        """Get tracing parameters for specific projection type."""
        
        params = {
            'axon': {
                'max_steps': 1000,
                'step_size': 1.0,
                'direction_weight': 0.9,
                'intensity_threshold': 0.5,
                'width_constraint': 2.0
            },
            'dendrite': {
                'max_steps': 500,
                'step_size': 1.0,
                'direction_weight': 0.7,
                'intensity_threshold': 0.6,
                'width_constraint': 3.0
            },
            'spine': {
                'max_steps': 50,
                'step_size': 0.5,
                'direction_weight': 0.8,
                'intensity_threshold': 0.7,
                'width_constraint': 1.5
            },
            'unknown': {
                'max_steps': 200,
                'step_size': 1.0,
                'direction_weight': 0.8,
                'intensity_threshold': 0.5,
                'width_constraint': 2.5
            }
        }
        
        return params.get(projection_type, params['unknown'])
    
    def _extract_local_context(self, volume: np.ndarray, 
                              point: np.ndarray, size: int = 11) -> np.ndarray:
        """Extract local context around a point."""
        
        z, y, x = point.astype(int)
        half_size = size // 2
        
        # Extract 3D context
        context = volume[
            max(0, z - half_size):min(volume.shape[0], z + half_size + 1),
            max(0, y - half_size):min(volume.shape[1], y + half_size + 1),
            max(0, x - half_size):min(volume.shape[2], x + half_size + 1)
        ]
        
        # Pad if necessary
        if context.shape != (size, size, size):
            padded = np.zeros((size, size, size), dtype=volume.dtype)
            padded[:context.shape[0], :context.shape[1], :context.shape[2]] = context
            context = padded
        
        return context
    
    def _predict_direction(self, context: np.ndarray) -> np.ndarray:
        """Predict next direction using neural network."""
        
        # Prepare input
        context_tensor = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(0)
        context_tensor = context_tensor.to(self.device)
        
        # Predict direction
        with torch.no_grad():
            direction = self.direction_predictor(context_tensor)
            direction = direction.cpu().numpy().flatten()
        
        # Normalize direction vector
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        return direction
    
    def _apply_projection_constraints(self, direction: np.ndarray, 
                                     projection_type: str) -> np.ndarray:
        """Apply constraints based on projection type."""
        
        if projection_type == "axon":
            # Axons prefer straight paths
            return direction
        
        elif projection_type == "dendrite":
            # Dendrites can branch and curve
            return direction
        
        elif projection_type == "spine":
            # Spines are short and directed
            return direction
        
        return direction
    
    def _is_valid_point(self, volume: np.ndarray, point: np.ndarray) -> bool:
        """Check if point is within volume bounds."""
        
        return (0 <= point[0] < volume.shape[0] and
                0 <= point[1] < volume.shape[1] and
                0 <= point[2] < volume.shape[2])
    
    def _should_terminate(self, volume: np.ndarray, point: np.ndarray, 
                         projection_type: str) -> bool:
        """Check if tracing should terminate."""
        
        # Check for synaptic terminals
        if self._is_synaptic_terminal(volume, point):
            return True
        
        # Check for branching points
        if self._is_branching_point(volume, point):
            return True
        
        # Check for intensity drop
        if volume[tuple(point)] < 0.3:
            return True
        
        return False
    
    def _is_synaptic_terminal(self, volume: np.ndarray, point: np.ndarray) -> bool:
        """Detect synaptic terminals."""
        
        # Extract local context
        context = self._extract_local_context(volume, point, size=7)
        
        # Check for terminal-like morphology
        center_intensity = context[3, 3, 3]
        surrounding_intensity = np.mean(context)
        
        # Terminal should have high center intensity and lower surrounding
        return center_intensity > 0.8 and center_intensity > surrounding_intensity * 1.5
    
    def _is_branching_point(self, volume: np.ndarray, point: np.ndarray) -> bool:
        """Detect branching points."""
        
        # Extract local context
        context = self._extract_local_context(volume, point, size=9)
        
        # Calculate branching factor
        branching_factor = self._calculate_branching_factor(context)
        
        return branching_factor > 0.7
    
    def _smooth_projection_path(self, path: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Smooth projection path using spline interpolation."""
        
        if len(path) < 3:
            return path
        
        # Convert to numpy array
        path_array = np.array(path)
        
        # Apply smoothing
        smoothed = ndimage.gaussian_filter1d(path_array, sigma=0.5, axis=0)
        
        # Convert back to list of tuples
        return [tuple(point.astype(int)) for point in smoothed]
    
    def _extract_projection_features(self, volume: np.ndarray, 
                                   path: List[Tuple[int, int, int]], 
                                   projection_type: str) -> Dict[str, Any]:
        """Extract features from traced projection."""
        
        if not path:
            return {}
        
        # Calculate basic features
        path_array = np.array(path)
        length = len(path)
        
        # Calculate path length
        path_length = 0
        for i in range(1, len(path)):
            path_length += np.linalg.norm(path_array[i] - path_array[i-1])
        
        # Calculate curvature
        curvature = self._calculate_path_curvature(path_array)
        
        # Calculate intensity profile
        intensities = [volume[point] for point in path]
        intensity_profile = {
            'mean': np.mean(intensities),
            'std': np.std(intensities),
            'min': np.min(intensities),
            'max': np.max(intensities)
        }
        
        # Calculate direction changes
        direction_changes = self._calculate_direction_changes(path_array)
        
        return {
            'length': length,
            'path_length': path_length,
            'curvature': curvature,
            'intensity_profile': intensity_profile,
            'direction_changes': direction_changes,
            'projection_type': projection_type
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_elongation(self, data: np.ndarray) -> float:
        """Calculate elongation factor of 3D data."""
        # Calculate principal components
        coords = np.where(data > np.mean(data))
        if len(coords[0]) < 3:
            return 0
        
        points = np.column_stack(coords)
        if len(points) < 3:
            return 0
        
        # Calculate covariance matrix
        cov_matrix = np.cov(points.T)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        if len(eigenvalues) < 2:
            return 0
        
        # Elongation is ratio of largest to smallest eigenvalue
        return np.max(eigenvalues) / np.min(eigenvalues)
    
    def _calculate_branching_factor(self, data: np.ndarray) -> float:
        """Calculate branching factor of 3D data."""
        # Threshold data
        threshold = np.mean(data) + np.std(data)
        binary_data = data > threshold
        
        # Calculate number of connected components
        labeled, num_features = ndimage.label(binary_data)
        
        # Branching factor is number of components
        return min(num_features / 10.0, 1.0)  # Normalize
    
    def _calculate_path_curvature(self, path: np.ndarray) -> float:
        """Calculate curvature of path."""
        if len(path) < 3:
            return 0
        
        curvatures = []
        for i in range(1, len(path) - 1):
            # Calculate curvature at each point
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            
            # Normalize vectors
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
            
            # Calculate angle
            cos_angle = np.dot(v1_norm, v2_norm)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            
            curvatures.append(angle)
        
        return np.mean(curvatures) if curvatures else 0
    
    def _calculate_direction_changes(self, path: np.ndarray) -> List[float]:
        """Calculate direction changes along path."""
        if len(path) < 2:
            return []
        
        direction_changes = []
        for i in range(1, len(path)):
            direction = path[i] - path[i-1]
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            if i > 1:
                prev_direction = path[i-1] - path[i-2]
                prev_direction = prev_direction / (np.linalg.norm(prev_direction) + 1e-8)
                
                # Calculate angle change
                cos_angle = np.dot(direction, prev_direction)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.arccos(cos_angle)
                direction_changes.append(angle_change)
        
        return direction_changes

class DendriticSpineClassifier:
    """Specialized classifier for dendritic spine types."""
    
    def __init__(self, config: ProjectionTracingConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Initialize spine classification model
        self.spine_classifier = self._build_spine_classifier()
        
        # Spine type templates
        self.spine_templates = self._load_spine_templates()
        
        logger.info(f"Dendritic spine classifier initialized on {self.device}")
    
    def _build_spine_classifier(self) -> nn.Module:
        """Build neural network for spine classification."""
        model = nn.Sequential(
            # 3D convolutional layers
            nn.Conv3d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            
            # Classification head
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, len(self.config.spine_types)),
            nn.Softmax(dim=1)
        )
        
        return model.to(self.device)
    
    def _load_spine_templates(self) -> Dict[str, np.ndarray]:
        """Load spine type templates."""
        
        templates = {}
        
        # Mushroom spine template (large, round head)
        mushroom = np.zeros((15, 15, 15))
        mushroom[5:10, 5:10, 5:10] = 1  # Head
        mushroom[7, 7, 10:13] = 1       # Neck
        templates['mushroom'] = mushroom
        
        # Thin spine template (long, thin)
        thin = np.zeros((15, 15, 15))
        thin[7, 7, 3:12] = 1            # Long shaft
        thin[7, 7, 12:15] = 0.5         # Small head
        templates['thin'] = thin
        
        # Stubby spine template (short, wide)
        stubby = np.zeros((15, 15, 15))
        stubby[6:9, 6:9, 8:12] = 1      # Short, wide structure
        templates['stubby'] = stubby
        
        # Filopodia template (very long, thin)
        filopodia = np.zeros((15, 15, 15))
        filopodia[7, 7, 2:13] = 1       # Very long shaft
        templates['filopodia'] = filopodia
        
        # Branched spine template (complex)
        branched = np.zeros((15, 15, 15))
        branched[7, 7, 8:12] = 1        # Main shaft
        branched[5:9, 7, 10] = 1        # Branch 1
        branched[7, 5:9, 10] = 1        # Branch 2
        templates['branched'] = branched
        
        return templates
    
    def detect_spines(self, volume: np.ndarray, dendrite_path: List[Tuple[int, int, int]]) -> List[Dict[str, Any]]:
        """Detect dendritic spines along dendrite path."""
        
        logger.info(f"Detecting spines along dendrite path of length {len(dendrite_path)}")
        
        detected_spines = []
        
        for i, point in enumerate(dendrite_path):
            # Extract local context around dendrite point
            context = self._extract_spine_context(volume, point)
            
            # Check if this point has a spine
            if self._is_spine_point(context):
                # Extract spine region
                spine_region = self._extract_spine_region(volume, point)
                
                # Classify spine type
                spine_type, confidence = self._classify_spine(spine_region)
                
                # Extract spine features
                features = self._extract_spine_features(spine_region, spine_type)
                
                detected_spines.append({
                    'position': point,
                    'spine_type': spine_type,
                    'confidence': confidence,
                    'features': features,
                    'region': spine_region
                })
        
        return detected_spines
    
    def _extract_spine_context(self, volume: np.ndarray, 
                              point: Tuple[int, int, int], 
                              size: int = 21) -> np.ndarray:
        """Extract context around potential spine point."""
        
        z, y, x = point
        half_size = size // 2
        
        # Extract 3D context
        context = volume[
            max(0, z - half_size):min(volume.shape[0], z + half_size + 1),
            max(0, y - half_size):min(volume.shape[1], y + half_size + 1),
            max(0, x - half_size):min(volume.shape[2], x + half_size + 1)
        ]
        
        # Pad if necessary
        if context.shape != (size, size, size):
            padded = np.zeros((size, size, size), dtype=volume.dtype)
            padded[:context.shape[0], :context.shape[1], :context.shape[2]] = context
            context = padded
        
        return context
    
    def _is_spine_point(self, context: np.ndarray) -> bool:
        """Check if context contains a spine."""
        
        # Calculate spine-like features
        center_intensity = context[10, 10, 10]  # Center point
        surrounding_intensity = np.mean(context)
        
        # Check for spine-like morphology
        intensity_ratio = center_intensity / (surrounding_intensity + 1e-8)
        
        # Check for protrusion
        protrusion_score = self._calculate_protrusion_score(context)
        
        return (intensity_ratio > 1.5 and 
                protrusion_score > 0.6 and 
                center_intensity > self.config.spine_intensity_threshold)
    
    def _calculate_protrusion_score(self, context: np.ndarray) -> float:
        """Calculate how much the center protrudes from surrounding."""
        
        center = context[10, 10, 10]
        center_region = context[8:13, 8:13, 8:13]
        surrounding_region = context[5:16, 5:16, 5:16]
        
        center_mean = np.mean(center_region)
        surrounding_mean = np.mean(surrounding_region)
        
        if surrounding_mean == 0:
            return 0
        
        return (center_mean - surrounding_mean) / surrounding_mean
    
    def _extract_spine_region(self, volume: np.ndarray, 
                             point: Tuple[int, int, int]) -> np.ndarray:
        """Extract the spine region around a point."""
        
        z, y, x = point
        size = 15
        
        # Extract spine region
        spine_region = volume[
            max(0, z - size//2):min(volume.shape[0], z + size//2 + 1),
            max(0, y - size//2):min(volume.shape[1], y + size//2 + 1),
            max(0, x - size//2):min(volume.shape[2], x + size//2 + 1)
        ]
        
        # Pad if necessary
        if spine_region.shape != (size, size, size):
            padded = np.zeros((size, size, size), dtype=volume.dtype)
            padded[:spine_region.shape[0], :spine_region.shape[1], :spine_region.shape[2]] = spine_region
            spine_region = padded
        
        return spine_region
    
    def _classify_spine(self, spine_region: np.ndarray) -> Tuple[str, float]:
        """Classify spine type using neural network."""
        
        # Prepare input
        spine_tensor = torch.from_numpy(spine_region).float().unsqueeze(0).unsqueeze(0)
        spine_tensor = spine_tensor.to(self.device)
        
        # Classify
        with torch.no_grad():
            predictions = self.spine_classifier(spine_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        # Get best prediction
        best_idx = np.argmax(predictions)
        spine_type = self.config.spine_types[best_idx]
        confidence = predictions[best_idx]
        
        return spine_type, confidence
    
    def _extract_spine_features(self, spine_region: np.ndarray, 
                               spine_type: str) -> Dict[str, Any]:
        """Extract morphological features from spine."""
        
        # Calculate basic features
        volume = np.sum(spine_region > 0.5)
        surface_area = self._calculate_surface_area(spine_region)
        
        # Calculate shape features
        elongation = self._calculate_elongation(spine_region)
        sphericity = self._calculate_sphericity(volume, surface_area)
        
        # Calculate intensity features
        intensity_profile = {
            'mean': np.mean(spine_region),
            'std': np.std(spine_region),
            'max': np.max(spine_region),
            'min': np.min(spine_region)
        }
        
        # Calculate position features
        center_of_mass = ndimage.center_of_mass(spine_region)
        
        return {
            'volume': volume,
            'surface_area': surface_area,
            'elongation': elongation,
            'sphericity': sphericity,
            'intensity_profile': intensity_profile,
            'center_of_mass': center_of_mass,
            'spine_type': spine_type
        }
    
    def _calculate_surface_area(self, region: np.ndarray) -> float:
        """Calculate surface area of spine region."""
        
        # Use morphological operations to find surface
        binary_region = region > 0.5
        eroded = ndimage.binary_erosion(binary_region)
        surface = binary_region & ~eroded
        
        return np.sum(surface)
    
    def _calculate_sphericity(self, volume: float, surface_area: float) -> float:
        """Calculate sphericity of spine."""
        
        if surface_area == 0:
            return 0
        
        # Sphericity = (π^(1/3) * (6V)^(2/3)) / A
        theoretical_surface = np.pi**(1/3) * (6 * volume)**(2/3)
        return theoretical_surface / surface_area

class ComprehensiveNeuralTracer:
    """Comprehensive neural tracing system combining projection tracing and spine classification."""
    
    def __init__(self, config: ProjectionTracingConfig):
        self.config = config
        
        # Initialize components
        self.projection_tracer = NeuronalProjectionTracer(config)
        self.spine_classifier = DendriticSpineClassifier(config)
        
        logger.info("Comprehensive neural tracer initialized")
    
    def trace_complete_neuron(self, volume: np.ndarray, 
                             soma_point: Tuple[int, int, int]) -> Dict[str, Any]:
        """Trace complete neuron including all projections and spines."""
        
        logger.info(f"Tracing complete neuron from soma at {soma_point}")
        
        # Trace dendrites
        dendrites = self._trace_dendrites(volume, soma_point)
        
        # Trace axons
        axons = self._trace_axons(volume, soma_point)
        
        # Detect and classify spines on dendrites
        all_spines = []
        for dendrite in dendrites:
            spines = self.spine_classifier.detect_spines(volume, dendrite['path'])
            all_spines.extend(spines)
        
        # Analyze connectivity
        connectivity = self._analyze_connectivity(volume, dendrites, axons)
        
        return {
            'soma_position': soma_point,
            'dendrites': dendrites,
            'axons': axons,
            'spines': all_spines,
            'connectivity': connectivity,
            'neuron_features': self._extract_neuron_features(dendrites, axons, all_spines)
        }
    
    def _trace_dendrites(self, volume: np.ndarray, 
                        soma_point: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Trace dendritic branches from soma."""
        
        dendrites = []
        
        # Find dendritic initiation points around soma
        init_points = self._find_dendritic_initiation_points(volume, soma_point)
        
        # Trace each dendritic branch
        for init_point in init_points:
            dendrite = self.projection_tracer.trace_projection(
                volume, init_point, "dendrite"
            )
            dendrites.append(dendrite)
        
        return dendrites
    
    def _trace_axons(self, volume: np.ndarray, 
                     soma_point: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Trace axonal projections from soma."""
        
        axons = []
        
        # Find axonal initiation points around soma
        init_points = self._find_axonal_initiation_points(volume, soma_point)
        
        # Trace each axonal branch
        for init_point in init_points:
            axon = self.projection_tracer.trace_projection(
                volume, init_point, "axon"
            )
            axons.append(axon)
        
        return axons
    
    def _find_dendritic_initiation_points(self, volume: np.ndarray, 
                                         soma_point: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Find dendritic initiation points around soma."""
        
        z, y, x = soma_point
        search_radius = 10
        
        init_points = []
        
        # Search in spherical region around soma
        for dz in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    point = (z + dz, y + dy, x + dx)
                    
                    # Check bounds
                    if not self.projection_tracer._is_valid_point(volume, np.array(point)):
                        continue
                    
                    # Check if point is dendritic initiation
                    if self._is_dendritic_initiation(volume, point):
                        init_points.append(point)
        
        return init_points
    
    def _find_axonal_initiation_points(self, volume: np.ndarray, 
                                      soma_point: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Find axonal initiation points around soma."""
        
        z, y, x = soma_point
        search_radius = 8
        
        init_points = []
        
        # Search in spherical region around soma
        for dz in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    point = (z + dz, y + dy, x + dx)
                    
                    # Check bounds
                    if not self.projection_tracer._is_valid_point(volume, np.array(point)):
                        continue
                    
                    # Check if point is axonal initiation
                    if self._is_axonal_initiation(volume, point):
                        init_points.append(point)
        
        return init_points
    
    def _is_dendritic_initiation(self, volume: np.ndarray, 
                                point: Tuple[int, int, int]) -> bool:
        """Check if point is dendritic initiation."""
        
        # Extract local context
        context = self.projection_tracer._extract_local_context(volume, np.array(point))
        
        # Dendritic initiation features
        intensity = volume[point]
        branching = self.projection_tracer._calculate_branching_factor(context)
        
        return (intensity > 0.6 and branching > 0.3)
    
    def _is_axonal_initiation(self, volume: np.ndarray, 
                             point: Tuple[int, int, int]) -> bool:
        """Check if point is axonal initiation."""
        
        # Extract local context
        context = self.projection_tracer._extract_local_context(volume, np.array(point))
        
        # Axonal initiation features
        intensity = volume[point]
        elongation = self.projection_tracer._calculate_elongation(context)
        
        return (intensity > 0.7 and elongation > 0.8)
    
    def _analyze_connectivity(self, volume: np.ndarray, 
                             dendrites: List[Dict[str, Any]], 
                             axons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze synaptic connectivity between neurons."""
        
        connectivity = {
            'synaptic_contacts': [],
            'connection_strength': 0.0,
            'connectivity_matrix': None
        }
        
        # Find synaptic contacts between dendrites and axons
        for dendrite in dendrites:
            for axon in axons:
                contacts = self._find_synaptic_contacts(volume, dendrite, axon)
                connectivity['synaptic_contacts'].extend(contacts)
        
        # Calculate connection strength
        if connectivity['synaptic_contacts']:
            connectivity['connection_strength'] = len(connectivity['synaptic_contacts'])
        
        return connectivity
    
    def _find_synaptic_contacts(self, volume: np.ndarray, 
                               dendrite: Dict[str, Any], 
                               axon: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find synaptic contacts between dendrite and axon."""
        
        contacts = []
        
        # Check for proximity between dendrite and axon paths
        for dendrite_point in dendrite['path']:
            for axon_point in axon['path']:
                distance = np.linalg.norm(np.array(dendrite_point) - np.array(axon_point))
                
                if distance < 3:  # Close proximity
                    # Check for synaptic morphology
                    if self._is_synaptic_contact(volume, dendrite_point, axon_point):
                        contacts.append({
                            'dendrite_point': dendrite_point,
                            'axon_point': axon_point,
                            'distance': distance,
                            'strength': self._calculate_synaptic_strength(volume, dendrite_point, axon_point)
                        })
        
        return contacts
    
    def _is_synaptic_contact(self, volume: np.ndarray, 
                            dendrite_point: Tuple[int, int, int], 
                            axon_point: Tuple[int, int, int]) -> bool:
        """Check if two points form a synaptic contact."""
        
        # Extract local context around both points
        dendrite_context = self.projection_tracer._extract_local_context(volume, np.array(dendrite_point))
        axon_context = self.projection_tracer._extract_local_context(volume, np.array(axon_point))
        
        # Check for synaptic morphology
        dendrite_intensity = volume[dendrite_point]
        axon_intensity = volume[axon_point]
        
        # Synaptic contacts should have high intensity at both points
        return (dendrite_intensity > 0.7 and axon_intensity > 0.7)
    
    def _calculate_synaptic_strength(self, volume: np.ndarray, 
                                   dendrite_point: Tuple[int, int, int], 
                                   axon_point: Tuple[int, int, int]) -> float:
        """Calculate synaptic strength between two points."""
        
        # Extract local context
        context = self.projection_tracer._extract_local_context(volume, np.array(dendrite_point))
        
        # Calculate features related to synaptic strength
        intensity = volume[dendrite_point]
        volume_density = np.sum(context > 0.5) / context.size
        
        # Synaptic strength is combination of intensity and volume density
        strength = (intensity * 0.7 + volume_density * 0.3)
        
        return min(strength, 1.0)
    
    def _extract_neuron_features(self, dendrites: List[Dict[str, Any]], 
                                axons: List[Dict[str, Any]], 
                                spines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract comprehensive features from traced neuron."""
        
        # Calculate dendritic features
        total_dendritic_length = sum(d['features']['path_length'] for d in dendrites)
        dendritic_complexity = len(dendrites)
        
        # Calculate axonal features
        total_axonal_length = sum(a['features']['path_length'] for a in axons)
        axonal_complexity = len(axons)
        
        # Calculate spine features
        spine_counts = {}
        for spine_type in self.config.spine_types:
            spine_counts[spine_type] = len([s for s in spines if s['spine_type'] == spine_type])
        
        total_spines = len(spines)
        spine_density = total_spines / max(total_dendritic_length, 1)
        
        return {
            'dendritic_features': {
                'total_length': total_dendritic_length,
                'complexity': dendritic_complexity,
                'average_length': total_dendritic_length / max(dendritic_complexity, 1)
            },
            'axonal_features': {
                'total_length': total_axonal_length,
                'complexity': axonal_complexity,
                'average_length': total_axonal_length / max(axonal_complexity, 1)
            },
            'spine_features': {
                'total_count': total_spines,
                'density': spine_density,
                'type_distribution': spine_counts
            },
            'neuron_complexity': dendritic_complexity + axonal_complexity,
            'total_connectivity': total_spines + len(axons)
        }

# Example usage
def test_neural_tracer():
    """Test the comprehensive neural tracing system."""
    
    # Configuration
    config = ProjectionTracingConfig(
        min_projection_length=10,
        max_projection_width=5,
        connectivity_threshold=0.7,
        spine_size_range=(3, 15),
        spine_intensity_threshold=0.6
    )
    
    # Initialize tracer
    tracer = ComprehensiveNeuralTracer(config)
    
    # Create sample volume (in practice, this would be real data)
    volume = np.random.random((100, 100, 100))
    
    # Trace complete neuron
    soma_point = (50, 50, 50)
    neuron_data = tracer.trace_complete_neuron(volume, soma_point)
    
    # Print results
    print(f"Traced neuron with {len(neuron_data['dendrites'])} dendrites")
    print(f"Found {len(neuron_data['axons'])} axons")
    print(f"Detected {len(neuron_data['spines'])} spines")
    print(f"Neuron complexity: {neuron_data['neuron_features']['neuron_complexity']}")
    
    return neuron_data

if __name__ == "__main__":
    test_neural_tracer() 