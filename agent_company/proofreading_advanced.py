#!/usr/bin/env python3
"""
Advanced Proofreading System for Production Connectomics
======================================================
Next-generation proofreading with sophisticated error detection,
correction algorithms, and production monitoring for petabyte-scale datasets.
"""

import os
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import json
import hashlib
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Production-grade imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
    logger.info("PyTorch available for advanced proofreading")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using stub implementation")

try:
    import dask.array as da
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask not available - distributed processing disabled")

try:
    import zarr
    import numcodecs
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    logger.warning("Zarr not available - chunked storage disabled")

try:
    from google.cloud import firestore
    from google.cloud import storage
    FIRESTORE_AVAILABLE = True
    logger.info("Firestore available for advanced proofreading")
except ImportError:
    FIRESTORE_AVAILABLE = False
    logger.warning("Firestore not available - using stub client")

@dataclass
class ProofreadingResult:
    """Advanced proofreading result with comprehensive metadata."""
    corrected_segmentation: np.ndarray
    error_map: np.ndarray
    confidence_scores: np.ndarray
    correction_metadata: Dict[str, Any]
    processing_time: float
    memory_usage: Dict[str, float]
    quality_metrics: Dict[str, float]
    distributed_info: Dict[str, Any]

class ErrorDetector:
    """Advanced error detection using multiple algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_methods = {
            'topology': self._detect_topological_errors,
            'morphology': self._detect_morphological_errors,
            'consistency': self._detect_consistency_errors,
            'boundary': self._detect_boundary_errors,
            'connectivity': self._detect_connectivity_errors
        }
        
        logger.info("Error detector initialized")
    
    def detect_errors(self, segmentation: np.ndarray, 
                     uncertainty_map: np.ndarray,
                     metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Detect errors using multiple methods."""
        errors = {}
        
        for method_name, method_func in self.detection_methods.items():
            try:
                if self.config.get(f'use_{method_name}', True):
                    errors[method_name] = method_func(segmentation, uncertainty_map, metadata)
                    logger.debug(f"Applied {method_name} error detection")
            except Exception as e:
                logger.error(f"Error detection method {method_name} failed: {e}")
                errors[method_name] = np.zeros_like(segmentation, dtype=bool)
        
        return errors
    
    def _detect_topological_errors(self, segmentation: np.ndarray, 
                                  uncertainty_map: np.ndarray,
                                  metadata: Dict[str, Any]) -> np.ndarray:
        """Detect topological errors (holes, handles, etc.)."""
        from scipy import ndimage
        
        # Label connected components
        labeled, num_components = ndimage.label(segmentation > 0.5)
        
        # Detect holes in each component
        holes = np.zeros_like(segmentation, dtype=bool)
        for component_id in range(1, num_components + 1):
            component_mask = labeled == component_id
            filled = ndimage.binary_fill_holes(component_mask)
            holes |= (filled & ~component_mask)
        
        # Combine with uncertainty
        high_uncertainty = uncertainty_map > 0.7
        topological_errors = holes | high_uncertainty
        
        return topological_errors
    
    def _detect_morphological_errors(self, segmentation: np.ndarray,
                                   uncertainty_map: np.ndarray,
                                   metadata: Dict[str, Any]) -> np.ndarray:
        """Detect morphological errors in segmentation."""
        try:
            # Use skimage for skeletonization if available
            try:
                from skimage.morphology import skeletonize_3d
                skeleton = skeletonize_3d(segmentation > 0.5)
            except ImportError:
                # Fallback to scipy if skimage not available
                try:
                    from scipy.ndimage import binary_skeletonize
                    skeleton = binary_skeletonize(segmentation > 0.5)
                except AttributeError:
                    # If neither available, use a simple edge detection
                    logger.warning("Skeletonization not available, using edge detection")
                    skeleton = np.zeros_like(segmentation, dtype=bool)
                    for axis in range(3):
                        skeleton |= np.abs(np.diff(segmentation > 0.5, axis=axis, prepend=0))
            
            # Detect thin structures that might be errors
            thin_structures = skeleton & (uncertainty_map > 0.3)
            
            # Detect isolated small components
            from scipy import ndimage
            labeled, num_features = ndimage.label(segmentation > 0.5)
            min_size = metadata.get('min_component_size', 100)
            
            isolated_components = np.zeros_like(segmentation, dtype=bool)
            for i in range(1, num_features + 1):
                component_size = np.sum(labeled == i)
                if component_size < min_size:
                    isolated_components |= (labeled == i)
            
            return thin_structures | isolated_components
            
        except Exception as e:
            logger.error(f"Error detection method morphology failed: {e}")
            return np.zeros_like(segmentation, dtype=bool)
    
    def _detect_consistency_errors(self, segmentation: np.ndarray,
                                 uncertainty_map: np.ndarray,
                                 metadata: Dict[str, Any]) -> np.ndarray:
        """Detect consistency errors across the volume."""
        # Check for sudden changes in segmentation
        gradients = np.zeros_like(segmentation, dtype=float)
        for axis in range(3):
            grad = np.abs(np.diff(segmentation, axis=axis, prepend=0))
            gradients = np.maximum(gradients, grad)
        
        # High gradient regions with high uncertainty
        consistency_errors = (gradients > 0.8) & (uncertainty_map > 0.6)
        return consistency_errors
    
    def _detect_boundary_errors(self, segmentation: np.ndarray,
                              uncertainty_map: np.ndarray,
                              metadata: Dict[str, Any]) -> np.ndarray:
        """Detect boundary errors."""
        from scipy import ndimage
        
        # Find boundaries
        boundaries = ndimage.binary_erosion(segmentation > 0.5)
        boundaries = (segmentation > 0.5) & ~boundaries
        
        # High uncertainty at boundaries
        boundary_errors = boundaries & (uncertainty_map > 0.5)
        return boundary_errors
    
    def _detect_connectivity_errors(self, segmentation: np.ndarray,
                                  uncertainty_map: np.ndarray,
                                  metadata: Dict[str, Any]) -> np.ndarray:
        """Detect connectivity errors."""
        from scipy import ndimage
        
        # Check for disconnected components that should be connected
        labeled, num_components = ndimage.label(segmentation > 0.5)
        
        # Find components that are close but not connected
        connectivity_errors = np.zeros_like(segmentation, dtype=bool)
        
        if num_components > 1:
            # Simple heuristic: components within distance threshold
            for i in range(1, num_components + 1):
                for j in range(i + 1, num_components + 1):
                    comp_i = labeled == i
                    comp_j = labeled == j
                    
                    # Calculate distance between components
                    dist_i = ndimage.distance_transform_edt(~comp_i)
                    dist_j = ndimage.distance_transform_edt(~comp_j)
                    
                    # Check if components are close
                    min_dist = np.min(dist_i[comp_j])
                    if min_dist < self.config.get('connectivity_threshold', 5):
                        connectivity_errors |= (comp_i | comp_j) & (uncertainty_map > 0.4)
        
        return connectivity_errors

class ErrorCorrector:
    """Advanced error correction using multiple strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.correction_strategies = {
            'morphological': self._morphological_correction,
            'topological': self._topological_correction,
            'interpolation': self._interpolation_correction,
            'smoothing': self._smoothing_correction,
            'reconstruction': self._reconstruction_correction
        }
        
        logger.info("Error corrector initialized")
    
    def correct_errors(self, segmentation: np.ndarray,
                      error_map: np.ndarray,
                      uncertainty_map: np.ndarray,
                      metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Correct errors using multiple strategies."""
        corrected = segmentation.copy()
        correction_metadata = {}
        
        for strategy_name, strategy_func in self.correction_strategies.items():
            try:
                if self.config.get(f'use_{strategy_name}', True):
                    corrected, strategy_metadata = strategy_func(
                        corrected, error_map, uncertainty_map, metadata
                    )
                    correction_metadata[strategy_name] = strategy_metadata
                    logger.debug(f"Applied {strategy_name} correction")
            except Exception as e:
                logger.error(f"Correction strategy {strategy_name} failed: {e}")
                correction_metadata[strategy_name] = {'error': str(e)}
        
        return corrected, correction_metadata
    
    def _morphological_correction(self, segmentation: np.ndarray,
                                error_map: np.ndarray,
                                uncertainty_map: np.ndarray,
                                metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply morphological corrections."""
        from scipy import ndimage
        
        # Fill holes
        filled = ndimage.binary_fill_holes(segmentation > 0.5)
        
        # Remove small components
        labeled, num_components = ndimage.label(filled)
        min_size = self.config.get('min_component_size', 100)
        
        for component_id in range(1, num_components + 1):
            component_size = np.sum(labeled == component_id)
            if component_size < min_size:
                filled[labeled == component_id] = False
        
        # Apply morphological closing to smooth boundaries
        kernel_size = self.config.get('morphology_kernel_size', 3)
        kernel = np.ones((kernel_size, kernel_size, kernel_size))
        smoothed = ndimage.binary_closing(filled, structure=kernel)
        
        metadata = {
            'holes_filled': np.sum(filled & ~(segmentation > 0.5)),
            'small_components_removed': np.sum((segmentation > 0.5) & ~filled),
            'boundaries_smoothed': np.sum(smoothed & ~filled)
        }
        
        return smoothed.astype(np.float32), metadata
    
    def _topological_correction(self, segmentation: np.ndarray,
                              error_map: np.ndarray,
                              uncertainty_map: np.ndarray,
                              metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply topological corrections."""
        from scipy import ndimage
        
        # Ensure single connected component
        labeled, num_components = ndimage.label(segmentation > 0.5)
        
        if num_components > 1:
            # Find largest component
            component_sizes = [np.sum(labeled == i) for i in range(1, num_components + 1)]
            largest_component = np.argmax(component_sizes) + 1
            
            # Keep only largest component
            corrected = labeled == largest_component
        else:
            corrected = segmentation > 0.5
        
        metadata = {
            'components_merged': max(0, num_components - 1),
            'largest_component_size': np.sum(corrected)
        }
        
        return corrected.astype(np.float32), metadata
    
    def _interpolation_correction(self, segmentation: np.ndarray,
                                error_map: np.ndarray,
                                uncertainty_map: np.ndarray,
                                metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply interpolation corrections."""
        from scipy import ndimage
        
        # Interpolate uncertain regions
        uncertain_regions = uncertainty_map > 0.7
        corrected = segmentation.copy()
        
        if np.any(uncertain_regions):
            # Use distance-weighted interpolation
            distance_map = ndimage.distance_transform_edt(~uncertain_regions)
            weight_map = 1.0 / (1.0 + distance_map)
            
            # Interpolate values
            corrected[uncertain_regions] = (
                weight_map[uncertain_regions] * segmentation[uncertain_regions]
            )
        
        metadata = {
            'uncertain_regions_interpolated': np.sum(uncertain_regions),
            'interpolation_weight': np.mean(weight_map[uncertain_regions]) if np.any(uncertain_regions) else 0.0
        }
        
        return corrected, metadata
    
    def _smoothing_correction(self, segmentation: np.ndarray,
                            error_map: np.ndarray,
                            uncertainty_map: np.ndarray,
                            metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply smoothing corrections."""
        from scipy import ndimage
        
        # Apply Gaussian smoothing
        sigma = self.config.get('smoothing_sigma', 1.0)
        smoothed = ndimage.gaussian_filter(segmentation, sigma=sigma)
        
        # Threshold back to binary
        corrected = smoothed > 0.5
        
        metadata = {
            'smoothing_sigma': sigma,
            'smoothing_applied': True
        }
        
        return corrected.astype(np.float32), metadata
    
    def _reconstruction_correction(self, segmentation: np.ndarray,
                                 error_map: np.ndarray,
                                 uncertainty_map: np.ndarray,
                                 metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply reconstruction-based corrections."""
        from scipy import ndimage
        
        # Use watershed for reconstruction
        distance = ndimage.distance_transform_edt(segmentation > 0.5)
        markers = ndimage.label(segmentation > 0.5)[0]
        
        # Watershed reconstruction
        reconstructed = ndimage.watershed_ift(
            (distance.max() - distance).astype(np.uint8),
            markers
        )
        
        corrected = reconstructed > 0
        
        metadata = {
            'watershed_applied': True,
            'reconstruction_volume': np.sum(corrected)
        }
        
        return corrected.astype(np.float32), metadata

class AdvancedProofreader:
    """Production-ready advanced proofreading system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'error_detection': {
                'use_topology': True,
                'use_morphology': True,
                'use_consistency': True,
                'use_boundary': True,
                'use_connectivity': True,
                'min_component_size': 100,
                'connectivity_threshold': 5
            },
            'error_correction': {
                'use_morphological': True,
                'use_topological': True,
                'use_interpolation': True,
                'use_smoothing': True,
                'use_reconstruction': True,
                'morphology_kernel_size': 3,
                'smoothing_sigma': 1.0
            },
            'distributed_config': {
                'num_processes': mp.cpu_count(),
                'num_threads': 10,
                'chunk_size': (128, 128, 128)
            },
            'storage_config': {
                'compression': 'blosc',
                'cache_size': '4GB'
            }
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.error_detector = ErrorDetector(default_config['error_detection'])
        self.error_corrector = ErrorCorrector(default_config['error_correction'])
        
        # Performance tracking
        self.stats = {
            'volumes_processed': 0,
            'total_processing_time': 0.0,
            'total_errors_detected': 0,
            'total_errors_corrected': 0
        }
        
        logger.info("Advanced proofreader initialized")
    
    def proofread(self, 
                  segmentation: np.ndarray,
                  uncertainty_map: np.ndarray,
                  metadata: Optional[Dict[str, Any]] = None) -> ProofreadingResult:
        """Perform advanced proofreading on segmentation."""
        start_time = time.time()
        
        if metadata is None:
            metadata = {}
        
        try:
            # Detect errors
            logger.info("Detecting errors...")
            error_maps = self.error_detector.detect_errors(
                segmentation, uncertainty_map, metadata
            )
            
            # Combine error maps
            combined_error_map = np.zeros_like(segmentation, dtype=bool)
            for error_map in error_maps.values():
                combined_error_map |= error_map
            
            self.stats['total_errors_detected'] += np.sum(combined_error_map)
            
            # Correct errors
            logger.info("Correcting errors...")
            corrected_segmentation, correction_metadata = self.error_corrector.correct_errors(
                segmentation, combined_error_map, uncertainty_map, metadata
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                corrected_segmentation, uncertainty_map, error_maps
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                segmentation, corrected_segmentation, error_maps, correction_metadata
            )
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['volumes_processed'] += 1
            
            return ProofreadingResult(
                corrected_segmentation=corrected_segmentation,
                error_map=combined_error_map,
                confidence_scores=confidence_scores,
                correction_metadata=correction_metadata,
                processing_time=processing_time,
                memory_usage=self._get_memory_usage(),
                quality_metrics=quality_metrics,
                distributed_info={
                    'num_processes': self.config['distributed_config']['num_processes'],
                    'num_threads': self.config['distributed_config']['num_threads']
                }
            )
            
        except Exception as e:
            logger.error(f"Proofreading failed: {e}")
            raise
    
    def _calculate_confidence_scores(self, 
                                   corrected_segmentation: np.ndarray,
                                   uncertainty_map: np.ndarray,
                                   error_maps: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate confidence scores for corrected segmentation."""
        # Base confidence from uncertainty
        confidence = 1.0 - uncertainty_map
        
        # Reduce confidence in error regions
        for error_map in error_maps.values():
            confidence[error_map] *= 0.5
        
        # Boost confidence in well-segmented regions
        well_segmented = (corrected_segmentation > 0.8) & (uncertainty_map < 0.2)
        confidence[well_segmented] = np.minimum(confidence[well_segmented] * 1.2, 1.0)
        
        return confidence
    
    def _calculate_quality_metrics(self,
                                 original_segmentation: np.ndarray,
                                 corrected_segmentation: np.ndarray,
                                 error_maps: Dict[str, np.ndarray],
                                 correction_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for the proofreading process."""
        # Volume changes
        original_volume = np.sum(original_segmentation > 0.5)
        corrected_volume = np.sum(corrected_segmentation > 0.5)
        volume_change = (corrected_volume - original_volume) / max(original_volume, 1)
        
        # Error statistics
        total_errors = sum(np.sum(error_map) for error_map in error_maps.values())
        error_density = total_errors / original_segmentation.size
        
        # Connectivity
        from scipy import ndimage
        original_components = ndimage.label(original_segmentation > 0.5)[1]
        corrected_components = ndimage.label(corrected_segmentation > 0.5)[1]
        
        metrics = {
            'volume_change_ratio': volume_change,
            'error_density': error_density,
            'component_change': corrected_components - original_components,
            'processing_time': self.stats['total_processing_time'],
            'total_errors_detected': total_errors
        }
        
        # Add correction-specific metrics
        for strategy, metadata in correction_metadata.items():
            if isinstance(metadata, dict) and 'error' not in metadata:
                for key, value in metadata.items():
                    metrics[f'{strategy}_{key}'] = value
        
        return metrics
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return {
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**3,
                'cpu_memory': 0.0
            }
        else:
            return {
                'gpu_memory_allocated': 0.0,
                'gpu_memory_cached': 0.0,
                'cpu_memory': 0.0
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get proofreading statistics."""
        return {
            **self.stats,
            'config': self.config
        }
    
    def save_results(self, result: ProofreadingResult, output_path: str):
        """Save proofreading results."""
        if ZARR_AVAILABLE:
            store = zarr.open(output_path, mode='w')
            store.create_dataset('corrected_segmentation', 
                               data=result.corrected_segmentation,
                               chunks=(64, 64, 64), 
                               compressor=numcodecs.Blosc())
            store.create_dataset('error_map', 
                               data=result.error_map,
                               chunks=(64, 64, 64), 
                               compressor=numcodecs.Blosc())
            store.create_dataset('confidence_scores', 
                               data=result.confidence_scores,
                               chunks=(64, 64, 64), 
                               compressor=numcodecs.Blosc())
            
            # Save metadata
            metadata_group = store.create_group('metadata')
            for key, value in result.correction_metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    metadata_group.attrs[key] = value
        else:
            # Save as numpy arrays
            np.save(f"{output_path}_corrected.npy", result.corrected_segmentation)
            np.save(f"{output_path}_errors.npy", result.error_map)
            np.save(f"{output_path}_confidence.npy", result.confidence_scores)
            
            # Convert metadata to JSON-serializable format
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            serializable_metadata = convert_numpy_types(result.correction_metadata)
            
            # Save metadata as JSON
            with open(f"{output_path}_metadata.json", 'w') as f:
                json.dump(serializable_metadata, f, indent=2) 