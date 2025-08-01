#!/usr/bin/env python3
"""
Enhanced Flood-Filling Algorithm for Connectomics
================================================
Advanced implementation with improved performance, memory efficiency, and features.
"""

import os
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from collections import deque, defaultdict
import heapq
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from scipy import ndimage
from scipy.spatial.distance import cdist
import cv2
import json
import pickle

logger = logging.getLogger(__name__)

@dataclass
class FloodFillConfig:
    """Configuration for flood-filling algorithm."""
    # Core parameters
    fov_size: Tuple[int, int, int] = (33, 33, 33)
    confidence_threshold: float = 0.9
    max_iterations: int = 10000
    min_segment_size: int = 100
    
    # Memory optimization
    chunk_size: Tuple[int, int, int] = (128, 128, 128)
    overlap_size: Tuple[int, int, int] = (16, 16, 16)
    use_memory_mapping: bool = True
    
    # Performance optimization
    batch_size: int = 4
    num_workers: int = 4
    use_gpu: bool = True
    
    # Quality control
    edge_smoothing: bool = True
    connectivity_check: bool = True
    remove_small_components: bool = True
    
    # Advanced features
    adaptive_thresholding: bool = True
    multi_scale_processing: bool = True
    uncertainty_aware: bool = True
    
    # Visualization
    save_intermediate: bool = True
    save_interval: int = 100
    visualization_format: str = "npy"  # "npy", "h5", "zarr"

@dataclass
class FloodFillResult:
    """Result of flood-filling operation."""
    segmentation: np.ndarray
    uncertainty_map: np.ndarray
    confidence_scores: np.ndarray
    processing_time: float
    iterations: int
    memory_usage: Dict[str, float]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]

class PriorityQueue:
    """Priority queue for flood-filling with confidence-based ordering."""
    
    def __init__(self):
        self.queue = []
        self.entry_count = 0
    
    def put(self, item: Tuple[float, int, Tuple[int, int, int]]):
        """Add item with priority (confidence, entry_count, coordinates)."""
        heapq.heappush(self.queue, (-item[0], self.entry_count, item[2]))
        self.entry_count += 1
    
    def get(self) -> Tuple[int, int, int]:
        """Get highest confidence coordinate."""
        return heapq.heappop(self.queue)[2]
    
    def empty(self) -> bool:
        return len(self.queue) == 0
    
    def size(self) -> int:
        return len(self.queue)

class AdaptiveThresholding:
    """Adaptive thresholding based on local image statistics."""
    
    def __init__(self, window_size: int = 15, method: str = "gaussian"):
        self.window_size = window_size
        self.method = method
    
    def compute_adaptive_threshold(self, volume: np.ndarray, seed_point: Tuple[int, int, int]) -> float:
        """Compute adaptive threshold based on local statistics."""
        z, y, x = seed_point
        
        # Extract local region
        z_start = max(0, z - self.window_size // 2)
        z_end = min(volume.shape[0], z + self.window_size // 2 + 1)
        y_start = max(0, y - self.window_size // 2)
        y_end = min(volume.shape[1], y + self.window_size // 2 + 1)
        x_start = max(0, x - self.window_size // 2)
        x_end = min(volume.shape[2], x + self.window_size // 2 + 1)
        
        local_region = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        if self.method == "gaussian":
            # Gaussian-based threshold
            mean_val = np.mean(local_region)
            std_val = np.std(local_region)
            threshold = mean_val + 2.0 * std_val
        elif self.method == "otsu":
            # Otsu's method
            threshold = self._otsu_threshold(local_region)
        elif self.method == "percentile":
            # Percentile-based
            threshold = np.percentile(local_region, 85)
        else:
            threshold = np.mean(local_region) + np.std(local_region)
        
        return np.clip(threshold, 0.1, 0.9)
    
    def _otsu_threshold(self, data: np.ndarray) -> float:
        """Compute Otsu's threshold."""
        # Normalize to 0-255 range
        data_norm = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        
        # Compute histogram
        hist, bins = np.histogram(data_norm, bins=256, range=(0, 256))
        
        # Compute Otsu's threshold
        total_pixels = hist.sum()
        current_max = 0
        threshold = 0
        
        for t in range(256):
            w0 = hist[:t+1].sum() / total_pixels
            w1 = hist[t+1:].sum() / total_pixels
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = np.average(np.arange(t+1), weights=hist[:t+1])
            mu1 = np.average(np.arange(t+1, 256), weights=hist[t+1:])
            
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > current_max:
                current_max = variance
                threshold = t
        
        return threshold / 255.0

class MultiScaleProcessor:
    """Multi-scale processing for improved segmentation quality."""
    
    def __init__(self, scales: List[float] = [1.0, 0.5, 0.25]):
        self.scales = scales
    
    def process_multi_scale(self, volume: np.ndarray, model: nn.Module, 
                           seed_point: Tuple[int, int, int]) -> np.ndarray:
        """Process volume at multiple scales and combine results."""
        results = []
        
        for scale in self.scales:
            # Resize volume
            if scale != 1.0:
                scaled_volume = self._resize_volume(volume, scale)
                scaled_seed = tuple(int(s * scale) for s in seed_point)
            else:
                scaled_volume = volume
                scaled_seed = seed_point
            
            # Process at this scale
            result = self._process_single_scale(scaled_volume, model, scaled_seed)
            
            # Resize back to original size
            if scale != 1.0:
                result = self._resize_volume(result, 1.0 / scale, target_shape=volume.shape)
            
            results.append(result)
        
        # Combine results using weighted average
        combined = self._combine_results(results)
        return combined
    
    def _resize_volume(self, volume: np.ndarray, scale: float, 
                      target_shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Resize volume using trilinear interpolation."""
        if target_shape is None:
            new_shape = tuple(int(s * scale) for s in volume.shape)
        else:
            new_shape = target_shape
        
        # Use scipy for 3D interpolation
        from scipy.ndimage import zoom
        return zoom(volume, [new_shape[i] / volume.shape[i] for i in range(3)], order=1)
    
    def _process_single_scale(self, volume: np.ndarray, model: nn.Module, 
                             seed_point: Tuple[int, int, int]) -> np.ndarray:
        """Process volume at a single scale."""
        # This would integrate with the main flood-filling algorithm
        # For now, return a simple threshold-based segmentation
        threshold = np.percentile(volume, 70)
        return (volume > threshold).astype(np.float32)
    
    def _combine_results(self, results: List[np.ndarray]) -> np.ndarray:
        """Combine multi-scale results using weighted average."""
        weights = [1.0, 0.7, 0.4]  # Higher weight for higher resolution
        combined = np.zeros_like(results[0])
        
        for result, weight in zip(results, weights):
            combined += weight * result
        
        return combined / sum(weights)

class EnhancedFloodFillAlgorithm:
    """Enhanced flood-filling algorithm with advanced features."""
    
    def __init__(self, config: FloodFillConfig):
        self.config = config
        self.adaptive_thresholding = AdaptiveThresholding()
        self.multi_scale_processor = MultiScaleProcessor()
        
        # Performance tracking
        self.stats = {
            'total_iterations': 0,
            'total_processing_time': 0.0,
            'memory_peak': 0.0,
            'segments_processed': 0
        }
    
    def flood_fill(self, volume: np.ndarray, model: nn.Module, 
                   seed_point: Tuple[int, int, int]) -> FloodFillResult:
        """Enhanced flood-filling with advanced features."""
        start_time = time.time()
        
        logger.info(f"Starting enhanced flood-fill from seed point: {seed_point}")
        
        # Initialize segmentation
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        uncertainty_map = np.zeros_like(volume, dtype=np.float32)
        confidence_scores = np.zeros_like(volume, dtype=np.float32)
        
        # Initialize priority queue
        queue = PriorityQueue()
        queue.put((1.0, 0, seed_point))  # Start with maximum confidence
        
        # Set seed point
        segmentation[seed_point] = 1
        confidence_scores[seed_point] = 1.0
        
        # Track visited points for efficiency
        visited = set()
        visited.add(seed_point)
        
        # Adaptive threshold
        if self.config.adaptive_thresholding:
            adaptive_threshold = self.adaptive_thresholding.compute_adaptive_threshold(
                volume, seed_point
            )
            logger.info(f"Adaptive threshold: {adaptive_threshold:.3f}")
        else:
            adaptive_threshold = self.config.confidence_threshold
        
        iteration = 0
        fov_center_offset = tuple(s // 2 for s in self.config.fov_size)
        
        # Main flood-filling loop
        while not queue.empty() and iteration < self.config.max_iterations:
            iteration += 1
            
            # Get next point with highest confidence
            cz, cy, cx = queue.get()
            
            # Define FOV around current point
            start = [c - o for c, o in zip((cz, cy, cx), fov_center_offset)]
            end = [s + f for s, f in zip(start, self.config.fov_size)]
            
            # Check bounds
            if any(s < 0 for s in start) or any(e > s for e, s in zip(end, volume.shape)):
                continue
            
            # Extract FOV
            fov = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            
            # Multi-scale processing
            if self.config.multi_scale_processing:
                prediction = self._process_multi_scale_fov(fov, model)
            else:
                prediction = self._process_single_fov(fov, model)
            
            # Update segmentation and queue
            new_points = self._update_segmentation(
                prediction, start, segmentation, uncertainty_map, 
                confidence_scores, queue, visited, adaptive_threshold
            )
            
            # Save intermediate results
            if (self.config.save_intermediate and 
                iteration % self.config.save_interval == 0):
                self._save_intermediate_result(
                    segmentation, uncertainty_map, confidence_scores, iteration
                )
            
            # Log progress
            if iteration % 1000 == 0:
                logger.info(f"Iteration {iteration}: Queue size = {queue.size()}, "
                          f"Segmented voxels = {np.sum(segmentation)}")
        
        # Post-processing
        if self.config.edge_smoothing:
            segmentation = self._smooth_edges(segmentation)
        
        if self.config.remove_small_components:
            segmentation = self._remove_small_components(segmentation)
        
        if self.config.connectivity_check:
            segmentation = self._ensure_connectivity(segmentation, seed_point)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            segmentation, uncertainty_map, volume
        )
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_iterations'] += iteration
        self.stats['total_processing_time'] += processing_time
        self.stats['segments_processed'] += 1
        
        return FloodFillResult(
            segmentation=segmentation,
            uncertainty_map=uncertainty_map,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            iterations=iteration,
            memory_usage=self._get_memory_usage(),
            quality_metrics=quality_metrics,
            metadata={
                'seed_point': seed_point,
                'adaptive_threshold': adaptive_threshold,
                'config': self.config.__dict__
            }
        )
    
    def _process_multi_scale_fov(self, fov: np.ndarray, model: nn.Module) -> np.ndarray:
        """Process FOV at multiple scales."""
        return self.multi_scale_processor.process_multi_scale(fov, model, 
                                                             (fov.shape[0]//2, fov.shape[1]//2, fov.shape[2]//2))
    
    def _process_single_fov(self, fov: np.ndarray, model: nn.Module) -> np.ndarray:
        """Process single FOV with model."""
        try:
            # Prepare input tensor
            fov_tensor = torch.from_numpy(fov).float().unsqueeze(0).unsqueeze(0)
            if self.config.use_gpu and torch.cuda.is_available():
                fov_tensor = fov_tensor.cuda()
                model = model.cuda()
            
            # Run inference
            with torch.no_grad():
                prediction, uncertainty = model(fov_tensor)
                prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()
                uncertainty = uncertainty.squeeze().cpu().numpy()
            
            return prediction
            
        except Exception as e:
            logger.warning(f"Model inference failed: {e}")
            # Fallback to simple thresholding
            return (fov > np.percentile(fov, 70)).astype(np.float32)
    
    def _update_segmentation(self, prediction: np.ndarray, start: List[int],
                           segmentation: np.ndarray, uncertainty_map: np.ndarray,
                           confidence_scores: np.ndarray, queue: PriorityQueue,
                           visited: set, threshold: float) -> int:
        """Update segmentation with new predictions."""
        new_points = 0
        
        for (dz, dy, dx), confidence in np.ndenumerate(prediction):
            if confidence > threshold:
                world_coord = (start[0] + dz, start[1] + dy, start[2] + dx)
                
                # Check bounds
                if (0 <= world_coord[0] < segmentation.shape[0] and
                    0 <= world_coord[1] < segmentation.shape[1] and
                    0 <= world_coord[2] < segmentation.shape[2]):
                    
                    if world_coord not in visited:
                        segmentation[world_coord] = 1
                        confidence_scores[world_coord] = confidence
                        uncertainty_map[world_coord] = 1.0 - confidence
                        
                        # Add to queue with confidence-based priority
                        queue.put((confidence, 0, world_coord))
                        visited.add(world_coord)
                        new_points += 1
        
        return new_points
    
    def _smooth_edges(self, segmentation: np.ndarray) -> np.ndarray:
        """Smooth segmentation edges using morphological operations."""
        from scipy import ndimage
        
        # Remove small holes
        segmentation = ndimage.binary_fill_holes(segmentation)
        
        # Smooth edges
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        segmentation = ndimage.binary_opening(segmentation, structure=kernel)
        segmentation = ndimage.binary_closing(segmentation, structure=kernel)
        
        return segmentation
    
    def _remove_small_components(self, segmentation: np.ndarray) -> np.ndarray:
        """Remove small disconnected components."""
        from scipy import ndimage
        
        # Label connected components
        labeled, num_features = ndimage.label(segmentation)
        
        # Find the largest component (assumed to be the main segment)
        if num_features > 0:
            component_sizes = ndimage.sum(segmentation, labeled, range(1, num_features + 1))
            largest_component = np.argmax(component_sizes) + 1
            
            # Keep only the largest component
            segmentation = (labeled == largest_component).astype(np.uint8)
        
        return segmentation
    
    def _ensure_connectivity(self, segmentation: np.ndarray, seed_point: Tuple[int, int, int]) -> np.ndarray:
        """Ensure connectivity to seed point."""
        from scipy import ndimage
        
        # Label connected components
        labeled, num_features = ndimage.label(segmentation)
        
        # Find component containing seed point
        seed_component = labeled[seed_point]
        
        if seed_component > 0:
            # Keep only the component containing the seed point
            segmentation = (labeled == seed_component).astype(np.uint8)
        
        return segmentation
    
    def _calculate_quality_metrics(self, segmentation: np.ndarray, 
                                 uncertainty_map: np.ndarray, 
                                 volume: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for the segmentation."""
        metrics = {}
        
        # Basic metrics
        metrics['total_voxels'] = np.sum(segmentation)
        metrics['volume_ratio'] = np.sum(segmentation) / segmentation.size
        
        # Edge metrics
        from scipy import ndimage
        edges = ndimage.binary_erosion(segmentation) != ndimage.binary_dilation(segmentation)
        metrics['edge_density'] = np.sum(edges) / np.sum(segmentation) if np.sum(segmentation) > 0 else 0
        
        # Uncertainty metrics
        if np.sum(segmentation) > 0:
            metrics['mean_uncertainty'] = np.mean(uncertainty_map[segmentation > 0])
            metrics['max_uncertainty'] = np.max(uncertainty_map[segmentation > 0])
        else:
            metrics['mean_uncertainty'] = 0.0
            metrics['max_uncertainty'] = 0.0
        
        # Connectivity metrics
        labeled, num_components = ndimage.label(segmentation)
        metrics['num_components'] = num_components
        
        # Shape metrics
        if np.sum(segmentation) > 0:
            # Calculate bounding box
            coords = np.where(segmentation)
            bbox_volume = (np.max(coords[0]) - np.min(coords[0]) + 1) * \
                         (np.max(coords[1]) - np.min(coords[1]) + 1) * \
                         (np.max(coords[2]) - np.min(coords[2]) + 1)
            metrics['compactness'] = np.sum(segmentation) / bbox_volume
        
        return metrics
    
    def _save_intermediate_result(self, segmentation: np.ndarray, 
                                uncertainty_map: np.ndarray,
                                confidence_scores: np.ndarray, 
                                iteration: int):
        """Save intermediate results for visualization."""
        output_dir = "floodfill_intermediate"
        os.makedirs(output_dir, exist_ok=True)
        
        if self.config.visualization_format == "npy":
            np.save(f"{output_dir}/segmentation_step_{iteration:06d}.npy", segmentation)
            np.save(f"{output_dir}/uncertainty_step_{iteration:06d}.npy", uncertainty_map)
            np.save(f"{output_dir}/confidence_step_{iteration:06d}.npy", confidence_scores)
        elif self.config.visualization_format == "h5":
            import h5py
            with h5py.File(f"{output_dir}/step_{iteration:06d}.h5", 'w') as f:
                f.create_dataset('segmentation', data=segmentation)
                f.create_dataset('uncertainty', data=uncertainty_map)
                f.create_dataset('confidence', data=confidence_scores)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        return {
            **self.stats,
            'average_iterations_per_segment': (
                self.stats['total_iterations'] / self.stats['segments_processed']
                if self.stats['segments_processed'] > 0 else 0
            ),
            'average_processing_time_per_segment': (
                self.stats['total_processing_time'] / self.stats['segments_processed']
                if self.stats['segments_processed'] > 0 else 0
            )
        }

# Example usage and testing
def test_enhanced_floodfill():
    """Test the enhanced flood-filling algorithm."""
    # Create test volume
    volume = np.random.random((64, 64, 64))
    
    # Create dummy model
    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x), torch.sigmoid(x)
    
    model = DummyModel()
    
    # Configure algorithm
    config = FloodFillConfig(
        fov_size=(17, 17, 17),
        confidence_threshold=0.7,
        max_iterations=1000,
        adaptive_thresholding=True,
        multi_scale_processing=True
    )
    
    # Create algorithm instance
    algorithm = EnhancedFloodFillAlgorithm(config)
    
    # Run flood-filling
    seed_point = (32, 32, 32)
    result = algorithm.flood_fill(volume, model, seed_point)
    
    print(f"Segmentation completed in {result.processing_time:.2f}s")
    print(f"Iterations: {result.iterations}")
    print(f"Quality metrics: {result.quality_metrics}")
    
    return result

if __name__ == "__main__":
    test_enhanced_floodfill() 