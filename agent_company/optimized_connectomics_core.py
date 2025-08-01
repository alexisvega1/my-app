#!/usr/bin/env python3
"""
Optimized Connectomics Core System
==================================
High-performance implementations using JAX, C++ extensions, and CUDA
for exabyte-scale connectomics processing.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import queue
import ctypes
from ctypes import cdll, c_int, c_float, c_double, POINTER, c_void_p
import mmap
import psutil
import gc

# JAX imports for GPU acceleration
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap, grad, value_and_grad
    from jax.lax import scan, while_loop, cond
    from jax.random import PRNGKey, split, normal
    from jax.experimental import maps, PartitionSpec as P
    from jax.experimental.maps import Mesh
    JAX_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("JAX available for GPU acceleration")
except ImportError:
    JAX_AVAILABLE = False
    logger.warning("JAX not available, falling back to NumPy")

# PyTorch for comparison
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.cpp_extension import load_inline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for optimized processing."""
    # Hardware configuration
    use_jax: bool = True
    use_cuda: bool = True
    use_cpp_extensions: bool = True
    use_memory_mapping: bool = True
    
    # Performance tuning
    chunk_size: Tuple[int, int, int] = (1024, 1024, 1024)
    batch_size: int = 32
    num_workers: int = mp.cpu_count()
    gpu_memory_fraction: float = 0.9
    
    # Memory optimization
    max_memory_gb: int = 1024
    cache_size_gb: int = 100
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    # I/O optimization
    use_async_io: bool = True
    io_buffer_size: int = 64 * 1024 * 1024  # 64MB
    compression_level: int = 6
    
    # Parallel processing
    use_distributed: bool = True
    num_nodes: int = 100
    gpus_per_node: int = 8

class JAXOptimizedFloodFill:
    """JAX-optimized flood-filling algorithm for maximum GPU efficiency."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for optimized flood-filling")
        
        # Configure JAX for maximum performance
        jax.config.update('jax_platform_name', 'gpu')
        jax.config.update('jax_enable_x64', False)  # Use float32 for speed
        
        # Get available devices
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        
        logger.info(f"JAX optimized flood-fill initialized with {self.num_devices} devices")
    
    @staticmethod
    @jit
    def _compute_priority_queue(volume: jnp.ndarray, 
                               visited: jnp.ndarray,
                               queue: jnp.ndarray,
                               queue_size: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JIT-compiled priority queue computation."""
        
        def update_queue(carry, x):
            queue, queue_size = carry
            pos, priority = x
            
            # Insert into priority queue (maintain heap property)
            new_size = queue_size + 1
            new_queue = queue.at[new_size - 1].set(jnp.array([pos[0], pos[1], pos[2], priority]))
            
            # Bubble up
            i = new_size - 1
            while i > 0:
                parent = (i - 1) // 2
                if new_queue[parent, 3] < new_queue[i, 3]:
                    new_queue = jax.lax.swap(new_queue, i, parent)
                    i = parent
                else:
                    break
            
            return (new_queue, new_size), None
        
        # Process all queue updates in parallel
        (updated_queue, updated_size), _ = scan(update_queue, (queue, queue_size), 
                                               (jnp.arange(volume.shape[0]), jnp.arange(volume.shape[0])))
        
        return updated_queue, updated_size
    
    @staticmethod
    @jit
    def _flood_fill_step(volume: jnp.ndarray,
                         segmentation: jnp.ndarray,
                         visited: jnp.ndarray,
                         queue: jnp.ndarray,
                         queue_size: jnp.ndarray,
                         threshold: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Single step of JAX-optimized flood-filling."""
        
        # Get highest priority position
        current_pos = queue[0, :3].astype(jnp.int32)
        current_priority = queue[0, 3]
        
        # Remove from queue (replace with last element and heapify down)
        last_element = queue[queue_size - 1]
        new_queue = queue.at[0].set(last_element)
        new_queue_size = queue_size - 1
        
        # Heapify down
        i = 0
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i
            
            if left < new_queue_size and new_queue[left, 3] > new_queue[largest, 3]:
                largest = left
            if right < new_queue_size and new_queue[right, 3] > new_queue[largest, 3]:
                largest = right
            
            if largest != i:
                new_queue = jax.lax.swap(new_queue, i, largest)
                i = largest
            else:
                break
        
        # Mark as visited and segmented
        new_visited = visited.at[current_pos[0], current_pos[1], current_pos[2]].set(True)
        new_segmentation = segmentation.at[current_pos[0], current_pos[1], current_pos[2]].set(1)
        
        # Check neighbors
        neighbors = jnp.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
        
        def check_neighbor(carry, neighbor):
            queue, queue_size = carry
            neighbor_pos = current_pos + neighbor
            
            # Check bounds
            in_bounds = jnp.all(neighbor_pos >= 0) & jnp.all(neighbor_pos < volume.shape)
            
            # Check if already visited
            not_visited = ~new_visited[neighbor_pos[0], neighbor_pos[1], neighbor_pos[2]]
            
            # Check threshold
            above_threshold = volume[neighbor_pos[0], neighbor_pos[1], neighbor_pos[2]] > threshold
            
            should_add = in_bounds & not_visited & above_threshold
            
            # Add to queue if conditions met
            priority = volume[neighbor_pos[0], neighbor_pos[1], neighbor_pos[2]]
            new_queue, new_size = JAXOptimizedFloodFill._compute_priority_queue(
                volume, new_visited, queue, queue_size
            )
            
            return (new_queue, new_size), None
        
        (final_queue, final_queue_size), _ = scan(check_neighbor, (new_queue, new_queue_size), neighbors)
        
        return new_segmentation, new_visited, final_queue, final_queue_size, current_priority
    
    @staticmethod
    @jit
    def flood_fill(volume: jnp.ndarray, 
                   seed_point: Tuple[int, int, int],
                   threshold: float,
                   max_steps: int = 10000) -> jnp.ndarray:
        """JAX-optimized flood-filling algorithm."""
        
        # Initialize arrays
        segmentation = jnp.zeros_like(volume, dtype=jnp.bool_)
        visited = jnp.zeros_like(volume, dtype=jnp.bool_)
        
        # Initialize priority queue
        queue_size = 1
        queue = jnp.zeros((max_steps, 4), dtype=jnp.float32)
        queue = queue.at[0].set(jnp.array([seed_point[0], seed_point[1], seed_point[2], 
                                          volume[seed_point[0], seed_point[1], seed_point[2]]]))
        
        # Main flood-fill loop
        def flood_fill_loop(carry, step):
            segmentation, visited, queue, queue_size = carry
            
            # Check if queue is empty
            continue_loop = queue_size > 0
            
            # Perform flood-fill step
            new_segmentation, new_visited, new_queue, new_queue_size, _ = \
                JAXOptimizedFloodFill._flood_fill_step(volume, segmentation, visited, queue, queue_size, threshold)
            
            return (new_segmentation, new_visited, new_queue, new_queue_size), continue_loop
        
        # Run flood-fill with while_loop for efficiency
        (final_segmentation, _, _, _), _ = scan(flood_fill_loop, 
                                               (segmentation, visited, queue, queue_size),
                                               jnp.arange(max_steps))
        
        return final_segmentation

class CppOptimizedVolumeProcessor:
    """C++ optimized volume processing for maximum CPU efficiency."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cpp_lib = self._compile_cpp_extensions()
        
        logger.info("C++ optimized volume processor initialized")
    
    def _compile_cpp_extensions(self):
        """Compile C++ extensions for maximum performance."""
        
        # C++ source code for optimized volume processing
        cpp_source = """
        #include <torch/extension.h>
        #include <vector>
        #include <algorithm>
        #include <omp.h>
        
        torch::Tensor optimized_volume_processing(
            torch::Tensor volume,
            torch::Tensor segmentation,
            float threshold,
            int max_iterations
        ) {
            // Enable OpenMP for parallel processing
            omp_set_num_threads(omp_get_max_threads());
            
            auto volume_accessor = volume.accessor<float, 3>();
            auto segmentation_accessor = segmentation.accessor<bool, 3>();
            
            int depth = volume.size(0);
            int height = volume.size(1);
            int width = volume.size(2);
            
            // Process volume in parallel chunks
            #pragma omp parallel for collapse(3)
            for (int z = 0; z < depth; z++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        float value = volume_accessor[z][y][x];
                        
                        if (value > threshold) {
                            segmentation_accessor[z][y][x] = true;
                        }
                    }
                }
            }
            
            return segmentation;
        }
        
        torch::Tensor optimized_flood_fill(
            torch::Tensor volume,
            int seed_z, int seed_y, int seed_x,
            float threshold,
            int max_steps
        ) {
            auto volume_accessor = volume.accessor<float, 3>();
            auto segmentation = torch::zeros_like(volume, torch::kBool);
            auto segmentation_accessor = segmentation.accessor<bool, 3>();
            
            int depth = volume.size(0);
            int height = volume.size(1);
            int width = volume.size(2);
            
            // Use std::priority_queue for efficient flood-fill
            std::priority_queue<std::pair<float, std::tuple<int, int, int>>> queue;
            std::vector<std::vector<std::vector<bool>>> visited(
                depth, std::vector<std::vector<bool>>(
                    height, std::vector<bool>(width, false)
                )
            );
            
            // Add seed point
            queue.push({volume_accessor[seed_z][seed_y][seed_x], {seed_z, seed_y, seed_x}});
            visited[seed_z][seed_y][seed_x] = true;
            
            int steps = 0;
            while (!queue.empty() && steps < max_steps) {
                auto [priority, pos] = queue.top();
                auto [z, y, x] = pos;
                queue.pop();
                
                segmentation_accessor[z][y][x] = true;
                
                // Check 6-connected neighbors
                int dz[] = {-1, 1, 0, 0, 0, 0};
                int dy[] = {0, 0, -1, 1, 0, 0};
                int dx[] = {0, 0, 0, 0, -1, 1};
                
                for (int i = 0; i < 6; i++) {
                    int nz = z + dz[i];
                    int ny = y + dy[i];
                    int nx = x + dx[i];
                    
                    if (nz >= 0 && nz < depth && ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        if (!visited[nz][ny][nx]) {
                            float neighbor_value = volume_accessor[nz][ny][nx];
                            if (neighbor_value > threshold) {
                                queue.push({neighbor_value, {nz, ny, nx}});
                                visited[nz][ny][nx] = true;
                            }
                        }
                    }
                }
                
                steps++;
            }
            
            return segmentation;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("optimized_volume_processing", &optimized_volume_processing, "Optimized volume processing");
            m.def("optimized_flood_fill", &optimized_flood_fill, "Optimized flood-fill algorithm");
        }
        """
        
        if TORCH_AVAILABLE:
            try:
                # Compile C++ extension
                cpp_extension = load_inline(
                    name="optimized_connectomics",
                    cpp_sources=[cpp_source],
                    extra_cflags=["-O3", "-march=native", "-fopenmp"],
                    extra_ldflags=["-fopenmp"],
                    verbose=True
                )
                return cpp_extension
            except Exception as e:
                logger.warning(f"Failed to compile C++ extensions: {e}")
                return None
        else:
            logger.warning("PyTorch not available for C++ extensions")
            return None
    
    def process_volume_cpp(self, volume: np.ndarray, threshold: float) -> np.ndarray:
        """Process volume using C++ optimized code."""
        if self.cpp_lib is None:
            return self._fallback_processing(volume, threshold)
        
        # Convert to PyTorch tensors
        volume_tensor = torch.from_numpy(volume).float()
        segmentation_tensor = torch.zeros_like(volume_tensor, dtype=torch.bool)
        
        # Process with C++ optimization
        result = self.cpp_lib.optimized_volume_processing(
            volume_tensor, segmentation_tensor, threshold, 1000
        )
        
        return result.numpy()
    
    def flood_fill_cpp(self, volume: np.ndarray, seed_point: Tuple[int, int, int], 
                       threshold: float, max_steps: int = 10000) -> np.ndarray:
        """Flood-fill using C++ optimized code."""
        if self.cpp_lib is None:
            return self._fallback_flood_fill(volume, seed_point, threshold, max_steps)
        
        # Convert to PyTorch tensor
        volume_tensor = torch.from_numpy(volume).float()
        
        # Process with C++ optimization
        result = self.cpp_lib.optimized_flood_fill(
            volume_tensor, seed_point[0], seed_point[1], seed_point[2], 
            threshold, max_steps
        )
        
        return result.numpy()
    
    def _fallback_processing(self, volume: np.ndarray, threshold: float) -> np.ndarray:
        """Fallback processing using NumPy."""
        return volume > threshold
    
    def _fallback_flood_fill(self, volume: np.ndarray, seed_point: Tuple[int, int, int],
                             threshold: float, max_steps: int) -> np.ndarray:
        """Fallback flood-fill using NumPy."""
        from queue import PriorityQueue
        
        segmentation = np.zeros_like(volume, dtype=bool)
        visited = np.zeros_like(volume, dtype=bool)
        
        queue = PriorityQueue()
        queue.put((-volume[seed_point], seed_point))
        visited[seed_point] = True
        
        steps = 0
        while not queue.empty() and steps < max_steps:
            priority, (z, y, x) = queue.get()
            segmentation[z, y, x] = True
            
            # Check neighbors
            for dz, dy, dx in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                nz, ny, nx = z + dz, y + dy, x + dx
                
                if (0 <= nz < volume.shape[0] and 0 <= ny < volume.shape[1] and 
                    0 <= nx < volume.shape[2] and not visited[nz, ny, nx]):
                    if volume[nz, ny, nx] > threshold:
                        queue.put((-volume[nz, ny, nx], (nz, ny, nx)))
                        visited[nz, ny, nx] = True
            
            steps += 1
        
        return segmentation

class MemoryOptimizedDataManager:
    """Memory-optimized data management for exabyte-scale processing."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_pool = {}
        self.cache = {}
        self.cache_order = []
        self.max_cache_size = config.cache_size_gb * 1024**3  # Convert to bytes
        self.current_cache_size = 0
        
        logger.info(f"Memory optimized data manager initialized with {config.cache_size_gb}GB cache")
    
    def load_chunk_memory_mapped(self, file_path: str, chunk_coords: Tuple[int, int, int],
                                 chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk using memory mapping for efficient I/O."""
        
        cache_key = f"{file_path}_{chunk_coords}_{chunk_size}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load using memory mapping
        with open(file_path, 'rb') as f:
            # Calculate offset in file
            offset = self._calculate_offset(chunk_coords, chunk_size)
            
            # Memory map the file
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Seek to offset
                mm.seek(offset)
                
                # Read chunk data
                chunk_data = np.frombuffer(mm.read(np.prod(chunk_size) * 4), dtype=np.float32)
                chunk_data = chunk_data.reshape(chunk_size)
        
        # Cache the result
        self._add_to_cache(cache_key, chunk_data)
        
        return chunk_data
    
    def _calculate_offset(self, chunk_coords: Tuple[int, int, int], 
                         chunk_size: Tuple[int, int, int]) -> int:
        """Calculate file offset for chunk coordinates."""
        # This would depend on the specific file format
        # For now, assume simple 3D array layout
        z, y, x = chunk_coords
        dz, dy, dx = chunk_size
        
        # Assuming row-major order
        offset = (z * dy * dx + y * dx + x) * 4  # 4 bytes per float32
        return offset
    
    def _add_to_cache(self, key: str, data: np.ndarray):
        """Add data to cache with LRU eviction."""
        data_size = data.nbytes
        
        # Evict if necessary
        while self.current_cache_size + data_size > self.max_cache_size and self.cache_order:
            oldest_key = self.cache_order.pop(0)
            if oldest_key in self.cache:
                evicted_data = self.cache.pop(oldest_key)
                self.current_cache_size -= evicted_data.nbytes
        
        # Add to cache
        self.cache[key] = data
        self.cache_order.append(key)
        self.current_cache_size += data_size
    
    def save_chunk_compressed(self, file_path: str, chunk_coords: Tuple[int, int, int],
                             chunk_data: np.ndarray):
        """Save chunk with compression for storage efficiency."""
        
        import zlib
        
        # Compress data
        compressed_data = zlib.compress(chunk_data.tobytes(), level=self.config.compression_level)
        
        # Save with metadata
        metadata = {
            'chunk_coords': chunk_coords,
            'chunk_shape': chunk_data.shape,
            'dtype': str(chunk_data.dtype),
            'compressed_size': len(compressed_data),
            'original_size': chunk_data.nbytes
        }
        
        # Save to file (this would be more sophisticated in production)
        with open(file_path, 'wb') as f:
            # Write metadata
            metadata_bytes = str(metadata).encode()
            f.write(len(metadata_bytes).to_bytes(8, 'big'))
            f.write(metadata_bytes)
            
            # Write compressed data
            f.write(compressed_data)

class OptimizedConnectomicsPipeline:
    """Optimized connectomics pipeline combining all optimizations."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # Initialize optimized components
        if config.use_jax and JAX_AVAILABLE:
            self.jax_flood_fill = JAXOptimizedFloodFill(config)
        
        if config.use_cpp_extensions:
            self.cpp_processor = CppOptimizedVolumeProcessor(config)
        
        self.memory_manager = MemoryOptimizedDataManager(config)
        
        # Performance monitoring
        self.performance_stats = {
            'processing_times': [],
            'memory_usage': [],
            'throughput': []
        }
        
        logger.info("Optimized connectomics pipeline initialized")
    
    def process_volume_optimized(self, volume_path: str, output_path: str,
                                 seed_points: List[Tuple[int, int, int]],
                                 threshold: float = 0.5) -> Dict[str, Any]:
        """Process volume with all optimizations enabled."""
        
        start_time = time.time()
        
        # Load volume in chunks
        chunk_size = self.config.chunk_size
        results = []
        
        for seed_point in seed_points:
            # Determine chunk coordinates
            chunk_coords = tuple(s // c for s, c in zip(seed_point, chunk_size))
            
            # Load chunk using memory mapping
            chunk_data = self.memory_manager.load_chunk_memory_mapped(
                volume_path, chunk_coords, chunk_size
            )
            
            # Process with optimal method
            if self.config.use_jax and JAX_AVAILABLE:
                # Use JAX for GPU acceleration
                chunk_data_jax = jnp.array(chunk_data)
                segmentation = self.jax_flood_fill.flood_fill(
                    chunk_data_jax, seed_point, threshold
                )
                segmentation = np.array(segmentation)
            elif self.config.use_cpp_extensions:
                # Use C++ for CPU optimization
                segmentation = self.cpp_processor.flood_fill_cpp(
                    chunk_data, seed_point, threshold
                )
            else:
                # Fallback to NumPy
                segmentation = self._fallback_flood_fill(chunk_data, seed_point, threshold)
            
            # Save result with compression
            output_chunk_path = f"{output_path}_chunk_{chunk_coords}.npz"
            self.memory_manager.save_chunk_compressed(
                output_chunk_path, chunk_coords, segmentation
            )
            
            results.append({
                'seed_point': seed_point,
                'chunk_coords': chunk_coords,
                'segmentation_shape': segmentation.shape,
                'output_path': output_chunk_path
            })
        
        # Record performance
        processing_time = time.time() - start_time
        self.performance_stats['processing_times'].append(processing_time)
        
        # Calculate throughput
        total_voxels = sum(r['segmentation_shape'][0] * r['segmentation_shape'][1] * 
                          r['segmentation_shape'][2] for r in results)
        throughput = total_voxels / processing_time
        self.performance_stats['throughput'].append(throughput)
        
        logger.info(f"Processed {len(seed_points)} seed points in {processing_time:.2f}s "
                   f"({throughput:.0f} voxels/s)")
        
        return {
            'results': results,
            'processing_time': processing_time,
            'throughput': throughput,
            'memory_usage': psutil.virtual_memory().percent
        }
    
    def _fallback_flood_fill(self, volume: np.ndarray, seed_point: Tuple[int, int, int],
                             threshold: float) -> np.ndarray:
        """Fallback flood-fill implementation."""
        from queue import PriorityQueue
        
        segmentation = np.zeros_like(volume, dtype=bool)
        visited = np.zeros_like(volume, dtype=bool)
        
        queue = PriorityQueue()
        queue.put((-volume[seed_point], seed_point))
        visited[seed_point] = True
        
        while not queue.empty():
            priority, (z, y, x) = queue.get()
            segmentation[z, y, x] = True
            
            # Check neighbors
            for dz, dy, dx in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                nz, ny, nx = z + dz, y + dy, x + dx
                
                if (0 <= nz < volume.shape[0] and 0 <= ny < volume.shape[1] and 
                    0 <= nx < volume.shape[2] and not visited[nz, ny, nx]):
                    if volume[nz, ny, nx] > threshold:
                        queue.put((-volume[nz, ny, nx], (nz, ny, nx)))
                        visited[nz, ny, nx] = True
        
        return segmentation
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_stats['processing_times']:
            return {'error': 'No performance data available'}
        
        return {
            'avg_processing_time': np.mean(self.performance_stats['processing_times']),
            'avg_throughput': np.mean(self.performance_stats['throughput']),
            'total_operations': len(self.performance_stats['processing_times']),
            'memory_usage': psutil.virtual_memory().percent,
            'optimizations_enabled': {
                'jax': self.config.use_jax and JAX_AVAILABLE,
                'cpp_extensions': self.config.use_cpp_extensions,
                'memory_mapping': self.config.use_memory_mapping,
                'mixed_precision': self.config.use_mixed_precision
            }
        }

# Performance comparison function
def benchmark_optimizations(volume_path: str, seed_points: List[Tuple[int, int, int]]):
    """Benchmark different optimization approaches."""
    
    # Test configurations
    configs = [
        OptimizationConfig(use_jax=False, use_cpp_extensions=False, use_memory_mapping=False),  # Baseline
        OptimizationConfig(use_jax=True, use_cpp_extensions=False, use_memory_mapping=True),   # JAX only
        OptimizationConfig(use_jax=False, use_cpp_extensions=True, use_memory_mapping=True),   # C++ only
        OptimizationConfig(use_jax=True, use_cpp_extensions=True, use_memory_mapping=True),    # All optimizations
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        logger.info(f"Testing configuration {i+1}/{len(configs)}")
        
        try:
            pipeline = OptimizedConnectomicsPipeline(config)
            result = pipeline.process_volume_optimized(volume_path, f"output_config_{i}", seed_points)
            
            results[f"config_{i}"] = {
                'config': config,
                'processing_time': result['processing_time'],
                'throughput': result['throughput'],
                'memory_usage': result['memory_usage']
            }
            
        except Exception as e:
            logger.error(f"Configuration {i} failed: {e}")
            results[f"config_{i}"] = {'error': str(e)}
    
    # Print comparison
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*80)
    
    for config_name, result in results.items():
        if 'error' in result:
            print(f"{config_name}: ERROR - {result['error']}")
        else:
            print(f"{config_name}:")
            print(f"  Processing time: {result['processing_time']:.2f}s")
            print(f"  Throughput: {result['throughput']:.0f} voxels/s")
            print(f"  Memory usage: {result['memory_usage']:.1f}%")
            print()
    
    return results

if __name__ == "__main__":
    # Example usage
    config = OptimizationConfig(
        use_jax=True,
        use_cpp_extensions=True,
        use_memory_mapping=True,
        chunk_size=(512, 512, 512),
        batch_size=16
    )
    
    pipeline = OptimizedConnectomicsPipeline(config)
    
    # Test with sample data
    # benchmark_optimizations("sample_volume.npy", [(100, 100, 100), (200, 200, 200)])
    
    # Get performance report
    report = pipeline.get_performance_report()
    print("Performance Report:", report) 