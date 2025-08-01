#!/usr/bin/env python3
"""
Production-Grade Distributed Connectomics System
===============================================
Designed for petabyte to exabyte scale processing with maximum robustness and efficiency.
"""

import os
import sys
import logging
import time
import json
import asyncio
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as tmp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import queue
import threading
from collections import deque
import signal
import psutil
import gc
import pickle
import hashlib
import zlib
from datetime import datetime, timedelta
import traceback

# Production imports
try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import dask.array as da
    import dask.distributed
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import zarr
    import numcodecs
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

try:
    from cloudvolume import CloudVolume
    CLOUDVOLUME_AVAILABLE = True
except ImportError:
    CLOUDVOLUME_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ProductionConfig:
    """Production configuration for exabyte-scale processing."""
    # System configuration
    max_memory_gb: int = 1024  # 1TB memory limit
    max_cpu_cores: int = 128
    max_gpu_memory_gb: int = 80
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    
    # Distributed processing
    num_nodes: int = 100
    gpus_per_node: int = 8
    workers_per_node: int = 16
    batch_size_per_gpu: int = 4
    
    # Data management
    chunk_size: Tuple[int, int, int] = (512, 512, 512)
    overlap_size: Tuple[int, int, int] = (64, 64, 64)
    compression_level: int = 6
    use_memory_mapping: bool = True
    cache_size_gb: int = 100
    
    # Fault tolerance
    max_retries: int = 3
    checkpoint_interval: int = 1000
    backup_interval: int = 10000
    health_check_interval: int = 30
    
    # Monitoring
    enable_telemetry: bool = True
    metrics_interval: int = 60
    alert_thresholds: Dict[str, float] = None
    
    # Storage
    storage_backend: str = "zarr"  # "zarr", "h5", "cloudvolume"
    storage_path: str = "/data/connectomics"
    temp_dir: str = "/tmp/connectomics"
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'memory_usage': 0.9,
                'gpu_memory_usage': 0.95,
                'disk_usage': 0.85,
                'error_rate': 0.01
            }

class MemoryManager:
    """Advanced memory management for large-scale processing."""
    
    def __init__(self, max_memory_gb: int, cache_size_gb: int):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.cache_size_bytes = cache_size_gb * 1024**3
        self.current_usage = 0
        self.cache = {}
        self.cache_order = deque()
        self.lock = threading.Lock()
        
        # Memory monitoring
        self.memory_history = deque(maxlen=1000)
        self.gc_threshold = 0.8
        
        logger.info(f"Memory manager initialized: {max_memory_gb}GB max, {cache_size_gb}GB cache")
    
    def allocate(self, size_bytes: int, key: str = None) -> bool:
        """Allocate memory with automatic garbage collection."""
        with self.lock:
            # Check if allocation would exceed limits
            if self.current_usage + size_bytes > self.max_memory_bytes:
                # Try to free cache
                self._free_cache(size_bytes)
                
                # If still not enough, force garbage collection
                if self.current_usage + size_bytes > self.max_memory_bytes:
                    self._force_gc()
                    
                    # If still not enough, fail
                    if self.current_usage + size_bytes > self.max_memory_bytes:
                        logger.warning(f"Memory allocation failed: {size_bytes} bytes")
                        return False
            
            self.current_usage += size_bytes
            self.memory_history.append((time.time(), self.current_usage))
            
            # Cache the allocation if key provided
            if key:
                self.cache[key] = size_bytes
                self.cache_order.append(key)
                
                # Maintain cache size
                while len(self.cache) > 0 and self._get_cache_size() > self.cache_size_bytes:
                    self._evict_oldest()
            
            return True
    
    def deallocate(self, size_bytes: int, key: str = None):
        """Deallocate memory."""
        with self.lock:
            self.current_usage = max(0, self.current_usage - size_bytes)
            
            if key and key in self.cache:
                del self.cache[key]
                self.cache_order.remove(key)
    
    def _free_cache(self, required_bytes: int):
        """Free cache to make room for required bytes."""
        freed_bytes = 0
        while freed_bytes < required_bytes and len(self.cache) > 0:
            freed_bytes += self._evict_oldest()
    
    def _evict_oldest(self) -> int:
        """Evict oldest cache entry."""
        if len(self.cache_order) == 0:
            return 0
        
        oldest_key = self.cache_order.popleft()
        size_bytes = self.cache.pop(oldest_key, 0)
        return size_bytes
    
    def _get_cache_size(self) -> int:
        """Get total cache size."""
        return sum(self.cache.values())
    
    def _force_gc(self):
        """Force garbage collection."""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self.lock:
            return {
                'current_usage_gb': self.current_usage / 1024**3,
                'max_memory_gb': self.max_memory_bytes / 1024**3,
                'cache_size_gb': self._get_cache_size() / 1024**3,
                'cache_entries': len(self.cache),
                'utilization_percent': (self.current_usage / self.max_memory_bytes) * 100
            }

class DistributedDataManager:
    """Distributed data management for petabyte-scale datasets."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.storage_backend = self._initialize_storage()
        self.chunk_cache = {}
        self.chunk_locks = {}
        self.global_lock = threading.Lock()
        
        # Initialize distributed storage
        if DASK_AVAILABLE:
            self.dask_client = self._initialize_dask()
        else:
            self.dask_client = None
        
        logger.info(f"Distributed data manager initialized with {config.storage_backend} backend")
    
    def _initialize_storage(self):
        """Initialize storage backend."""
        if self.config.storage_backend == "zarr" and ZARR_AVAILABLE:
            return "zarr"
        elif self.config.storage_backend == "cloudvolume" and CLOUDVOLUME_AVAILABLE:
            return "cloudvolume"
        else:
            return "h5"  # Fallback
    
    def _initialize_dask(self):
        """Initialize Dask distributed client."""
        try:
            cluster = LocalCluster(
                n_workers=self.config.workers_per_node,
                memory_limit=f"{self.config.max_memory_gb}GB",
                threads_per_worker=2
            )
            client = Client(cluster)
            logger.info(f"Dask cluster initialized with {len(cluster.workers)} workers")
            return client
        except Exception as e:
            logger.warning(f"Failed to initialize Dask: {e}")
            return None
    
    def load_chunk(self, chunk_coords: Tuple[int, int, int], 
                   chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load data chunk with caching and error handling."""
        chunk_key = f"{chunk_coords}_{chunk_size}"
        
        # Check cache first
        if chunk_key in self.chunk_cache:
            return self.chunk_cache[chunk_key]
        
        # Load from storage
        try:
            chunk_data = self._load_from_storage(chunk_coords, chunk_size)
            
            # Cache the chunk
            with self.global_lock:
                self.chunk_cache[chunk_key] = chunk_data
                
                # Limit cache size
                if len(self.chunk_cache) > 1000:
                    oldest_key = next(iter(self.chunk_cache))
                    del self.chunk_cache[oldest_key]
            
            return chunk_data
            
        except Exception as e:
            logger.error(f"Failed to load chunk {chunk_coords}: {e}")
            raise
    
    def _load_from_storage(self, chunk_coords: Tuple[int, int, int], 
                          chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk from storage backend."""
        if self.storage_backend == "zarr":
            return self._load_zarr_chunk(chunk_coords, chunk_size)
        elif self.storage_backend == "cloudvolume":
            return self._load_cloudvolume_chunk(chunk_coords, chunk_size)
        else:
            return self._load_h5_chunk(chunk_coords, chunk_size)
    
    def _load_zarr_chunk(self, chunk_coords: Tuple[int, int, int], 
                        chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk from Zarr storage."""
        import zarr
        
        z, y, x = chunk_coords
        dz, dy, dx = chunk_size
        
        # Load from Zarr array
        zarr_array = zarr.open(self.config.storage_path, mode='r')
        chunk = zarr_array[z:z+dz, y:y+dy, x:x+dx]
        
        return np.array(chunk)
    
    def _load_cloudvolume_chunk(self, chunk_coords: Tuple[int, int, int], 
                               chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk from CloudVolume storage."""
        from cloudvolume import CloudVolume
        
        vol = CloudVolume(self.config.storage_path)
        z, y, x = chunk_coords
        dz, dy, dx = chunk_size
        
        chunk = vol[z:z+dz, y:y+dy, x:x+dx]
        return chunk[:, :, :, 0]  # Remove channel dimension
    
    def _load_h5_chunk(self, chunk_coords: Tuple[int, int, int], 
                      chunk_size: Tuple[int, int, int]) -> np.ndarray:
        """Load chunk from HDF5 storage."""
        import h5py
        
        z, y, x = chunk_coords
        dz, dy, dx = chunk_size
        
        with h5py.File(self.config.storage_path, 'r') as f:
            dataset = f['data']
            chunk = dataset[z:z+dz, y:y+dy, x:x+dx]
        
        return np.array(chunk)
    
    def save_chunk(self, chunk_coords: Tuple[int, int, int], 
                   chunk_data: np.ndarray, compression: bool = True):
        """Save chunk to storage with compression."""
        try:
            if compression:
                chunk_data = self._compress_chunk(chunk_data)
            
            self._save_to_storage(chunk_coords, chunk_data)
            
        except Exception as e:
            logger.error(f"Failed to save chunk {chunk_coords}: {e}")
            raise
    
    def _compress_chunk(self, data: np.ndarray) -> np.ndarray:
        """Compress chunk data."""
        # Use zlib compression for efficiency
        compressed = zlib.compress(data.tobytes(), level=self.config.compression_level)
        return np.frombuffer(compressed, dtype=np.uint8)
    
    def _save_to_storage(self, chunk_coords: Tuple[int, int, int], chunk_data: np.ndarray):
        """Save chunk to storage backend."""
        if self.storage_backend == "zarr":
            self._save_zarr_chunk(chunk_coords, chunk_data)
        elif self.storage_backend == "cloudvolume":
            self._save_cloudvolume_chunk(chunk_coords, chunk_data)
        else:
            self._save_h5_chunk(chunk_coords, chunk_data)

class DistributedModelManager:
    """Distributed model management for efficient inference."""
    
    def __init__(self, config: ProductionConfig, model_class, model_config: Dict[str, Any]):
        self.config = config
        self.model_class = model_class
        self.model_config = model_config
        self.models = {}
        self.model_locks = {}
        self.inference_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Initialize distributed training
        if dist.is_available():
            self._initialize_distributed()
        
        # Start inference workers
        self.workers = []
        self._start_workers()
        
        logger.info(f"Distributed model manager initialized with {config.gpus_per_node} GPUs per node")
    
    def _initialize_distributed(self):
        """Initialize distributed training."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                rank=rank,
                world_size=world_size
            )
            
            logger.info(f"Distributed training initialized: rank {rank}/{world_size}")
    
    def _start_workers(self):
        """Start inference worker processes."""
        for gpu_id in range(self.config.gpus_per_node):
            worker = threading.Thread(
                target=self._inference_worker,
                args=(gpu_id,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _inference_worker(self, gpu_id: int):
        """Inference worker process."""
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model for this GPU
        model = self.model_class(self.model_config)
        model.to(device)
        model.eval()
        
        if dist.is_available() and dist.is_initialized():
            model = DDP(model, device_ids=[gpu_id])
        
        self.models[gpu_id] = model
        self.model_locks[gpu_id] = threading.Lock()
        
        logger.info(f"Inference worker {gpu_id} started on device {device}")
        
        while True:
            try:
                # Get inference task
                task = self.inference_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Process inference
                result = self._process_inference(model, task, device)
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference worker {gpu_id} error: {e}")
                self.result_queue.put({'error': str(e), 'task_id': task.get('task_id')})
    
    def _process_inference(self, model: nn.Module, task: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """Process inference task."""
        task_id = task['task_id']
        data = task['data']
        
        try:
            # Move data to device
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).to(device)
            
            # Run inference
            with torch.no_grad():
                with self.model_locks[device.index]:
                    if self.config.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                    else:
                        output = model(data)
            
            # Move output back to CPU
            if isinstance(output, dict):
                output = {k: v.cpu().numpy() if torch.is_tensor(v) else v 
                         for k, v in output.items()}
            else:
                output = output.cpu().numpy()
            
            return {
                'task_id': task_id,
                'result': output,
                'device': device.index,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Inference error for task {task_id}: {e}")
            return {
                'task_id': task_id,
                'error': str(e),
                'device': device.index,
                'timestamp': time.time()
            }
    
    def submit_inference(self, data: np.ndarray, task_id: str = None) -> str:
        """Submit inference task."""
        if task_id is None:
            task_id = hashlib.md5(f"{time.time()}_{data.shape}".encode()).hexdigest()
        
        task = {
            'task_id': task_id,
            'data': data,
            'timestamp': time.time()
        }
        
        self.inference_queue.put(task)
        return task_id
    
    def get_result(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Get inference result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class ProductionConnectomicsPipeline:
    """Production-grade connectomics pipeline for exabyte-scale processing."""
    
    def __init__(self, config: ProductionConfig, model_class, model_config: Dict[str, Any]):
        self.config = config
        self.memory_manager = MemoryManager(config.max_memory_gb, config.cache_size_gb)
        self.data_manager = DistributedDataManager(config)
        self.model_manager = DistributedModelManager(config, model_class, model_config)
        
        # Processing state
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.error_queue = queue.Queue()
        
        # Monitoring
        self.stats = {
            'chunks_processed': 0,
            'chunks_failed': 0,
            'total_processing_time': 0.0,
            'start_time': time.time(),
            'last_checkpoint': time.time()
        }
        
        # Health monitoring
        self.health_checker = threading.Thread(target=self._health_monitor, daemon=True)
        self.health_checker.start()
        
        # Start processing workers
        self.processing_workers = []
        self._start_processing_workers()
        
        logger.info("Production connectomics pipeline initialized")
    
    def _start_processing_workers(self):
        """Start processing worker threads."""
        for worker_id in range(self.config.workers_per_node):
            worker = threading.Thread(
                target=self._processing_worker,
                args=(worker_id,),
                daemon=True
            )
            worker.start()
            self.processing_workers.append(worker)
    
    def _processing_worker(self, worker_id: int):
        """Processing worker thread."""
        logger.info(f"Processing worker {worker_id} started")
        
        while True:
            try:
                # Get processing task
                task = self.processing_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Process chunk
                result = self._process_chunk(task, worker_id)
                
                if result.get('success', False):
                    self.results_queue.put(result)
                    self.stats['chunks_processed'] += 1
                else:
                    self.error_queue.put(result)
                    self.stats['chunks_failed'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker {worker_id} error: {e}")
                self.error_queue.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'timestamp': time.time()
                })
    
    def _process_chunk(self, task: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
        """Process a single chunk."""
        chunk_coords = task['chunk_coords']
        chunk_size = task['chunk_size']
        task_id = task['task_id']
        
        start_time = time.time()
        
        try:
            # Load chunk data
            chunk_data = self.data_manager.load_chunk(chunk_coords, chunk_size)
            
            # Check memory allocation
            data_size = chunk_data.nbytes
            if not self.memory_manager.allocate(data_size, f"chunk_{task_id}"):
                raise MemoryError(f"Insufficient memory for chunk {task_id}")
            
            # Submit inference
            inference_task_id = self.model_manager.submit_inference(chunk_data, task_id)
            
            # Get inference result
            inference_result = self.model_manager.get_result(timeout=60.0)
            
            if inference_result is None:
                raise TimeoutError(f"Inference timeout for task {task_id}")
            
            if 'error' in inference_result:
                raise RuntimeError(f"Inference error: {inference_result['error']}")
            
            # Save results
            result_data = inference_result['result']
            self.data_manager.save_chunk(chunk_coords, result_data)
            
            # Cleanup
            self.memory_manager.deallocate(data_size, f"chunk_{task_id}")
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            
            return {
                'task_id': task_id,
                'chunk_coords': chunk_coords,
                'worker_id': worker_id,
                'processing_time': processing_time,
                'success': True,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Chunk processing error for {task_id}: {e}")
            return {
                'task_id': task_id,
                'chunk_coords': chunk_coords,
                'worker_id': worker_id,
                'error': str(e),
                'success': False,
                'timestamp': time.time()
            }
    
    def _health_monitor(self):
        """Health monitoring thread."""
        while True:
            try:
                # Check system health
                health_stats = self._get_health_stats()
                
                # Check alert thresholds
                for metric, threshold in self.config.alert_thresholds.items():
                    if health_stats.get(metric, 0) > threshold:
                        logger.warning(f"Health alert: {metric} = {health_stats[metric]:.3f} > {threshold}")
                
                # Log health stats
                if self.config.enable_telemetry:
                    logger.info(f"Health stats: {health_stats}")
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _get_health_stats(self) -> Dict[str, float]:
        """Get system health statistics."""
        # Memory stats
        memory_stats = self.memory_manager.get_memory_stats()
        
        # System stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # GPU stats
        gpu_memory_percent = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            gpu_memory_percent = gpu_memory * 100
        
        # Error rate
        total_chunks = self.stats['chunks_processed'] + self.stats['chunks_failed']
        error_rate = self.stats['chunks_failed'] / max(total_chunks, 1)
        
        return {
            'memory_usage': memory_stats['utilization_percent'] / 100,
            'gpu_memory_usage': gpu_memory_percent / 100,
            'disk_usage': disk_percent / 100,
            'cpu_usage': cpu_percent / 100,
            'error_rate': error_rate
        }
    
    def process_volume(self, volume_path: str, output_path: str, 
                      chunk_coords_list: List[Tuple[int, int, int]]) -> bool:
        """Process entire volume with distributed processing."""
        logger.info(f"Starting volume processing: {len(chunk_coords_list)} chunks")
        
        # Submit all chunks for processing
        for i, chunk_coords in enumerate(chunk_coords_list):
            task = {
                'task_id': f"chunk_{i:06d}",
                'chunk_coords': chunk_coords,
                'chunk_size': self.config.chunk_size,
                'volume_path': volume_path,
                'output_path': output_path
            }
            self.processing_queue.put(task)
        
        # Monitor processing
        total_chunks = len(chunk_coords_list)
        completed_chunks = 0
        failed_chunks = 0
        
        while completed_chunks + failed_chunks < total_chunks:
            # Check results
            try:
                result = self.results_queue.get(timeout=1.0)
                if result.get('success', False):
                    completed_chunks += 1
                else:
                    failed_chunks += 1
            except queue.Empty:
                pass
            
            # Check errors
            try:
                error = self.error_queue.get_nowait()
                logger.error(f"Processing error: {error}")
                failed_chunks += 1
            except queue.Empty:
                pass
            
            # Log progress
            if (completed_chunks + failed_chunks) % 100 == 0:
                progress = (completed_chunks + failed_chunks) / total_chunks * 100
                logger.info(f"Progress: {progress:.1f}% ({completed_chunks} completed, {failed_chunks} failed)")
            
            # Checkpoint
            if time.time() - self.stats['last_checkpoint'] > self.config.checkpoint_interval:
                self._save_checkpoint()
        
        # Final statistics
        success_rate = completed_chunks / total_chunks
        logger.info(f"Processing completed: {success_rate:.1%} success rate")
        
        return success_rate > 0.95  # 95% success threshold
    
    def _save_checkpoint(self):
        """Save processing checkpoint."""
        checkpoint = {
            'stats': self.stats,
            'timestamp': time.time(),
            'memory_stats': self.memory_manager.get_memory_stats()
        }
        
        checkpoint_path = os.path.join(self.config.temp_dir, 'checkpoint.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.stats['last_checkpoint'] = time.time()
        logger.info("Checkpoint saved")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        runtime = time.time() - self.stats['start_time']
        
        return {
            **self.stats,
            'runtime_hours': runtime / 3600,
            'chunks_per_hour': self.stats['chunks_processed'] / (runtime / 3600),
            'success_rate': self.stats['chunks_processed'] / max(self.stats['chunks_processed'] + self.stats['chunks_failed'], 1),
            'health_stats': self._get_health_stats()
        }

# Utility functions for production deployment
def create_production_pipeline(config_dict: Dict[str, Any], model_class, model_config: Dict[str, Any]) -> ProductionConnectomicsPipeline:
    """Create production pipeline from configuration."""
    config = ProductionConfig(**config_dict)
    return ProductionConnectomicsPipeline(config, model_class, model_config)

def estimate_processing_time(volume_size_gb: float, config: ProductionConfig) -> float:
    """Estimate processing time for a volume."""
    # Estimate chunks
    chunk_volume_gb = np.prod(config.chunk_size) * 4 / 1024**3  # 4 bytes per voxel
    num_chunks = volume_size_gb / chunk_volume_gb
    
    # Estimate time per chunk (including I/O)
    time_per_chunk = 30  # seconds (conservative estimate)
    
    # Parallel processing
    total_workers = config.num_nodes * config.workers_per_node
    
    # Estimated time
    estimated_time_hours = (num_chunks * time_per_chunk) / (total_workers * 3600)
    
    return estimated_time_hours

# Example usage
def test_production_pipeline():
    """Test the production pipeline."""
    # Configuration for petabyte-scale processing
    config_dict = {
        'max_memory_gb': 1024,
        'num_nodes': 10,
        'gpus_per_node': 8,
        'workers_per_node': 16,
        'chunk_size': (512, 512, 512),
        'storage_backend': 'zarr',
        'enable_telemetry': True
    }
    
    # Create pipeline
    from transformer_connectomics import TransformerConnectomicsModel, TransformerConfig
    
    model_config = TransformerConfig(
        embed_dim=256,
        num_layers=12,
        num_heads=8
    )
    
    pipeline = create_production_pipeline(config_dict, TransformerConnectomicsModel, model_config.__dict__)
    
    # Test with sample chunk coordinates
    chunk_coords = [(0, 0, 0), (512, 0, 0), (0, 512, 0), (0, 0, 512)]
    
    # Process chunks
    success = pipeline.process_volume(
        volume_path="/data/sample_volume.zarr",
        output_path="/data/output_volume.zarr",
        chunk_coords_list=chunk_coords
    )
    
    # Get statistics
    stats = pipeline.get_statistics()
    print(f"Processing statistics: {stats}")
    
    return pipeline, success

if __name__ == "__main__":
    test_production_pipeline() 