#!/usr/bin/env python3
"""
Production H01 Processing Pipeline
=================================
Scalable pipeline for processing large H01 brain regions with:
- Distributed processing
- Memory optimization
- Progress monitoring
- Error handling and recovery
- Batch processing
"""

import numpy as np
import os
import json
import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil
import threading
from datetime import datetime
import queue

# Import our pipeline components
from neuron_tracer_3d import NeuronTracer3D
from visualization import H01Visualizer
from extract_h01_brain_regions import H01RegionExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for production processing."""
    # Processing parameters
    chunk_size: Tuple[int, int, int] = (256, 256, 128)
    overlap: Tuple[int, int, int] = (32, 32, 16)
    batch_size: int = 4
    
    # Resource limits
    max_memory_gb: float = 32.0
    max_cpu_percent: float = 80.0
    num_workers: int = mp.cpu_count()
    
    # Output settings
    output_dir: str = "h01_production_results"
    save_intermediate: bool = True
    compression: str = "gzip"
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: int = 30  # seconds

class ResourceMonitor:
    """Monitor system resources during processing."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.monitoring = config.enable_monitoring
        self.interval = config.monitoring_interval
        self.stats = []
        self.stop_event = threading.Event()
        self.monitor_thread = None
        
    def start(self):
        """Start resource monitoring."""
        if not self.monitoring:
            return
            
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop resource monitoring."""
        if self.monitor_thread:
            self.stop_event.set()
            self.monitor_thread.join()
            logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                stats = self._get_system_stats()
                self.stats.append(stats)
                
                # Log if resources are high
                if stats['memory_percent'] > 80 or stats['cpu_percent'] > 80:
                    logger.warning(f"High resource usage: CPU {stats['cpu_percent']:.1f}%, "
                                 f"Memory {stats['memory_percent']:.1f}%")
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.interval)
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        if not self.stats:
            return {}
        
        cpu_values = [s['cpu_percent'] for s in self.stats]
        memory_values = [s['memory_percent'] for s in self.stats]
        
        return {
            'duration_seconds': len(self.stats) * self.interval,
            'cpu_avg': np.mean(cpu_values),
            'cpu_max': np.max(cpu_values),
            'memory_avg': np.mean(memory_values),
            'memory_max': np.max(memory_values),
            'samples': len(self.stats)
        }

class ProductionH01Processor:
    """Production-scale H01 data processor."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.extractor = H01RegionExtractor()
        self.monitor = ResourceMonitor(config)
        
        # Processing state
        self.processed_regions = []
        self.failed_regions = []
        self.processing_stats = {}
        
        logger.info(f"Production H01 processor initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Workers: {config.num_workers}")
        logger.info(f"Memory limit: {config.max_memory_gb} GB")
    
    def process_region(self, region_name: str, size: str = "medium") -> Dict[str, Any]:
        """Process a single brain region."""
        start_time = time.time()
        region_output_dir = self.output_dir / f"{region_name}_{size}"
        region_output_dir.mkdir(exist_ok=True)
        
        try:
            logger.info(f"Processing region: {region_name} ({size})")
            
            # Step 1: Extract region
            logger.info(f"  Step 1: Extracting {region_name}...")
            region_data = self.extractor.extract_region(region_name, size)
            
            if region_data is None:
                raise RuntimeError(f"Failed to extract region {region_name}")
            
            # Save raw data
            raw_file = region_output_dir / "raw_data.npy"
            np.save(raw_file, region_data)
            logger.info(f"  ✓ Saved raw data: {raw_file}")
            
            # Step 2: Create segmentation
            logger.info(f"  Step 2: Creating segmentation...")
            segmentation = self._create_segmentation(region_data)
            
            seg_file = region_output_dir / "segmentation.npy"
            np.save(seg_file, segmentation)
            logger.info(f"  ✓ Saved segmentation: {seg_file}")
            
            # Step 3: Neuron tracing
            logger.info(f"  Step 3: Tracing neurons...")
            tracer = NeuronTracer3D(segmentation_data=segmentation)
            
            # Analyze connectivity
            tracer.analyze_connectivity(distance_threshold=10.0)
            
            # Export traces
            traces_file = region_output_dir / "traces.json"
            tracer.export_traces(str(traces_file))
            logger.info(f"  ✓ Saved traces: {traces_file}")
            
            # Step 4: Visualization
            logger.info(f"  Step 4: Creating visualizations...")
            self._create_visualizations(region_data, segmentation, tracer, region_output_dir)
            
            # Step 5: Create comprehensive report
            logger.info(f"  Step 5: Generating report...")
            self._create_region_report(region_name, size, region_data, segmentation, 
                                     tracer, region_output_dir)
            
            processing_time = time.time() - start_time
            
            result = {
                'region_name': region_name,
                'size': size,
                'status': 'success',
                'processing_time': processing_time,
                'data_shape': region_data.shape,
                'num_neurons': len(tracer.traced_neurons),
                'output_dir': str(region_output_dir)
            }
            
            logger.info(f"✓ Successfully processed {region_name} in {processing_time:.1f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"✗ Failed to process {region_name}: {e}")
            
            return {
                'region_name': region_name,
                'size': size,
                'status': 'failed',
                'error': str(e),
                'processing_time': processing_time
            }
    
    def _create_segmentation(self, data: np.ndarray) -> np.ndarray:
        """Create segmentation from raw data."""
        # Threshold the data
        threshold = np.percentile(data[data > 0], 50)
        binary = (data > threshold).astype(np.uint8)
        
        # Label connected components
        from skimage import measure
        labeled = measure.label(binary)
        
        return labeled
    
    def _create_visualizations(self, volume: np.ndarray, segmentation: np.ndarray, 
                             tracer: NeuronTracer3D, output_dir: Path):
        """Create visualizations for the region."""
        try:
            # Create comprehensive visualization
            tracer.create_comprehensive_visualization(str(output_dir))
            
            # Create additional visualizations
            viz = H01Visualizer(str(output_dir))
            if "segmentation" in viz.get_available_datasets():
                viz.create_2d_slice_viewer("segmentation", 
                                         save_path=str(output_dir / "slice_view.png"))
            
            logger.info(f"  ✓ Created visualizations")
            
        except Exception as e:
            logger.warning(f"  ⚠ Visualization failed: {e}")
    
    def _create_region_report(self, region_name: str, size: str, volume: np.ndarray,
                            segmentation: np.ndarray, tracer: NeuronTracer3D, 
                            output_dir: Path):
        """Create a comprehensive report for the region."""
        report = {
            'region_name': region_name,
            'size': size,
            'processing_timestamp': datetime.now().isoformat(),
            'data_info': {
                'volume_shape': list(volume.shape),
                'volume_size_mb': volume.nbytes / (1024 * 1024),
                'segmentation_shape': list(segmentation.shape),
                'num_components': int(np.max(segmentation))
            },
            'tracing_results': {
                'num_neurons': len(tracer.traced_neurons),
                'neuron_info': [
                    {
                        'id': int(neuron_id),
                        'volume': int(neuron.volume),
                        'confidence': float(neuron.confidence),
                        'connections': int(len(neuron.connectivity))
                    }
                    for neuron_id, neuron in tracer.traced_neurons.items()
                ]
            },
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
            }
        }
        
        report_file = output_dir / "processing_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"  ✓ Created report: {report_file}")
    
    def process_multiple_regions(self, regions: List[str], size: str = "medium") -> Dict[str, Any]:
        """Process multiple brain regions."""
        logger.info(f"Starting batch processing of {len(regions)} regions")
        
        # Start monitoring
        self.monitor.start()
        
        results = []
        start_time = time.time()
        
        try:
            # Process regions sequentially for now (can be parallelized)
            for region_name in regions:
                result = self.process_region(region_name, size)
                results.append(result)
                
                if result['status'] == 'success':
                    self.processed_regions.append(result)
                else:
                    self.failed_regions.append(result)
                
                # Check resource limits
                if self._check_resource_limits():
                    logger.warning("Resource limits reached, pausing...")
                    time.sleep(60)  # Wait for resources to free up
        
        finally:
            # Stop monitoring
            self.monitor.stop()
        
        total_time = time.time() - start_time
        
        # Create batch summary
        summary = self._create_batch_summary(results, total_time)
        
        logger.info(f"Batch processing completed in {total_time:.1f}s")
        logger.info(f"Success: {len(self.processed_regions)}, Failed: {len(self.failed_regions)}")
        
        return summary
    
    def _check_resource_limits(self) -> bool:
        """Check if we're approaching resource limits."""
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent()
        
        return (memory_percent > self.config.max_cpu_percent or 
                cpu_percent > self.config.max_cpu_percent)
    
    def _create_batch_summary(self, results: List[Dict], total_time: float) -> Dict[str, Any]:
        """Create a summary of batch processing."""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        summary = {
            'batch_info': {
                'total_regions': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'total_time': total_time,
                'avg_time_per_region': total_time / len(results) if results else 0
            },
            'resource_usage': self.monitor.get_summary(),
            'results': results
        }
        
        # Save summary
        summary_file = self.output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch summary saved: {summary_file}")
        return summary

def main():
    """Main function for production processing."""
    print("Production H01 Processing Pipeline")
    print("=" * 50)
    
    # Configuration
    config = ProcessingConfig(
        chunk_size=(256, 256, 128),
        overlap=(32, 32, 16),
        batch_size=4,
        max_memory_gb=32.0,
        max_cpu_percent=80.0,
        num_workers=mp.cpu_count(),
        output_dir="h01_production_results",
        enable_monitoring=True
    )
    
    # Initialize processor
    processor = ProductionH01Processor(config)
    
    # Define regions to process
    regions_to_process = [
        "prefrontal_cortex",
        "hippocampus", 
        "visual_cortex"
    ]
    
    size = "medium"  # Start with medium size for testing
    
    print(f"Processing regions: {regions_to_process}")
    print(f"Size: {size}")
    print(f"Output directory: {config.output_dir}")
    
    try:
        # Process regions
        summary = processor.process_multiple_regions(regions_to_process, size)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*50}")
        
        batch_info = summary['batch_info']
        print(f"Total regions: {batch_info['total_regions']}")
        print(f"Successful: {batch_info['successful']}")
        print(f"Failed: {batch_info['failed']}")
        print(f"Total time: {batch_info['total_time']:.1f}s")
        print(f"Average time per region: {batch_info['avg_time_per_region']:.1f}s")
        
        if summary['resource_usage']:
            res = summary['resource_usage']
            print(f"\nResource Usage:")
            print(f"  CPU avg/max: {res['cpu_avg']:.1f}% / {res['cpu_max']:.1f}%")
            print(f"  Memory avg/max: {res['memory_avg']:.1f}% / {res['memory_max']:.1f}%")
            print(f"  Duration: {res['duration_seconds']:.0f}s")
        
        print(f"\nResults saved in: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 