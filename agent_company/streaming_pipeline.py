"""
Real-time Streaming Pipeline for Connectomics Data
==================================================

Enables real-time processing of connectomics data streams with
continuous learning and live analysis capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
import threading
import queue
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import PipelineConfig, load_config
from enhanced_pipeline import EnhancedConnectomicsPipeline
from monitoring import MetricsCollector, PerformanceMonitor
from model_server import ModelServer

logger = logging.getLogger(__name__)

class DataStream:
    """
    Real-time data stream for connectomics data.
    """
    
    def __init__(self, buffer_size: int = 100, max_wait_time: float = 1.0):
        """
        Initialize data stream.
        
        Args:
            buffer_size: Maximum size of data buffer
            max_wait_time: Maximum time to wait for data
        """
        self.buffer_size = buffer_size
        self.max_wait_time = max_wait_time
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.is_running = False
        self.callbacks = []
        
        logger.info(f"Data stream initialized with buffer size {buffer_size}")
    
    def add_callback(self, callback: Callable):
        """Add callback function for data processing."""
        self.callbacks.append(callback)
    
    def start(self):
        """Start the data stream."""
        self.is_running = True
        logger.info("Data stream started")
    
    def stop(self):
        """Stop the data stream."""
        self.is_running = False
        logger.info("Data stream stopped")
    
    def put_data(self, data: np.ndarray, metadata: Optional[Dict] = None):
        """
        Add data to the stream.
        
        Args:
            data: Input data array
            metadata: Optional metadata
        """
        if not self.is_running:
            logger.warning("Data stream not running, ignoring data")
            return
        
        try:
            # Add timestamp
            if metadata is None:
                metadata = {}
            metadata['timestamp'] = datetime.now().isoformat()
            
            # Put data in queue
            self.data_queue.put((data, metadata), timeout=self.max_wait_time)
            
        except queue.Full:
            logger.warning("Data buffer full, dropping oldest data")
            try:
                self.data_queue.get_nowait()  # Remove oldest data
                self.data_queue.put((data, metadata), timeout=self.max_wait_time)
            except queue.Full:
                logger.error("Failed to add data to stream")
    
    def get_data(self) -> Optional[tuple]:
        """Get data from the stream."""
        try:
            return self.data_queue.get(timeout=self.max_wait_time)
        except queue.Empty:
            return None
    
    def process_stream(self):
        """Process data from the stream."""
        while self.is_running:
            data_item = self.get_data()
            if data_item is not None:
                data, metadata = data_item
                
                # Call all registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(data, metadata)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")


class StreamingProcessor:
    """
    Real-time processor for connectomics data streams.
    """
    
    def __init__(self, config: PipelineConfig, model_path: Optional[str] = None):
        """
        Initialize streaming processor.
        
        Args:
            config: Pipeline configuration
            model_path: Path to trained model
        """
        self.config = config
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.data_stream = None
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
        
        # Processing statistics
        self.processed_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Setup model
        self._setup_model()
        
        logger.info(f"Streaming processor initialized on device: {self.device}")
    
    def _setup_model(self):
        """Setup the model for inference."""
        try:
            # Create pipeline
            pipeline = EnhancedConnectomicsPipeline(config=self.config)
            
            # Setup model
            if not pipeline.setup_model():
                raise RuntimeError("Failed to setup model")
            
            # Load trained model if provided
            if self.model_path and Path(self.model_path).exists():
                pipeline.trainer.load_checkpoint(self.model_path)
                logger.info(f"Model loaded from: {self.model_path}")
            
            self.model = pipeline.model
            self.model.eval()
            
            logger.info("Model setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def process_data(self, data: np.ndarray, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a single data sample.
        
        Args:
            data: Input data array
            metadata: Optional metadata
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        try:
            # Convert to tensor
            data_tensor = torch.from_numpy(data).float()
            if data_tensor.dim() == 3:
                data_tensor = data_tensor.unsqueeze(0)  # Add batch dimension
            
            # Move to device
            data_tensor = data_tensor.to(self.device)
            
            # Normalize data
            data_tensor = (data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min() + 1e-8)
            
            # Run inference
            with torch.no_grad():
                output = self.model(data_tensor)
            
            # Convert output to numpy
            output_np = output.cpu().numpy()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self.processed_count += 1
            self.total_processing_time += processing_time
            
            # Record metrics
            self.metrics_collector.record_inference_time(processing_time)
            self.performance_monitor.record_performance('inference_time', processing_time, metadata)
            
            # Prepare results
            results = {
                'input_shape': data.shape,
                'output_shape': output_np.shape,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'output': output_np
            }
            
            logger.debug(f"Processed data in {processing_time:.4f}s")
            return results
            
        except Exception as e:
            self.error_count += 1
            processing_time = time.time() - start_time
            
            logger.error(f"Processing error: {e}")
            self.metrics_collector.record_error('processing_error')
            
            return {
                'error': str(e),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
    
    def start_streaming(self, data_source: Callable, interval: float = 1.0):
        """
        Start streaming processing.
        
        Args:
            data_source: Function that provides data
            interval: Time interval between data samples
        """
        self.data_stream = DataStream()
        self.data_stream.add_callback(self.process_data)
        
        # Start data stream
        self.data_stream.start()
        
        # Start data source thread
        def data_source_thread():
            while self.data_stream.is_running:
                try:
                    data = data_source()
                    if data is not None:
                        self.data_stream.put_data(data)
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Data source error: {e}")
                    time.sleep(interval)
        
        # Start processing thread
        def processing_thread():
            self.data_stream.process_stream()
        
        # Start threads
        self.source_thread = threading.Thread(target=data_source_thread, daemon=True)
        self.processing_thread = threading.Thread(target=processing_thread, daemon=True)
        
        self.source_thread.start()
        self.processing_thread.start()
        
        logger.info("Streaming processing started")
    
    def stop_streaming(self):
        """Stop streaming processing."""
        if self.data_stream:
            self.data_stream.stop()
        
        logger.info("Streaming processing stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        avg_processing_time = (self.total_processing_time / self.processed_count 
                              if self.processed_count > 0 else 0.0)
        
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'success_rate': (self.processed_count - self.error_count) / max(self.processed_count, 1),
            'performance_summary': self.performance_monitor.get_performance_summary()
        }


class ContinuousLearningPipeline:
    """
    Continuous learning pipeline that updates the model with new data.
    """
    
    def __init__(self, config: PipelineConfig, model_path: Optional[str] = None,
                 update_frequency: int = 100, learning_rate: float = 1e-4):
        """
        Initialize continuous learning pipeline.
        
        Args:
            config: Pipeline configuration
            model_path: Path to initial model
            update_frequency: How often to update the model
            learning_rate: Learning rate for continuous learning
        """
        self.config = config
        self.model_path = model_path
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        
        # Initialize components
        self.streaming_processor = StreamingProcessor(config, model_path)
        self.data_buffer = []
        self.update_count = 0
        
        # Setup optimizer for continuous learning
        self.optimizer = torch.optim.Adam(
            self.streaming_processor.model.parameters(),
            lr=learning_rate
        )
        
        logger.info("Continuous learning pipeline initialized")
    
    def process_and_learn(self, data: np.ndarray, metadata: Optional[Dict] = None):
        """
        Process data and potentially update the model.
        
        Args:
            data: Input data
            metadata: Optional metadata
        """
        # Process data
        results = self.streaming_processor.process_data(data, metadata)
        
        # Add to buffer for learning
        if 'error' not in results:
            self.data_buffer.append((data, results['output'], metadata))
        
        # Check if we should update the model
        if len(self.data_buffer) >= self.update_frequency:
            self._update_model()
    
    def _update_model(self):
        """Update the model with buffered data."""
        try:
            logger.info(f"Updating model with {len(self.data_buffer)} samples")
            
            # Prepare training data
            inputs = []
            targets = []
            
            for data, target, _ in self.data_buffer:
                inputs.append(torch.from_numpy(data).float())
                targets.append(torch.from_numpy(target).float())
            
            # Stack into batches
            inputs = torch.stack(inputs).to(self.streaming_processor.device)
            targets = torch.stack(targets).to(self.streaming_processor.device)
            
            # Training step
            self.optimizer.zero_grad()
            
            outputs = self.streaming_processor.model(inputs)
            loss = F.mse_loss(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            # Clear buffer
            self.data_buffer.clear()
            self.update_count += 1
            
            logger.info(f"Model updated. Loss: {loss.item():.6f}")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def save_model(self, path: str):
        """Save the current model."""
        try:
            torch.save({
                'model_state_dict': self.streaming_processor.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'update_count': self.update_count,
                'config': self.config.to_dict()
            }, path)
            logger.info(f"Model saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")


class RealTimeAnalytics:
    """
    Real-time analytics for streaming connectomics data.
    """
    
    def __init__(self):
        """Initialize real-time analytics."""
        self.analytics_data = []
        self.alert_thresholds = {}
        self.alert_callbacks = []
        
    def add_data_point(self, data: np.ndarray, results: Dict[str, Any]):
        """Add a data point for analytics."""
        analytics_point = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'processing_time': results.get('processing_time', 0),
            'output_shape': results.get('output_shape', None),
            'error': results.get('error', None)
        }
        
        self.analytics_data.append(analytics_point)
        
        # Check for alerts
        self._check_alerts(analytics_point)
    
    def set_alert_threshold(self, metric: str, threshold: float, condition: str = 'gt'):
        """Set alert threshold for a metric."""
        self.alert_thresholds[metric] = {'threshold': threshold, 'condition': condition}
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, data_point: Dict[str, Any]):
        """Check if any alerts should be triggered."""
        for metric, config in self.alert_thresholds.items():
            if metric in data_point:
                value = data_point[metric]
                threshold = config['threshold']
                condition = config['condition']
                
                triggered = False
                if condition == 'gt' and value > threshold:
                    triggered = True
                elif condition == 'lt' and value < threshold:
                    triggered = True
                elif condition == 'eq' and value == threshold:
                    triggered = True
                
                if triggered:
                    alert = {
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'condition': condition,
                        'timestamp': data_point['timestamp']
                    }
                    
                    # Call alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback error: {e}")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary."""
        if not self.analytics_data:
            return {}
        
        processing_times = [d['processing_time'] for d in self.analytics_data if 'processing_time' in d]
        error_count = len([d for d in self.analytics_data if d.get('error')])
        
        return {
            'total_data_points': len(self.analytics_data),
            'average_processing_time': np.mean(processing_times) if processing_times else 0,
            'max_processing_time': np.max(processing_times) if processing_times else 0,
            'error_rate': error_count / len(self.analytics_data),
            'last_update': self.analytics_data[-1]['timestamp'] if self.analytics_data else None
        }


def create_streaming_pipeline(config_path: Optional[str] = None,
                            model_path: Optional[str] = None,
                            enable_continuous_learning: bool = False) -> StreamingProcessor:
    """
    Create a streaming pipeline for real-time processing.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model
        enable_continuous_learning: Whether to enable continuous learning
        
    Returns:
        Streaming processor or continuous learning pipeline
    """
    config = load_config(config_path, "production")
    
    if enable_continuous_learning:
        return ContinuousLearningPipeline(config, model_path)
    else:
        return StreamingProcessor(config, model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Streaming Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--continuous-learning", action="store_true", 
                       help="Enable continuous learning")
    parser.add_argument("--interval", type=float, default=1.0, 
                       help="Data processing interval")
    
    args = parser.parse_args()
    
    # Create streaming pipeline
    pipeline = create_streaming_pipeline(
        config_path=args.config,
        model_path=args.model,
        enable_continuous_learning=args.continuous_learning
    )
    
    # Example data source function
    def mock_data_source():
        """Mock data source for testing."""
        return np.random.rand(64, 64, 64).astype(np.float32)
    
    # Start streaming
    pipeline.start_streaming(mock_data_source, args.interval)
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop_streaming()
        print("Streaming stopped") 