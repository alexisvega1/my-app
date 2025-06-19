#!/usr/bin/env python3
"""
Telemetry and Monitoring System
==============================
Comprehensive monitoring and metrics collection using Prometheus.
"""

import os
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    from prometheus_client import (
        start_http_server, Counter, Gauge, Summary, Histogram,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        multiprocess, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus client available")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available - using stub metrics")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - system metrics disabled")

@dataclass
class MetricValue:
    """A metric value with timestamp."""
    value: float
    timestamp: datetime
    labels: Dict[str, str]

class StubMetrics:
    """Stub metrics for when Prometheus is not available."""
    
    def __init__(self, name: str, description: str, labelnames: List[str] = None):
        self.name = name
        self.description = description
        self.labelnames = labelnames or []
        self.values = []
    
    def inc(self, amount: float = 1.0, **labels):
        """Increment the metric."""
        self.values.append(MetricValue(
            value=amount,
            timestamp=datetime.now(),
            labels=labels
        ))
    
    def set(self, value: float, **labels):
        """Set the metric value."""
        self.values.append(MetricValue(
            value=value,
            timestamp=datetime.now(),
            labels=labels
        ))
    
    def observe(self, value: float, **labels):
        """Observe a value (for histograms/summaries)."""
        self.values.append(MetricValue(
            value=value,
            timestamp=datetime.now(),
            labels=labels
        ))

class TelemetrySystem:
    """Comprehensive telemetry and monitoring system."""
    
    def __init__(self, 
                 port: int = 8000,
                 enable_prometheus: bool = True,
                 enable_system_metrics: bool = True,
                 metrics_prefix: str = "agent_system"):
        """
        Initialize the telemetry system.
        
        Args:
            port: Port for metrics HTTP server
            enable_prometheus: Whether to enable Prometheus metrics
            enable_system_metrics: Whether to collect system metrics
            metrics_prefix: Prefix for all metrics
        """
        self.port = port
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_system_metrics = enable_system_metrics and PSUTIL_AVAILABLE
        self.metrics_prefix = metrics_prefix
        
        # Initialize metrics
        self.metrics = {}
        self._initialize_metrics()
        
        # System monitoring
        self.system_monitor_thread = None
        self.is_monitoring = False
        self.monitoring_interval = 30  # seconds
        
        # Performance tracking
        self.performance_data = {
            'requests': [],
            'errors': [],
            'processing_times': []
        }
        
        # Custom collectors
        self.custom_collectors = {}
        
        logger.info(f"Telemetry system initialized on port {port}")
    
    def _initialize_metrics(self):
        """Initialize all metrics."""
        if self.enable_prometheus:
            self._initialize_prometheus_metrics()
        else:
            self._initialize_stub_metrics()
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.metrics['requests_total'] = Counter(
            f'{self.metrics_prefix}_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.metrics['request_duration'] = Summary(
            f'{self.metrics_prefix}_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.metrics['request_size'] = Histogram(
            f'{self.metrics_prefix}_request_size_bytes',
            'Request size in bytes',
            ['method', 'endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000]
        )
        
        # Error metrics
        self.metrics['errors_total'] = Counter(
            f'{self.metrics_prefix}_errors_total',
            'Total number of errors',
            ['type', 'component']
        )
        
        # Processing metrics
        self.metrics['processing_time'] = Summary(
            f'{self.metrics_prefix}_processing_time_seconds',
            'Processing time in seconds',
            ['component', 'operation']
        )
        
        # System metrics
        self.metrics['cpu_usage'] = Gauge(
            f'{self.metrics_prefix}_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.metrics['memory_usage'] = Gauge(
            f'{self.metrics_prefix}_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.metrics['disk_usage'] = Gauge(
            f'{self.metrics_prefix}_disk_usage_percent',
            'Disk usage percentage',
            ['mount_point']
        )
        
        # Agent-specific metrics
        self.metrics['active_agents'] = Gauge(
            f'{self.metrics_prefix}_active_agents',
            'Number of active agents',
            ['agent_type']
        )
        
        self.metrics['segmentation_requests'] = Counter(
            f'{self.metrics_prefix}_segmentation_requests_total',
            'Total segmentation requests',
            ['model', 'status']
        )
        
        self.metrics['segmentation_accuracy'] = Gauge(
            f'{self.metrics_prefix}_segmentation_accuracy',
            'Segmentation accuracy',
            ['model']
        )
        
        self.metrics['proofreading_requests'] = Counter(
            f'{self.metrics_prefix}_proofreading_requests_total',
            'Total proofreading requests',
            ['triggered', 'status']
        )
        
        self.metrics['training_requests'] = Counter(
            f'{self.metrics_prefix}_training_requests_total',
            'Total training requests',
            ['model', 'status']
        )
        
        self.metrics['model_checkpoints'] = Gauge(
            f'{self.metrics_prefix}_model_checkpoints',
            'Number of model checkpoints',
            ['model']
        )
        
        # Queue metrics
        self.metrics['queue_size'] = Gauge(
            f'{self.metrics_prefix}_queue_size',
            'Queue size',
            ['queue_name']
        )
        
        self.metrics['queue_latency'] = Summary(
            f'{self.metrics_prefix}_queue_latency_seconds',
            'Queue processing latency',
            ['queue_name']
        )
        
        logger.info("Prometheus metrics initialized")
    
    def _initialize_stub_metrics(self):
        """Initialize stub metrics for testing."""
        metric_definitions = {
            'requests_total': 'Total number of requests',
            'request_duration': 'Request duration in seconds',
            'request_size': 'Request size in bytes',
            'errors_total': 'Total number of errors',
            'processing_time': 'Processing time in seconds',
            'cpu_usage': 'CPU usage percentage',
            'memory_usage': 'Memory usage in bytes',
            'disk_usage': 'Disk usage percentage',
            'active_agents': 'Number of active agents',
            'segmentation_requests': 'Total segmentation requests',
            'segmentation_accuracy': 'Segmentation accuracy',
            'proofreading_requests': 'Total proofreading requests',
            'training_requests': 'Total training requests',
            'model_checkpoints': 'Number of model checkpoints',
            'queue_size': 'Queue size',
            'queue_latency': 'Queue processing latency'
        }
        
        for name, description in metric_definitions.items():
            self.metrics[name] = StubMetrics(name, description)
        
        logger.info("Stub metrics initialized")
    
    def start_metrics_server(self, port: Optional[int] = None) -> bool:
        """
        Start the Prometheus metrics HTTP server.
        
        Args:
            port: Port to start server on (overrides instance port)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_prometheus:
            logger.warning("Prometheus not enabled - cannot start metrics server")
            return False
        
        try:
            server_port = port or self.port
            start_http_server(server_port)
            logger.info(f"Prometheus metrics server started on port {server_port}")
            
            # Start system monitoring
            if self.enable_system_metrics:
                self.start_system_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False
    
    def start_system_monitoring(self):
        """Start system monitoring in background thread."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - system monitoring disabled")
            return
        
        if self.is_monitoring:
            logger.warning("System monitoring already running")
            return
        
        self.is_monitoring = True
        self.system_monitor_thread = threading.Thread(
            target=self._system_monitoring_worker,
            daemon=True
        )
        self.system_monitor_thread.start()
        
        logger.info("System monitoring started")
    
    def _system_monitoring_worker(self):
        """Background worker for system monitoring."""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_cpu_usage(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_memory_usage(memory.used)
                
                # Disk usage
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        self.record_disk_usage(partition.mountpoint, usage.percent)
                    except PermissionError:
                        continue
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def stop_system_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if self.system_monitor_thread and self.system_monitor_thread.is_alive():
            self.system_monitor_thread.join(timeout=5.0)
        
        logger.info("System monitoring stopped")
    
    # Metric recording methods
    def record_request(self, method: str, endpoint: str, status: int, duration: float, size: int = 0):
        """Record a request."""
        if self.enable_prometheus:
            self.metrics['requests_total'].labels(method=method, endpoint=endpoint, status=str(status)).inc()
            self.metrics['request_duration'].labels(method=method, endpoint=endpoint).observe(duration)
            if size > 0:
                self.metrics['request_size'].labels(method=method, endpoint=endpoint).observe(size)
        else:
            self.metrics['requests_total'].inc(amount=1.0, method=method, endpoint=endpoint, status=str(status))
            self.metrics['request_duration'].observe(duration, method=method, endpoint=endpoint)
            if size > 0:
                self.metrics['request_size'].observe(size, method=method, endpoint=endpoint)
        
        # Store for analysis
        self.performance_data['requests'].append({
            'method': method,
            'endpoint': endpoint,
            'status': status,
            'duration': duration,
            'size': size,
            'timestamp': datetime.now()
        })
    
    def record_error(self, error_type: str, component: str):
        """Record an error."""
        if self.enable_prometheus:
            self.metrics['errors_total'].labels(type=error_type, component=component).inc()
        else:
            self.metrics['errors_total'].inc(amount=1.0, type=error_type, component=component)
        
        # Store for analysis
        self.performance_data['errors'].append({
            'type': error_type,
            'component': component,
            'timestamp': datetime.now()
        })
    
    def record_processing_time(self, component: str, operation: str, duration: float):
        """Record processing time."""
        if self.enable_prometheus:
            self.metrics['processing_time'].labels(component=component, operation=operation).observe(duration)
        else:
            self.metrics['processing_time'].observe(duration, component=component, operation=operation)
        
        # Store for analysis
        self.performance_data['processing_times'].append({
            'component': component,
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now()
        })
    
    def record_cpu_usage(self, usage_percent: float):
        """Record CPU usage."""
        if self.enable_prometheus:
            self.metrics['cpu_usage'].set(usage_percent)
        else:
            self.metrics['cpu_usage'].set(usage_percent)
    
    def record_memory_usage(self, usage_bytes: int):
        """Record memory usage."""
        if self.enable_prometheus:
            self.metrics['memory_usage'].set(usage_bytes)
        else:
            self.metrics['memory_usage'].set(usage_bytes)
    
    def record_disk_usage(self, mount_point: str, usage_percent: float):
        """Record disk usage."""
        if self.enable_prometheus:
            self.metrics['disk_usage'].labels(mount_point=mount_point).set(usage_percent)
        else:
            self.metrics['disk_usage'].set(usage_percent, mount_point=mount_point)
    
    def record_active_agents(self, agent_type: str, count: int):
        """Record number of active agents."""
        if self.enable_prometheus:
            self.metrics['active_agents'].labels(agent_type=agent_type).set(count)
        else:
            self.metrics['active_agents'].set(count, agent_type=agent_type)
    
    def record_segmentation_request(self, model: str, status: str):
        """Record a segmentation request."""
        if self.enable_prometheus:
            self.metrics['segmentation_requests'].labels(model=model, status=status).inc()
        else:
            self.metrics['segmentation_requests'].inc(amount=1.0, model=model, status=status)
    
    def record_segmentation_accuracy(self, model: str, accuracy: float):
        """Record segmentation accuracy."""
        if self.enable_prometheus:
            self.metrics['segmentation_accuracy'].labels(model=model).set(accuracy)
        else:
            self.metrics['segmentation_accuracy'].set(accuracy, model=model)
    
    def record_proofreading_request(self, triggered: bool, status: str):
        """Record a proofreading request."""
        if self.enable_prometheus:
            self.metrics['proofreading_requests'].labels(
                triggered=str(triggered), status=status
            ).inc()
        else:
            self.metrics['proofreading_requests'].inc(
                amount=1.0, triggered=str(triggered), status=status
            )
    
    def record_training_request(self, model: str, status: str):
        """Record a training request."""
        if self.enable_prometheus:
            self.metrics['training_requests'].labels(model=model, status=status).inc()
        else:
            self.metrics['training_requests'].inc(amount=1.0, model=model, status=status)
    
    def record_model_checkpoints(self, model: str, count: int):
        """Record number of model checkpoints."""
        if self.enable_prometheus:
            self.metrics['model_checkpoints'].labels(model=model).set(count)
        else:
            self.metrics['model_checkpoints'].set(count, model=model)
    
    def record_queue_size(self, queue_name: str, size: int):
        """Record queue size."""
        if self.enable_prometheus:
            self.metrics['queue_size'].labels(queue_name=queue_name).set(size)
        else:
            self.metrics['queue_size'].set(size, queue_name=queue_name)
    
    def record_queue_latency(self, queue_name: str, latency: float):
        """Record queue latency."""
        if self.enable_prometheus:
            self.metrics['queue_latency'].observe(latency, queue=queue_name)
        else:
            self.metrics['queue_latency'].observe(latency, queue=queue_name)
    
    def update_metrics(self, metrics_data: Dict[str, Any]):
        """Update metrics with custom data from pipeline."""
        try:
            # Update H01-specific metrics
            if 'volumes_processed' in metrics_data:
                self.record_segmentation_request('h01_ffn_v2', 'completed')
            
            if 'processing_time' in metrics_data:
                self.record_processing_time('h01_pipeline', 'total', metrics_data['processing_time'])
            
            if 'segmentation_confidence' in metrics_data:
                self.record_segmentation_accuracy('h01_ffn_v2', metrics_data['segmentation_confidence'])
            
            if 'proofreading_quality' in metrics_data:
                self.record_proofreading_request(True, 'completed')
            
            if 'region_size_gb' in metrics_data:
                # Create a custom gauge for region size
                if 'h01_region_size' not in self.metrics:
                    if self.enable_prometheus:
                        self.metrics['h01_region_size'] = Gauge(
                            f'{self.metrics_prefix}_h01_region_size_gb',
                            'H01 region size in GB'
                        )
                    else:
                        self.metrics['h01_region_size'] = StubMetrics(
                            f'{self.metrics_prefix}_h01_region_size_gb',
                            'H01 region size in GB'
                        )
                self.metrics['h01_region_size'].set(metrics_data['region_size_gb'])
            
            # Update system metrics if available
            if self.enable_system_metrics and PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                self.record_cpu_usage(cpu_percent)
                self.record_memory_usage(memory.used)
                
                # Convert to GB for display
                if 'h01_agentic_tracer_cpu_usage_percent' not in self.metrics:
                    if self.enable_prometheus:
                        self.metrics['h01_agentic_tracer_cpu_usage_percent'] = Gauge(
                            f'{self.metrics_prefix}_cpu_usage_percent',
                            'CPU usage percentage'
                        )
                        self.metrics['h01_agentic_tracer_memory_usage_gb'] = Gauge(
                            f'{self.metrics_prefix}_memory_usage_gb',
                            'Memory usage in GB'
                        )
                    else:
                        self.metrics['h01_agentic_tracer_cpu_usage_percent'] = StubMetrics(
                            f'{self.metrics_prefix}_cpu_usage_percent',
                            'CPU usage percentage'
                        )
                        self.metrics['h01_agentic_tracer_memory_usage_gb'] = StubMetrics(
                            f'{self.metrics_prefix}_memory_usage_gb',
                            'Memory usage in GB'
                        )
                
                self.metrics['h01_agentic_tracer_cpu_usage_percent'].set(cpu_percent)
                self.metrics['h01_agentic_tracer_memory_usage_gb'].set(memory.used / (1024**3))
            
            logger.debug(f"Updated metrics with data: {list(metrics_data.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    # Custom collectors
    def add_custom_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Add a custom metric collector."""
        self.custom_collectors[name] = collector_func
        logger.info(f"Added custom collector: {name}")
    
    def remove_custom_collector(self, name: str):
        """Remove a custom metric collector."""
        if name in self.custom_collectors:
            del self.custom_collectors[name]
            logger.info(f"Removed custom collector: {name}")
    
    # Performance analysis
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter data
        recent_requests = [
            req for req in self.performance_data['requests']
            if req['timestamp'] > cutoff_time
        ]
        recent_errors = [
            err for err in self.performance_data['errors']
            if err['timestamp'] > cutoff_time
        ]
        recent_processing = [
            proc for proc in self.performance_data['processing_times']
            if proc['timestamp'] > cutoff_time
        ]
        
        # Calculate statistics
        total_requests = len(recent_requests)
        total_errors = len(recent_errors)
        
        if recent_requests:
            avg_duration = sum(req['duration'] for req in recent_requests) / total_requests
            max_duration = max(req['duration'] for req in recent_requests)
            min_duration = min(req['duration'] for req in recent_requests)
        else:
            avg_duration = max_duration = min_duration = 0.0
        
        if recent_processing:
            avg_processing = sum(proc['duration'] for proc in recent_processing) / len(recent_processing)
        else:
            avg_processing = 0.0
        
        # Error rate
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'period_hours': hours,
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate_percent': error_rate,
            'avg_request_duration': avg_duration,
            'max_request_duration': max_duration,
            'min_request_duration': min_duration,
            'avg_processing_time': avg_processing,
            'requests_per_hour': total_requests / hours if hours > 0 else 0.0
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        if format == 'json':
            return json.dumps(self.get_performance_summary(), indent=2)
        elif format == 'prometheus' and self.enable_prometheus:
            return generate_latest(REGISTRY).decode('utf-8')
        else:
            return "Unsupported format"
    
    def get_metrics_data(self) -> Dict[str, Any]:
        """Get all current metrics data."""
        data = {}
        
        for name, metric in self.metrics.items():
            if hasattr(metric, 'values'):  # Stub metrics
                data[name] = [
                    {
                        'value': mv.value,
                        'timestamp': mv.timestamp.isoformat(),
                        'labels': mv.labels
                    }
                    for mv in metric.values[-100:]  # Last 100 values
                ]
            else:  # Prometheus metrics
                # For Prometheus metrics, we can't easily extract current values
                # without scraping the metrics endpoint
                data[name] = {'type': 'prometheus_metric'}
        
        return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get telemetry system statistics."""
        return {
            'port': self.port,
            'enable_prometheus': self.enable_prometheus,
            'enable_system_metrics': self.enable_system_metrics,
            'metrics_prefix': self.metrics_prefix,
            'is_monitoring': self.is_monitoring,
            'monitoring_interval': self.monitoring_interval,
            'custom_collectors': len(self.custom_collectors),
            'performance_data': {
                'requests_count': len(self.performance_data['requests']),
                'errors_count': len(self.performance_data['errors']),
                'processing_times_count': len(self.performance_data['processing_times'])
            }
        }
    
    def cleanup(self):
        """Clean up telemetry resources."""
        self.stop_system_monitoring()
        logger.info("Telemetry system cleaned up")

# Global telemetry instance
_telemetry_instance = None

def get_telemetry() -> TelemetrySystem:
    """Get the global telemetry instance."""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = TelemetrySystem()
    return _telemetry_instance

def start_metrics_server(port: int = 8000) -> bool:
    """
    Start the Prometheus metrics HTTP server on the specified port.
    
    Args:
        port: Port to start server on
        
    Returns:
        True if successful, False otherwise
    """
    telemetry = get_telemetry()
    return telemetry.start_metrics_server(port)

# Convenience functions for common metrics
def record_request(method: str, endpoint: str, status: int, duration: float, size: int = 0):
    """Record a request using the global telemetry instance."""
    telemetry = get_telemetry()
    telemetry.record_request(method, endpoint, status, duration, size)

def record_error(error_type: str, component: str):
    """Record an error using the global telemetry instance."""
    telemetry = get_telemetry()
    telemetry.record_error(error_type, component)

def record_processing_time(component: str, operation: str, duration: float):
    """Record processing time using the global telemetry instance."""
    telemetry = get_telemetry()
    telemetry.record_processing_time(component, operation, duration) 