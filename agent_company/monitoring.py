"""
Advanced Monitoring and Observability for Connectomics Pipeline
==============================================================

Provides comprehensive monitoring with Prometheus metrics, Grafana dashboards,
and real-time observability for production deployments.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Monitoring will be limited.")

# GPU monitoring
try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    logging.warning("GPUtil not available. GPU monitoring will be disabled.")

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Collects and manages metrics for the connectomics pipeline.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available. Using basic metrics.")
            self.metrics = {}
            return
        
        # Training metrics
        self.training_loss = Gauge('connectomics_training_loss', 'Training loss value')
        self.validation_loss = Gauge('connectomics_validation_loss', 'Validation loss value')
        self.learning_rate = Gauge('connectomics_learning_rate', 'Current learning rate')
        
        # Performance metrics
        self.inference_time = Histogram('connectomics_inference_time_seconds', 
                                       'Time spent on inference', buckets=[0.1, 0.5, 1.0, 2.0, 5.0])
        self.training_time = Histogram('connectomics_training_time_seconds', 
                                      'Time spent on training', buckets=[1.0, 5.0, 10.0, 30.0, 60.0])
        
        # System metrics
        self.gpu_memory_usage = Gauge('connectomics_gpu_memory_bytes', 'GPU memory usage in bytes')
        self.gpu_utilization = Gauge('connectomics_gpu_utilization_percent', 'GPU utilization percentage')
        self.cpu_usage = Gauge('connectomics_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('connectomics_memory_usage_bytes', 'Memory usage in bytes')
        
        # Counters
        self.inference_requests = Counter('connectomics_inference_requests_total', 'Total inference requests')
        self.training_epochs = Counter('connectomics_training_epochs_total', 'Total training epochs')
        self.errors = Counter('connectomics_errors_total', 'Total errors', ['type'])
        
        # Model metrics
        self.model_parameters = Gauge('connectomics_model_parameters', 'Number of model parameters')
        self.batch_size = Gauge('connectomics_batch_size', 'Current batch size')
        
        logger.info("Metrics collector initialized")
    
    def record_training_metrics(self, train_loss: float, val_loss: float, lr: float):
        """Record training metrics."""
        if PROMETHEUS_AVAILABLE:
            self.training_loss.set(train_loss)
            self.validation_loss.set(val_loss)
            self.learning_rate.set(lr)
        else:
            self.metrics.update({
                'training_loss': train_loss,
                'validation_loss': val_loss,
                'learning_rate': lr
            })
    
    def record_inference_time(self, duration: float):
        """Record inference time."""
        if PROMETHEUS_AVAILABLE:
            self.inference_time.observe(duration)
        else:
            self.metrics['inference_time'] = duration
    
    def record_training_time(self, duration: float):
        """Record training time."""
        if PROMETHEUS_AVAILABLE:
            self.training_time.observe(duration)
        else:
            self.metrics['training_time'] = duration
    
    def record_system_metrics(self):
        """Record system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if PROMETHEUS_AVAILABLE:
            self.cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        if PROMETHEUS_AVAILABLE:
            self.memory_usage.set(memory.used)
        
        # GPU metrics
        if GPU_MONITORING_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    if PROMETHEUS_AVAILABLE:
                        self.gpu_memory_usage.labels(gpu_id=i).set(gpu.memoryUsed * 1024 * 1024)
                        self.gpu_utilization.labels(gpu_id=i).set(gpu.load * 100)
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")
        
        if not PROMETHEUS_AVAILABLE:
            self.metrics.update({
                'cpu_usage': cpu_percent,
                'memory_usage': memory.used,
                'timestamp': datetime.now().isoformat()
            })
    
    def increment_inference_requests(self):
        """Increment inference request counter."""
        if PROMETHEUS_AVAILABLE:
            self.inference_requests.inc()
    
    def increment_training_epochs(self):
        """Increment training epoch counter."""
        if PROMETHEUS_AVAILABLE:
            self.training_epochs.inc()
    
    def record_error(self, error_type: str):
        """Record error occurrence."""
        if PROMETHEUS_AVAILABLE:
            self.errors.labels(type=error_type).inc()
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest()
        else:
            return json.dumps(self.metrics, indent=2)


class PerformanceMonitor:
    """
    Monitors performance and provides insights.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.start_time = time.time()
        self.performance_data = []
        
    def record_performance(self, metric: str, value: float, metadata: Optional[Dict] = None):
        """Record performance metric."""
        data_point = {
            'metric': metric,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.performance_data.append(data_point)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_data:
            return {}
        
        metrics = {}
        for data_point in self.performance_data:
            metric_name = data_point['metric']
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(data_point['value'])
        
        summary = {}
        for metric_name, values in metrics.items():
            summary[metric_name] = {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1]
            }
        
        return summary
    
    def save_performance_data(self, filepath: str):
        """Save performance data to file."""
        with open(filepath, 'w') as f:
            json.dump(self.performance_data, f, indent=2)
        logger.info(f"Performance data saved to: {filepath}")


class HealthChecker:
    """
    Performs health checks on the pipeline components.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks = {}
    
    def register_check(self, name: str, check_func):
        """Register a health check function."""
        self.checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = {
                    'status': 'healthy',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                results[name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return results


class MonitoringDashboard:
    """
    Provides monitoring dashboard functionality.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 performance_monitor: PerformanceMonitor,
                 health_checker: HealthChecker):
        """Initialize monitoring dashboard."""
        self.metrics_collector = metrics_collector
        self.performance_monitor = performance_monitor
        self.health_checker = health_checker
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        # Record current system metrics
        self.metrics_collector.record_system_metrics()
        
        # Get health check results
        health_results = self.health_checker.run_health_checks()
        
        # Get performance summary
        performance_summary = self.performance_monitor.get_performance_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health': health_results,
            'performance': performance_summary,
            'system': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
    
    def generate_grafana_dashboard(self, output_path: str):
        """Generate Grafana dashboard configuration."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Connectomics Pipeline Dashboard",
                "tags": ["connectomics", "pipeline"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Training Loss",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "connectomics_training_loss",
                                "legendFormat": "Training Loss"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Inference Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(connectomics_inference_time_seconds_sum[5m])",
                                "legendFormat": "Inference Rate"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "GPU Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "connectomics_gpu_memory_bytes",
                                "legendFormat": "GPU Memory"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "System Resources",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "connectomics_cpu_usage_percent",
                                "legendFormat": "CPU Usage"
                            },
                            {
                                "expr": "connectomics_memory_usage_bytes",
                                "legendFormat": "Memory Usage"
                            }
                        ]
                    }
                ]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        logger.info(f"Grafana dashboard configuration saved to: {output_path}")


def create_monitoring_system() -> tuple:
    """
    Create a complete monitoring system.
    
    Returns:
        Tuple of (MetricsCollector, PerformanceMonitor, HealthChecker, MonitoringDashboard)
    """
    metrics_collector = MetricsCollector()
    performance_monitor = PerformanceMonitor()
    health_checker = HealthChecker()
    dashboard = MonitoringDashboard(metrics_collector, performance_monitor, health_checker)
    
    return metrics_collector, performance_monitor, health_checker, dashboard


# Example health check functions
def check_gpu_availability():
    """Check if GPU is available."""
    import torch
    return {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

def check_memory_usage():
    """Check memory usage."""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total,
        'available': memory.available,
        'percent': memory.percent
    }

def check_disk_space():
    """Check disk space."""
    disk = psutil.disk_usage('/')
    return {
        'total': disk.total,
        'free': disk.free,
        'percent': disk.percent
    } 