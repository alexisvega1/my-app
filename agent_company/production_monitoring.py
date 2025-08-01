#!/usr/bin/env python3
"""
Production Monitoring and Observability System
=============================================
Comprehensive monitoring for exabyte-scale connectomics processing with
real-time metrics, alerting, and performance optimization.
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import psutil
import torch
from collections import defaultdict, deque
import queue
import signal
import traceback

# Monitoring imports
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, 
        generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import influxdb_client
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for production monitoring."""
    # Metrics collection
    metrics_port: int = 8080
    metrics_interval: int = 30  # seconds
    enable_prometheus: bool = True
    enable_influxdb: bool = True
    enable_elasticsearch: bool = True
    
    # Alerting
    enable_alerts: bool = True
    alert_webhook_url: str = ""
    alert_cooldown: int = 300  # seconds
    
    # Performance tracking
    track_memory: bool = True
    track_gpu: bool = True
    track_throughput: bool = True
    track_latency: bool = True
    track_errors: bool = True
    
    # Storage
    metrics_retention_days: int = 30
    log_retention_days: int = 90
    
    # Thresholds
    memory_threshold: float = 0.9
    gpu_memory_threshold: float = 0.95
    error_rate_threshold: float = 0.01
    latency_threshold_ms: float = 1000.0
    throughput_threshold: float = 100.0

class ProductionMetrics:
    """Production metrics collection and management."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.registry = CollectorRegistry()
        
        # Initialize Prometheus metrics
        if PROMETHEUS_AVAILABLE and config.enable_prometheus:
            self._init_prometheus_metrics()
        
        # Initialize InfluxDB client
        if INFLUXDB_AVAILABLE and config.enable_influxdb:
            self._init_influxdb_client()
        
        # Initialize Elasticsearch client
        if ELASTICSEARCH_AVAILABLE and config.enable_elasticsearch:
            self._init_elasticsearch_client()
        
        # Metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=10000))
        self.alert_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_stats = {
            'chunks_processed': 0,
            'chunks_failed': 0,
            'total_processing_time': 0.0,
            'average_latency': 0.0,
            'peak_memory_usage': 0.0,
            'peak_gpu_memory_usage': 0.0
        }
        
        # Start metrics collection
        self.metrics_thread = threading.Thread(target=self._metrics_collector, daemon=True)
        self.metrics_thread.start()
        
        # Start alert monitoring
        if config.enable_alerts:
            self.alert_thread = threading.Thread(target=self._alert_monitor, daemon=True)
            self.alert_thread.start()
        
        logger.info("Production metrics system initialized")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Counters
        self.chunks_processed_total = Counter(
            'connectomics_chunks_processed_total',
            'Total number of chunks processed',
            ['node_id', 'gpu_id', 'status'],
            registry=self.registry
        )
        
        self.chunks_failed_total = Counter(
            'connectomics_chunks_failed_total',
            'Total number of chunks that failed processing',
            ['node_id', 'error_type'],
            registry=self.registry
        )
        
        self.processing_errors_total = Counter(
            'connectomics_processing_errors_total',
            'Total number of processing errors',
            ['node_id', 'error_type'],
            registry=self.registry
        )
        
        # Gauges
        self.memory_usage_gauge = Gauge(
            'connectomics_memory_usage_bytes',
            'Current memory usage in bytes',
            ['node_id'],
            registry=self.registry
        )
        
        self.gpu_memory_usage_gauge = Gauge(
            'connectomics_gpu_memory_usage_bytes',
            'Current GPU memory usage in bytes',
            ['node_id', 'gpu_id'],
            registry=self.registry
        )
        
        self.cpu_usage_gauge = Gauge(
            'connectomics_cpu_usage_percent',
            'Current CPU usage percentage',
            ['node_id'],
            registry=self.registry
        )
        
        self.disk_usage_gauge = Gauge(
            'connectomics_disk_usage_percent',
            'Current disk usage percentage',
            ['node_id', 'mount_point'],
            registry=self.registry
        )
        
        # Histograms
        self.processing_latency_histogram = Histogram(
            'connectomics_processing_latency_seconds',
            'Processing latency in seconds',
            ['node_id', 'chunk_size'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.chunk_size_histogram = Histogram(
            'connectomics_chunk_size_bytes',
            'Chunk size in bytes',
            ['node_id'],
            buckets=[1e6, 1e7, 1e8, 1e9, 1e10, 1e11],
            registry=self.registry
        )
        
        # Summaries
        self.throughput_summary = Summary(
            'connectomics_throughput_chunks_per_second',
            'Processing throughput in chunks per second',
            ['node_id'],
            registry=self.registry
        )
        
        self.error_rate_summary = Summary(
            'connectomics_error_rate',
            'Error rate as percentage of total chunks',
            ['node_id'],
            registry=self.registry
        )
    
    def _init_influxdb_client(self):
        """Initialize InfluxDB client."""
        try:
            self.influx_client = InfluxDBClient(
                url=os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
                token=os.getenv('INFLUXDB_TOKEN', ''),
                org=os.getenv('INFLUXDB_ORG', 'connectomics'),
                bucket=os.getenv('INFLUXDB_BUCKET', 'connectomics_metrics')
            )
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            logger.info("InfluxDB client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize InfluxDB: {e}")
            self.influx_client = None
    
    def _init_elasticsearch_client(self):
        """Initialize Elasticsearch client."""
        try:
            self.es_client = Elasticsearch(
                hosts=[os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')],
                basic_auth=(
                    os.getenv('ELASTICSEARCH_USER', 'elastic'),
                    os.getenv('ELASTICSEARCH_PASSWORD', '')
                )
            )
            logger.info("Elasticsearch client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Elasticsearch: {e}")
            self.es_client = None
    
    def _metrics_collector(self):
        """Background metrics collection thread."""
        while True:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect GPU metrics
                if self.config.track_gpu:
                    self._collect_gpu_metrics()
                
                # Collect performance metrics
                if self.config.track_throughput:
                    self._collect_performance_metrics()
                
                # Send metrics to external systems
                self._send_metrics()
                
                # Store metrics in history
                self._store_metrics_history()
                
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.config.metrics_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        node_id = os.getenv('NODE_ID', 'unknown')
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage_bytes = memory.used
        
        if PROMETHEUS_AVAILABLE:
            self.memory_usage_gauge.labels(node_id=node_id).set(memory_usage_bytes)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if PROMETHEUS_AVAILABLE:
            self.cpu_usage_gauge.labels(node_id=node_id).set(cpu_percent)
        
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        disk_percent = disk_usage.percent
        
        if PROMETHEUS_AVAILABLE:
            self.disk_usage_gauge.labels(node_id=node_id, mount_point='/').set(disk_percent)
        
        # Update performance stats
        self.performance_stats['peak_memory_usage'] = max(
            self.performance_stats['peak_memory_usage'],
            memory_usage_bytes / (1024**3)  # Convert to GB
        )
        
        # Store in history
        timestamp = time.time()
        self.metrics_history['memory_usage'].append((timestamp, memory_usage_bytes))
        self.metrics_history['cpu_usage'].append((timestamp, cpu_percent))
        self.metrics_history['disk_usage'].append((timestamp, disk_percent))
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics."""
        if not torch.cuda.is_available():
            return
        
        node_id = os.getenv('NODE_ID', 'unknown')
        
        for gpu_id in range(torch.cuda.device_count()):
            try:
                # GPU memory usage
                gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id)
                gpu_memory_reserved = torch.cuda.memory_reserved(gpu_id)
                
                if PROMETHEUS_AVAILABLE:
                    self.gpu_memory_usage_gauge.labels(
                        node_id=node_id, gpu_id=str(gpu_id)
                    ).set(gpu_memory_allocated)
                
                # Update performance stats
                self.performance_stats['peak_gpu_memory_usage'] = max(
                    self.performance_stats['peak_gpu_memory_usage'],
                    gpu_memory_allocated / (1024**3)  # Convert to GB
                )
                
                # Store in history
                timestamp = time.time()
                self.metrics_history[f'gpu_memory_{gpu_id}'].append((timestamp, gpu_memory_allocated))
                
            except Exception as e:
                logger.warning(f"Failed to collect GPU {gpu_id} metrics: {e}")
    
    def _collect_performance_metrics(self):
        """Collect performance metrics."""
        # Calculate throughput
        if len(self.metrics_history['chunks_processed']) >= 2:
            recent_chunks = list(self.metrics_history['chunks_processed'])[-10:]
            if len(recent_chunks) >= 2:
                time_diff = recent_chunks[-1][0] - recent_chunks[0][0]
                chunk_diff = recent_chunks[-1][1] - recent_chunks[0][1]
                throughput = chunk_diff / time_diff if time_diff > 0 else 0
                
                node_id = os.getenv('NODE_ID', 'unknown')
                if PROMETHEUS_AVAILABLE:
                    self.throughput_summary.labels(node_id=node_id).observe(throughput)
                
                self.metrics_history['throughput'].append((time.time(), throughput))
    
    def _send_metrics(self):
        """Send metrics to external systems."""
        # Send to InfluxDB
        if self.influx_client is not None:
            self._send_to_influxdb()
        
        # Send to Elasticsearch
        if self.es_client is not None:
            self._send_to_elasticsearch()
    
    def _send_to_influxdb(self):
        """Send metrics to InfluxDB."""
        try:
            points = []
            node_id = os.getenv('NODE_ID', 'unknown')
            timestamp = datetime.utcnow()
            
            # System metrics
            if self.metrics_history['memory_usage']:
                memory_usage = self.metrics_history['memory_usage'][-1][1]
                points.append(Point("system_metrics")
                    .tag("node_id", node_id)
                    .field("memory_usage_bytes", memory_usage)
                    .time(timestamp))
            
            if self.metrics_history['cpu_usage']:
                cpu_usage = self.metrics_history['cpu_usage'][-1][1]
                points.append(Point("system_metrics")
                    .tag("node_id", node_id)
                    .field("cpu_usage_percent", cpu_usage)
                    .time(timestamp))
            
            # Performance metrics
            if self.metrics_history['throughput']:
                throughput = self.metrics_history['throughput'][-1][1]
                points.append(Point("performance_metrics")
                    .tag("node_id", node_id)
                    .field("throughput_chunks_per_second", throughput)
                    .time(timestamp))
            
            # Write points
            if points:
                self.write_api.write(bucket=os.getenv('INFLUXDB_BUCKET', 'connectomics_metrics'), record=points)
                
        except Exception as e:
            logger.warning(f"Failed to send metrics to InfluxDB: {e}")
    
    def _send_to_elasticsearch(self):
        """Send metrics to Elasticsearch."""
        try:
            node_id = os.getenv('NODE_ID', 'unknown')
            timestamp = datetime.utcnow()
            
            # Create document
            doc = {
                'timestamp': timestamp.isoformat(),
                'node_id': node_id,
                'metrics': {}
            }
            
            # Add recent metrics
            for metric_name, history in self.metrics_history.items():
                if history:
                    doc['metrics'][metric_name] = history[-1][1]
            
            # Add performance stats
            doc['performance_stats'] = self.performance_stats
            
            # Index document
            self.es_client.index(
                index=f'connectomics-metrics-{timestamp.strftime("%Y.%m.%d")}',
                document=doc
            )
            
        except Exception as e:
            logger.warning(f"Failed to send metrics to Elasticsearch: {e}")
    
    def _store_metrics_history(self):
        """Store metrics in local history."""
        timestamp = time.time()
        
        # Store current performance stats
        self.metrics_history['performance_stats'].append((timestamp, self.performance_stats.copy()))
    
    def _alert_monitor(self):
        """Background alert monitoring thread."""
        while True:
            try:
                # Check alert conditions
                alerts = self._check_alert_conditions()
                
                # Send alerts
                for alert in alerts:
                    self._send_alert(alert)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                time.sleep(60)
    
    def _check_alert_conditions(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        node_id = os.getenv('NODE_ID', 'unknown')
        timestamp = time.time()
        
        # Check memory usage
        if self.metrics_history['memory_usage']:
            memory_usage = self.metrics_history['memory_usage'][-1][1]
            memory_percent = memory_usage / psutil.virtual_memory().total
            
            if memory_percent > self.config.memory_threshold:
                alerts.append({
                    'type': 'high_memory_usage',
                    'severity': 'warning',
                    'message': f'Memory usage is {memory_percent:.1%}',
                    'value': memory_percent,
                    'threshold': self.config.memory_threshold,
                    'node_id': node_id,
                    'timestamp': timestamp
                })
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_allocated(gpu_id)
                gpu_memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
                gpu_memory_percent = gpu_memory / gpu_memory_total
                
                if gpu_memory_percent > self.config.gpu_memory_threshold:
                    alerts.append({
                        'type': 'high_gpu_memory_usage',
                        'severity': 'warning',
                        'message': f'GPU {gpu_id} memory usage is {gpu_memory_percent:.1%}',
                        'value': gpu_memory_percent,
                        'threshold': self.config.gpu_memory_threshold,
                        'node_id': node_id,
                        'gpu_id': gpu_id,
                        'timestamp': timestamp
                    })
        
        # Check error rate
        total_chunks = self.performance_stats['chunks_processed'] + self.performance_stats['chunks_failed']
        if total_chunks > 0:
            error_rate = self.performance_stats['chunks_failed'] / total_chunks
            
            if error_rate > self.config.error_rate_threshold:
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'critical',
                    'message': f'Error rate is {error_rate:.1%}',
                    'value': error_rate,
                    'threshold': self.config.error_rate_threshold,
                    'node_id': node_id,
                    'timestamp': timestamp
                })
        
        # Check throughput
        if self.metrics_history['throughput']:
            throughput = self.metrics_history['throughput'][-1][1]
            
            if throughput < self.config.throughput_threshold:
                alerts.append({
                    'type': 'low_throughput',
                    'severity': 'warning',
                    'message': f'Throughput is {throughput:.1f} chunks/second',
                    'value': throughput,
                    'threshold': self.config.throughput_threshold,
                    'node_id': node_id,
                    'timestamp': timestamp
                })
        
        return alerts
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to configured endpoints."""
        # Check cooldown
        alert_key = f"{alert['type']}_{alert.get('node_id', 'unknown')}"
        last_alert_time = None
        
        for historical_alert in reversed(self.alert_history):
            if historical_alert.get('alert_key') == alert_key:
                last_alert_time = historical_alert['timestamp']
                break
        
        if last_alert_time and (time.time() - last_alert_time) < self.config.alert_cooldown:
            return  # Skip due to cooldown
        
        # Store alert
        alert['alert_key'] = alert_key
        self.alert_history.append(alert)
        
        # Send to webhook
        if self.config.alert_webhook_url:
            self._send_webhook_alert(alert)
        
        # Log alert
        logger.warning(f"Alert: {alert['message']} (severity: {alert['severity']})")
    
    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """Send alert to webhook endpoint."""
        try:
            import requests
            
            payload = {
                'text': f"[{alert['severity'].upper()}] {alert['message']}",
                'attachments': [{
                    'color': 'red' if alert['severity'] == 'critical' else 'yellow',
                    'fields': [
                        {'title': 'Type', 'value': alert['type'], 'short': True},
                        {'title': 'Node', 'value': alert.get('node_id', 'unknown'), 'short': True},
                        {'title': 'Value', 'value': f"{alert['value']:.3f}", 'short': True},
                        {'title': 'Threshold', 'value': f"{alert['threshold']:.3f}", 'short': True}
                    ],
                    'timestamp': alert['timestamp']
                }]
            }
            
            response = requests.post(
                self.config.alert_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to send webhook alert: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to send webhook alert: {e}")
    
    def record_chunk_processed(self, chunk_size: int, processing_time: float, 
                              node_id: str = None, gpu_id: str = None):
        """Record a successfully processed chunk."""
        if node_id is None:
            node_id = os.getenv('NODE_ID', 'unknown')
        
        # Update counters
        if PROMETHEUS_AVAILABLE:
            self.chunks_processed_total.labels(
                node_id=node_id, gpu_id=gpu_id or 'unknown', status='success'
            ).inc()
        
        # Update histograms
        if PROMETHEUS_AVAILABLE:
            self.processing_latency_histogram.labels(
                node_id=node_id, chunk_size='large' if chunk_size > 1e9 else 'medium'
            ).observe(processing_time)
            
            self.chunk_size_histogram.labels(node_id=node_id).observe(chunk_size)
        
        # Update performance stats
        self.performance_stats['chunks_processed'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        
        # Update average latency
        total_chunks = self.performance_stats['chunks_processed']
        current_avg = self.performance_stats['average_latency']
        self.performance_stats['average_latency'] = (
            (current_avg * (total_chunks - 1) + processing_time) / total_chunks
        )
        
        # Store in history
        timestamp = time.time()
        self.metrics_history['chunks_processed'].append((timestamp, total_chunks))
        self.metrics_history['processing_latency'].append((timestamp, processing_time))
    
    def record_chunk_failed(self, error_type: str, node_id: str = None):
        """Record a failed chunk."""
        if node_id is None:
            node_id = os.getenv('NODE_ID', 'unknown')
        
        # Update counters
        if PROMETHEUS_AVAILABLE:
            self.chunks_failed_total.labels(node_id=node_id, error_type=error_type).inc()
            self.processing_errors_total.labels(node_id=node_id, error_type=error_type).inc()
        
        # Update performance stats
        self.performance_stats['chunks_failed'] += 1
        
        # Store in history
        timestamp = time.time()
        self.metrics_history['chunks_failed'].append((timestamp, self.performance_stats['chunks_failed']))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {
            'performance_stats': self.performance_stats.copy(),
            'current_metrics': {},
            'alert_count': len(self.alert_history)
        }
        
        # Add current metrics
        for metric_name, history in self.metrics_history.items():
            if history:
                summary['current_metrics'][metric_name] = history[-1][1]
        
        return summary
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry)
        else:
            return "# Prometheus client not available"

class ProductionHealthChecker:
    """Production health checking system."""
    
    def __init__(self, metrics: ProductionMetrics):
        self.metrics = metrics
        self.health_status = {
            'overall': 'healthy',
            'components': {},
            'last_check': time.time()
        }
        
        # Start health checking
        self.health_thread = threading.Thread(target=self._health_checker, daemon=True)
        self.health_thread.start()
        
        logger.info("Production health checker initialized")
    
    def _health_checker(self):
        """Background health checking thread."""
        while True:
            try:
                # Check system health
                self._check_system_health()
                
                # Check application health
                self._check_application_health()
                
                # Update overall health
                self._update_overall_health()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(30)
    
    def _check_system_health(self):
        """Check system-level health."""
        # Memory health
        memory = psutil.virtual_memory()
        memory_percent = memory.percent / 100
        
        if memory_percent > 0.95:
            self.health_status['components']['memory'] = 'critical'
        elif memory_percent > 0.85:
            self.health_status['components']['memory'] = 'warning'
        else:
            self.health_status['components']['memory'] = 'healthy'
        
        # CPU health
        cpu_percent = psutil.cpu_percent(interval=1) / 100
        
        if cpu_percent > 0.95:
            self.health_status['components']['cpu'] = 'critical'
        elif cpu_percent > 0.85:
            self.health_status['components']['cpu'] = 'warning'
        else:
            self.health_status['components']['cpu'] = 'healthy'
        
        # Disk health
        disk_usage = psutil.disk_usage('/')
        disk_percent = disk_usage.percent / 100
        
        if disk_percent > 0.95:
            self.health_status['components']['disk'] = 'critical'
        elif disk_percent > 0.85:
            self.health_status['components']['disk'] = 'warning'
        else:
            self.health_status['components']['disk'] = 'healthy'
    
    def _check_application_health(self):
        """Check application-level health."""
        # Error rate health
        total_chunks = (self.metrics.performance_stats['chunks_processed'] + 
                       self.metrics.performance_stats['chunks_failed'])
        
        if total_chunks > 0:
            error_rate = self.metrics.performance_stats['chunks_failed'] / total_chunks
            
            if error_rate > 0.05:
                self.health_status['components']['error_rate'] = 'critical'
            elif error_rate > 0.01:
                self.health_status['components']['error_rate'] = 'warning'
            else:
                self.health_status['components']['error_rate'] = 'healthy'
        else:
            self.health_status['components']['error_rate'] = 'unknown'
        
        # Throughput health
        if self.metrics.metrics_history['throughput']:
            throughput = self.metrics.metrics_history['throughput'][-1][1]
            
            if throughput < 10:
                self.health_status['components']['throughput'] = 'critical'
            elif throughput < 50:
                self.health_status['components']['throughput'] = 'warning'
            else:
                self.health_status['components']['throughput'] = 'healthy'
        else:
            self.health_status['components']['throughput'] = 'unknown'
    
    def _update_overall_health(self):
        """Update overall health status."""
        # Count health statuses
        status_counts = defaultdict(int)
        for status in self.health_status['components'].values():
            status_counts[status] += 1
        
        # Determine overall health
        if status_counts['critical'] > 0:
            self.health_status['overall'] = 'critical'
        elif status_counts['warning'] > 0:
            self.health_status['overall'] = 'warning'
        else:
            self.health_status['overall'] = 'healthy'
        
        self.health_status['last_check'] = time.time()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health_status.copy()

# Example usage
def test_production_monitoring():
    """Test the production monitoring system."""
    # Configuration
    config = MonitoringConfig(
        enable_prometheus=True,
        enable_influxdb=False,  # Set to True if InfluxDB is available
        enable_elasticsearch=False,  # Set to True if Elasticsearch is available
        enable_alerts=True,
        alert_webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    )
    
    # Initialize metrics
    metrics = ProductionMetrics(config)
    
    # Initialize health checker
    health_checker = ProductionHealthChecker(metrics)
    
    # Simulate some processing
    for i in range(100):
        # Simulate successful chunk processing
        chunk_size = np.random.randint(1e6, 1e9)
        processing_time = np.random.uniform(0.1, 5.0)
        metrics.record_chunk_processed(chunk_size, processing_time)
        
        # Simulate occasional failures
        if np.random.random() < 0.05:  # 5% failure rate
            metrics.record_chunk_failed("timeout")
        
        time.sleep(0.1)
    
    # Get metrics summary
    summary = metrics.get_metrics_summary()
    print(f"Metrics summary: {json.dumps(summary, indent=2)}")
    
    # Get health status
    health_status = health_checker.get_health_status()
    print(f"Health status: {json.dumps(health_status, indent=2)}")
    
    # Get Prometheus metrics
    prometheus_metrics = metrics.get_prometheus_metrics()
    print(f"Prometheus metrics:\n{prometheus_metrics}")
    
    return metrics, health_checker

if __name__ == "__main__":
    test_production_monitoring() 