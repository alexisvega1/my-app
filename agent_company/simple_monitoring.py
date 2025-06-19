#!/usr/bin/env python3
"""
Simple Real-time Monitoring for Agentic Tracer
==============================================
Displays real-time metrics from the Prometheus endpoint without requiring Docker.
"""

import time
import requests
import json
import os
import sys
from datetime import datetime
import threading
from typing import Dict, Any, Optional
import argparse

class SimpleMonitor:
    """Simple real-time monitoring display."""
    
    def __init__(self, prometheus_url: str = "http://localhost:8000/metrics"):
        self.prometheus_url = prometheus_url
        self.running = False
        self.metrics_history = []
        self.max_history = 100
        
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.running = True
        print("🔍 Starting real-time monitoring...")
        print("📊 Prometheus URL:", self.prometheus_url)
        print("=" * 80)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            self.running = False
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self._fetch_metrics()
                if metrics:
                    self._display_metrics(metrics)
                    self._update_history(metrics)
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(10)
    
    def _fetch_metrics(self) -> Optional[Dict[str, Any]]:
        """Fetch metrics from Prometheus endpoint."""
        try:
            response = requests.get(self.prometheus_url, timeout=5)
            if response.status_code == 200:
                return self._parse_metrics(response.text)
            else:
                print(f"⚠️  HTTP {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Connection error: {e}")
            return None
    
    def _parse_metrics(self, metrics_text: str) -> Dict[str, Any]:
        """Parse Prometheus metrics text into structured data."""
        metrics = {}
        
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            # Parse metric line
            if '{' in line:
                # Metric with labels
                metric_name = line.split('{')[0]
                value_part = line.split('}')[-1].strip()
            else:
                # Simple metric
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0]
                    value_part = parts[1]
                else:
                    continue
            
            try:
                value = float(value_part)
                metrics[metric_name] = value
            except ValueError:
                continue
        
        return metrics
    
    def _display_metrics(self, metrics: Dict[str, Any]):
        """Display current metrics."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Processing metrics
        if 'h01_agentic_tracer_volumes_processed' in metrics:
            print(f"📦 Volumes Processed: {metrics['h01_agentic_tracer_volumes_processed']}")
        
        if 'h01_agentic_tracer_processing_time_seconds' in metrics:
            print(f"⏱️  Processing Time: {metrics['h01_agentic_tracer_processing_time_seconds']:.2f}s")
        
        if 'h01_agentic_tracer_segmentation_confidence' in metrics:
            print(f"🎯 Segmentation Confidence: {metrics['h01_agentic_tracer_segmentation_confidence']:.3f}")
        
        if 'h01_agentic_tracer_proofreading_quality' in metrics:
            print(f"✏️  Proofreading Quality: {metrics['h01_agentic_tracer_proofreading_quality']:.3f}")
        
        if 'h01_agentic_tracer_h01_region_size_gb' in metrics:
            print(f"💾 Region Size: {metrics['h01_agentic_tracer_h01_region_size_gb']:.2f} GB")
        
        # System metrics
        if 'h01_agentic_tracer_cpu_usage_percent' in metrics:
            print(f"🖥️  CPU Usage: {metrics['h01_agentic_tracer_cpu_usage_percent']:.1f}%")
        
        if 'h01_agentic_tracer_memory_usage_gb' in metrics:
            print(f"🧠 Memory Usage: {metrics['h01_agentic_tracer_memory_usage_gb']:.2f} GB")
        
        if 'h01_agentic_tracer_gpu_memory_usage_gb' in metrics:
            print(f"🎮 GPU Memory: {metrics['h01_agentic_tracer_gpu_memory_usage_gb']:.2f} GB")
        
        # Error metrics
        if 'h01_agentic_tracer_errors_total' in metrics:
            print(f"❌ Total Errors: {metrics['h01_agentic_tracer_errors_total']}")
        
        # Throughput metrics
        if 'h01_agentic_tracer_processing_throughput_volumes_per_hour' in metrics:
            print(f"🚀 Throughput: {metrics['h01_agentic_tracer_processing_throughput_volumes_per_hour']:.2f} volumes/hour")
        
        print("=" * 80)
        
        # Show recent history
        if self.metrics_history:
            print("📈 Recent Activity:")
            for i, (timestamp, hist_metrics) in enumerate(self.metrics_history[-5:]):
                if 'h01_agentic_tracer_volumes_processed' in hist_metrics:
                    print(f"  {timestamp.strftime('%H:%M:%S')}: {hist_metrics['h01_agentic_tracer_volumes_processed']} volumes")
        
        print("\n💡 Press Ctrl+C to stop monitoring")
    
    def _update_history(self, metrics: Dict[str, Any]):
        """Update metrics history."""
        self.metrics_history.append((datetime.now(), metrics))
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1][1]
        return {
            'latest_metrics': latest,
            'history_length': len(self.metrics_history),
            'monitoring_active': self.running
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Real-time Monitoring for Agentic Tracer")
    parser.add_argument("--url", default="http://localhost:8000/metrics", 
                       help="Prometheus metrics URL")
    parser.add_argument("--interval", type=int, default=5,
                       help="Update interval in seconds")
    
    args = parser.parse_args()
    
    monitor = SimpleMonitor(args.url)
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
    finally:
        summary = monitor.get_summary()
        if summary:
            print(f"\n📊 Monitoring Summary:")
            print(f"  - History points: {summary['history_length']}")
            print(f"  - Latest metrics: {len(summary['latest_metrics'])} available")

if __name__ == "__main__":
    main() 