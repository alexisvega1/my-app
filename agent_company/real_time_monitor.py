#!/usr/bin/env python3
"""
Real-Time Monitor for H01 Production Batch Processing
====================================================
Live monitoring dashboard for batch processing with:
- Real-time progress tracking
- Live log monitoring
- Performance metrics
- Results summary generation
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psutil
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class BatchProgressTracker:
    """Track batch processing progress in real-time."""
    
    def __init__(self, output_dir: str = "h01_production_batch"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Progress tracking
        self.regions_to_process = []
        self.completed_regions = []
        self.failed_regions = []
        self.current_processing = []
        
        # Performance metrics
        self.start_time = None
        self.performance_stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'processing_times': []
        }
        
        # Results aggregation
        self.aggregated_results = {
            'total_neurons': 0,
            'total_synapses': 0,
            'total_circuits': 0,
            'synapse_types': Counter(),
            'neuron_types': Counter(),
            'motif_types': Counter(),
            'region_stats': {}
        }
        
        logger.info(f"Batch progress tracker initialized for: {self.output_dir}")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.start_time = datetime.now()
        logger.info(f"Started monitoring at {self.start_time}")
        
        # Start performance monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_thread.start()
        
        # Start file system monitoring
        self._setup_file_monitoring()
    
    def _monitor_performance(self):
        """Monitor system performance metrics."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.performance_stats['cpu_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': cpu_percent
                })
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.performance_stats['memory_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': memory.percent,
                    'used_gb': memory.used / (1024**3),
                    'total_gb': memory.total / (1024**3)
                })
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.performance_stats['disk_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': (disk.used / disk.total) * 100,
                    'used_gb': disk.used / (1024**3),
                    'total_gb': disk.total / (1024**3)
                })
                
                # Keep only last 1000 measurements
                for key in self.performance_stats:
                    if len(self.performance_stats[key]) > 1000:
                        self.performance_stats[key] = self.performance_stats[key][-1000:]
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(10)
    
    def _setup_file_monitoring(self):
        """Set up file system monitoring for new results."""
        event_handler = BatchFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.output_dir), recursive=True)
        observer.start()
        
        logger.info("File system monitoring started")
    
    def update_progress(self, region_name: str, status: str, details: Dict = None):
        """Update progress for a region."""
        if status == 'started':
            self.current_processing.append(region_name)
            logger.info(f"Started processing: {region_name}")
        
        elif status == 'completed':
            if region_name in self.current_processing:
                self.current_processing.remove(region_name)
            self.completed_regions.append({
                'region_name': region_name,
                'completion_time': datetime.now().isoformat(),
                'details': details or {}
            })
            logger.info(f"Completed processing: {region_name}")
        
        elif status == 'failed':
            if region_name in self.current_processing:
                self.current_processing.remove(region_name)
            self.failed_regions.append({
                'region_name': region_name,
                'failure_time': datetime.now().isoformat(),
                'error': details.get('error', 'Unknown error') if details else 'Unknown error'
            })
            logger.error(f"Failed processing: {region_name}")
    
    def process_new_results(self, region_dir: Path):
        """Process new results from a region directory."""
        try:
            # Check for advanced analysis report
            adv_report_file = region_dir / "advanced_analysis_report.json"
            if adv_report_file.exists():
                with open(adv_report_file, 'r') as f:
                    report = json.load(f)
                
                # Update aggregated results
                summary = report.get('summary', {})
                self.aggregated_results['total_neurons'] += summary.get('total_neurons', 0)
                self.aggregated_results['total_synapses'] += summary.get('total_synapses', 0)
                self.aggregated_results['total_circuits'] += summary.get('total_motifs', 0)
                
                # Update type distributions
                stats = report.get('statistics', {})
                
                # Synapse types
                synapse_stats = stats.get('synapse_analysis', {})
                type_dist = synapse_stats.get('type_distribution', {})
                for synapse_type, count in type_dist.items():
                    self.aggregated_results['synapse_types'][synapse_type] += count
                
                # Neuron types
                morph_stats = stats.get('morphological_analysis', {})
                type_dist = morph_stats.get('type_distribution', {})
                for neuron_type, count in type_dist.items():
                    self.aggregated_results['neuron_types'][neuron_type] += count
                
                # Motif types
                motif_stats = stats.get('motif_analysis', {})
                type_dist = motif_stats.get('statistics', {}).get('type_distribution', {})
                for motif_type, count in type_dist.items():
                    self.aggregated_results['motif_types'][motif_type] += count
                
                # Region-specific stats
                region_name = region_dir.name
                self.aggregated_results['region_stats'][region_name] = {
                    'summary': summary,
                    'key_findings': report.get('key_findings', {}),
                    'processing_time': details.get('processing_time', 0) if 'details' in locals() else 0
                }
                
                logger.info(f"Processed results for: {region_name}")
        
        except Exception as e:
            logger.error(f"Error processing results for {region_dir}: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        elapsed_time = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        # Calculate progress
        total_regions = len(self.completed_regions) + len(self.failed_regions) + len(self.current_processing)
        completed_count = len(self.completed_regions)
        failed_count = len(self.failed_regions)
        current_count = len(self.current_processing)
        
        progress_percent = (completed_count / total_regions * 100) if total_regions > 0 else 0
        
        # Performance summary
        recent_cpu = np.mean([x['value'] for x in self.performance_stats['cpu_usage'][-10:]]) if self.performance_stats['cpu_usage'] else 0
        recent_memory = np.mean([x['value'] for x in self.performance_stats['memory_usage'][-10:]]) if self.performance_stats['memory_usage'] else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': str(elapsed_time),
            'progress': {
                'total_regions': total_regions,
                'completed': completed_count,
                'failed': failed_count,
                'currently_processing': current_count,
                'progress_percent': progress_percent
            },
            'current_processing': self.current_processing,
            'performance': {
                'cpu_percent': recent_cpu,
                'memory_percent': recent_memory,
                'disk_usage_gb': self.performance_stats['disk_usage'][-1]['used_gb'] if self.performance_stats['disk_usage'] else 0
            },
            'results_summary': {
                'total_neurons': self.aggregated_results['total_neurons'],
                'total_synapses': self.aggregated_results['total_synapses'],
                'total_circuits': self.aggregated_results['total_circuits'],
                'synapse_type_distribution': dict(self.aggregated_results['synapse_types']),
                'neuron_type_distribution': dict(self.aggregated_results['neuron_types']),
                'motif_type_distribution': dict(self.aggregated_results['motif_types'])
            }
        }
    
    def generate_live_report(self) -> str:
        """Generate a live HTML report."""
        status = self.get_current_status()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>H01 Batch Processing - Live Monitor</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .status {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .progress-bar {{ background: #bdc3c7; height: 20px; border-radius: 10px; overflow: hidden; }}
                .progress-fill {{ background: #27ae60; height: 100%; width: {status['progress']['progress_percent']}%; transition: width 0.3s; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .processing {{ background: #f39c12; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
                .completed {{ background: #27ae60; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
                .failed {{ background: #e74c3c; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 H01 Connectomics Batch Processing</h1>
                <p>Live Monitor - Last Updated: {status['timestamp']}</p>
            </div>
            
            <div class="status">
                <h2>Processing Status</h2>
                <p>Elapsed Time: {status['elapsed_time']}</p>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <p>Progress: {status['progress']['completed']}/{status['progress']['total_regions']} regions ({status['progress']['progress_percent']:.1f}%)</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>System Performance</h3>
                    <p>CPU: {status['performance']['cpu_percent']:.1f}%</p>
                    <p>Memory: {status['performance']['memory_percent']:.1f}%</p>
                    <p>Disk: {status['performance']['disk_usage_gb']:.1f} GB</p>
                </div>
                
                <div class="metric">
                    <h3>Results Summary</h3>
                    <p>Neurons: {status['results_summary']['total_neurons']:,}</p>
                    <p>Synapses: {status['results_summary']['total_synapses']:,}</p>
                    <p>Circuits: {status['results_summary']['total_circuits']:,}</p>
                </div>
            </div>
            
            <div class="status">
                <h3>Currently Processing</h3>
                {''.join([f'<div class="processing">{region}</div>' for region in status['current_processing']])}
            </div>
            
            <div class="status">
                <h3>Completed Regions</h3>
                {''.join([f'<div class="completed">{region["region_name"]}</div>' for region in self.completed_regions[-5:]])}
            </div>
            
            <div class="status">
                <h3>Failed Regions</h3>
                {''.join([f'<div class="failed">{region["region_name"]}: {region["error"]}</div>' for region in self.failed_regions[-5:]])}
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def save_live_report(self):
        """Save the live report to file."""
        html_content = self.generate_live_report()
        report_file = self.output_dir / "live_monitor.html"
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Live report saved: {report_file}")

class BatchFileHandler(FileSystemEventHandler):
    """Handle file system events for batch processing."""
    
    def __init__(self, tracker: BatchProgressTracker):
        self.tracker = tracker
    
    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.name == "advanced_analysis_report.json":
                region_dir = file_path.parent
                self.tracker.process_new_results(region_dir)
                self.tracker.save_live_report()

def main():
    """Main function for real-time monitoring."""
    print("Real-Time H01 Batch Monitor")
    print("=" * 40)
    
    # Initialize tracker
    tracker = BatchProgressTracker()
    
    # Start monitoring
    tracker.start_monitoring()
    
    print("Monitoring started! Check 'h01_production_batch/live_monitor.html' for live updates.")
    print("Press Ctrl+C to stop monitoring.")
    
    try:
        while True:
            # Update live report every 30 seconds
            tracker.save_live_report()
            
            # Print status to console
            status = tracker.get_current_status()
            print(f"\n[{status['timestamp']}] Progress: {status['progress']['completed']}/{status['progress']['total_regions']} ({status['progress']['progress_percent']:.1f}%)")
            print(f"CPU: {status['performance']['cpu_percent']:.1f}% | Memory: {status['performance']['memory_percent']:.1f}%")
            print(f"Neurons: {status['results_summary']['total_neurons']:,} | Synapses: {status['results_summary']['total_synapses']:,}")
            
            time.sleep(30)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        # Generate final summary
        final_status = tracker.get_current_status()
        print(f"\nFinal Summary:")
        print(f"Total regions processed: {final_status['progress']['completed']}")
        print(f"Failed regions: {final_status['progress']['failed']}")
        print(f"Total neurons: {final_status['results_summary']['total_neurons']:,}")
        print(f"Total synapses: {final_status['results_summary']['total_synapses']:,}")
        print(f"Total circuits: {final_status['results_summary']['total_circuits']:,}")

if __name__ == "__main__":
    main() 