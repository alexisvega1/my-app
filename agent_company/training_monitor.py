#!/usr/bin/env python3
"""
Real-time Training Monitor Dashboard
===================================
Live monitoring dashboard for H01 training progress.
"""

import os
import sys
import time
import json
import logging
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np

# Optional imports for GUI
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.animation as animation
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI components not available - using console monitoring")

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Real-time training progress monitor."""
    
    def __init__(self, log_file: str = None, model_file: str = None):
        self.log_file = log_file or "train_ffn_v2.log"
        self.model_file = model_file or "best_ffn_v2_model.pt"
        self.monitoring = False
        self.stats = {
            'start_time': None,
            'current_epoch': 0,
            'total_epochs': 20,
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'memory_usage': [],
            'processing_times': [],
            'last_update': None
        }
        
        # Performance tracking
        self.performance_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': []
        }
        
        logger.info("Training monitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.monitoring = True
        self.stats['start_time'] = datetime.now()
        
        # Start monitoring threads
        self.log_monitor_thread = threading.Thread(target=self._monitor_log_file, daemon=True)
        self.system_monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        
        self.log_monitor_thread.start()
        self.system_monitor_thread.start()
        
        logger.info("Training monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        logger.info("Training monitoring stopped")
    
    def _monitor_log_file(self):
        """Monitor training log file for progress updates."""
        last_position = 0
        
        while self.monitoring:
            try:
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                        
                        for line in new_lines:
                            self._parse_log_line(line.strip())
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error monitoring log file: {e}")
                time.sleep(5)
    
    def _parse_log_line(self, line: str):
        """Parse log line for training progress."""
        if not line:
            return
        
        # Parse epoch progress
        if "Epoch" in line and "Train Loss=" in line:
            try:
                # Extract epoch number
                epoch_part = line.split("Epoch ")[1].split("/")[0]
                self.stats['current_epoch'] = int(epoch_part)
                
                # Extract metrics
                if "Train Loss=" in line:
                    train_loss = float(line.split("Train Loss=")[1].split(",")[0])
                    self.stats['train_loss'].append(train_loss)
                
                if "Train Acc=" in line:
                    train_acc = float(line.split("Train Acc=")[1].split(",")[0])
                    self.stats['train_acc'].append(train_acc)
                
                if "Val Loss=" in line:
                    val_loss = float(line.split("Val Loss=")[1].split(",")[0])
                    self.stats['val_loss'].append(val_loss)
                
                if "Val Acc=" in line:
                    val_acc = float(line.split("Val Acc=")[1].split(",")[0])
                    self.stats['val_acc'].append(val_acc)
                
                if "Time=" in line:
                    time_str = line.split("Time=")[1].split("s")[0]
                    self.stats['processing_times'].append(float(time_str))
                
                if "Memory=" in line:
                    memory_str = line.split("Memory=")[1].split("GB")[0]
                    self.stats['memory_usage'].append(float(memory_str))
                
                self.stats['last_update'] = datetime.now()
                
            except Exception as e:
                logger.debug(f"Could not parse log line: {e}")
    
    def _monitor_system(self):
        """Monitor system resources."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.performance_data['cpu_usage'].append({
                    'timestamp': time.time(),
                    'value': cpu_percent
                })
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.performance_data['memory_usage'].append({
                    'timestamp': time.time(),
                    'value': memory.percent
                })
                
                # Keep only last 1000 data points
                for key in self.performance_data:
                    if len(self.performance_data[key]) > 1000:
                        self.performance_data[key] = self.performance_data[key][-1000:]
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system: {e}")
                time.sleep(10)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current training progress summary."""
        if not self.stats['start_time']:
            return {'status': 'not_started'}
        
        elapsed_time = datetime.now() - self.stats['start_time']
        progress_percent = (self.stats['current_epoch'] / self.stats['total_epochs']) * 100
        
        # Estimate remaining time
        if self.stats['processing_times']:
            avg_epoch_time = np.mean(self.stats['processing_times'])
            remaining_epochs = self.stats['total_epochs'] - self.stats['current_epoch']
            estimated_remaining = timedelta(seconds=avg_epoch_time * remaining_epochs)
        else:
            estimated_remaining = timedelta(0)
        
        return {
            'status': 'training',
            'current_epoch': self.stats['current_epoch'],
            'total_epochs': self.stats['total_epochs'],
            'progress_percent': progress_percent,
            'elapsed_time': str(elapsed_time).split('.')[0],
            'estimated_remaining': str(estimated_remaining).split('.')[0],
            'train_loss': self.stats['train_loss'][-1] if self.stats['train_loss'] else None,
            'val_loss': self.stats['val_loss'][-1] if self.stats['val_loss'] else None,
            'train_acc': self.stats['train_acc'][-1] if self.stats['train_acc'] else None,
            'val_acc': self.stats['val_acc'][-1] if self.stats['val_acc'] else None,
            'memory_usage': self.stats['memory_usage'][-1] if self.stats['memory_usage'] else None,
            'last_update': self.stats['last_update'].isoformat() if self.stats['last_update'] else None
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        if not self.performance_data['cpu_usage']:
            return {
                'status': 'no_data',
                'cpu_usage_avg': 0.0,
                'cpu_usage_max': 0.0,
                'memory_usage_avg': 0.0,
                'memory_usage_max': 0.0,
                'system_memory_total': psutil.virtual_memory().total / (1024**3),
                'system_memory_available': psutil.virtual_memory().available / (1024**3)
            }
        
        # Calculate averages over last 10 minutes
        cutoff_time = time.time() - 600  # 10 minutes
        
        recent_cpu = [d['value'] for d in self.performance_data['cpu_usage'] 
                     if d['timestamp'] > cutoff_time]
        recent_memory = [d['value'] for d in self.performance_data['memory_usage'] 
                        if d['timestamp'] > cutoff_time]
        
        return {
            'cpu_usage_avg': np.mean(recent_cpu) if recent_cpu else 0,
            'cpu_usage_max': np.max(recent_cpu) if recent_cpu else 0,
            'memory_usage_avg': np.mean(recent_memory) if recent_memory else 0,
            'memory_usage_max': np.max(recent_memory) if recent_memory else 0,
            'system_memory_total': psutil.virtual_memory().total / (1024**3),
            'system_memory_available': psutil.virtual_memory().available / (1024**3)
        }

class ConsoleMonitor:
    """Console-based monitoring interface."""
    
    def __init__(self, monitor: TrainingMonitor):
        self.monitor = monitor
    
    def start_console_monitoring(self):
        """Start console-based monitoring."""
        print("🚀 H01 Training Monitor Started")
        print("=" * 50)
        
        try:
            while self.monitor.monitoring:
                self._display_progress()
                time.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            self.monitor.stop_monitoring()
    
    def _display_progress(self):
        """Display current progress in console."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        progress = self.monitor.get_progress_summary()
        performance = self.monitor.get_performance_summary()
        
        print("🎯 H01 Training Progress Monitor")
        print("=" * 50)
        
        if progress['status'] == 'not_started':
            print("⏳ Training not started yet...")
            return
        
        # Progress bar
        progress_bar = "█" * int(progress['progress_percent'] / 5) + "░" * (20 - int(progress['progress_percent'] / 5))
        print(f"📊 Progress: [{progress_bar}] {progress['progress_percent']:.1f}%")
        print(f"🔄 Epoch: {progress['current_epoch']}/{progress['total_epochs']}")
        print(f"⏱️  Elapsed: {progress['elapsed_time']}")
        print(f"⏳ Remaining: {progress['estimated_remaining']}")
        
        # Metrics
        print("\n📈 Training Metrics:")
        if progress['train_loss']:
            print(f"   Train Loss: {progress['train_loss']:.4f}")
        if progress['val_loss']:
            print(f"   Val Loss: {progress['val_loss']:.4f}")
        if progress['train_acc']:
            print(f"   Train Acc: {progress['train_acc']:.4f}")
        if progress['val_acc']:
            print(f"   Val Acc: {progress['val_acc']:.4f}")
        
        # System performance
        print("\n💻 System Performance:")
        print(f"   CPU Usage: {performance.get('cpu_usage_avg', 0.0):.1f}% (max: {performance.get('cpu_usage_max', 0.0):.1f}%)")
        print(f"   Memory Usage: {performance.get('memory_usage_avg', 0.0):.1f}% (max: {performance.get('memory_usage_max', 0.0):.1f}%)")
        print(f"   Available Memory: {performance.get('system_memory_available', 0.0):.1f}GB")
        
        if progress['memory_usage']:
            print(f"   Training Memory: {progress['memory_usage']:.2f}GB")
        
        # Last update
        if progress['last_update']:
            last_update = datetime.fromisoformat(progress['last_update'])
            time_since = datetime.now() - last_update
            print(f"\n🕒 Last Update: {time_since.total_seconds():.0f}s ago")
        
        print("\n" + "=" * 50)

def main():
    """Main entry point for training monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="H01 Training Monitor")
    parser.add_argument("--log-file", default="train_ffn_v2.log", help="Training log file")
    parser.add_argument("--model-file", default="best_ffn_v2_model.pt", help="Model checkpoint file")
    parser.add_argument("--console", action="store_true", help="Use console interface")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor
    monitor = TrainingMonitor(args.log_file, args.model_file)
    monitor.start_monitoring()
    
    try:
        console_monitor = ConsoleMonitor(monitor)
        console_monitor.start_console_monitoring()
    
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main() 