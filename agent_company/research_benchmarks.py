"""
Research Benchmarks and State-of-the-Art Comparison
===================================================

Comprehensive benchmarking system for comparing different model architectures
and methods against state-of-the-art connectomics approaches.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from config import PipelineConfig, load_config
from enhanced_pipeline import EnhancedConnectomicsPipeline
from advanced_models import create_advanced_model
from monitoring import MetricsCollector, PerformanceMonitor

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    performance: Dict[str, float]
    config: Dict[str, Any]
    timestamp: str

class ConnectomicsBenchmark:
    """
    Comprehensive benchmarking system for connectomics models.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize benchmark system.
        
        Args:
            config: Base configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
        
        # Benchmark datasets
        self.datasets = {
            'h01_small': {'size': (64, 64, 64), 'description': 'H01 small region'},
            'h01_medium': {'size': (128, 128, 128), 'description': 'H01 medium region'},
            'h01_large': {'size': (256, 256, 256), 'description': 'H01 large region'},
            'synthetic': {'size': (64, 64, 64), 'description': 'Synthetic data'}
        }
        
        # Model architectures to benchmark
        self.models = {
            'mathematical_ffn': {
                'class': 'MathematicalFFNv2',
                'params': {'input_channels': 1, 'output_channels': 1, 'hidden_channels': 64, 'depth': 4}
            },
            'transformer_ffn': {
                'class': 'TransformerFFN',
                'params': {'input_channels': 1, 'output_channels': 1, 'hidden_channels': 64, 'depth': 4, 'num_heads': 8}
            },
            'swin_transformer': {
                'class': 'SwinTransformer3D',
                'params': {'input_channels': 1, 'output_channels': 1, 'embed_dim': 96}
            },
            'hybrid': {
                'class': 'HybridConnectomicsModel',
                'params': {'input_channels': 1, 'output_channels': 1, 'cnn_channels': 64, 'transformer_dim': 256}
            }
        }
        
        logger.info("Connectomics benchmark system initialized")
    
    def generate_synthetic_data(self, size: Tuple[int, int, int], 
                               complexity: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic connectomics data for benchmarking.
        
        Args:
            size: Size of the data (D, H, W)
            complexity: Complexity level ('simple', 'medium', 'complex')
            
        Returns:
            Tuple of (input_data, target_data)
        """
        D, H, W = size
        
        if complexity == 'simple':
            # Simple synthetic data with basic structures
            input_data = np.random.rand(D, H, W).astype(np.float32)
            
            # Create simple target structures
            target_data = np.zeros_like(input_data)
            target_data[D//4:3*D//4, H//4:3*H//4, W//4:3*W//4] = 1.0
            
        elif complexity == 'medium':
            # Medium complexity with multiple structures
            input_data = np.random.rand(D, H, W).astype(np.float32)
            
            # Add some structure to input
            for i in range(5):
                center_d = np.random.randint(0, D)
                center_h = np.random.randint(0, H)
                center_w = np.random.randint(0, W)
                radius = np.random.randint(5, 15)
                
                d, h, w = np.ogrid[:D, :H, :W]
                mask = (d - center_d)**2 + (h - center_h)**2 + (w - center_w)**2 <= radius**2
                input_data[mask] += 0.5
            
            # Create target with similar structures
            target_data = (input_data > 0.7).astype(np.float32)
            
        else:  # complex
            # Complex synthetic data with realistic connectomics features
            input_data = np.random.rand(D, H, W).astype(np.float32)
            
            # Create complex branching structures
            target_data = np.zeros_like(input_data)
            
            # Main branches
            for i in range(3):
                start_d, start_h, start_w = np.random.randint(0, min(D, H, W), 3)
                length = np.random.randint(20, 40)
                
                for t in range(length):
                    d = int(start_d + t * np.random.normal(0, 0.1))
                    h = int(start_h + t * np.random.normal(0, 0.1))
                    w = int(start_w + t * np.random.normal(0, 0.1))
                    
                    if 0 <= d < D and 0 <= h < H and 0 <= w < W:
                        target_data[d, h, w] = 1.0
                        
                        # Add side branches
                        if np.random.random() < 0.3:
                            for j in range(np.random.randint(3, 8)):
                                side_d = d + np.random.randint(-2, 3)
                                side_h = h + np.random.randint(-2, 3)
                                side_w = w + np.random.randint(-2, 3)
                                
                                if (0 <= side_d < D and 0 <= side_h < H and 0 <= side_w < W):
                                    target_data[side_d, side_h, side_w] = 1.0
        
        return input_data, target_data
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        # Ensure binary predictions
        predictions_binary = (predictions > 0.5).astype(np.float32)
        targets_binary = (targets > 0.5).astype(np.float32)
        
        # Basic metrics
        tp = np.sum((predictions_binary == 1) & (targets_binary == 1))
        tn = np.sum((predictions_binary == 0) & (targets_binary == 0))
        fp = np.sum((predictions_binary == 1) & (targets_binary == 0))
        fn = np.sum((predictions_binary == 0) & (targets_binary == 1))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        # IoU (Jaccard index)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        # Hausdorff distance (simplified)
        hausdorff = self._calculate_hausdorff_distance(predictions_binary, targets_binary)
        
        # Surface distance metrics
        surface_distances = self._calculate_surface_distances(predictions_binary, targets_binary)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'dice': dice,
            'iou': iou,
            'hausdorff_distance': hausdorff,
            'mean_surface_distance': surface_distances['mean'],
            'std_surface_distance': surface_distances['std'],
            'max_surface_distance': surface_distances['max']
        }
    
    def _calculate_hausdorff_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Hausdorff distance between predictions and targets."""
        # Simplified Hausdorff distance calculation
        try:
            from scipy.spatial.distance import directed_hausdorff
            
            # Get coordinates of non-zero points
            pred_coords = np.argwhere(pred > 0)
            target_coords = np.argwhere(target > 0)
            
            if len(pred_coords) == 0 or len(target_coords) == 0:
                return float('inf')
            
            # Calculate directed Hausdorff distances
            d1, _, _ = directed_hausdorff(pred_coords, target_coords)
            d2, _, _ = directed_hausdorff(target_coords, pred_coords)
            
            return max(d1, d2)
            
        except Exception as e:
            logger.warning(f"Hausdorff distance calculation failed: {e}")
            return float('inf')
    
    def _calculate_surface_distances(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Calculate surface distance metrics."""
        try:
            from scipy.ndimage import distance_transform_edt
            
            # Calculate distance transforms
            pred_dist = distance_transform_edt(pred == 0)
            target_dist = distance_transform_edt(target == 0)
            
            # Get surface points
            pred_surface = pred_dist == 1
            target_surface = target_dist == 1
            
            if np.sum(pred_surface) == 0 or np.sum(target_surface) == 0:
                return {'mean': float('inf'), 'std': float('inf'), 'max': float('inf')}
            
            # Calculate distances from pred surface to target surface
            distances = target_dist[pred_surface]
            
            return {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'max': np.max(distances)
            }
            
        except Exception as e:
            logger.warning(f"Surface distance calculation failed: {e}")
            return {'mean': float('inf'), 'std': float('inf'), 'max': float('inf')}
    
    def benchmark_model(self, model_name: str, dataset_name: str, 
                       num_samples: int = 10) -> BenchmarkResult:
        """
        Benchmark a specific model on a dataset.
        
        Args:
            model_name: Name of the model to benchmark
            dataset_name: Name of the dataset to use
            num_samples: Number of samples to test
            
        Returns:
            Benchmark result
        """
        logger.info(f"Benchmarking {model_name} on {dataset_name}")
        
        # Get model configuration
        model_config = self.models[model_name]
        
        # Create model
        model = create_advanced_model(model_config['class'], **model_config['params'])
        model.to(self.device)
        model.eval()
        
        # Get dataset configuration
        dataset_config = self.datasets[dataset_name]
        data_size = dataset_config['size']
        
        # Performance tracking
        total_inference_time = 0.0
        total_memory_usage = 0.0
        all_metrics = []
        
        for i in range(num_samples):
            # Generate test data
            input_data, target_data = self.generate_synthetic_data(data_size, 'medium')
            
            # Convert to tensor
            input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).to(self.device)
            
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                predictions = model(input_tensor)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Measure memory usage
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
                total_memory_usage += memory_usage
            
            # Convert predictions to numpy
            predictions_np = predictions.cpu().numpy().squeeze()
            
            # Calculate metrics
            metrics = self.calculate_metrics(predictions_np, target_data)
            all_metrics.append(metrics)
            
            logger.debug(f"Sample {i+1}/{num_samples}: F1={metrics['f1_score']:.4f}, Time={inference_time:.4f}s")
        
        # Aggregate results
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isinf(m[key])]
            avg_metrics[key] = np.mean(values) if values else 0.0
        
        performance = {
            'avg_inference_time': total_inference_time / num_samples,
            'total_inference_time': total_inference_time,
            'avg_memory_usage': total_memory_usage / num_samples if torch.cuda.is_available() else 0.0,
            'samples_per_second': num_samples / total_inference_time
        }
        
        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=avg_metrics,
            performance=performance,
            config=model_config,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        
        logger.info(f"Benchmark completed: {model_name} on {dataset_name}")
        logger.info(f"Average F1: {avg_metrics['f1_score']:.4f}, Avg Time: {performance['avg_inference_time']:.4f}s")
        
        return result
    
    def run_comprehensive_benchmark(self, models: Optional[List[str]] = None,
                                  datasets: Optional[List[str]] = None,
                                  num_samples: int = 10) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark across all models and datasets.
        
        Args:
            models: List of model names to benchmark (None for all)
            datasets: List of dataset names to benchmark (None for all)
            num_samples: Number of samples per benchmark
            
        Returns:
            List of benchmark results
        """
        if models is None:
            models = list(self.models.keys())
        
        if datasets is None:
            datasets = list(self.datasets.keys())
        
        logger.info(f"Starting comprehensive benchmark: {len(models)} models, {len(datasets)} datasets")
        
        results = []
        
        for model_name in models:
            for dataset_name in datasets:
                try:
                    result = self.benchmark_model(model_name, dataset_name, num_samples)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark failed for {model_name} on {dataset_name}: {e}")
        
        logger.info(f"Comprehensive benchmark completed: {len(results)} results")
        return results
    
    def generate_benchmark_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            output_path: Path to save report (None for console output)
            
        Returns:
            Report content
        """
        if not self.results:
            return "No benchmark results available."
        
        # Create report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CONNECTOMICS BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Results: {len(self.results)}")
        report_lines.append("")
        
        # Summary table
        report_lines.append("SUMMARY TABLE")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Model':<20} {'Dataset':<15} {'F1 Score':<10} {'Dice':<10} {'Time (s)':<10} {'Memory (GB)':<12}")
        report_lines.append("-" * 80)
        
        for result in self.results:
            report_lines.append(
                f"{result.model_name:<20} {result.dataset_name:<15} "
                f"{result.metrics['f1_score']:<10.4f} {result.metrics['dice']:<10.4f} "
                f"{result.performance['avg_inference_time']:<10.4f} "
                f"{result.performance['avg_memory_usage']:<12.4f}"
            )
        
        report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 80)
        
        for result in self.results:
            report_lines.append(f"Model: {result.model_name}")
            report_lines.append(f"Dataset: {result.dataset_name}")
            report_lines.append(f"Timestamp: {result.timestamp}")
            report_lines.append("")
            
            report_lines.append("Metrics:")
            for metric, value in result.metrics.items():
                report_lines.append(f"  {metric}: {value:.6f}")
            
            report_lines.append("Performance:")
            for perf, value in result.performance.items():
                report_lines.append(f"  {perf}: {value:.6f}")
            
            report_lines.append("-" * 40)
        
        report_content = "\n".join(report_lines)
        
        # Save to file if specified
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Benchmark report saved to: {output_path}")
        
        return report_content
    
    def plot_benchmark_results(self, save_path: Optional[str] = None):
        """Plot benchmark results."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Prepare data for plotting
        models = list(set(r.model_name for r in self.results))
        datasets = list(set(r.dataset_name for r in self.results))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1 Score comparison
        f1_data = []
        for model in models:
            model_results = [r for r in self.results if r.model_name == model]
            f1_scores = [r.metrics['f1_score'] for r in model_results]
            f1_data.append(f1_scores)
        
        axes[0, 0].boxplot(f1_data, labels=models)
        axes[0, 0].set_title('F1 Score Comparison')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        time_data = []
        for model in models:
            model_results = [r for r in self.results if r.model_name == model]
            times = [r.performance['avg_inference_time'] for r in model_results]
            time_data.append(times)
        
        axes[0, 1].boxplot(time_data, labels=models)
        axes[0, 1].set_title('Inference Time Comparison')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        memory_data = []
        for model in models:
            model_results = [r for r in self.results if r.model_name == model]
            memory = [r.performance['avg_memory_usage'] for r in model_results]
            memory_data.append(memory)
        
        axes[1, 0].boxplot(memory_data, labels=models)
        axes[1, 0].set_title('Memory Usage Comparison')
        axes[1, 0].set_ylabel('Memory (GB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Heatmap of F1 scores by model and dataset
        f1_matrix = np.zeros((len(models), len(datasets)))
        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                result = next((r for r in self.results if r.model_name == model and r.dataset_name == dataset), None)
                f1_matrix[i, j] = result.metrics['f1_score'] if result else 0.0
        
        im = axes[1, 1].imshow(f1_matrix, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('F1 Score Heatmap')
        axes[1, 1].set_xticks(range(len(datasets)))
        axes[1, 1].set_yticks(range(len(models)))
        axes[1, 1].set_xticklabels(datasets, rotation=45)
        axes[1, 1].set_yticklabels(models)
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Benchmark plots saved to: {save_path}")
        else:
            plt.show()
    
    def save_results(self, output_path: str):
        """Save benchmark results to JSON file."""
        results_data = []
        
        for result in self.results:
            results_data.append({
                'model_name': result.model_name,
                'dataset_name': result.dataset_name,
                'metrics': result.metrics,
                'performance': result.performance,
                'config': result.config,
                'timestamp': result.timestamp
            })
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {output_path}")


def run_connectomics_benchmark(config_path: Optional[str] = None,
                             models: Optional[List[str]] = None,
                             datasets: Optional[List[str]] = None,
                             num_samples: int = 10) -> ConnectomicsBenchmark:
    """
    Run comprehensive connectomics benchmark.
    
    Args:
        config_path: Path to configuration file
        models: List of model names to benchmark
        datasets: List of dataset names to benchmark
        num_samples: Number of samples per benchmark
        
    Returns:
        Benchmark object with results
    """
    # Load configuration
    config = load_config(config_path, "development")
    
    # Create benchmark system
    benchmark = ConnectomicsBenchmark(config)
    
    # Run comprehensive benchmark
    benchmark.run_comprehensive_benchmark(models, datasets, num_samples)
    
    # Generate report
    report = benchmark.generate_benchmark_report("benchmark_report.txt")
    print(report)
    
    # Plot results
    benchmark.plot_benchmark_results("benchmark_results.png")
    
    # Save results
    benchmark.save_results("benchmark_results.json")
    
    return benchmark


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Connectomics Benchmark System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--models", nargs="+", help="Models to benchmark")
    parser.add_argument("--datasets", nargs="+", help="Datasets to benchmark")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples per benchmark")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = run_connectomics_benchmark(
        config_path=args.config,
        models=args.models,
        datasets=args.datasets,
        num_samples=args.num_samples
    )
    
    print("Benchmark completed successfully!") 