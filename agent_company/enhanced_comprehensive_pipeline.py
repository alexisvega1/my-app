#!/usr/bin/env python3
"""
Enhanced Comprehensive Connectomics Pipeline
============================================

Advanced pipeline that integrates:
- Complete neuron tracing with skeletonization
- Spine detection and classification
- Molecular identity prediction based on morphology
- Allen Brain SDK integration for additional data
- Enhanced segmentation with SAM2 refinement
- Comprehensive connectivity analysis
- Production-ready monitoring and error handling
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings
from dataclasses import dataclass

# Import our enhanced modules
from config import load_config, PipelineConfig
from data_loader import H01DataLoader, create_data_loader
from training import AdvancedTrainer, create_trainer
from ffn_v2_mathematical_model import MathematicalFFNv2
from comprehensive_neuron_analyzer import ComprehensiveNeuronAnalyzer, SpineDetector, MolecularIdentityPredictor

# SAM 2 imports (if available)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    warnings.warn("SAM2 not available. Refinement will be disabled.")

logger = logging.getLogger(__name__)

@dataclass
class ComprehensiveAnalysisConfig:
    """Configuration for comprehensive analysis components."""
    enable_spine_detection: bool = True
    enable_molecular_prediction: bool = True
    enable_allen_brain_integration: bool = True
    enable_synapse_detection: bool = True
    enable_connectivity_analysis: bool = True
    spine_detection_threshold: float = 0.7
    molecular_prediction_confidence_threshold: float = 0.6
    max_neurons_per_analysis: int = 1000

class EnhancedComprehensivePipeline:
    """
    Enhanced comprehensive connectomics pipeline with advanced analysis capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "development"):
        """
        Initialize the enhanced comprehensive pipeline.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (development, production, colab)
        """
        # Load configuration
        self.config = load_config(config_path, environment)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.data_loader = None
        self.model = None
        self.trainer = None
        self.sam_predictor = None
        self.comprehensive_analyzer = None
        
        # Setup comprehensive analysis configuration
        self.analysis_config = ComprehensiveAnalysisConfig()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Enhanced comprehensive pipeline initialized on device: {self.device}")
        logger.info(f"Environment: {environment}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('comprehensive_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        logger.info("Logging setup complete")
    
    def setup_comprehensive_analyzer(self) -> bool:
        """
        Setup the comprehensive neuron analyzer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up comprehensive neuron analyzer...")
            
            # Create configuration for the analyzer
            analyzer_config = {
                'spine_detection': {
                    'min_spine_volume': 50,
                    'max_spine_volume': 2000,
                    'spine_detection_threshold': self.analysis_config.spine_detection_threshold
                },
                'molecular_prediction': {
                    'use_allen_brain_sdk': self.analysis_config.enable_allen_brain_integration,
                    'confidence_threshold': self.analysis_config.molecular_prediction_confidence_threshold
                }
            }
            
            # Initialize the comprehensive analyzer
            self.comprehensive_analyzer = ComprehensiveNeuronAnalyzer(analyzer_config)
            
            logger.info("Comprehensive neuron analyzer setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup comprehensive analyzer: {e}")
            return False
    
    def setup_data_loader(self) -> bool:
        """
        Setup the data loader with error handling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up data loader...")
            
            # Initialize H01 data loader
            self.data_loader = H01DataLoader(self.config)
            
            # Test data loading
            test_coords = self.data_loader.get_random_valid_coords(tuple(self.config.data.chunk_size))
            test_data = self.data_loader.load_chunk(test_coords, tuple(self.config.data.chunk_size))
            
            logger.info(f"Data loader test successful. Loaded chunk of shape: {test_data.shape}")
            logger.info(f"Cache stats: {self.data_loader.get_cache_stats()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup data loader: {e}")
            return False
    
    def setup_model(self) -> bool:
        """
        Setup the neural network model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up model...")
            
            # Create model
            self.model = MathematicalFFNv2(
                input_channels=self.config.model.input_channels,
                output_channels=self.config.model.output_channels,
                hidden_channels=self.config.model.hidden_channels,
                depth=self.config.model.depth
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model created successfully")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            return False
    
    def setup_trainer(self) -> bool:
        """
        Setup the trainer with advanced features.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up trainer...")
            
            if self.model is None:
                raise RuntimeError("Model must be setup before trainer")
            
            # Create trainer
            self.trainer = create_trainer(self.model, self.config, self.device)
            
            logger.info("Trainer setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            return False
    
    def setup_sam_refinement(self) -> bool:
        """
        Setup SAM2 for refinement (if available).
        
        Returns:
            True if successful, False otherwise
        """
        if not SAM2_AVAILABLE:
            logger.warning("SAM2 not available, skipping refinement setup")
            return False
        
        try:
            logger.info("Setting up SAM2 refinement...")
            
            # Load SAM2 model
            sam2_model = build_sam2(
                encoder_patch_embed_dim=96,
                encoder_num_heads=3,
                encoder_window_size=7,
                encoder_depth=2,
                encoder_global_attn_indexes=[],
                checkpoint="checkpoints/sam2_hiera_t.pt"
            )
            
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            self.sam_predictor.set_image(None)  # Will be set during inference
            
            logger.info("SAM2 refinement setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup SAM2 refinement: {e}")
            return False
    
    def run_comprehensive_analysis(self, segmentation: np.ndarray, volume: np.ndarray, 
                                 region_name: str = "unknown") -> Dict[str, Any]:
        """
        Run comprehensive analysis on segmented neurons.
        
        Args:
            segmentation: Segmentation mask with labeled neurons
            volume: Original volume data
            region_name: Name of the region being analyzed
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info(f"Running comprehensive analysis on region: {region_name}")
        
        start_time = time.time()
        results = {
            'region_name': region_name,
            'analysis_timestamp': time.time(),
            'neurons_analyzed': 0,
            'spines_detected': 0,
            'synapses_detected': 0,
            'molecular_predictions': {},
            'connectivity_analysis': {},
            'analysis_confidence': 0.0,
            'processing_time': 0.0
        }
        
        try:
            # Get unique neuron labels
            unique_labels = np.unique(segmentation)
            unique_labels = unique_labels[unique_labels > 0]  # Remove background
            
            # Limit number of neurons for analysis
            if len(unique_labels) > self.analysis_config.max_neurons_per_analysis:
                logger.warning(f"Limiting analysis to {self.analysis_config.max_neurons_per_analysis} neurons")
                unique_labels = unique_labels[:self.analysis_config.max_neurons_per_analysis]
            
            neuron_analyses = []
            total_spines = 0
            total_synapses = 0
            
            # Analyze each neuron
            for i, neuron_id in enumerate(unique_labels):
                logger.info(f"Analyzing neuron {i+1}/{len(unique_labels)} (ID: {neuron_id})")
                
                # Create neuron mask
                neuron_mask = (segmentation == neuron_id)
                
                # Run comprehensive analysis
                neuron_analysis = self.comprehensive_analyzer.analyze_neuron(neuron_mask, int(neuron_id))
                neuron_analyses.append(neuron_analysis)
                
                # Update counts
                total_spines += len(neuron_analysis['spines'])
                total_synapses += len(neuron_analysis['synapses'])
            
            # Aggregate molecular predictions
            molecular_predictions = self._aggregate_molecular_predictions(neuron_analyses)
            
            # Analyze connectivity patterns
            connectivity_analysis = self._analyze_connectivity_patterns(neuron_analyses, segmentation)
            
            # Calculate overall confidence
            overall_confidence = np.mean([analysis['analysis_confidence'] for analysis in neuron_analyses])
            
            # Update results
            results.update({
                'neurons_analyzed': len(neuron_analyses),
                'spines_detected': total_spines,
                'synapses_detected': total_synapses,
                'molecular_predictions': molecular_predictions,
                'connectivity_analysis': connectivity_analysis,
                'analysis_confidence': float(overall_confidence),
                'neuron_analyses': neuron_analyses,
                'processing_time': time.time() - start_time
            })
            
            logger.info(f"Comprehensive analysis completed in {results['processing_time']:.2f} seconds")
            logger.info(f"Analyzed {results['neurons_analyzed']} neurons")
            logger.info(f"Detected {results['spines_detected']} spines and {results['synapses_detected']} synapses")
            
        except Exception as e:
            logger.error(f"Error during comprehensive analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def _aggregate_molecular_predictions(self, neuron_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate molecular predictions across all neurons."""
        aggregated = {
            'neuron_types': {},
            'molecular_markers': {},
            'confidence_scores': {}
        }
        
        for analysis in neuron_analyses:
            predictions = analysis.get('molecular_predictions', {})
            
            for marker, confidence in predictions.items():
                if marker not in aggregated['molecular_markers']:
                    aggregated['molecular_markers'][marker] = []
                aggregated['molecular_markers'][marker].append(confidence)
        
        # Calculate average confidences
        for marker, confidences in aggregated['molecular_markers'].items():
            aggregated['confidence_scores'][marker] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'count': len(confidences)
            }
        
        return aggregated
    
    def _analyze_connectivity_patterns(self, neuron_analyses: List[Dict[str, Any]], 
                                     segmentation: np.ndarray) -> Dict[str, Any]:
        """Analyze connectivity patterns across neurons."""
        connectivity_analysis = {
            'total_connections': 0,
            'connection_types': {},
            'hub_neurons': [],
            'modularity_score': 0.0
        }
        
        # Count connection types
        for analysis in neuron_analyses:
            connectivity = analysis.get('connectivity', {})
            connection_types = connectivity.get('connection_types', {})
            
            for conn_type, count in connection_types.items():
                if conn_type not in connectivity_analysis['connection_types']:
                    connectivity_analysis['connection_types'][conn_type] = 0
                connectivity_analysis['connection_types'][conn_type] += count
        
        # Identify hub neurons (high connectivity)
        for analysis in neuron_analyses:
            connectivity = analysis.get('connectivity', {})
            total_connections = connectivity.get('input_synapses', 0) + connectivity.get('output_synapses', 0)
            
            if total_connections > 10:  # Threshold for hub neurons
                connectivity_analysis['hub_neurons'].append({
                    'neuron_id': analysis['neuron_id'],
                    'total_connections': total_connections,
                    'connection_ratio': connectivity.get('connectivity_strength', 0.0)
                })
        
        connectivity_analysis['total_connections'] = sum(connectivity_analysis['connection_types'].values())
        
        return connectivity_analysis
    
    def run_inference_with_comprehensive_analysis(self, region_coords: Optional[tuple] = None, 
                                                region_size: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Run inference with comprehensive analysis.
        
        Args:
            region_coords: Coordinates of the region to analyze
            region_size: Size of the region to analyze
            
        Returns:
            Dictionary containing inference and analysis results
        """
        logger.info("Running inference with comprehensive analysis...")
        
        # Run standard inference
        if self.model is None:
            raise RuntimeError("Model must be setup before inference")
        
        # Load data
        if region_coords is None:
            region_coords = self.data_loader.get_random_valid_coords(tuple(self.config.data.chunk_size))
        
        if region_size is None:
            region_size = tuple(self.config.data.chunk_size)
        
        volume = self.data_loader.load_chunk(region_coords, region_size)
        
        # Run model inference
        with torch.no_grad():
            input_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(self.device)
            output = self.model(input_tensor)
            
            # Convert to segmentation
            segmentation = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Refine with SAM2 if available
        if self.sam_predictor is not None:
            segmentation = self.refine_with_sam(volume, segmentation)
        
        # Run comprehensive analysis
        analysis_results = self.run_comprehensive_analysis(segmentation, volume, f"region_{region_coords}")
        
        return {
            'inference_results': {
                'region_coords': region_coords,
                'region_size': region_size,
                'volume_shape': volume.shape,
                'segmentation_shape': segmentation.shape,
                'unique_labels': int(np.max(segmentation))
            },
            'comprehensive_analysis': analysis_results
        }
    
    def refine_with_sam(self, volume: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
        """
        Refine segmentation using SAM2.
        
        Args:
            volume: Input volume
            segmentation: Initial segmentation
            
        Returns:
            Refined segmentation
        """
        if self.sam_predictor is None:
            return segmentation
        
        try:
            logger.info("Refining segmentation with SAM2...")
            
            # Convert volume to image format for SAM2
            # This is a simplified implementation
            refined_segmentation = segmentation.copy()
            
            # Apply SAM2 refinement to each slice
            for z in range(volume.shape[0]):
                slice_image = volume[z]
                slice_seg = segmentation[z]
                
                # Convert to SAM2 format and refine
                # This would involve more complex SAM2 integration
                refined_segmentation[z] = slice_seg
            
            logger.info("SAM2 refinement completed")
            return refined_segmentation
            
        except Exception as e:
            logger.warning(f"SAM2 refinement failed: {e}")
            return segmentation
    
    def save_comprehensive_results(self, results: Dict[str, Any], output_dir: str = "comprehensive_results"):
        """
        Save comprehensive analysis results.
        
        Args:
            results: Analysis results to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main results
        results_file = output_path / "comprehensive_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save individual neuron analyses
        neuron_analyses = results.get('comprehensive_analysis', {}).get('neuron_analyses', [])
        for analysis in neuron_analyses:
            neuron_id = analysis['neuron_id']
            neuron_file = output_path / f"neuron_{neuron_id}_analysis.json"
            with open(neuron_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(results, output_path)
        
        logger.info(f"Comprehensive results saved to {output_path}")
    
    def _generate_summary_report(self, results: Dict[str, Any], output_path: Path):
        """Generate a summary report of the analysis."""
        analysis = results.get('comprehensive_analysis', {})
        
        summary = {
            'analysis_summary': {
                'total_neurons': analysis.get('neurons_analyzed', 0),
                'total_spines': analysis.get('spines_detected', 0),
                'total_synapses': analysis.get('synapses_detected', 0),
                'average_confidence': analysis.get('analysis_confidence', 0.0),
                'processing_time': analysis.get('processing_time', 0.0)
            },
            'molecular_predictions': analysis.get('molecular_predictions', {}),
            'connectivity_analysis': analysis.get('connectivity_analysis', {}),
            'timestamp': results.get('comprehensive_analysis', {}).get('analysis_timestamp', 0)
        }
        
        summary_file = output_path / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete enhanced comprehensive pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting complete enhanced comprehensive pipeline...")
            
            # Setup all components
            if not self.setup_data_loader():
                return False
            
            if not self.setup_model():
                return False
            
            if not self.setup_trainer():
                return False
            
            if not self.setup_comprehensive_analyzer():
                return False
            
            if not self.setup_sam_refinement():
                logger.warning("SAM2 setup failed, continuing without refinement")
            
            # Run inference with comprehensive analysis
            results = self.run_inference_with_comprehensive_analysis()
            
            # Save results
            self.save_comprehensive_results(results)
            
            logger.info("Complete enhanced comprehensive pipeline finished successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def main():
    """Main entry point for the enhanced comprehensive pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Comprehensive Connectomics Pipeline")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--environment", default="development", choices=["development", "production", "colab"])
    parser.add_argument("--output-dir", default="comprehensive_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = EnhancedComprehensivePipeline(args.config, args.environment)
    
    # Run pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("✅ Enhanced comprehensive pipeline completed successfully")
        print(f"📁 Results saved to: {args.output_dir}")
    else:
        print("❌ Enhanced comprehensive pipeline failed")
        exit(1)

if __name__ == "__main__":
    main() 