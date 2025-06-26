# Enhanced Comprehensive Connectomics Pipeline

## Overview

The Enhanced Comprehensive Connectomics Pipeline is a state-of-the-art system for complete neuron analysis, including:

- **Complete Neuron Tracing**: 3D skeletonization and morphological analysis
- **Spine Detection & Classification**: Automated detection and classification of dendritic spines
- **Molecular Identity Prediction**: Prediction of neuron types based on morphological features
- **Allen Brain SDK Integration**: Access to additional molecular and connectivity data
- **Synapse Detection**: Identification and classification of synaptic connections
- **Connectivity Analysis**: Circuit motif detection and connectivity mapping
- **SAM2 Refinement**: Advanced segmentation refinement using Segment Anything Model 2

## Key Features

### 🧠 Complete Neuron Analysis
- **3D Skeletonization**: Full 3D tracing of neuron morphology
- **Morphological Feature Extraction**: Comprehensive feature analysis
- **Branching Analysis**: Dendritic arborization quantification
- **Volume and Surface Analysis**: Detailed geometric measurements

### 🦴 Spine Detection & Classification
- **Automated Detection**: AI-powered spine detection algorithms
- **Type Classification**: Mushroom, thin, stubby, and filopodia spines
- **Morphological Analysis**: Length, head diameter, neck diameter measurements
- **Confidence Scoring**: Reliability assessment for each detection

### 🧬 Molecular Identity Prediction
- **Morphology-Based Classification**: Predict neuron types from structure
- **Multiple Markers**: Glutamatergic, GABAergic, cholinergic, etc.
- **Machine Learning Models**: Trained classifiers for accurate prediction
- **Allen Brain Integration**: Additional data from Allen Brain Atlas

### 🔗 Connectivity Analysis
- **Synapse Detection**: Automated synapse identification
- **Circuit Motifs**: Detection of common connectivity patterns
- **Hub Neuron Identification**: Find highly connected neurons
- **Connectivity Strength**: Quantify connection strengths

### 🎯 Advanced Segmentation
- **SAM2 Refinement**: State-of-the-art segmentation refinement
- **Multi-scale Analysis**: Analysis at different resolution levels
- **Uncertainty Quantification**: Confidence measures for all predictions
- **Quality Control**: Automated quality assessment

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 100GB+ storage

### Core Installation
```bash
# Clone the repository
git clone <repository-url>
cd agent_company

# Install core dependencies
pip install -r requirements-comprehensive.txt

# Install Allen Brain SDK (optional but recommended)
pip install allensdk

# Install SAM2 (optional)
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

### Configuration
1. Copy the configuration template:
```bash
cp comprehensive_config.yaml my_config.yaml
```

2. Edit `my_config.yaml` to match your environment:
```yaml
environment: "development"  # or "production"
data:
  volume_path: "your_data_path"
  cache_path: "your_cache_path"
```

## Usage

### Basic Usage

#### 1. Setup Pipeline
```python
from enhanced_comprehensive_pipeline import EnhancedComprehensivePipeline

# Initialize pipeline
pipeline = EnhancedComprehensivePipeline(
    config_path="comprehensive_config.yaml",
    environment="development"
)

# Setup components
pipeline.setup_data_loader()
pipeline.setup_model()
pipeline.setup_comprehensive_analyzer()
pipeline.setup_sam_refinement()
```

#### 2. Run Complete Analysis
```python
# Run inference with comprehensive analysis
results = pipeline.run_inference_with_comprehensive_analysis(
    region_coords=(1000, 1000, 1000),
    region_size=(128, 128, 128)
)

# Save results
pipeline.save_comprehensive_results(results, "output_directory")
```

### Advanced Usage

#### Custom Spine Detection
```python
from comprehensive_neuron_analyzer import SpineDetector

# Configure spine detection
config = {
    'min_spine_volume': 50,
    'max_spine_volume': 2000,
    'spine_detection_threshold': 0.7
}

spine_detector = SpineDetector(config)

# Detect spines on a neuron mask
spines = spine_detector.detect_spines(neuron_mask, dendrite_skeleton)

# Analyze spine types
for spine in spines:
    print(f"Spine {spine.id}: {spine.spine_type} (confidence: {spine.confidence:.2f})")
```

#### Molecular Identity Prediction
```python
from comprehensive_neuron_analyzer import MolecularIdentityPredictor

# Create predictor
predictor = MolecularIdentityPredictor({'use_allen_brain_sdk': True})

# Train with custom data
training_data = [
    {
        'soma_volume': 1500,
        'dendritic_length': 2000,
        'spine_density': 0.2,
        'molecular_type': 'pyramidal'
    }
    # ... more training data
]

predictor.train_classifier(training_data)

# Predict molecular identity
predictions = predictor.predict_molecular_identity(neuron_morphology)
print(f"Predicted types: {predictions}")
```

#### Allen Brain SDK Integration
```python
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.api.queries.cell_types_api import CellTypesApi

# Initialize Allen Brain SDK
boc = BrainObservatoryCache(manifest_file='brain_observatory_manifest.json')
cell_types_api = CellTypesApi()

# Query cell data
cells = cell_types_api.list_cells()
print(f"Retrieved {len(cells)} cell records")
```

## Configuration Options

### Spine Detection
```yaml
comprehensive_analysis:
  spine_detection:
    enabled: true
    min_spine_volume: 50
    max_spine_volume: 2000
    spine_detection_threshold: 0.7
    spine_types:
      mushroom:
        min_length: 3.0
        min_head_diameter: 1.5
      thin:
        min_length: 2.0
        max_head_diameter: 1.0
```

### Molecular Prediction
```yaml
comprehensive_analysis:
  molecular_prediction:
    enabled: true
    use_allen_brain_sdk: true
    confidence_threshold: 0.6
    markers:
      - "glutamatergic"
      - "gabaergic"
      - "pyramidal"
      - "interneuron"
```

### Performance Tuning
```yaml
performance:
  max_memory_usage: "16GB"
  max_neurons_per_batch: 100
  max_spines_per_analysis: 1000
  enable_result_caching: true
```

## Output Formats

### JSON Output
```json
{
  "neuron_id": 1,
  "morphology": {
    "soma_volume": 1200.5,
    "dendritic_length": 1800.2,
    "spine_density": 0.15
  },
  "spines": [
    {
      "id": 1,
      "spine_type": "mushroom",
      "volume": 150.0,
      "confidence": 0.85
    }
  ],
  "molecular_predictions": {
    "pyramidal": 0.82,
    "glutamatergic": 0.91
  }
}
```

### SWC Format
Standard SWC format for neuron skeletons:
```
# n T X Y Z R PARENT
1 1 64.0 64.0 64.0 8.0 -1
2 3 64.0 64.0 72.0 3.0 1
```

### HDF5 Format
Hierarchical data format for complex analyses:
```python
import h5py

with h5py.File('analysis_results.h5', 'r') as f:
    neurons = f['neurons']
    spines = f['spines']
    predictions = f['molecular_predictions']
```

## Testing

### Run All Tests
```bash
python test_comprehensive_pipeline.py
```

### Individual Test Components
```bash
# Test spine detection
python -c "from test_comprehensive_pipeline import test_spine_detection; test_spine_detection()"

# Test molecular prediction
python -c "from test_comprehensive_pipeline import test_molecular_prediction; test_molecular_prediction()"

# Test Allen Brain integration
python -c "from test_comprehensive_pipeline import test_allen_brain_integration; test_allen_brain_integration()"
```

## Performance Optimization

### Memory Management
- Use appropriate chunk sizes for your hardware
- Enable result caching for repeated analyses
- Monitor memory usage with the built-in profiler

### GPU Acceleration
- Ensure CUDA is properly installed
- Use mixed precision training for faster inference
- Batch processing for multiple neurons

### Parallel Processing
- Multi-worker data loading
- Parallel spine detection
- Distributed training (for large datasets)

## Troubleshooting

### Common Issues

#### Allen Brain SDK Connection
```python
# Check connection
try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    boc = BrainObservatoryCache()
    print("✅ Allen Brain SDK connected")
except Exception as e:
    print(f"❌ Allen Brain SDK error: {e}")
```

#### Memory Issues
```yaml
# Reduce memory usage
performance:
  max_memory_usage: "8GB"
  max_neurons_per_batch: 50
  max_spines_per_analysis: 500
```

#### SAM2 Not Available
```python
# Graceful fallback
if SAM2_AVAILABLE:
    pipeline.setup_sam_refinement()
else:
    print("⚠️ SAM2 not available, continuing without refinement")
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
pipeline = EnhancedComprehensivePipeline()
pipeline.run_complete_pipeline()
```

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-comprehensive.txt
pip install black flake8 mypy pytest

# Run code formatting
black agent_company/

# Run linting
flake8 agent_company/

# Run type checking
mypy agent_company/

# Run tests
pytest tests/
```

### Adding New Features
1. Create feature branch
2. Implement feature with tests
3. Update documentation
4. Submit pull request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{comprehensive_connectomics_pipeline,
  title={Enhanced Comprehensive Connectomics Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: See the docs/ directory
- **Issues**: Report bugs on GitHub
- **Discussions**: Join our community discussions
- **Email**: contact@your-organization.com

## Acknowledgments

- Allen Brain Institute for the Allen Brain SDK
- Facebook Research for SAM2
- The connectomics community for inspiration and feedback 