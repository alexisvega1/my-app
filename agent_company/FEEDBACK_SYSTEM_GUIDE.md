# Human Feedback Reinforcement Learning System Guide

## 🎯 Overview

This guide covers the comprehensive human feedback RL system we've built for your neuron tracing pipeline. The system enables human-in-the-loop feedback collection, uncertainty-based intervention, and continuous learning for your connectomics pipeline.

## 🏗️ System Architecture

### Core Components

1. **`FeedbackCollector`** - Collects and stores human feedback
2. **`UncertaintyEstimator`** - Estimates model uncertainty and triggers interventions
3. **`HumanInTheLoopCallback`** - Manages human intervention requests
4. **`HumanFeedbackRLTracer`** - Full tracing pipeline with feedback integration

### Key Features

- ✅ **Uncertainty-Based Intervention**: Only request human input when model is uncertain
- ✅ **Multiple Autonomy Levels**: Fully autonomous, semi-autonomous, or manual review
- ✅ **Structured Feedback Logging**: JSONL format with timestamps and metadata
- ✅ **Custom Callbacks**: Extensible interface for different decision types
- ✅ **Web Interface**: Flask-based UI for human annotators
- ✅ **Training Data Preparation**: Ready-to-use data for model retraining

## 🚀 Quick Start

### 1. Basic Integration

```python
from human_feedback_rl import (
    FeedbackCollector, UncertaintyEstimator, HumanInTheLoopCallback,
    AutonomyLevel, FeedbackType, TracingFeedback
)

# Initialize components
feedback_collector = FeedbackCollector("my_feedback")
uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.7)
human_callback = HumanInTheLoopCallback(AutonomyLevel.SEMI_AUTONOMOUS)

# Use in your pipeline
for neuron_id in range(1, max_neuron_id + 1):
    uncertainty = uncertainty_estimator.estimate_uncertainty(model_output)
    if uncertainty_estimator.should_intervene(uncertainty):
        human_decision = human_callback.request_intervention(...)
        feedback_collector.add_feedback(...)
```

### 2. Web Interface

```bash
# Start the web interface
python web_feedback_interface.py

# Open browser to: http://localhost:5000
# Use setup page to initialize system
# Process demo interventions
# Submit feedback through web UI
```

### 3. Enhanced H01 Pipeline

```bash
# Run enhanced pipeline with feedback
python enhanced_h01_pipeline.py

# This will:
# - Process H01 regions with uncertainty estimation
# - Request human intervention when uncertain
# - Collect feedback for model improvement
# - Save results and feedback data
```

## 📊 Uncertainty Thresholds

### Recommended Settings

| Use Case | Threshold | Description |
|----------|-----------|-------------|
| **High Quality** | 0.5 | More human interventions, higher quality |
| **Balanced** | 0.7 | Good balance of automation and oversight |
| **High Throughput** | 0.8 | Fewer interventions, faster processing |
| **Research** | 0.6 | Good for collecting training data |

### Adjusting Thresholds

```python
# For high-quality annotation
uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.5)

# For high-throughput processing
uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.8)

# For research/data collection
uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.6)
```

## 🎨 Custom Callbacks

### Creating Custom Interfaces

```python
def my_custom_trace_callback(request):
    """Custom callback for your specific interface."""
    print(f"Neuron {request['neuron_id']} needs review")
    print(f"Uncertainty: {request['uncertainty_score']:.3f}")
    
    # Your custom logic here
    decision = your_interface.get_decision()
    return decision

# Register the callback
human_callback.register_callback("trace_continuation", my_custom_trace_callback)
```

### Decision Types

- **`trace_continuation`**: Continue or stop tracing
- **`branch_detection`**: Detect neuron branches
- **`synapse_detection`**: Identify synapses
- **`segmentation`**: Validate neuron segmentation

## 📈 Feedback Analysis

### Getting Statistics

```python
# Get feedback statistics
stats = feedback_collector.get_feedback_stats()
print(f"Total feedback: {stats['total_feedback']}")
print(f"Distribution: {stats['feedback_distribution']}")

# Get uncertainty statistics
uncertainty_stats = uncertainty_estimator.get_uncertainty_stats()
print(f"Mean uncertainty: {uncertainty_stats['mean']:.3f}")
```

### Training Data Preparation

```python
# Get data for model training
training_data = feedback_collector.get_training_data(min_feedback_count=100)

if training_data:
    print(f"Training data available: {len(training_data['states'])} samples")
    # Use for model retraining
else:
    print("Need more feedback data")
```

## 🌐 Web Interface Features

### Dashboard
- Real-time feedback statistics
- Uncertainty metrics
- Processing status
- Quick actions

### Intervention Interface
- 3D visualization of neuron data
- Agent suggestions with uncertainty scores
- Human decision input forms
- Feedback type selection
- Reasoning text fields

### API Endpoints
- `/api/intervention/<id>` - Get intervention details
- `/api/feedback` - Submit human feedback
- `/api/stats` - Get current statistics
- `/api/initialize` - Initialize system

## 🔧 Integration Examples

### 1. Add to Existing Pipeline

```python
# In your production_pipeline.py
from human_feedback_rl import *

# Initialize feedback system
feedback_collector = FeedbackCollector("h01_feedback")
uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold=0.7)
human_callback = HumanInTheLoopCallback(AutonomyLevel.SEMI_AUTONOMOUS)

# Add to processing loop
for region_name, volume in regions.items():
    segmentation = segment_volume(volume)
    
    for neuron_id in range(1, max_neuron_id + 1):
        uncertainty = uncertainty_estimator.estimate_uncertainty(model_output)
        if uncertainty_estimator.should_intervene(uncertainty):
            human_decision = human_callback.request_intervention(...)
            feedback_collector.add_feedback(...)
```

### 2. Batch Processing Integration

```python
# In your batch processor
def process_batch_with_feedback(batch_data):
    feedback_collector = FeedbackCollector("batch_feedback")
    
    for item in batch_data:
        result = process_single_item(item)
        
        if result['uncertainty'] > threshold:
            feedback = TracingFeedback(...)
            feedback_collector.add_feedback(feedback)
    
    # Get training data
    training_data = feedback_collector.get_training_data(min_feedback_count=100)
    if training_data:
        update_model(training_data)
```

### 3. Custom Uncertainty Estimation

```python
# Use your model's uncertainty
def estimate_model_uncertainty(model_output, confidence_scores):
    # Your uncertainty estimation logic
    uncertainty = your_uncertainty_function(model_output, confidence_scores)
    return uncertainty

# Integrate with feedback system
uncertainty = estimate_model_uncertainty(model_output, confidence_scores)
if uncertainty_estimator.should_intervene(uncertainty):
    # Request human intervention
```

## 📁 File Structure

```
agent_company/
├── human_feedback_rl.py              # Core feedback system
├── web_feedback_interface.py         # Web interface
├── enhanced_h01_pipeline.py          # Enhanced H01 pipeline
├── integrate_feedback_system.py      # Integration examples
├── feedback_collection_demo.py       # Demo scripts
├── h01_feedback_integration.py       # H01-specific integration
├── templates/                        # Web interface templates
│   ├── dashboard.html
│   ├── setup.html
│   └── interventions.html
├── enhanced_h01_feedback/            # Feedback data
├── enhanced_h01_results/             # Processing results
└── web_feedback/                     # Web interface data
```

## 🎯 Deployment Options

### 1. CLI Interface (Default)
- Simple command-line prompts
- Good for development and testing
- Easy to integrate with existing scripts

### 2. Web Interface
- User-friendly web UI
- Multiple annotators can work simultaneously
- Real-time statistics and visualizations
- Suitable for production deployment

### 3. Custom Interface
- Integrate with your existing tools
- Use your preferred UI framework
- Customize for your specific workflow

## 📊 Monitoring and Analytics

### Feedback Metrics
- Total feedback collected
- Feedback distribution (accept/reject/correct)
- Average uncertainty scores
- Intervention frequency by decision type

### Performance Metrics
- Processing time per region
- Neurons detected per region
- Human intervention rate
- Model confidence trends

### Quality Metrics
- Human agreement rates
- Feedback consistency
- Uncertainty reduction over time
- Model improvement tracking

## 🔄 Continuous Learning

### Model Updates
1. Collect sufficient feedback (100+ samples)
2. Prepare training data
3. Retrain model with feedback
4. Deploy updated model
5. Monitor performance improvements

### Feedback Loop
```
Model Prediction → Uncertainty Estimation → Human Intervention → 
Feedback Collection → Training Data → Model Update → Improved Predictions
```

## 🚀 Next Steps

### Immediate Actions
1. **Test the system** with your data
2. **Adjust uncertainty thresholds** based on your needs
3. **Customize callbacks** for your interface
4. **Deploy web interface** for human annotators

### Advanced Features
1. **Multi-user support** for team annotation
2. **Advanced visualizations** with 3D rendering
3. **Automated model retraining** pipeline
4. **Quality assurance** workflows
5. **Integration with external tools** (Neurolucida, Vaa3D)

### Production Deployment
1. **Set up monitoring** and alerting
2. **Implement backup** and recovery
3. **Scale for large datasets**
4. **Add authentication** and access control
5. **Optimize performance** for real-time processing

## 📞 Support

For questions or issues:
1. Check the demo scripts for examples
2. Review the integration examples
3. Test with small datasets first
4. Monitor feedback quality and adjust thresholds

## 🎉 Success Metrics

- **Reduced human workload** through smart intervention
- **Improved model accuracy** through feedback learning
- **Faster processing** with uncertainty-based automation
- **Higher quality annotations** through human oversight
- **Continuous improvement** through feedback loops

---

**The system is designed to make your tracing agent partially autonomous while maintaining human oversight where it matters most!** 🧠✨ 