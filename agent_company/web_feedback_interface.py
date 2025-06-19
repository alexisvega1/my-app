#!/usr/bin/env python3
"""
Web-Based Human Feedback Interface
=================================
Flask web application for human annotators to provide feedback on neuron tracing decisions.
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import torch
import json
import base64
import io
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import our feedback system
from human_feedback_rl import (
    FeedbackCollector,
    UncertaintyEstimator,
    HumanInTheLoopCallback,
    AutonomyLevel,
    FeedbackType,
    TracingFeedback,
    InterventionPoint
)

app = Flask(__name__)

# Global state for the feedback system
feedback_collector = None
uncertainty_estimator = None
human_callback = None
current_intervention = None

class WebFeedbackCallback:
    """Web-based callback for human feedback."""
    
    def __init__(self, feedback_collector, uncertainty_estimator):
        self.feedback_collector = feedback_collector
        self.uncertainty_estimator = uncertainty_estimator
        self.pending_interventions = {}
        self.intervention_counter = 0
    
    def request_intervention(self, intervention_point):
        """Queue intervention for web interface."""
        self.intervention_counter += 1
        intervention_id = f"intervention_{self.intervention_counter}"
        
        # Store intervention
        self.pending_interventions[intervention_id] = {
            'intervention_point': intervention_point,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        # Return intervention ID for web interface
        return intervention_id
    
    def get_pending_interventions(self):
        """Get all pending interventions."""
        return self.pending_interventions
    
    def submit_feedback(self, intervention_id, human_decision, feedback_type, reasoning=""):
        """Submit human feedback for an intervention."""
        if intervention_id not in self.pending_interventions:
            return False, "Intervention not found"
        
        intervention_data = self.pending_interventions[intervention_id]
        intervention_point = intervention_data['intervention_point']
        
        # Create feedback
        feedback = TracingFeedback(
            timestamp=datetime.now().isoformat(),
            region_name=intervention_point.region_name,
            neuron_id=intervention_point.neuron_id,
            decision_type=intervention_point.decision_type,
            agent_decision=intervention_point.agent_suggestion,
            human_feedback=FeedbackType(feedback_type),
            human_correction=human_decision if human_decision != intervention_point.agent_suggestion else None,
            uncertainty_score=intervention_point.uncertainty_score,
            confidence_score=intervention_point.confidence_score,
            reasoning=reasoning
        )
        
        # Add to collector
        success = self.feedback_collector.add_feedback(feedback)
        
        if success:
            # Mark as completed
            self.pending_interventions[intervention_id]['status'] = 'completed'
            self.pending_interventions[intervention_id]['human_decision'] = human_decision
            self.pending_interventions[intervention_id]['feedback_type'] = feedback_type
        
        return success, "Feedback submitted successfully"

def create_visualization(volume, segmentation, neuron_id, point=None):
    """Create visualization for web interface."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create neuron mask
    neuron_mask = (segmentation == neuron_id)
    
    if point:
        z, y, x = point
        # Show slices through the point
        axes[0].imshow(volume[z, :, :], cmap='gray')
        axes[0].contour(neuron_mask[z, :, :], colors='red', alpha=0.7)
        axes[0].plot(x, y, 'go', markersize=10, label='Current Point')
        axes[0].set_title(f'Z-slice {z}')
        axes[0].legend()
        
        axes[1].imshow(volume[:, y, :], cmap='gray')
        axes[1].contour(neuron_mask[:, y, :], colors='red', alpha=0.7)
        axes[1].plot(x, z, 'go', markersize=10, label='Current Point')
        axes[1].set_title(f'Y-slice {y}')
        axes[1].legend()
        
        axes[2].imshow(volume[:, :, x], cmap='gray')
        axes[2].contour(neuron_mask[:, :, x], colors='red', alpha=0.7)
        axes[2].plot(y, z, 'go', markersize=10, label='Current Point')
        axes[2].set_title(f'X-slice {x}')
        axes[2].legend()
    else:
        # Show middle slices
        mid_z, mid_y, mid_x = [s//2 for s in volume.shape]
        
        axes[0].imshow(volume[mid_z, :, :], cmap='gray')
        axes[0].contour(neuron_mask[mid_z, :, :], colors='red', alpha=0.7)
        axes[0].set_title(f'Z-slice {mid_z}')
        
        axes[1].imshow(volume[:, mid_y, :], cmap='gray')
        axes[1].contour(neuron_mask[:, mid_y, :], colors='red', alpha=0.7)
        axes[1].set_title(f'Y-slice {mid_y}')
        
        axes[2].imshow(volume[:, :, mid_x], cmap='gray')
        axes[2].contour(neuron_mask[:, :, mid_x], colors='red', alpha=0.7)
        axes[2].set_title(f'X-slice {mid_x}')
    
    plt.tight_layout()
    
    # Convert to base64 for web display
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    """Main dashboard."""
    global feedback_collector, uncertainty_estimator
    
    if feedback_collector is None:
        return render_template('setup.html')
    
    # Get feedback statistics
    feedback_stats = feedback_collector.get_feedback_stats()
    uncertainty_stats = uncertainty_estimator.get_uncertainty_stats()
    
    return render_template('dashboard.html', 
                         feedback_stats=feedback_stats,
                         uncertainty_stats=uncertainty_stats)

@app.route('/interventions')
def interventions():
    """Show pending interventions."""
    global human_callback
    
    if human_callback is None:
        return jsonify({'error': 'System not initialized'})
    
    pending = human_callback.get_pending_interventions()
    return render_template('interventions.html', interventions=pending)

@app.route('/api/intervention/<intervention_id>')
def get_intervention(intervention_id):
    """Get specific intervention details."""
    global human_callback
    
    if human_callback is None:
        return jsonify({'error': 'System not initialized'})
    
    pending = human_callback.get_pending_interventions()
    
    if intervention_id not in pending:
        return jsonify({'error': 'Intervention not found'})
    
    intervention_data = pending[intervention_id]
    intervention_point = intervention_data['intervention_point']
    
    # Create visualization (simplified for demo)
    # In practice, you'd load the actual volume data
    volume = np.random.randint(0, 255, (64, 64, 64))
    segmentation = np.random.randint(0, 5, (64, 64, 64))
    
    visualization = create_visualization(
        volume, segmentation, 
        intervention_point.neuron_id,
        intervention_point.current_state
    )
    
    return jsonify({
        'intervention_id': intervention_id,
        'region_name': intervention_point.region_name,
        'neuron_id': intervention_point.neuron_id,
        'decision_type': intervention_point.decision_type,
        'uncertainty_score': intervention_point.uncertainty_score,
        'confidence_score': intervention_point.confidence_score,
        'agent_suggestion': intervention_point.agent_suggestion,
        'current_state': intervention_point.current_state,
        'context': intervention_point.context,
        'visualization': visualization,
        'timestamp': intervention_data['timestamp']
    })

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit human feedback."""
    global human_callback
    
    if human_callback is None:
        return jsonify({'error': 'System not initialized'})
    
    data = request.json
    intervention_id = data.get('intervention_id')
    human_decision = data.get('human_decision')
    feedback_type = data.get('feedback_type')
    reasoning = data.get('reasoning', '')
    
    success, message = human_callback.submit_feedback(
        intervention_id, human_decision, feedback_type, reasoning
    )
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/stats')
def get_stats():
    """Get current statistics."""
    global feedback_collector, uncertainty_estimator
    
    if feedback_collector is None:
        return jsonify({'error': 'System not initialized'})
    
    feedback_stats = feedback_collector.get_feedback_stats()
    uncertainty_stats = uncertainty_estimator.get_uncertainty_stats()
    
    return jsonify({
        'feedback_stats': feedback_stats,
        'uncertainty_stats': uncertainty_stats
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the feedback system."""
    global feedback_collector, uncertainty_estimator, human_callback
    
    data = request.json
    feedback_dir = data.get('feedback_dir', 'web_feedback')
    uncertainty_threshold = data.get('uncertainty_threshold', 0.7)
    
    # Initialize components
    feedback_collector = FeedbackCollector(feedback_dir)
    uncertainty_estimator = UncertaintyEstimator(uncertainty_threshold)
    human_callback = WebFeedbackCallback(feedback_collector, uncertainty_estimator)
    
    return jsonify({'success': True, 'message': 'System initialized'})

@app.route('/api/process_demo')
def process_demo():
    """Process a demo intervention."""
    global human_callback
    
    if human_callback is None:
        return jsonify({'error': 'System not initialized'})
    
    # Create a demo intervention
    intervention_point = InterventionPoint(
        timestamp=datetime.now().isoformat(),
        region_name="demo_region",
        neuron_id=1,
        decision_type="trace_continuation",
        current_state=(32, 32, 32),
        agent_suggestion={"next_point": (33, 32, 32)},
        uncertainty_score=0.8,
        confidence_score=0.2,
        context={"volume_shape": (64, 64, 64)}
    )
    
    intervention_id = human_callback.request_intervention(intervention_point)
    
    return jsonify({'success': True, 'intervention_id': intervention_id})

# Create templates directory and HTML files
def create_templates():
    """Create HTML templates for the web interface."""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Dashboard template
    dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Neuron Tracing Feedback Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat-card { background: #ecf0f1; padding: 15px; border-radius: 5px; flex: 1; }
        .button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .button:hover { background: #2980b9; }
        .intervention { background: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .visualization { text-align: center; margin: 20px 0; }
        .visualization img { max-width: 100%; height: auto; }
        .feedback-form { margin: 20px 0; }
        .feedback-form input, .feedback-form textarea { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Neuron Tracing Feedback Interface</h1>
        <p>Human-in-the-Loop Feedback System for Connectomics</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>Feedback Statistics</h3>
            <p>Total Feedback: <span id="total-feedback">0</span></p>
            <p>Accept: <span id="accept-count">0</span></p>
            <p>Reject: <span id="reject-count">0</span></p>
            <p>Correct: <span id="correct-count">0</span></p>
        </div>
        <div class="stat-card">
            <h3>Uncertainty Statistics</h3>
            <p>Mean: <span id="uncertainty-mean">0.0</span></p>
            <p>Std: <span id="uncertainty-std">0.0</span></p>
            <p>Max: <span id="uncertainty-max">0.0</span></p>
        </div>
    </div>
    
    <div>
        <button class="button" onclick="processDemo()">Process Demo Intervention</button>
        <button class="button" onclick="loadInterventions()">Load Pending Interventions</button>
        <button class="button" onclick="refreshStats()">Refresh Statistics</button>
    </div>
    
    <div id="interventions-container"></div>
    
    <script>
        function refreshStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error(data.error);
                        return;
                    }
                    
                    const feedback = data.feedback_stats;
                    const uncertainty = data.uncertainty_stats;
                    
                    document.getElementById('total-feedback').textContent = feedback.total_feedback || 0;
                    document.getElementById('accept-count').textContent = feedback.feedback_distribution?.accept || 0;
                    document.getElementById('reject-count').textContent = feedback.feedback_distribution?.reject || 0;
                    document.getElementById('correct-count').textContent = feedback.feedback_distribution?.correct || 0;
                    
                    document.getElementById('uncertainty-mean').textContent = uncertainty.mean?.toFixed(3) || '0.000';
                    document.getElementById('uncertainty-std').textContent = uncertainty.std?.toFixed(3) || '0.000';
                    document.getElementById('uncertainty-max').textContent = uncertainty.max?.toFixed(3) || '0.000';
                });
        }
        
        function processDemo() {
            fetch('/api/process_demo')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        loadInterventions();
                    }
                });
        }
        
        function loadInterventions() {
            fetch('/interventions')
                .then(response => response.text())
                .then(html => {
                    document.getElementById('interventions-container').innerHTML = html;
                });
        }
        
        function submitFeedback(interventionId) {
            const decision = document.getElementById('decision-' + interventionId).value;
            const feedbackType = document.getElementById('feedback-type-' + interventionId).value;
            const reasoning = document.getElementById('reasoning-' + interventionId).value;
            
            fetch('/api/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    intervention_id: interventionId,
                    human_decision: decision,
                    feedback_type: feedbackType,
                    reasoning: reasoning
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Feedback submitted successfully!');
                    loadInterventions();
                    refreshStats();
                } else {
                    alert('Error: ' + data.message);
                }
            });
        }
        
        // Load initial data
        refreshStats();
    </script>
</body>
</html>
    """
    
    # Setup template
    setup_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Setup - Neuron Tracing Feedback Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .setup-form { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .setup-form input { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 3px; }
        .button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Setup Feedback System</h1>
        <p>Initialize the human feedback system</p>
    </div>
    
    <div class="setup-form">
        <h3>System Configuration</h3>
        <label>Feedback Directory:</label>
        <input type="text" id="feedback-dir" value="web_feedback">
        
        <label>Uncertainty Threshold:</label>
        <input type="number" id="uncertainty-threshold" value="0.7" step="0.1" min="0" max="1">
        
        <button class="button" onclick="initializeSystem()">Initialize System</button>
    </div>
    
    <script>
        function initializeSystem() {
            const feedbackDir = document.getElementById('feedback-dir').value;
            const uncertaintyThreshold = parseFloat(document.getElementById('uncertainty-threshold').value);
            
            fetch('/api/initialize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    feedback_dir: feedbackDir,
                    uncertainty_threshold: uncertaintyThreshold
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    window.location.href = '/';
                } else {
                    alert('Error: ' + data.message);
                }
            });
        }
    </script>
</body>
</html>
    """
    
    # Interventions template
    interventions_html = """
{% for intervention_id, data in interventions.items() %}
<div class="intervention">
    <h3>Intervention: {{ intervention_id }}</h3>
    <p><strong>Region:</strong> {{ data.intervention_point.region_name }}</p>
    <p><strong>Neuron ID:</strong> {{ data.intervention_point.neuron_id }}</p>
    <p><strong>Decision Type:</strong> {{ data.intervention_point.decision_type }}</p>
    <p><strong>Uncertainty:</strong> {{ "%.3f"|format(data.intervention_point.uncertainty_score) }}</p>
    <p><strong>Status:</strong> {{ data.status }}</p>
    
    {% if data.status == 'pending' %}
    <div class="visualization" id="viz-{{ intervention_id }}">
        <p>Loading visualization...</p>
    </div>
    
    <div class="feedback-form">
        <label>Agent Suggestion:</label>
        <input type="text" value="{{ data.intervention_point.agent_suggestion }}" readonly>
        
        <label>Your Decision:</label>
        <input type="text" id="decision-{{ intervention_id }}" placeholder="Enter your decision">
        
        <label>Feedback Type:</label>
        <select id="feedback-type-{{ intervention_id }}">
            <option value="accept">Accept</option>
            <option value="reject">Reject</option>
            <option value="correct">Correct</option>
            <option value="uncertain">Uncertain</option>
        </select>
        
        <label>Reasoning:</label>
        <textarea id="reasoning-{{ intervention_id }}" placeholder="Explain your decision"></textarea>
        
        <button class="button" onclick="submitFeedback('{{ intervention_id }}')">Submit Feedback</button>
    </div>
    
    <script>
        // Load visualization for this intervention
        fetch('/api/intervention/{{ intervention_id }}')
            .then(response => response.json())
            .then(data => {
                if (data.visualization) {
                    document.getElementById('viz-{{ intervention_id }}').innerHTML = 
                        '<img src="data:image/png;base64,' + data.visualization + '" alt="Visualization">';
                }
            });
    </script>
    {% endif %}
</div>
{% endfor %}
    """
    
    # Write templates
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(dashboard_html)
    
    with open(templates_dir / "setup.html", "w") as f:
        f.write(setup_html)
    
    with open(templates_dir / "interventions.html", "w") as f:
        f.write(interventions_html)

def main():
    """Run the web feedback interface."""
    print("🧠 Web Feedback Interface")
    print("=" * 50)
    
    # Create templates
    create_templates()
    
    print("✅ Templates created")
    print("🌐 Starting web server...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🔧 Use the setup page to initialize the system")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main() 