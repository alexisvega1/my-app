#!/usr/bin/env python3
"""
Advanced Analysis Module for H01 Connectomics
=============================================
Advanced analysis capabilities including:
- Synapse type classification
- Circuit motif detection
- Morphological analysis
- Connectivity statistics
- Machine learning integration
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats, spatial
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SynapseFeatures:
    """Features extracted from synapse detection."""
    intensity: float
    volume: float
    shape_factor: float
    distance_to_soma: float
    connectivity_degree: int
    local_density: float
    morphological_type: str = "unknown"

@dataclass
class CircuitMotif:
    """Circuit motif information."""
    motif_type: str
    neurons: List[int]
    synapses: List[Dict]
    strength: float
    confidence: float
    properties: Dict[str, Any]

class AdvancedSynapseAnalyzer:
    """Advanced synapse analysis and classification."""
    
    def __init__(self):
        self.synapse_types = {
            'excitatory': {'intensity_range': (150, 255), 'volume_range': (50, 500)},
            'inhibitory': {'intensity_range': (100, 200), 'volume_range': (30, 300)},
            'modulatory': {'intensity_range': (120, 180), 'volume_range': (40, 400)}
        }
        
    def classify_synapses(self, synapses: List[Dict], volume: np.ndarray, 
                         neuron_traces: Dict) -> Dict[str, Any]:
        """Classify synapses by type and properties."""
        logger.info("Classifying synapses by type and properties...")
        
        classified_synapses = []
        synapse_features = []
        
        for synapse in synapses:
            # Extract features
            features = self._extract_synapse_features(synapse, volume, neuron_traces)
            synapse_features.append(features)
            
            # Classify by type
            synapse_type = self._classify_synapse_type(features)
            
            # Enhanced synapse data
            classified_synapse = {
                **synapse,
                'type': synapse_type,
                'features': {
                    'intensity': features.intensity,
                    'volume': features.volume,
                    'shape_factor': features.shape_factor,
                    'distance_to_soma': features.distance_to_soma,
                    'connectivity_degree': features.connectivity_degree,
                    'local_density': features.local_density,
                    'morphological_type': features.morphological_type
                },
                'confidence': self._calculate_classification_confidence(features)
            }
            
            classified_synapses.append(classified_synapse)
        
        # Cluster analysis for additional insights
        clusters = self._cluster_synapses(synapse_features)
        
        # Statistics
        type_counts = Counter([s['type'] for s in classified_synapses])
        confidence_stats = {
            'mean': np.mean([s['confidence'] for s in classified_synapses]),
            'std': np.std([s['confidence'] for s in classified_synapses]),
            'min': np.min([s['confidence'] for s in classified_synapses]),
            'max': np.max([s['confidence'] for s in classified_synapses])
        }
        
        logger.info(f"Classified {len(classified_synapses)} synapses")
        logger.info(f"Type distribution: {dict(type_counts)}")
        
        return {
            'classified_synapses': classified_synapses,
            'type_distribution': dict(type_counts),
            'confidence_stats': confidence_stats,
            'clusters': clusters,
            'total_synapses': len(classified_synapses)
        }
    
    def _extract_synapse_features(self, synapse: Dict, volume: np.ndarray, 
                                 neuron_traces: Dict) -> SynapseFeatures:
        """Extract comprehensive features from a synapse."""
        z, y, x = synapse['coordinates']
        
        # Local intensity analysis
        local_region = volume[
            max(0, z-3):min(volume.shape[0], z+4),
            max(0, y-3):min(volume.shape[1], y+4),
            max(0, x-3):min(volume.shape[2], x+4)
        ]
        
        intensity = float(np.mean(local_region)) if local_region.size > 0 else 0
        
        # Volume estimation
        volume_estimate = float(np.sum(local_region > intensity * 0.8))
        
        # Shape factor (sphericity)
        binary_region = local_region > intensity * 0.8
        if np.sum(binary_region) > 0:
            from skimage import measure
            props = measure.regionprops(binary_region.astype(int))
            if props:
                shape_factor = props[0].eccentricity
            else:
                shape_factor = 0.5
        else:
            shape_factor = 0.5
        
        # Distance to soma
        neuron_id = synapse['neuron_id']
        if neuron_id in neuron_traces:
            neuron = neuron_traces[neuron_id]
            soma_coords = neuron.coordinates[0] if neuron.coordinates else [0, 0, 0]
            distance_to_soma = np.linalg.norm(np.array(synapse['coordinates']) - np.array(soma_coords))
        else:
            distance_to_soma = 0
        
        # Connectivity degree
        connectivity_degree = len(neuron_traces.get(neuron_id, {}).connectivity) if neuron_id in neuron_traces else 0
        
        # Local density
        local_density = float(np.sum(local_region > 0) / local_region.size) if local_region.size > 0 else 0
        
        # Morphological type (simplified)
        morphological_type = "axo_dendritic" if distance_to_soma > 50 else "somatic"
        
        return SynapseFeatures(
            intensity=intensity,
            volume=volume_estimate,
            shape_factor=shape_factor,
            distance_to_soma=distance_to_soma,
            connectivity_degree=connectivity_degree,
            local_density=local_density,
            morphological_type=morphological_type
        )
    
    def _classify_synapse_type(self, features: SynapseFeatures) -> str:
        """Classify synapse type based on features."""
        # Simple rule-based classification
        if (features.intensity > 200 and features.volume > 200):
            return "excitatory"
        elif (features.intensity < 150 and features.volume < 150):
            return "inhibitory"
        else:
            return "modulatory"
    
    def _calculate_classification_confidence(self, features: SynapseFeatures) -> float:
        """Calculate confidence in synapse classification."""
        # Confidence based on feature consistency
        intensity_conf = min(features.intensity / 255.0, 1.0)
        volume_conf = min(features.volume / 500.0, 1.0)
        shape_conf = 1.0 - features.shape_factor  # Lower eccentricity = higher confidence
        
        return (intensity_conf + volume_conf + shape_conf) / 3.0
    
    def _cluster_synapses(self, synapse_features: List[SynapseFeatures]) -> Dict[str, Any]:
        """Cluster synapses for additional insights."""
        if len(synapse_features) < 2:
            return {'clusters': [], 'silhouette_score': 0}
        
        # Prepare features for clustering
        feature_matrix = np.array([
            [f.intensity, f.volume, f.shape_factor, f.distance_to_soma, f.local_density]
            for f in synapse_features
        ])
        
        # Normalize features
        feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / np.std(feature_matrix, axis=0)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(feature_matrix)
        labels = clustering.labels_
        
        # Calculate silhouette score
        if len(set(labels)) > 1:
            silhouette = silhouette_score(feature_matrix, labels)
        else:
            silhouette = 0
        
        return {
            'clusters': labels.tolist(),
            'silhouette_score': silhouette,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
        }

class CircuitMotifDetector:
    """Detect and analyze neural circuit motifs."""
    
    def __init__(self):
        self.motif_patterns = {
            'feedforward': self._detect_feedforward,
            'recurrent': self._detect_recurrent,
            'divergent': self._detect_divergent,
            'convergent': self._detect_convergent,
            'lateral': self._detect_lateral
        }
    
    def detect_motifs(self, connectivity_graph: Dict, synapses: List[Dict]) -> Dict[str, Any]:
        """Detect various circuit motifs."""
        logger.info("Detecting neural circuit motifs...")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        for neuron_id, neuron_data in connectivity_graph.items():
            G.add_node(neuron_id)
            for connected_id in neuron_data['connections']:
                G.add_edge(neuron_id, connected_id)
        
        motifs = []
        
        # Detect each motif type
        for motif_type, detector_func in self.motif_patterns.items():
            detected_motifs = detector_func(G, synapses)
            motifs.extend(detected_motifs)
        
        # Calculate motif statistics
        motif_stats = self._calculate_motif_statistics(motifs, G)
        
        logger.info(f"Detected {len(motifs)} circuit motifs")
        
        return {
            'motifs': motifs,
            'statistics': motif_stats,
            'total_motifs': len(motifs)
        }
    
    def _detect_feedforward(self, G: nx.DiGraph, synapses: List[Dict]) -> List[CircuitMotif]:
        """Detect feedforward motifs (A -> B -> C)."""
        motifs = []
        
        for node in G.nodes():
            successors = list(G.successors(node))
            for succ in successors:
                succ_successors = list(G.successors(succ))
                for succ_succ in succ_successors:
                    if succ_succ != node:  # Avoid self-loops
                        motif = CircuitMotif(
                            motif_type="feedforward",
                            neurons=[node, succ, succ_succ],
                            synapses=self._get_motif_synapses([node, succ, succ_succ], synapses),
                            strength=1.0,
                            confidence=0.8,
                            properties={'length': 3}
                        )
                        motifs.append(motif)
        
        return motifs
    
    def _detect_recurrent(self, G: nx.DiGraph, synapses: List[Dict]) -> List[CircuitMotif]:
        """Detect recurrent motifs (A -> B -> A)."""
        motifs = []
        
        for node in G.nodes():
            successors = list(G.successors(node))
            for succ in successors:
                if node in G.successors(succ):
                    motif = CircuitMotif(
                        motif_type="recurrent",
                        neurons=[node, succ],
                        synapses=self._get_motif_synapses([node, succ], synapses),
                        strength=1.0,
                        confidence=0.9,
                        properties={'cycle_length': 2}
                    )
                    motifs.append(motif)
        
        return motifs
    
    def _detect_divergent(self, G: nx.DiGraph, synapses: List[Dict]) -> List[CircuitMotif]:
        """Detect divergent motifs (A -> B, A -> C)."""
        motifs = []
        
        for node in G.nodes():
            successors = list(G.successors(node))
            if len(successors) >= 2:
                for i in range(len(successors)):
                    for j in range(i+1, len(successors)):
                        motif = CircuitMotif(
                            motif_type="divergent",
                            neurons=[node, successors[i], successors[j]],
                            synapses=self._get_motif_synapses([node, successors[i], successors[j]], synapses),
                            strength=1.0,
                            confidence=0.7,
                            properties={'fan_out': len(successors)}
                        )
                        motifs.append(motif)
        
        return motifs
    
    def _detect_convergent(self, G: nx.DiGraph, synapses: List[Dict]) -> List[CircuitMotif]:
        """Detect convergent motifs (A -> C, B -> C)."""
        motifs = []
        
        for node in G.nodes():
            predecessors = list(G.predecessors(node))
            if len(predecessors) >= 2:
                for i in range(len(predecessors)):
                    for j in range(i+1, len(predecessors)):
                        motif = CircuitMotif(
                            motif_type="convergent",
                            neurons=[predecessors[i], predecessors[j], node],
                            synapses=self._get_motif_synapses([predecessors[i], predecessors[j], node], synapses),
                            strength=1.0,
                            confidence=0.7,
                            properties={'fan_in': len(predecessors)}
                        )
                        motifs.append(motif)
        
        return motifs
    
    def _detect_lateral(self, G: nx.DiGraph, synapses: List[Dict]) -> List[CircuitMotif]:
        """Detect lateral motifs (A -> B, B -> A)."""
        motifs = []
        
        for node in G.nodes():
            successors = list(G.successors(node))
            for succ in successors:
                if node in G.successors(succ):
                    motif = CircuitMotif(
                        motif_type="lateral",
                        neurons=[node, succ],
                        synapses=self._get_motif_synapses([node, succ], synapses),
                        strength=1.0,
                        confidence=0.8,
                        properties={'bidirectional': True}
                    )
                    motifs.append(motif)
        
        return motifs
    
    def _get_motif_synapses(self, neurons: List[int], synapses: List[Dict]) -> List[Dict]:
        """Get synapses associated with a motif."""
        motif_synapses = []
        for synapse in synapses:
            if synapse['neuron_id'] in neurons:
                motif_synapses.append(synapse)
        return motif_synapses
    
    def _calculate_motif_statistics(self, motifs: List[CircuitMotif], G: nx.DiGraph) -> Dict[str, Any]:
        """Calculate statistics about detected motifs."""
        motif_types = [m.motif_type for m in motifs]
        type_counts = Counter(motif_types)
        
        # Network-level statistics
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        
        # Motif density
        motif_density = len(motifs) / total_nodes if total_nodes > 0 else 0
        
        # Average motif strength
        avg_strength = np.mean([m.strength for m in motifs]) if motifs else 0
        
        return {
            'type_distribution': dict(type_counts),
            'total_motifs': len(motifs),
            'motif_density': motif_density,
            'average_strength': avg_strength,
            'network_nodes': total_nodes,
            'network_edges': total_edges
        }

class MorphologicalAnalyzer:
    """Analyze neuronal morphology and structure."""
    
    def __init__(self):
        self.morphological_types = {
            'pyramidal': {'soma_volume_range': (1000, 5000), 'dendrite_complexity': 'high'},
            'interneuron': {'soma_volume_range': (500, 1500), 'dendrite_complexity': 'medium'},
            'granule': {'soma_volume_range': (200, 800), 'dendrite_complexity': 'low'}
        }
    
    def analyze_morphology(self, neuron_traces: Dict) -> Dict[str, Any]:
        """Analyze neuronal morphology."""
        logger.info("Analyzing neuronal morphology...")
        
        morphological_data = []
        
        for neuron_id, neuron in neuron_traces.items():
            # Basic morphological features
            volume = neuron.volume
            coordinates = neuron.coordinates
            
            # Calculate morphological features
            features = self._calculate_morphological_features(coordinates, volume)
            
            # Classify neuron type
            neuron_type = self._classify_neuron_type(features)
            
            morphological_data.append({
                'neuron_id': neuron_id,
                'type': neuron_type,
                'features': features,
                'volume': volume,
                'coordinates_count': len(coordinates)
            })
        
        # Population statistics
        type_counts = Counter([d['type'] for d in morphological_data])
        volume_stats = {
            'mean': np.mean([d['volume'] for d in morphological_data]),
            'std': np.std([d['volume'] for d in morphological_data]),
            'min': np.min([d['volume'] for d in morphological_data]),
            'max': np.max([d['volume'] for d in morphological_data])
        }
        
        logger.info(f"Analyzed {len(morphological_data)} neurons")
        logger.info(f"Type distribution: {dict(type_counts)}")
        
        return {
            'morphological_data': morphological_data,
            'type_distribution': dict(type_counts),
            'volume_statistics': volume_stats,
            'total_neurons': len(morphological_data)
        }
    
    def _calculate_morphological_features(self, coordinates: List[List[int]], volume: int) -> Dict[str, float]:
        """Calculate morphological features from coordinates."""
        if len(coordinates) < 2:
            return {'complexity': 0, 'branching': 0, 'elongation': 0}
        
        coords_array = np.array(coordinates)
        
        # Complexity (based on coordinate variance)
        complexity = np.std(coords_array, axis=0).mean()
        
        # Branching (based on coordinate clustering)
        if len(coords_array) > 10:
            clustering = DBSCAN(eps=10, min_samples=3).fit(coords_array)
            branching = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        else:
            branching = 1
        
        # Elongation (ratio of max to min extent)
        extents = np.max(coords_array, axis=0) - np.min(coords_array, axis=0)
        elongation = np.max(extents) / (np.min(extents) + 1e-6)
        
        return {
            'complexity': float(complexity),
            'branching': float(branching),
            'elongation': float(elongation)
        }
    
    def _classify_neuron_type(self, features: Dict[str, float]) -> str:
        """Classify neuron type based on morphological features."""
        complexity = features['complexity']
        branching = features['branching']
        elongation = features['elongation']
        
        if complexity > 50 and branching > 3:
            return "pyramidal"
        elif complexity > 20 and branching > 1:
            return "interneuron"
        else:
            return "granule"

class AdvancedAnalysisPipeline:
    """Complete advanced analysis pipeline."""
    
    def __init__(self):
        self.synapse_analyzer = AdvancedSynapseAnalyzer()
        self.motif_detector = CircuitMotifDetector()
        self.morphological_analyzer = MorphologicalAnalyzer()
    
    def run_complete_analysis(self, volume: np.ndarray, neuron_traces: Dict, 
                            synapses: List[Dict], connectivity_graph: Dict) -> Dict[str, Any]:
        """Run complete advanced analysis pipeline."""
        logger.info("Running complete advanced analysis pipeline...")
        
        # 1. Synapse classification
        synapse_analysis = self.synapse_analyzer.classify_synapses(synapses, volume, neuron_traces)
        
        # 2. Circuit motif detection
        motif_analysis = self.motif_detector.detect_motifs(connectivity_graph, synapses)
        
        # 3. Morphological analysis
        morphological_analysis = self.morphological_analyzer.analyze_morphology(neuron_traces)
        
        # 4. Integration analysis
        integration_analysis = self._integrate_analyses(synapse_analysis, motif_analysis, morphological_analysis)
        
        # 5. Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_report(
            synapse_analysis, motif_analysis, morphological_analysis, integration_analysis
        )
        
        logger.info("Advanced analysis pipeline completed")
        
        return {
            'synapse_analysis': synapse_analysis,
            'motif_analysis': motif_analysis,
            'morphological_analysis': morphological_analysis,
            'integration_analysis': integration_analysis,
            'comprehensive_report': comprehensive_report
        }
    
    def _integrate_analyses(self, synapse_analysis: Dict, motif_analysis: Dict, 
                          morphological_analysis: Dict) -> Dict[str, Any]:
        """Integrate different analyses for cross-validation and insights."""
        
        # Cross-correlation between synapse types and circuit motifs
        synapse_types = [s['type'] for s in synapse_analysis['classified_synapses']]
        motif_types = [m.motif_type for m in motif_analysis['motifs']]
        
        # Neuron type vs connectivity patterns
        neuron_types = [d['type'] for d in morphological_analysis['morphological_data']]
        
        integration_stats = {
            'synapse_motif_correlation': self._calculate_correlation(synapse_types, motif_types),
            'neuron_type_distribution': Counter(neuron_types),
            'synapse_type_distribution': Counter(synapse_types),
            'motif_type_distribution': Counter(motif_types)
        }
        
        return integration_stats
    
    def _calculate_correlation(self, list1: List, list2: List) -> float:
        """Calculate correlation between two categorical lists."""
        if len(list1) != len(list2) or len(list1) == 0:
            return 0.0
        
        # Convert to numerical for correlation
        unique1 = list(set(list1))
        unique2 = list(set(list2))
        
        num1 = [unique1.index(x) for x in list1]
        num2 = [unique2.index(x) for x in list2]
        
        return float(np.corrcoef(num1, num2)[0, 1])
    
    def _generate_comprehensive_report(self, synapse_analysis: Dict, motif_analysis: Dict,
                                     morphological_analysis: Dict, integration_analysis: Dict) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        
        report = {
            'summary': {
                'total_neurons': morphological_analysis['total_neurons'],
                'total_synapses': synapse_analysis['total_synapses'],
                'total_motifs': motif_analysis['total_motifs'],
                'analysis_timestamp': str(np.datetime64('now'))
            },
            'key_findings': {
                'dominant_neuron_type': max(morphological_analysis['type_distribution'].items(), key=lambda x: x[1])[0],
                'dominant_synapse_type': max(synapse_analysis['type_distribution'].items(), key=lambda x: x[1])[0],
                'dominant_motif_type': max(motif_analysis['statistics']['type_distribution'].items(), key=lambda x: x[1])[0],
                'average_synapse_confidence': synapse_analysis['confidence_stats']['mean']
            },
            'statistics': {
                'synapse_analysis': synapse_analysis,
                'motif_analysis': motif_analysis,
                'morphological_analysis': morphological_analysis,
                'integration_analysis': integration_analysis
            }
        }
        
        return report

def main():
    """Test the advanced analysis pipeline."""
    print("Advanced Analysis Module Test")
    print("=" * 40)
    
    # Create test data
    volume = np.random.randint(0, 255, (100, 100, 100))
    
    # Mock neuron traces
    neuron_traces = {
        1: type('Neuron', (), {
            'coordinates': [[50, 50, 50], [51, 50, 50], [52, 50, 50]],
            'volume': 1000,
            'connectivity': [2, 3]
        })(),
        2: type('Neuron', (), {
            'coordinates': [[60, 60, 60], [61, 60, 60]],
            'volume': 800,
            'connectivity': [1]
        })()
    }
    
    # Mock synapses
    synapses = [
        {'id': 1, 'neuron_id': 1, 'coordinates': [50, 50, 50], 'intensity': 200},
        {'id': 2, 'neuron_id': 2, 'coordinates': [60, 60, 60], 'intensity': 150}
    ]
    
    # Mock connectivity graph
    connectivity_graph = {
        1: {'connections': [2, 3], 'synapses': [synapses[0]]},
        2: {'connections': [1], 'synapses': [synapses[1]]}
    }
    
    # Run analysis
    pipeline = AdvancedAnalysisPipeline()
    results = pipeline.run_complete_analysis(volume, neuron_traces, synapses, connectivity_graph)
    
    print("Analysis completed successfully!")
    print(f"Processed {results['comprehensive_report']['summary']['total_neurons']} neurons")
    print(f"Detected {results['comprehensive_report']['summary']['total_synapses']} synapses")
    print(f"Found {results['comprehensive_report']['summary']['total_motifs']} circuit motifs")

if __name__ == "__main__":
    main() 