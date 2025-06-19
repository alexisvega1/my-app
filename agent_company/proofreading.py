#!/usr/bin/env python3
"""
Uncertainty-Triggered Proofreader
================================
Automatically proofreads segmentation results when uncertainty exceeds threshold.
"""

import os
import logging
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import google.cloud.firestore as firestore
    from google.cloud.firestore import Client as FirestoreClient
    from google.cloud.firestore_v1.base_document import DocumentSnapshot
    from google.auth.exceptions import DefaultCredentialsError
    FIRESTORE_AVAILABLE = True
    logger.info("Google Cloud Firestore available")
except ImportError:
    FIRESTORE_AVAILABLE = False
    logger.warning("Google Cloud Firestore not available - using stub client")

try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - some proofreading features disabled")

@dataclass
class ProofreadingResult:
    """Result of a proofreading operation."""
    original_text: str
    proofread_text: str
    corrections_made: List[Dict[str, Any]]
    confidence_improvement: float
    processing_time: float
    metadata: Dict[str, Any]

class FirestoreClientStub:
    """Stub Firestore client for when Google Cloud is not available."""
    
    def __init__(self, project_id: str = "stub-project"):
        self.project_id = project_id
        self.collections = {}
        logger.info(f"Initialized stub Firestore client for project: {project_id}")
    
    def collection(self, collection_name: str):
        """Get or create a collection."""
        if collection_name not in self.collections:
            self.collections[collection_name] = {}
        return FirestoreCollectionStub(self.collections[collection_name])
    
    def close(self):
        """Close the stub client."""
        logger.info("Stub Firestore client closed")

class FirestoreCollectionStub:
    """Stub collection for testing."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    def document(self, doc_id: str):
        """Get or create a document."""
        if doc_id not in self.data:
            self.data[doc_id] = {}
        return FirestoreDocumentStub(self.data, doc_id)
    
    def add(self, data: Dict[str, Any]) -> Tuple[str, Any]:
        """Add a document with auto-generated ID."""
        doc_id = f"stub_doc_{len(self.data)}"
        self.data[doc_id] = data
        return doc_id, None

class FirestoreDocumentStub:
    """Stub document for testing."""
    
    def __init__(self, data: Dict[str, Any], doc_id: str):
        self.data = data
        self.doc_id = doc_id
    
    def get(self) -> Optional[Dict[str, Any]]:
        """Get document data."""
        return self.data.get(self.doc_id, None)
    
    def set(self, data: Dict[str, Any]) -> None:
        """Set document data."""
        self.data[self.doc_id] = data
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update document data."""
        if self.doc_id in self.data:
            self.data[self.doc_id].update(data)
        else:
            self.data[self.doc_id] = data

class UncertaintyTriggeredProofreader:
    """Proofreader that activates based on uncertainty scores."""
    
    def __init__(self, 
                 firestore_project_id: Optional[str] = None,
                 uncertainty_threshold: float = 0.5,
                 enable_firestore: bool = True):
        """
        Initialize the proofreader.
        
        Args:
            firestore_project_id: Google Cloud project ID
            uncertainty_threshold: Threshold above which proofreading is triggered
            enable_firestore: Whether to enable Firestore integration
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.enable_firestore = enable_firestore and FIRESTORE_AVAILABLE
        
        # Initialize Firestore client
        if self.enable_firestore:
            try:
                if firestore_project_id:
                    self.firestore_client = FirestoreClient(project=firestore_project_id)
                else:
                    self.firestore_client = FirestoreClient()
                logger.info("Firestore client initialized successfully")
            except DefaultCredentialsError:
                logger.warning("Firestore credentials not found - using stub client")
                self.firestore_client = FirestoreClientStub()
                self.enable_firestore = False
        else:
            self.firestore_client = FirestoreClientStub()
        
        # Proofreading statistics
        self.total_proofreads = 0
        self.total_corrections = 0
        self.total_processing_time = 0.0
        
        # Initialize proofreading rules
        self._initialize_proofreading_rules()
        
        logger.info(f"UncertaintyTriggeredProofreader initialized with threshold: {uncertainty_threshold}")
    
    def _initialize_proofreading_rules(self):
        """Initialize proofreading rules and patterns."""
        self.proofreading_rules = {
            'segmentation_artifacts': {
                'description': 'Remove small segmentation artifacts',
                'min_size': 10,
                'max_size': 1000,
                'enabled': True
            },
            'boundary_smoothing': {
                'description': 'Smooth segmentation boundaries',
                'kernel_size': 3,
                'iterations': 1,
                'enabled': True
            },
            'hole_filling': {
                'description': 'Fill small holes in segmentation',
                'max_hole_size': 50,
                'enabled': True
            },
            'noise_reduction': {
                'description': 'Reduce noise in segmentation',
                'threshold': 0.3,
                'enabled': True
            },
            'connectivity_check': {
                'description': 'Ensure connectivity of segmentation',
                'connectivity': 26,  # 26-connectivity for 3D
                'enabled': True
            }
        }
    
    def proofread(self, 
                  segmentation: np.ndarray, 
                  uncertainty_map: np.ndarray,
                  metadata: Optional[Dict[str, Any]] = None) -> ProofreadingResult:
        """
        Proofread the segmentation if uncertainty exceeds threshold.
        
        Args:
            segmentation: Binary segmentation array
            uncertainty_map: Uncertainty scores array
            metadata: Additional metadata about the segmentation
            
        Returns:
            ProofreadingResult with original and corrected segmentation
        """
        start_time = time.time()
        
        # Calculate average uncertainty
        avg_uncertainty = np.mean(uncertainty_map)
        
        # Store original segmentation
        original_segmentation = segmentation.copy()
        
        if avg_uncertainty > self.uncertainty_threshold:
            logger.info(f"Uncertainty {avg_uncertainty:.3f} exceeds threshold {self.uncertainty_threshold} - triggering proofreading")
            
            # Perform proofreading
            corrected_segmentation, corrections = self._perform_proofreading(segmentation, uncertainty_map)
            
            # Calculate confidence improvement
            confidence_improvement = self._calculate_confidence_improvement(
                original_segmentation, corrected_segmentation, uncertainty_map
            )
            
            # Store results in Firestore
            if self.enable_firestore:
                self._store_proofreading_result(
                    original_segmentation, corrected_segmentation, 
                    corrections, confidence_improvement, metadata
                )
            
            # Update statistics
            self.total_proofreads += 1
            self.total_corrections += len(corrections)
            
        else:
            logger.info(f"Uncertainty {avg_uncertainty:.3f} below threshold {self.uncertainty_threshold} - no proofreading needed")
            corrected_segmentation = original_segmentation
            corrections = []
            confidence_improvement = 0.0
        
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        return ProofreadingResult(
            original_text=str(original_segmentation),  # Convert to string for compatibility
            proofread_text=str(corrected_segmentation),
            corrections_made=corrections,
            confidence_improvement=confidence_improvement,
            processing_time=processing_time,
            metadata={
                'average_uncertainty': avg_uncertainty,
                'uncertainty_threshold': self.uncertainty_threshold,
                'proofreading_triggered': avg_uncertainty > self.uncertainty_threshold,
                'original_metadata': metadata or {}
            }
        )
    
    def _perform_proofreading(self, 
                            segmentation: np.ndarray, 
                            uncertainty_map: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Perform the actual proofreading operations."""
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available - returning original segmentation")
            return segmentation, []
        
        corrected_segmentation = segmentation.copy()
        corrections = []
        
        # Apply proofreading rules
        for rule_name, rule_config in self.proofreading_rules.items():
            if not rule_config['enabled']:
                continue
            
            try:
                if rule_name == 'segmentation_artifacts':
                    corrected_segmentation, correction = self._remove_artifacts(
                        corrected_segmentation, rule_config
                    )
                    if correction:
                        corrections.append(correction)
                
                elif rule_name == 'boundary_smoothing':
                    corrected_segmentation, correction = self._smooth_boundaries(
                        corrected_segmentation, rule_config
                    )
                    if correction:
                        corrections.append(correction)
                
                elif rule_name == 'hole_filling':
                    corrected_segmentation, correction = self._fill_holes(
                        corrected_segmentation, rule_config
                    )
                    if correction:
                        corrections.append(correction)
                
                elif rule_name == 'noise_reduction':
                    corrected_segmentation, correction = self._reduce_noise(
                        corrected_segmentation, uncertainty_map, rule_config
                    )
                    if correction:
                        corrections.append(correction)
                
                elif rule_name == 'connectivity_check':
                    corrected_segmentation, correction = self._ensure_connectivity(
                        corrected_segmentation, rule_config
                    )
                    if correction:
                        corrections.append(correction)
                
            except Exception as e:
                logger.error(f"Error applying rule {rule_name}: {e}")
                continue
        
        return corrected_segmentation, corrections
    
    def _remove_artifacts(self, segmentation: np.ndarray, rule_config: Dict[str, Any]) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Remove small segmentation artifacts."""
        min_size = rule_config['min_size']
        max_size = rule_config['max_size']
        
        # Label connected components
        labeled, num_features = ndimage.label(segmentation)
        
        # Count pixels in each component
        component_sizes = np.bincount(labeled.ravel())[1:]  # Skip background (label 0)
        
        # Find components to remove
        to_remove = np.where((component_sizes < min_size) | (component_sizes > max_size))[0] + 1
        
        if len(to_remove) == 0:
            return segmentation, None
        
        # Remove artifacts
        corrected = segmentation.copy()
        for label in to_remove:
            corrected[labeled == label] = 0
        
        return corrected, {
            'rule': 'segmentation_artifacts',
            'artifacts_removed': len(to_remove),
            'min_size': min_size,
            'max_size': max_size
        }
    
    def _smooth_boundaries(self, segmentation: np.ndarray, rule_config: Dict[str, Any]) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Smooth segmentation boundaries."""
        kernel_size = rule_config['kernel_size']
        iterations = rule_config['iterations']
        
        # Create smoothing kernel
        kernel = np.ones((kernel_size, kernel_size, kernel_size))
        
        # Apply morphological operations
        corrected = segmentation.copy()
        for _ in range(iterations):
            corrected = ndimage.binary_closing(corrected, structure=kernel)
            corrected = ndimage.binary_opening(corrected, structure=kernel)
        
        changes = np.sum(corrected != segmentation)
        
        return corrected, {
            'rule': 'boundary_smoothing',
            'pixels_changed': int(changes),
            'kernel_size': kernel_size,
            'iterations': iterations
        } if changes > 0 else None
    
    def _fill_holes(self, segmentation: np.ndarray, rule_config: Dict[str, Any]) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Fill small holes in segmentation."""
        max_hole_size = rule_config['max_hole_size']
        
        # Invert segmentation to find holes
        inverted = ~segmentation
        
        # Label holes
        labeled_holes, num_holes = ndimage.label(inverted)
        
        # Count hole sizes
        hole_sizes = np.bincount(labeled_holes.ravel())[1:]  # Skip background
        
        # Find holes to fill
        holes_to_fill = np.where(hole_sizes <= max_hole_size)[0] + 1
        
        if len(holes_to_fill) == 0:
            return segmentation, None
        
        # Fill holes
        corrected = segmentation.copy()
        for hole_label in holes_to_fill:
            corrected[labeled_holes == hole_label] = 1
        
        return corrected, {
            'rule': 'hole_filling',
            'holes_filled': len(holes_to_fill),
            'max_hole_size': max_hole_size
        }
    
    def _reduce_noise(self, 
                     segmentation: np.ndarray, 
                     uncertainty_map: np.ndarray, 
                     rule_config: Dict[str, Any]) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Reduce noise based on uncertainty map."""
        threshold = rule_config['threshold']
        
        # Find high-uncertainty regions
        high_uncertainty = uncertainty_map > threshold
        
        # Remove segmentation in high-uncertainty regions
        corrected = segmentation.copy()
        corrected[high_uncertainty] = 0
        
        changes = np.sum(corrected != segmentation)
        
        return corrected, {
            'rule': 'noise_reduction',
            'pixels_removed': int(changes),
            'uncertainty_threshold': threshold
        } if changes > 0 else None
    
    def _ensure_connectivity(self, segmentation: np.ndarray, rule_config: Dict[str, Any]) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Ensure connectivity of segmentation."""
        connectivity = rule_config['connectivity']
        
        # Label connected components
        labeled, num_components = ndimage.label(segmentation, structure=np.ones((3, 3, 3)))
        
        if num_components <= 1:
            return segmentation, None
        
        # Keep only the largest component
        component_sizes = np.bincount(labeled.ravel())[1:]
        largest_component = np.argmax(component_sizes) + 1
        
        corrected = np.zeros_like(segmentation)
        corrected[labeled == largest_component] = 1
        
        return corrected, {
            'rule': 'connectivity_check',
            'components_removed': num_components - 1,
            'connectivity': connectivity
        }
    
    def _calculate_confidence_improvement(self, 
                                        original: np.ndarray, 
                                        corrected: np.ndarray, 
                                        uncertainty_map: np.ndarray) -> float:
        """Calculate confidence improvement after proofreading."""
        # Simple metric: reduction in high-uncertainty pixels
        high_uncertainty_original = np.sum((original == 1) & (uncertainty_map > 0.7))
        high_uncertainty_corrected = np.sum((corrected == 1) & (uncertainty_map > 0.7))
        
        if high_uncertainty_original == 0:
            return 0.0
        
        improvement = (high_uncertainty_original - high_uncertainty_corrected) / high_uncertainty_original
        return max(0.0, min(1.0, improvement))
    
    def _store_proofreading_result(self, 
                                 original: np.ndarray, 
                                 corrected: np.ndarray, 
                                 corrections: List[Dict[str, Any]], 
                                 confidence_improvement: float,
                                 metadata: Optional[Dict[str, Any]]):
        """Store proofreading results in Firestore."""
        try:
            # Create document data
            doc_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'original_shape': list(original.shape),
                'corrected_shape': list(corrected.shape),
                'corrections': corrections,
                'confidence_improvement': confidence_improvement,
                'total_corrections': len(corrections),
                'metadata': metadata or {},
                'statistics': {
                    'original_pixels': int(np.sum(original)),
                    'corrected_pixels': int(np.sum(corrected)),
                    'pixel_change': int(np.sum(corrected) - np.sum(original))
                }
            }
            
            # Store in Firestore
            collection = self.firestore_client.collection('proofreading_results')
            doc_id = f"proofread_{int(time.time())}"
            collection.document(doc_id).set(doc_data)
            
            logger.info(f"Stored proofreading result in Firestore: {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to store proofreading result in Firestore: {e}")
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document from Firestore."""
        try:
            collection = self.firestore_client.collection('proofreading_results')
            doc = collection.document(doc_id).get()
            return doc if doc else None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def update_document(self, doc_id: str, data: Dict[str, Any]) -> None:
        """Update a document in Firestore."""
        try:
            collection = self.firestore_client.collection('proofreading_results')
            collection.document(doc_id).update(data)
            logger.info(f"Updated document {doc_id} in Firestore")
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get proofreading statistics."""
        return {
            'total_proofreads': self.total_proofreads,
            'total_corrections': self.total_corrections,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': (
                self.total_processing_time / self.total_proofreads 
                if self.total_proofreads > 0 else 0.0
            ),
            'uncertainty_threshold': self.uncertainty_threshold,
            'firestore_enabled': self.enable_firestore,
            'scipy_available': SCIPY_AVAILABLE,
            'proofreading_rules': {
                name: {'enabled': config['enabled']} 
                for name, config in self.proofreading_rules.items()
            }
        }
    
    def update_uncertainty_threshold(self, new_threshold: float) -> None:
        """Update the uncertainty threshold."""
        self.uncertainty_threshold = new_threshold
        logger.info(f"Updated uncertainty threshold to {new_threshold}")
    
    def enable_rule(self, rule_name: str, enabled: bool = True) -> bool:
        """Enable or disable a proofreading rule."""
        if rule_name in self.proofreading_rules:
            self.proofreading_rules[rule_name]['enabled'] = enabled
            logger.info(f"{'Enabled' if enabled else 'Disabled'} rule: {rule_name}")
            return True
        else:
            logger.warning(f"Unknown rule: {rule_name}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'firestore_client'):
            self.firestore_client.close()
        logger.info("UncertaintyTriggeredProofreader cleaned up") 