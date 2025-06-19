#!/usr/bin/env python3
"""
Memory System
============
FAISS-based memory system for storing and retrieving experiences.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using simple memory")

logger = logging.getLogger(__name__)

class Memory:
    """FAISS-based memory system for storing experiences."""
    
    def __init__(self, memory_file: str = "memory.json", vector_dim: int = 128):
        self.memory_file = memory_file
        self.vector_dim = vector_dim
        self.experiences = []
        
        # Initialize FAISS index if available
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(vector_dim)
            self.vectors = []
        else:
            self.index = None
            self.vectors = []
        
        self._load_memory()
    
    def _load_memory(self):
        """Load memory from file."""
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                self.experiences = data.get('experiences', [])
                if FAISS_AVAILABLE and data.get('vectors'):
                    vectors = np.array(data['vectors'])
                    self.index = faiss.IndexFlatL2(vectors.shape[1])
                    self.index.add(vectors)
                    self.vectors = vectors.tolist()
        except FileNotFoundError:
            logger.info(f"Creating new memory file: {self.memory_file}")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")
    
    def _save_memory(self):
        """Save memory to file."""
        try:
            data = {
                'experiences': self.experiences,
                'vectors': self.vectors
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector representation."""
        # Simple hash-based vector for demo
        # In production, use proper embeddings (e.g., OpenAI embeddings)
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to vector
        vector = np.zeros(self.vector_dim)
        for i, byte in enumerate(hash_bytes):
            if i < self.vector_dim:
                vector[i] = byte / 255.0
        
        return vector
    
    def store_experience(self, ticket: Dict, plan: Dict, spec: Dict, 
                        review: Dict, result: Dict):
        """Store a complete experience in memory."""
        experience = {
            'id': len(self.experiences),
            'timestamp': datetime.now().isoformat(),
            'ticket': ticket,
            'plan': plan,
            'spec': spec,
            'review': review,
            'result': result,
            'success': result.get('status') == 'success'
        }
        
        self.experiences.append(experience)
        
        # Create vector representation
        text = f"{ticket['description']} {plan.get('action', '')} {spec.get('tool_name', '')}"
        vector = self._text_to_vector(text)
        
        if FAISS_AVAILABLE:
            self.index.add(vector.reshape(1, -1))
        
        self.vectors.append(vector.tolist())
        self._save_memory()
        
        logger.info(f"Stored experience {experience['id']} in memory")
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar experiences."""
        if not self.experiences:
            return []
        
        query_vector = self._text_to_vector(query)
        
        if FAISS_AVAILABLE and self.index.ntotal > 0:
            # Use FAISS for similarity search
            D, I = self.index.search(query_vector.reshape(1, -1), min(k, self.index.ntotal))
            similar_experiences = [self.experiences[i] for i in I[0] if i < len(self.experiences)]
        else:
            # Fallback to simple similarity
            similarities = []
            for i, exp in enumerate(self.experiences):
                exp_text = f"{exp['ticket']['description']} {exp['plan'].get('action', '')}"
                exp_vector = self._text_to_vector(exp_text)
                similarity = np.dot(query_vector, exp_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(exp_vector))
                similarities.append((similarity, exp))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            similar_experiences = [exp for _, exp in similarities[:k]]
        
        return similar_experiences
    
    def get_successful_experiences(self) -> List[Dict]:
        """Get all successful experiences."""
        return [exp for exp in self.experiences if exp.get('success', False)]
    
    def get_failed_experiences(self) -> List[Dict]:
        """Get all failed experiences."""
        return [exp for exp in self.experiences if not exp.get('success', True)]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total = len(self.experiences)
        successful = len(self.get_successful_experiences())
        failed = len(self.get_failed_experiences())
        
        return {
            'total_experiences': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'faiss_available': FAISS_AVAILABLE
        } 