#!/usr/bin/env python3
"""
Connectomics RAG System for Tracing and Proofreading
===================================================
Specialized Retrieval-Augmented Generation system for connectomics applications,
leveraging domain knowledge, anatomical context, and expert guidance.
"""

import os
import sys
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import pickle
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import faiss
import sentence_transformers
from transformers import AutoTokenizer, AutoModel, pipeline
import openai
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for connectomics RAG system."""
    # Model configuration
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "gpt-4"  # or "claude-3-opus" for better reasoning
    max_context_length: int = 8192
    temperature: float = 0.1  # Low temperature for consistent results
    
    # Retrieval configuration
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    max_chunk_size: int = 512
    overlap_size: int = 50
    
    # Knowledge base configuration
    knowledge_base_path: str = "/data/connectomics_knowledge"
    cache_dir: str = "/cache/rag_cache"
    update_frequency_hours: int = 24
    
    # Specialized features
    enable_anatomical_context: bool = True
    enable_expert_guidance: bool = True
    enable_uncertainty_estimation: bool = True
    enable_multi_modal: bool = True
    
    # Performance optimization
    use_gpu: bool = True
    batch_size: int = 32
    num_workers: int = 4

class ConnectomicsKnowledgeBase:
    """Specialized knowledge base for connectomics domain."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.knowledge_base_path = Path(config.knowledge_base_path)
        self.cache_dir = Path(config.cache_dir)
        
        # Create directories
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize knowledge components
        self.anatomical_knowledge = {}
        self.tracing_guidelines = {}
        self.proofreading_rules = {}
        self.expert_annotations = {}
        self.case_studies = {}
        
        # Load knowledge base
        self._load_knowledge_base()
        
        logger.info("Connectomics knowledge base initialized")
    
    def _load_knowledge_base(self):
        """Load connectomics knowledge from various sources."""
        # Load anatomical knowledge
        self._load_anatomical_knowledge()
        
        # Load tracing guidelines
        self._load_tracing_guidelines()
        
        # Load proofreading rules
        self._load_proofreading_rules()
        
        # Load expert annotations
        self._load_expert_annotations()
        
        # Load case studies
        self._load_case_studies()
    
    def _load_anatomical_knowledge(self):
        """Load anatomical context and brain region information."""
        anatomical_file = self.knowledge_base_path / "anatomical_knowledge.json"
        
        if anatomical_file.exists():
            with open(anatomical_file, 'r') as f:
                self.anatomical_knowledge = json.load(f)
        else:
            # Initialize with default anatomical knowledge
            self.anatomical_knowledge = {
                "brain_regions": {
                    "cerebral_cortex": {
                        "description": "Outer layer of the brain responsible for higher cognitive functions",
                        "tracing_guidelines": [
                            "Follow continuous membrane boundaries",
                            "Look for characteristic cell density patterns",
                            "Consider laminar organization"
                        ],
                        "common_errors": [
                            "Breaking at weak membrane boundaries",
                            "Missing small processes",
                            "Including adjacent tissue"
                        ]
                    },
                    "hippocampus": {
                        "description": "Medial temporal lobe structure involved in memory formation",
                        "tracing_guidelines": [
                            "Follow pyramidal cell layer",
                            "Include dentate gyrus",
                            "Respect subfield boundaries"
                        ],
                        "common_errors": [
                            "Mixing subfields",
                            "Missing small projections",
                            "Breaking at cell layer boundaries"
                        ]
                    },
                    "thalamus": {
                        "description": "Relay station for sensory and motor information",
                        "tracing_guidelines": [
                            "Follow nuclear boundaries",
                            "Include all major nuclei",
                            "Respect white matter tracts"
                        ],
                        "common_errors": [
                            "Mixing nuclei",
                            "Including adjacent structures",
                            "Breaking at nuclear boundaries"
                        ]
                    }
                },
                "cell_types": {
                    "pyramidal_neurons": {
                        "morphology": "Triangular cell body with apical and basal dendrites",
                        "distribution": "Cerebral cortex, hippocampus",
                        "tracing_tips": [
                            "Follow apical dendrite to pial surface",
                            "Include all major dendritic branches",
                            "Look for characteristic spine patterns"
                        ]
                    },
                    "interneurons": {
                        "morphology": "Small, diverse cell types with local projections",
                        "distribution": "Throughout brain",
                        "tracing_tips": [
                            "Focus on cell body and proximal dendrites",
                            "Look for characteristic branching patterns",
                            "Consider inhibitory vs excitatory markers"
                        ]
                    }
                }
            }
            
            # Save default knowledge
            with open(anatomical_file, 'w') as f:
                json.dump(self.anatomical_knowledge, f, indent=2)
    
    def _load_tracing_guidelines(self):
        """Load tracing guidelines and best practices."""
        guidelines_file = self.knowledge_base_path / "tracing_guidelines.json"
        
        if guidelines_file.exists():
            with open(guidelines_file, 'r') as f:
                self.tracing_guidelines = json.load(f)
        else:
            # Initialize with default tracing guidelines
            self.tracing_guidelines = {
                "general_principles": [
                    "Always trace from soma to terminal",
                    "Follow the strongest membrane boundary",
                    "Include all visible processes",
                    "Respect anatomical boundaries",
                    "Maintain connectivity consistency"
                ],
                "quality_checks": [
                    "Verify membrane continuity",
                    "Check for missing branches",
                    "Ensure proper termination",
                    "Validate anatomical plausibility",
                    "Confirm connectivity patterns"
                ],
                "common_mistakes": [
                    "Breaking at weak boundaries",
                    "Missing small processes",
                    "Including adjacent cells",
                    "Incorrect branch connections",
                    "Ignoring anatomical context"
                ],
                "troubleshooting": {
                    "weak_boundaries": "Use multiple channels, check adjacent sections",
                    "missing_processes": "Increase contrast, check different channels",
                    "ambiguous_connections": "Follow strongest signal, check context",
                    "complex_branching": "Trace systematically, use anatomical landmarks"
                }
            }
            
            # Save default guidelines
            with open(guidelines_file, 'w') as f:
                json.dump(self.tracing_guidelines, f, indent=2)
    
    def _load_proofreading_rules(self):
        """Load proofreading rules and validation criteria."""
        rules_file = self.knowledge_base_path / "proofreading_rules.json"
        
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                self.proofreading_rules = json.load(f)
        else:
            # Initialize with default proofreading rules
            self.proofreading_rules = {
                "validation_criteria": {
                    "membrane_continuity": {
                        "description": "Check for breaks in cell membrane",
                        "severity": "critical",
                        "fix_strategies": [
                            "Interpolate between visible points",
                            "Check adjacent sections",
                            "Use multiple channels"
                        ]
                    },
                    "branch_completeness": {
                        "description": "Ensure all visible branches are included",
                        "severity": "high",
                        "fix_strategies": [
                            "Add missing branches",
                            "Check for weak signals",
                            "Verify termination points"
                        ]
                    },
                    "anatomical_plausibility": {
                        "description": "Verify tracing follows anatomical rules",
                        "severity": "high",
                        "fix_strategies": [
                            "Check against anatomical atlas",
                            "Verify cell type consistency",
                            "Validate connectivity patterns"
                        ]
                    },
                    "connectivity_consistency": {
                        "description": "Ensure connectivity makes sense",
                        "severity": "medium",
                        "fix_strategies": [
                            "Check synaptic partners",
                            "Verify projection patterns",
                            "Validate circuit logic"
                        ]
                    }
                },
                "automated_checks": [
                    "Membrane continuity analysis",
                    "Branch completeness scoring",
                    "Anatomical boundary validation",
                    "Connectivity pattern verification",
                    "Quality metric calculation"
                ]
            }
            
            # Save default rules
            with open(rules_file, 'w') as f:
                json.dump(self.proofreading_rules, f, indent=2)
    
    def _load_expert_annotations(self):
        """Load expert annotations and corrections."""
        annotations_file = self.knowledge_base_path / "expert_annotations.json"
        
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                self.expert_annotations = json.load(f)
        else:
            # Initialize with sample expert annotations
            self.expert_annotations = {
                "correction_patterns": {
                    "membrane_breaks": {
                        "frequency": 0.15,
                        "common_causes": ["Weak staining", "Section artifacts"],
                        "expert_solutions": [
                            "Use adjacent sections for continuity",
                            "Check multiple channels",
                            "Apply morphological constraints"
                        ]
                    },
                    "missing_branches": {
                        "frequency": 0.25,
                        "common_causes": ["Low contrast", "Small processes"],
                        "expert_solutions": [
                            "Increase contrast sensitivity",
                            "Check for weak signals",
                            "Use anatomical context"
                        ]
                    },
                    "wrong_connections": {
                        "frequency": 0.10,
                        "common_causes": ["Ambiguous boundaries", "Complex branching"],
                        "expert_solutions": [
                            "Follow strongest boundary",
                            "Check anatomical landmarks",
                            "Verify connectivity logic"
                        ]
                    }
                },
                "expert_tips": [
                    "Always start from the soma and work outward",
                    "Use anatomical landmarks for orientation",
                    "Check multiple channels for confirmation",
                    "When in doubt, follow the strongest signal",
                    "Consider the cell type and its typical morphology"
                ]
            }
            
            # Save default annotations
            with open(annotations_file, 'w') as f:
                json.dump(self.expert_annotations, f, indent=2)
    
    def _load_case_studies(self):
        """Load case studies and examples."""
        cases_file = self.knowledge_base_path / "case_studies.json"
        
        if cases_file.exists():
            with open(cases_file, 'r') as f:
                self.case_studies = json.load(f)
        else:
            # Initialize with sample case studies
            self.case_studies = {
                "complex_branching": {
                    "description": "Neuron with extensive dendritic arborization",
                    "challenges": [
                        "Many small branches",
                        "Complex 3D structure",
                        "Weak signals in distal processes"
                    ],
                    "solutions": [
                        "Systematic branch-by-branch tracing",
                        "Use of anatomical landmarks",
                        "Multiple channel verification"
                    ],
                    "quality_metrics": {
                        "completeness": 0.95,
                        "accuracy": 0.92,
                        "time_efficiency": 0.85
                    }
                },
                "ambiguous_boundaries": {
                    "description": "Neuron with unclear membrane boundaries",
                    "challenges": [
                        "Weak membrane staining",
                        "Adjacent cell interference",
                        "Section artifacts"
                    ],
                    "solutions": [
                        "Multi-channel analysis",
                        "Adjacent section comparison",
                        "Morphological constraints"
                    ],
                    "quality_metrics": {
                        "completeness": 0.88,
                        "accuracy": 0.90,
                        "time_efficiency": 0.75
                    }
                }
            }
            
            # Save default case studies
            with open(cases_file, 'w') as f:
                json.dump(self.case_studies, f, indent=2)
    
    def get_relevant_knowledge(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge based on query and context."""
        relevant_knowledge = {
            "anatomical_context": {},
            "tracing_guidelines": [],
            "proofreading_rules": [],
            "expert_tips": [],
            "case_studies": {}
        }
        
        # Extract brain region and cell type from context
        brain_region = context.get('brain_region', 'unknown')
        cell_type = context.get('cell_type', 'unknown')
        issue_type = context.get('issue_type', 'general')
        
        # Get anatomical context
        if brain_region in self.anatomical_knowledge.get('brain_regions', {}):
            relevant_knowledge['anatomical_context'] = self.anatomical_knowledge['brain_regions'][brain_region]
        
        if cell_type in self.anatomical_knowledge.get('cell_types', {}):
            relevant_knowledge['anatomical_context']['cell_type'] = self.anatomical_knowledge['cell_types'][cell_type]
        
        # Get tracing guidelines
        relevant_knowledge['tracing_guidelines'] = self.tracing_guidelines.get('general_principles', [])
        
        if issue_type in self.tracing_guidelines.get('troubleshooting', {}):
            relevant_knowledge['tracing_guidelines'].append(
                self.tracing_guidelines['troubleshooting'][issue_type]
            )
        
        # Get proofreading rules
        if issue_type in self.proofreading_rules.get('validation_criteria', {}):
            relevant_knowledge['proofreading_rules'] = [
                self.proofreading_rules['validation_criteria'][issue_type]
            ]
        
        # Get expert tips
        relevant_knowledge['expert_tips'] = self.expert_annotations.get('expert_tips', [])
        
        # Get relevant case studies
        for case_name, case_data in self.case_studies.items():
            if any(challenge in query.lower() for challenge in case_data.get('challenges', [])):
                relevant_knowledge['case_studies'][case_name] = case_data
        
        return relevant_knowledge

class ConnectomicsRAGSystem:
    """Specialized RAG system for connectomics applications."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.knowledge_base = ConnectomicsKnowledgeBase(config)
        self.embedding_model = self._load_embedding_model()
        self.llm_client = self._setup_llm_client()
        self.vector_index = self._build_vector_index()
        
        # Cache for performance
        self.query_cache = {}
        self.embedding_cache = {}
        
        logger.info(f"Connectomics RAG system initialized on {self.device}")
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings."""
        try:
            model = sentence_transformers.SentenceTransformer(self.config.embedding_model)
            model.to(self.device)
            logger.info(f"Embedding model loaded: {self.config.embedding_model}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    
    def _setup_llm_client(self):
        """Setup LLM client (OpenAI, Anthropic, etc.)."""
        try:
            if self.config.llm_model.startswith("gpt"):
                # OpenAI setup
                openai.api_key = os.getenv('OPENAI_API_KEY')
                return "openai"
            elif self.config.llm_model.startswith("claude"):
                # Anthropic setup
                return "anthropic"
            else:
                # Local model setup
                return "local"
        except Exception as e:
            logger.error(f"Failed to setup LLM client: {e}")
            return None
    
    def _build_vector_index(self):
        """Build FAISS vector index for efficient retrieval."""
        try:
            # Create FAISS index
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Add knowledge base documents to index
            documents = self._extract_documents()
            embeddings = self._compute_embeddings(documents)
            
            if embeddings is not None and len(embeddings) > 0:
                index.add(embeddings.astype('float32'))
                logger.info(f"Vector index built with {len(documents)} documents")
            
            return index
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            return None
    
    def _extract_documents(self) -> List[str]:
        """Extract documents from knowledge base for indexing."""
        documents = []
        
        # Extract from anatomical knowledge
        for region, data in self.knowledge_base.anatomical_knowledge.get('brain_regions', {}).items():
            documents.append(f"Brain region {region}: {data.get('description', '')}")
            documents.extend([f"Tracing guideline for {region}: {guideline}" 
                            for guideline in data.get('tracing_guidelines', [])])
        
        # Extract from tracing guidelines
        documents.extend(self.knowledge_base.tracing_guidelines.get('general_principles', []))
        documents.extend(self.knowledge_base.tracing_guidelines.get('quality_checks', []))
        
        # Extract from proofreading rules
        for rule_name, rule_data in self.knowledge_base.proofreading_rules.get('validation_criteria', {}).items():
            documents.append(f"Proofreading rule {rule_name}: {rule_data.get('description', '')}")
        
        # Extract from expert annotations
        documents.extend(self.knowledge_base.expert_annotations.get('expert_tips', []))
        
        return documents
    
    def _compute_embeddings(self, documents: List[str]) -> Optional[np.ndarray]:
        """Compute embeddings for documents."""
        try:
            if self.embedding_model is None:
                return None
            
            embeddings = self.embedding_model.encode(
                documents,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return None
    
    def retrieve_relevant_context(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant context for the query."""
        try:
            # Check cache first
            query_hash = hashlib.md5(query.encode()).hexdigest()
            if query_hash in self.query_cache:
                return self.query_cache[query_hash]
            
            # Get domain-specific knowledge
            domain_knowledge = self.knowledge_base.get_relevant_knowledge(query, context)
            
            # Get vector-based retrieval results
            vector_results = self._vector_retrieval(query)
            
            # Combine and rank results
            combined_results = self._combine_and_rank_results(domain_knowledge, vector_results, query)
            
            # Cache results
            self.query_cache[query_hash] = combined_results
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant context: {e}")
            return []
    
    def _vector_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Perform vector-based retrieval."""
        try:
            if self.vector_index is None or self.embedding_model is None:
                return []
            
            # Compute query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Search index
            scores, indices = self.vector_index.search(
                query_embedding.astype('float32'), 
                self.config.top_k_retrieval
            )
            
            # Get documents
            documents = self._extract_documents()
            results = []
            
            for score, idx in zip(scores[0], indices[0]):
                if score > self.config.similarity_threshold and idx < len(documents):
                    results.append({
                        'content': documents[idx],
                        'score': float(score),
                        'source': 'vector_retrieval'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform vector retrieval: {e}")
            return []
    
    def _combine_and_rank_results(self, domain_knowledge: Dict[str, Any], 
                                 vector_results: List[Dict[str, Any]], 
                                 query: str) -> List[Dict[str, Any]]:
        """Combine and rank retrieval results."""
        combined_results = []
        
        # Add domain knowledge with high priority
        if domain_knowledge.get('anatomical_context'):
            combined_results.append({
                'content': f"Anatomical context: {json.dumps(domain_knowledge['anatomical_context'])}",
                'score': 0.95,
                'source': 'domain_knowledge'
            })
        
        if domain_knowledge.get('tracing_guidelines'):
            combined_results.append({
                'content': f"Tracing guidelines: {'; '.join(domain_knowledge['tracing_guidelines'])}",
                'score': 0.90,
                'source': 'domain_knowledge'
            })
        
        if domain_knowledge.get('expert_tips'):
            combined_results.append({
                'content': f"Expert tips: {'; '.join(domain_knowledge['expert_tips'])}",
                'score': 0.85,
                'source': 'domain_knowledge'
            })
        
        # Add vector results
        combined_results.extend(vector_results)
        
        # Sort by score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to top results
        return combined_results[:self.config.top_k_retrieval]
    
    def generate_response(self, query: str, context: Dict[str, Any], 
                         retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using LLM with retrieved context."""
        try:
            # Prepare prompt
            prompt = self._build_prompt(query, context, retrieved_context)
            
            # Generate response
            if self.llm_client == "openai":
                response = self._generate_openai_response(prompt)
            elif self.llm_client == "anthropic":
                response = self._generate_anthropic_response(prompt)
            else:
                response = self._generate_local_response(prompt)
            
            # Post-process response
            processed_response = self._post_process_response(response, context)
            
            return processed_response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                'response': f"Error generating response: {str(e)}",
                'confidence': 0.0,
                'sources': []
            }
    
    def _build_prompt(self, query: str, context: Dict[str, Any], 
                     retrieved_context: List[Dict[str, Any]]) -> str:
        """Build prompt for LLM."""
        prompt = f"""You are an expert connectomics researcher helping with neural tracing and proofreading. 

Context:
- Brain region: {context.get('brain_region', 'unknown')}
- Cell type: {context.get('cell_type', 'unknown')}
- Issue type: {context.get('issue_type', 'general')}
- Current task: {context.get('task', 'tracing')}

Relevant knowledge:
"""
        
        for i, ctx in enumerate(retrieved_context):
            prompt += f"{i+1}. {ctx['content']}\n"
        
        prompt += f"""

Query: {query}

Please provide a detailed, actionable response that:
1. Addresses the specific query
2. Uses the relevant knowledge provided
3. Gives step-by-step guidance if applicable
4. Includes confidence level and reasoning
5. Suggests additional checks or considerations

Response:"""
        
        return prompt
    
    def _generate_openai_response(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert connectomics researcher."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_context_length // 2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {str(e)}"
    
    def _generate_anthropic_response(self, prompt: str) -> str:
        """Generate response using Anthropic API."""
        try:
            # This would use Anthropic's API
            # For now, return a placeholder
            return "Anthropic API response placeholder"
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error: {str(e)}"
    
    def _generate_local_response(self, prompt: str) -> str:
        """Generate response using local model."""
        try:
            # This would use a local model like Llama or Mistral
            # For now, return a placeholder
            return "Local model response placeholder"
        except Exception as e:
            logger.error(f"Local model error: {e}")
            return f"Error: {str(e)}"
    
    def _post_process_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process LLM response."""
        # Extract confidence and sources
        confidence = 0.8  # Default confidence
        sources = []
        
        # Try to extract confidence from response
        if "confidence:" in response.lower():
            try:
                confidence_line = [line for line in response.split('\n') if 'confidence:' in line.lower()][0]
                confidence = float(confidence_line.split(':')[1].strip())
            except:
                pass
        
        # Extract sources
        if "sources:" in response.lower():
            try:
                sources_section = response.split("sources:")[1].split('\n')[0]
                sources = [s.strip() for s in sources_section.split(',')]
            except:
                pass
        
        return {
            'response': response,
            'confidence': confidence,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        }
    
    def query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main query interface for the RAG system."""
        if context is None:
            context = {}
        
        start_time = time.time()
        
        try:
            # Retrieve relevant context
            retrieved_context = self.retrieve_relevant_context(query, context)
            
            # Generate response
            response = self.generate_response(query, context, retrieved_context)
            
            # Add metadata
            response['query'] = query
            response['context'] = context
            response['retrieved_context'] = retrieved_context
            response['processing_time'] = time.time() - start_time
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'response': f"Error processing query: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'processing_time': time.time() - start_time
            }

# Example usage
def test_connectomics_rag():
    """Test the connectomics RAG system."""
    # Configuration
    config = RAGConfig(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        llm_model="gpt-4",
        top_k_retrieval=5,
        similarity_threshold=0.7
    )
    
    # Initialize RAG system
    rag_system = ConnectomicsRAGSystem(config)
    
    # Test queries
    test_queries = [
        {
            "query": "How should I trace a pyramidal neuron in the cerebral cortex?",
            "context": {
                "brain_region": "cerebral_cortex",
                "cell_type": "pyramidal_neurons",
                "task": "tracing"
            }
        },
        {
            "query": "What should I check when proofreading a complex dendritic arborization?",
            "context": {
                "brain_region": "hippocampus",
                "cell_type": "pyramidal_neurons",
                "task": "proofreading",
                "issue_type": "complex_branching"
            }
        },
        {
            "query": "How do I handle weak membrane boundaries during tracing?",
            "context": {
                "brain_region": "thalamus",
                "cell_type": "unknown",
                "task": "tracing",
                "issue_type": "weak_boundaries"
            }
        }
    ]
    
    # Process queries
    for test_case in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {test_case['query']}")
        print(f"Context: {test_case['context']}")
        
        response = rag_system.query(test_case['query'], test_case['context'])
        
        print(f"\nResponse: {response['response']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Sources: {response['sources']}")
        print(f"Processing time: {response['processing_time']:.2f}s")
    
    return rag_system

if __name__ == "__main__":
    test_connectomics_rag() 