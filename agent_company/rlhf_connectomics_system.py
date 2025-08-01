#!/usr/bin/env python3
"""
RLHF Connectomics System
========================
Reinforcement Learning from Human Feedback system for continuous improvement
of connectomics tracing, proofreading, and RAG systems.
"""

import os
import sys
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import pickle
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from collections import defaultdict, deque
import uuid
import sqlite3
from transformers import AutoTokenizer, AutoModel, pipeline
import openai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class RLHFConfig:
    """Configuration for RLHF connectomics system."""
    # Model configuration
    base_model_path: str = "/models/connectomics_base"
    reward_model_path: str = "/models/reward_model"
    policy_model_path: str = "/models/policy_model"
    
    # Training configuration
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 10
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # RLHF specific
    kl_penalty: float = 0.1
    reward_scale: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Feedback collection
    feedback_queue_size: int = 1000
    feedback_batch_size: int = 32
    min_feedback_samples: int = 100
    
    # Human feedback types
    feedback_types: List[str] = None
    
    # Storage
    feedback_db_path: str = "/data/rlhf_feedback.db"
    model_checkpoint_dir: str = "/models/checkpoints"
    
    # Performance tracking
    evaluation_interval: int = 100
    save_interval: int = 500
    
    def __post_init__(self):
        if self.feedback_types is None:
            self.feedback_types = [
                "tracing_accuracy",
                "proofreading_quality", 
                "rag_helpfulness",
                "user_satisfaction",
                "time_efficiency",
                "error_correction"
            ]

class HumanFeedbackCollector:
    """Collect and manage human feedback for RLHF training."""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.feedback_queue = queue.Queue(maxsize=config.feedback_queue_size)
        self.feedback_db = self._init_feedback_database()
        self.feedback_stats = defaultdict(int)
        self.feedback_processor = threading.Thread(target=self._process_feedback, daemon=True)
        self.feedback_processor.start()
        
        logger.info("Human feedback collector initialized")
    
    def _init_feedback_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for feedback storage."""
        db_path = Path(self.config.feedback_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                feedback_type TEXT,
                task_id TEXT,
                user_id TEXT,
                model_output TEXT,
                human_rating REAL,
                human_correction TEXT,
                context TEXT,
                metadata TEXT
            )
        ''')
        
        # Create feedback statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_stats (
                feedback_type TEXT PRIMARY KEY,
                total_samples INTEGER,
                average_rating REAL,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        return conn
    
    def collect_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Collect human feedback for a specific task."""
        feedback_id = str(uuid.uuid4())
        
        # Add metadata
        feedback_data.update({
            'id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': json.dumps(feedback_data.get('metadata', {}))
        })
        
        # Store in database
        cursor = self.feedback_db.cursor()
        cursor.execute('''
            INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_data['id'],
            feedback_data['timestamp'],
            feedback_data['feedback_type'],
            feedback_data.get('task_id', ''),
            feedback_data.get('user_id', ''),
            json.dumps(feedback_data.get('model_output', {})),
            feedback_data.get('human_rating', 0.0),
            json.dumps(feedback_data.get('human_correction', {})),
            json.dumps(feedback_data.get('context', {})),
            feedback_data['metadata']
        ))
        self.feedback_db.commit()
        
        # Add to processing queue
        try:
            self.feedback_queue.put(feedback_data, timeout=1.0)
            self.feedback_stats[feedback_data['feedback_type']] += 1
        except queue.Full:
            logger.warning("Feedback queue is full, dropping feedback")
        
        return feedback_id
    
    def collect_tracing_feedback(self, task_id: str, user_id: str, 
                                model_tracing: Dict[str, Any], 
                                human_rating: float, 
                                human_correction: Dict[str, Any] = None,
                                context: Dict[str, Any] = None) -> str:
        """Collect feedback for tracing tasks."""
        feedback_data = {
            'feedback_type': 'tracing_accuracy',
            'task_id': task_id,
            'user_id': user_id,
            'model_output': model_tracing,
            'human_rating': human_rating,
            'human_correction': human_correction or {},
            'context': context or {},
            'metadata': {
                'brain_region': context.get('brain_region', 'unknown'),
                'cell_type': context.get('cell_type', 'unknown'),
                'tracing_complexity': context.get('complexity', 'medium')
            }
        }
        
        return self.collect_feedback(feedback_data)
    
    def collect_proofreading_feedback(self, task_id: str, user_id: str,
                                     model_proofreading: Dict[str, Any],
                                     human_rating: float,
                                     human_correction: Dict[str, Any] = None,
                                     context: Dict[str, Any] = None) -> str:
        """Collect feedback for proofreading tasks."""
        feedback_data = {
            'feedback_type': 'proofreading_quality',
            'task_id': task_id,
            'user_id': user_id,
            'model_output': model_proofreading,
            'human_rating': human_rating,
            'human_correction': human_correction or {},
            'context': context or {},
            'metadata': {
                'error_types': context.get('error_types', []),
                'correction_strategies': context.get('correction_strategies', []),
                'quality_metrics': context.get('quality_metrics', {})
            }
        }
        
        return self.collect_feedback(feedback_data)
    
    def collect_rag_feedback(self, task_id: str, user_id: str,
                            rag_response: Dict[str, Any],
                            human_rating: float,
                            human_correction: Dict[str, Any] = None,
                            context: Dict[str, Any] = None) -> str:
        """Collect feedback for RAG system responses."""
        feedback_data = {
            'feedback_type': 'rag_helpfulness',
            'task_id': task_id,
            'user_id': user_id,
            'model_output': rag_response,
            'human_rating': human_rating,
            'human_correction': human_correction or {},
            'context': context or {},
            'metadata': {
                'query_type': context.get('query_type', 'general'),
                'response_helpfulness': context.get('helpfulness', 'medium'),
                'knowledge_relevance': context.get('relevance', 'medium')
            }
        }
        
        return self.collect_feedback(feedback_data)
    
    def _process_feedback(self):
        """Background thread for processing feedback."""
        while True:
            try:
                # Get feedback batch
                feedback_batch = []
                for _ in range(self.config.feedback_batch_size):
                    try:
                        feedback = self.feedback_queue.get(timeout=1.0)
                        feedback_batch.append(feedback)
                    except queue.Empty:
                        break
                
                if feedback_batch:
                    self._update_feedback_statistics(feedback_batch)
                
                time.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Error processing feedback: {e}")
                time.sleep(10)
    
    def _update_feedback_statistics(self, feedback_batch: List[Dict[str, Any]]):
        """Update feedback statistics in database."""
        cursor = self.feedback_db.cursor()
        
        for feedback in feedback_batch:
            feedback_type = feedback['feedback_type']
            rating = feedback['human_rating']
            
            # Update statistics
            cursor.execute('''
                INSERT OR REPLACE INTO feedback_stats 
                (feedback_type, total_samples, average_rating, last_updated)
                VALUES (
                    ?,
                    COALESCE((SELECT total_samples FROM feedback_stats WHERE feedback_type = ?), 0) + 1,
                    COALESCE((SELECT average_rating FROM feedback_stats WHERE feedback_type = ?), 0) * 
                    COALESCE((SELECT total_samples FROM feedback_stats WHERE feedback_type = ?), 0) / 
                    (COALESCE((SELECT total_samples FROM feedback_stats WHERE feedback_type = ?), 0) + 1) + 
                    ? / (COALESCE((SELECT total_samples FROM feedback_stats WHERE feedback_type = ?), 0) + 1),
                    ?
                )
            ''', (feedback_type, feedback_type, feedback_type, feedback_type, 
                  feedback_type, rating, feedback_type, feedback['timestamp']))
        
        self.feedback_db.commit()
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        cursor = self.feedback_db.cursor()
        cursor.execute('SELECT * FROM feedback_stats')
        rows = cursor.fetchall()
        
        stats = {}
        for row in rows:
            stats[row[0]] = {
                'total_samples': row[1],
                'average_rating': row[2],
                'last_updated': row[3]
            }
        
        return stats
    
    def get_feedback_batch(self, feedback_type: str = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get a batch of feedback for training."""
        cursor = self.feedback_db.cursor()
        
        if feedback_type:
            cursor.execute('''
                SELECT * FROM feedback 
                WHERE feedback_type = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (feedback_type, limit))
        else:
            cursor.execute('''
                SELECT * FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        
        feedback_batch = []
        for row in rows:
            feedback_batch.append({
                'id': row[0],
                'timestamp': row[1],
                'feedback_type': row[2],
                'task_id': row[3],
                'user_id': row[4],
                'model_output': json.loads(row[5]),
                'human_rating': row[6],
                'human_correction': json.loads(row[7]),
                'context': json.loads(row[8]),
                'metadata': json.loads(row[9])
            })
        
        return feedback_batch

class RewardModel(nn.Module):
    """Reward model for RLHF training."""
    
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(config.base_model_path)
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass."""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        reward = self.reward_head(pooled_output)
        return reward

class PolicyModel(nn.Module):
    """Policy model for RLHF training."""
    
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(config.base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass."""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        value = self.value_head(pooled_output)
        return value

class RLHFTrainer:
    """RLHF trainer for connectomics models."""
    
    def __init__(self, config: RLHFConfig, feedback_collector: HumanFeedbackCollector):
        self.config = config
        self.feedback_collector = feedback_collector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.reward_model = RewardModel(config).to(self.device)
        self.policy_model = PolicyModel(config).to(self.device)
        
        # Optimizers
        self.reward_optimizer = optim.AdamW(
            self.reward_model.parameters(), 
            lr=config.learning_rate
        )
        self.policy_optimizer = optim.AdamW(
            self.policy_model.parameters(), 
            lr=config.learning_rate
        )
        
        # Training state
        self.training_stats = {
            'reward_loss': [],
            'policy_loss': [],
            'kl_divergence': [],
            'human_ratings': [],
            'model_ratings': []
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.model_checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RLHF trainer initialized on {self.device}")
    
    def train_reward_model(self, feedback_batch: List[Dict[str, Any]]) -> float:
        """Train the reward model on human feedback."""
        self.reward_model.train()
        
        total_loss = 0.0
        batch_size = len(feedback_batch)
        
        for feedback in feedback_batch:
            # Prepare input
            model_output = feedback['model_output']
            human_rating = feedback['human_rating']
            
            # Tokenize input
            input_text = self._prepare_input_text(model_output, feedback['context'])
            inputs = self.policy_model.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Forward pass
            predicted_reward = self.reward_model(**inputs)
            
            # Loss (MSE between predicted and human rating)
            target_reward = torch.tensor([human_rating], dtype=torch.float32).to(self.device)
            loss = nn.MSELoss()(predicted_reward.squeeze(), target_reward)
            
            # Backward pass
            self.reward_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.config.max_grad_norm)
            self.reward_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / batch_size
        self.training_stats['reward_loss'].append(avg_loss)
        
        return avg_loss
    
    def train_policy_model(self, feedback_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train the policy model using PPO-style RLHF."""
        self.policy_model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl_loss = 0.0
        batch_size = len(feedback_batch)
        
        for feedback in feedback_batch:
            # Prepare input
            model_output = feedback['model_output']
            human_rating = feedback['human_rating']
            human_correction = feedback.get('human_correction', {})
            
            # Tokenize input
            input_text = self._prepare_input_text(model_output, feedback['context'])
            inputs = self.policy_model.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get reward from reward model
            with torch.no_grad():
                reward = self.reward_model(**inputs)
            
            # Get value from policy model
            value = self.policy_model(**inputs)
            
            # Policy loss (simplified PPO)
            advantage = reward - value.detach()
            policy_loss = -advantage.mean()
            
            # Value loss
            value_loss = nn.MSELoss()(value.squeeze(), reward.squeeze())
            
            # KL divergence penalty (simplified)
            kl_loss = self.config.kl_penalty * torch.mean(value ** 2)
            
            # Total loss
            total_loss = (policy_loss + 
                         self.config.value_coef * value_loss + 
                         kl_loss)
            
            # Backward pass
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_policy_loss = total_policy_loss / batch_size
        avg_value_loss = total_value_loss / batch_size
        avg_kl_loss = total_kl_loss / batch_size
        
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['kl_divergence'].append(avg_kl_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'kl_loss': avg_kl_loss
        }
    
    def _prepare_input_text(self, model_output: Dict[str, Any], 
                           context: Dict[str, Any]) -> str:
        """Prepare input text for model training."""
        # Format based on feedback type
        if 'tracing' in context.get('task', ''):
            return self._format_tracing_input(model_output, context)
        elif 'proofreading' in context.get('task', ''):
            return self._format_proofreading_input(model_output, context)
        elif 'rag' in context.get('task', ''):
            return self._format_rag_input(model_output, context)
        else:
            return json.dumps(model_output)
    
    def _format_tracing_input(self, model_output: Dict[str, Any], 
                             context: Dict[str, Any]) -> str:
        """Format tracing input for training."""
        brain_region = context.get('brain_region', 'unknown')
        cell_type = context.get('cell_type', 'unknown')
        
        return f"""
        Task: Trace neuron in {brain_region} ({cell_type})
        Model Output: {json.dumps(model_output)}
        Context: {json.dumps(context)}
        """
    
    def _format_proofreading_input(self, model_output: Dict[str, Any], 
                                  context: Dict[str, Any]) -> str:
        """Format proofreading input for training."""
        error_types = context.get('error_types', [])
        quality_metrics = context.get('quality_metrics', {})
        
        return f"""
        Task: Proofread neural tracing
        Error Types: {error_types}
        Model Output: {json.dumps(model_output)}
        Quality Metrics: {json.dumps(quality_metrics)}
        """
    
    def _format_rag_input(self, model_output: Dict[str, Any], 
                         context: Dict[str, Any]) -> str:
        """Format RAG input for training."""
        query_type = context.get('query_type', 'general')
        response = model_output.get('response', '')
        
        return f"""
        Task: RAG Response ({query_type})
        Query: {context.get('query', '')}
        Response: {response}
        Confidence: {model_output.get('confidence', 0.0)}
        """
    
    def evaluate_model(self, feedback_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model performance on feedback."""
        self.reward_model.eval()
        self.policy_model.eval()
        
        predicted_ratings = []
        human_ratings = []
        
        with torch.no_grad():
            for feedback in feedback_batch:
                # Prepare input
                input_text = self._prepare_input_text(feedback['model_output'], feedback['context'])
                inputs = self.policy_model.tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # Get predicted rating
                predicted_reward = self.reward_model(**inputs)
                predicted_rating = predicted_reward.item()
                
                predicted_ratings.append(predicted_rating)
                human_ratings.append(feedback['human_rating'])
        
        # Calculate metrics
        mse = np.mean((np.array(predicted_ratings) - np.array(human_ratings)) ** 2)
        correlation = np.corrcoef(predicted_ratings, human_ratings)[0, 1]
        
        self.training_stats['human_ratings'].extend(human_ratings)
        self.training_stats['model_ratings'].extend(predicted_ratings)
        
        return {
            'mse': mse,
            'correlation': correlation,
            'mean_human_rating': np.mean(human_ratings),
            'mean_model_rating': np.mean(predicted_ratings)
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'reward_model_state_dict': self.reward_model.state_dict(),
            'policy_model_state_dict': self.policy_model.state_dict(),
            'reward_optimizer_state_dict': self.reward_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self, num_epochs: int = None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Get feedback batch
            feedback_batch = self.feedback_collector.get_feedback_batch(limit=self.config.batch_size)
            
            if len(feedback_batch) < self.config.min_feedback_samples:
                logger.warning(f"Insufficient feedback samples: {len(feedback_batch)}")
                continue
            
            # Train reward model
            reward_loss = self.train_reward_model(feedback_batch)
            
            # Train policy model
            policy_metrics = self.train_policy_model(feedback_batch)
            
            # Evaluate
            if epoch % self.config.evaluation_interval == 0:
                eval_metrics = self.evaluate_model(feedback_batch)
                logger.info(f"Epoch {epoch + 1} - Reward Loss: {reward_loss:.4f}, "
                          f"Policy Loss: {policy_metrics['policy_loss']:.4f}, "
                          f"MSE: {eval_metrics['mse']:.4f}, "
                          f"Correlation: {eval_metrics['correlation']:.4f}")
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch, eval_metrics if 'eval_metrics' in locals() else {})
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward loss
        axes[0, 0].plot(self.training_stats['reward_loss'])
        axes[0, 0].set_title('Reward Model Loss')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        
        # Policy loss
        axes[0, 1].plot(self.training_stats['policy_loss'])
        axes[0, 1].set_title('Policy Model Loss')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        
        # KL divergence
        axes[1, 0].plot(self.training_stats['kl_divergence'])
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('KL Divergence')
        
        # Human vs Model ratings
        if self.training_stats['human_ratings'] and self.training_stats['model_ratings']:
            axes[1, 1].scatter(self.training_stats['human_ratings'], 
                             self.training_stats['model_ratings'], alpha=0.5)
            axes[1, 1].plot([0, 1], [0, 1], 'r--')  # Perfect correlation line
            axes[1, 1].set_title('Human vs Model Ratings')
            axes[1, 1].set_xlabel('Human Rating')
            axes[1, 1].set_ylabel('Model Rating')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

class RLHFConnectomicsSystem:
    """Main RLHF system for connectomics applications."""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        
        # Initialize components
        self.feedback_collector = HumanFeedbackCollector(config)
        self.rlhf_trainer = RLHFTrainer(config, self.feedback_collector)
        
        # Integration with existing systems
        self.rag_system = None  # Will be set by external system
        self.tracing_model = None  # Will be set by external system
        self.proofreading_model = None  # Will be set by external system
        
        logger.info("RLHF connectomics system initialized")
    
    def set_rag_system(self, rag_system):
        """Set RAG system for integration."""
        self.rag_system = rag_system
    
    def set_tracing_model(self, tracing_model):
        """Set tracing model for integration."""
        self.tracing_model = tracing_model
    
    def set_proofreading_model(self, proofreading_model):
        """Set proofreading model for integration."""
        self.proofreading_model = proofreading_model
    
    def collect_tracing_feedback(self, task_id: str, user_id: str,
                                model_tracing: Dict[str, Any],
                                human_rating: float,
                                human_correction: Dict[str, Any] = None,
                                context: Dict[str, Any] = None) -> str:
        """Collect feedback for tracing tasks."""
        return self.feedback_collector.collect_tracing_feedback(
            task_id, user_id, model_tracing, human_rating, 
            human_correction, context
        )
    
    def collect_proofreading_feedback(self, task_id: str, user_id: str,
                                     model_proofreading: Dict[str, Any],
                                     human_rating: float,
                                     human_correction: Dict[str, Any] = None,
                                     context: Dict[str, Any] = None) -> str:
        """Collect feedback for proofreading tasks."""
        return self.feedback_collector.collect_proofreading_feedback(
            task_id, user_id, model_proofreading, human_rating,
            human_correction, context
        )
    
    def collect_rag_feedback(self, task_id: str, user_id: str,
                            rag_response: Dict[str, Any],
                            human_rating: float,
                            human_correction: Dict[str, Any] = None,
                            context: Dict[str, Any] = None) -> str:
        """Collect feedback for RAG system responses."""
        return self.feedback_collector.collect_rag_feedback(
            task_id, user_id, rag_response, human_rating,
            human_correction, context
        )
    
    def train_on_feedback(self, num_epochs: int = None):
        """Train models on collected feedback."""
        self.rlhf_trainer.train(num_epochs)
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return self.feedback_collector.get_feedback_statistics()
    
    def evaluate_performance(self) -> Dict[str, float]:
        """Evaluate current model performance."""
        feedback_batch = self.feedback_collector.get_feedback_batch(limit=100)
        return self.rlhf_trainer.evaluate_model(feedback_batch)
    
    def save_models(self, save_dir: str):
        """Save trained models."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save reward model
        torch.save(self.rlhf_trainer.reward_model.state_dict(), 
                  save_path / "reward_model.pt")
        
        # Save policy model
        torch.save(self.rlhf_trainer.policy_model.state_dict(), 
                  save_path / "policy_model.pt")
        
        # Save training stats
        with open(save_path / "training_stats.json", 'w') as f:
            json.dump(self.rlhf_trainer.training_stats, f, indent=2)
        
        logger.info(f"Models saved to {save_path}")
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress."""
        self.rlhf_trainer.plot_training_progress(save_path)

# Example usage
def test_rlhf_system():
    """Test the RLHF connectomics system."""
    # Configuration
    config = RLHFConfig(
        base_model_path="sentence-transformers/all-mpnet-base-v2",
        learning_rate=1e-5,
        batch_size=16,
        num_epochs=5
    )
    
    # Initialize RLHF system
    rlhf_system = RLHFConnectomicsSystem(config)
    
    # Simulate feedback collection
    for i in range(50):
        # Tracing feedback
        rlhf_system.collect_tracing_feedback(
            task_id=f"tracing_{i}",
            user_id="expert_1",
            model_tracing={
                "segments": [(0, 0, 0), (1, 1, 1), (2, 2, 2)],
                "confidence": 0.85,
                "completeness": 0.90
            },
            human_rating=0.8,
            human_correction={
                "missing_segments": [(3, 3, 3)],
                "incorrect_segments": []
            },
            context={
                "brain_region": "cerebral_cortex",
                "cell_type": "pyramidal_neurons",
                "complexity": "high"
            }
        )
        
        # Proofreading feedback
        rlhf_system.collect_proofreading_feedback(
            task_id=f"proofreading_{i}",
            user_id="expert_1",
            model_proofreading={
                "errors_detected": ["membrane_break", "missing_branch"],
                "corrections_suggested": ["interpolate", "add_branch"],
                "confidence": 0.75
            },
            human_rating=0.7,
            context={
                "error_types": ["membrane_break", "missing_branch"],
                "quality_metrics": {"completeness": 0.85, "accuracy": 0.80}
            }
        )
        
        # RAG feedback
        rlhf_system.collect_rag_feedback(
            task_id=f"rag_{i}",
            user_id="expert_1",
            rag_response={
                "response": "Follow the apical dendrite to the pial surface...",
                "confidence": 0.9,
                "sources": ["anatomical_knowledge", "expert_tips"]
            },
            human_rating=0.85,
            context={
                "query_type": "tracing_guidance",
                "helpfulness": "high",
                "relevance": "high"
            }
        )
    
    # Get feedback statistics
    stats = rlhf_system.get_feedback_statistics()
    print(f"Feedback statistics: {json.dumps(stats, indent=2)}")
    
    # Train on feedback
    rlhf_system.train_on_feedback(num_epochs=3)
    
    # Evaluate performance
    metrics = rlhf_system.evaluate_performance()
    print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
    
    # Plot training progress
    rlhf_system.plot_training_progress("training_progress.png")
    
    return rlhf_system

if __name__ == "__main__":
    test_rlhf_system() 