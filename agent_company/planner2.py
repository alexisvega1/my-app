#!/usr/bin/env python3
"""
Planner-2
=========
Advanced planning agent that decides on actions based on tickets and memory.
"""

import logging
import re
from typing import Dict, List, Any
from memory import Memory

logger = logging.getLogger(__name__)

class Planner2:
    """Advanced planning agent that creates action plans."""
    
    def __init__(self):
        self.action_patterns = {
            'build_spec': [
                r'build.*function',
                r'create.*function',
                r'function.*that',
                r'fetch.*price',
                r'get.*data',
                r'calculate.*',
                r'convert.*',
                r'generate.*',
                r'process.*',
                r'analyze.*'
            ],
            'search_info': [
                r'what.*is',
                r'how.*to',
                r'explain.*',
                r'information.*about',
                r'details.*about'
            ],
            'list_tools': [
                r'list.*tools',
                r'show.*tools',
                r'available.*tools',
                r'what.*can.*you.*do'
            ],
            'help': [
                r'help',
                r'usage',
                r'how.*use',
                r'commands'
            ]
        }
    
    def create_plan(self, ticket: Dict[str, Any], memory: Memory) -> Dict[str, Any]:
        """Create a plan based on the ticket and memory."""
        description = ticket['description'].lower()
        
        # Check for similar experiences in memory
        similar_experiences = memory.search_similar(description, k=3)
        
        # Determine action based on patterns and memory
        action = self._determine_action(description, similar_experiences)
        
        # Create plan
        plan = {
            'action': action,
            'ticket_id': ticket['id'],
            'description': description,
            'similar_experiences': len(similar_experiences),
            'confidence': self._calculate_confidence(description, action, similar_experiences),
            'reasoning': self._generate_reasoning(description, action, similar_experiences)
        }
        
        logger.info(f"Created plan: {action} for ticket {ticket['id']}")
        return plan
    
    def _determine_action(self, description: str, similar_experiences: List[Dict]) -> str:
        """Determine the best action based on description and similar experiences."""
        
        # Check if we have successful similar experiences
        successful_similar = [exp for exp in similar_experiences if exp.get('success', False)]
        
        # If we have successful similar experiences, use the same action
        if successful_similar:
            most_recent = max(successful_similar, key=lambda x: x['timestamp'])
            return most_recent['plan']['action']
        
        # Pattern matching for new requests
        for action, patterns in self.action_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description):
                    return action
        
        # Default to build_spec for unknown requests
        return 'build_spec'
    
    def _calculate_confidence(self, description: str, action: str, 
                            similar_experiences: List[Dict]) -> float:
        """Calculate confidence in the chosen action."""
        base_confidence = 0.5
        
        # Boost confidence if we have similar successful experiences
        successful_similar = [exp for exp in similar_experiences if exp.get('success', False)]
        if successful_similar:
            base_confidence += 0.3
        
        # Boost confidence for clear pattern matches
        for patterns in self.action_patterns.values():
            for pattern in patterns:
                if re.search(pattern, description):
                    base_confidence += 0.2
                    break
        
        return min(base_confidence, 1.0)
    
    def _generate_reasoning(self, description: str, action: str, 
                          similar_experiences: List[Dict]) -> str:
        """Generate reasoning for the chosen action."""
        if similar_experiences:
            successful_count = len([exp for exp in similar_experiences if exp.get('success', False)])
            return f"Found {len(similar_experiences)} similar experiences ({successful_count} successful). Using {action} based on historical success."
        
        # Pattern-based reasoning
        for action_name, patterns in self.action_patterns.items():
            if action == action_name:
                for pattern in patterns:
                    if re.search(pattern, description):
                        return f"Pattern match: '{pattern}' suggests {action} action."
        
        return f"No clear pattern match. Defaulting to {action} for new request type."
    
    def get_available_actions(self) -> List[str]:
        """Get list of available actions."""
        return list(self.action_patterns.keys())
    
    def add_action_pattern(self, action: str, pattern: str):
        """Add a new action pattern."""
        if action not in self.action_patterns:
            self.action_patterns[action] = []
        self.action_patterns[action].append(pattern)
        logger.info(f"Added pattern '{pattern}' for action '{action}'") 