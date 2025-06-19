"""
Agent Company Framework

A comprehensive AI framework for computer vision tasks with focus on:
- Segmentation with uncertainty estimation
- Memory and learning systems
- Tool orchestration and automation
- Production-ready deployment
"""

from .segmenters import FFNv2Plugin
from .tool_registry import ModelRegistry

__version__ = "0.1.0"

__all__ = [
    'FFNv2Plugin',
    'ModelRegistry'
]