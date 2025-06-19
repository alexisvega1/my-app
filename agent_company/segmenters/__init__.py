"""
Segmentation models and plugins for the agent company framework.

This module contains various neural network architectures for 3D segmentation
tasks, including FFN-v2 with Inception-based backbones.
"""

from .ffn_v2_plugin import FFNv2Plugin, InceptionBlock3D, ConvBnRelu, AuxiliaryClassifier

__all__ = [
    'FFNv2Plugin',
    'InceptionBlock3D', 
    'ConvBnRelu',
    'AuxiliaryClassifier'
]