"""
Two-Layer Architecture Package
Thay thế OPTIMIZED_GENERAL_MODEL_SOLUTION.md

Layer 1: FrameworkGenerator - Tái sử dụng UnbeatableSequenceModel
Layer 2: StyleLearner - Học style đánh dựa trên framework
"""

from .framework_generator import FrameworkGenerator
from .style_learner import StyleLearner

__all__ = ['FrameworkGenerator', 'StyleLearner']
