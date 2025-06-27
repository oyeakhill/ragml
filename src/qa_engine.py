"""
Backward compatibility module - redirects to enhanced_qa_engine.

This module exists to maintain compatibility with existing code that imports from qa_engine.
All functionality has been moved to enhanced_qa_engine.py.
"""

# Import everything from enhanced_qa_engine for backward compatibility
from src.enhanced_qa_engine import *

# Explicitly import commonly used classes
from src.enhanced_qa_engine import (
    EnhancedQAEngine as QAEngine,  # Alias for backward compatibility
    QueryType,
    QueryAnalysis,
    ValidationResult,
    QueryClassifier,
    IntelligentRetriever,
    ManagedIntelligentRetriever,
    ContextAssembler,
    AnswerValidator
)

# For complete backward compatibility
EnhancedQAEngine = QAEngine

__all__ = [
    'QAEngine',
    'EnhancedQAEngine',
    'QueryType',
    'QueryAnalysis',
    'ValidationResult',
    'QueryClassifier',
    'IntelligentRetriever',
    'ManagedIntelligentRetriever',
    'ContextAssembler',
    'AnswerValidator'
]
