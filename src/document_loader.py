"""
Backward compatibility module - redirects to enhanced_document_loader.

This module exists to maintain compatibility with existing code that imports from document_loader.
All functionality has been moved to enhanced_document_loader.py.
"""

# Import everything from enhanced_document_loader for backward compatibility
from src.enhanced_document_loader import *

# Import and alias the main class for backward compatibility
from src.enhanced_document_loader import EnhancedDocumentLoader as DocumentLoader

# Also keep the original name available
from src.enhanced_document_loader import EnhancedDocumentLoader

__all__ = ['DocumentLoader', 'EnhancedDocumentLoader']
