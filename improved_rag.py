#!/usr/bin/env python3
"""
Improved RAG System with Complete PDF Processing and Answer Validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import hashlib
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from openai import OpenAI
import re
import json
from datetime import datetime

from src.utils.config import get_config
from src.utils.logger import logger
from src.vector_store import VectorStore
from src.document_manager import DocumentManager, JsonDocumentRepository


class ImprovedPDFProcessor:
    """Enhanced PDF processor that captures ALL text content"""
    
    def __init__(self):
        self.table_patterns = [
            r'\b[A-Z]{2,}\s*[-:]\s*[A-Za-z0-9\s]+',  # Pattern like "GLS: description"
            r'\b[A-Z]{2,}\s+[A-Za-z0-9\s\-]+\s+\d+[A-Za-z]*',  # Pattern with measurements
            r'^\s*[•·▪▫◦‣⁃]\s*.+',  # Bullet points
            r'^\s*\d+\.\s*.+',  # Numbered lists
        ]
    
    def extract_complete_text(self, pdf_path: str) -> Dict[str, Any]:
        """Extract ALL text from PDF with multiple methods"""
        try:
            doc = fitz.open(pdf_path)
            all_content = []
            tables_found = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Method 1: Standard text extraction
                text = page.get_text("text")
                
                # Method 2: Block-based extraction for better structure
                blocks = page.get_text("blocks")
                
                # Method 3: Dictionary extraction for detailed content
                page_dict = page.get_text("dict")
                
                # Combine all methods
                page_content = f"\n=== Page {page_num + 1} ===\n"
                
                # Process blocks for structured content
                if blocks:
                    for block in blocks:
                        if block[6] == 0:  # Text block
                            block_text = block[4].strip()
                            if block_text:
                                # Check if it's a table-like structure
                                if self._is_table_content(block_text):
                                    tables_found.append({
                                        "page": page_num + 1,
