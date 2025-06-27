"""
Enhanced Document Loader with table extraction and structure-aware chunking.

This module provides production-grade document processing capabilities with:
- Intelligent PDF parsing with layout preservation
- Table detection and extraction
- Structure-aware text chunking
- Metadata enrichment
- Error handling and validation

Architecture:
- Strategy Pattern for different document types
- Template Method for processing pipeline
- Factory Pattern for chunker creation
- Single Responsibility Principle adherence
"""

import re
import hashlib
import time
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import fitz  # PyMuPDF
import pandas as pd
try:
    from PIL import Image
    import pytesseract
    import numpy as np
    OCR_AVAILABLE = True
    try:
        pytesseract.get_tesseract_version()
        logger.info("✅ Tesseract OCR is installed and available")
    except Exception as e:
        logger.warning(f"⚠️ Tesseract OCR not available: {e}")
        OCR_AVAILABLE = False
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("⚠️ OCR dependencies not installed. Install with: pip install pytesseract pillow pdf2image")

from src.utils.logger import logger


@dataclass
class DocumentMetadata:
    """Value object for document metadata."""
    filename: str
    file_path: str
    total_pages: int
    file_size: int
    document_type: str
    has_tables: bool = False
    table_count: int = 0
    processing_time: float = 0.0
    content_hash: Optional[str] = None


@dataclass
class TableInfo:
    """Value object for table information."""
    content: str
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0
    table_type: str = "detected"


class DocumentProcessor(ABC):
    """Abstract base class for document processors using Strategy pattern."""
    
    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the file type."""
        pass
    
    @abstractmethod
    def extract_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from the document."""
        pass


class PDFProcessor(DocumentProcessor):
    """PDF processor with advanced table detection, layout preservation, and OCR support."""
    
    def __init__(self, use_ocr: bool = True, ocr_threshold: int = 50, enable_logging: bool = True):
        self.supported_extensions = {'.pdf'}
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.ocr_threshold = ocr_threshold  # Min chars to skip OCR
        self.enable_logging = enable_logging
        
        if self.use_ocr:
            logger.info("🔍 OCR-Enhanced PDF processing enabled")
        else:
            logger.info("📄 Standard PDF processing (no OCR)")
        
    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported PDF."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def extract_content(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract content from PDF with table detection and layout preservation.
        
        Returns:
            Dictionary containing text, tables, and metadata
        """
        try:
            doc = fitz.open(str(file_path))
            text_pages = []
            all_tables = []
            total_text_length = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with layout preservation
                text = self._extract_text_with_layout(page)
                
                # Detect and extract tables
                tables = self._extract_tables_from_page(page, page_num + 1)
                all_tables.extend(tables)
                
                if text.strip() or tables:
                    page_info = {
                        "page_number": page_num + 1,
                        "text": text,
                        "has_tables": len(tables) > 0,
                        "table_count": len(tables),
                        "char_count": len(text)
                    }
                    text_pages.append(page_info)
                    total_text_length += len(text)
            
            doc.close()
            
            # Merge content with table markers
            full_text = self._merge_content_with_tables(text_pages, all_tables)
            
            # Create metadata
            metadata = DocumentMetadata(
                filename=file_path.name,
                file_path=str(file_path.absolute()),
                total_pages=len(text_pages),
                file_size=file_path.stat().st_size,
                document_type="pdf",
                has_tables=len(all_tables) > 0,
                table_count=len(all_tables)
            )
            
            # Add content hash for deduplication
            metadata.content_hash = hashlib.sha256(full_text.encode()).hexdigest()[:16]
            
            logger.info(
                f"Successfully processed PDF: {file_path.name} "
                f"({len(text_pages)} pages, {len(all_tables)} tables, "
                f"{total_text_length:,} characters)"
            )
            
            return {
                "text": full_text,
                "pages": text_pages,
                "tables": all_tables,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
    
    def _extract_text_with_layout(self, page) -> str:
        """
        Extract text while preserving document layout and structure.
        
        Uses multiple extraction methods to capture ALL text including:
        - Regular text blocks
        - Text in different layers
        - Text spans that might be in graphics
        - OCR for image-based text
        """
        page_num = page.number + 1
        start_time = time.time()
        
        try:
            all_text_parts = []
            
            # Method 1: Get text blocks with position information
            blocks = page.get_text("blocks")
            pymupdf_text = ""
            
            if blocks:
                # Sort blocks by reading order (top-to-bottom, left-to-right)
                blocks.sort(key=lambda block: (block[1], block[0]))  # y-coordinate, then x-coordinate
                
                for block in blocks:
                    if len(block) > 6 and block[6] == 0:  # Text block (not image)
                        block_text = block[4].strip()
                        if block_text:
                            # Clean up the text while preserving structure
                            cleaned_text = self._clean_text_block(block_text)
                            if cleaned_text:
                                all_text_parts.append(cleaned_text)
                                pymupdf_text += cleaned_text + "\n"
            
            # Method 2: Extract from dictionary format to catch more text
            dict_text = page.get_text("dict")
            additional_texts = set()
            
            if "blocks" in dict_text:
                for block in dict_text["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    if "text" in span:
                                        span_text = span["text"].strip()
                                        # Add short texts that might be model names
                                        if span_text and len(span_text) <= 20 and span_text not in "\n".join(all_text_parts):
                                            additional_texts.add(span_text)
            
            # Add additional unique texts found
            if additional_texts:
                all_text_parts.append("\n[Additional text elements found:]")
                all_text_parts.extend(sorted(additional_texts))
            
            # Method 3: Simple extraction
            simple_text = page.get_text("text", sort=True)
            
            # Calculate how much text we got from PyMuPDF
            pymupdf_char_count = len(pymupdf_text)
            
            # Log extraction details
            if self.enable_logging:
                logger.info(f"  Page {page_num} - PyMuPDF extracted: {pymupdf_char_count} chars")
                
                # Check for specific terms
                combined_text = "\n".join(all_text_parts).lower()
                logger.info(f"  Page {page_num} - Contains 'GLS': {'✅' if 'gls' in combined_text else '❌'}")
                logger.info(f"  Page {page_num} - Contains 'GLX': {'✅' if 'glx' in combined_text else '❌'}")
            
            # Method 4: OCR if enabled and text is minimal
            ocr_text = ""
            if self.use_ocr and pymupdf_char_count < self.ocr_threshold:
                logger.info(f"  Page {page_num} - Text below threshold ({pymupdf_char_count} < {self.ocr_threshold}), applying OCR...")
                ocr_text = self._ocr_page(page)
                
                if ocr_text:
                    ocr_char_count = len(ocr_text)
                    logger.info(f"  Page {page_num} - OCR extracted: {ocr_char_count} chars")
                    
                    # Check OCR results for specific terms
                    ocr_lower = ocr_text.lower()
                    logger.info(f"  Page {page_num} - OCR contains 'GLS': {'✅' if 'gls' in ocr_lower else '❌'}")
                    logger.info(f"  Page {page_num} - OCR contains 'GLX': {'✅' if 'glx' in ocr_lower else '❌'}")
                    
                    # Add OCR text to results
                    all_text_parts.append("\n[OCR Extracted Text:]")
                    all_text_parts.append(ocr_text)
            
            # Combine all text
            if all_text_parts:
                final_text = "\n\n".join(all_text_parts)
            elif ocr_text:
                final_text = ocr_text
            else:
                final_text = simple_text
            
            # Log processing time
            processing_time = time.time() - start_time
            if self.enable_logging:
                logger.info(f"  Page {page_num} - Processing time: {processing_time:.2f}s")
                logger.info(f"  Page {page_num} - Final text length: {len(final_text)} chars")
            
            return final_text
            
        except Exception as e:
            logger.warning(f"Error in layout extraction for page {page_num}, falling back to simple: {e}")
            return page.get_text("text", sort=True)
    
    def _ocr_page(self, page) -> str:
        """
        Perform OCR on a PDF page.
        
        Returns:
            Extracted text from OCR
        """
        try:
            # Render page to image
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(img, lang='eng')
            
            # Clean up OCR text
            ocr_text = self._clean_ocr_text(ocr_text)
            
            return ocr_text
            
        except Exception as e:
            logger.error(f"OCR failed for page: {e}")
            return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean up OCR-extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common OCR errors
        # GLS/GLX specific fixes
        text = re.sub(r'\bGL5\b', 'GLS', text)  # Common OCR error: S -> 5
        text = re.sub(r'\bGLX\s*\b', 'GLX', text)
        text = re.sub(r'\bGL\$\b', 'GLS', text)  # Common OCR error: S -> $
        
        return text.strip()
    
    def _clean_text_block(self, text: str) -> str:
        """Clean and normalize text block while preserving meaning."""
        if not text:
            return ""
        
        # Normalize whitespace but preserve paragraph breaks
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces on lines
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces on lines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        return text.strip()
    
    def _extract_tables_from_page(self, page, page_num: int) -> List[TableInfo]:
        """
        Extract ALL structured content from PDF page.
        This captures tables, lists, specifications, and any structured data.
        """
        tables = []
        
        # Get all text extraction methods for comprehensive coverage
        text = page.get_text()
        blocks = page.get_text("blocks")
        dict_content = page.get_text("dict")
        
        if not text.strip():
            return tables
        
        lines = text.split('\n')
        current_table_lines = []
        in_table = False
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                if in_table and current_table_lines:
                    # End of table/structured content
                    table_content = self._format_table_content(current_table_lines)
                    if table_content:
                        tables.append(TableInfo(
                            content=table_content,
                            page_number=page_num,
                            confidence=self._assess_table_confidence(current_table_lines),
                            table_type="structured_content"
                        ))
                    current_table_lines = []
                    in_table = False
                continue
            
            # Detect ANY structured content (not just tables)
            is_structured = self._is_structured_content(line)
            
            if is_structured:
                if not in_table:
                    in_table = True
                current_table_lines.append(line)
            else:
                if in_table and current_table_lines:
                    # Check if we should continue the structured content
                    if len(current_table_lines) >= 2:  # Minimum size
                        table_content = self._format_table_content(current_table_lines)
                        if table_content:
                            tables.append(TableInfo(
                                content=table_content,
                                page_number=page_num,
                                confidence=self._assess_table_confidence(current_table_lines),
                                table_type="structured_content"
                            ))
                    current_table_lines = []
                    in_table = False
        
        # Handle structured content at end of page
        if in_table and current_table_lines and len(current_table_lines) >= 2:
            table_content = self._format_table_content(current_table_lines)
            if table_content:
                tables.append(TableInfo(
                    content=table_content,
                    page_number=page_num,
                    confidence=self._assess_table_confidence(current_table_lines),
                    table_type="structured_content"
                ))
        
        return tables
    
    def _is_structured_content(self, line: str) -> bool:
        """
        Determine if a line contains ANY structured content.
        This is more comprehensive than just table detection.
        """
        if not line.strip():
            return False
        
        # Pattern 1: Contains pipe separators
        if '|' in line:
            return True
        
        # Pattern 2: Contains tabs
        if '\t' in line:
            return True
        
        # Pattern 3: Multiple columns separated by 2+ spaces
        if re.search(r'  {2,}', line):
            return True
        
        # Pattern 4: Key-value pairs (specifications)
        if ':' in line and re.match(r'^[A-Za-z0-9\s\-_]+:\s*.+', line):
            return True
        
        # Pattern 5: Lines with measurements/units
        if re.search(r'\b\d+\s*(mm|cm|m|kg|W|Hz|V|A|°C|%|x|X)\b', line, re.IGNORECASE):
            return True
        
        # Pattern 6: Bullet points or lists
        if re.match(r'^\s*[•·▪▫◦‣⁃\-\*]\s+', line):
            return True
        
        # Pattern 7: Numbered lists
        if re.match(r'^\s*\d+[\.\)]\s+', line):
            return True
        
        # Pattern 8: All caps headers (likely section titles)
        words = line.split()
        if len(words) >= 2 and all(word.isupper() for word in words[:2]):
            return True
        
        # Pattern 9: Model numbers or product codes (mix of letters and numbers)
        if re.search(r'\b[A-Z]{2,}[\s\-]?\d+\b', line):
            return True
        
        return False
    
    def _is_table_line(self, line: str) -> bool:
        """
        Determine if a line is likely part of a table.
        
        Uses multiple heuristics to detect tabular data.
        """
        if not line.strip():
            return False
        
        # Pattern 1: Contains pipe separators
        if '|' in line and line.count('|') >= 2:
            return True
        
        # Pattern 2: Contains multiple tabs
        if '\t' in line and line.count('\t') >= 2:
            return True
        
        # Pattern 3: Multiple columns separated by spaces (2+ spaces)
        if re.search(r'  {2,}', line):
            parts = re.split(r'  {2,}', line.strip())
            if len(parts) >= 3:  # At least 3 columns
                return True
        
        # Pattern 4: Colon-separated key-value pairs (specifications)
        if re.match(r'^[A-Za-z\s]+:\s+.+', line):
            return True
        
        # Pattern 5: Lines with numbers and units (measurements, specs)
        if re.search(r'\b\d+\s*(mm|cm|m|kg|W|Hz|V|A|°C|%)\b', line, re.IGNORECASE):
            return True
        
        # Pattern 6: Technical specifications pattern
        if re.search(r'^[A-Za-z\s]+\s+[A-Za-z0-9\s\-\|\+]+$', line):
            # Check if it looks like "Property Value" format
            parts = line.split()
            if len(parts) >= 2:
                return True
        
        return False
    
    def _format_table_content(self, table_lines: List[str]) -> str:
        """Format table lines into a clean table representation."""
        if not table_lines:
            return ""
        
        # Clean and format the table
        formatted_lines = []
        for line in table_lines:
            # Normalize spacing
            if '|' in line:
                # Pipe-separated table
                parts = [part.strip() for part in line.split('|')]
                formatted_line = ' | '.join(part for part in parts if part)
            elif '\t' in line:
                # Tab-separated table
                parts = [part.strip() for part in line.split('\t')]
                formatted_line = ' | '.join(part for part in parts if part)
            else:
                # Space-separated table
                formatted_line = re.sub(r'  +', ' | ', line.strip())
            
            if formatted_line.strip():
                formatted_lines.append(formatted_line)
        
        return '\n'.join(formatted_lines)
    
    def _assess_table_confidence(self, table_lines: List[str]) -> float:
        """Assess confidence that the detected content is actually a table."""
        if not table_lines:
            return 0.0
        
        score = 0.0
        
        # Factor 1: Number of rows (more rows = higher confidence)
        score += min(len(table_lines) / 10.0, 0.3)
        
        # Factor 2: Consistent column structure
        column_counts = []
        for line in table_lines:
            if '|' in line:
                column_counts.append(line.count('|') + 1)
            elif '\t' in line:
                column_counts.append(line.count('\t') + 1)
            else:
                column_counts.append(len(re.split(r'  +', line.strip())))
        
        if column_counts:
            consistency = 1.0 - (max(column_counts) - min(column_counts)) / max(column_counts, 1)
            score += consistency * 0.4
        
        # Factor 3: Contains numeric data
        numeric_lines = sum(1 for line in table_lines if re.search(r'\d', line))
        score += (numeric_lines / len(table_lines)) * 0.3
        
        return min(score, 1.0)
    
    def _merge_content_with_tables(
        self, 
        text_pages: List[Dict], 
        tables: List[TableInfo]
    ) -> str:
        """
        Merge page text content with table information.
        
        Tables are inserted at appropriate positions with clear markers.
        """
        merged_content = []
        
        # Group tables by page
        tables_by_page = {}
        for table in tables:
            page = table.page_number
            if page not in tables_by_page:
                tables_by_page[page] = []
            tables_by_page[page].append(table)
        
        for page_info in text_pages:
            page_num = page_info["page_number"]
            
            # Add page header
            merged_content.append(f"\n--- Page {page_num} ---\n")
            
            # Add tables for this page first (if any)
            if page_num in tables_by_page:
                for table in tables_by_page[page_num]:
                    merged_content.append(f"\n[TABLE START - Page {page_num}]")
                    merged_content.append(table.content)
                    merged_content.append(f"[TABLE END]\n")
            
            # Add page text content
            if page_info["text"].strip():
                merged_content.append(page_info["text"])
        
        return "\n".join(merged_content)


class StructureAwareChunker:
    """
    Advanced text chunker that preserves document structure.
    
    Features:
    - Table-aware chunking (keeps tables intact)
    - Semantic boundary detection
    - Configurable overlap strategies
    - Metadata preservation
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        preserve_tables: bool = True,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables
        self.min_chunk_size = min_chunk_size
        
        # Semantic boundary patterns (in order of preference)
        self.boundary_patterns = [
            r'\n\n---.*?---\n\n',  # Page breaks
            r'\n\n',  # Paragraph breaks
            r'\. ',   # Sentence ends
            r', ',    # Comma breaks
            r' '      # Word breaks
        ]
    
    def chunk_document(
        self, 
        text: str, 
        metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """
        Split document into structure-aware chunks.
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        chunks = []
        
        if self.preserve_tables:
            # Split by table boundaries first
            segments = self._split_by_tables(text)
        else:
            segments = [text]
        
        chunk_index = 0
        
        for segment in segments:
            if self._is_table_segment(segment):
                # Keep table as single chunk (unless too large)
                table_chunks = self._chunk_table_segment(segment, metadata, chunk_index)
                chunks.extend(table_chunks)
                chunk_index += len(table_chunks)
            else:
                # Regular text chunking with overlap
                text_chunks = self._chunk_text_segment(segment, metadata, chunk_index)
                chunks.extend(text_chunks)
                chunk_index += len(text_chunks)
        
        # Add overlap between adjacent chunks
        chunks = self._add_intelligent_overlap(chunks)
        
        logger.info(f"Created {len(chunks)} structure-aware chunks")
        return chunks
    
    def _split_by_tables(self, text: str) -> List[str]:
        """Split text into segments, separating tables from regular text."""
        segments = []
        current_segment = []
        lines = text.split('\n')
        in_table = False
        
        for line in lines:
            if '[TABLE START' in line:
                # End current text segment
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = []
                # Start table segment
                current_segment.append(line)
                in_table = True
            elif '[TABLE END]' in line:
                # End table segment
                current_segment.append(line)
                segments.append('\n'.join(current_segment))
                current_segment = []
                in_table = False
            else:
                current_segment.append(line)
        
        # Add remaining content
        if current_segment:
            segments.append('\n'.join(current_segment))
        
        return [seg for seg in segments if seg.strip()]
    
    def _is_table_segment(self, segment: str) -> bool:
        """Check if segment contains a table."""
        return '[TABLE START' in segment and '[TABLE END]' in segment
    
    def _chunk_table_segment(
        self, 
        segment: str, 
        metadata: DocumentMetadata, 
        start_index: int
    ) -> List[Dict[str, Any]]:
        """Handle table segments - keep intact unless too large."""
        if len(segment) <= self.chunk_size:
            # Table fits in one chunk
            return [{
                "text": segment,
                "chunk_index": start_index,
                "chunk_type": "table",
                "is_table": True,
                "char_count": len(segment),
                **self._create_chunk_metadata(metadata)
            }]
        else:
            # Large table - split carefully
            logger.warning(f"Large table ({len(segment)} chars) being split")
            return self._split_large_table(segment, metadata, start_index)
    
    def _split_large_table(
        self, 
        segment: str, 
        metadata: DocumentMetadata, 
        start_index: int
    ) -> List[Dict[str, Any]]:
        """Split large tables while preserving structure."""
        chunks = []
        lines = segment.split('\n')
        
        # Find table boundaries
        table_start_idx = None
        table_end_idx = None
        
        for i, line in enumerate(lines):
            if '[TABLE START' in line:
                table_start_idx = i
            elif '[TABLE END]' in line:
                table_end_idx = i
                break
        
        if table_start_idx is None or table_end_idx is None:
            # Fallback to regular text chunking
            return self._chunk_text_segment(segment, metadata, start_index)
        
        # Extract table header and content
        header_lines = lines[:table_start_idx + 1]
        table_lines = lines[table_start_idx + 1:table_end_idx]
        footer_lines = lines[table_end_idx:]
        
        # Split table content into chunks
        current_chunk_lines = header_lines.copy()
        current_size = sum(len(line) + 1 for line in current_chunk_lines)
        chunk_index = start_index
        
        for line in table_lines:
            line_size = len(line) + 1
            
            if current_size + line_size + len(footer_lines[0]) > self.chunk_size:
                # Finish current chunk
                chunk_lines = current_chunk_lines + footer_lines
                chunks.append({
                    "text": '\n'.join(chunk_lines),
                    "chunk_index": chunk_index,
                    "chunk_type": "table_part",
                    "is_table": True,
                    "char_count": sum(len(l) + 1 for l in chunk_lines),
                    **self._create_chunk_metadata(metadata)
                })
                
                # Start new chunk
                current_chunk_lines = header_lines.copy()
                current_size = sum(len(line) + 1 for line in current_chunk_lines)
                chunk_index += 1
            
            current_chunk_lines.append(line)
            current_size += line_size
        
        # Add final chunk
        if len(current_chunk_lines) > len(header_lines):
            chunk_lines = current_chunk_lines + footer_lines
            chunks.append({
                "text": '\n'.join(chunk_lines),
                "chunk_index": chunk_index,
                "chunk_type": "table_part",
                "is_table": True,
                "char_count": sum(len(l) + 1 for l in chunk_lines),
                **self._create_chunk_metadata(metadata)
            })
        
        return chunks
    
    def _chunk_text_segment(
        self, 
        segment: str, 
        metadata: DocumentMetadata, 
        start_index: int
    ) -> List[Dict[str, Any]]:
        """Chunk regular text with semantic boundary awareness."""
        if len(segment) <= self.chunk_size:
            return [{
                "text": segment,
                "chunk_index": start_index,
                "chunk_type": "text",
                "is_table": False,
                "char_count": len(segment),
                **self._create_chunk_metadata(metadata)
            }]
        
        chunks = []
        remaining_text = segment
        chunk_index = start_index
        
        while remaining_text:
            if len(remaining_text) <= self.chunk_size:
                # Remaining text fits in one chunk
                chunks.append({
                    "text": remaining_text,
                    "chunk_index": chunk_index,
                    "chunk_type": "text",
                    "is_table": False,
                    "char_count": len(remaining_text),
                    **self._create_chunk_metadata(metadata)
                })
                break
            
            # Find best split point
            split_point = self._find_best_split_point(remaining_text, self.chunk_size)
            
            if split_point == -1:
                # No good split point found, force split at chunk_size
                split_point = self.chunk_size
            
            chunk_text = remaining_text[:split_point].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "chunk_type": "text",
                    "is_table": False,
                    "char_count": len(chunk_text),
                    **self._create_chunk_metadata(metadata)
                })
                chunk_index += 1
            
            # Move to next chunk with overlap
            overlap_start = max(0, split_point - self.chunk_overlap)
            remaining_text = remaining_text[overlap_start:].strip()
            
            # Avoid infinite loop
            if not remaining_text or len(remaining_text) < self.min_chunk_size:
                break
        
        return chunks
    
    def _find_best_split_point(self, text: str, max_length: int) -> int:
        """Find the best point to split text using semantic boundaries."""
        if len(text) <= max_length:
            return len(text)
        
        # Search backwards from max_length for the best boundary
        search_start = max_length
        search_end = max(max_length - 200, max_length // 2)  # Don't search too far back
        
        for pattern in self.boundary_patterns:
            for match in re.finditer(pattern, text[search_end:search_start]):
                return search_end + match.end()
        
        # No good boundary found
        return -1
    
    def _add_intelligent_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add intelligent overlap between chunks for better context continuity."""
        if len(chunks) <= 1:
            return chunks
        
        enhanced_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i].copy()
            prev_chunk = chunks[i-1]
            
            # Skip overlap for table chunks
            if current_chunk.get("is_table", False) or prev_chunk.get("is_table", False):
                enhanced_chunks.append(current_chunk)
                continue
            
            # Add contextual overlap from previous chunk
            prev_text = prev_chunk["text"]
            overlap_text = self._extract_overlap_context(prev_text, self.chunk_overlap)
            
            if overlap_text:
                current_chunk["text"] = overlap_text + "\n\n" + current_chunk["text"]
                current_chunk["char_count"] = len(current_chunk["text"])
                current_chunk["has_overlap"] = True
            
            enhanced_chunks.append(current_chunk)
        
        return enhanced_chunks
    
    def _extract_overlap_context(self, text: str, max_overlap: int) -> str:
        """Extract meaningful context for overlap from end of previous chunk."""
        if len(text) <= max_overlap:
            return text
        
        # Try to find a good boundary for overlap (complete sentences)
        overlap_start = len(text) - max_overlap
        
        # Look for sentence boundaries
        for match in re.finditer(r'\. ', text[overlap_start:]):
            return text[overlap_start + match.end():]
        
        # Fallback to word boundary
        words = text[overlap_start:].split()
        if len(words) > 1:
            return ' '.join(words[1:])  # Skip partial first word
        
        return text[overlap_start:]
    
    def _create_chunk_metadata(self, doc_metadata: DocumentMetadata) -> Dict[str, Any]:
        """Create metadata for individual chunks."""
        return {
            "filename": doc_metadata.filename,
            "file_path": doc_metadata.file_path,
            "document_type": doc_metadata.document_type,
            "total_pages": doc_metadata.total_pages,
            "has_tables": doc_metadata.has_tables,
            "content_hash": doc_metadata.content_hash
        }


class EnhancedDocumentLoader:
    """
    Production-grade document loader with extensible architecture.
    
    Features:
    - Strategy pattern for different document types
    - Robust error handling and validation
    - Memory-efficient processing
    - Comprehensive logging and metrics
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        preserve_tables: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables
        
        # Initialize processors
        self.processors = [
            PDFProcessor()
        ]
        
        # Initialize chunker
        self.chunker = StructureAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_tables=preserve_tables
        )
        
        # Supported file types
        self.supported_extensions = {'.pdf'}
    
    def load_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and process a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing processed document data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type not supported or processing fails
        """
        file_path = Path(file_path)
        
        # Validation
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Find appropriate processor
        processor = self._get_processor(file_path)
        if not processor:
            raise ValueError(f"No processor available for file type: {file_path.suffix}")
        
        # Process document
        try:
            result = processor.extract_content(file_path)
            logger.info(f"Successfully loaded document: {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {str(e)}")
            raise
    
    def load_and_chunk_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load document and split into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        # Load document
        doc_data = self.load_document(file_path)
        
        if not doc_data.get("success", False):
            raise ValueError(f"Failed to load document: {file_path}")
        
        # Extract text and metadata
        text = doc_data.get("text", "")
        metadata = doc_data.get("metadata")
        
        if not text.strip():
            logger.warning(f"No text content extracted from {file_path}")
            return []
        
        # Chunk the document
        chunks = self.chunker.chunk_document(text, metadata)
        
        logger.info(
            f"Document {metadata.filename} processed: "
            f"{len(chunks)} chunks created from {len(text):,} characters"
        )
        
        return chunks
    
    def process_uploaded_file(
        self, 
        file_content: bytes, 
        filename: str, 
        save_dir: str = "uploads"
    ) -> List[Dict[str, Any]]:
        """
        Process an uploaded file.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            save_dir: Directory to save the file
            
        Returns:
            List of document chunks
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save file temporarily
        file_path = save_path / filename
        
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"Saved uploaded file: {file_path}")
            
            # Process the saved file
            chunks = self.load_and_chunk_document(file_path)
            
            return chunks
            
        except Exception as e:
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            logger.error(f"Error processing uploaded file {filename}: {str(e)}")
            raise
    
    def _get_processor(self, file_path: Path) -> Optional[DocumentProcessor]:
        """Get the appropriate processor for the file type."""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None
    
    def get_supported_extensions(self) -> set:
        """Get set of supported file extensions."""
        return self.supported_extensions.copy()
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a file before processing.
        
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        
        validation_result = {
            "is_valid": False,
            "file_exists": False,
            "is_supported": False,
            "file_size": 0,
            "errors": []
        }
        
        # Check if file exists
        if not file_path.exists():
            validation_result["errors"].append(f"File not found: {file_path}")
            return validation_result
        
        validation_result["file_exists"] = True
        validation_result["file_size"] = file_path.stat().st_size
        
        # Check file size (limit to 100MB for safety)
        max_size = 100 * 1024 * 1024  # 100MB
        if validation_result["file_size"] > max_size:
            validation_result["errors"].append(
                f"File too large: {validation_result['file_size']:,} bytes "
                f"(max: {max_size:,} bytes)"
            )
            return validation_result
        
        # Check if supported
        if file_path.suffix.lower() not in self.supported_extensions:
            validation_result["errors"].append(
                f"Unsupported file type: {file_path.suffix}"
            )
            return validation_result
        
        validation_result["is_supported"] = True
        
        # Additional format-specific validation
        try:
            if file_path.suffix.lower() == '.pdf':
                # Try to open PDF to verify it's valid
                doc = fitz.open(str(file_path))
                if len(doc) == 0:
                    validation_result["errors"].append("PDF contains no pages")
                else:
                    validation_result["page_count"] = len(doc)
                doc.close()
        except Exception as e:
            validation_result["errors"].append(f"File validation error: {str(e)}")
            return validation_result
        
        validation_result["is_valid"] = len(validation_result["errors"]) == 0
        return validation_result


class DocumentLoaderFactory:
    """Factory for creating document loaders with different configurations."""
    
    @staticmethod
    def create_default_loader() -> EnhancedDocumentLoader:
        """Create loader with default settings."""
        return EnhancedDocumentLoader(
            chunk_size=1000,
            chunk_overlap=200,
            preserve_tables=True
        )
    
    @staticmethod
    def create_large_document_loader() -> EnhancedDocumentLoader:
        """Create loader optimized for large documents."""
        return EnhancedDocumentLoader(
            chunk_size=1500,
            chunk_overlap=150,
            preserve_tables=True
        )
    
    @staticmethod
    def create_precise_loader() -> EnhancedDocumentLoader:
        """Create loader for precise, small-chunk processing."""
        return EnhancedDocumentLoader(
            chunk_size=500,
            chunk_overlap=100,
            preserve_tables=True
        )
    
    @staticmethod
    def create_table_focused_loader() -> EnhancedDocumentLoader:
        """Create loader with enhanced table processing."""
        return EnhancedDocumentLoader(
            chunk_size=800,
            chunk_overlap=100,
            preserve_tables=True
        )


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    loader = DocumentLoaderFactory.create_default_loader()
    
    # Test file validation
    test_file = Path("test_document.pdf")
    if test_file.exists():
        validation = loader.validate_file(test_file)
        print(f"Validation result: {validation}")
        
        if validation["is_valid"]:
            # Load and process document
            try:
                chunks = loader.load_and_chunk_document(test_file)
                print(f"Created {len(chunks)} chunks")
                
                # Show first chunk as example
                if chunks:
                    first_chunk = chunks[0]
                    print(f"First chunk preview:")
                    print(f"- Type: {first_chunk.get('chunk_type', 'unknown')}")
                    print(f"- Is table: {first_chunk.get('is_table', False)}")
                    print(f"- Character count: {first_chunk.get('char_count', 0)}")
                    print(f"- Text preview: {first_chunk['text'][:200]}...")
                    
            except Exception as e:
                print(f"Error processing document: {e}")
    else:
        print("Test file not found")
