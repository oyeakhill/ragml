"""Enhanced Question-Answering engine with advanced RAG capabilities and document lifecycle management."""

from typing import List, Dict, Any, Optional, Generator, Tuple
from enum import Enum
from dataclasses import dataclass
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import re
import os
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

from src.utils.logger import logger
from src.utils.config import get_config
from src.vector_store import VectorStore
from src.utils.text_splitter import DocumentChunker
from src.document_manager import (
    DocumentManager,
    JsonDocumentRepository,
    DocumentStatus
)


class QueryType(Enum):
    """Types of queries for intelligent routing."""
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    SYNTHESIS = "synthesis"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


@dataclass
class QueryAnalysis:
    """Results of query analysis."""
    query_type: QueryType
    entities: List[str]
    required_context: List[str]
    complexity_score: float
    multi_hop: bool


@dataclass
class ValidationResult:
    """Results of answer validation."""
    is_complete: bool
    confidence_level: str  # High, Medium, Low
    missing_information: List[str]
    contradictions: List[str]
    suggestions: List[str]


class ComparisonQueryHandler:
    """Integrated comparison query handling."""
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        self.comparison_patterns = [
            r'which (?:is|are) better',
            r'compare (?:the )?(\w+) (?:and|vs|versus|with) (\w+)',
            r'what(?:\'s| is) the difference between',
            r'(\w+) or (\w+)',
            r'(\w+) vs\.? (\w+)',
            r'better.*(\w+).*(\w+)',
            r'comparison (?:of|between)',
            r'pros and cons'
        ]
    
    def is_comparison_query(self, query: str) -> bool:
        """Check if query is asking for comparison."""
        query_lower = query.lower()
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                return True
        comparison_words = ['better', 'worse', 'superior', 'inferior', 'versus', 
                           'compare', 'comparison', 'difference', 'vs', 'or']
        return any(word in query_lower for word in comparison_words)
    
    def extract_comparison_entities(self, query: str) -> List[str]:
        """Extract entities being compared from the query."""
        entities = []
        query_lower = query.lower()
        
        patterns = [
            r'compare (?:the )?(\w+) (?:and|vs|versus|with) (\w+)',
            r'(\w+) (?:or|vs\.?|versus) (\w+)',
            r'between (\w+) and (\w+)',
            r'is (\w+) better than (\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities.extend(match.groups())
                break
        
        if not entities:
            model_pattern = r'\b([A-Z][A-Z0-9]+)\b'
            matches = re.findall(model_pattern, query)
            entities.extend(matches)
        
        entities = [e.strip() for e in entities if e and len(e) > 1]
        return list(dict.fromkeys(entities))


class StructuredContentChunker:
    """Chunker that preserves document structure."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_with_structure(self, text: str, tables: List[Dict], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text while preserving tables and structure."""
        chunks = []
        parts = self._split_by_tables(text)
        
        for part in parts:
            if "[TABLE START]" in part and "[TABLE END]" in part:
                chunk_data = {
                    "text": part,
                    "chunk_index": len(chunks),
                    "is_table": True,
                    "chunk_type": "table",
                    **metadata
                }
                chunks.append(chunk_data)
            else:
                text_chunks = self._split_text_intelligently(part)
                for text_chunk in text_chunks:
                    chunk_data = {
                        "text": text_chunk,
                        "chunk_index": len(chunks),
                        "is_table": False,
                        "chunk_type": "text",
                        **metadata
                    }
                    chunks.append(chunk_data)
        
        return self._add_overlap_context(chunks)
    
    def _split_by_tables(self, text: str) -> List[str]:
        """Split text by table markers."""
        parts = []
        current_part = []
        lines = text.split('\n')
        
        for line in lines:
            if "[TABLE START]" in line:
                if current_part:
                    parts.append('\n'.join(current_part))
                    current_part = []
                current_part.append(line)
            elif "[TABLE END]" in line:
                current_part.append(line)
                parts.append('\n'.join(current_part))
                current_part = []
            else:
                current_part.append(line)
        
        if current_part:
            parts.append('\n'.join(current_part))
        
        return [p for p in parts if p.strip()]
    
    def _split_text_intelligently(self, text: str) -> List[str]:
        """Split text preserving semantic units."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _add_overlap_context(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add overlap context between chunks."""
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            if chunk.get("is_table", False):
                enhanced_chunks.append(chunk)
                continue
            
            # Add context from previous chunk
            if i > 0 and not chunks[i-1].get("is_table", False):
                prev_text = chunks[i-1]["text"]
                overlap_text = prev_text[-self.chunk_overlap:] if len(prev_text) > self.chunk_overlap else prev_text
                chunk["text"] = overlap_text + "\n\n" + chunk["text"]
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks


class EnhancedDocumentLoader:
    """Consolidated enhanced document loader."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = {'.pdf'}
        self.structured_chunker = StructuredContentChunker(chunk_size, chunk_overlap)
    
    def load_pdf(self, file_path: str) -> Dict[str, Any]:
        """Enhanced PDF loading with table extraction."""
        try:
            doc = fitz.open(file_path)
            text_pages = []
            all_tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = self._extract_text_with_layout(page)
                tables = self._extract_tables_from_page(page, page_num + 1)
                all_tables.extend(tables)
                
                if text.strip() or tables:
                    text_pages.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "has_tables": len(tables) > 0,
                        "table_count": len(tables)
                    })
            
            doc.close()
            full_text = self._merge_content_with_tables(text_pages, all_tables)
            
            file_path_obj = Path(file_path)
            metadata = {
                "filename": file_path_obj.name,
                "file_path": str(file_path_obj.absolute()),
                "total_pages": len(text_pages),
                "file_size": file_path_obj.stat().st_size,
                "document_type": "pdf",
                "has_tables": len(all_tables) > 0,
                "table_count": len(all_tables)
            }
            
            logger.info(f"Successfully loaded PDF: {file_path_obj.name} "
                       f"({len(text_pages)} pages, {len(all_tables)} tables)")
            
            return {
                "text": full_text,
                "pages": text_pages,
                "tables": all_tables,
                **metadata
            }
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to load PDF: {str(e)}")
    
    def _extract_text_with_layout(self, page) -> str:
        """Extract text while preserving layout."""
        text = page.get_text("text", sort=True)
        blocks = page.get_text("blocks")
        
        if blocks:
            structured_text = []
            for block in blocks:
                if block[6] == 0:  # Text block
                    block_text = block[4].strip()
                    if block_text:
                        structured_text.append(block_text)
            if structured_text:
                text = "\n\n".join(structured_text)
        return text
    
    def _extract_tables_from_page(self, page, page_num: int) -> List[Dict]:
        """Extract tables from PDF page."""
        tables = []
        text = page.get_text()
        
        # Simple table detection based on patterns
        lines = text.split('\n')
        table_lines = []
        
        for line in lines:
            # Check for table-like patterns
            if ('|' in line or '\t' in line or 
                re.search(r'  {2,}', line) or 
                re.match(r'^[A-Za-z\s]+:\s+.+', line)):
                table_lines.append(line)
            elif table_lines and len(table_lines) >= 2:
                table_content = '\n'.join(table_lines)
                tables.append({
                    "content": table_content,
                    "page_number": page_num
                })
                table_lines = []
        
        return tables
    
    def _merge_content_with_tables(self, text_pages: List[Dict], tables: List[Dict]) -> str:
        """Merge text content with tables."""
        merged_content = []
        tables_by_page = {}
        
        for table in tables:
            page = table["page_number"]
            if page not in tables_by_page:
                tables_by_page[page] = []
            tables_by_page[page].append(table)
        
        for page_info in text_pages:
            page_num = page_info["page_number"]
            merged_content.append(f"\n--- Page {page_num} ---\n")
            
            if page_num in tables_by_page:
                for table in tables_by_page[page_num]:
                    merged_content.append("\n[TABLE START]\n")
                    merged_content.append(table["content"])
                    merged_content.append("\n[TABLE END]\n")
            
            merged_content.append(page_info["text"])
        
        return "\n".join(merged_content)
    
    def load_and_chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and chunk document with structure awareness."""
        doc_data = self.load_document(file_path)
        chunk_metadata = {
            "filename": doc_data["filename"],
            "file_path": doc_data["file_path"],
            "document_type": doc_data["document_type"],
            "has_tables": doc_data.get("has_tables", False)
        }
        
        chunks = self.structured_chunker.split_with_structure(
            doc_data["text"], 
            doc_data.get("tables", []),
            chunk_metadata
        )
        
        logger.info(f"Document {doc_data['filename']} split into {len(chunks)} chunks")
        return chunks
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load document based on file type."""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path_obj.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        if extension == '.pdf':
            return self.load_pdf(file_path)
        raise ValueError(f"File type {extension} not implemented")
    
    def process_uploaded_file(self, file_content: bytes, filename: str, save_dir: str = "uploads") -> List[Dict[str, Any]]:
        """Process uploaded file."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        file_path = save_path / filename
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        try:
            chunks = self.load_and_chunk_document(str(file_path))
            return chunks
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise e


class QueryClassifier:
    """Classifies queries and extracts key information."""
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        
    def classify_query(self, query: str) -> QueryAnalysis:
        """
        Classify the query type and extract key information.
        
        Args:
            query: User query
            
        Returns:
            QueryAnalysis object
        """
        classification_prompt = f"""Analyze this query and provide a JSON response with the following structure:
{{
    "query_type": "factual|comparative|analytical|synthesis|temporal|causal",
    "entities": ["list", "of", "key", "entities"],
    "required_context": ["types", "of", "information", "needed"],
    "complexity_score": 0.0-1.0,
    "multi_hop": true/false
}}

Query: {query}

Analysis:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a query analysis expert. Respond only with valid JSON."},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            analysis_text = response.choices[0].message.content
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
            
            return QueryAnalysis(
                query_type=QueryType(analysis_data.get("query_type", "factual")),
                entities=analysis_data.get("entities", []),
                required_context=analysis_data.get("required_context", []),
                complexity_score=float(analysis_data.get("complexity_score", 0.5)),
                multi_hop=analysis_data.get("multi_hop", False)
            )
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Return default analysis
            return QueryAnalysis(
                query_type=QueryType.FACTUAL,
                entities=[],
                required_context=[],
                complexity_score=0.5,
                multi_hop=False
            )


class IntelligentRetriever:
    """Advanced retrieval strategies for different query types."""
    
    def __init__(self, vector_store: VectorStore, openai_client: OpenAI):
        self.vector_store = vector_store
        self.openai_client = openai_client
        
    def multi_stage_retrieval(
        self, 
        query: str, 
        query_analysis: QueryAnalysis,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-stage retrieval based on query analysis.
        
        Args:
            query: User query
            query_analysis: Analysis of the query
            k: Number of results to retrieve
            
        Returns:
            List of relevant chunks
        """
        all_results = []
        seen_texts = set()
        
        # Stage 1: Direct semantic retrieval
        direct_results = self.vector_store.search(query, k=k)
        for result in direct_results:
            text_hash = hash(result['text'])
            if text_hash not in seen_texts:
                all_results.append(result)
                seen_texts.add(text_hash)
        
        # Stage 2: Entity-based retrieval
        for entity in query_analysis.entities[:3]:  # Limit to top 3 entities
            entity_results = self.vector_store.search(entity, k=2)
            for result in entity_results:
                text_hash = hash(result['text'])
                if text_hash not in seen_texts:
                    all_results.append(result)
                    seen_texts.add(text_hash)
        
        # Stage 3: Context expansion for multi-hop queries
        if query_analysis.multi_hop and all_results:
            # Generate follow-up queries based on initial results
            context_queries = self._generate_context_queries(query, all_results[:3])
            for ctx_query in context_queries[:2]:  # Limit expansion
                ctx_results = self.vector_store.search(ctx_query, k=2)
                for result in ctx_results:
                    text_hash = hash(result['text'])
                    if text_hash not in seen_texts:
                        all_results.append(result)
                        seen_texts.add(text_hash)
        
        # Re-rank results based on relevance
        ranked_results = self._rerank_results(query, all_results, query_analysis)
        
        return ranked_results[:k*2]  # Return more than k for context assembly
    
    def _generate_context_queries(
        self, 
        original_query: str, 
        initial_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate follow-up queries for context expansion."""
        context = "\n".join([r['text'][:200] for r in initial_results])
        
        prompt = f"""Based on this query and initial context, generate 2-3 follow-up search queries that would help answer the original question more completely.

Original query: {original_query}

Initial context:
{context}

Follow-up queries (one per line):"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate relevant follow-up search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            queries = response.choices[0].message.content.strip().split('\n')
            return [q.strip() for q in queries if q.strip()][:3]
            
        except Exception as e:
            logger.error(f"Error generating context queries: {e}")
            return []
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        query_analysis: QueryAnalysis
    ) -> List[Dict[str, Any]]:
        """Re-rank results based on query type and relevance."""
        # Simple re-ranking based on entity presence
        for result in results:
            score = result.get('distance', 1.0)
            text_lower = result['text'].lower()
            
            # Boost score for entity matches
            entity_boost = sum(1 for entity in query_analysis.entities 
                             if entity.lower() in text_lower)
            
            # Adjust score based on query type
            if query_analysis.query_type == QueryType.COMPARATIVE:
                # Boost chunks that contain comparison words
                comparison_words = ['compare', 'versus', 'vs', 'difference', 'similar', 'unlike']
                comparison_boost = sum(1 for word in comparison_words if word in text_lower)
                score -= comparison_boost * 0.1
            
            result['adjusted_score'] = score - (entity_boost * 0.2)
        
        # Sort by adjusted score (lower is better)
        return sorted(results, key=lambda x: x.get('adjusted_score', x.get('distance', 1.0)))


class ManagedIntelligentRetriever(IntelligentRetriever):
    """
    Enhanced retriever that uses document manager for filtered search.
    
    This ensures we only search within properly managed documents
    and never return results from deleted or failed documents.
    """
    
    def __init__(self, document_manager: DocumentManager, openai_client: OpenAI):
        self.document_manager = document_manager
        self.openai_client = openai_client
        # We don't call super().__init__ as we're replacing vector_store with document_manager
    
    def multi_stage_retrieval(
        self, 
        query: str, 
        query_analysis: QueryAnalysis,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-stage retrieval with document filtering.
        
        This ensures we only retrieve from active, properly indexed documents.
        """
        all_results = []
        seen_texts = set()
        
        # Stage 1: Direct semantic retrieval
        direct_results = self.document_manager.search_with_filtering(
            query=query,
            k=k
        )
        
        for result in direct_results:
            text_hash = hash(result['text'])
            if text_hash not in seen_texts:
                all_results.append(result)
                seen_texts.add(text_hash)
        
        # Stage 2: Entity-based retrieval
        for entity in query_analysis.entities[:3]:
            entity_results = self.document_manager.search_with_filtering(
                query=entity,
                k=2
            )
            
            for result in entity_results:
                text_hash = hash(result['text'])
                if text_hash not in seen_texts:
                    all_results.append(result)
                    seen_texts.add(text_hash)
        
        # Stage 3: Context expansion for multi-hop queries
        if query_analysis.multi_hop and all_results:
            context_queries = self._generate_context_queries(query, all_results[:3])
            
            for ctx_query in context_queries[:2]:
                ctx_results = self.document_manager.search_with_filtering(
                    query=ctx_query,
                    k=2
                )
                
                for result in ctx_results:
                    text_hash = hash(result['text'])
                    if text_hash not in seen_texts:
                        all_results.append(result)
                        seen_texts.add(text_hash)
        
        # Re-rank results
        ranked_results = self._rerank_results(query, all_results, query_analysis)
        
        return ranked_results[:k*2]


class ContextAssembler:
    """Intelligent context assembly with quality assessment."""
    
    def __init__(self, openai_client: OpenAI, max_context_length: int = 3000):
        self.openai_client = openai_client
        self.max_context_length = max_context_length
        
    def assemble_context(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        query_analysis: QueryAnalysis
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Assemble context intelligently based on query requirements.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            query_analysis: Query analysis results
            
        Returns:
            Tuple of (assembled context, selected chunks)
        """
        # Remove redundancy
        unique_chunks = self._remove_redundancy(chunks)
        
        # Assess completeness
        completeness_score = self._assess_completeness(query, unique_chunks, query_analysis)
        
        # Select and order chunks based on query type
        selected_chunks = self._select_chunks(unique_chunks, query_analysis, completeness_score)
        
        # Format context
        context = self._format_context(selected_chunks, query_analysis)
        
        return context, selected_chunks
    
    def _remove_redundancy(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove redundant chunks using similarity threshold."""
        unique_chunks = []
        seen_content = []
        
        for chunk in chunks:
            # Simple deduplication - could be enhanced with semantic similarity
            is_duplicate = False
            for seen in seen_content:
                if self._calculate_overlap(chunk['text'], seen) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_content.append(chunk['text'])
        
        return unique_chunks
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """Calculate text overlap ratio."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _assess_completeness(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        query_analysis: QueryAnalysis
    ) -> float:
        """Assess if retrieved context is sufficient to answer the query."""
        if not chunks:
            return 0.0
        
        # Check if required entities are present
        all_text = " ".join([c['text'].lower() for c in chunks])
        entities_found = sum(1 for entity in query_analysis.entities 
                           if entity.lower() in all_text)
        
        entity_coverage = entities_found / len(query_analysis.entities) if query_analysis.entities else 1.0
        
        # Check for required context types
        context_found = sum(1 for ctx_type in query_analysis.required_context
                          if any(ctx_type.lower() in c['text'].lower() for c in chunks))
        
        context_coverage = context_found / len(query_analysis.required_context) if query_analysis.required_context else 1.0
        
        # Combined score
        completeness = (entity_coverage + context_coverage) / 2
        
        return completeness
    
    def _select_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        query_analysis: QueryAnalysis,
        completeness_score: float
    ) -> List[Dict[str, Any]]:
        """Select chunks based on query type and completeness."""
        selected = []
        current_length = 0
        
        # If completeness is low, include more chunks
        target_chunks = 5 if completeness_score < 0.5 else 3
        
        for chunk in chunks[:target_chunks]:
            chunk_length = len(chunk['text'])
            if current_length + chunk_length <= self.max_context_length:
                selected.append(chunk)
                current_length += chunk_length
            else:
                # Truncate the last chunk if needed
                remaining_space = self.max_context_length - current_length
                if remaining_space > 100:  # Only include if meaningful
                    chunk['text'] = chunk['text'][:remaining_space] + "..."
                    selected.append(chunk)
                break
        
        return selected
    
    def _format_context(
        self, 
        chunks: List[Dict[str, Any]], 
        query_analysis: QueryAnalysis
    ) -> str:
        """Format context based on query type."""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        
        # Add header based on query type
        if query_analysis.query_type == QueryType.COMPARATIVE:
            context_parts.append("COMPARATIVE CONTEXT - Multiple perspectives found:\n")
        elif query_analysis.query_type == QueryType.TEMPORAL:
            context_parts.append("TEMPORAL CONTEXT - Time-related information:\n")
        
        # Format chunks
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            chunk_index = metadata.get("chunk_index", "?")
            
            context_parts.append(
                f"[Source {i}: {filename} (chunk {chunk_index})]\n{chunk['text']}\n"
            )
        
        return "\n---\n".join(context_parts)


class AnswerValidator:
    """Validates and enhances answer quality."""
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        
    def validate_answer(
        self, 
        query: str, 
        context: str, 
        answer: str,
        query_analysis: QueryAnalysis
    ) -> ValidationResult:
        """
        Validate the generated answer.
        
        Args:
            query: Original query
            context: Retrieved context
            answer: Generated answer
            query_analysis: Query analysis results
            
        Returns:
            ValidationResult
        """
        validation_prompt = f"""Analyze this Q&A interaction and provide a JSON assessment:

Query: {query}
Query Type: {query_analysis.query_type.value}

Context provided:
{context[:1000]}...

Answer given:
{answer}

Provide JSON with:
{{
    "is_complete": true/false,
    "confidence_level": "High|Medium|Low",
    "missing_information": ["list of missing elements"],
    "contradictions": ["list of contradictions if any"],
    "suggestions": ["improvements if needed"]
}}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an answer quality validator. Respond only with valid JSON."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            validation_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', validation_text, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
            
            return ValidationResult(
                is_complete=validation_data.get("is_complete", True),
                confidence_level=validation_data.get("confidence_level", "Medium"),
                missing_information=validation_data.get("missing_information", []),
                contradictions=validation_data.get("contradictions", []),
                suggestions=validation_data.get("suggestions", [])
            )
            
        except Exception as e:
            logger.error(f"Error validating answer: {e}")
            return ValidationResult(
                is_complete=True,
                confidence_level="Medium",
                missing_information=[],
                contradictions=[],
                suggestions=[]
            )


def enhance_qa_engine_with_comparison(qa_engine):
    """Enhance QA engine with comparison capabilities."""
    comparison_handler = ComparisonQueryHandler(qa_engine.openai_client)
    original_answer_question = qa_engine.answer_question
    
    def enhanced_answer_question(question: str, use_rag: bool = True, stream: bool = False, validate: bool = True) -> Dict[str, Any]:
        """Enhanced answer method with comparison detection."""
        if use_rag and comparison_handler.is_comparison_query(question):
            logger.info("Detected comparison query, using specialized handler")
            entities = comparison_handler.extract_comparison_entities(question)
            
            # Enhanced retrieval for comparison
            all_results = []
            for entity in entities[:3]:  # Limit entities
                results = qa_engine.document_manager.search_with_filtering(query=entity, k=5)
                all_results.extend(results)
            
            # Format context for comparison
            context_str = "\n".join([f"[Source]: {r['text'][:300]}..." for r in all_results[:5]])
            
            # Enhanced prompt for comparison
            messages = [
                {"role": "system", "content": "You are a technical expert specializing in detailed comparisons. Always provide specific numbers and specifications when available."},
                {"role": "user", "content": f"Compare the following based on this context:\n\nContext: {context_str}\n\nQuestion: {question}"}
            ]
            
            if stream:
                return {
                    "success": True,
                    "stream": qa_engine._stream_answer(messages),
                    "sources": all_results[:5],
                    "comparison_entities": entities
                }
            else:
                response = qa_engine._call_llm(messages)
                return {
                    "success": True,
                    "answer": response.choices[0].message.content,
                    "sources": all_results[:5],
                    "comparison_entities": entities,
                    "query_type": "comparison"
                }
        else:
            return original_answer_question(question, use_rag, stream, validate)
    
    qa_engine.answer_question = enhanced_answer_question
    qa_engine.comparison_handler = comparison_handler
    return qa_engine


class EnhancedQAEngine:
    """Enhanced QA Engine with intelligent retrieval, validation, and document lifecycle management."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """Initialize the enhanced QA engine with document management."""
        self.config = get_config()
        self.openai_client = OpenAI(api_key=self.config.openai_api_key)
        self.vector_store = vector_store or VectorStore()
        
        # Use integrated enhanced document loader
        self.document_loader = EnhancedDocumentLoader(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Initialize document management
        repository_path = Path(self.config.chroma_persist_dir) / "document_metadata"
        self.document_repository = JsonDocumentRepository(repository_path)
        self.document_manager = DocumentManager(
            repository=self.document_repository,
            vector_store=self.vector_store
        )
        
        # Initialize components
        self.query_classifier = QueryClassifier(self.openai_client)
        self.intelligent_retriever = ManagedIntelligentRetriever(
            self.document_manager, self.openai_client
        )
        self.context_assembler = ContextAssembler(self.openai_client)
        self.answer_validator = AnswerValidator(self.openai_client)
        
        # Enhanced system prompt
        self.system_prompt = """You are an intelligent assistant. When answering questions:

1. Read and understand the full context provided
2. Give direct, specific answers based on the document
3. Make logical inferences when needed
4. Connect related information across different parts of the document
5. Be comprehensive but concise

Focus on being helpful and intelligent in your responses."""
        
        # Enhance with comparison capabilities
        enhance_qa_engine_with_comparison(self)
        
        # Verify system integrity on startup (but don't fail if empty)
        try:
            integrity_check = self.verify_system_integrity()
            if not integrity_check["is_healthy"] and integrity_check.get("total_documents", 0) > 0:
                logger.warning(f"System integrity issues found: {integrity_check['issues']}")
        except Exception as e:
            logger.warning(f"Could not verify system integrity on startup: {e}")
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load and index a document with proper lifecycle management.
        
        This method ensures:
        1. Duplicate documents are detected by content hash
        2. Updated documents replace old versions atomically
        3. Failed operations don't leave partial state
        """
        try:
            # Load and chunk the document
            chunks = self.document_loader.load_and_chunk_document(file_path)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No content extracted from document"
                }
            
            # Extract filename
            filename = Path(file_path).name
            
            # Use document manager for atomic operation
            result = self.document_manager.add_document(filename, chunks)
            
            # Log operation result
            if result["success"]:
                action = result.get("action", "unknown")
                if action == "unchanged":
                    logger.info(f"Document {filename} already up to date")
                elif action == "added":
                    logger.info(f"Successfully indexed document {filename}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def load_uploaded_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Load and index an uploaded file with lifecycle management.
        
        This ensures proper handling of document updates and prevents
        stale references in the vector store.
        """
        try:
            # Process the uploaded file
            chunks = self.document_loader.process_uploaded_file(
                file_content, 
                filename
            )
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No content extracted from uploaded file"
                }
            
            # Use document manager for atomic operation
            result = self.document_manager.add_document(filename, chunks)
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading uploaded file {filename}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Delete a document with proper cleanup."""
        return self.document_manager.delete_document(filename)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """Get list of all indexed documents."""
        return self.document_manager.list_documents()
    
    def get_document_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document."""
        return self.document_manager.get_document_info(filename)
    
    def verify_system_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire system."""
        return self.document_manager.verify_integrity()
    
    def answer_question(
        self, 
        question: str, 
        use_rag: bool = True,
        stream: bool = False,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using enhanced RAG and LLM.
        
        Args:
            question: User question
            use_rag: Whether to use RAG retrieval
            stream: Whether to stream the response
            validate: Whether to validate the answer
            
        Returns:
            Answer with metadata
        """
        try:
            # Step 1: Analyze the query
            query_analysis = self.query_classifier.classify_query(question)
            logger.info(f"Query classified as: {query_analysis.query_type.value}")
            
            context_chunks = []
            context_str = ""
            
            if use_rag:
                # Step 2: Intelligent retrieval
                retrieved_chunks = self.intelligent_retriever.multi_stage_retrieval(
                    question, 
                    query_analysis,
                    k=self.config.top_k_results
                )
                
                # Step 3: Assemble context
                context_str, context_chunks = self.context_assembler.assemble_context(
                    question,
                    retrieved_chunks,
                    query_analysis
                )
            
            # Step 4: Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            if use_rag and context_str:
                user_message = f"""Context from document:
{context_str}

Question: {question}"""
            else:
                user_message = question
            
            messages.append({"role": "user", "content": user_message})
            
            # Step 5: Generate answer
            if stream:
                return {
                    "success": True,
                    "stream": self._stream_answer(messages),
                    "sources": context_chunks,
                    "query_analysis": query_analysis
                }
            else:
                response = self._call_llm(messages)
                answer = response.choices[0].message.content
                
                # Step 6: Validate answer (optional)
                validation_result = None
                if validate and use_rag:
                    validation_result = self.answer_validator.validate_answer(
                        question,
                        context_str,
                        answer,
                        query_analysis
                    )
                
                return {
                    "success": True,
                    "answer": answer,
                    "sources": context_chunks,
                    "model": self.config.llm_model,
                    "used_rag": use_rag,
                    "query_analysis": query_analysis,
                    "validation": validation_result
                }
                
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {"success": False, "error": str(e)}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_llm(self, messages: List[Dict[str, str]], stream: bool = False) -> Any:
        """Call the OpenAI LLM with retry logic."""
        return self.openai_client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens,
            stream=stream
        )
    
    def _stream_answer(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Stream the LLM response."""
        try:
            stream = self._call_llm(messages, stream=True)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield f"\n\nError: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced system statistics including document management."""
        vector_stats = self.vector_store.get_collection_stats()
        
        # Add document management stats
        doc_stats = {
            "total_documents": len(self.document_manager.list_documents()),
            "repository_type": type(self.document_repository).__name__,
            "integrity_check": self.document_manager.verify_integrity()["is_healthy"]
        }
        
        return {
            "vector_store": vector_stats,
            "document_management": doc_stats,
            "config": {
                "llm_model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "top_k_results": self.config.top_k_results
            },
            "enhanced_features": {
                "query_classification": True,
                "multi_stage_retrieval": True,
                "context_quality_assessment": True,
                "answer_validation": True,
                "document_lifecycle_management": True
            }
        }
    
    def clear_documents(self):
        """
        Clear all documents with proper cleanup.
        
        This ensures both vector store and metadata are cleared atomically.
        """
        # Get all documents
        documents = self.document_manager.list_documents()
        
        # Delete each document properly
        for doc in documents:
            self.document_manager.delete_document(doc["filename"])
        
        # Clear vector store collection
        self.vector_store.clear_collection()
        
        # Reinitialize collection
        self.vector_store.initialize_collection(reset=True)
        
        logger.info("Cleared all documents from the system")
