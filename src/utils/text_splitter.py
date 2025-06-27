"""Text splitting utilities for document chunking."""

from typing import List, Dict, Any
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import logger


class DocumentChunker:
    """Handles document text splitting with metadata preservation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}")
            self.encoding = None
    
    def split_text(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: The text to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for splitting")
            return []
        
        # Split the text
        chunks = self.text_splitter.split_text(text)
        
        # Prepare results with metadata
        results = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk)
            }
            
            # Add token count if available
            if self.encoding:
                chunk_data["token_count"] = len(self.encoding.encode(chunk))
            
            # Add provided metadata
            if metadata:
                chunk_data.update(metadata)
            
            results.append(chunk_data)
        
        logger.info(f"Split text into {len(results)} chunks")
        return results
    
    def split_documents(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of document dictionaries with 'text' and optional metadata
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            if "text" not in doc:
                logger.warning(f"Document {doc_idx} missing 'text' field")
                continue
            
            # Extract metadata
            metadata = {k: v for k, v in doc.items() if k != "text"}
            metadata["document_index"] = doc_idx
            
            # Split and add chunks
            chunks = self.split_text(doc["text"], metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
