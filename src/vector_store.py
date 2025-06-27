"""Vector store module for document embeddings and retrieval."""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.logger import logger
from src.utils.config import get_config


class VectorStore:
    """Manages document embeddings and similarity search using ChromaDB."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        self.config = get_config()
        self.persist_directory = persist_directory or self.config.chroma_persist_dir
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.config.openai_api_key)
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Collection name
        self.collection_name = "document_chunks"
        self.collection = None
        
    def _init_chromadb(self):
        """Initialize ChromaDB client."""
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        logger.info(f"Initialized ChromaDB with persist directory: {self.persist_directory}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def initialize_collection(self, reset: bool = False):
        """
        Initialize or get the document collection.
        
        Args:
            reset: Whether to reset the collection if it exists
        """
        try:
            if reset:
                # Delete collection if it exists
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except Exception:
                    pass
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for Q&A"}
            )
            
            logger.info(f"Initialized collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def add_documents(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int = 50
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with text and metadata
            batch_size: Number of chunks to process at once
            
        Returns:
            Number of chunks added
        """
        if not self.collection:
            self.initialize_collection()
        
        total_added = 0
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract texts and metadata
            texts = [chunk["text"] for chunk in batch]
            metadatas = []
            ids = []
            
            for j, chunk in enumerate(batch):
                # Create metadata
                metadata = {
                    k: v for k, v in chunk.items() 
                    if k != "text" and isinstance(v, (str, int, float, bool))
                }
                metadatas.append(metadata)
                
                # Create unique ID
                chunk_id = f"{chunk.get('filename', 'unknown')}_{i+j}"
                ids.append(chunk_id)
            
            # Get embeddings
            try:
                embeddings = self._get_embeddings(texts)
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_added += len(batch)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} chunks")
                
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Total chunks added to vector store: {total_added}")
        return total_added
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with text, metadata, and scores
        """
        if not self.collection:
            logger.warning("Collection not initialized")
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_embeddings([query])[0]
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0,
                        "id": results['ids'][0][i] if results['ids'] else ""
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.config.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def delete_by_filename(self, filename: str) -> int:
        """
        Delete all chunks from a specific file.
        
        Args:
            filename: Name of the file to delete chunks for
            
        Returns:
            Number of chunks deleted
        """
        if not self.collection:
            return 0
        
        try:
            # Get all IDs for the filename
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for file: {filename}")
                return len(results['ids'])
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return 0
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        if self.collection:
            try:
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = None
                logger.info("Cleared vector store collection")
            except Exception as e:
                logger.error(f"Error clearing collection: {e}")
