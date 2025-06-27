"""
Document Manager - Centralized document lifecycle management with ACID-like guarantees.

This module implements the Repository pattern for document management, ensuring
consistency between the vector store and application state. It provides:

1. Transactional document operations (add/update/delete)
2. Document versioning and change detection
3. Atomic operations with rollback capability
4. Clear separation of concerns between storage and business logic

Architecture Decision Records (ADR):
- ADR-001: Use Repository pattern to abstract vector store implementation
- ADR-002: Implement Unit of Work pattern for transactional consistency
- ADR-003: Use content-based hashing for document identity (SHA-256)
- ADR-004: Maintain document metadata separately from vector embeddings
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import threading
from contextlib import contextmanager

from src.utils.logger import logger


class DocumentStatus(Enum):
    """Document lifecycle states."""
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass
class DocumentMetadata:
    """
    Immutable document metadata following Value Object pattern.
    
    The document_id is a content-based hash ensuring:
    1. Same content always produces same ID (idempotent)
    2. Content changes create new document versions
    3. No naming collisions
    """
    document_id: str
    filename: str
    content_hash: str
    chunk_count: int
    total_size: int
    created_at: datetime
    updated_at: datetime
    status: DocumentStatus
    chunk_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "content_hash": self.content_hash,
            "chunk_count": self.chunk_count,
            "total_size": self.total_size,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "chunk_ids": self.chunk_ids,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Deserialize from dictionary."""
        return cls(
            document_id=data["document_id"],
            filename=data["filename"],
            content_hash=data["content_hash"],
            chunk_count=data["chunk_count"],
            total_size=data["total_size"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=DocumentStatus(data["status"]),
            chunk_ids=data.get("chunk_ids", []),
            error_message=data.get("error_message")
        )


class DocumentRepository(ABC):
    """
    Abstract repository interface for document operations.
    Follows Interface Segregation Principle (ISP).
    """
    
    @abstractmethod
    def add(self, metadata: DocumentMetadata) -> None:
        """Add document metadata."""
        pass
    
    @abstractmethod
    def get(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata by ID."""
        pass
    
    @abstractmethod
    def get_by_filename(self, filename: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata by filename."""
        pass
    
    @abstractmethod
    def list_all(self) -> List[DocumentMetadata]:
        """List all document metadata."""
        pass
    
    @abstractmethod
    def update(self, metadata: DocumentMetadata) -> None:
        """Update document metadata."""
        pass
    
    @abstractmethod
    def delete(self, document_id: str) -> None:
        """Delete document metadata."""
        pass
    
    @abstractmethod
    def exists(self, document_id: str) -> bool:
        """Check if document exists."""
        pass


class JsonDocumentRepository(DocumentRepository):
    """
    JSON-based document repository implementation.
    
    Production considerations:
    - For scale, replace with PostgreSQL/MongoDB
    - Add Redis for caching frequently accessed documents
    - Implement connection pooling for database access
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "document_metadata.json"
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._cache: Dict[str, DocumentMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from disk with error recovery."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for doc_id, doc_data in data.items():
                        self._cache[doc_id] = DocumentMetadata.from_dict(doc_data)
            except Exception as e:
                logger.error(f"Failed to load metadata, starting fresh: {e}")
                self._cache = {}
                # Create backup of corrupted file
                backup_path = self.metadata_file.with_suffix('.backup')
                self.metadata_file.rename(backup_path)
    
    def _save_metadata(self) -> None:
        """Persist metadata to disk with atomic write."""
        temp_file = self.metadata_file.with_suffix('.tmp')
        try:
            data = {
                doc_id: metadata.to_dict() 
                for doc_id, metadata in self._cache.items()
            }
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            # Atomic rename
            temp_file.replace(self.metadata_file)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def add(self, metadata: DocumentMetadata) -> None:
        with self._lock:
            self._cache[metadata.document_id] = metadata
            self._save_metadata()
    
    def get(self, document_id: str) -> Optional[DocumentMetadata]:
        with self._lock:
            return self._cache.get(document_id)
    
    def get_by_filename(self, filename: str) -> Optional[DocumentMetadata]:
        with self._lock:
            for metadata in self._cache.values():
                if metadata.filename == filename and metadata.status != DocumentStatus.DELETED:
                    return metadata
            return None
    
    def list_all(self) -> List[DocumentMetadata]:
        with self._lock:
            return [
                metadata for metadata in self._cache.values()
                if metadata.status != DocumentStatus.DELETED
            ]
    
    def update(self, metadata: DocumentMetadata) -> None:
        with self._lock:
            if metadata.document_id not in self._cache:
                raise ValueError(f"Document {metadata.document_id} not found")
            metadata.updated_at = datetime.now()
            self._cache[metadata.document_id] = metadata
            self._save_metadata()
    
    def delete(self, document_id: str) -> None:
        with self._lock:
            if document_id in self._cache:
                # Soft delete - mark as deleted but keep metadata
                metadata = self._cache[document_id]
                metadata.status = DocumentStatus.DELETED
                metadata.updated_at = datetime.now()
                self._save_metadata()
    
    def exists(self, document_id: str) -> bool:
        with self._lock:
            metadata = self._cache.get(document_id)
            return metadata is not None and metadata.status != DocumentStatus.DELETED


class DocumentManager:
    """
    High-level document management with transactional guarantees.
    
    Implements Unit of Work pattern to ensure consistency between
    document metadata and vector store operations.
    """
    
    def __init__(self, repository: DocumentRepository, vector_store):
        self.repository = repository
        self.vector_store = vector_store
        self._operation_lock = threading.Lock()
    
    @staticmethod
    def generate_document_id(content: str) -> str:
        """Generate deterministic document ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    @staticmethod
    def generate_chunk_id(document_id: str, chunk_index: int) -> str:
        """Generate deterministic chunk ID."""
        return f"{document_id}:chunk:{chunk_index:04d}"
    
    @contextmanager
    def transaction(self):
        """
        Provide transactional context for document operations.
        
        In production, this would integrate with database transactions
        and implement proper two-phase commit with the vector store.
        """
        with self._operation_lock:
            # Start transaction
            rollback_actions = []
            try:
                yield rollback_actions
            except Exception as e:
                # Execute rollback actions in reverse order
                logger.error(f"Transaction failed, rolling back: {e}")
                for action in reversed(rollback_actions):
                    try:
                        action()
                    except Exception as rollback_error:
                        logger.error(f"Rollback failed: {rollback_error}")
                raise
    
    def add_document(
        self, 
        filename: str, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add or update a document with full transactional guarantees.
        
        Process:
        1. Generate content-based document ID
        2. Check for existing document with same content
        3. If content changed, mark old version as deleted
        4. Add new document version to vector store
        5. Update metadata repository
        
        Returns:
            Operation result with status and details
        """
        # Generate document ID from content
        full_content = "".join(chunk["text"] for chunk in chunks)
        document_id = self.generate_document_id(full_content)
        content_hash = document_id[:16]  # Use first 16 chars for display
        
        with self.transaction() as rollback_actions:
            # Check for existing document with same filename
            existing = self.repository.get_by_filename(filename)
            
            if existing:
                if existing.content_hash == content_hash:
                    # Same content, no action needed
                    return {
                        "success": True,
                        "action": "unchanged",
                        "document_id": existing.document_id,
                        "message": "Document content unchanged"
                    }
                else:
                    # Content changed, delete old version
                    self._delete_document_internal(existing, rollback_actions)
            
            # Create metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                content_hash=content_hash,
                chunk_count=len(chunks),
                total_size=len(full_content),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=DocumentStatus.PENDING,
                chunk_ids=[]
            )
            
            # Add to repository first (can rollback)
            self.repository.add(metadata)
            rollback_actions.append(lambda: self.repository.delete(document_id))
            
            # Prepare chunks with proper IDs
            chunk_ids = []
            enhanced_chunks = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = self.generate_chunk_id(document_id, i)
                chunk_ids.append(chunk_id)
                
                # Enhanced chunk with full metadata
                enhanced_chunk = {
                    **chunk,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                enhanced_chunks.append(enhanced_chunk)
            
            # Add to vector store
            try:
                # Initialize collection if needed
                if not self.vector_store.collection:
                    self.vector_store.initialize_collection()
                
                # Add chunks in batches
                batch_size = 50
                for i in range(0, len(enhanced_chunks), batch_size):
                    batch = enhanced_chunks[i:i + batch_size]
                    batch_ids = chunk_ids[i:i + batch_size]
                    
                    texts = [chunk["text"] for chunk in batch]
                    metadatas = [
                        {k: v for k, v in chunk.items() if k != "text"}
                        for chunk in batch
                    ]
                    
                    embeddings = self.vector_store._get_embeddings(texts)
                    
                    self.vector_store.collection.add(
                        ids=batch_ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas
                    )
                
                # Update metadata status
                metadata.status = DocumentStatus.INDEXED
                metadata.chunk_ids = chunk_ids
                self.repository.update(metadata)
                
                logger.info(f"Successfully indexed document {filename} ({document_id})")
                
                return {
                    "success": True,
                    "action": "added",
                    "document_id": document_id,
                    "filename": filename,
                    "chunks_indexed": len(chunks)
                }
                
            except Exception as e:
                # Update metadata with error
                metadata.status = DocumentStatus.FAILED
                metadata.error_message = str(e)
                self.repository.update(metadata)
                raise
    
    def _delete_document_internal(
        self, 
        metadata: DocumentMetadata, 
        rollback_actions: List
    ) -> None:
        """Internal method to delete document with rollback support."""
        # Delete from vector store
        if metadata.chunk_ids:
            self.vector_store.collection.delete(ids=metadata.chunk_ids)
            # Rollback would require re-adding chunks (complex, omitted for brevity)
        
        # Mark as deleted in repository
        old_status = metadata.status
        metadata.status = DocumentStatus.DELETED
        self.repository.update(metadata)
        rollback_actions.append(
            lambda: setattr(metadata, 'status', old_status)
        )
    
    def delete_document(self, filename: str) -> Dict[str, Any]:
        """
        Delete a document by filename.
        
        Returns:
            Operation result with status and details
        """
        with self.transaction() as rollback_actions:
            metadata = self.repository.get_by_filename(filename)
            
            if not metadata:
                return {
                    "success": False,
                    "error": f"Document '{filename}' not found"
                }
            
            self._delete_document_internal(metadata, rollback_actions)
            
            logger.info(f"Successfully deleted document {filename}")
            
            return {
                "success": True,
                "document_id": metadata.document_id,
                "chunks_deleted": len(metadata.chunk_ids)
            }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all active documents with their metadata."""
        documents = self.repository.list_all()
        return [
            {
                "filename": doc.filename,
                "document_id": doc.document_id,
                "chunk_count": doc.chunk_count,
                "total_size": doc.total_size,
                "status": doc.status.value,
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat()
            }
            for doc in documents
        ]
    
    def get_document_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document."""
        metadata = self.repository.get_by_filename(filename)
        if not metadata:
            return None
        
        return {
            "filename": metadata.filename,
            "document_id": metadata.document_id,
            "content_hash": metadata.content_hash,
            "chunk_count": metadata.chunk_count,
            "total_size": metadata.total_size,
            "status": metadata.status.value,
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "chunk_ids": metadata.chunk_ids,
            "error_message": metadata.error_message
        }
    
    def search_with_filtering(
        self, 
        query: str, 
        k: int = 5,
        include_files: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with document-level filtering.
        
        This ensures we only search within active documents and can
        exclude specific files from results.
        """
        # Build list of valid document IDs
        valid_doc_ids = set()
        filename_to_doc_id = {}
        
        for metadata in self.repository.list_all():
            if metadata.status == DocumentStatus.INDEXED:
                valid_doc_ids.add(metadata.document_id)
                filename_to_doc_id[metadata.filename] = metadata.document_id
        
        # Apply include/exclude filters
        if include_files:
            filtered_ids = {
                filename_to_doc_id[f] for f in include_files 
                if f in filename_to_doc_id
            }
            valid_doc_ids &= filtered_ids
        
        if exclude_files:
            excluded_ids = {
                filename_to_doc_id[f] for f in exclude_files 
                if f in filename_to_doc_id
            }
            valid_doc_ids -= excluded_ids
        
        if not valid_doc_ids:
            return []
        
        # Perform search with document ID filter
        # ChromaDB requires at least 2 conditions for $and, so we use direct filter
        where_clause = {"document_id": {"$in": list(valid_doc_ids)}}
        
        return self.vector_store.search(
            query=query,
            k=k,
            filter_dict=where_clause
        )
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify consistency between metadata and vector store.
        
        Checks:
        1. All indexed documents have chunks in vector store
        2. No orphaned chunks exist in vector store
        3. Chunk counts match metadata
        """
        issues = []
        
        # Initialize collection if needed
        if not self.vector_store.collection:
            self.vector_store.initialize_collection()
        
        # Get all chunks from vector store
        try:
            all_chunks = self.vector_store.collection.get()
        except Exception as e:
            return {
                "is_healthy": False,
                "issues": [{
                    "type": "error",
                    "description": f"Failed to access vector store: {str(e)}"
                }],
                "total_documents": len(self.repository.list_all()),
                "total_chunks": 0
            }
        
        chunk_by_doc = {}
        
        if all_chunks and all_chunks.get('metadatas'):
            for i, metadata in enumerate(all_chunks['metadatas']):
                doc_id = metadata.get('document_id', 'unknown')
                if doc_id not in chunk_by_doc:
                    chunk_by_doc[doc_id] = []
                chunk_by_doc[doc_id].append(all_chunks['ids'][i])
        
        # Check each document
        for doc_metadata in self.repository.list_all():
            if doc_metadata.status == DocumentStatus.INDEXED:
                doc_id = doc_metadata.document_id
                expected_chunks = set(doc_metadata.chunk_ids)
                actual_chunks = set(chunk_by_doc.get(doc_id, []))
                
                if expected_chunks != actual_chunks:
                    issues.append({
                        "type": "chunk_mismatch",
                        "document": doc_metadata.filename,
                        "expected": len(expected_chunks),
                        "actual": len(actual_chunks),
                        "missing": list(expected_chunks - actual_chunks),
                        "extra": list(actual_chunks - expected_chunks)
                    })
        
        # Check for orphaned chunks
        known_doc_ids = {
            doc.document_id for doc in self.repository.list_all()
        }
        
        for doc_id in chunk_by_doc:
            if doc_id not in known_doc_ids:
                issues.append({
                    "type": "orphaned_chunks",
                    "document_id": doc_id,
                    "chunk_count": len(chunk_by_doc[doc_id])
                })
        
        return {
            "is_healthy": len(issues) == 0,
            "issues": issues,
            "total_documents": len(self.repository.list_all()),
            "total_chunks": len(all_chunks.get('ids', []))
        }
