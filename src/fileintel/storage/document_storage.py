"""
Document storage operations for collections, documents, and chunks.

Handles core CRUD operations for the document management system
without vector search or GraphRAG functionality.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import and_
from .base_storage import BaseStorageInfrastructure
from .models import Collection, Document, DocumentChunk

logger = logging.getLogger(__name__)


class DocumentStorage:
    """
    Handles core document storage operations.

    Manages collections, documents, and chunks without complex
    search or GraphRAG functionality.
    """

    def __init__(self, config_or_session):
        """Initialize with shared infrastructure."""
        self.base = BaseStorageInfrastructure(config_or_session)
        self.db = self.base.db

    # Collection Operations
    def create_collection(self, name: str, description: str = None) -> Collection:
        """Create a new collection."""
        name = self.base._validate_collection_name(name)

        if description:
            description = self.base._validate_input_security(description, "description")

        try:
            import uuid

            collection_id = str(uuid.uuid4())
            collection = Collection(
                id=collection_id, name=name, description=description
            )
            self.db.add(collection)
            self.base._safe_commit()
            return collection
        except Exception as e:
            self.base._handle_session_error(e)

    def get_collection(self, collection_id: str) -> Collection:
        """Get collection by ID."""
        return self.db.query(Collection).filter(Collection.id == collection_id).first()

    def get_collection_by_name(self, name: str) -> Collection:
        """Get collection by name."""
        return self.db.query(Collection).filter(Collection.name == name).first()

    def get_all_collections(self) -> List[Collection]:
        """Get all collections."""
        return self.db.query(Collection).all()

    def delete_collection(self, collection_id: str):
        """Delete a collection and all associated data."""
        collection = self.get_collection(collection_id)
        if collection:
            self.db.delete(collection)
            self.base._safe_commit()

    def update_collection_status(self, collection_id: str, status: str) -> bool:
        """Update collection processing status."""
        try:
            collection = self.get_collection(collection_id)
            if not collection:
                logger.warning(
                    f"Collection {collection_id} not found for status update"
                )
                return False

            collection.processing_status = status
            self.base._safe_commit()
            logger.info(f"Updated collection {collection_id} status to {status}")
            return True

        except Exception as e:
            logger.error(f"Error updating collection status: {e}")
            self.base._handle_session_error(e)
            return False

    # Document Operations
    def create_document(
        self,
        filename: str,
        content_hash: str,
        file_size: int,
        mime_type: str,
        collection_id: str,
        file_path: str = None,
        original_filename: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Document:
        """Create a new document."""
        try:
            # Validate inputs
            filename = self.base._validate_input_security(filename, "filename")
            content_hash = self.base._validate_input_security(
                content_hash, "content_hash"
            )
            mime_type = self.base._validate_input_security(mime_type, "mime_type")

            if file_path:
                file_path = self.base._validate_input_security(file_path, "file_path")
            if original_filename:
                original_filename = self.base._validate_input_security(
                    original_filename, "original_filename"
                )

            import uuid

            document_id = str(uuid.uuid4())

            # Store file_path in metadata if provided
            doc_metadata = metadata or {}
            if file_path:
                doc_metadata["file_path"] = file_path

            document = Document(
                id=document_id,
                filename=filename,
                content_hash=content_hash,
                file_size=file_size,
                mime_type=mime_type,
                collection_id=collection_id,
                original_filename=original_filename,
                document_metadata=doc_metadata,
            )

            self.db.add(document)
            self.base._safe_commit()
            return document

        except Exception as e:
            self.base._handle_session_error(e)

    def get_document(self, document_id: str) -> Document:
        """Get document by ID."""
        return self.db.query(Document).filter(Document.id == document_id).first()

    def get_document_by_hash(self, content_hash: str) -> Document:
        """Get document by content hash."""
        return (
            self.db.query(Document)
            .filter(Document.content_hash == content_hash)
            .first()
        )

    def get_document_by_hash_and_collection(
        self, content_hash: str, collection_id: str
    ) -> Document:
        """Get document by hash within a specific collection."""
        return (
            self.db.query(Document)
            .filter(
                and_(
                    Document.content_hash == content_hash,
                    Document.collection_id == collection_id,
                )
            )
            .first()
        )

    def get_document_by_filename_and_collection(
        self, filename: str, collection_id: str
    ) -> Document:
        """Get document by filename within a collection."""
        return (
            self.db.query(Document)
            .filter(
                and_(
                    Document.filename == filename,
                    Document.collection_id == collection_id,
                )
            )
            .first()
        )

    def get_document_by_original_filename_and_collection(
        self, original_filename: str, collection_id: str
    ) -> Document:
        """Get document by original filename within a collection."""
        return (
            self.db.query(Document)
            .filter(
                and_(
                    Document.original_filename == original_filename,
                    Document.collection_id == collection_id,
                )
            )
            .first()
        )

    def get_documents_by_collection(self, collection_id: str) -> List[Document]:
        """Get all documents in a collection."""
        return (
            self.db.query(Document)
            .filter(Document.collection_id == collection_id)
            .all()
        )

    def delete_document(self, document_id: str):
        """Delete a document."""
        document = self.get_document(document_id)
        if document:
            self.db.delete(document)
            self.base._safe_commit()

    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]):
        """Update document metadata."""
        document = self.get_document(document_id)
        if document:
            document.document_metadata = metadata
            self.base._safe_commit()

    # Chunk Operations
    def add_document_chunks(
        self, document_id: str, chunks_or_collection_id, chunk_metadata_or_chunks=None
    ) -> List[DocumentChunk]:
        """Add chunks for a document."""
        # Handle backward compatibility with old signature:
        # add_document_chunks(document_id, collection_id, chunks)
        # vs new signature:
        # add_document_chunks(document_id, chunks, chunk_metadata)

        if isinstance(chunks_or_collection_id, str):
            # Old signature: chunks_or_collection_id is collection_id
            collection_id = chunks_or_collection_id
            chunks = chunk_metadata_or_chunks
            chunk_metadata = None
        else:
            # New signature: chunks_or_collection_id is chunks
            chunks = chunks_or_collection_id
            chunk_metadata = chunk_metadata_or_chunks

        document = self.get_document(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")

        chunk_objects = []
        for i, chunk_data in enumerate(chunks):
            chunk_text = self.base._clean_text(chunk_data.get("text", ""))

            if not chunk_text.strip():
                continue  # Skip empty chunks

            import uuid

            chunk_id = str(uuid.uuid4())

            # Store additional chunk data in metadata
            metadata = chunk_metadata or {}

            # Preserve ALL metadata from chunking process (pages, extraction_methods, etc.)
            chunk_meta = chunk_data.get("metadata", {})
            if chunk_meta:
                metadata.update(chunk_meta)

            # Add token and position info (may override chunk_meta if present)
            metadata.update(
                {
                    "token_count": chunk_data.get("token_count", 0),
                    "start_char": chunk_data.get("start_char", 0),
                    "end_char": chunk_data.get("end_char", 0),
                }
            )

            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                collection_id=collection_id,
                position=i,
                chunk_text=chunk_text,
                chunk_metadata=metadata,
            )

            self.db.add(chunk)
            chunk_objects.append(chunk)

        self.base._safe_commit()
        return chunk_objects

    def get_all_chunks_for_document(self, document_id: str):
        """Get all chunks for a document."""
        chunks = (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.position)
            .all()
        )
        return chunks

    def get_all_chunks_for_collection(self, collection_id: str):
        """Get all chunks for a collection."""
        chunks = (
            self.db.query(DocumentChunk)
            .join(Document)
            .filter(Document.collection_id == collection_id)
            .order_by(DocumentChunk.document_id, DocumentChunk.position)
            .all()
        )
        return chunks

    def get_chunks_by_type_for_collection(self, collection_id: str, chunk_type: str = None):
        """Get chunks for a collection filtered by chunk type."""
        query = (
            self.db.query(DocumentChunk)
            .join(Document)
            .filter(Document.collection_id == collection_id)
        )

        if chunk_type:
            # Filter by chunk_type in metadata using JSON operations
            query = query.filter(
                DocumentChunk.chunk_metadata.op('->')('chunk_type').astext == chunk_type
            )

        chunks = query.order_by(DocumentChunk.document_id, DocumentChunk.position).all()
        return chunks

    def update_chunk_embedding(self, chunk_id: str, embedding: List[float]):
        """Update chunk with embedding vector."""
        chunk = (
            self.db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
        )
        if chunk:
            chunk.embedding = embedding
            self.base._safe_commit()
            return True
        return False

    def close(self):
        """Close the storage connection."""
        self.base.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
