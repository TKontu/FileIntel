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

    def update_collection_status(
        self, collection_id: str, status: str, task_id: str = None
    ) -> bool:
        """Update collection processing status with optional task tracking.

        Args:
            collection_id: ID of the collection to update
            status: New status value (must be valid CollectionStatus)
            task_id: Optional task ID to associate with this status update

        Returns:
            True if update successful, False otherwise
        """
        try:
            from datetime import datetime
            from fileintel.storage.models import CollectionStatus

            # Define valid state transitions
            VALID_TRANSITIONS = {
                CollectionStatus.CREATED.value: [
                    CollectionStatus.PROCESSING.value,
                    CollectionStatus.FAILED.value
                ],
                CollectionStatus.PROCESSING.value: [
                    CollectionStatus.COMPLETED.value,
                    CollectionStatus.FAILED.value
                ],
                CollectionStatus.COMPLETED.value: [
                    CollectionStatus.PROCESSING.value  # Allow reprocessing
                ],
                CollectionStatus.FAILED.value: [
                    CollectionStatus.PROCESSING.value  # Allow retry
                ],
            }

            # Use row-level locking to prevent race conditions
            from fileintel.storage.models import Collection

            collection = (
                self.base.db.query(Collection)
                .filter(Collection.id == collection_id)
                .with_for_update()  # Lock row for update
                .first()
            )
            if not collection:
                logger.warning(
                    f"Collection {collection_id} not found for status update"
                )
                return False

            # Validate status is a valid enum value
            valid_statuses = [s.value for s in CollectionStatus]
            if status not in valid_statuses:
                logger.error(
                    f"Invalid status '{status}'. Must be one of: {valid_statuses}"
                )
                return False

            # Validate state transition is allowed
            current_status = collection.processing_status
            allowed_transitions = VALID_TRANSITIONS.get(current_status, [])
            if status not in allowed_transitions and status != current_status:
                logger.warning(
                    f"Invalid status transition: {current_status} -> {status}. "
                    f"Allowed transitions: {allowed_transitions}"
                )
                return False

            collection.processing_status = status
            collection.status_updated_at = datetime.utcnow()

            if task_id:
                collection.current_task_id = task_id

                # Append to task history
                history = collection.task_history or []
                history.append({
                    "task_id": task_id,
                    "status": status,
                    "timestamp": datetime.utcnow().isoformat()
                })
                collection.task_history = history

            self.base._safe_commit()
            logger.info(
                f"Updated collection {collection_id} status to {status}"
                + (f" with task {task_id}" if task_id else "")
            )
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
        file_path: str,
        original_filename: str = None,
        metadata: Dict[str, Any] = None,
        content_fingerprint: str = None,
    ) -> Document:
        """Create a new document (without collection association)."""
        try:
            # Validate inputs
            filename = self.base._validate_input_security(filename, "filename")
            content_hash = self.base._validate_input_security(
                content_hash, "content_hash"
            )
            mime_type = self.base._validate_input_security(mime_type, "mime_type")
            file_path = self.base._validate_input_security(file_path, "file_path")

            if original_filename:
                original_filename = self.base._validate_input_security(
                    original_filename, "original_filename"
                )

            import uuid

            document_id = str(uuid.uuid4())

            document = Document(
                id=document_id,
                filename=filename,
                content_hash=content_hash,
                content_fingerprint=content_fingerprint,
                file_size=file_size,
                mime_type=mime_type,
                file_path=file_path,
                original_filename=original_filename,
                document_metadata=metadata or {},
            )

            self.db.add(document)
            self.base._safe_commit()
            return document

        except Exception as e:
            self.base._handle_session_error(e)

    def add_document_to_collection(self, document_id: str, collection_id: str) -> bool:
        """Add a document to a collection (many-to-many relationship)."""
        try:
            document = self.get_document(document_id)
            collection = self.get_collection(collection_id)

            if not document or not collection:
                logger.warning(f"Document {document_id} or collection {collection_id} not found")
                return False

            # Check if association already exists
            if collection in document.collections:
                logger.debug(f"Document {document_id} already in collection {collection_id}")
                return True

            # Add collection to document's collections
            document.collections.append(collection)
            self.base._safe_commit()
            logger.info(f"Added document {document_id} to collection {collection_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document to collection: {e}")
            self.base._handle_session_error(e)
            return False

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
        """Get document by hash within a specific collection (via many-to-many)."""
        from .models import collection_documents

        return (
            self.db.query(Document)
            .join(collection_documents, Document.id == collection_documents.c.document_id)
            .filter(
                and_(
                    Document.content_hash == content_hash,
                    collection_documents.c.collection_id == collection_id,
                )
            )
            .first()
        )

    def get_document_by_filename_and_collection(
        self, filename: str, collection_id: str
    ) -> Document:
        """Get document by filename within a collection (via many-to-many)."""
        from .models import collection_documents

        return (
            self.db.query(Document)
            .join(collection_documents, Document.id == collection_documents.c.document_id)
            .filter(
                and_(
                    Document.filename == filename,
                    collection_documents.c.collection_id == collection_id,
                )
            )
            .first()
        )

    def get_document_by_original_filename_and_collection(
        self, original_filename: str, collection_id: str
    ) -> Document:
        """Get document by original filename within a collection (via many-to-many)."""
        from .models import collection_documents

        return (
            self.db.query(Document)
            .join(collection_documents, Document.id == collection_documents.c.document_id)
            .filter(
                and_(
                    Document.original_filename == original_filename,
                    collection_documents.c.collection_id == collection_id,
                )
            )
            .first()
        )

    def get_document_by_fingerprint(
        self, fingerprint: str, collection_id: str = None
    ) -> Document:
        """
        Get document by content fingerprint.

        Args:
            fingerprint: Content fingerprint UUID (deterministic, content-based)
            collection_id: Optional collection filter (for scoped lookup)

        Returns:
            Document if found, None otherwise

        Examples:
            # Global lookup (any collection)
            doc = storage.get_document_by_fingerprint("8f3d2c1b-...")

            # Scoped lookup (specific collection)
            doc = storage.get_document_by_fingerprint("8f3d2c1b-...", collection_id="abc123")
        """
        query = self.db.query(Document).filter(
            Document.content_fingerprint == fingerprint
        )

        if collection_id:
            # Scoped to specific collection (via many-to-many)
            from .models import collection_documents
            query = query.join(collection_documents, Document.id == collection_documents.c.document_id).filter(
                collection_documents.c.collection_id == collection_id
            )

        return query.first()

    def get_all_documents_by_fingerprint(self, fingerprint: str) -> List[Document]:
        """
        Get all documents with this fingerprint across all collections.

        Useful for:
        - Checking if content exists anywhere in system
        - Finding duplicate uploads in different collections
        - Tracking content reuse

        Args:
            fingerprint: Content fingerprint UUID

        Returns:
            List of Documents (may be empty if not found)

        Example:
            docs = storage.get_all_documents_by_fingerprint("8f3d2c1b-...")
            if docs:
                print(f"This content exists in {len(docs)} collections")
        """
        return (
            self.db.query(Document)
            .filter(Document.content_fingerprint == fingerprint)
            .all()
        )

    def get_documents_by_collection(self, collection_id: str) -> List[Document]:
        """Get all documents in a collection (via many-to-many relationship)."""
        from .models import collection_documents

        return (
            self.db.query(Document)
            .join(collection_documents, Document.id == collection_documents.c.document_id)
            .filter(collection_documents.c.collection_id == collection_id)
            .all()
        )

    def delete_document(self, document_id: str):
        """Delete a document."""
        document = self.get_document(document_id)
        if document:
            self.db.delete(document)
            self.base._safe_commit()

    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any], replace: bool = False):
        """Update document metadata by merging with existing metadata.

        This method merges new metadata with existing metadata, preserving
        any existing fields that are not being updated. This prevents data
        loss when updating metadata from different sources (e.g., file metadata
        and LLM-extracted metadata).

        Args:
            document_id: The document ID to update
            metadata: New metadata to merge or replace
            replace: If True, completely replace existing metadata instead of merging
        """
        document = self.get_document(document_id)
        if document:
            if replace:
                # Complete replacement - use for force re-extraction
                document.document_metadata = metadata
            else:
                # Merge new metadata with existing, preserving existing fields
                existing = document.document_metadata or {}
                document.document_metadata = {**existing, **metadata}
            self.base._safe_commit()

    # Chunk Operations
    def add_document_chunks(
        self, document_id: str, chunks: List[Dict], chunk_metadata: Dict = None
    ) -> List[DocumentChunk]:
        """Add chunks for a document (global chunks, not per-collection)."""
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
                position=i,
                chunk_text=chunk_text,
                chunk_metadata=metadata,
            )

            self.db.add(chunk)
            chunk_objects.append(chunk)

        self.base._safe_commit()
        return chunk_objects

    def get_chunk_by_id(self, chunk_id: str):
        """Get a single chunk by its UUID."""
        return self.db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()

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
        """Get all chunks for documents in a collection (via many-to-many relationship)."""
        from .models import collection_documents

        chunks = (
            self.db.query(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .join(collection_documents, Document.id == collection_documents.c.document_id)
            .filter(collection_documents.c.collection_id == collection_id)
            .order_by(DocumentChunk.document_id, DocumentChunk.position)
            .all()
        )
        return chunks

    def get_chunks_by_type_for_collection(self, collection_id: str, chunk_type: str = None):
        """Get chunks for documents in a collection, filtered by chunk type (via many-to-many relationship)."""
        from .models import collection_documents

        query = (
            self.db.query(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .join(collection_documents, Document.id == collection_documents.c.document_id)
            .filter(collection_documents.c.collection_id == collection_id)
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
