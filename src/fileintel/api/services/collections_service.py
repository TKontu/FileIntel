"""
Collections business logic service.

Extracts complex business logic from API route handlers to improve
maintainability and follow the Single Responsibility Principle.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.storage.models import Collection, Document
from fileintel.core.config import get_config
from fileintel.core.validation import validate_file_upload, FileValidationError
from fileintel.tasks.workflow_tasks import (
    complete_collection_analysis,
    incremental_collection_update,
)
from fileintel.tasks.document_tasks import process_document, process_collection

logger = logging.getLogger(__name__)


class CollectionsService:
    """Service class for collections business logic."""

    def __init__(self, storage: PostgreSQLStorage):
        """Initialize with storage dependency."""
        self.storage = storage
        self.config = get_config()

    def create_collection(
        self, name: str, description: Optional[str] = None
    ) -> Collection:
        """Create a new collection."""
        return self.storage.create_collection(name, description)

    def get_all_collections(self) -> List[Dict[str, Any]]:
        """Get all collections with formatted data."""
        collections = self.storage.get_all_collections()
        return [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "status": getattr(c, "processing_status", "unknown"),
                "status_description": self._get_status_description(
                    getattr(c, "processing_status", "unknown")
                ),
                "created_at": c.created_at.isoformat()
                if hasattr(c, "created_at") and c.created_at
                else None,
                "updated_at": c.updated_at.isoformat()
                if hasattr(c, "updated_at") and c.updated_at
                else None,
            }
            for c in collections
        ]

    def get_collection_details(self, collection: Collection) -> Dict[str, Any]:
        """Get detailed collection information."""
        documents = self.storage.get_documents_by_collection(collection.id)

        return {
            "id": collection.id,
            "name": collection.name,
            "description": collection.description,
            "status": getattr(collection, "processing_status", "unknown"),
            "status_description": self._get_status_description(
                getattr(collection, "processing_status", "unknown")
            ),
            "document_count": len(documents),
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "original_filename": doc.original_filename,
                    "file_size": doc.file_size,
                    "mime_type": doc.mime_type,
                }
                for doc in documents
            ],
            "created_at": collection.created_at.isoformat()
            if hasattr(collection, "created_at") and collection.created_at
            else None,
            "updated_at": collection.updated_at.isoformat()
            if hasattr(collection, "updated_at") and collection.updated_at
            else None,
        }

    async def upload_document_to_collection(
        self, collection: Collection, file, file_content: bytes
    ) -> Dict[str, Any]:
        """Handle document upload to collection (non-blocking)."""
        import asyncio
        import hashlib
        import aiofiles

        # Validate file upload
        validate_file_upload(file)

        # Create unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        unique_filename = f"{file_id}{file_extension}"
        file_path = Path(self.config.paths.uploads) / unique_filename

        # Ensure upload directory exists (run in thread to avoid blocking)
        await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)

        # Save file asynchronously using aiofiles (prevents event loop blocking)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)

        # Calculate file hash in thread pool (CPU-bound operation)
        def _calculate_hash():
            return hashlib.sha256(file_content).hexdigest()

        content_hash = await asyncio.to_thread(_calculate_hash)

        # Store document in database
        # Note: Keeping synchronous for now to avoid SQLAlchemy session thread-safety issues
        # The session is created in the request thread and shouldn't be accessed from other threads
        document = self.storage.create_document(
            filename=unique_filename,
            content_hash=content_hash,
            file_size=len(file_content),
            mime_type=file.content_type or "application/octet-stream",
            file_path=str(file_path),
            original_filename=file.filename,
        )

        # Link document to collection
        self.storage.add_document_to_collection(document.id, collection.id)

        logger.info(f"Document {document.id} uploaded and linked to collection {collection.id}")

        return {
            "document_id": document.id,
            "filename": document.filename,
            "original_filename": document.original_filename,
            "file_size": document.file_size,
            "collection_id": collection.id,
            "file_path": str(file_path),
        }

    def submit_collection_processing_task(
        self,
        collection_id: str,
        include_embeddings: bool = True,
        task_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit collection processing task."""
        collection = self.storage.get_collection(collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")

        documents = self.storage.get_documents_by_collection(collection_id)
        if not documents:
            raise ValueError(f"No documents found in collection {collection_id}")

        # Update collection status
        self.storage.update_collection_status(collection_id, "processing")

        # Get file paths from documents (file_path is required field)
        file_paths = []
        for doc in documents:
            if not doc.file_path:
                raise ValueError(f"Document {doc.id} missing file_path")
            file_paths.append(doc.file_path)

        # Submit task
        task_result = complete_collection_analysis.delay(
            collection_id=collection_id,
            file_paths=file_paths,
            generate_embeddings=include_embeddings,
            **(task_options or {}),
        )

        estimated_duration = self._estimate_processing_time(
            len(documents), include_embeddings
        )

        logger.info(
            f"Submitted collection processing task {task_result.id} for collection {collection_id}"
        )

        return {
            "task_id": task_result.id,
            "collection_id": collection_id,
            "include_embeddings": include_embeddings,
            "document_count": len(documents),
            "estimated_duration_seconds": estimated_duration,
            "status": "submitted",
        }

    def submit_incremental_update_task(
        self, collection_id: str, new_documents_only: bool = True
    ) -> Dict[str, Any]:
        """Submit incremental collection update task."""
        collection = self.storage.get_collection(collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")

        # Submit incremental update task
        task_result = incremental_collection_update.delay(
            collection_id=collection_id, new_documents_only=new_documents_only
        )

        documents = self.storage.get_documents_by_collection(collection_id)
        estimated_duration = self._estimate_incremental_time(len(documents))

        logger.info(
            f"Submitted incremental update task {task_result.id} for collection {collection_id}"
        )

        return {
            "task_id": task_result.id,
            "collection_id": collection_id,
            "update_type": "incremental",
            "new_documents_only": new_documents_only,
            "estimated_duration_seconds": estimated_duration,
            "status": "submitted",
        }

    def get_processing_status(self, collection_id: str) -> Dict[str, Any]:
        """Get collection processing status."""
        collection = self.storage.get_collection(collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")

        status = getattr(collection, "processing_status", "unknown")
        documents = self.storage.get_documents_by_collection(collection_id)

        # Get embedding statistics
        embedding_stats = self.storage.get_embedding_statistics(collection_id)

        return {
            "collection_id": collection_id,
            "status": status,
            "status_description": self._get_status_description(status),
            "document_count": len(documents),
            "embedding_statistics": embedding_stats,
            "last_updated": collection.updated_at.isoformat()
            if hasattr(collection, "updated_at") and collection.updated_at
            else None,
        }

    def _get_status_description(self, status: str) -> str:
        """Get human-readable description for collection processing status."""
        status_descriptions = {
            "created": "Collection created, ready for processing",
            "processing": "Document processing in progress",
            "processing_with_embeddings": "Processing documents and generating embeddings",
            "processing_documents": "Processing documents only",
            "processing_embeddings": "Generating embeddings for processed documents",
            "completed": "All processing completed successfully",
            "failed": "Processing failed, check logs for details",
        }
        return status_descriptions.get(status, f"Unknown status: {status}")

    def _estimate_processing_time(
        self, document_count: int, include_embeddings: bool
    ) -> int:
        """Estimate processing time for documents."""
        # Import constants from error_handlers
        from fileintel.api.error_handlers import (
            DEFAULT_TASK_ESTIMATION_SECONDS_PER_DOCUMENT,
            INCREMENTAL_TASK_ESTIMATION_SECONDS_PER_DOCUMENT,
        )

        base_time = document_count * DEFAULT_TASK_ESTIMATION_SECONDS_PER_DOCUMENT
        if include_embeddings:
            base_time *= 2  # Embeddings take approximately 2x longer
        return base_time

    def _estimate_incremental_time(self, document_count: int) -> int:
        """Estimate incremental processing time."""
        from fileintel.api.error_handlers import (
            INCREMENTAL_TASK_ESTIMATION_SECONDS_PER_DOCUMENT,
        )

        return document_count * INCREMENTAL_TASK_ESTIMATION_SECONDS_PER_DOCUMENT
