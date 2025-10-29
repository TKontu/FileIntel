"""
Business logic services for API operations.

Separates business logic from HTTP request handling to improve testability
and follow separation of concerns principle.
"""

import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import uuid

try:
    import aiofiles

    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
from fastapi import HTTPException, UploadFile

from ..storage.base import DocumentStorageInterface
from ..storage.models import Collection, Document
from ..core.config import get_config
from .validators import validate_content_type
from .error_handlers import APIErrorHandler
import re


class DocumentUploadService:
    """
    Service for handling document upload operations.

    Encapsulates file upload business logic separately from HTTP request handling.
    """

    def __init__(self, storage: DocumentStorageInterface):
        self.storage = storage

    async def save_uploaded_file(
        self, file: UploadFile, upload_dir: str, max_file_size: int
    ) -> Tuple[str, str, int]:
        """
        Save uploaded file to disk with security measures.

        Returns:
            tuple: (file_path, content_hash, file_size)
        """
        os.makedirs(upload_dir, exist_ok=True)

        # Generate secure filename
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1]
        secure_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(upload_dir, secure_filename)

        # Stream file to disk with hash calculation
        file_hash = hashlib.sha256()
        file_size = 0

        if not HAS_AIOFILES:
            raise HTTPException(
                status_code=500,
                detail="aiofiles package not available for async file operations",
            )

        try:
            async with aiofiles.open(file_path, "wb") as f:
                while contents := await file.read(1024 * 1024):  # 1MB chunks
                    file_size += len(contents)

                    if file_size > max_file_size:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        raise APIErrorHandler.file_too_large(max_file_size)

                    await f.write(contents)
                    file_hash.update(contents)
        except HTTPException:
            raise
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise APIErrorHandler.internal_error(f"Failed to save file: {e}")

        return file_path, file_hash.hexdigest(), file_size

    def check_duplicate_document(
        self, content_hash: str, collection_id: str, file_path: str
    ) -> Optional[Document]:
        """
        Check for duplicate documents and clean up if found.

        Returns:
            Document: Existing document if duplicate found, None otherwise
        """
        existing_doc = self.storage.get_document_by_hash_and_collection(
            content_hash, collection_id
        )
        if existing_doc:
            if os.path.exists(file_path):
                os.remove(file_path)
        return existing_doc

    def create_document_record(
        self,
        secure_filename: str,
        original_filename: str,
        content_hash: str,
        file_size: int,
        file_path: str,
        collection_id: str,
    ) -> Document:
        """Create database record for uploaded document."""
        mime_type, _ = mimetypes.guess_type(original_filename)
        if mime_type is None:
            mime_type = "application/octet-stream"

        # Create document without collection association
        document = self.storage.create_document(
            filename=secure_filename,
            original_filename=original_filename,
            content_hash=content_hash,
            file_size=file_size,
            mime_type=mime_type,
            file_path=file_path,
            metadata={},
        )

        # Link document to collection
        self.storage.add_document_to_collection(document.id, collection_id)

        return document

    async def upload_single_document(
        self, file: UploadFile, collection: Collection
    ) -> Dict[str, Any]:
        """
        Complete single document upload workflow.

        Returns:
            dict: Upload result with document_id and file_path
        """
        config = get_config()
        # Parse file size string like '100MB' to bytes
        size_str = config.document_processing.max_file_size.upper()
        match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?B)$", size_str)
        if not match:
            raise ValueError(f"Invalid file size format: {size_str}")
        size, unit = match.groups()
        multipliers = {
            "B": 1,
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
            "TB": 1024**4,
        }
        max_file_size = int(float(size) * multipliers[unit])

        supported_formats = config.document_processing.supported_formats

        # Validate file
        validate_content_type(file, supported_formats)

        # Save file securely
        file_path, content_hash, file_size = await self.save_uploaded_file(
            file, config.paths.uploads, max_file_size
        )

        # Check for duplicates
        existing_doc = self.check_duplicate_document(
            content_hash, collection.id, file_path
        )
        if existing_doc:
            raise APIErrorHandler.duplicate_document(existing_doc.id)

        # Create database record
        document = self.create_document_record(
            os.path.basename(file_path),
            file.filename,
            content_hash,
            file_size,
            file_path,
            collection.id,
        )

        return {
            "document_id": document.id,
            "file_path": file_path,
            "filename": file.filename,
        }

    def validate_batch_upload_request(self, files: List[UploadFile]) -> None:
        """Validate batch upload request."""
        if not files:
            raise APIErrorHandler.validation_error("No files were uploaded.")

    async def process_single_file_for_batch(
        self,
        file: UploadFile,
        collection: Collection,
        upload_dir: str,
        max_file_size: int,
        supported_formats: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
        """
        Process single file for batch upload.

        Returns:
            tuple: (processed_file_info or None, skip_info or None)
        """
        try:
            validate_content_type(file, supported_formats)

            # Save file using existing helper
            file_path, content_hash, file_size = await self.save_uploaded_file(
                file, upload_dir, max_file_size
            )

            # Check for duplicates - return skip info if found
            existing_doc = self.check_duplicate_document(
                content_hash, collection.id, file_path
            )
            if existing_doc:
                return None, {
                    "filename": file.filename,
                    "reason": f"Document with same content already exists with ID: {existing_doc.id}",
                }

            # Create database record
            document = self.create_document_record(
                os.path.basename(file_path),
                file.filename,
                content_hash,
                file_size,
                file_path,
                collection.id,
            )

            return {
                "document_id": document.id,
                "filename": file.filename,
                "file_path": file_path,
            }, None

        except HTTPException as e:
            # Handle file size errors
            return None, {
                "filename": file.filename,
                "reason": e.detail,
            }
        except Exception as e:
            return None, {
                "filename": file.filename,
                "reason": f"Processing error: {str(e)}",
            }


class CollectionService:
    """
    Service for handling collection operations.

    Encapsulates collection business logic and reduces coupling to storage layer.
    """

    def __init__(self, storage: DocumentStorageInterface):
        self.storage = storage

    def create_collection(
        self, name: str, description: Optional[str] = None
    ) -> Collection:
        """Create a new collection with validation."""
        # Input validation
        if not name or not name.strip():
            raise APIErrorHandler.validation_error("Collection name cannot be empty.")
        if len(name) > 100:
            raise APIErrorHandler.validation_error(
                "Collection name cannot exceed 100 characters."
            )
        if not re.match(r"^[a-zA-Z0-9 _-]+$", name):
            raise APIErrorHandler.validation_error(
                "Collection name can only contain alphanumeric characters, spaces, underscores, and hyphens."
            )

        return self.storage.create_collection(name)

    def get_all_collections(self) -> List[Collection]:
        """Get all collections."""
        return self.storage.get_all_collections()

    def delete_collection(self, collection_id: str) -> None:
        """Delete a collection."""
        self.storage.delete_collection(collection_id)

    def get_document_details(self, document_id: str) -> Dict[str, Any]:
        """Get detailed document information with proper error handling."""
        document = self.storage.get_document(document_id)
        if not document:
            raise APIErrorHandler.document_not_found(document_id)

        return {
            "id": document.id,
            "collection_ids": [c.id for c in document.collections],
            "collections": [{"id": c.id, "name": c.name} for c in document.collections],
            "filename": document.filename,
            "original_filename": document.original_filename,
            "content_hash": document.content_hash,
            "file_size": document.file_size,
            "mime_type": document.mime_type,
            "file_path": document.file_path,
            "document_metadata": document.document_metadata,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
        }

    def update_document_metadata(
        self, document_id: str, metadata_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update document metadata with proper validation and error handling."""
        document = self.storage.get_document(document_id)
        if not document:
            raise APIErrorHandler.document_not_found(document_id)

        # Get current metadata
        current_metadata = document.document_metadata or {}

        # Filter out None values
        update_data = {k: v for k, v in metadata_update.items() if v is not None}

        # Merge with existing metadata
        updated_metadata = current_metadata.copy()
        updated_metadata.update(update_data)

        # Mark as manually updated
        updated_metadata["manually_updated"] = True
        updated_metadata["manual_update_fields"] = list(update_data.keys())

        # Update in storage
        self.storage.update_document_metadata(document_id, updated_metadata)

        return {
            "message": "Metadata updated successfully",
            "updated_fields": list(update_data.keys()),
            "document_id": document_id,
        }

    def delete_document(self, document_id: str) -> None:
        """Delete a document."""
        self.storage.delete_document(document_id)

    def validate_document_exists_for_processing(self, document_id: str) -> Document:
        """Validate that document exists and return it for processing tasks."""
        document = self.storage.get_document(document_id)
        if not document:
            raise APIErrorHandler.document_not_found(document_id)
        return document


class TaskService:
    """
    Service for handling task creation and management.

    Encapsulates task creation business logic separate from HTTP handlers.
    """

    def __init__(self):
        pass

    def create_document_processing_task(
        self,
        file_path: str,
        document_id: str,
        collection_id: str,
        prompt: Optional[str] = None,
    ) -> str:
        """Create a document processing task and return task ID."""
        from ...tasks.document_tasks import process_document

        task_result = process_document.delay(
            file_path=file_path,
            document_id=document_id,
            collection_id=collection_id,
            prompt=prompt,
        )
        return task_result.id

    def create_collection_analysis_task(
        self,
        collection_id: str,
        file_paths: List[str],
        build_graph: bool = True,
        extract_metadata: bool = True,
        generate_embeddings: bool = True,
    ) -> str:
        """Create a collection analysis task and return task ID."""
        from ...tasks.workflow_tasks import complete_collection_analysis

        task_result = complete_collection_analysis.delay(
            collection_id=collection_id,
            file_paths=file_paths,
            build_graph=build_graph,
            extract_metadata=extract_metadata,
            generate_embeddings=generate_embeddings,
        )
        return task_result.id

    def create_question_answering_task(
        self,
        question: str,
        collection_id: str,
        task_name: Optional[str] = None,
        job_type: str = "question_merge",
        document_id: Optional[str] = None,
    ) -> str:
        """Create a question answering task and return task ID."""
        from ...tasks.llm_tasks import generate_answer

        task_result = generate_answer.delay(
            question=question,
            collection_id=collection_id,
            document_id=document_id,
            task_name=task_name,
            job_type=job_type,
        )
        return task_result.id

    def create_analysis_task(
        self,
        collection_id: str,
        task_name: Optional[str] = None,
        job_type: str = "analysis_merge",
        document_id: Optional[str] = None,
    ) -> str:
        """Create an analysis task and return task ID."""
        from ...tasks.llm_tasks import generate_answer

        task_result = generate_answer.delay(
            question="Analyze this collection"
            if not document_id
            else "Analyze this document",
            collection_id=collection_id,
            document_id=document_id,
            task_name=task_name,
            job_type=job_type,
        )
        return task_result.id


async def get_collection_by_identifier(
    storage: DocumentStorageInterface, identifier: str
) -> Optional[Collection]:
    """
    Get collection by ID or name.

    Args:
        storage: Storage interface
        identifier: Collection ID (UUID) or name

    Returns:
        Collection if found, None otherwise
    """
    # First try to get by ID (assuming UUID format)
    try:
        import uuid

        uuid.UUID(identifier)  # This will raise ValueError if not a valid UUID
        collection = storage.get_collection(identifier)
        if collection:
            return collection
    except (ValueError, TypeError):
        # Not a valid UUID, continue to try by name
        pass

    # Try to get by name
    collections = storage.get_all_collections()
    for collection in collections:
        if collection.name == identifier:
            return collection

    return None
