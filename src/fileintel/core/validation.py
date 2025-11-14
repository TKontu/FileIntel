"""
Centralized validation functions for FileIntel.

Consolidates all validation logic to reduce duplication and improve maintainability.
All validation errors are standardized and follow consistent error message patterns.
"""

import mimetypes
from typing import List, Optional, Any, Dict
from fastapi import HTTPException, UploadFile
from fileintel.storage.postgresql_storage import PostgreSQLStorage


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class FileValidationError(ValidationError):
    """Raised when file validation fails."""

    pass


class CollectionValidationError(ValidationError):
    """Raised when collection validation fails."""

    pass


class TaskValidationError(ValidationError):
    """Raised when task validation fails."""

    pass


# Supported file formats and MIME types
SUPPORTED_FILE_FORMATS = ["pdf", "epub", "mobi", "txt", "md"]
SUPPORTED_MIME_TYPES = {
    "pdf": "application/pdf",
    "epub": "application/epub+zip",
    "mobi": "application/x-mobipocket-ebook",
    "txt": "text/plain",
    "md": "text/markdown",
}

VALID_OPERATION_TYPES = [
    "complete_analysis",
    "document_processing_only",
    "incremental_update",
]
VALID_SEARCH_TYPES = ["vector", "graph", "adaptive", "global", "local"]


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """
    Validate that all required fields are present and not None.

    Args:
        data: Dictionary containing the data to validate
        required_fields: List of field names that are required

    Raises:
        TaskValidationError: If any required fields are missing
    """
    missing_fields = [
        field for field in required_fields if field not in data or data[field] is None
    ]
    if missing_fields:
        raise TaskValidationError(f"Missing required fields: {missing_fields}")


def validate_file_upload(
    file: UploadFile, supported_formats: Optional[List[str]] = None
) -> None:
    """
    Validate uploaded file format and content type.

    Args:
        file: The uploaded file to validate
        supported_formats: List of supported file formats (defaults to SUPPORTED_FILE_FORMATS)

    Raises:
        FileValidationError: If file validation fails
    """
    if not file.filename:
        raise FileValidationError("No filename provided")

    if supported_formats is None:
        supported_formats = SUPPORTED_FILE_FORMATS

    supported_mime_types = [
        SUPPORTED_MIME_TYPES.get(fmt, fmt) for fmt in supported_formats
    ]

    if file.content_type not in supported_mime_types:
        # Try to guess from filename as fallback
        mime_type, _ = mimetypes.guess_type(file.filename)
        if mime_type not in supported_mime_types:
            raise FileValidationError(
                f"Unsupported file type: {file.content_type}. "
                f"Supported types: {supported_mime_types}"
            )


def validate_collection_exists(
    collection_identifier: str, storage: PostgreSQLStorage
) -> Any:
    """
    Validate that a collection exists and return it.

    Args:
        collection_identifier: Collection ID or name
        storage: PostgreSQL storage instance

    Returns:
        The collection object if found

    Raises:
        CollectionValidationError: If collection is not found
    """
    import uuid

    # Try to get by ID first
    try:
        uuid.UUID(collection_identifier)
        collection = storage.get_collection(collection_identifier)
    except (ValueError, TypeError):
        # Not a UUID, try by name
        collection = storage.get_collection_by_name(collection_identifier)

    if not collection:
        raise CollectionValidationError(
            f"Collection '{collection_identifier}' not found"
        )
    return collection


def validate_collection_has_documents(
    collection_id: str, storage: PostgreSQLStorage
) -> List[Any]:
    """
    Validate that a collection contains documents.

    Args:
        collection_id: Collection ID
        storage: PostgreSQL storage instance

    Returns:
        List of documents in the collection

    Raises:
        CollectionValidationError: If collection has no documents
    """
    documents = storage.get_documents_by_collection(collection_id)
    if not documents:
        raise CollectionValidationError("Collection contains no documents to process")
    return documents


def validate_file_paths(documents: List[Any]) -> List[str]:
    """
    Validate that documents have valid file paths.

    Args:
        documents: List of document objects

    Returns:
        List of valid file paths

    Raises:
        CollectionValidationError: If no valid file paths found
    """
    file_paths = []
    for doc in documents:
        # File path is now a direct column on the document model
        if hasattr(doc, 'file_path') and doc.file_path:
            file_paths.append(doc.file_path)
        # Fallback to metadata for backward compatibility
        elif doc.document_metadata and doc.document_metadata.get("file_path"):
            file_paths.append(doc.document_metadata["file_path"])

    if not file_paths:
        raise CollectionValidationError("No file paths found in collection documents")
    return file_paths


def validate_operation_type(operation_type: str) -> None:
    """
    Validate operation type for collection processing.

    Args:
        operation_type: The operation type to validate

    Raises:
        ValidationError: If operation type is invalid
    """
    if operation_type not in VALID_OPERATION_TYPES:
        raise ValidationError(
            f"Unknown operation type: {operation_type}. Valid types: {VALID_OPERATION_TYPES}"
        )


def validate_search_type(search_type: str) -> None:
    """
    Validate search type for query processing.

    Args:
        search_type: The search type to validate

    Raises:
        ValidationError: If search type is invalid
    """
    if search_type not in VALID_SEARCH_TYPES:
        raise ValidationError(
            f"Unknown search type: {search_type}. Valid types: {VALID_SEARCH_TYPES}"
        )


def validate_non_empty_list(items: List[Any], item_name: str) -> None:
    """
    Validate that a list is not empty.

    Args:
        items: List to validate
        item_name: Name of the items for error message

    Raises:
        ValidationError: If list is empty
    """
    if not items:
        raise ValidationError(f"No {item_name} provided")


def validate_task_batch(tasks: List[Any]) -> None:
    """
    Validate batch task submission.

    Args:
        tasks: List of tasks to validate

    Raises:
        TaskValidationError: If no tasks provided
    """
    if not tasks:
        raise TaskValidationError("No tasks provided in batch")


def validate_batch_size(items: List[Any], max_size: int, item_name: str) -> None:
    """
    Validate that batch size doesn't exceed maximum limit.

    Args:
        items: List of items in the batch
        max_size: Maximum allowed batch size
        item_name: Name of items for error message (e.g., "files", "collections")

    Raises:
        ValidationError: If batch size exceeds maximum
    """
    if len(items) > max_size:
        raise ValidationError(
            f"Batch size {len(items)} exceeds maximum {max_size} {item_name}"
        )


def validate_file_size(file: UploadFile, max_size_mb: int) -> None:
    """
    Validate that file size doesn't exceed maximum limit.

    Args:
        file: The uploaded file to validate
        max_size_mb: Maximum file size in megabytes

    Raises:
        FileValidationError: If file size exceeds limit
    """
    # Note: file.size might not be available for all upload types
    # This is a best-effort check
    if hasattr(file, 'size') and file.size:
        max_size_bytes = max_size_mb * 1024 * 1024
        if file.size > max_size_bytes:
            size_mb = file.size / (1024 * 1024)
            raise FileValidationError(
                f"File size {size_mb:.2f}MB exceeds maximum {max_size_mb}MB"
            )


def validate_uploaded_files(files: List[UploadFile]) -> None:
    """
    Validate that files were uploaded.

    Args:
        files: List of uploaded files

    Raises:
        FileValidationError: If no files uploaded
    """
    if not files:
        raise FileValidationError("No files were uploaded")


def validate_llm_response_content(content: str) -> None:
    """
    Validate LLM response content.

    Args:
        content: Response content to validate

    Raises:
        ValidationError: If content is empty
    """
    if not content or not content.strip():
        raise ValidationError("LLM response content is empty")


def validate_llm_response_model(model: str) -> None:
    """
    Validate LLM response model information.

    Args:
        model: Model name to validate

    Raises:
        ValidationError: If model is missing
    """
    if not model:
        raise ValidationError("LLM response is missing model information")


# Helper function to convert validation errors to HTTP exceptions
def to_http_exception(error: ValidationError, status_code: int = 400) -> HTTPException:
    """
    Convert validation error to HTTP exception.

    Args:
        error: The validation error
        status_code: HTTP status code to use

    Returns:
        HTTPException with appropriate error message
    """
    return HTTPException(status_code=status_code, detail=str(error))
