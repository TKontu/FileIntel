"""
Shared error handling utilities for API v2 endpoints.

Consolidates duplicate error handling patterns across route handlers.
"""

import logging
from datetime import datetime
from typing import Any, Callable, TypeVar, Optional
from functools import wraps

from fastapi import HTTPException
from .models import ApiResponseV2

logger = logging.getLogger(__name__)

T = TypeVar("T")


def api_error_handler(operation_name: str):
    """
    Decorator to handle common API errors and return standardized responses.

    Args:
        operation_name: Description of the operation for logging
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTP exceptions to maintain FastAPI error handling
                raise
            except Exception as e:
                logger.error(f"Error in {operation_name}: {e}")
                return ApiResponseV2(
                    success=False, error=str(e), timestamp=datetime.utcnow()
                )

        return wrapper

    return decorator


def celery_error_handler(task_operation: str):
    """
    Decorator specifically for Celery task operations with specialized error handling.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except (ValueError, ConnectionError, RuntimeError) as e:
                logger.error(f"Error in {task_operation}: {e}")
                return ApiResponseV2(
                    success=False, error=str(e), timestamp=datetime.utcnow()
                )
            except Exception as e:
                logger.error(f"Unexpected error in {task_operation}: {e}")
                return ApiResponseV2(
                    success=False,
                    error=f"Internal error: {str(e)}",
                    timestamp=datetime.utcnow(),
                )

        return wrapper

    return decorator


def create_success_response(data: Any, message: Optional[str] = None) -> ApiResponseV2:
    """Create a standardized success response."""
    response_data = data
    if message:
        if isinstance(data, dict):
            response_data = {**data, "message": message}
        else:
            response_data = {"data": data, "message": message}

    return ApiResponseV2(success=True, data=response_data, timestamp=datetime.utcnow())


def create_error_response(error: str) -> ApiResponseV2:
    """Create a standardized error response."""
    return ApiResponseV2(success=False, error=error, timestamp=datetime.utcnow())


def validate_collection_exists(
    collection_identifier: str, storage, operation_name: str
):
    """
    DEPRECATED: Use fileintel.core.validation.validate_collection_exists instead.
    Validate that a collection exists and return it.

    Raises HTTPException if not found.
    """
    from fileintel.core.validation import (
        validate_collection_exists as core_validate_collection_exists,
    )
    from fileintel.core.validation import to_http_exception, CollectionValidationError

    try:
        return core_validate_collection_exists(collection_identifier, storage)
    except CollectionValidationError as e:
        raise to_http_exception(e, status_code=404)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating collection in {operation_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error accessing collection: {str(e)}"
        )


class APIErrorHandler:
    """Centralized error handling for common API scenarios."""

    @staticmethod
    def document_not_found(document_id: Optional[str] = None) -> HTTPException:
        """Standard document not found error."""
        detail = (
            f"Document with ID {document_id} not found"
            if document_id
            else "Document not found"
        )
        return HTTPException(status_code=404, detail=detail)

    @staticmethod
    def collection_not_found(collection_id: Optional[str] = None) -> HTTPException:
        """Standard collection not found error."""
        detail = (
            f"Collection {collection_id} not found"
            if collection_id
            else "Collection not found"
        )
        return HTTPException(status_code=404, detail=detail)

    @staticmethod
    def task_not_found(task_id: str) -> HTTPException:
        """Standard task not found error."""
        return HTTPException(status_code=404, detail=f"Task {task_id} not found")

    @staticmethod
    def duplicate_document(existing_document_id: str) -> HTTPException:
        """Standard duplicate document error."""
        return HTTPException(
            status_code=409,
            detail=f"Document with same content already exists in this collection with ID: {existing_document_id}",
        )

    @staticmethod
    def file_too_large(max_size: int) -> HTTPException:
        """Standard file too large error."""
        return HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size} bytes",
        )

    @staticmethod
    def unsupported_file_type(
        content_type: str, supported_types: list
    ) -> HTTPException:
        """Standard unsupported file type error."""
        return HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Supported types are: {', '.join(supported_types)}",
        )

    @staticmethod
    def validation_error(message: str) -> HTTPException:
        """Standard validation error."""
        return HTTPException(status_code=400, detail=message)

    @staticmethod
    def internal_error(message: str) -> HTTPException:
        """Standard internal server error."""
        return HTTPException(status_code=500, detail=message)


# Constants for common configuration
DEFAULT_TASK_ESTIMATION_SECONDS_PER_DOCUMENT = 30
INCREMENTAL_TASK_ESTIMATION_SECONDS_PER_DOCUMENT = 25
BATCH_TASK_ESTIMATION_SECONDS_PER_COLLECTION = 120
