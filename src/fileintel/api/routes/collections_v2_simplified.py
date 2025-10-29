"""
Collections API v2 - Simplified using service layer.

Refactored to use CollectionsService for business logic,
following Single Responsibility Principle.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from fileintel.api.dependencies import (
    get_storage,
    get_collection_by_id_or_name,
    get_api_key,
)
from fileintel.api.error_handlers import (
    api_error_handler,
    create_success_response,
    validate_collection_exists,
    to_http_exception,
)
from fileintel.api.models import (
    TaskSubmissionRequest,
    TaskSubmissionResponse,
    DocumentProcessingRequest,
    BatchTaskSubmissionRequest,
    BatchTaskSubmissionResponse,
    ApiResponseV2,
)
from fileintel.api.services import CollectionsService
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.core.validation import FileValidationError

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_api_key)])


class CollectionCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


def get_collections_service(
    storage: PostgreSQLStorage = Depends(get_storage),
) -> CollectionsService:
    """Dependency to get CollectionsService."""
    return CollectionsService(storage)


# CRUD Operations for Collections
@router.post("/collections", response_model=ApiResponseV2)
@api_error_handler("create collection")
async def create_collection(
    request: CollectionCreateRequest,
    service: CollectionsService = Depends(get_collections_service),
) -> ApiResponseV2:
    """Create a new collection."""
    collection = service.create_collection(request.name, request.description)
    return create_success_response(
        {
            "id": collection.id,
            "name": collection.name,
            "description": collection.description,
        }
    )


@router.get("/collections", response_model=ApiResponseV2)
@api_error_handler("list collections")
async def list_collections(
    service: CollectionsService = Depends(get_collections_service),
) -> ApiResponseV2:
    """List all collections."""
    collections = service.get_all_collections()
    return create_success_response(collections)


@router.get("/collections/{collection_identifier}", response_model=ApiResponseV2)
@api_error_handler("get collection")
async def get_collection(
    collection_identifier: str,
    storage: PostgreSQLStorage = Depends(get_storage),
    service: CollectionsService = Depends(get_collections_service),
) -> ApiResponseV2:
    """Get collection by ID or name."""
    collection = validate_collection_exists(
        collection_identifier, storage, "get collection"
    )
    collection_data = service.get_collection_details(collection)
    return create_success_response(collection_data)


@router.delete("/collections/{collection_identifier}", response_model=ApiResponseV2)
@api_error_handler("delete collection")
async def delete_collection(
    collection_identifier: str, storage: PostgreSQLStorage = Depends(get_storage)
) -> ApiResponseV2:
    """Delete a collection."""
    collection = validate_collection_exists(
        collection_identifier, storage, "delete collection"
    )
    storage.delete_collection(collection.id)
    return create_success_response(
        {"message": f"Collection '{collection.name}' deleted successfully"}
    )


@router.post(
    "/collections/{collection_identifier}/documents", response_model=ApiResponseV2
)
@api_error_handler("upload document")
async def upload_document_to_collection(
    collection_identifier: str,
    file: UploadFile = File(...),
    storage: PostgreSQLStorage = Depends(get_storage),
    service: CollectionsService = Depends(get_collections_service),
) -> ApiResponseV2:
    """Upload a single document to a collection."""
    try:
        import aiofiles
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="aiofiles package not available for async file operations",
        )

    # Validate collection exists
    collection = get_collection_by_id_or_name(collection_identifier, storage)
    if not collection:
        raise HTTPException(
            status_code=404, detail=f"Collection {collection_identifier} not found"
        )

    # Read file content
    content = await file.read()

    # Use service to handle upload
    try:
        result = await service.upload_document_to_collection(collection, file, content)
        return create_success_response(result)
    except FileValidationError as e:
        raise to_http_exception(e)


@router.post("/collections/{collection_id}/process", response_model=ApiResponseV2)
@api_error_handler("process collection")
async def process_collection(
    collection_id: str,
    request: TaskSubmissionRequest,
    service: CollectionsService = Depends(get_collections_service),
) -> ApiResponseV2:
    """Submit collection for processing."""
    try:
        result = service.submit_collection_processing_task(
            collection_id=collection_id,
            include_embeddings=request.generate_embeddings,
            task_options=request.parameters,
        )
        return create_success_response(result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/collections/{collection_id}/documents/add", response_model=ApiResponseV2)
@api_error_handler("add documents to collection")
async def add_documents_to_collection(
    collection_id: str,
    request: DocumentProcessingRequest,
    service: CollectionsService = Depends(get_collections_service),
) -> ApiResponseV2:
    """Add documents to collection and optionally process incrementally."""
    try:
        if request.incremental_update:
            result = service.submit_incremental_update_task(
                collection_id=collection_id, new_documents_only=True
            )
        else:
            result = service.submit_collection_processing_task(
                collection_id=collection_id,
                include_embeddings=request.include_embeddings,
            )
        return create_success_response(result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/collections/batch-process", response_model=ApiResponseV2)
@api_error_handler("batch process collections")
async def batch_process_collections(
    request: BatchTaskSubmissionRequest,
    service: CollectionsService = Depends(get_collections_service),
) -> ApiResponseV2:
    """Process multiple collections in batch."""
    results = []

    for collection_id in request.collection_ids:
        try:
            result = service.submit_collection_processing_task(
                collection_id=collection_id,
                include_embeddings=request.include_embeddings,
                task_options=request.options,
            )
            results.append(
                {
                    "collection_id": collection_id,
                    "status": "submitted",
                    "task_id": result["task_id"],
                }
            )
        except Exception as e:
            results.append(
                {"collection_id": collection_id, "status": "error", "error": str(e)}
            )

    return create_success_response(
        {
            "batch_id": f"batch_{len(results)}_collections",
            "results": results,
            "summary": {
                "total_requested": len(request.collection_ids),
                "submitted": len([r for r in results if r["status"] == "submitted"]),
                "errors": len([r for r in results if r["status"] == "error"]),
            },
        }
    )


@router.get(
    "/collections/{collection_id}/processing-status", response_model=ApiResponseV2
)
@api_error_handler("get collection processing status")
async def get_collection_processing_status(
    collection_id: str, service: CollectionsService = Depends(get_collections_service)
) -> ApiResponseV2:
    """Get current processing status of a collection."""
    try:
        status = service.get_processing_status(collection_id)
        return create_success_response(status)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post(
    "/collections/{collection_id}/upload-and-process", response_model=ApiResponseV2
)
@api_error_handler("upload and process")
async def upload_and_process_document(
    collection_id: str,
    file: UploadFile = File(...),
    include_embeddings: bool = Form(True),
    storage: PostgreSQLStorage = Depends(get_storage),
    service: CollectionsService = Depends(get_collections_service),
) -> ApiResponseV2:
    """Upload document and immediately start processing."""
    try:
        import aiofiles
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="aiofiles package not available for async file operations",
        )

    # Validate collection exists
    collection = storage.get_collection(collection_id)
    if not collection:
        raise HTTPException(
            status_code=404, detail=f"Collection {collection_id} not found"
        )

    # Read file content
    content = await file.read()

    try:
        # Upload document
        upload_result = await service.upload_document_to_collection(
            collection, file, content
        )

        # Submit processing task
        process_result = service.submit_collection_processing_task(
            collection_id=collection_id, include_embeddings=include_embeddings
        )

        return create_success_response(
            {"upload": upload_result, "processing": process_result}
        )
    except FileValidationError as e:
        raise to_http_exception(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
