"""
Collections API v2 - Task-based processing endpoints.

Replaces job-based processing with Celery task submission and monitoring.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from fileintel.api.dependencies import (
    get_storage,
    get_collection_by_id_or_name,
    get_api_key,
)
from fileintel.api.services import get_collection_by_identifier
from fileintel.api.error_handlers import (
    api_error_handler,
    create_success_response,
    validate_collection_exists,
    DEFAULT_TASK_ESTIMATION_SECONDS_PER_DOCUMENT,
    INCREMENTAL_TASK_ESTIMATION_SECONDS_PER_DOCUMENT,
    BATCH_TASK_ESTIMATION_SECONDS_PER_COLLECTION,
)
from fileintel.api.models import (
    TaskSubmissionRequest,
    TaskSubmissionResponse,
    DocumentProcessingRequest,
    BatchTaskSubmissionRequest,
    BatchTaskSubmissionResponse,
    ApiResponseV2,
    TaskState,
)
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.storage.models import Collection
from fileintel.tasks.workflow_tasks import (
    complete_collection_analysis,
    incremental_collection_update,
)
from fileintel.tasks.document_tasks import process_document, process_collection
import uuid

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_api_key)])


def _get_status_description(status: str) -> str:
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


class CollectionCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


# CRUD Operations for Collections
@router.post("/collections", response_model=ApiResponseV2)
@api_error_handler("create collection")
async def create_collection(
    request: CollectionCreateRequest, storage: PostgreSQLStorage = Depends(get_storage)
) -> ApiResponseV2:
    """Create a new collection."""
    collection = storage.create_collection(request.name, request.description)
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
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """List all collections."""
    collections = storage.get_all_collections()
    collection_data = [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "status": getattr(c, "processing_status", "unknown"),
        }
        for c in collections
    ]
    return create_success_response(collection_data)


@router.get("/collections/{collection_identifier}", response_model=ApiResponseV2)
@api_error_handler("get collection")
async def get_collection(
    collection_identifier: str, storage: PostgreSQLStorage = Depends(get_storage)
) -> ApiResponseV2:
    """Get collection by ID or name."""
    collection = validate_collection_exists(
        collection_identifier, storage, "get collection"
    )
    collection_data = {
        "id": collection.id,
        "name": collection.name,
        "description": collection.description,
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "original_filename": d.original_filename,
                "mime_type": d.mime_type,
            }
            for d in collection.documents
        ],
    }
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


# Document Operations
@router.post(
    "/collections/{collection_identifier}/documents", response_model=ApiResponseV2
)
async def upload_document_to_collection(
    collection_identifier: str,
    file: UploadFile = File(...),
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """Upload a single document to a collection."""
    try:
        from fileintel.core.config import get_config
        from pathlib import Path

        try:
            import aiofiles
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="aiofiles package not available for async file operations",
            )

        # Validate collection exists
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        config = get_config()

        # Validate file upload
        from fileintel.core.validation import (
            validate_file_upload,
            to_http_exception,
            FileValidationError,
        )

        try:
            validate_file_upload(file)
        except FileValidationError as e:
            raise to_http_exception(e)

        # Create unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        unique_filename = f"{file_id}{file_extension}"
        file_path = Path(config.paths.uploads) / unique_filename

        # Ensure upload directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save file and calculate metadata
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # Calculate file metadata
        import hashlib

        content_hash = hashlib.sha256(content).hexdigest()
        file_size = len(content)
        mime_type = file.content_type or "application/octet-stream"

        # Store document in database (file_path is stored in metadata)
        document = storage.create_document(
            filename=unique_filename,
            content_hash=content_hash,
            file_size=file_size,
            mime_type=mime_type,
            collection_id=collection.id,
            original_filename=file.filename,
            metadata={
                "uploaded_via": "api_v2",
                "original_filename": file.filename,
                "file_path": str(file_path),
            },
        )

        return ApiResponseV2(
            success=True,
            data={
                "document_id": document.id,
                "filename": document.filename,
                "file_path": str(file_path),
                "message": "Document uploaded successfully",
            },
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())


# Individual Document Operations
@router.get("/documents/{document_id}", response_model=ApiResponseV2)
@api_error_handler("get document")
async def get_document(
    document_id: str, storage: PostgreSQLStorage = Depends(get_storage)
) -> ApiResponseV2:
    """Get a specific document by ID."""
    document = storage.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    document_data = {
        "id": document.id,
        "filename": document.filename,
        "original_filename": document.original_filename,
        "content_hash": document.content_hash,
        "file_size": document.file_size,
        "mime_type": document.mime_type,
        "collection_id": document.collection_id,
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "document_metadata": document.document_metadata or {},
    }
    return create_success_response(document_data)


@router.delete("/documents/{document_id}", response_model=ApiResponseV2)
@api_error_handler("delete document")
async def delete_document(
    document_id: str, storage: PostgreSQLStorage = Depends(get_storage)
) -> ApiResponseV2:
    """Delete a specific document by ID."""
    document = storage.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    filename = document.original_filename or document.filename
    storage.delete_document(document_id)
    return create_success_response(
        {"message": f"Document '{filename}' deleted successfully"}
    )


@router.get("/documents/{document_id}/chunks", response_model=ApiResponseV2)
@api_error_handler("get document chunks")
async def get_document_chunks(
    document_id: str,
    limit: Optional[int] = 100,
    offset: Optional[int] = 0,
    storage: PostgreSQLStorage = Depends(get_storage)
) -> ApiResponseV2:
    """Get chunks for a specific document."""
    document = storage.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Get chunks for the document
    chunks = storage.get_all_chunks_for_document(document_id)

    # Apply pagination
    total_chunks = len(chunks)
    paginated_chunks = chunks[offset:offset + limit] if chunks else []

    # Format chunks for API response
    chunk_data = []
    for i, chunk in enumerate(paginated_chunks):
        chunk_info = {
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index if hasattr(chunk, 'chunk_index') else offset + i,
            "text": chunk.chunk_text,
            "has_embedding": chunk.embedding is not None,
            "embedding_dimensions": len(chunk.embedding) if chunk.embedding else None,
            "chunk_metadata": chunk.chunk_metadata or {},
        }
        chunk_data.append(chunk_info)

    response_data = {
        "document_id": document_id,
        "total_chunks": total_chunks,
        "chunks_returned": len(chunk_data),
        "offset": offset,
        "limit": limit,
        "chunks": chunk_data,
    }

    return create_success_response(response_data)


# Processing Operations
@router.post(
    "/collections/{collection_identifier}/process", response_model=ApiResponseV2
)
async def submit_collection_processing_task(
    collection_identifier: str,
    request: TaskSubmissionRequest,
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """
    Submit a collection for complete processing using Celery tasks.

    This endpoint replaces the v1 job-based processing with distributed task execution.
    """
    try:
        # Validate collection exists
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        # Validate collection has documents and file paths
        from fileintel.core.validation import (
            validate_collection_has_documents,
            validate_file_paths,
            to_http_exception,
            CollectionValidationError,
        )

        try:
            documents = validate_collection_has_documents(collection.id, storage)
            file_paths = validate_file_paths(documents)
        except CollectionValidationError as e:
            raise to_http_exception(e)

        # Submit the appropriate task based on operation type
        if request.operation_type == "complete_analysis":
            # Use the complete workflow task
            task = complete_collection_analysis.delay(
                collection_id=collection.id,
                file_paths=file_paths,
                build_graph=request.build_graph,
                extract_metadata=request.extract_metadata,
                generate_embeddings=request.generate_embeddings,
                **request.parameters,
            )
        elif request.operation_type == "document_processing_only":
            # Use basic document processing
            task = process_collection.delay(
                collection_identifier=collection.id,
                file_paths=file_paths,
                **request.parameters,
            )
        else:
            from fileintel.core.validation import (
                validate_operation_type,
                to_http_exception,
                ValidationError,
            )

            try:
                validate_operation_type(request.operation_type)
            except ValidationError as e:
                raise to_http_exception(e)

        # Create response
        response_data = TaskSubmissionResponse(
            task_id=task.id,
            task_type=request.operation_type,
            status=TaskState.PENDING,
            submitted_at=datetime.utcnow(),
            collection_identifier=collection.id,
            estimated_duration=len(file_paths)
            * DEFAULT_TASK_ESTIMATION_SECONDS_PER_DOCUMENT,
        )

        return ApiResponseV2(
            success=True, data=response_data.dict(), timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting collection processing task: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())


@router.post(
    "/collections/{collection_identifier}/documents/add", response_model=ApiResponseV2
)
async def add_documents_to_collection(
    collection_identifier: str,
    request: DocumentProcessingRequest,
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """
    Add new documents to existing collection with incremental processing.

    Uses incremental update task to process new documents and update indexes.
    """
    try:
        # Validate collection exists
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        # Validate file paths
        if not request.file_paths:
            raise HTTPException(status_code=400, detail="No file paths provided")

        # Get existing embeddings for incremental update
        existing_documents = storage.get_documents_by_collection(collection.id)
        existing_embeddings = []
        # Note: In a real implementation, you'd fetch actual embeddings from storage
        # For now, we'll pass None and let the task handle it

        # Submit incremental update task
        task = incremental_collection_update.delay(
            collection_identifier=collection.id,
            new_file_paths=request.file_paths,
            existing_embeddings=existing_embeddings,
        )

        response_data = TaskSubmissionResponse(
            task_id=task.id,
            task_type="incremental_update",
            status=TaskState.PENDING,
            submitted_at=datetime.utcnow(),
            collection_identifier=collection.id,
            estimated_duration=len(request.file_paths)
            * INCREMENTAL_TASK_ESTIMATION_SECONDS_PER_DOCUMENT,
        )

        return ApiResponseV2(
            success=True, data=response_data.dict(), timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting incremental update task: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())


@router.post("/collections/batch-process", response_model=ApiResponseV2)
async def submit_batch_processing_tasks(
    request: BatchTaskSubmissionRequest,
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """
    Submit multiple collection processing tasks in a batch.

    Supports parallel or sequential execution workflows.
    """
    try:
        if not request.tasks:
            raise HTTPException(status_code=400, detail="No tasks provided in batch")

        submitted_tasks = []
        batch_id = str(uuid.uuid4())

        for task_request in request.tasks:
            # Validate collection exists
            collection = get_collection_by_id_or_name(
                task_request.collection_identifier, storage
            )
            if not collection:
                logger.warning(
                    f"Collection {task_request.collection_identifier} not found, skipping"
                )
                continue

            # Get documents for this collection
            documents = storage.get_documents_by_collection(collection.id)
            file_paths = [doc.file_path for doc in documents if doc.file_path]

            if not file_paths:
                logger.warning(
                    f"No documents found for collection {collection.id}, skipping"
                )
                continue

            # Submit task
            if request.workflow_type == "parallel":
                # Submit all tasks immediately for parallel execution
                task = complete_collection_analysis.delay(
                    collection_id=str(collection.id),
                    file_paths=file_paths,
                    build_graph=task_request.build_graph,
                    extract_metadata=task_request.extract_metadata,
                    generate_embeddings=task_request.generate_embeddings,
                    batch_id=batch_id,
                    **task_request.parameters,
                )
            else:
                # For sequential, we'd need to implement a chain workflow
                # For now, treat as parallel
                task = complete_collection_analysis.delay(
                    collection_id=str(collection.id),
                    file_paths=file_paths,
                    build_graph=task_request.build_graph,
                    extract_metadata=task_request.extract_metadata,
                    generate_embeddings=task_request.generate_embeddings,
                    batch_id=batch_id,
                    **task_request.parameters,
                )

            submitted_tasks.append(task.id)

        if not submitted_tasks:
            raise HTTPException(
                status_code=400, detail="No valid tasks could be submitted"
            )

        response_data = BatchTaskSubmissionResponse(
            batch_id=batch_id,
            task_ids=submitted_tasks,
            submitted_count=len(submitted_tasks),
            workflow_type=request.workflow_type,
            estimated_duration=len(submitted_tasks)
            * BATCH_TASK_ESTIMATION_SECONDS_PER_COLLECTION,
        )

        return ApiResponseV2(
            success=True, data=response_data.dict(), timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting batch processing tasks: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())


@router.get(
    "/collections/{collection_identifier}/processing-status",
    response_model=ApiResponseV2,
)
async def get_collection_processing_status(
    collection_identifier: str, storage: PostgreSQLStorage = Depends(get_storage)
) -> ApiResponseV2:
    """
    Get the current processing status for a collection.

    Returns information about any active or recent tasks for the collection.
    """
    try:
        # Validate collection exists
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        # Get collection status and related information
        documents = storage.get_documents_by_collection(collection.id)
        chunks = storage.get_all_chunks_for_collection(collection.id)

        # Count chunks with embeddings
        chunks_with_embeddings = (
            sum(1 for chunk in chunks if chunk.embedding is not None) if chunks else 0
        )

        # Get actual processing status from collection
        processing_status = getattr(collection, "processing_status", "created")

        # Determine available operations based on current status
        available_operations = []
        if processing_status in ["created", "completed", "failed"]:
            available_operations.extend(
                ["complete_analysis", "document_processing_only", "incremental_update"]
            )

        status_info = {
            "collection_identifier": collection.id,
            "collection_name": collection.name,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "chunks_with_embeddings": chunks_with_embeddings,
            "embedding_coverage": (chunks_with_embeddings / len(chunks) * 100)
            if chunks
            else 0,
            "last_updated": collection.updated_at.isoformat()
            if collection.updated_at
            else None,
            "created_at": collection.created_at.isoformat()
            if collection.created_at
            else None,
            "processing_status": processing_status,
            "status_description": _get_status_description(processing_status),
            "available_operations": available_operations,
        }

        return ApiResponseV2(
            success=True, data=status_info, timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection processing status: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())


@router.post(
    "/collections/{collection_identifier}/upload-and-process",
    response_model=ApiResponseV2,
)
async def upload_and_process_documents(
    collection_identifier: str,
    files: List[UploadFile] = File(...),
    process_immediately: bool = Form(default=True),
    build_graph: bool = Form(default=True),
    extract_metadata: bool = Form(default=True),
    generate_embeddings: bool = Form(default=True),
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """
    Upload documents to a collection and optionally process them immediately.

    Combines file upload with task submission in a single endpoint.
    """
    try:
        from fileintel.core.config import get_config
        import os
        from pathlib import Path

        try:
            import aiofiles
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="aiofiles package not available for async file operations",
            )

        # Validate collection exists
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        config = get_config()
        uploaded_files = []

        # Upload files
        for file in files:
            if not file.filename:
                continue

            # Create unique filename
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix
            unique_filename = f"{file_id}{file_extension}"
            file_path = Path(config.paths.uploads) / unique_filename

            # Ensure upload directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save file and calculate metadata
            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            # Calculate file metadata
            import hashlib

            content_hash = hashlib.sha256(content).hexdigest()
            file_size = len(content)
            mime_type = file.content_type or "application/octet-stream"

            # Store document in database
            document = storage.create_document(
                filename=unique_filename,
                original_filename=file.filename,
                content_hash=content_hash,
                file_size=file_size,
                mime_type=mime_type,
                collection_id=collection.id,
                metadata={"uploaded_via": "api_v2", "file_path": str(file_path)},
            )

            uploaded_files.append(str(file_path))

        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No files were uploaded")

        result_data = {
            "uploaded_files": len(uploaded_files),
            "file_paths": uploaded_files,
        }

        # Process immediately if requested
        if process_immediately:
            task = complete_collection_analysis.delay(
                collection_id=str(collection.id),
                file_paths=uploaded_files,
                build_graph=build_graph,
                extract_metadata=extract_metadata,
                generate_embeddings=generate_embeddings,
            )

            result_data.update(
                {
                    "task_id": task.id,
                    "processing_status": "submitted",
                    "estimated_duration": len(uploaded_files)
                    * DEFAULT_TASK_ESTIMATION_SECONDS_PER_DOCUMENT,
                }
            )

        return ApiResponseV2(
            success=True, data=result_data, timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading and processing documents: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())
