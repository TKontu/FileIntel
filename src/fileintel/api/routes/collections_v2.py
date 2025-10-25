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
    generate_collection_embeddings_simple,
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
                "file_size": d.file_size,
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

        # Check for duplicate in this collection
        existing_document = storage.get_document_by_hash_and_collection(
            content_hash, collection.id
        )

        if existing_document:
            logger.info(
                f"Duplicate detected: {file.filename} (hash: {content_hash[:16]}...) "
                f"already exists as document {existing_document.id}"
            )

            # Return existing document instead of creating duplicate
            document = existing_document
            duplicate_detected = True
        else:
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
            duplicate_detected = False

        return ApiResponseV2(
            success=True,
            data={
                "document_id": document.id,
                "filename": document.filename,
                "original_filename": document.original_filename,
                "content_hash": document.content_hash,
                "file_size": document.file_size,
                "file_path": str(file_path),
                "duplicate": duplicate_detected,
                "message": "Duplicate file detected, linked to existing document" if duplicate_detected else "Document uploaded successfully",
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
    """Get a specific document by ID or ID prefix."""
    # Try exact match first
    document = storage.get_document(document_id)

    # If not found and looks like a prefix (< 36 chars), try prefix search
    if not document and len(document_id) < 36:
        from sqlalchemy import or_
        from fileintel.storage.models import Document

        # Search for documents starting with this prefix
        matching_docs = storage.db.query(Document).filter(
            Document.id.like(f"{document_id}%")
        ).all()

        if len(matching_docs) == 1:
            document = matching_docs[0]
        elif len(matching_docs) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Ambiguous ID prefix '{document_id}' matches {len(matching_docs)} documents. Please provide more characters."
            )

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
    """Delete a specific document by ID or ID prefix."""
    # Try exact match first
    document = storage.get_document(document_id)

    # If not found and looks like a prefix (< 36 chars), try prefix search
    if not document and len(document_id) < 36:
        from fileintel.storage.models import Document

        matching_docs = storage.db.query(Document).filter(
            Document.id.like(f"{document_id}%")
        ).all()

        if len(matching_docs) == 1:
            document = matching_docs[0]
            document_id = document.id  # Use full ID for deletion
        elif len(matching_docs) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Ambiguous ID prefix '{document_id}' matches {len(matching_docs)} documents. Please provide more characters."
            )

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
    """Get chunks for a specific document by ID or ID prefix."""
    # Try exact match first
    document = storage.get_document(document_id)

    # If not found and looks like a prefix (< 36 chars), try prefix search
    if not document and len(document_id) < 36:
        from fileintel.storage.models import Document

        matching_docs = storage.db.query(Document).filter(
            Document.id.like(f"{document_id}%")
        ).all()

        if len(matching_docs) == 1:
            document = matching_docs[0]
            document_id = document.id  # Use full ID for chunk lookup
        elif len(matching_docs) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Ambiguous ID prefix '{document_id}' matches {len(matching_docs)} documents. Please provide more characters."
            )

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
        has_embedding = chunk.embedding is not None
        chunk_info = {
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index if hasattr(chunk, 'chunk_index') else offset + i,
            "text": chunk.chunk_text,
            "has_embedding": has_embedding,
            "embedding_dimensions": len(chunk.embedding) if has_embedding else None,
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
            all_file_paths = validate_file_paths(documents)
        except CollectionValidationError as e:
            raise to_http_exception(e)

        # Filter out documents that already have chunks (skip re-processing)
        file_paths = []
        documents_with_chunks = []
        documents_without_chunks = []

        for doc in documents:
            doc_chunks = storage.get_all_chunks_for_document(doc.id)
            if doc_chunks:
                # Document already has chunks - skip processing
                documents_with_chunks.append({
                    "document_id": str(doc.id),
                    "filename": doc.original_filename or doc.filename,
                    "chunk_count": len(doc_chunks)
                })
                logger.info(
                    f"Skipping document {doc.id} ({doc.original_filename}) - "
                    f"already has {len(doc_chunks)} chunks"
                )
            else:
                # Document needs processing
                file_path = doc.metadata.get("file_path") if doc.metadata else None
                if not file_path and doc.document_metadata:
                    file_path = doc.document_metadata.get("file_path")

                if file_path:
                    file_paths.append(file_path)
                    documents_without_chunks.append({
                        "document_id": str(doc.id),
                        "filename": doc.original_filename or doc.filename
                    })

        logger.info(
            f"Collection {collection.id}: {len(documents_with_chunks)} documents already processed, "
            f"{len(file_paths)} documents need processing"
        )

        # Validate operation type first
        from fileintel.core.validation import (
            validate_operation_type,
            to_http_exception,
            ValidationError,
        )

        try:
            validate_operation_type(request.operation_type)
        except ValidationError as e:
            raise to_http_exception(e)

        # Check if all documents already have chunks
        if not file_paths and len(documents_with_chunks) > 0:
            # All documents already processed - skip to embeddings/GraphRAG
            logger.info(
                f"All {len(documents)} documents already have chunks. "
                f"Skipping MinerU processing, proceeding directly to embeddings/GraphRAG."
            )

            # Go straight to embeddings (which will trigger GraphRAG automatically)
            # Pass empty document_results since documents are already processed
            task = generate_collection_embeddings_simple.delay(
                document_results=[],
                collection_id=str(collection.id)
            )

            storage.update_collection_status(
                collection.id, "processing", task_id=task.id
            )

            return ApiResponseV2(
                success=True,
                data=TaskSubmissionResponse(
                    task_id=task.id,
                    task_type="embeddings_and_graphrag",
                    status=TaskState.PENDING,
                    submitted_at=datetime.utcnow(),
                    collection_id=collection.id,
                ).dict(),
                message=f"Skipped re-processing {len(documents_with_chunks)} documents (already have chunks). Generating embeddings and GraphRAG index.",
                timestamp=datetime.utcnow(),
            )

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
            # Should never reach here after validation, but handle defensively
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported operation type: {request.operation_type}"
            )

        # Store task ID in collection for tracking
        storage.update_collection_status(
            collection.id, "processing", task_id=task.id
        )

        # Create response
        response_data = TaskSubmissionResponse(
            task_id=task.id,
            task_type=request.operation_type,
            status=TaskState.PENDING,
            submitted_at=datetime.utcnow(),
            collection_id=collection.id,
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
            collection_id=collection.id,
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
        from fileintel.core.config import get_config
        from fileintel.core.validation import validate_batch_size

        if not request.tasks:
            raise HTTPException(status_code=400, detail="No tasks provided in batch")

        # Validate batch size doesn't exceed configured limit
        config = get_config()
        validate_batch_size(
            request.tasks,
            config.batch_processing.max_processing_batch_size,
            "collections"
        )

        submitted_tasks = []
        failed_submissions = []  # Track failures instead of silently skipping
        batch_id = str(uuid.uuid4())

        for task_request in request.tasks:
            try:
                # Validate collection exists
                collection = get_collection_by_id_or_name(
                    task_request.collection_identifier, storage
                )
                if not collection:
                    failed_submissions.append({
                        "collection_identifier": task_request.collection_identifier,
                        "error": "Collection not found"
                    })
                    logger.warning(
                        f"Collection {task_request.collection_identifier} not found"
                    )
                    continue

                # Get documents for this collection
                documents = storage.get_documents_by_collection(collection.id)
                # Extract file paths from document metadata
                file_paths = []
                for doc in documents:
                    # Try both metadata and document_metadata fields
                    file_path = (
                        doc.metadata.get("file_path") if doc.metadata
                        else doc.document_metadata.get("file_path") if doc.document_metadata
                        else None
                    )
                    if file_path:
                        file_paths.append(file_path)

                if not file_paths:
                    failed_submissions.append({
                        "collection_identifier": task_request.collection_identifier,
                        "error": "No documents found in collection"
                    })
                    logger.warning(
                        f"No documents found for collection {collection.id}"
                    )
                    continue

                # Submit task based on workflow type
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
                    submitted_tasks.append(task.id)
                else:
                    # Sequential workflow: use Celery chain for ordered execution
                    from celery import chain

                    # Create chain of tasks (executed one after another)
                    task_chain = chain(
                        complete_collection_analysis.s(
                            collection_id=str(collection.id),
                            file_paths=file_paths,
                            build_graph=task_request.build_graph,
                            extract_metadata=task_request.extract_metadata,
                            generate_embeddings=task_request.generate_embeddings,
                            batch_id=batch_id,
                            **task_request.parameters,
                        )
                    )

                    # Apply chain and get the task ID
                    task = task_chain.apply_async()
                    submitted_tasks.append(task.id)

            except Exception as e:
                # Track any exceptions during task submission
                failed_submissions.append({
                    "collection_identifier": task_request.collection_identifier,
                    "error": str(e)
                })
                logger.error(f"Failed to submit task for collection {task_request.collection_identifier}: {e}")

        if not submitted_tasks and not failed_submissions:
            raise HTTPException(
                status_code=400, detail="No tasks provided"
            )

        if not submitted_tasks:
            raise HTTPException(
                status_code=400,
                detail=f"No valid tasks could be submitted. {len(failed_submissions)} tasks failed."
            )

        response_data = BatchTaskSubmissionResponse(
            batch_id=batch_id,
            task_ids=submitted_tasks,
            submitted_count=len(submitted_tasks),
            failed_count=len(failed_submissions),
            failures=failed_submissions,
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

        # Validate task state if collection is processing
        warning_message = None
        if collection.current_task_id and processing_status == "processing":
            from celery.result import AsyncResult

            task = AsyncResult(collection.current_task_id)

            # Check if task state contradicts collection status
            if task.state in ["SUCCESS", "FAILURE", "REVOKED"]:
                warning_message = (
                    f"Collection status is 'processing' but task {task.state}. "
                    "This may indicate a callback failure. Task ID: {collection.current_task_id}"
                )
                logger.warning(f"Stale status detected for collection {collection.id}: {warning_message}")

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
            "current_task_id": collection.current_task_id,
            "status_updated_at": collection.status_updated_at.isoformat()
            if collection.status_updated_at
            else None,
        }

        # Add warning if status appears stale
        if warning_message:
            status_info["warning"] = warning_message

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
        from fileintel.core.validation import validate_batch_size, validate_file_size
        import os
        from pathlib import Path

        try:
            import aiofiles
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="aiofiles package not available for async file operations",
            )

        config = get_config()

        # Validate batch size
        validate_batch_size(
            files,
            config.batch_processing.max_upload_batch_size,
            "files"
        )

        # Validate individual file sizes
        for file in files:
            validate_file_size(file, config.batch_processing.max_file_size_mb)

        # Validate collection exists
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        config = get_config()
        uploaded_files = []
        duplicate_files = []

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

            # Check for duplicate in this collection
            existing_document = storage.get_document_by_hash_and_collection(
                content_hash, collection.id
            )

            if existing_document:
                logger.info(
                    f"Duplicate detected in batch: {file.filename} (hash: {content_hash[:16]}...) "
                    f"already exists as document {existing_document.id}"
                )

                # Track duplicate but don't create new document
                duplicate_files.append({
                    "filename": file.filename,
                    "existing_document_id": str(existing_document.id),
                    "content_hash": content_hash,
                    "file_size": file_size,
                    "reason": "Duplicate file already exists in collection"
                })

                # Remove the uploaded file since it's a duplicate
                file_path.unlink(missing_ok=True)

                # Skip to next file
                continue
            else:
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

        if not uploaded_files and not duplicate_files:
            raise HTTPException(status_code=400, detail="No files were uploaded")

        result_data = {
            "uploaded_files": len(uploaded_files),
            "duplicates_skipped": len(duplicate_files),
            "file_paths": uploaded_files,
            "duplicates": duplicate_files,
        }

        # Process immediately if requested (only if there are new files)
        if process_immediately and uploaded_files:
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
        elif process_immediately and not uploaded_files:
            # All files were duplicates
            result_data.update(
                {
                    "processing_status": "skipped",
                    "message": "All files were duplicates, no processing needed",
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
