"""
Metadata extraction API endpoints for document analysis.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..models import ApiResponseV2
from ..dependencies import get_storage
from ..services import get_collection_by_identifier
from ...core.config import get_config
from ...tasks.llm_tasks import extract_document_metadata
from ...storage.postgresql_storage import PostgreSQLStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metadata", tags=["Metadata"])


class MetadataExtractionRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to extract metadata for")
    force_reextract: bool = Field(False, description="Force re-extraction even if metadata exists")
    max_chunks: int = Field(6, description="Number of chunks to use for extraction (default: 3)")


class MetadataBulkUpdateRequest(BaseModel):
    updates: List[Dict[str, Any]] = Field(
        ...,
        description="List of metadata updates, each with document_id and metadata dict"
    )
    replace: bool = Field(
        False,
        description="If True, replace existing metadata entirely. If False, merge with existing."
    )


class MetadataExtractionResponse(BaseModel):
    task_id: str
    document_id: str
    status: str
    message: str


@router.post(
    "/extract",
    response_model=ApiResponseV2,
    summary="Extract document metadata (async task)",
    description="""
    Extract structured metadata from a document using LLM analysis.

    Returns task ID immediately. Use `/api/v2/tasks/{task_id}` to check status and retrieve extracted metadata.

    **What Gets Extracted:**
    - Title, authors, publication date
    - Document type (academic paper, report, book, etc.)
    - Abstract/summary
    - Keywords and topics
    - DOI/ISBN/other identifiers
    - Publisher information

    **How It Works:**
    1. Analyzes first N chunks of document (configurable via `max_chunks`)
    2. Uses LLM to extract structured metadata
    3. Merges with existing file metadata
    4. Stores in document_metadata field

    **Parameters:**
    - `document_id` (required): Document UUID to analyze
    - `force_reextract` (default: false): Re-extract even if metadata exists
    - `max_chunks` (default: 6): Number of chunks to analyze (more = better but slower)

    **Workflow:**
    1. POST to this endpoint → Get `task_id`
    2. GET `/api/v2/tasks/{task_id}` → Monitor progress
    3. When `status=SUCCESS` → Metadata available in result
    4. GET `/api/v2/metadata/document/{document_id}` → Retrieve stored metadata

    **Example Request:**
    ```json
    {
      "document_id": "3b9e6ac7-2152-4133-bd87-2cd0ffc09863",
      "force_reextract": false,
      "max_chunks": 6
    }
    ```

    **Example Response:**
    ```json
    {
      "success": true,
      "data": {
        "task_id": "abc-123-def-456",
        "document_id": "3b9e6ac7-2152-4133-bd87-2cd0ffc09863",
        "status": "started",
        "message": "Metadata extraction started for document 'paper.pdf'"
      }
    }
    ```

    **Task Result Format:**
    ```json
    {
      "title": "Introduction to Machine Learning",
      "authors": ["Smith, J.", "Jones, A."],
      "publication_date": "2023-05-15",
      "document_type": "academic_paper",
      "abstract": "This paper introduces...",
      "keywords": ["machine learning", "neural networks"],
      "doi": "10.1234/example.2023.001",
      "llm_extracted": true
    }
    ```
    """
)
async def extract_document_metadata_endpoint(
    request: MetadataExtractionRequest,
    background_tasks: BackgroundTasks,
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """Extract metadata from a document using LLM analysis."""
    import asyncio

    try:
        config = get_config()
        # Get document by ID (wrap blocking storage call)
        document = await asyncio.to_thread(storage.get_document, request.document_id)
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{request.document_id}' not found",
            )

        # Check if metadata already exists
        if not request.force_reextract and document.document_metadata:
            has_extracted_metadata = (
                document.document_metadata.get("llm_extracted", False) or
                document.document_metadata.get("extraction_method") == "llm_analysis"
            )
            if has_extracted_metadata:
                return ApiResponseV2(
                    success=True,
                    message="Document already has extracted metadata. Use force_reextract=true to re-extract.",
                    data={
                        "document_id": request.document_id,
                        "existing_metadata": document.document_metadata,
                        "status": "exists"
                    },
                    timestamp=datetime.utcnow()
                )

        # Get document chunks for analysis (wrap blocking storage call)
        chunks = await asyncio.to_thread(
            storage.get_all_chunks_for_document, request.document_id
        )
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"No processed chunks found for document '{request.document_id}'. Please process document first.",
            )

        # Extract text from chunks (use first N chunks for metadata extraction)
        max_chunks = min(request.max_chunks, len(chunks))  # Don't exceed available chunks
        text_chunks = [chunk.chunk_text for chunk in chunks[:max_chunks]]

        # Get existing file metadata
        file_metadata = document.document_metadata if document.document_metadata else None

        # Log metadata extraction request
        logger.info(
            f"Received metadata extraction request: document_id={request.document_id} "
            f"filename='{document.original_filename}' max_chunks={request.max_chunks} "
            f"force_reextract={request.force_reextract}"
        )

        # Start background metadata extraction task with timeout
        config = get_config()
        task_result = extract_document_metadata.apply_async(
            args=[request.document_id, text_chunks, file_metadata, request.max_chunks],
            soft_time_limit=config.llm.task_timeout_seconds,
            time_limit=config.llm.task_hard_limit_seconds,
        )

        # Log task submission
        logger.info(
            f"Submitted metadata extraction task: task_id={task_result.id} "
            f"document_id={request.document_id}"
        )

        response_data = MetadataExtractionResponse(
            task_id=str(task_result.id),
            document_id=request.document_id,
            status="started",
            message=f"Metadata extraction started for document '{document.original_filename}'",
        )

        return ApiResponseV2(
            success=True,
            message="Metadata extraction task started successfully",
            data=response_data.dict(),
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start metadata extraction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start metadata extraction: {str(e)}"
        )


@router.get("/document/{document_id}", response_model=ApiResponseV2)
async def get_document_metadata(
    document_id: str,
    storage: PostgreSQLStorage = Depends(get_storage)
):
    """Get extracted metadata for a document."""
    import asyncio

    try:
        # Get document by ID (wrap blocking storage call)
        document = await asyncio.to_thread(storage.get_document, document_id)
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found",
            )

        metadata = document.document_metadata or {}

        # Log metadata fetch request
        logger.info(f"Document metadata requested: document_id={document_id} filename='{document.original_filename}'")

        # Check if document has LLM-extracted metadata
        has_extracted_metadata = (
            metadata.get("llm_extracted", False) or
            metadata.get("extraction_method") == "llm_analysis"
        )

        return ApiResponseV2(
            success=True,
            message=f"Metadata for document '{document.original_filename}'",
            data={
                "document_id": document_id,
                "filename": document.original_filename,
                "has_extracted_metadata": has_extracted_metadata,
                "metadata": metadata,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
            },
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document metadata: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get document metadata: {str(e)}"
        )


@router.post(
    "/collection/{collection_identifier}/extract-all",
    response_model=ApiResponseV2,
    summary="Extract metadata for all documents in collection (async tasks)",
    description="""
    Batch extract metadata for all documents in a collection using LLM analysis.

    Submits multiple async tasks (one per document). Use `/api/v2/tasks/{task_id}` to monitor each task.

    **Smart Processing:**
    - Automatically skips documents that already have extracted metadata
    - Only processes documents without LLM metadata (unless force_reextract=true)
    - Submits tasks with rate limiting (pause every 10 tasks to avoid overwhelming LLM)

    **Query Parameters:**
    - `force_reextract` (default: false): Re-extract all documents, even if metadata exists
    - `max_chunks` (default: 3): Number of chunks to analyze per document

    **Use Cases:**
    - Initial metadata extraction for new collection
    - Re-extracting metadata after prompt improvements
    - Filling in missing metadata for partial extractions

    **Workflow:**
    1. POST to this endpoint → Get list of `task_id`s
    2. Monitor each task via `/api/v2/tasks/{task_id}`
    3. Check collection status via `/api/v2/metadata/collection/{id}/status`

    **Example Request:**
    ```bash
    POST /api/v2/metadata/collection/papers/extract-all?force_reextract=false&max_chunks=6
    ```

    **Example Response:**
    ```json
    {
      "success": true,
      "data": {
        "collection_id": "collection-uuid",
        "collection_name": "Research Papers",
        "total_documents": 50,
        "documents_processed": 30,
        "tasks_started": [
          {
            "document_id": "doc-1",
            "filename": "paper1.pdf",
            "task_id": "task-abc-123"
          },
          {
            "document_id": "doc-2",
            "filename": "paper2.pdf",
            "task_id": "task-def-456"
          }
        ]
      },
      "message": "Started metadata extraction for 30 documents in collection 'Research Papers'"
    }
    ```

    **Rate Limiting:**
    - Automatic 1-second pause every 10 tasks
    - Prevents overwhelming LLM provider with concurrent requests
    """
)
async def extract_collection_metadata(
    collection_identifier: str,
    force_reextract: bool = Query(False, description="Force re-extraction for all documents"),
    max_chunks: int = Query(6, description="Number of chunks to analyze per document (default: 6)"),
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """Extract metadata for all documents in a collection."""
    try:
        config = get_config()
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Get all documents in collection (wrap blocking storage call)
        import asyncio
        documents = await asyncio.to_thread(
            storage.get_documents_by_collection, collection.id
        )

        # Log collection extraction request
        logger.info(
            f"Received collection metadata extraction: collection='{collection.name}' ({collection.id}) "
            f"documents={len(documents)} force_reextract={force_reextract} max_chunks={max_chunks}"
        )

        if not documents:
            return ApiResponseV2(
                success=True,
                message=f"No documents found in collection '{collection.name}'",
                data={
                    "collection_id": collection.id,
                    "documents_processed": 0,
                    "tasks_started": []
                },
                timestamp=datetime.utcnow()
            )

        tasks_started = []
        processed_count = 0

        # Filter documents that need processing
        docs_to_process = []
        for document in documents:
            # Check if metadata already exists
            if not force_reextract and document.document_metadata:
                has_extracted_metadata = (
                    document.document_metadata.get("llm_extracted", False) or
                    document.document_metadata.get("extraction_method") == "llm_analysis"
                )
                if has_extracted_metadata:
                    continue  # Skip documents that already have extracted metadata
            docs_to_process.append(document)

        # Parallelize chunk fetching for all documents
        if docs_to_process:
            chunk_tasks = [
                asyncio.to_thread(storage.get_all_chunks_for_document, doc.id)
                for doc in docs_to_process
            ]
            all_doc_chunks = await asyncio.gather(*chunk_tasks)
        else:
            all_doc_chunks = []

        # Process each document with its chunks
        for document, chunks in zip(docs_to_process, all_doc_chunks):
            if not chunks:
                logger.warning(f"No chunks found for document {document.id}, skipping")
                continue

            # Extract text from chunks (use first N chunks)
            num_chunks = min(max_chunks, len(chunks))  # Don't exceed available chunks
            text_chunks = [chunk.chunk_text for chunk in chunks[:num_chunks]]

            # Get existing file metadata
            file_metadata = document.document_metadata if document.document_metadata else None

            # Start background task with timeout
            task_result = extract_document_metadata.apply_async(
                args=[document.id, text_chunks, file_metadata, max_chunks],
                soft_time_limit=config.llm.task_timeout_seconds,
                time_limit=config.llm.task_hard_limit_seconds,
            )

            tasks_started.append({
                "document_id": document.id,
                "filename": document.original_filename,
                "task_id": str(task_result.id),
            })

            # Rate limiting: add small delay every 10 tasks to prevent overwhelming LLM
            if len(tasks_started) % 10 == 0:
                await asyncio.sleep(1)  # 1 second pause every 10 tasks (non-blocking)
            processed_count += 1

        # Log batch completion
        logger.info(
            f"Collection metadata extraction started: collection='{collection.name}' "
            f"tasks_started={len(tasks_started)} processed={processed_count}"
        )

        return ApiResponseV2(
            success=True,
            message=f"Started metadata extraction for {processed_count} documents in collection '{collection.name}'",
            data={
                "collection_id": collection.id,
                "collection_name": collection.name,
                "total_documents": len(documents),
                "documents_processed": processed_count,
                "tasks_started": tasks_started,
            },
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start collection metadata extraction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start collection metadata extraction: {str(e)}"
        )


@router.get("/collection/{collection_identifier}/status", response_model=ApiResponseV2)
async def get_collection_metadata_status(
    collection_identifier: str,
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """Get metadata extraction status for all documents in a collection."""
    try:
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Get all documents in collection (wrap blocking storage call)
        import asyncio
        documents = await asyncio.to_thread(
            storage.get_documents_by_collection, collection.id
        )

        # Log status request
        logger.info(f"Collection metadata status requested: collection='{collection.name}' ({collection.id})")

        status_summary = {
            "total_documents": len(documents),
            "with_extracted_metadata": 0,
            "without_metadata": 0,
            "with_file_metadata_only": 0,
        }

        document_status = []

        for document in documents:
            metadata = document.document_metadata or {}
            has_extracted_metadata = (
                metadata.get("llm_extracted", False) or
                metadata.get("extraction_method") == "llm_analysis"
            )

            if has_extracted_metadata:
                status_summary["with_extracted_metadata"] += 1
                status = "extracted"
            elif metadata:
                status_summary["with_file_metadata_only"] += 1
                status = "file_metadata_only"
            else:
                status_summary["without_metadata"] += 1
                status = "no_metadata"

            document_status.append({
                "document_id": document.id,
                "filename": document.original_filename,
                "status": status,
                "metadata_fields": len(metadata),
                "updated_at": document.updated_at,
            })

        return ApiResponseV2(
            success=True,
            message=f"Metadata status for collection '{collection.name}'",
            data={
                "collection_id": collection.id,
                "collection_name": collection.name,
                "summary": status_summary,
                "documents": document_status,
            },
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection metadata status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get collection metadata status: {str(e)}"
        )


@router.get("/collection/{collection_identifier}/export", response_model=ApiResponseV2)
async def export_collection_metadata(
    collection_identifier: str,
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """Export all documents with metadata for a collection.

    This endpoint returns all documents in a collection with their full metadata
    in a structured format suitable for export to CSV, Markdown, or JSON.
    """
    import asyncio

    try:
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Get all documents with metadata (SINGLE OPTIMIZED QUERY - wrap blocking call)
        documents = await asyncio.to_thread(
            storage.get_documents_by_collection, collection.id
        )

        # Helper function to check if document has LLM-extracted metadata
        def has_llm_metadata(metadata):
            if not metadata:
                return False
            return (
                metadata.get("llm_extracted", False) or
                metadata.get("extraction_method") == "llm_analysis"
            )

        # Transform to export format
        export_docs = [
            {
                "document_id": doc.id,
                "filename": doc.original_filename,
                "has_extracted_metadata": has_llm_metadata(doc.document_metadata),
                "metadata": doc.document_metadata or {},
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
            }
            for doc in documents
        ]

        return ApiResponseV2(
            success=True,
            message=f"Exported metadata for {len(documents)} documents from collection '{collection.name}'",
            data={
                "collection_id": collection.id,
                "collection_name": collection.name,
                "total_documents": len(documents),
                "documents": export_docs,
            },
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export collection metadata: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to export collection metadata: {str(e)}"
        )


@router.post("/bulk-update", response_model=ApiResponseV2)
async def bulk_update_metadata(
    request: MetadataBulkUpdateRequest,
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """Bulk update metadata for multiple documents.

    This endpoint allows updating metadata for multiple documents at once,
    useful for importing corrected metadata from CSV files.

    Args:
        request: Bulk update request with list of document updates
        storage: Database storage instance

    Returns:
        Summary of updates applied
    """
    try:
        if not request.updates:
            raise HTTPException(
                status_code=400,
                detail="No updates provided"
            )

        # Log bulk update request
        logger.info(f"Received bulk metadata update: updates={len(request.updates)} replace={request.replace}")

        import asyncio

        success_count = 0
        failed_updates = []
        updated_documents = []

        for update in request.updates:
            document_id = update.get("document_id")
            metadata = update.get("metadata", {})

            if not document_id:
                failed_updates.append({
                    "document_id": None,
                    "error": "Missing document_id"
                })
                continue

            try:
                # Get document (wrap blocking storage call)
                document = await asyncio.to_thread(storage.get_document, document_id)
                if not document:
                    failed_updates.append({
                        "document_id": document_id,
                        "error": "Document not found"
                    })
                    continue

                # Update metadata (wrap blocking storage call)
                await asyncio.to_thread(
                    storage.update_document_metadata,
                    document_id,
                    metadata,
                    replace=request.replace
                )

                success_count += 1
                updated_documents.append({
                    "document_id": document_id,
                    "filename": document.original_filename,
                    "fields_updated": len(metadata)
                })

            except Exception as e:
                logger.error(f"Failed to update document {document_id}: {e}")
                failed_updates.append({
                    "document_id": document_id,
                    "error": str(e)
                })

        # Log update results
        logger.info(f"Bulk metadata update completed: success={success_count} failed={len(failed_updates)}")

        return ApiResponseV2(
            success=True,
            message=f"Updated metadata for {success_count} documents ({len(failed_updates)} failed)",
            data={
                "total_updates": len(request.updates),
                "success_count": success_count,
                "failed_count": len(failed_updates),
                "updated_documents": updated_documents,
                "failed_updates": failed_updates,
                "replace_mode": request.replace
            },
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to perform bulk metadata update: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform bulk metadata update: {str(e)}"
        )


@router.get("/system-status", response_model=ApiResponseV2)
async def get_metadata_system_status(storage: PostgreSQLStorage = Depends(get_storage)):
    """Get metadata extraction system status."""
    try:
        # Check if metadata extraction dependencies are available
        from ...document_processing.metadata_extractor import MetadataExtractor
        from ...llm_integration.unified_provider import UnifiedLLMProvider
        from ...prompt_management.simple_prompts import load_prompt_components
        from pathlib import Path

        # Check if prompt templates exist
        config = get_config()
        prompts_dir = Path(config.rag.root_dir) / "prompts" / "templates" / "metadata_extraction"
        templates_exist = prompts_dir.exists() and (prompts_dir / "prompt.md").exists()

        # Basic system status
        status = {
            "status": "operational",
            "metadata_extractor_available": True,
            "llm_provider_available": True,
            "prompt_templates_available": templates_exist,
            "supported_operations": [
                "document_metadata_extraction",
                "collection_batch_extraction",
                "metadata_status_monitoring",
            ],
            "prompt_templates_path": str(prompts_dir) if templates_exist else None,
        }

        return ApiResponseV2(
            success=True,
            message="Metadata extraction system is operational",
            data=status,
            timestamp=datetime.utcnow()
        )

    except ImportError as e:
        status = {
            "status": "unavailable",
            "metadata_extractor_available": False,
            "error": f"Metadata extraction dependencies not available: {e}",
        }

        return ApiResponseV2(
            success=False,
            message="Metadata extraction system unavailable",
            data=status,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to get metadata system status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get metadata system status: {str(e)}"
        )