"""
Metadata extraction API endpoints for document analysis.
"""

import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..models import ApiResponseV2
from ..dependencies import get_storage, get_config
from ..services import get_collection_by_identifier
from ...tasks.llm_tasks import extract_document_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metadata", tags=["Metadata"])


class MetadataExtractionRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to extract metadata for")
    force_reextract: bool = Field(False, description="Force re-extraction even if metadata exists")


class MetadataExtractionResponse(BaseModel):
    task_id: str
    document_id: str
    status: str
    message: str


@router.post("/extract", response_model=ApiResponseV2)
async def extract_document_metadata_endpoint(
    request: MetadataExtractionRequest,
    background_tasks: BackgroundTasks,
    storage=Depends(get_storage),
    config=Depends(get_config),
):
    """Extract metadata from a document using LLM analysis."""
    try:
        # Get document by ID
        document = storage.get_document(request.document_id)
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

        # Get document chunks for analysis
        chunks = storage.get_all_chunks_for_document(request.document_id)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"No processed chunks found for document '{request.document_id}'. Please process document first.",
            )

        # Extract text from chunks (use first 3 chunks for metadata extraction)
        text_chunks = [chunk.chunk_text for chunk in chunks[:3]]

        # Get existing file metadata
        file_metadata = document.document_metadata if document.document_metadata else None

        # Start background metadata extraction task with timeout
        config = get_config()
        task_result = extract_document_metadata.apply_async(
            args=[request.document_id, text_chunks, file_metadata],
            soft_time_limit=config.llm.task_timeout_seconds,
            time_limit=config.llm.task_hard_limit_seconds,
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
    storage=Depends(get_storage)
):
    """Get extracted metadata for a document."""
    try:
        # Get document by ID
        document = storage.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found",
            )

        metadata = document.document_metadata or {}

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


@router.post("/collection/{collection_identifier}/extract-all", response_model=ApiResponseV2)
async def extract_collection_metadata(
    collection_identifier: str,
    force_reextract: bool = Query(False, description="Force re-extraction for all documents"),
    storage=Depends(get_storage),
    config=Depends(get_config),
):
    """Extract metadata for all documents in a collection."""
    try:
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Get all documents in collection
        documents = storage.get_documents_by_collection(collection.id)
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

        for document in documents:
            # Check if metadata already exists
            if not force_reextract and document.document_metadata:
                has_extracted_metadata = (
                    document.document_metadata.get("llm_extracted", False) or
                    document.document_metadata.get("extraction_method") == "llm_analysis"
                )
                if has_extracted_metadata:
                    continue  # Skip documents that already have extracted metadata

            # Get document chunks
            chunks = storage.get_all_chunks_for_document(document.id)
            if not chunks:
                logger.warning(f"No chunks found for document {document.id}, skipping")
                continue

            # Extract text from chunks
            text_chunks = [chunk.chunk_text for chunk in chunks[:3]]

            # Get existing file metadata
            file_metadata = document.document_metadata if document.document_metadata else None

            # Start background task with timeout
            config = get_config()
            task_result = extract_document_metadata.apply_async(
                args=[document.id, text_chunks, file_metadata],
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
                from time import sleep
                sleep(1)  # 1 second pause every 10 tasks
            processed_count += 1

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
    storage=Depends(get_storage),
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

        # Get all documents in collection
        documents = storage.get_documents_by_collection(collection.id)

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


@router.get("/system-status", response_model=ApiResponseV2)
async def get_metadata_system_status(storage=Depends(get_storage)):
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