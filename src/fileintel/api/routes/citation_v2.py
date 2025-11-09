"""
Citation generation API endpoints.

Provides endpoints for generating Harvard-style citations for text segments
using vector similarity search to find matching source documents.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..models import ApiResponseV2
from ..dependencies import get_storage
from ..services import get_collection_by_identifier
from ...core.config import get_config
from ...storage.postgresql_storage import PostgreSQLStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/citations", tags=["Citations"])


class CitationRequest(BaseModel):
    """Request model for citation generation."""

    text_segment: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Text segment that needs citation (10-5000 characters)"
    )

    document_id: Optional[str] = Field(
        None,
        description="Optional: Restrict search to specific document"
    )

    min_similarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0-1.0, default from config)"
    )

    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Number of candidate sources to retrieve (default from config)"
    )

    include_llm_analysis: bool = Field(
        False,
        description="Include LLM-based relevance analysis (adds latency)"
    )


class CitationResponse(BaseModel):
    """Response model for citation generation."""

    citation: Dict[str, str] = Field(
        ...,
        description="Citation information with in_text, full, and style fields"
    )

    source: Dict[str, Any] = Field(
        ...,
        description="Source document details including similarity score and excerpt"
    )

    confidence: str = Field(
        ...,
        description="Confidence level: 'high', 'medium', or 'low'"
    )

    relevance_note: Optional[str] = Field(
        None,
        description="Optional LLM-generated relevance explanation (if requested)"
    )

    warning: Optional[str] = Field(
        None,
        description="Warning message if metadata is incomplete"
    )


@router.post(
    "/collections/{collection_identifier}/generate-citation",
    response_model=ApiResponseV2,
    summary="Generate citation for text segment (async task)",
    description="""
    Submit a citation generation task for a text segment using vector similarity search.

    Returns a task ID immediately. Use `/api/v2/tasks/{task_id}` to check status and retrieve results.

    The service:
    1. Validates the input text segment (10-5000 characters)
    2. Uses vector search to find the most similar source document
    3. Validates metadata availability for proper citation
    4. Generates both in-text and full Harvard-style citations
    5. Determines confidence level based on match quality
    6. Optionally provides LLM-based relevance analysis

    Confidence levels:
    - HIGH: similarity ≥0.85 AND complete metadata
    - MEDIUM: similarity ≥0.70 OR has metadata
    - LOW: similarity <0.70 AND no metadata

    **Workflow:**
    1. POST to this endpoint → Get `task_id`
    2. GET `/api/v2/tasks/{task_id}` → Check status
    3. When `status=SUCCESS` → Get citation from result

    Example successful result:
    ```json
    {
      "citation": {
        "in_text": "(Smith et al., 2023, p. 45)",
        "full": "Smith, J., Jones, A., & Brown, B. (2023). Title. Publisher.",
        "style": "harvard"
      },
      "source": {
        "document_id": "doc-123",
        "similarity_score": 0.92,
        "text_excerpt": "...",
        "page_numbers": [45]
      },
      "confidence": "high",
      "status": "completed"
    }
    ```
    """
)
async def generate_citation(
    collection_identifier: str,
    request: CitationRequest,
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """
    Submit citation generation task.

    Args:
        collection_identifier: Collection ID or name
        request: Citation request with text segment and parameters
        storage: PostgreSQL storage instance (dependency injection)

    Returns:
        ApiResponseV2 with task_id for async processing

    Raises:
        HTTPException: If collection not found or input invalid
    """
    import asyncio

    try:
        # Get collection (wrap blocking call)
        collection = await asyncio.to_thread(
            get_collection_by_identifier, storage, collection_identifier
        )
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found"
            )

        # Import citation task
        from fileintel.tasks.llm_tasks import generate_citation as generate_citation_task

        # Submit task to Celery
        task_result = generate_citation_task.delay(
            text_segment=request.text_segment,
            collection_id=collection.id,
            document_id=request.document_id,
            min_similarity=request.min_similarity,
            include_llm_analysis=request.include_llm_analysis,
            top_k=request.top_k
        )

        logger.info(
            f"Submitted citation generation task: task_id={task_result.id} "
            f"collection='{collection.name}' text_length={len(request.text_segment)}"
        )

        # Return task ID immediately (non-blocking)
        response_data = {
            "task_id": str(task_result.id),
            "status": "processing",
            "collection_id": collection.id,
            "collection_name": collection.name,
            "text_segment": request.text_segment[:100] + "..." if len(request.text_segment) > 100 else request.text_segment,
            "message": "Citation generation task submitted. Use task_id to check status.",
            "status_endpoint": f"/api/v2/tasks/{task_result.id}",
        }

        return ApiResponseV2(
            success=True,
            message="Citation generation task submitted successfully",
            data=response_data,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to submit citation generation task: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit citation generation task: {str(e)}"
        )


@router.get(
    "/collections/{collection_identifier}/citation-config",
    response_model=ApiResponseV2,
    summary="Get citation configuration",
    description="Get current citation generation configuration for the system"
)
async def get_citation_config():
    """
    Get citation generation configuration.

    Returns:
        ApiResponseV2 with current citation settings
    """
    try:
        config = get_config()

        citation_config = {
            "min_similarity": config.citation.min_similarity,
            "default_top_k": config.citation.default_top_k,
            "confidence_thresholds": config.citation.confidence_thresholds,
            "max_excerpt_length": config.citation.max_excerpt_length,
            "enable_llm_analysis": config.citation.enable_llm_analysis,
            "llm_analysis_model": config.citation.llm_analysis_model,
        }

        return ApiResponseV2(
            success=True,
            message="Citation configuration retrieved",
            data=citation_config,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Failed to get citation config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get citation configuration: {str(e)}"
        )
