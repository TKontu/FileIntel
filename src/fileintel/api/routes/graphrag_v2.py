"""
GraphRAG API endpoints for index management and operations.
"""

import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..models import ApiResponseV2
from ..dependencies import get_storage
from ..services import get_collection_by_identifier
from ...core.config import get_config
from ...rag.graph_rag.services.graphrag_service import GraphRAGService
from ...rag.graph_rag.validators import CompletenessValidator
from ...storage.postgresql_storage import PostgreSQLStorage
from ...tasks.graphrag_tasks import build_graphrag_index_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graphrag", tags=["GraphRAG"])


class GraphRAGIndexRequest(BaseModel):
    collection_id: str = Field(..., description="Collection ID or name to index")
    force_rebuild: bool = Field(
        False,
        description="Force full rebuild from scratch, ignoring checkpoints. "
                    "When False (default), automatically resumes from last successful checkpoint if available."
    )


class GraphRAGIndexResponse(BaseModel):
    task_id: str
    collection_id: str
    status: str
    message: str


@router.post("/index", response_model=ApiResponseV2)
async def create_graphrag_index(
    request: GraphRAGIndexRequest,
    background_tasks: BackgroundTasks,
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """
    Create or rebuild GraphRAG index for a collection.

    This endpoint now supports automatic checkpoint resume, allowing indexing to
    continue from the last successful workflow step if interrupted.

    - force_rebuild=False (default): Automatically resumes from checkpoints if available
    - force_rebuild=True: Ignores checkpoints and rebuilds from scratch
    """
    try:
        config = get_config()
        # Get collection by identifier (ID or name)
        collection = await get_collection_by_identifier(storage, request.collection_id)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{request.collection_id}' not found",
            )

        # Check if index already exists
        graphrag_service = GraphRAGService(storage, config)

        if not request.force_rebuild:
            status = await graphrag_service.get_index_status(collection.id)
            # Check for "ready" (schema value) or "indexed" (legacy compatibility)
            # Allow resume if status is "building" (checkpoint resume enabled)
            if status.get("status") in ["ready", "indexed"]:
                return ApiResponseV2(
                    success=False,
                    message="GraphRAG index already exists. Use force_rebuild=true to rebuild.",
                    data={"status": "exists", "index_status": status},
                    timestamp=datetime.utcnow()
                )

        # Get documents for indexing (wrap blocking call)
        import asyncio
        documents = await asyncio.to_thread(
            storage.get_documents_by_collection, collection.id
        )
        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"No documents found in collection '{request.collection_id}'",
            )

        # Get processed chunks in parallel for all documents
        chunk_tasks = [
            asyncio.to_thread(storage.get_all_chunks_for_document, doc.id)
            for doc in documents
        ]
        all_chunks = await asyncio.gather(*chunk_tasks)

        # Flatten results
        chunks = []
        for doc_chunks in all_chunks:
            chunks.extend(doc_chunks)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"No processed chunks found in collection '{request.collection_id}'. Please process documents first.",
            )

        # Log index creation request
        logger.info(
            f"Received GraphRAG index creation: collection='{collection.name}' ({collection.id}) "
            f"documents={len(documents)} chunks={len(chunks)} force_rebuild={request.force_rebuild}"
        )

        # Start background indexing task
        task_result = build_graphrag_index_task.delay(
            collection_id=str(collection.id), force_rebuild=request.force_rebuild
        )

        # Log task submission
        logger.info(f"Submitted GraphRAG indexing task: task_id={task_result.id} collection='{collection.name}'")

        response_data = GraphRAGIndexResponse(
            task_id=str(task_result.id),
            collection_id=str(collection.id),
            status="started",
            message=f"GraphRAG indexing started for collection '{collection.name}'",
        )

        return ApiResponseV2(
            success=True,
            message="GraphRAG indexing task started successfully",
            data=response_data.dict(),
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start GraphRAG indexing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start GraphRAG indexing: {str(e)}"
        )


@router.get("/{collection_identifier}/status", response_model=ApiResponseV2)
async def get_graphrag_status(
    collection_identifier: str, storage: PostgreSQLStorage = Depends(get_storage)
):
    """Get GraphRAG index status for a collection."""
    try:
        config = get_config()
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Log status request
        logger.info(f"GraphRAG status requested: collection='{collection.name}' ({collection.id})")

        # Get index status
        graphrag_service = GraphRAGService(storage, config)
        status = await graphrag_service.get_index_status(collection.id)

        return ApiResponseV2(
            success=True,
            message=f"GraphRAG status for collection '{collection.name}'",
            data=status,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get GraphRAG status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get GraphRAG status: {str(e)}"
        )


@router.get("/{collection_identifier}/entities", response_model=ApiResponseV2)
async def get_graphrag_entities(
    collection_identifier: str,
    limit: Optional[int] = Query(
        20, description="Maximum number of entities to return"
    ),
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """
    Get GraphRAG entities for a collection.

    Returns entities from database with pagination support, ordered by importance score.
    Fast database queries (< 100ms) replace slower parquet file loading (2-5s).
    """
    try:
        config = get_config()
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Log entities request
        logger.info(f"GraphRAG entities requested: collection='{collection.name}' limit={limit}")

        # Check if index exists
        graphrag_service = GraphRAGService(storage, config)
        status = await graphrag_service.get_index_status(collection.id)
        if status.get("status") not in ["ready", "indexed"]:
            raise HTTPException(
                status_code=404,
                detail=f"No GraphRAG index found for collection '{collection_identifier}'. Please create index first.",
            )

        # Query entities from database with limit
        import asyncio
        entities_data = await asyncio.to_thread(
            storage.get_graphrag_entities,
            collection.id,
            limit=limit if limit and limit > 0 else None
        )

        if not entities_data:
            logger.warning(f"No entities found for collection {collection.id}")
            return ApiResponseV2(
                success=True,
                message=f"No entities found for collection '{collection.name}'",
                data=[],
                timestamp=datetime.utcnow()
            )

        # Transform to API format
        entities = []
        for entity_record in entities_data:
            entity = {
                "name": entity_record.get("name"),  # Storage returns "name"
                "type": entity_record.get("type"),
                "description": entity_record.get("description", ""),
                "importance_score": entity_record.get("importance_score", 0),  # Already integer
            }
            entities.append(entity)

        return ApiResponseV2(
            success=True,
            message=f"Found {len(entities)} entities in collection '{collection.name}'",
            data=entities,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get GraphRAG entities: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get GraphRAG entities: {str(e)}"
        )


@router.get("/{collection_identifier}/communities", response_model=ApiResponseV2)
async def get_graphrag_communities(
    collection_identifier: str,
    limit: Optional[int] = Query(
        10, description="Maximum number of communities to return"
    ),
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """
    Get GraphRAG communities for a collection.

    Returns hierarchical community structure with level information from database.
    Fast database queries (< 100ms) replace slower parquet file loading (2-5s).
    """
    try:
        config = get_config()
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Log communities request
        logger.info(f"GraphRAG communities requested: collection='{collection.name}' limit={limit}")

        # Check if index exists
        graphrag_service = GraphRAGService(storage, config)
        status = await graphrag_service.get_index_status(collection.id)
        if status.get("status") not in ["ready", "indexed"]:
            raise HTTPException(
                status_code=404,
                detail=f"No GraphRAG index found for collection '{collection_identifier}'. Please create index first.",
            )

        # Query communities from database with limit
        import asyncio
        communities_data = await asyncio.to_thread(
            storage.get_graphrag_communities,
            collection.id,
            limit=limit if limit and limit > 0 else None
        )

        if not communities_data:
            logger.warning(f"No communities found for collection {collection.id}")
            return ApiResponseV2(
                success=True,
                message=f"No communities found for collection '{collection.name}'",
                data=[],
                timestamp=datetime.utcnow()
            )

        # Transform to API format
        communities = []
        for community_record in communities_data:
            title = community_record.get("title", "Unknown")
            summary = community_record.get("summary", "")

            # If title is generic "Community N" and we have a summary, extract first sentence as title
            if title.startswith("Community ") and summary:
                first_sentence = summary.split('.')[0] if '.' in summary else summary.split('\n')[0]
                if len(first_sentence) > 10 and len(first_sentence) < 100:
                    title = first_sentence.strip()

            community = {
                "title": title,
                "community_id": str(community_record.get("community_id")),
                "level": community_record.get("level", 0),
                "size": community_record.get("size", 0),
                "summary": summary,
            }
            communities.append(community)

        return ApiResponseV2(
            success=True,
            message=f"Found {len(communities)} communities in collection '{collection.name}'",
            data=communities,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get GraphRAG communities: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get GraphRAG communities: {str(e)}"
        )


@router.get("/{collection_identifier}/communities/{community_id}", response_model=ApiResponseV2)
async def get_graphrag_community_by_id(
    collection_identifier: str,
    community_id: str,
    storage: PostgreSQLStorage = Depends(get_storage),
):
    """
    Get a specific GraphRAG community by ID.

    Returns detailed community information from database including summary and entities.
    Fast database query (< 100ms) replaces slower parquet file loading (2-5s).
    """
    try:
        config = get_config()
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Log community request
        logger.info(f"GraphRAG community requested: collection='{collection.name}' community_id={community_id}")

        # Check if index exists
        graphrag_service = GraphRAGService(storage, config)
        status = await graphrag_service.get_index_status(collection.id)
        if status.get("status") not in ["ready", "indexed"]:
            raise HTTPException(
                status_code=404,
                detail=f"No GraphRAG index found for collection '{collection_identifier}'. Please create index first.",
            )

        # Convert community_id from URL string to integer
        try:
            community_id_int = int(community_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid community_id: must be an integer"
            )

        # Query community from database
        import asyncio
        community_data = await asyncio.to_thread(
            storage.get_graphrag_community_by_id,
            collection.id,
            community_id_int
        )

        if not community_data:
            raise HTTPException(
                status_code=404,
                detail=f"Community '{community_id}' not found in collection '{collection.name}'",
            )

        # Build response with title enhancement
        title = community_data.get("title", "Unknown")
        summary = community_data.get("summary", "")

        # If title is just "Community N" and we have a summary, extract first sentence as title
        if title.startswith("Community ") and summary:
            first_sentence = summary.split('.')[0] if '.' in summary else summary.split('\n')[0]
            if len(first_sentence) > 10 and len(first_sentence) < 100:
                title = first_sentence.strip()

        response_data = {
            "title": title,
            "community_id": str(community_data.get("community_id")),
            "level": community_data.get("level", 0),
            "size": community_data.get("size", 0),
            "summary": summary,
            "entities": community_data.get("entities", []),
        }

        return ApiResponseV2(
            success=True,
            message=f"Found community '{title}' in collection '{collection.name}'",
            data=response_data,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get GraphRAG community: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get GraphRAG community: {str(e)}"
        )


@router.delete("/{collection_identifier}/index", response_model=ApiResponseV2)
async def remove_graphrag_index(
    collection_identifier: str, storage: PostgreSQLStorage = Depends(get_storage)
):
    """Remove GraphRAG index for a collection."""
    try:
        config = get_config()
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Log index removal request
        logger.info(f"GraphRAG index removal requested: collection='{collection.name}' ({collection.id})")

        # Remove index
        graphrag_service = GraphRAGService(storage, config)
        result = await graphrag_service.remove_index(collection.id)

        # Log removal result
        logger.info(f"GraphRAG index removed: collection='{collection.name}' status={result['status']}")

        if result["status"] == "no_index":
            return ApiResponseV2(
                success=False,
                message=f"No GraphRAG index found for collection '{collection.name}'",
                data=result,
                timestamp=datetime.utcnow()
            )

        return ApiResponseV2(
            success=True,
            message=f"GraphRAG index removed for collection '{collection.name}'",
            data=result,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove GraphRAG index: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to remove GraphRAG index: {str(e)}"
        )


@router.get("/{collection_identifier}/completeness", response_model=ApiResponseV2)
async def get_graphrag_completeness(
    collection_identifier: str, storage: PostgreSQLStorage = Depends(get_storage)
):
    """
    Get completeness analysis for a GraphRAG index.

    Returns raw completeness data including:
    - Overall completeness score
    - Per-phase breakdown (entities, communities)
    - Hierarchy-level details for communities
    - Missing item IDs

    No assessment is provided - just raw data for analysis.
    """
    try:
        config = get_config()
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Log completeness request
        logger.info(f"GraphRAG completeness requested: collection='{collection.name}' ({collection.id})")

        # Check if index exists
        graphrag_service = GraphRAGService(storage, config)
        status = await graphrag_service.get_index_status(collection.id)
        if status.get("status") not in ["ready", "indexed", "building"]:
            raise HTTPException(
                status_code=404,
                detail=f"No GraphRAG index found for collection '{collection_identifier}'. Please create index first.",
            )

        # Validate completeness
        from pathlib import Path
        workspace_path = Path(status["index_path"])
        validator = CompletenessValidator(workspace_path)

        reports = validator.validate_all()

        # Calculate overall completeness
        total_items = sum(r.total_items for r in reports.values())
        complete_items = sum(r.complete_items for r in reports.values())
        overall_completeness = complete_items / total_items if total_items > 0 else 0.0

        # Build response
        completeness_data = {
            "collection_id": str(collection.id),
            "collection_name": collection.name,
            "overall_completeness": overall_completeness,
            "total_items": total_items,
            "complete_items": complete_items,
            "missing_items": total_items - complete_items,
            "phases": {
                phase_name: report.to_dict()
                for phase_name, report in reports.items()
            }
        }

        return ApiResponseV2(
            success=True,
            message=f"Completeness analysis for collection '{collection.name}'",
            data=completeness_data,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get GraphRAG completeness: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get GraphRAG completeness: {str(e)}"
        )


@router.get("/status", response_model=ApiResponseV2)
async def get_graphrag_system_status(storage: PostgreSQLStorage = Depends(get_storage)):
    """Get GraphRAG system status."""
    try:
        # Check if GraphRAG dependencies are available
        from ...rag.graph_rag._graphrag_imports import (
            global_search,
            local_search,
            build_index,
        )

        # Basic system status
        status = {
            "status": "operational",
            "graphrag_available": True,
            "supported_operations": [
                "index_creation",
                "global_search",
                "local_search",
                "entity_extraction",
                "community_detection",
            ],
        }

        return ApiResponseV2(
            success=True, message="GraphRAG system is operational", data=status, timestamp=datetime.utcnow()
        )

    except ImportError:
        status = {
            "status": "unavailable",
            "graphrag_available": False,
            "error": "GraphRAG dependencies not available",
        }

        return ApiResponseV2(
            success=False, message="GraphRAG system unavailable", data=status, timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to get GraphRAG system status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get GraphRAG system status: {str(e)}"
        )
