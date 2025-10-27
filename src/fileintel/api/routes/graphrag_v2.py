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
from ...storage.postgresql_storage import PostgreSQLStorage
from ...tasks.graphrag_tasks import build_graphrag_index_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graphrag", tags=["GraphRAG"])


class GraphRAGIndexRequest(BaseModel):
    collection_id: str = Field(..., description="Collection ID or name to index")
    force_rebuild: bool = Field(False, description="Force rebuild existing index")


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
    """Create or rebuild GraphRAG index for a collection."""
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
            if status.get("status") == "indexed":
                return ApiResponseV2(
                    success=False,
                    message="GraphRAG index already exists. Use force_rebuild=true to rebuild.",
                    data={"status": "exists", "index_status": status},
                    timestamp=datetime.utcnow()
                )

        # Get documents for indexing
        documents = storage.get_documents_by_collection(collection.id)
        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"No documents found in collection '{request.collection_id}'",
            )

        # Get processed chunks
        chunks = []
        for doc in documents:
            doc_chunks = storage.get_all_chunks_for_document(doc.id)
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

    Field Mapping (GraphRAG parquet -> API response):
    - 'title' -> 'name' (entity name from GraphRAG)
    - 'degree' -> 'importance_score' (centrality measure)
    - 'type' -> 'type' (entity type classification)

    Note: This endpoint reads from parquet files (faster) not database.
    The storage layer uses 'entity_name' field when persisting to database.
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
        if status.get("status") != "indexed":
            raise HTTPException(
                status_code=404,
                detail=f"No GraphRAG index found for collection '{collection_identifier}'. Please create index first.",
            )

        # Load entities from parquet files
        import os
        import pandas as pd

        workspace_path = status["index_path"]
        entities_file = os.path.join(workspace_path, "entities.parquet")

        if not os.path.exists(entities_file):
            return ApiResponseV2(
                success=True,
                message=f"No entities found for collection '{collection.name}'",
                data=[],
                timestamp=datetime.utcnow()
            )

        # Read entities and limit results
        try:
            entities_df = pd.read_parquet(entities_file)
        except Exception as e:
            logger.error(f"Failed to read entities parquet file: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load entities from index: {str(e)}"
            )

        if limit:
            if limit < 0:
                # Negative limit means "last N rows"
                entities_df = entities_df.tail(abs(limit))
            else:
                # Positive limit means "first N rows"
                entities_df = entities_df.head(limit)

        # Convert to list of dicts
        entities = []
        for _, row in entities_df.iterrows():
            entity = {
                "name": row.get("title", "Unknown"),  # GraphRAG uses "title" not "name"
                "type": row.get("type", "Unknown"),
                "description": row.get("description", ""),
                "importance_score": float(row.get("degree", 0.0)),  # GraphRAG uses "degree" not "rank"
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

    Returns hierarchical community structure with level information.
    Communities are read directly from GraphRAG parquet files for performance.
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
        if status.get("status") != "indexed":
            raise HTTPException(
                status_code=404,
                detail=f"No GraphRAG index found for collection '{collection_identifier}'. Please create index first.",
            )

        # Load communities from parquet files
        import os
        import pandas as pd

        workspace_path = status["index_path"]
        communities_file = os.path.join(workspace_path, "communities.parquet")

        if not os.path.exists(communities_file):
            return ApiResponseV2(
                success=True,
                message=f"No communities found for collection '{collection.name}'",
                data=[],
                timestamp=datetime.utcnow()
            )

        # Read communities and limit results
        try:
            communities_df = pd.read_parquet(communities_file)
        except Exception as e:
            logger.error(f"Failed to read communities parquet file: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load communities from index: {str(e)}"
            )

        if limit:
            if limit < 0:
                # Negative limit means "last N rows"
                communities_df = communities_df.tail(abs(limit))
            else:
                # Positive limit means "first N rows"
                communities_df = communities_df.head(limit)

        # Convert to list of dicts
        communities = []
        for _, row in communities_df.iterrows():
            # Use summary as title if available and more descriptive than generic "Community N"
            title = row.get("title", "Unknown")
            summary = row.get("summary", "")

            # If title is just "Community N" and we have a summary, extract first sentence as title
            if title.startswith("Community ") and summary:
                # Take first sentence/line of summary as a better title
                first_sentence = summary.split('.')[0] if '.' in summary else summary.split('\n')[0]
                if len(first_sentence) > 10 and len(first_sentence) < 100:
                    title = first_sentence.strip()

            community = {
                "title": title,
                "community_id": row.get("human_readable_id", row.get("id", "N/A")),  # Use human_readable_id or fall back to UUID
                "level": int(row.get("level", 0)),
                "rank": float(row.get("rank", 0.0)),
                "summary": summary,
                "size": int(row.get("size", 0)),
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

    Returns detailed community information including full summary, findings, and content.
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
        if status.get("status") != "indexed":
            raise HTTPException(
                status_code=404,
                detail=f"No GraphRAG index found for collection '{collection_identifier}'. Please create index first.",
            )

        # Load community from parquet file
        import os
        import pandas as pd

        workspace_path = status["index_path"]
        communities_file = os.path.join(workspace_path, "communities.parquet")

        if not os.path.exists(communities_file):
            raise HTTPException(
                status_code=404,
                detail=f"No communities file found for collection '{collection.name}'",
            )

        # Read communities and find the specific one
        try:
            communities_df = pd.read_parquet(communities_file)
        except Exception as e:
            logger.error(f"Failed to read communities parquet file: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load communities from index: {str(e)}"
            )

        # Find community by ID (match both human_readable_id and id columns)
        community_row = None
        for _, row in communities_df.iterrows():
            human_id = str(row.get("human_readable_id", ""))
            uuid_id = str(row.get("id", ""))
            if human_id == community_id or uuid_id == community_id:
                community_row = row
                break

        if community_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Community '{community_id}' not found in collection '{collection.name}'",
            )

        # Build community response with all available fields
        title = community_row.get("title", "Unknown")
        summary = community_row.get("summary", "")

        # If title is just "Community N" and we have a summary, extract first sentence as title
        if title.startswith("Community ") and summary:
            first_sentence = summary.split('.')[0] if '.' in summary else summary.split('\n')[0]
            if len(first_sentence) > 10 and len(first_sentence) < 100:
                title = first_sentence.strip()

        community_data = {
            "title": title,
            "community_id": community_row.get("human_readable_id", community_row.get("id", "N/A")),
            "level": int(community_row.get("level", 0)),
            "rank": float(community_row.get("rank", 0.0)),
            "size": int(community_row.get("size", 0)),
            "summary": summary,
            "full_content": community_row.get("full_content", ""),
            "findings": community_row.get("findings", ""),
        }

        return ApiResponseV2(
            success=True,
            message=f"Found community '{title}' in collection '{collection.name}'",
            data=community_data,
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
