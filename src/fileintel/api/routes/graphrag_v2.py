"""
GraphRAG API endpoints for index management and operations.
"""

import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..models import ApiResponseV2
from ..dependencies import get_storage, get_config
from ..services import get_collection_by_identifier
from ...rag.graph_rag.services.graphrag_service import GraphRAGService
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
    storage=Depends(get_storage),
    config=Depends(get_config),
):
    """Create or rebuild GraphRAG index for a collection."""
    try:
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

        # Start background indexing task
        task_result = build_graphrag_index_task.delay(
            collection_id=str(collection.id), force_rebuild=request.force_rebuild
        )

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
    collection_identifier: str, storage=Depends(get_storage), config=Depends(get_config)
):
    """Get GraphRAG index status for a collection."""
    try:
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

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
    storage=Depends(get_storage),
    config=Depends(get_config),
):
    """Get GraphRAG entities for a collection."""
    try:
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

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
        entities_df = pd.read_parquet(entities_file)
        if limit:
            entities_df = entities_df.head(limit)

        # Convert to list of dicts
        entities = []
        for _, row in entities_df.iterrows():
            entity = {
                "name": row.get("name", "Unknown"),
                "type": row.get("type", "Unknown"),
                "description": row.get("description", ""),
                "importance_score": float(row.get("rank", 0.0)),
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
    storage=Depends(get_storage),
    config=Depends(get_config),
):
    """Get GraphRAG communities for a collection."""
    try:
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

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
        communities_df = pd.read_parquet(communities_file)
        if limit:
            communities_df = communities_df.head(limit)

        # Convert to list of dicts
        communities = []
        for _, row in communities_df.iterrows():
            community = {
                "title": row.get("title", "Unknown"),
                "rank": float(row.get("rank", 0.0)),
                "summary": row.get("summary", ""),
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


@router.delete("/{collection_identifier}/index", response_model=ApiResponseV2)
async def remove_graphrag_index(
    collection_identifier: str, storage=Depends(get_storage), config=Depends(get_config)
):
    """Remove GraphRAG index for a collection."""
    try:
        # Get collection by identifier
        collection = await get_collection_by_identifier(storage, collection_identifier)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_identifier}' not found",
            )

        # Remove index
        graphrag_service = GraphRAGService(storage, config)
        result = await graphrag_service.remove_index(collection.id)

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
async def get_graphrag_system_status(storage=Depends(get_storage)):
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
