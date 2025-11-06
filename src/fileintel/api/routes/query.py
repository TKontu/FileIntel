"""
Query API v2 - RAG Query Endpoints.

Provides direct REST API access to RAG functionality without requiring task management.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from fileintel.api.dependencies import (
    get_storage,
    get_collection_by_id_or_name,
    get_api_key,
)
from fileintel.api.error_handlers import api_error_handler, create_success_response
from fileintel.api.models import ApiResponseV2
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.core.config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_api_key)])


class QueryRequest(BaseModel):
    question: str
    search_type: Optional[str] = "adaptive"  # vector, graph, adaptive, global, local
    max_results: Optional[int] = 5
    include_sources: Optional[bool] = True
    query_mode: Optional[str] = "sync"  # "sync" (default) or "async" for backwards compatibility


class QueryResponse(BaseModel):
    answer: str
    sources: list
    query_type: str
    routing_explanation: Optional[str] = None
    collection_id: str
    question: str
    processing_time_ms: Optional[int] = None


@router.post("/collections/{collection_identifier}/query", response_model=ApiResponseV2)
@api_error_handler("query collection")
async def query_collection(
    collection_identifier: str,
    request: QueryRequest,
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """
    Query a collection using RAG (Retrieval-Augmented Generation).

    Supports:
    - Vector search: Semantic similarity search through embeddings
    - Graph search: Relationship-based queries through GraphRAG
    - Adaptive: Automatically routes to best search method
    - Global: GraphRAG global community summaries
    - Local: GraphRAG local entity relationships
    """
    start_time = datetime.now()

    try:
        # Validate collection exists (now async)
        collection = await get_collection_by_id_or_name(collection_identifier, storage)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        # Log query request
        logger.info(
            f"Received query request: collection='{collection.name}' search_type={request.search_type} "
            f"query_mode={request.query_mode} question_length={len(request.question)} max_results={request.max_results}"
        )

        # FORCE ALL QUERIES TO ASYNC MODE - API is non-blocking router only
        # All processing (vector, graph, adaptive) happens in Celery workers
        # This ensures:
        # 1. API can handle high request concurrency (just routing to queues)
        # 2. No blocking operations in API container
        # 3. Consistent architecture: API = router, Workers = processing
        if request.query_mode == "sync":
            logger.info(
                f"Auto-forcing async mode for {request.search_type} query "
                "(API is non-blocking router - all queries run as Celery tasks)"
            )
            request.query_mode = "async"

        # ALL QUERIES: Submit as Celery task (non-blocking)
        return await _submit_query_task(request, collection, storage)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _submit_query_task(
    request: QueryRequest, collection, storage: PostgreSQLStorage
) -> ApiResponseV2:
    """
    Submit query as async Celery task.

    Returns task ID immediately for non-blocking operation.
    Recommended for graph queries which can take 30-60 seconds.
    """
    try:
        # Import Celery tasks
        from fileintel.tasks.graphrag_tasks import (
            query_graph_global,
            query_graph_local,
            adaptive_graphrag_query,
            query_vector,
        )

        # Route to appropriate Celery task based on search type
        task_result = None

        if request.search_type in ["graph", "global"]:
            # Submit global graph query task
            task_result = query_graph_global.delay(
                query=request.question, collection_id=collection.id
            )
            query_type = "graph_global"

        elif request.search_type == "local":
            # Submit local graph query task
            task_result = query_graph_local.delay(
                query=request.question, collection_id=collection.id
            )
            query_type = "graph_local"

        elif request.search_type == "adaptive":
            # Adaptive routing: analyze query and create chain directly
            # This avoids nested task IDs - user gets single task ID for the complete chain
            from celery import chain
            from celery.exceptions import OperationalError as CeleryOperationalError
            import socket

            # Routing logic: determine local vs global strategy
            query_lower = request.question.lower()

            # Keywords that suggest local search
            local_indicators = ["who is", "what is", "tell me about", "specific", "person", "company", "entity"]
            # Keywords that suggest global search
            global_indicators = ["overall", "summary", "general", "trend", "pattern", "across", "all", "total"]

            local_score = sum(1 for indicator in local_indicators if indicator in query_lower)
            global_score = sum(1 for indicator in global_indicators if indicator in query_lower)

            # Default to global search for broad queries, local for specific
            use_local = local_score > global_score and len(request.question.split()) <= 10
            search_type = "local" if use_local else "global"

            logger.info(
                f"Adaptive routing: chose '{search_type}' search "
                f"(local_score={local_score}, global_score={global_score})"
            )

            # Import tasks
            from fileintel.tasks.graphrag_tasks import enhance_adaptive_result

            # Create chain: search task | enhancement task
            # Wrap in try/except to handle Celery broker failures
            try:
                if use_local:
                    task_result = chain(
                        query_graph_local.s(request.question, collection.id),
                        # Use .si() (immutable signature) with explicit kwargs for clarity and type safety
                        enhance_adaptive_result.si(
                            strategy=search_type,
                            local_score=local_score,
                            global_score=global_score,
                            query=request.question,
                            collection_id=collection.id
                        )
                    ).apply_async()
                else:
                    task_result = chain(
                        query_graph_global.s(request.question, collection.id),
                        # Use .si() (immutable signature) with explicit kwargs for clarity and type safety
                        enhance_adaptive_result.si(
                            strategy=search_type,
                            local_score=local_score,
                            global_score=global_score,
                            query=request.question,
                            collection_id=collection.id
                        )
                    ).apply_async()

                query_type = "adaptive"

            except (ConnectionError, CeleryOperationalError, socket.error, TimeoutError) as e:
                logger.error(f"Failed to submit adaptive query chain: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Task queue unavailable: {str(e)}. "
                        "The Celery broker may be down or overloaded. Please try again later."
                    )
                )
            except Exception as e:
                logger.error(f"Unexpected error creating adaptive query chain: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create adaptive query: {str(e)}"
                )

        elif request.search_type == "vector":
            # Submit vector query task
            task_result = query_vector.delay(
                query=request.question,
                collection_id=collection.id,
                top_k=request.max_results
            )
            query_type = "vector"

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported search_type '{request.search_type}'. "
                "Supported: vector, graph, global, local, adaptive",
            )

        # Log task submission
        logger.info(
            f"Submitted async query task: task_id={task_result.id} "
            f"collection='{collection.name}' query_type={query_type}"
        )

        # Return task ID immediately (non-blocking)
        response_data = {
            "task_id": str(task_result.id),
            "status": "processing",
            "query_type": query_type,
            "collection_id": collection.id,
            "question": request.question,
            "message": f"Query submitted for async processing. Use task_id to check status.",
            "status_endpoint": f"/api/v2/tasks/{task_result.id}",
        }

        return ApiResponseV2(
            success=True,
            message="Query submitted successfully",
            data=response_data,
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting async query task: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit async query: {str(e)}"
        )


@router.post(
    "/collections/{collection_identifier}/documents/{document_identifier}/query",
    response_model=ApiResponseV2,
)
@api_error_handler("query document")
async def query_document(
    collection_identifier: str,
    document_identifier: str,
    request: QueryRequest,
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """
    Query a specific document within a collection.

    Uses vector search restricted to chunks from the specified document.
    Routes to Celery worker for consistency with collection queries.
    """
    import asyncio

    try:
        # Validate collection exists (now async)
        collection = await get_collection_by_id_or_name(collection_identifier, storage)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        # Find document in collection (wrap in asyncio.to_thread)
        documents = await asyncio.to_thread(
            storage.get_documents_by_collection, collection.id
        )
        document = None
        for doc in documents:
            if (
                doc.id == document_identifier
                or doc.filename == document_identifier
                or doc.original_filename == document_identifier
            ):
                document = doc
                break

        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_identifier} not found in collection",
            )

        # Log document query request
        logger.info(
            f"Received document query request: collection='{collection.name}' "
            f"document='{document.original_filename}' question_length={len(request.question)}"
        )

        # ARCHITECTURE: Route to Celery worker for consistency
        # Collection queries → Celery (line 47)
        # Document queries → Celery (here)
        # This ensures consistent architecture: API as non-blocking router
        from fileintel.tasks.graphrag_tasks import query_vector

        try:
            # Submit as Celery task (non-blocking)
            task_result = query_vector.delay(
                query=request.question,
                collection_id=collection.id,
                document_id=document.id,  # Restrict to specific document
                top_k=request.max_results or 5
            )

            # Log task submission
            logger.info(
                f"Submitted document query task: task_id={task_result.id} "
                f"collection='{collection.name}' document='{document.original_filename}'"
            )

            # Return task ID immediately (non-blocking)
            response_data = {
                "task_id": str(task_result.id),
                "status": "processing",
                "query_type": "vector_document",
                "collection_id": collection.id,
                "document_id": document.id,
                "question": request.question,
                "message": f"Document query submitted for async processing. Use task_id to check status.",
                "status_endpoint": f"/api/v2/tasks/{task_result.id}",
            }

            return ApiResponseV2(
                success=True,
                message="Document query submitted successfully",
                data=response_data,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Failed to submit document query task: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to submit document query: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/endpoints", response_model=ApiResponseV2)
async def list_query_endpoints() -> ApiResponseV2:
    """
    List all available query endpoints and their usage.

    Provides helpful guidance for users on correct API paths.
    """
    endpoints_info = {
        "available_endpoints": [
            {
                "method": "POST",
                "path": "/api/v2/collections/{collection_id}/query",
                "description": "Query a collection using RAG (Retrieval-Augmented Generation)",
                "parameters": {
                    "collection_id": "Collection ID or name",
                    "request_body": {
                        "question": "Your question (required)",
                        "search_type": "vector|graph|adaptive|global|local (default: adaptive)",
                        "max_results": "Number of results (default: 5)",
                        "include_sources": "Include source documents (default: true)",
                    },
                },
            },
            {
                "method": "POST",
                "path": "/api/v2/collections/{collection_id}/documents/{document_id}/query",
                "description": "Query a specific document within a collection",
                "parameters": {
                    "collection_id": "Collection ID or name",
                    "document_id": "Document ID, filename, or original filename",
                    "request_body": "Same as collection query",
                },
            },
            {
                "method": "GET",
                "path": "/api/v2/query/status",
                "description": "Get query system status and capabilities",
            },
        ],
        "common_mistakes": [
            {
                "incorrect": "/api/v2/query/collection",
                "correct": "/api/v2/collections/{collection_id}/query",
                "note": "Use 'collections' (plural) and include the collection identifier",
            }
        ],
        "example_usage": {
            "curl_example": 'curl -X POST \'/api/v2/collections/my-collection/query\' -H \'Content-Type: application/json\' -d \'{"question": "What is this about?", "search_type": "adaptive"}\''
        },
    }

    return create_success_response(endpoints_info)


@router.get("/query/status", response_model=ApiResponseV2)
@api_error_handler("get query status")
async def get_query_status(
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """
    Get the status of the query system and available capabilities.

    Returns information about available collections, search methods, and system health.
    """
    try:
        # Log status request
        logger.info("Query system status requested")

        # Get collections with their document counts (wrap blocking storage call)
        import asyncio

        collections = await asyncio.to_thread(storage.get_all_collections)
        collection_info = []

        for collection in collections:
            # Wrap blocking storage calls
            documents = await asyncio.to_thread(
                storage.get_documents_by_collection, collection.id
            )
            chunks = await asyncio.to_thread(
                storage.get_all_chunks_for_collection, collection.id
            )

            collection_info.append(
                {
                    "id": collection.id,
                    "name": collection.name,
                    "description": collection.description,
                    "processing_status": getattr(
                        collection, "processing_status", "unknown"
                    ),
                    "document_count": len(documents),
                    "chunk_count": len(chunks),
                    "has_embeddings": any(
                        chunk.embedding is not None for chunk in chunks
                    )
                    if chunks
                    else False,
                }
            )

        # Check system capabilities
        capabilities = {
            "vector_search": True,  # Always available if we have the basic stack
            "graph_search": False,
            "adaptive_routing": False,
        }

        # Test GraphRAG availability
        try:
            from fileintel.rag.graph_rag.services.graphrag_service import (
                GraphRAGService,
            )

            capabilities["graph_search"] = True
            capabilities["adaptive_routing"] = True
        except Exception:
            pass

        status_info = {
            "status": "operational",
            "api_version": "v2",
            "collections": collection_info,
            "total_collections": len(collections),
            "capabilities": capabilities,
            "supported_search_types": [
                "vector",
                "graph",
                "adaptive",
                "global",
                "local",
            ],
            "endpoints": [
                "POST /collections/{id}/query - Query a collection",
                "POST /collections/{id}/documents/{doc_id}/query - Query a document",
                "GET /query/status - Get system status",
            ],
        }

        return create_success_response(status_info)

    except Exception as e:
        logger.error(f"Error getting query status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
