"""
Query API v2 - RAG Query Endpoints.

Provides direct REST API access to RAG functionality without requiring task management.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

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
    question: str = Field(..., description="The question to ask about the collection")
    search_type: Optional[str] = Field(
        default="adaptive",
        description="RAG search type: 'vector' (semantic search), 'graph' (knowledge graph), 'adaptive' (auto-select), 'global' (GraphRAG global), or 'local' (GraphRAG local)"
    )
    max_results: Optional[int] = Field(default=5, description="Maximum number of results to retrieve")
    include_sources: Optional[bool] = Field(default=True, description="Include source citations in response")
    include_cited_chunks: Optional[bool] = Field(default=False, description="Include full text content of cited chunks in response")
    query_mode: Optional[str] = Field(
        default="sync",
        description="Query execution mode: 'sync' (wait for result) or 'async' (return task ID)"
    )
    answer_format: Optional[str] = Field(
        default="default",
        description="Answer format: 'default' (standard), 'single_paragraph' (concise single paragraph), 'table' (markdown table), 'list' (bulleted/numbered), 'json' (structured JSON), 'essay' (multi-paragraph), or 'markdown' (rich formatting)"
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list
    query_type: str
    routing_explanation: Optional[str] = None
    collection_id: str
    question: str
    processing_time_ms: Optional[int] = None


@router.post(
    "/collections/{collection_identifier}/query",
    response_model=ApiResponseV2,
    summary="Query collection using RAG (async task)",
    description="""
    Submit a query to a collection using RAG (Retrieval-Augmented Generation).

    Returns a task ID immediately. Use `/api/v2/tasks/{task_id}` to check status and retrieve results.

    **IMPORTANT: Async-Only Architecture**

    All queries execute asynchronously as Celery tasks regardless of the `query_mode` parameter.
    This ensures the API remains non-blocking and can handle high request concurrency.
    - The API acts as a non-blocking router
    - All processing happens in Celery workers
    - Synchronous mode is NOT supported - `query_mode: "sync"` is automatically converted to async
    - Always returns a `task_id` immediately

    **Search Types:**
    - `vector`: Semantic similarity search (best for factual questions)
    - `graph`: Knowledge graph search (best for relationship queries)
    - `adaptive`: Auto-selects best method using LLM (recommended)
    - `global`: GraphRAG global community summaries (broad analysis)
    - `local`: GraphRAG local entity relationships (specific entities)

    **Answer Formats:**
    - `default`: Standard prose answer
    - `single_paragraph`: Concise single paragraph
    - `table`: Markdown table format
    - `list`: Bulleted/numbered list
    - `json`: Structured JSON
    - `essay`: Multi-paragraph detailed essay
    - `markdown`: Rich markdown formatting

    **Workflow:**
    1. POST to this endpoint → Get `task_id`
    2. GET `/api/v2/tasks/{task_id}` → Monitor progress
    3. When `status=SUCCESS` → Get answer from result

    **Example Request:**
    ```json
    {
      "question": "What are the main themes in the documents?",
      "search_type": "adaptive",
      "max_results": 5,
      "include_sources": true,
      "answer_format": "default"
    }
    ```

    **Example Response:**
    ```json
    {
      "success": true,
      "data": {
        "task_id": "abc-123",
        "status": "processing",
        "query_type": "adaptive_vector",
        "collection_id": "col-456",
        "question": "What are the main themes?",
        "message": "Query submitted for async processing",
        "status_endpoint": "/api/v2/tasks/abc-123"
      }
    }
    ```
    """
)
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
                query=request.question,
                collection_id=collection.id,
                answer_format=request.answer_format,
                include_cited_chunks=request.include_cited_chunks
            )
            query_type = "graph_global"

        elif request.search_type == "local":
            # Submit local graph query task
            task_result = query_graph_local.delay(
                query=request.question,
                collection_id=collection.id,
                answer_format=request.answer_format,
                include_cited_chunks=request.include_cited_chunks
            )
            query_type = "graph_local"

        elif request.search_type == "adaptive":
            # LLM-based adaptive routing: classify query intent and route optimally
            from celery import chain
            from celery.exceptions import OperationalError as CeleryOperationalError
            import socket
            from fileintel.rag.query_classifier import QueryClassifier

            # Initialize classifier with current config
            config = get_config()
            classifier = QueryClassifier(config)

            # Classify query using LLM/hybrid/keyword method (based on config)
            try:
                classification = classifier.classify(request.question)
                query_class = classification["type"].lower()  # "vector", "graph", or "hybrid"
                confidence = classification.get("confidence", 0.0)
                reasoning = classification.get("reasoning", "No reasoning provided")
                method = classification.get("method", "unknown")
                cached = classification.get("cached", False)

                logger.info(
                    f"Adaptive routing classified as '{query_class}' "
                    f"(confidence: {confidence:.2f}, method: {method}, cached: {cached}). "
                    f"Reasoning: {reasoning}"
                )
            except Exception as e:
                # Fallback to graph_global if classification completely fails
                logger.error(f"Query classification failed: {e}. Falling back to graph_global")
                query_class = "graph"
                confidence = 0.5
                reasoning = f"Classification error: {str(e)}"

            # Route based on classification
            try:
                if query_class == "vector":
                    # Route to Vector RAG
                    task_result = query_vector.delay(
                        query=request.question,
                        collection_id=collection.id,
                        top_k=request.max_results,
                        answer_format=request.answer_format,
                        include_cited_chunks=request.include_cited_chunks
                    )
                    query_type = "adaptive_vector"
                    logger.info(f"Adaptive routing: executing vector search (confidence: {confidence:.2f})")

                elif query_class == "graph":
                    # Route to GraphRAG - determine local vs global using heuristic
                    query_lower = request.question.lower()
                    local_indicators = ["who is", "what is", "tell me about", "specific", "person", "company", "entity"]
                    use_local = any(ind in query_lower for ind in local_indicators) and len(request.question.split()) <= 10

                    if use_local:
                        task_result = query_graph_local.delay(
                            request.question,
                            collection.id,
                            answer_format=request.answer_format,
                            include_cited_chunks=request.include_cited_chunks
                        )
                        query_type = "adaptive_graph_local"
                        logger.info(f"Adaptive routing: executing graph local search (confidence: {confidence:.2f})")
                    else:
                        task_result = query_graph_global.delay(
                            request.question,
                            collection.id,
                            answer_format=request.answer_format,
                            include_cited_chunks=request.include_cited_chunks
                        )
                        query_type = "adaptive_graph_global"
                        logger.info(f"Adaptive routing: executing graph global search (confidence: {confidence:.2f})")

                elif query_class == "hybrid":
                    # Execute hybrid query: vector + graph combined
                    from fileintel.tasks.hybrid_tasks import combine_hybrid_results

                    # Create chain: vector → graph → combine
                    task_result = chain(
                        # First: Vector search
                        query_vector.si(
                            query=request.question,
                            collection_id=collection.id,
                            top_k=request.max_results,
                            answer_format=request.answer_format,
                            include_cited_chunks=request.include_cited_chunks
                        ),
                        # Second: Graph search (run in parallel would be better, but chain is simpler)
                        query_graph_global.si(
                            request.question,
                            collection.id,
                            answer_format=request.answer_format,
                            include_cited_chunks=request.include_cited_chunks
                        ),
                        # Third: Combine results using LLM synthesis
                        combine_hybrid_results.s(
                            query=request.question,
                            collection_id=collection.id,
                            answer_format=request.answer_format
                        )
                    ).apply_async()
                    query_type = "adaptive_hybrid"
                    logger.info(f"Adaptive routing: executing hybrid search (vector + graph, confidence: {confidence:.2f})")

                else:
                    # Unknown classification - fallback to graph_global
                    logger.warning(f"Unknown classification type '{query_class}', falling back to graph_global")
                    task_result = query_graph_global.delay(
                        request.question,
                        collection.id,
                        answer_format=request.answer_format,
                        include_cited_chunks=request.include_cited_chunks
                    )
                    query_type = "adaptive_graph_global"

            except (ConnectionError, CeleryOperationalError, socket.error, TimeoutError) as e:
                logger.error(f"Failed to submit adaptive query: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Task queue unavailable: {str(e)}. "
                        "The Celery broker may be down or overloaded. Please try again later."
                    )
                )
            except Exception as e:
                logger.error(f"Unexpected error creating adaptive query: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create adaptive query: {str(e)}"
                )

        elif request.search_type == "vector":
            # Submit vector query task
            task_result = query_vector.delay(
                query=request.question,
                collection_id=collection.id,
                top_k=request.max_results,
                answer_format=request.answer_format,
                include_cited_chunks=request.include_cited_chunks
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
                top_k=request.max_results or 5,
                answer_format=request.answer_format
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
                        "answer_format": "default|single_paragraph|table|list|json|essay|markdown (default: default)",
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

        # Parallelize all collection data queries using asyncio.gather
        collection_data_tasks = []
        for collection in collections:
            # Create parallel tasks for documents and chunks
            collection_data_tasks.append(
                asyncio.gather(
                    asyncio.to_thread(
                        storage.get_documents_by_collection, collection.id
                    ),
                    asyncio.to_thread(
                        storage.get_all_chunks_for_collection, collection.id
                    )
                )
            )

        # Execute all collection queries in parallel
        all_collection_data = await asyncio.gather(*collection_data_tasks)

        # Build collection info from parallel results
        collection_info = []
        for collection, (documents, chunks) in zip(collections, all_collection_data):
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
