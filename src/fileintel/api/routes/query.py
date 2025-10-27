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
        # Validate collection exists
        collection = get_collection_by_id_or_name(collection_identifier, storage)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        # Log query request
        logger.info(
            f"Received query request: collection='{collection.name}' search_type={request.search_type} "
            f"question_length={len(request.question)} max_results={request.max_results}"
        )

        # Get configuration
        config = get_config()

        # Route the query based on search type
        if request.search_type in ["vector"]:
            result = await _process_vector_query(
                request, collection.id, config, storage
            )
        elif request.search_type in ["graph", "global", "local"]:
            result = await _process_graph_query(request, collection.id, config, storage)
        elif request.search_type == "adaptive":
            result = await _process_adaptive_query(
                request, collection.id, config, storage
            )
        else:
            from fileintel.core.validation import (
                validate_search_type,
                to_http_exception,
                ValidationError,
            )

            try:
                validate_search_type(request.search_type)
            except ValidationError as e:
                raise to_http_exception(e)

        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Log query completion
        logger.info(
            f"Query completed: collection='{collection.name}' query_type={result.get('query_type')} "
            f"processing_time={processing_time}ms"
        )

        # Format response
        query_response = QueryResponse(
            answer=result.get("answer", "No answer generated"),
            sources=result.get("sources", []),
            query_type=result.get("query_type", request.search_type),
            routing_explanation=result.get("routing_explanation"),
            collection_id=collection.id,
            question=request.question,
            processing_time_ms=processing_time,
        )

        return create_success_response(query_response.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_vector_query(
    request: QueryRequest, collection_id: str, config, storage: PostgreSQLStorage
) -> Dict[str, Any]:
    """Process vector-based RAG query."""
    try:
        import asyncio
        from fileintel.rag.vector_rag.services.vector_rag_service import (
            VectorRAGService,
        )

        vector_service = VectorRAGService(config, storage)
        result = await asyncio.to_thread(
            vector_service.query, request.question, collection_id
        )

        return {
            "answer": result.get("answer", "No answer found"),
            "sources": result.get("sources", []),
            "query_type": "vector",
            "routing_explanation": "Processed using vector similarity search",
        }
    except Exception as e:
        logger.error(f"Vector query error: {e}")
        return {
            "answer": "Vector search temporarily unavailable",
            "sources": [],
            "query_type": "vector",
            "routing_explanation": f"Vector search failed: {str(e)}",
        }


async def _process_graph_query(
    request: QueryRequest, collection_id: str, config, storage: PostgreSQLStorage
) -> Dict[str, Any]:
    """Process graph-based RAG query."""
    try:
        from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService

        graph_service = GraphRAGService(storage, config)

        # Route to specific GraphRAG method based on search type
        if request.search_type == "global":
            result = await graph_service.global_query(collection_id, request.question)
        elif request.search_type == "local":
            result = await graph_service.local_query(collection_id, request.question)
        else:  # generic "graph"
            result = await graph_service.query(request.question, collection_id)

        return {
            "answer": result.get("answer", "No answer found"),
            "sources": result.get("sources", []),
            "query_type": f"graph_{request.search_type}"
            if request.search_type in ["global", "local"]
            else "graph",
            "routing_explanation": f"Processed using GraphRAG {request.search_type} search",
        }
    except Exception as e:
        logger.error(f"Graph query error: {e}")
        return {
            "answer": "Graph search temporarily unavailable",
            "sources": [],
            "query_type": "graph",
            "routing_explanation": f"Graph search failed: {str(e)}",
        }


async def _process_adaptive_query(
    request: QueryRequest, collection_id: str, config, storage: PostgreSQLStorage
) -> Dict[str, Any]:
    """Process adaptive query using intelligent routing."""
    try:
        from fileintel.rag.query_orchestrator import QueryOrchestrator
        from fileintel.rag.vector_rag.services.vector_rag_service import (
            VectorRAGService,
        )
        from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService
        from fileintel.rag.query_classifier import QueryClassifier

        # Create services
        vector_service = VectorRAGService(config, storage)
        graph_service = GraphRAGService(storage, config)
        query_classifier = QueryClassifier(config)

        # Create orchestrator
        orchestrator = QueryOrchestrator(
            vector_rag_service=vector_service,
            graphrag_service=graph_service,
            query_classifier=query_classifier,
            config=config.rag,
        )

        # Route query
        result = await orchestrator.route_query(request.question, collection_id)

        return {
            "answer": result.answer,
            "sources": result.sources,
            "query_type": result.query_type,
            "routing_explanation": result.routing_explanation,
        }
    except Exception as e:
        logger.error(f"Adaptive query error: {e}")
        # Fallback to vector search
        return await _process_vector_query(request, collection_id, config, storage)


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
    """
    start_time = datetime.now()

    try:
        # Validate collection exists
        collection = get_collection_by_id_or_name(collection_identifier, storage)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_identifier} not found"
            )

        # Find document in collection
        documents = storage.get_documents_by_collection(collection.id)
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

        # Get configuration
        config = get_config()

        # Process as vector query restricted to this document
        try:
            import asyncio
            from fileintel.rag.vector_rag.services.vector_rag_service import (
                VectorRAGService,
            )

            vector_service = VectorRAGService(config, storage)

            # Query with document restriction
            result = await asyncio.to_thread(
                vector_service.query,
                request.question,
                collection.id,
                document_id=document.id,
            )

            # Calculate processing time
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Log query completion
            logger.info(
                f"Document query completed: document='{document.original_filename}' "
                f"processing_time={processing_time}ms"
            )

            # Format response
            query_response = QueryResponse(
                answer=result.get("answer", "No answer found in document"),
                sources=result.get("sources", []),
                query_type="vector_document",
                routing_explanation=f"Searched within document '{document.original_filename}' using vector similarity",
                collection_id=collection.id,
                question=request.question,
                processing_time_ms=processing_time,
            )

            return create_success_response(query_response.dict())

        except Exception as e:
            logger.error(f"Document query error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error querying document: {str(e)}"
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

        # Get collections with their document counts
        collections = storage.get_all_collections()
        collection_info = []

        for collection in collections:
            documents = storage.get_documents_by_collection(collection.id)
            chunks = storage.get_all_chunks_for_collection(collection.id)

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
