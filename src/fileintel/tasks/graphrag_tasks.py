"""
GraphRAG Celery tasks.

Converts GraphRAG operations to distributed Celery tasks for parallel processing
of entity extraction, community detection, and graph-based querying.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from celery import group, chain, chord

from fileintel.celery_config import app
from .base import BaseFileIntelTask
from fileintel.core.config import get_config

logger = logging.getLogger(__name__)


def prepare_graphrag_data(
    documents: List[Dict[str, Any]], collection_id: str
) -> Dict[str, Any]:
    """
    Pure function to prepare document data for GraphRAG indexing.

    Args:
        documents: List of document data with text content
        collection_id: Collection identifier

    Returns:
        Prepared data structure for GraphRAG
    """
    import pandas as pd
    import os
    from pathlib import Path

    config = get_config()
    root_dir = Path(config.graphrag.index_base_path) / collection_id

    # Create directory structure
    root_dir.mkdir(parents=True, exist_ok=True)
    input_dir = root_dir / "input"
    input_dir.mkdir(exist_ok=True)

    # Prepare documents as text files (GraphRAG expects file inputs)
    prepared_docs = []
    for i, doc in enumerate(documents):
        doc_id = doc.get("document_id", f"doc_{i}")
        content = doc.get("content", "")

        # Write content to file
        file_path = input_dir / f"{doc_id}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        prepared_docs.append(
            {
                "document_id": doc_id,
                "file_path": str(file_path),
                "content_length": len(content),
            }
        )

    return {
        "collection_id": collection_id,
        "root_dir": str(root_dir),
        "input_dir": str(input_dir),
        "documents": prepared_docs,
        "document_count": len(prepared_docs),
    }


def create_graphrag_config(collection_id: str, root_dir: str) -> Dict[str, Any]:
    """
    DEPRECATED: Use GraphRAGConfigAdapter.adapt_config() instead.

    This function creates a dictionary config that was meant to be used with
    GraphRagConfig.from_dict(), but that method is not the correct way to create
    GraphRAG configs. Use the config adapter instead.

    Pure function to create GraphRAG configuration for a collection.

    Args:
        collection_id: Collection identifier
        root_dir: Root directory for GraphRAG data

    Returns:
        GraphRAG configuration dictionary (DEPRECATED)
    """
    config = get_config()

    graphrag_config = {
        "root_dir": root_dir,
        "input": {
            "type": "file",
            "file_type": "text",
            "base_dir": f"{root_dir}/input",
            "file_encoding": "utf-8",
        },
        "cache": {"type": "file", "base_dir": f"{root_dir}/cache"},
        "storage": {"type": "file", "base_dir": f"{root_dir}/output"},
        "llm": {
            "api_key": config.get("llm.openai.api_key"),
            "type": "openai_chat",
            "model": config.graphrag.llm_model,
            "api_base": config.get("llm.openai.base_url"),  # Custom API endpoint
            "max_tokens": config.graphrag.max_tokens,
            "temperature": config.get("llm.temperature", 0.1),
            # Resilience settings - MORE aggressive than Vector RAG due to high failure cost
            # GraphRAG indexing can take 24 hours; failure means losing ALL progress
            "max_retries": 10,  # 10 retries (vs Vector RAG's 3) - GraphRAG failure is catastrophic
            "request_timeout": 600,  # 10 minutes per LLM request (community summarization can be very slow)
            "concurrent_requests": config.graphrag.async_processing.max_concurrent_requests,  # GraphRAG async concurrency
        },
        "embeddings": {
            "api_key": config.get("llm.openai.api_key"),
            "type": "openai_embedding",
            "model": config.graphrag.embedding_model,
            "api_base": config.get("llm.openai.embedding_base_url") or config.get("llm.openai.base_url"),  # Custom API endpoint
            "batch_size": config.graphrag.embedding_batch_max_tokens,
            # Resilience settings - MORE aggressive than Vector RAG due to high failure cost
            # GraphRAG indexing can take 24 hours; failure means losing ALL progress
            "max_retries": 10,  # 10 retries (vs Vector RAG's 3) - GraphRAG failure is catastrophic
            "request_timeout": 60,  # 60s timeout (vs Vector RAG's 30s) - give more time for transient issues
            # Connection pool race condition fixed in models.py with custom httpx client
            # See: docs/graphrag_embedding_todo.md for root cause analysis
            "concurrent_requests": config.graphrag.async_processing.max_concurrent_requests,  # GraphRAG async concurrency
        },
        "chunks": {
            "size": config.rag.chunking.chunk_size,
            "overlap": config.rag.chunking.chunk_overlap,
        },
        "entity_extraction": {"strategy": {"type": "graph_intelligence"}},
        "community_reports": {"strategy": {"type": "graph_intelligence"}},
    }

    return graphrag_config


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="graphrag_indexing",
    soft_time_limit=345600,  # 96 hours - GraphRAG indexing can be very slow for large collections
    time_limit=349200,       # 97 hours hard limit
)
def build_graph_index(
    self, documents: List[Dict[str, Any]], collection_id: str, **kwargs
) -> Dict[str, Any]:
    """
    Build GraphRAG index from documents with distributed processing.

    Args:
        documents: List of documents with content
        collection_id: Collection identifier
        **kwargs: Additional GraphRAG parameters

    Returns:
        Dict containing indexing results
    """
    self.validate_input(
        ["documents", "collection_id"], documents=documents, collection_id=collection_id
    )

    try:
        self.update_progress(0, 5, "Preparing GraphRAG data")

        # Prepare data for GraphRAG
        prepared_data = prepare_graphrag_data(documents, collection_id)
        root_dir = prepared_data["root_dir"]

        self.update_progress(1, 5, "Creating GraphRAG configuration")

        # Import GraphRAG components (with fallback handling)
        try:
            from fileintel.rag.graph_rag._graphrag_imports import (
                build_index,
                GraphRagConfig,
            )
        except ImportError as e:
            logger.warning(f"GraphRAG imports not available: {e}")
            return {
                "collection_id": collection_id,
                "error": "GraphRAG dependencies not available",
                "status": "failed",
            }

        self.update_progress(2, 5, "Initializing GraphRAG indexing")

        # Create configuration using the config adapter (correct approach)
        from fileintel.rag.graph_rag.adapters.config_adapter import GraphRAGConfigAdapter
        config = get_config()
        config_adapter = GraphRAGConfigAdapter()
        config_obj = config_adapter.adapt_config(config, collection_id, config.graphrag.index_base_path)

        # Build the index
        self.update_progress(3, 5, "Building GraphRAG index (this may take a while)")

        # Run indexing (async function - run in event loop)
        import asyncio
        loop = asyncio.get_event_loop()
        future = asyncio.run_coroutine_threadsafe(
            build_index(config=config_obj),
            loop
        )
        index_result = future.result()  # Wait for completion

        self.update_progress(4, 5, "Processing index results")

        # Parse results
        if index_result:
            # Check for output files
            output_dir = Path(root_dir) / "output"
            entities_file = output_dir / "create_final_entities.parquet"
            relationships_file = output_dir / "create_final_relationships.parquet"
            communities_file = output_dir / "create_final_communities.parquet"

            # Count results if files exist
            entities_count = 0
            relationships_count = 0
            communities_count = 0

            if entities_file.exists():
                import pandas as pd

                entities_df = pd.read_parquet(entities_file)
                entities_count = len(entities_df)

            if relationships_file.exists():
                import pandas as pd

                relationships_df = pd.read_parquet(relationships_file)
                relationships_count = len(relationships_df)

            if communities_file.exists():
                import pandas as pd

                communities_df = pd.read_parquet(communities_file)
                communities_count = len(communities_df)

            result = {
                "collection_id": collection_id,
                "root_dir": root_dir,
                "documents_processed": len(documents),
                "entities_extracted": entities_count,
                "relationships_created": relationships_count,
                "communities_detected": communities_count,
                "index_files_created": [
                    str(f) for f in output_dir.glob("*.parquet") if f.exists()
                ],
                "status": "completed",
            }
        else:
            result = {
                "collection_id": collection_id,
                "error": "GraphRAG indexing failed",
                "status": "failed",
            }

        self.update_progress(5, 5, "GraphRAG index building completed")
        return result

    except Exception as e:
        logger.error(
            f"Error building GraphRAG index for collection {collection_id}: {e}"
        )
        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(
    base=BaseFileIntelTask, bind=True, queue="graphrag_queries", rate_limit="60/m", max_retries=3
)
def query_graph_global(
    self, query: str, collection_id: str, answer_format: str = "default", **kwargs
) -> Dict[str, Any]:
    """
    Perform global GraphRAG query across the entire graph.

    Args:
        query: Query string
        collection_id: Collection to query
        answer_format: Answer format template name (default: "default")
        **kwargs: Additional query parameters

    Returns:
        Dict containing query results
    """
    self.validate_input(
        ["query", "collection_id"], query=query, collection_id=collection_id
    )

    from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService
    from fileintel.celery_config import get_shared_storage

    storage = get_shared_storage()

    try:
        self.update_progress(0, 3, "Preparing global GraphRAG query")

        config = get_config()

        # Initialize GraphRAG service
        graphrag_service = GraphRAGService(storage, config)

        self.update_progress(1, 3, "Executing global search with citation tracing")

        # Perform global search using GraphRAG service's query() method
        # This calls global_search() internally + applies citation tracing and formatting
        import asyncio
        loop = asyncio.get_event_loop()
        future = asyncio.run_coroutine_threadsafe(
            graphrag_service.query(
                query,
                collection_id,
                search_type="global",
                answer_format=answer_format
            ),
            loop
        )
        search_result = future.result()  # Wait for completion

        self.update_progress(2, 3, "Processing query results")

        # GraphRAGService.query() returns dict with 'answer', 'sources', 'confidence', 'metadata'
        # The answer is already formatted with Harvard-style citations
        result = {
            "query": query,
            "collection_id": collection_id,
            "answer": search_result.get("answer", ""),
            "sources": search_result.get("sources", []),
            "search_type": "global",
            "confidence": search_result.get("confidence", 0.8),
            "status": "completed",
        }

        self.update_progress(3, 3, "Global GraphRAG query completed")
        return result

    except Exception as e:
        logger.error(f"Error in global GraphRAG query: {e}")
        return {
            "query": query,
            "collection_id": collection_id,
            "error": str(e),
            "status": "failed",
        }
    finally:
        # CRITICAL: Always close storage connection to prevent leaks
        storage.close()


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="graphrag_queries",
    rate_limit="60/m",
    max_retries=3,
)
def query_graph_local(self, query: str, collection_id: str, answer_format: str = "default", **kwargs) -> Dict[str, Any]:
    """
    Perform local GraphRAG query focused on specific entities.

    Args:
        query: Query string
        collection_id: Collection to query
        answer_format: Answer format template name (default: "default")
        **kwargs: Additional query parameters

    Returns:
        Dict containing query results
    """
    self.validate_input(
        ["query", "collection_id"], query=query, collection_id=collection_id
    )

    from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService
    from fileintel.celery_config import get_shared_storage

    storage = get_shared_storage()

    try:
        self.update_progress(0, 3, "Preparing local GraphRAG query")

        config = get_config()

        # Initialize GraphRAG service
        graphrag_service = GraphRAGService(storage, config)

        self.update_progress(1, 3, "Executing local search with citation tracing")

        # Perform local search using GraphRAG service's query() method
        # This calls local_search() internally + applies citation tracing and formatting
        #
        # Use asyncio.run() instead of get_event_loop() + run_coroutine_threadsafe()
        # because local_search contains blocking sync operations that would deadlock
        # if run in an existing event loop. asyncio.run() creates a fresh isolated loop.
        import asyncio
        search_result = asyncio.run(
            graphrag_service.query(
                query,
                collection_id,
                search_type="local",
                answer_format=answer_format
            )
        )

        self.update_progress(2, 3, "Processing query results")

        # GraphRAGService.query() returns dict with 'answer', 'sources', 'confidence', 'metadata'
        # The answer is already formatted with Harvard-style citations
        result = {
            "query": query,
            "collection_id": collection_id,
            "answer": search_result.get("answer", ""),
            "sources": search_result.get("sources", []),
            "search_type": "local",
            "confidence": search_result.get("confidence", 0.9),
            "status": "completed",
        }

        self.update_progress(3, 3, "Local GraphRAG query completed")
        return result

    except Exception as e:
        logger.error(f"Error in local GraphRAG query: {e}")
        return {
            "query": query,
            "collection_id": collection_id,
            "error": str(e),
            "status": "failed",
        }
    finally:
        # CRITICAL: Always close storage connection to prevent leaks
        storage.close()


@app.task(base=BaseFileIntelTask, bind=True, queue="graphrag_queries")
def adaptive_graphrag_query(
    self, query: str, collection_id: str, **kwargs
) -> Dict[str, Any]:
    """
    DEPRECATED: This task is no longer used by the API.

    The API now creates Celery chains directly in src/fileintel/api/routes/query.py
    for better control flow and to avoid nested task ID confusion.

    Chain pattern: query_graph_X.s() | enhance_adaptive_result.s()

    If you need adaptive querying:
    - Use API endpoint: POST /api/v2/collections/{id}/query with search_type="adaptive"
    - API returns single chain task ID for polling
    - Poll GET /api/v2/tasks/{chain_id} for final result

    This task remains for backward compatibility but should not be called directly.
    """
    logger.warning(
        f"adaptive_graphrag_query task called directly (deprecated). "
        f"Use API endpoint POST /api/v2/collections/{{id}}/query instead. "
        f"query='{query}', collection_id='{collection_id}'"
    )

    # For backward compatibility, still execute the query but log deprecation
    self.validate_input(
        ["query", "collection_id"], query=query, collection_id=collection_id
    )

    try:
        self.update_progress(0, 3, "Analyzing query for optimal search strategy")

        # Simple heuristic for choosing search strategy
        query_lower = query.lower()

        # Keywords that suggest local search
        local_indicators = [
            "who is",
            "what is",
            "tell me about",
            "specific",
            "person",
            "company",
            "entity",
        ]
        # Keywords that suggest global search
        global_indicators = [
            "overall",
            "summary",
            "general",
            "trend",
            "pattern",
            "across",
            "all",
            "total",
        ]

        local_score = sum(
            1 for indicator in local_indicators if indicator in query_lower
        )
        global_score = sum(
            1 for indicator in global_indicators if indicator in query_lower
        )

        # Default to global search for broad queries, local for specific
        use_local = local_score > global_score and len(query.split()) <= 10

        search_type = "local" if use_local else "global"
        self.update_progress(1, 3, f"Using {search_type} search strategy")

        # Create non-blocking chain
        self.update_progress(2, 3, f"Creating chain for {search_type} search")

        if use_local:
            task_chain = chain(
                query_graph_local.s(query, collection_id, **kwargs),
                enhance_adaptive_result.s(search_type, local_score, global_score, query, collection_id)
            )
        else:
            task_chain = chain(
                query_graph_global.s(query, collection_id, **kwargs),
                enhance_adaptive_result.s(search_type, local_score, global_score, query, collection_id)
            )

        # Execute chain asynchronously
        chain_result = task_chain.apply_async()

        self.update_progress(3, 3, "Adaptive chain submitted (non-blocking)")

        logger.info(
            f"[DEPRECATED CALL] Adaptive query chain created: strategy={search_type} "
            f"chain_id={chain_result.id} collection={collection_id}"
        )

        # Return task info for backward compatibility
        # NOTE: This is metadata, not the actual query result!
        # Poll the chain_id (task_id) to get the final result
        return {
            "status": "processing",
            "task_id": str(chain_result.id),
            "query": query,
            "collection_id": collection_id,
            "adaptive_strategy": search_type,
            "strategy_reasoning": f"Local score: {local_score}, Global score: {global_score}",
            "message": "Adaptive query submitted as non-blocking chain. Poll task_id for result.",
            "warning": "This task is deprecated. Use API endpoint instead."
        }

    except Exception as e:
        logger.error(f"Error creating adaptive GraphRAG query chain: {e}")
        return {
            "query": query,
            "collection_id": collection_id,
            "error": str(e),
            "status": "failed",
        }


@app.task(queue="graphrag_queries")
def enhance_adaptive_result(
    search_result: Dict[str, Any],
    *,  # Force all following parameters to be keyword-only
    strategy: str,
    local_score: int,
    global_score: int,
    query: str,
    collection_id: str
) -> Dict[str, Any]:
    """
    Enhance search result with adaptive strategy metadata.

    This task receives the result from a chained GraphRAG search task
    (query_graph_local or query_graph_global) and enriches it with
    information about why that strategy was chosen.

    Used in Celery chain to avoid worker blocking:
    chain(query_graph_X.s(...), enhance_adaptive_result.s(...))

    CRITICAL: This task is designed for Celery chains.
    - search_result: Automatically passed from previous task (query_graph_X)
    - Other parameters: Passed explicitly via .s() at chain creation

    Args:
        search_result: Result from query_graph_local or query_graph_global
        strategy: Strategy used ("local" or "global")
        local_score: Score for local search indicators
        global_score: Score for global search indicators
        query: Original query string
        collection_id: Collection ID

    Returns:
        Enhanced result dict with adaptive_strategy and strategy_reasoning
    """
    logger.info(f"Enhancing adaptive result for collection {collection_id} with strategy '{strategy}'")

    # Validate input from previous task
    if not isinstance(search_result, dict):
        logger.error(
            f"enhance_adaptive_result received invalid input: {type(search_result)}. "
            f"Expected dict from query_graph_{strategy} task."
        )
        return {
            "query": query,
            "collection_id": collection_id,
            "status": "failed",
            "error": f"Chain error: Previous task returned invalid result type: {type(search_result)}",
            "adaptive_strategy": strategy,
            "strategy_reasoning": f"Local score: {local_score}, Global score: {global_score}"
        }

    # Check if previous task failed
    if search_result.get("status") == "failed":
        logger.warning(
            f"enhance_adaptive_result received failed result from {strategy} search: "
            f"{search_result.get('error', 'Unknown error')}"
        )
        # Preserve error from previous task but add adaptive metadata
        result = search_result.copy()
        result["adaptive_strategy"] = strategy
        result["strategy_reasoning"] = f"Local score: {local_score}, Global score: {global_score}"
        result["chain_note"] = f"Search task ({strategy}) failed before enhancement"
        return result

    # Enhance successful result with adaptive metadata
    result = search_result.copy()
    result["adaptive_strategy"] = strategy
    result["strategy_reasoning"] = f"Local score: {local_score}, Global score: {global_score}"

    # Preserve original query and collection info if not already present
    if "query" not in result:
        result["query"] = query
    if "collection_id" not in result:
        result["collection_id"] = collection_id

    logger.info(
        f"Successfully enhanced adaptive result: strategy={strategy}, "
        f"has_answer={('answer' in result)}, status={result.get('status', 'unknown')}"
    )

    return result


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="rag_processing",
    rate_limit="120/m",  # Vector queries are faster, allow higher rate
    max_retries=3
)
def query_vector(
    self, query: str, collection_id: str, top_k: int = 5, **kwargs
) -> Dict[str, Any]:
    """
    Perform vector similarity search with RAG.

    Args:
        query: Query string
        collection_id: Collection to query
        top_k: Number of similar chunks to retrieve
        **kwargs: Additional query parameters

    Returns:
        Dict containing query results
    """
    self.validate_input(
        ["query", "collection_id"], query=query, collection_id=collection_id
    )

    config = get_config()
    from fileintel.celery_config import get_shared_storage
    from fileintel.rag.vector_rag.services.vector_rag_service import VectorRAGService

    storage = get_shared_storage()

    try:
        self.update_progress(0, 3, "Preparing vector RAG query")

        self.update_progress(1, 3, "Executing vector similarity search")

        # Initialize vector service and execute query
        vector_service = VectorRAGService(config, storage)
        result = vector_service.query(
            query=query,
            collection_id=collection_id,
            top_k=top_k,
            **kwargs
        )

        self.update_progress(2, 3, "Formatting results")

        # Format result for task response
        task_result = {
            "query": query,
            "collection_id": collection_id,
            "answer": result.get("answer", "No answer found"),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "search_type": "vector",
            "chunks_retrieved": result.get("metadata", {}).get("chunks_retrieved", 0),
            "status": "completed",
        }

        self.update_progress(3, 3, "Vector RAG query completed")
        return task_result

    except Exception as e:
        logger.error(f"Error in vector RAG query: {e}")
        return {
            "query": query,
            "collection_id": collection_id,
            "error": str(e),
            "status": "failed",
        }
    finally:
        # CRITICAL: Always close storage connection to prevent leaks
        storage.close()


@app.task(base=BaseFileIntelTask, bind=True, queue="rag_processing")
def get_graphrag_index_status(self, collection_id: str) -> Dict[str, Any]:
    """
    Check the status of GraphRAG index for a collection.

    Args:
        collection_id: Collection identifier

    Returns:
        Dict containing index status information
    """
    self.validate_input(["collection_id"], collection_id=collection_id)

    try:
        config = get_config()
        root_dir = Path(config.graphrag.index_base_path) / collection_id
        output_dir = root_dir / "output"

        if not root_dir.exists():
            return {
                "collection_id": collection_id,
                "index_exists": False,
                "status": "no_index",
            }

        # Check for key index files
        key_files = [
            "create_final_entities.parquet",
            "create_final_relationships.parquet",
            "create_final_communities.parquet",
        ]

        file_status = {}
        total_size = 0

        for filename in key_files:
            file_path = output_dir / filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_status[filename] = {
                    "exists": True,
                    "size_bytes": file_size,
                    "modified": file_path.stat().st_mtime,
                }
                total_size += file_size
            else:
                file_status[filename] = {"exists": False}

        # Count entities, relationships, communities if files exist
        counts = {}
        if output_dir.exists():
            import pandas as pd

            for filename in key_files:
                file_path = output_dir / filename
                if file_path.exists():
                    try:
                        df = pd.read_parquet(file_path)
                        counts[filename.replace(".parquet", "")] = len(df)
                    except Exception:
                        counts[filename.replace(".parquet", "")] = 0

        return {
            "collection_id": collection_id,
            "index_exists": any(status["exists"] for status in file_status.values()),
            "root_dir": str(root_dir),
            "total_size_bytes": total_size,
            "files": file_status,
            "counts": counts,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error checking GraphRAG index status: {e}")
        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="graphrag_indexing",
    soft_time_limit=345600,  # 96 hours - GraphRAG indexing can be very slow for large collections
    time_limit=349200,       # 97 hours hard limit
)
def build_graphrag_index_task(
    self, collection_id: str, force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Build GraphRAG index for a collection from existing document chunks.

    This task now supports checkpoint resume, allowing automatic recovery from
    failures without losing progress.

    Note: GraphRAG indexing can take hours for large collections due to:
    - Entity extraction for each chunk (~1-3s per chunk)
    - Community detection algorithms (scales with relationship count)
    - Community summarization (recursive, multi-level)
    - LLM API latency and rate limits

    Args:
        collection_id: Collection identifier
        force_rebuild: Whether to force full rebuild ignoring checkpoints (default: False)
                      When False, will automatically resume from last checkpoint if available

    Returns:
        Dict containing indexing results
    """
    from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService
    from fileintel.celery_config import get_shared_storage
    from fileintel.core.config import get_config

    self.validate_input(["collection_id"], collection_id=collection_id)

    try:
        logger.info(f"Starting GraphRAG index build for collection {collection_id}")

        if force_rebuild:
            logger.info("🔄 Force rebuild requested - ignoring checkpoints")
        else:
            logger.info("📌 Checkpoint resume enabled - will resume from last checkpoint if available")

        self.update_progress(0, 5, "Initializing GraphRAG indexing")

        # Initialize services
        config = get_config()
        storage = get_shared_storage()
        try:
            graphrag_service = GraphRAGService(storage, config)

            # Get collection
            collection = storage.get_collection(collection_id)
            if not collection:
                return {
                    "collection_id": collection_id,
                    "error": f"Collection {collection_id} not found",
                    "status": "failed",
                }

            self.update_progress(1, 5, "Loading document chunks")

            # Get all document chunks for the collection
            documents = storage.get_documents_by_collection(collection_id)
            if not documents:
                return {
                    "collection_id": collection_id,
                    "error": "No documents found in collection",
                    "status": "failed",
                }

            # Get chunks for GraphRAG processing
            # For two-tier chunking, use graph chunks; otherwise use all chunks
            config = get_config()
            if getattr(config.rag, 'enable_two_tier_chunking', False):
                all_chunks = storage.get_chunks_by_type_for_collection(collection_id, 'graph')
                logger.info(f"Two-tier chunking enabled: using graph chunks for GraphRAG indexing")
            else:
                all_chunks = storage.get_all_chunks_for_collection(collection_id)

            if not all_chunks:
                return {
                    "collection_id": collection_id,
                    "error": "No processed chunks found. Please process documents first.",
                    "status": "failed",
                }

            self.update_progress(
                2, 5, f"Building GraphRAG index from {len(all_chunks)} chunks"
            )

            # Use GraphRAG service to build index with checkpoint resume
            import asyncio
            import concurrent.futures

            # Determine if resume should be enabled
            # force_rebuild=True -> disable resume (start from scratch)
            # force_rebuild=False -> enable resume (use checkpoints)
            enable_resume = not force_rebuild

            # Set status to "building" when indexing starts
            storage.update_graphrag_index_status(collection_id, "building")
            logger.info(f"Set GraphRAG index status to 'building' for collection {collection_id}")

            # Wrap indexing in try/except to ensure status is set correctly on failure
            try:
                # For gevent: Run async code using asyncio.run_coroutine_threadsafe
                # This submits the coroutine to the event loop from outside
                loop = asyncio.get_event_loop()
                future = asyncio.run_coroutine_threadsafe(
                    graphrag_service.build_index_with_resume(
                        all_chunks,
                        collection_id,
                        enable_resume=enable_resume,
                        validate_checkpoints=True,
                    ),
                    loop
                )
                workspace_path = future.result()  # Wait for completion

                self.update_progress(4, 5, "GraphRAG index completed")

                # Note: Status will be set to "ready" by GraphRAGService after ALL validations pass
                # This prevents race condition where status is "ready" before PostgreSQL save + validation complete

            except Exception as index_error:
                # If indexing fails (timeout, incomplete embeddings, etc.), set status to "error"
                logger.error(f"Indexing failed for collection {collection_id}: {index_error}")

                # Use atomic status update with retry logic
                # This prevents status from being stuck at "building" if update fails
                error_message = str(index_error)[:500]  # Limit error message length
                retry_count = 0
                max_retries = 3
                retry_delay = 1.0  # Start with 1 second

                while retry_count < max_retries:
                    try:
                        storage.update_graphrag_index_status_atomic(
                            collection_id,
                            "error",
                            error_message
                        )
                        logger.info(
                            f"Atomically set GraphRAG index status to 'error' for collection {collection_id} "
                            f"(attempt {retry_count + 1}/{max_retries})"
                        )
                        break  # Success, exit retry loop
                    except Exception as status_error:
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.critical(
                                f"CRITICAL: Failed to update status to 'error' after {max_retries} retries. "
                                f"Collection {collection_id} status may be stuck at 'building'. "
                                f"Manual intervention required. Last error: {status_error}"
                            )
                            # Continue to re-raise original error even if status update failed
                        else:
                            logger.warning(
                                f"Status update failed (attempt {retry_count}/{max_retries}), "
                                f"retrying in {retry_delay}s: {status_error}"
                            )
                            import time
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff

                # Re-raise original error (not status error)
                raise index_error

            # Get final status
            future = asyncio.run_coroutine_threadsafe(
                graphrag_service.get_index_status(collection_id),
                loop
            )
            status = future.result()

            result = {
                "collection_id": collection_id,
                "workspace_path": workspace_path,
                "documents_processed": len(documents),
                "chunks_processed": len(all_chunks),
                "status": "completed",
                "index_status": status,
            }

            self.update_progress(5, 5, "GraphRAG indexing task completed")
            return result
        finally:
            storage.close()

            # Force aggressive memory cleanup after GraphRAG processing
            # GraphRAG loads large DataFrames (entities, communities, relationships)
            # that can consume 2-6GB. Without cleanup, worker holds memory during
            # fork(), causing OOM kills due to copy-on-write memory pressure.
            import gc
            gc.collect()  # Force garbage collection
            logger.info("Completed memory cleanup after GraphRAG task")

    except Exception as e:
        logger.error(
            f"Error building GraphRAG index for collection {collection_id}: {e}"
        )

        # Set status to "failed" when indexing fails (ensure connection cleanup)
        try:
            storage = get_shared_storage()
            try:
                storage.update_graphrag_index_status(collection_id, "failed")
                logger.info(f"Set GraphRAG index status to 'failed' for collection {collection_id}")
            finally:
                # CRITICAL: Always close storage connection to prevent leaks
                storage.close()
        except Exception as status_error:
            logger.error(f"Failed to update status to 'failed': {status_error}")

        # Force cleanup even on failure to prevent memory leaks
        import gc
        gc.collect()
        return {"collection_id": collection_id, "error": str(e), "status": "failed"}


@app.task(base=BaseFileIntelTask, bind=True, queue="rag_processing")
def remove_graphrag_index(self, collection_id: str) -> Dict[str, Any]:
    """
    Remove GraphRAG index for a collection.

    Args:
        collection_id: Collection identifier

    Returns:
        Dict with removal status and details
    """
    from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService
    from fileintel.celery_config import get_shared_storage

    try:
        logger.info(f"Removing GraphRAG index for collection {collection_id}")

        # Initialize services
        config = get_config()
        storage = get_shared_storage()
        try:
            graphrag_service = GraphRAGService(storage=storage, settings=config)
            # Remove index using GraphRAG service
            import asyncio

            loop = asyncio.get_event_loop()
            future = asyncio.run_coroutine_threadsafe(
                graphrag_service.remove_index(collection_id),
                loop
            )
            result = future.result()

            # Clean up database index info
            if hasattr(storage, "remove_graphrag_index_info"):
                storage.remove_graphrag_index_info(collection_id)

            logger.info(f"GraphRAG index removed for collection {collection_id}")

            return {
                "collection_id": collection_id,
                "status": "success",
                "message": f"GraphRAG index removed for collection {collection_id}",
                "details": result,
            }

        finally:
            storage.close()

    except Exception as e:
        logger.error(
            f"Error removing GraphRAG index for collection {collection_id}: {e}"
        )
        return {"collection_id": collection_id, "error": str(e), "status": "failed"}
