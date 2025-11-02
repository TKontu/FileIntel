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
    root_dir = Path(config.rag.root_dir) / collection_id

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
    Pure function to create GraphRAG configuration for a collection.

    Args:
        collection_id: Collection identifier
        root_dir: Root directory for GraphRAG data

    Returns:
        GraphRAG configuration dictionary
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
            "model": config.rag.llm_model,
            "api_base": config.get("llm.openai.base_url"),  # Custom API endpoint
            "max_tokens": config.rag.max_tokens,
            "temperature": config.get("llm.temperature", 0.1),
            # Resilience settings - MORE aggressive than Vector RAG due to high failure cost
            # GraphRAG indexing can take 24 hours; failure means losing ALL progress
            "max_retries": 10,  # 10 retries (vs Vector RAG's 3) - GraphRAG failure is catastrophic
            "request_timeout": 600,  # 10 minutes per LLM request (community summarization can be very slow)
            "concurrent_requests": config.rag.async_processing.max_concurrent_requests,  # Match Vector RAG concurrency (8)
        },
        "embeddings": {
            "api_key": config.get("llm.openai.api_key"),
            "type": "openai_embedding",
            "model": config.rag.embedding_model,
            "api_base": config.get("llm.openai.embedding_base_url") or config.get("llm.openai.base_url"),  # Custom API endpoint
            "batch_size": config.rag.embedding_batch_max_tokens,
            # Resilience settings - MORE aggressive than Vector RAG due to high failure cost
            # GraphRAG indexing can take 24 hours; failure means losing ALL progress
            "max_retries": 10,  # 10 retries (vs Vector RAG's 3) - GraphRAG failure is catastrophic
            "request_timeout": 60,  # 60s timeout (vs Vector RAG's 30s) - give more time for transient issues
            # Connection pool race condition fixed in models.py with custom httpx client
            # See: docs/graphrag_embedding_todo.md for root cause analysis
            "concurrent_requests": config.rag.async_processing.max_concurrent_requests,  # Match Vector RAG concurrency (8)
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

        # Create GraphRAG configuration
        graphrag_config = create_graphrag_config(collection_id, root_dir)

        self.update_progress(2, 5, "Initializing GraphRAG indexing")

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

        # Build the index
        self.update_progress(3, 5, "Building GraphRAG index (this may take a while)")

        # Create config object and run indexing
        config_obj = GraphRagConfig.from_dict(graphrag_config)
        index_result = build_index(config=config_obj)

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
    self, query: str, collection_id: str, **kwargs
) -> Dict[str, Any]:
    """
    Perform global GraphRAG query across the entire graph.

    Args:
        query: Query string
        collection_id: Collection to query
        **kwargs: Additional query parameters

    Returns:
        Dict containing query results
    """
    self.validate_input(
        ["query", "collection_id"], query=query, collection_id=collection_id
    )

    try:
        self.update_progress(0, 3, "Preparing global GraphRAG query")

        config = get_config()
        root_dir = Path(config.rag.root_dir) / collection_id

        # Check if index exists
        output_dir = root_dir / "output"
        if not output_dir.exists():
            return {
                "query": query,
                "collection_id": collection_id,
                "error": "GraphRAG index not found. Please build index first.",
                "status": "failed",
            }

        self.update_progress(1, 3, "Executing global search")

        # Import GraphRAG components
        try:
            from fileintel.rag.graph_rag._graphrag_imports import (
                global_search,
                GraphRagConfig,
            )
        except ImportError as e:
            return {
                "query": query,
                "collection_id": collection_id,
                "error": "GraphRAG dependencies not available",
                "status": "failed",
            }

        # Create configuration for querying
        graphrag_config = create_graphrag_config(collection_id, str(root_dir))
        config_obj = GraphRagConfig.from_dict(graphrag_config)

        # Perform global search
        search_result = global_search(query=query, config=config_obj, **kwargs)

        self.update_progress(2, 3, "Processing query results")

        # Extract results
        if hasattr(search_result, "response"):
            answer = search_result.response
            sources = getattr(search_result, "context_data", [])
        else:
            answer = str(search_result)
            sources = []

        result = {
            "query": query,
            "collection_id": collection_id,
            "answer": answer,
            "sources": sources,
            "search_type": "global",
            "confidence": kwargs.get("confidence", 0.8),
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


@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="graphrag_queries",
    rate_limit="60/m",
    max_retries=3,
)
def query_graph_local(self, query: str, collection_id: str, **kwargs) -> Dict[str, Any]:
    """
    Perform local GraphRAG query focused on specific entities.

    Args:
        query: Query string
        collection_id: Collection to query
        **kwargs: Additional query parameters

    Returns:
        Dict containing query results
    """
    self.validate_input(
        ["query", "collection_id"], query=query, collection_id=collection_id
    )

    try:
        self.update_progress(0, 3, "Preparing local GraphRAG query")

        config = get_config()
        root_dir = Path(config.rag.root_dir) / collection_id

        # Check if index exists
        output_dir = root_dir / "output"
        if not output_dir.exists():
            return {
                "query": query,
                "collection_id": collection_id,
                "error": "GraphRAG index not found. Please build index first.",
                "status": "failed",
            }

        self.update_progress(1, 3, "Executing local search")

        # Import GraphRAG components
        try:
            from fileintel.rag.graph_rag._graphrag_imports import (
                local_search,
                GraphRagConfig,
            )
        except ImportError as e:
            return {
                "query": query,
                "collection_id": collection_id,
                "error": "GraphRAG dependencies not available",
                "status": "failed",
            }

        # Create configuration for querying
        graphrag_config = create_graphrag_config(collection_id, str(root_dir))
        config_obj = GraphRagConfig.from_dict(graphrag_config)

        # Perform local search
        search_result = local_search(query=query, config=config_obj, **kwargs)

        self.update_progress(2, 3, "Processing query results")

        # Extract results
        if hasattr(search_result, "response"):
            answer = search_result.response
            sources = getattr(search_result, "context_data", [])
        else:
            answer = str(search_result)
            sources = []

        result = {
            "query": query,
            "collection_id": collection_id,
            "answer": answer,
            "sources": sources,
            "search_type": "local",
            "confidence": kwargs.get("confidence", 0.9),
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


@app.task(base=BaseFileIntelTask, bind=True, queue="graphrag_queries")
def adaptive_graphrag_query(
    self, query: str, collection_id: str, **kwargs
) -> Dict[str, Any]:
    """
    Adaptive GraphRAG query that chooses between global and local search.

    Args:
        query: Query string
        collection_id: Collection to query
        **kwargs: Additional query parameters

    Returns:
        Dict containing query results from optimal search strategy
    """
    self.validate_input(
        ["query", "collection_id"], query=query, collection_id=collection_id
    )

    try:
        self.update_progress(0, 4, "Analyzing query for optimal search strategy")

        # Simple heuristic for choosing search strategy
        # More specific queries (with entities/names) -> local search
        # Broader questions -> global search
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
        self.update_progress(1, 4, f"Using {search_type} search strategy")

        # Execute the chosen search (avoiding .get() to prevent blocking)
        if use_local:
            search_task = query_graph_local.apply_async(
                args=[query, collection_id], kwargs=kwargs
            )
            search_result = {"status": "processing", "task_id": search_task.id}
        else:
            search_task = query_graph_global.apply_async(
                args=[query, collection_id], kwargs=kwargs
            )
            search_result = {"status": "processing", "task_id": search_task.id}

        self.update_progress(3, 4, "GraphRAG adaptive query initiated")

        # Return task information instead of blocking for completion
        result = search_result.copy()
        result["adaptive_strategy"] = search_type
        result["strategy_reasoning"] = f"Local score: {local_score}, Global score: {global_score}"
        result["query"] = query
        result["collection_id"] = collection_id

        self.update_progress(4, 4, "Adaptive GraphRAG query task started")
        return result

    except Exception as e:
        logger.error(f"Error in adaptive GraphRAG query: {e}")
        return {
            "query": query,
            "collection_id": collection_id,
            "error": str(e),
            "status": "failed",
        }


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
        root_dir = Path(config.rag.root_dir) / collection_id
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

            # Set status to "ready" when indexing completes successfully
            storage.update_graphrag_index_status(collection_id, "ready")
            logger.info(f"Set GraphRAG index status to 'ready' for collection {collection_id}")

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

        # Set status to "failed" when indexing fails
        try:
            storage = get_shared_storage()
            storage.update_graphrag_index_status(collection_id, "failed")
            storage.close()
            logger.info(f"Set GraphRAG index status to 'failed' for collection {collection_id}")
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
