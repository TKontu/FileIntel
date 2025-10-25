"""This module defines the GraphRAG service for direct API interaction."""

import os
import pandas as pd
import asyncio
import logging
import shutil
from typing import List, Dict, Any
from fileintel.storage.models import DocumentChunk
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.rag.graph_rag.adapters.data_adapter import GraphRAGDataAdapter
from fileintel.core.config import Settings, get_config
from fileintel.rag.graph_rag.adapters.config_adapter import GraphRAGConfigAdapter
from fileintel.rag.reranker_service import RerankerService
from .parquet_loader import ParquetLoader
from .dataframe_cache import GraphRAGDataFrameCache
from .._graphrag_imports import global_search, local_search, build_index, GraphRagConfig

logger = logging.getLogger(__name__)


class GraphRAGService:
    """A service to interact with the GraphRAG API."""

    def __init__(self, storage: PostgreSQLStorage, settings: Settings):
        self.storage = storage
        self.settings = settings
        self.data_adapter = GraphRAGDataAdapter()
        self.config_adapter = GraphRAGConfigAdapter()
        self.cache = GraphRAGDataFrameCache(self.settings)
        self.parquet_loader = ParquetLoader(self.cache)
        # Cache configs by collection_id to avoid repeated adaptation
        self._config_cache = {}

        # Initialize reranker service if enabled for graph results
        self.reranker = None
        if settings.rag.reranking.enabled and settings.rag.reranking.rerank_graph_results:
            try:
                self.reranker = RerankerService(settings)
                logger.info("RerankerService initialized for GraphRAG")
            except Exception as e:
                logger.warning(f"Failed to initialize RerankerService: {e}. Continuing without reranking.")

    async def _get_cached_config(self, collection_id: str):
        """Get cached GraphRAG config for collection, creating if not exists."""
        if collection_id not in self._config_cache:
            self._config_cache[collection_id] = await asyncio.to_thread(
                self.config_adapter.adapt_config,
                self.settings,
                collection_id,
                self.settings.rag.root_dir,
            )
        return self._config_cache[collection_id]

    async def query(self, query: str, collection_id: str) -> Dict[str, Any]:
        """
        Standard query interface for the orchestrator. Defaults to global search.
        """
        # Validate collection exists (run in thread to avoid blocking)
        collection = await asyncio.to_thread(self.storage.get_collection, collection_id)
        if not collection:
            logger.warning(f"Collection '{collection_id}' not found")
            return {
                "answer": f"Collection '{collection_id}' not found. Please verify the collection ID and try again.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "error": "collection_not_found",
                    "search_type": "global",
                    "collection_id": collection_id,
                },
            }

        raw_response = await self.global_search(query, collection_id)

        # Extract response from dict (global_search returns a dict from data_adapter)
        if isinstance(raw_response, dict):
            answer = raw_response.get("response", str(raw_response))
            sources = raw_response.get("context", {}).get("data", [])
        else:
            answer = getattr(raw_response, "response", str(raw_response))
            sources = getattr(raw_response, "context_data", [])

        # Rerank sources if enabled
        sources = await self._rerank_sources_if_enabled(query, sources)

        # Calculate confidence based on result quality
        confidence = self._calculate_confidence(raw_response, sources)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "metadata": {"search_type": "global"},
        }

    async def build_index(self, documents: List[DocumentChunk], collection_id: str):
        """Builds a GraphRAG index from a set of documents."""
        import logging
        import time

        logger = logging.getLogger(__name__)

        logger.info(f"Starting GraphRAG index build for collection {collection_id}")
        logger.info(f"Processing {len(documents)} document chunks")

        documents_df = self.data_adapter.adapt_documents(documents)
        logger.info(f"Converted to DataFrame with {len(documents_df)} rows")

        graphrag_config = await self._get_cached_config(collection_id)
        logger.info(
            f"GraphRAG config created, output dir: {graphrag_config.output.base_dir}"
        )

        # Log LLM configuration
        chat_model = graphrag_config.models.get("default_chat_model")
        embedding_model = graphrag_config.models.get("default_embedding_model")
        if chat_model:
            logger.info(f"GraphRAG Chat model: {chat_model.model}")
            logger.info(
                f"GraphRAG Chat model base_url: {getattr(chat_model, 'api_base', 'N/A')}"
            )
        if embedding_model:
            logger.info(f"GraphRAG Embedding model: {embedding_model.model}")
            logger.info(
                f"GraphRAG Embedding model base_url: {getattr(embedding_model, 'api_base', 'N/A')}"
            )

        start_time = time.time()
        logger.info(
            "Starting GraphRAG index build (this may take several minutes for entity/relationship extraction)"
        )

        result = await build_index(config=graphrag_config, input_documents=documents_df)

        elapsed = time.time() - start_time
        logger.info(f"GraphRAG index build completed in {elapsed:.1f} seconds")

        workspace_path = graphrag_config.output.base_dir

        # Count the generated entities and communities
        (
            documents_count,
            entities_count,
            communities_count,
        ) = self._count_graphrag_results(workspace_path)

        await asyncio.to_thread(
            self.storage.save_graphrag_index_info,
            collection_id=collection_id,
            index_path=workspace_path,
            documents_count=documents_count,
            entities_count=entities_count,
            communities_count=communities_count,
        )

        # Load and save entities and communities to database
        await self._save_graphrag_data_to_database(collection_id, workspace_path)

        return workspace_path

    def _count_graphrag_results(self, workspace_path: str) -> tuple[int, int, int]:
        """Count documents, entities, and communities from GraphRAG output files."""
        try:
            import os

            documents_count = 0
            entities_count = 0
            communities_count = 0

            # Count documents
            documents_file = os.path.join(workspace_path, "documents.parquet")
            if os.path.exists(documents_file):
                documents_df = pd.read_parquet(documents_file)
                documents_count = len(documents_df)

            # Count entities
            entities_file = os.path.join(workspace_path, "entities.parquet")
            if os.path.exists(entities_file):
                entities_df = pd.read_parquet(entities_file)
                entities_count = len(entities_df)

            # Count communities
            communities_file = os.path.join(workspace_path, "communities.parquet")
            if os.path.exists(communities_file):
                communities_df = pd.read_parquet(communities_file)
                communities_count = len(communities_df)

            return documents_count, entities_count, communities_count

        except Exception as e:
            # If counting fails, return zeros but don't fail the indexing
            print(f"Warning: Could not count GraphRAG results: {e}")
            return 0, 0, 0

    async def global_search(self, query: str, collection_id: str):
        """Performs a global search on the GraphRAG index."""
        index_info = await asyncio.to_thread(
            self.storage.get_graphrag_index_info, collection_id
        )
        if not index_info or not index_info.get("index_path"):
            raise ValueError(f"GraphRAG index not found for collection {collection_id}")

        workspace_path = index_info["index_path"]

        dataframes = await self.parquet_loader.load_parquet_files(
            workspace_path, collection_id
        )

        graphrag_config = await self._get_cached_config(collection_id)

        result, context = await global_search(
            config=graphrag_config,
            entities=dataframes["entities"],
            communities=dataframes["communities"],
            community_reports=dataframes["community_reports"],
            community_level=self.settings.rag.community_levels,
            dynamic_community_selection=True,
            response_type="text",
            query=query,
        )
        return self.data_adapter.convert_response(result, context)

    async def local_search(
        self, query: str, collection_id: str, community: str, MOCK_ARGM=None
    ):
        """Performs a local search within a specific community."""
        index_info = await asyncio.to_thread(
            self.storage.get_graphrag_index_info, collection_id
        )
        if not index_info or not index_info.get("index_path"):
            raise ValueError(f"GraphRAG index not found for collection {collection_id}")

        workspace_path = index_info["index_path"]

        dataframes = await self.parquet_loader.load_parquet_files(
            workspace_path, collection_id
        )

        # Handle optional covariates (workspace_path is already the output directory)
        covariates_path = os.path.join(workspace_path, "covariates.parquet")
        covariates = None
        if os.path.exists(covariates_path):
            covariates = pd.read_parquet(covariates_path)

        graphrag_config = await self._get_cached_config(collection_id)

        result, context = await local_search(
            config=graphrag_config,
            entities=dataframes["entities"],
            communities=dataframes["communities"],
            community_reports=dataframes["community_reports"],
            text_units=dataframes.get("text_units"),
            relationships=dataframes.get("relationships"),
            covariates=covariates,
            community_level=self.settings.rag.community_levels,
            response_type="text",
            query=query,
        )
        return self.data_adapter.convert_response(result, context)

    # Orchestrator interface methods
    async def global_query(self, collection_id: str, query: str):
        """Wrapper for global_search to match orchestrator interface."""
        raw_response = await self.global_search(query, collection_id)

        # Convert response format to match API expectations
        if isinstance(raw_response, dict):
            answer = raw_response.get("response", str(raw_response))
            context = raw_response.get("context", {})

            # Extract text_unit IDs for source tracing (DataFrames can't be JSON serialized)
            text_unit_ids = self._extract_text_unit_ids_from_context(context)

            return {
                "answer": answer,
                "sources": [],
                "context": {},  # Empty - can't serialize DataFrames
                "metadata": {
                    "search_type": "global",
                    "text_unit_ids": list(text_unit_ids)  # Add for source tracing
                }
            }
        return raw_response

    async def local_query(self, collection_id: str, query: str):
        """Wrapper for local_search to match orchestrator interface."""
        raw_response = await self.local_search(query, collection_id, community="")

        # Convert response format to match API expectations
        if isinstance(raw_response, dict):
            answer = raw_response.get("response", str(raw_response))
            context = raw_response.get("context", {})
            return {
                "answer": answer,
                "sources": [],
                "context": context,
                "metadata": {"search_type": "local"}
            }
        return raw_response

    def _extract_text_unit_ids_from_context(self, context: dict) -> set:
        """Extract text_unit IDs from GraphRAG context for source tracing."""
        import pandas as pd

        text_unit_ids = set()

        # From community reports
        if "reports" in context:
            reports_df = context["reports"]
            if isinstance(reports_df, pd.DataFrame) and not reports_df.empty:
                # Get workspace path for this collection
                # Note: This is a simplified extraction - full traversal happens in source_tracer
                # For now, just return empty set - CLI will do full traversal
                pass

        return text_unit_ids

    def is_collection_indexed(self, collection_id: str) -> bool:
        """Check if a collection has a GraphRAG index."""
        try:
            index_info = self.storage.get_graphrag_index_info(collection_id)
            return index_info is not None and index_info.get("index_path") is not None
        except Exception:
            return False

    async def remove_index(self, collection_id: str) -> Dict[str, Any]:
        """Remove GraphRAG index for a collection."""

        try:
            # Get index info to find workspace path
            index_info = await asyncio.to_thread(
                self.storage.get_graphrag_index_info, collection_id
            )
            if not index_info or not index_info.get("index_path"):
                logger.warning(
                    f"No GraphRAG index found for collection {collection_id}"
                )
                return {"status": "no_index", "message": "No index found to remove"}

            workspace_path = index_info["index_path"]

            # Remove the workspace directory
            if os.path.exists(workspace_path):
                shutil.rmtree(workspace_path)
                logger.info(f"Removed GraphRAG workspace at {workspace_path}")

            # Remove database index information
            await asyncio.to_thread(
                self.storage.remove_graphrag_index_info, collection_id
            )
            logger.info(f"Removed GraphRAG index info from database for collection {collection_id}")

            return {
                "status": "success",
                "message": f"GraphRAG index removed for collection {collection_id}",
                "workspace_removed": workspace_path,
            }

        except Exception as e:
            logger.error(
                f"Failed to remove GraphRAG index for collection {collection_id}: {e}"
            )
            raise

    async def get_index_status(self, collection_id: str) -> Dict[str, Any]:
        """Get GraphRAG index status for a collection."""
        try:
            index_info = await asyncio.to_thread(
                self.storage.get_graphrag_index_info, collection_id
            )
            if not index_info:
                return {"status": "not_indexed"}

            workspace_path = index_info.get("index_path")
            if not workspace_path or not os.path.exists(workspace_path):
                return {"status": "index_missing", "path": workspace_path}

            # Count current results
            (
                documents_count,
                entities_count,
                communities_count,
            ) = self._count_graphrag_results(workspace_path)

            return {
                "status": "indexed",
                "index_path": workspace_path,
                "documents_count": documents_count,
                "entities_count": entities_count,
                "communities_count": communities_count,
                "created_at": index_info.get("created_at"),
                "updated_at": index_info.get("updated_at"),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _save_graphrag_data_to_database(self, collection_id: str, workspace_path: str):
        """Load GraphRAG parquet files and save entities/communities to database."""
        try:
            import os
            import numpy as np
            import json

            # Load entities from parquet
            entities_file = os.path.join(workspace_path, "entities.parquet")
            if os.path.exists(entities_file):
                entities_df = pd.read_parquet(entities_file)
                entities_data = entities_df.to_dict('records')

                # Convert numpy arrays to lists and handle NaN values for JSON serialization
                def convert_numpy_arrays(obj):
                    """Recursively convert numpy arrays to lists and NaN to None."""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_arrays(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_arrays(item) for item in obj]
                    elif pd.isna(obj):  # Handle NaN values
                        return None
                    else:
                        return obj

                # Clean the entities data for JSON serialization
                entities_data_clean = [convert_numpy_arrays(entity) for entity in entities_data]

                await asyncio.to_thread(
                    self.storage.save_graphrag_entities,
                    collection_id,
                    entities_data_clean
                )
                logger.info(f"Saved {len(entities_data_clean)} entities to database for collection {collection_id}")

            # Load communities from parquet
            communities_file = os.path.join(workspace_path, "communities.parquet")
            if os.path.exists(communities_file):
                communities_df = pd.read_parquet(communities_file)
                communities_data = communities_df.to_dict('records')

                # Clean the communities data for JSON serialization
                communities_data_clean = [convert_numpy_arrays(community) for community in communities_data]

                await asyncio.to_thread(
                    self.storage.save_graphrag_communities,
                    collection_id,
                    communities_data_clean
                )
                logger.info(f"Saved {len(communities_data_clean)} communities to database for collection {collection_id}")

        except Exception as e:
            logger.error(f"Error saving GraphRAG data to database: {e}")

    async def _rerank_sources_if_enabled(self, query: str, sources: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank GraphRAG sources if reranker is enabled.

        Args:
            query: The search query
            sources: List of source dicts from GraphRAG
            top_k: Number of top results to return (defaults to config.final_top_k)

        Returns:
            Reranked sources or original sources if reranking disabled/failed
        """
        if not self.reranker or not sources:
            return sources

        top_k = top_k or self.settings.rag.reranking.final_top_k

        try:
            # Convert GraphRAG sources to reranker format
            # GraphRAG sources may have various text fields - try common ones
            passages = []
            for source in sources:
                # Try to extract text from various possible fields
                text = (
                    source.get("content") or
                    source.get("text") or
                    source.get("description") or
                    source.get("title", "")
                )

                if text:
                    passages.append({
                        "content": text,
                        "relevance_score": source.get("score", source.get("weight", 0.0)),
                        **source  # Include all other fields
                    })

            if not passages:
                logger.debug("No text content found in GraphRAG sources for reranking")
                return sources

            # Rerank passages
            reranked_passages = await asyncio.to_thread(
                self.reranker.rerank,
                query=query,
                passages=passages,
                top_k=top_k,
                passage_text_key="content"
            )

            # Convert back to source format
            reranked_sources = []
            for passage in reranked_passages:
                source = passage.copy()
                # Restore original text field name if it wasn't "content"
                if "content" in source:
                    text = source.pop("content")
                    # Update the appropriate field
                    if "text" in sources[0]:
                        source["text"] = text
                    elif "description" in sources[0]:
                        source["description"] = text
                source["reranked_score"] = passage["reranked_score"]
                source["original_score"] = passage.get("original_score", 0.0)
                reranked_sources.append(source)

            logger.info(f"Reranked {len(passages)} → {len(reranked_sources)} GraphRAG sources")
            return reranked_sources

        except Exception as e:
            logger.error(f"GraphRAG source reranking failed: {e}. Using original sources.")
            return sources[:top_k] if top_k else sources

    def _calculate_confidence(self, raw_response, sources) -> float:
        """
        Calculate confidence score based on response quality.
        Simplified approach using configurable default confidence.
        """
        # Use configurable default confidence for GraphRAG responses
        default_confidence = getattr(
            self.settings.rag, "default_confidence", 0.8
        )

        # Return default confidence if we have sources, otherwise reduce confidence
        return default_confidence if sources else default_confidence * 0.5
