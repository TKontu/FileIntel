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

    def _sync_graphrag_logger_level(self):
        """
        Synchronize the GraphRAG and fnllm logger levels with application logging configuration.

        Respects component_levels from config to allow fine-grained control over different
        GraphRAG subsystems (e.g., show workflow progress but hide verbose operation logs).
        """
        # Apply component-specific log levels from config
        if hasattr(self.settings.logging, 'component_levels'):
            for component, level in self.settings.logging.component_levels.items():
                if component.startswith('graphrag') or component.startswith('fnllm'):
                    logger_obj = logging.getLogger(component)
                    logger_obj.setLevel(level.upper())

        # Fallback: if no component levels defined, use root level
        else:
            app_log_level = getattr(logging, self.settings.logging.level.upper(), logging.WARNING)
            logging.getLogger("graphrag").setLevel(app_log_level)
            logging.getLogger("fnllm").setLevel(app_log_level)

    async def _get_cached_config(self, collection_id: str):
        """Get cached GraphRAG config for collection, creating if not exists."""
        if collection_id not in self._config_cache:
            self._config_cache[collection_id] = await asyncio.to_thread(
                self.config_adapter.adapt_config,
                self.settings,
                collection_id,
                self.settings.graphrag.index_base_path,
            )
        return self._config_cache[collection_id]

    async def query(self, query: str, collection_id: str, search_type: str = "global") -> Dict[str, Any]:
        """
        Standard query interface with citation tracing. Routes to global or local search.

        Args:
            query: The search query
            collection_id: Collection to search
            search_type: "global" for global search, "local" for local search (default: "global")

        Returns:
            Dict with answer (formatted with citations), sources, confidence, and metadata
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
                    "search_type": search_type,
                    "collection_id": collection_id,
                },
            }

        # Route to appropriate search method
        if search_type == "local":
            raw_response = await self.local_search(query, collection_id, community="")
        else:
            raw_response = await self.global_search(query, collection_id)

        # Extract response from dict (global_search returns a dict from data_adapter)
        if isinstance(raw_response, dict):
            answer = raw_response.get("response", str(raw_response))

            # Debug: Log what's in the context
            context = raw_response.get("context", {})
            logger.info(f"GraphRAG context type: {type(context)}, keys: {context.keys() if isinstance(context, dict) else 'N/A'}")

            # Extract sources from context (check each key individually to avoid DataFrame boolean ambiguity)
            sources = []
            if isinstance(context, dict):
                # Try common field names
                for key in ["reports", "data", "sources"]:
                    if key in context:
                        potential_sources = context[key]
                        # Convert DataFrame to list of dicts if needed
                        import pandas as pd
                        if isinstance(potential_sources, pd.DataFrame):
                            sources = potential_sources.to_dict('records')
                            logger.info(f"Extracted {len(sources)} sources from context['{key}'] (DataFrame)")
                            break
                        elif potential_sources and len(potential_sources) > 0:
                            sources = potential_sources
                            logger.info(f"Extracted {len(sources)} sources from context['{key}']")
                            break
            else:
                # Context might be a SearchResult object
                sources = getattr(context, "reports", []) or getattr(context, "data", []) or []
        else:
            answer = getattr(raw_response, "response", str(raw_response))
            sources = getattr(raw_response, "context_data", [])
            logger.info(f"Extracted {len(sources)} sources from context_data")

        logger.info(f"Total sources before reranking: {len(sources)}")

        # Rerank sources if enabled
        sources = await self._rerank_sources_if_enabled(query, sources)

        # SERVER-SIDE: Trace citations and format answer with Harvard references
        formatted_answer = answer
        traced_sources = sources
        try:
            # Get workspace path for this collection
            index_info = await asyncio.to_thread(
                self.storage.get_graphrag_index_info, collection_id
            )

            if index_info and index_info.get("index_path"):
                workspace_path = index_info["index_path"]
                formatted_answer, traced_sources = await self._trace_and_format_citations(
                    answer, collection_id, workspace_path, sources
                )
            else:
                logger.warning(f"Citation tracing skipped: No index_path found for collection {collection_id}")
        except Exception as e:
            import traceback
            logger.error(f"Citation tracing failed: {e}")
            logger.debug(f"Citation tracing traceback: {traceback.format_exc()}")
            # Fallback to raw answer if tracing fails

        # Calculate confidence based on result quality
        confidence = self._calculate_confidence(raw_response, traced_sources)

        return {
            "answer": formatted_answer,      # Formatted with Harvard citations
            "raw_answer": answer,            # Original with [Data: Reports (X)]
            "sources": traced_sources,
            "confidence": confidence,
            "metadata": {"search_type": search_type},
        }

    async def build_index(self, documents: List[DocumentChunk], collection_id: str):
        """
        Builds a GraphRAG index from a set of documents.

        This method uses default settings without checkpoint resume.
        For resume capability, use build_index_with_resume instead.
        """
        return await self.build_index_with_resume(
            documents, collection_id, enable_resume=False
        )

    async def build_index_with_resume(
        self,
        documents: List[DocumentChunk],
        collection_id: str,
        enable_resume: bool = True,
        validate_checkpoints: bool = True,
    ):
        """
        Builds a GraphRAG index from a set of documents with checkpoint resume capability.

        Args:
            documents: List of document chunks to index
            collection_id: Collection identifier
            enable_resume: Whether to enable checkpoint detection and resume (default: True)
            validate_checkpoints: Whether to validate checkpoint consistency (default: True)

        Returns:
            Workspace path where the index was built
        """
        import logging
        import time
        from .progress_callback import GraphRAGProgressCallback

        logger = logging.getLogger(__name__)

        logger.info(f"Starting GraphRAG index build for collection {collection_id}")
        logger.info(f"Processing {len(documents)} document chunks")

        if enable_resume:
            logger.info("📌 Checkpoint resume enabled")
        else:
            logger.info("🔄 Full rebuild mode (checkpoints disabled)")

        documents_df = self.data_adapter.adapt_documents(documents)
        logger.info(f"Converted to DataFrame with {len(documents_df)} rows")

        graphrag_config = await self._get_cached_config(collection_id)
        logger.debug(
            f"GraphRAG config created, output dir: {graphrag_config.output.base_dir}"
        )

        # Log LLM configuration
        chat_model = graphrag_config.models.get("default_chat_model")
        embedding_model = graphrag_config.models.get("default_embedding_model")
        if chat_model:
            logger.debug(f"GraphRAG Chat model: {chat_model.model}")
            logger.debug(
                f"GraphRAG Chat model base_url: {getattr(chat_model, 'api_base', 'N/A')}"
            )
        if embedding_model:
            logger.debug(f"GraphRAG Embedding model: {embedding_model.model}")
            logger.debug(
                f"GraphRAG Embedding model base_url: {getattr(embedding_model, 'api_base', 'N/A')}"
            )

        start_time = time.time()

        # Create progress callback for clean workflow tracking
        progress_callback = GraphRAGProgressCallback(collection_id)

        # Call build_index with resume parameters
        result = await build_index(
            config=graphrag_config,
            input_documents=documents_df,
            callbacks=[progress_callback],
            enable_resume=enable_resume,
            validate_checkpoints=validate_checkpoints,
        )

        # Sync GraphRAG logger level with application config (build_index calls init_loggers)
        self._sync_graphrag_logger_level()

        elapsed = time.time() - start_time
        logger.info(f"GraphRAG index build completed in {elapsed:.1f} seconds")

        workspace_path = graphrag_config.output.base_dir

        # Count the generated entities and communities (wrap blocking pandas operations)
        (
            documents_count,
            entities_count,
            communities_count,
        ) = await asyncio.to_thread(self._count_graphrag_results, workspace_path)

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

        # Validate completeness if enabled
        if self.settings.graphrag.validate_completeness:
            self._validate_index_completeness(workspace_path)

        # Critical validation: Ensure embeddings are complete before marking as ready
        # Run in thread pool to avoid blocking event loop (LanceDB operations are blocking)
        await asyncio.to_thread(
            self._validate_embedding_completeness,
            workspace_path
        )

        # Set status to "ready" ONLY after ALL operations succeed:
        # 1. Index build complete
        # 2. PostgreSQL data saved
        # 3. Completeness validation passed
        # 4. Embedding validation passed
        # This prevents race condition where API shows "ready" before data is accessible
        #
        # Use atomic update with row locking to prevent concurrent status inconsistencies
        await asyncio.to_thread(
            self.storage.update_graphrag_index_status_atomic,
            collection_id,
            "ready",
            None  # error_message
        )
        logger.info(f"Atomically set GraphRAG index status to 'ready' for collection {collection_id} after all validations passed")

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

    def _validate_index_completeness(self, workspace_path: str) -> None:
        """Validate completeness of GraphRAG index and log warnings for gaps.

        Args:
            workspace_path: Path to GraphRAG workspace directory (output dir)
        """
        try:
            from pathlib import Path
            from ..validators import CompletenessValidator

            logger.info("Validating GraphRAG index completeness...")

            validator = CompletenessValidator(Path(workspace_path))
            reports = validator.validate_all()

            # Log results for each phase
            for phase_name, report in reports.items():
                if report.completeness >= self.settings.graphrag.completeness_threshold:
                    logger.info(
                        f"✅ Phase '{phase_name}' {report.completeness:.2%} complete "
                        f"({report.complete_items:,}/{report.total_items:,})"
                    )
                else:
                    logger.warning(
                        f"⚠️  Phase '{phase_name}' only {report.completeness:.2%} complete. "
                        f"Missing {report.missing_items:,} items. "
                        f"Sample IDs: {report.missing_ids[:10]}"
                    )

                    # Log hierarchy-level details if available
                    if report.details_by_level:
                        logger.info(f"   Hierarchy level breakdown for '{phase_name}':")
                        for level in sorted(report.details_by_level.keys()):
                            level_data = report.details_by_level[level]
                            level_completeness = level_data['completeness']
                            if level_completeness >= self.settings.graphrag.completeness_threshold:
                                status = "✅"
                            else:
                                status = "⚠️ "
                            logger.info(
                                f"   {status} Level {level}: {level_completeness:.2%} "
                                f"({level_data['complete']:,}/{level_data['total']:,})"
                            )

            # Calculate and log overall completeness
            total_items = sum(r.total_items for r in reports.values())
            complete_items = sum(r.complete_items for r in reports.values())
            overall_completeness = complete_items / total_items if total_items > 0 else 0.0

            if overall_completeness >= self.settings.graphrag.completeness_threshold:
                logger.info(
                    f"✅ Overall index completeness: {overall_completeness:.2%} "
                    f"({complete_items:,}/{total_items:,})"
                )
            else:
                logger.warning(
                    f"⚠️  Overall index completeness: {overall_completeness:.2%} "
                    f"({complete_items:,}/{total_items:,})"
                )

        except Exception as e:
            # Don't fail indexing if validation fails
            logger.error(f"Error validating index completeness: {e}", exc_info=True)

    def _validate_embedding_completeness(self, workspace_path: str) -> None:
        """Validate that embeddings were created for all expected items.

        This is a CRITICAL validation that raises an exception if embeddings are incomplete.
        Unlike _validate_index_completeness() which only logs warnings, this method will
        prevent the index from being marked as "ready" if embeddings are missing.

        Args:
            workspace_path: Path to GraphRAG workspace directory (output dir)

        Raises:
            ValueError: If embeddings are incomplete
        """
        import os
        import pandas as pd

        logger.info("🔍 Validating embedding completeness (CRITICAL)...")

        # CRITICAL FIX: workspace_path already points to the output directory
        # (e.g., "/data/collection_id/output"), so don't add another "output" level
        lancedb_dir = os.path.join(workspace_path, "lancedb")

        # Check if LanceDB directory exists
        if not os.path.exists(lancedb_dir):
            raise ValueError(
                f"LanceDB directory not found at {lancedb_dir}. "
                "Embeddings were not created. This indicates the generate_text_embeddings "
                "workflow did not complete successfully."
            )

        try:
            import lancedb

            # Connect to LanceDB
            db = lancedb.connect(lancedb_dir)

            # Validate text unit embeddings (most critical)
            text_units_file = os.path.join(workspace_path, "text_units.parquet")
            if os.path.exists(text_units_file):
                text_units_df = pd.read_parquet(text_units_file)
                expected_count = len(text_units_df)

                try:
                    table = db.open_table("default-text_unit-text")
                    actual_count = table.count_rows()

                    completeness = actual_count / expected_count if expected_count > 0 else 0.0

                    if completeness < 0.95:  # Less than 95% complete
                        raise ValueError(
                            f"Text unit embeddings INCOMPLETE: {actual_count:,}/{expected_count:,} "
                            f"({completeness:.1%} complete). Missing {expected_count - actual_count:,} embeddings. "
                            "This indicates the embedding workflow hung or was killed before completion. "
                            "Status cannot be set to 'ready' with incomplete embeddings."
                        )

                    logger.info(
                        f"✅ Text unit embeddings complete: {actual_count:,}/{expected_count:,} "
                        f"({completeness:.1%})"
                    )

                except Exception as e:
                    if "not found" in str(e).lower():
                        raise ValueError(
                            f"Text unit embedding table not found in LanceDB. "
                            f"Expected table 'default-text_unit-text' does not exist. "
                            "This indicates embeddings were never created."
                        )
                    raise

            # Validate entity embeddings
            entities_file = os.path.join(workspace_path, "entities.parquet")
            if os.path.exists(entities_file):
                entities_df = pd.read_parquet(entities_file)
                expected_count = len(entities_df)

                try:
                    table = db.open_table("default-entity-description")
                    actual_count = table.count_rows()

                    completeness = actual_count / expected_count if expected_count > 0 else 0.0

                    if completeness < 0.95:
                        raise ValueError(
                            f"Entity embeddings INCOMPLETE: {actual_count:,}/{expected_count:,} "
                            f"({completeness:.1%} complete)"
                        )

                    logger.info(
                        f"✅ Entity embeddings complete: {actual_count:,}/{expected_count:,} "
                        f"({completeness:.1%})"
                    )

                except Exception as e:
                    if "not found" in str(e).lower():
                        logger.warning(
                            "Entity embedding table not found - may not be configured"
                        )
                    else:
                        raise

            # Validate community embeddings
            communities_file = os.path.join(workspace_path, "community_reports.parquet")
            if os.path.exists(communities_file):
                communities_df = pd.read_parquet(communities_file)
                expected_count = len(communities_df)

                try:
                    table = db.open_table("default-community_report-summary")
                    actual_count = table.count_rows()

                    completeness = actual_count / expected_count if expected_count > 0 else 0.0

                    if completeness < 0.95:
                        raise ValueError(
                            f"Community embeddings INCOMPLETE: {actual_count:,}/{expected_count:,} "
                            f"({completeness:.1%} complete)"
                        )

                    logger.info(
                        f"✅ Community embeddings complete: {actual_count:,}/{expected_count:,} "
                        f"({completeness:.1%})"
                    )

                except Exception as e:
                    if "not found" in str(e).lower():
                        logger.warning(
                            "Community embedding table not found - may not be configured"
                        )
                    else:
                        raise

            logger.info("✅ All embedding validations passed - index is complete")

        except ImportError:
            logger.warning(
                "lancedb not installed - skipping embedding validation. "
                "This is not recommended for production."
            )
        except Exception as e:
            # Re-raise ValueError as-is (validation failures)
            if isinstance(e, ValueError):
                raise
            # Wrap other exceptions
            raise ValueError(f"Error validating embeddings: {e}") from e

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
            community_level=self.settings.graphrag.community_levels,
            dynamic_community_selection=True,
            response_type="text",
            query=query,
        )

        # Sync GraphRAG logger level with application config (global_search calls init_loggers)
        self._sync_graphrag_logger_level()

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
            community_level=self.settings.graphrag.community_levels,
            response_type="text",
            query=query,
        )

        # Sync GraphRAG logger level with application config (local_search calls init_loggers)
        self._sync_graphrag_logger_level()

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

            # Count current results (wrap blocking pandas operations)
            (
                documents_count,
                entities_count,
                communities_count,
            ) = await asyncio.to_thread(self._count_graphrag_results, workspace_path)

            # Use actual index_status from database (building/ready/failed/updating)
            # instead of hardcoded "indexed" to support checkpoint resume
            return {
                "status": index_info.get("index_status", "unknown"),
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

            # Load communities - need to merge two files for complete data
            communities_structure_file = os.path.join(workspace_path, "communities.parquet")
            communities_content_file = os.path.join(workspace_path, "community_reports.parquet")

            # CRITICAL: Both files required for complete community data
            # Missing community_reports.parquet means no summaries → broken display
            if os.path.exists(communities_structure_file) and not os.path.exists(communities_content_file):
                logger.error(
                    f"CRITICAL: Missing {communities_content_file} - "
                    f"communities will have no summaries. This indicates incomplete GraphRAG index."
                )
                raise ValueError(
                    f"GraphRAG index incomplete: community_reports.parquet not found at {communities_content_file}. "
                    "Index build may have failed or been interrupted. Please rebuild the index."
                )

            if os.path.exists(communities_structure_file) and os.path.exists(communities_content_file):
                # Load both files
                communities_df = pd.read_parquet(communities_structure_file)  # Has entity_ids, relationship_ids
                reports_df = pd.read_parquet(communities_content_file)  # Has summary, full_content, findings

                # Merge on community and level to get complete data
                merged_df = pd.merge(
                    communities_df,
                    reports_df[['community', 'level', 'summary', 'full_content', 'findings', 'rank', 'rating_explanation']],
                    on=['community', 'level'],
                    how='left'  # Keep all communities even if no report
                )

                communities_data = merged_df.to_dict('records')

                # Clean the communities data for JSON serialization
                communities_data_clean = [convert_numpy_arrays(community) for community in communities_data]

                await asyncio.to_thread(
                    self.storage.save_graphrag_communities,
                    collection_id,
                    communities_data_clean
                )
                logger.info(f"Saved {len(communities_data_clean)} communities (merged from structure + reports) to database for collection {collection_id}")

        except Exception as e:
            logger.error(f"Error saving GraphRAG data to database: {e}")
            # Re-raise to prevent marking index as "ready" with broken data
            raise

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
            source_field_map = []  # Track which field was used for each source

            for source in sources:
                # Try to extract text from various possible fields (in priority order)
                text = None
                text_field = None

                if source.get("content"):
                    text = source["content"]
                    text_field = "content"
                elif source.get("text"):
                    text = source["text"]
                    text_field = "text"
                elif source.get("description"):
                    text = source["description"]
                    text_field = "description"
                elif source.get("title"):
                    text = source["title"]
                    text_field = "title"

                if text:
                    passages.append({
                        "content": text,
                        "relevance_score": source.get("score", source.get("weight", 0.0)),
                        **source  # Include all other fields
                    })
                    source_field_map.append(text_field)

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
            for idx, passage in enumerate(reranked_passages):
                source = passage.copy()

                # Restore original text field using tracked field name
                if "content" in source and idx < len(source_field_map):
                    text = source.pop("content", None)
                    original_field = source_field_map[idx]
                    if text and original_field:
                        source[original_field] = text

                source["reranked_score"] = passage.get("reranked_score", 0.0)
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

    async def _trace_and_format_citations(
        self, answer: str, collection_id: str, workspace_path: str, reranked_sources: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        SERVER-SIDE: Trace GraphRAG citations to source documents and format with Harvard citations.

        Args:
            answer: Raw GraphRAG answer with inline citations like [Data: Reports (6826)]
            collection_id: Collection ID
            workspace_path: Path to GraphRAG output directory (e.g., /data/.../output)
            reranked_sources: Sources that have been reranked (if enabled)

        Returns:
            Tuple of (formatted_answer, traced_sources)
        """
        import re
        import os
        import pandas as pd

        # 1. Parse citation IDs from answer
        citation_ids = self._parse_citation_ids(answer)

        if not citation_ids["report_ids"] and not citation_ids["entity_ids"]:
            # No citations to trace
            return answer, reranked_sources

        logger.info(f"Tracing {len(citation_ids['report_ids'])} report citations, {len(citation_ids['entity_ids'])} entity citations")

        # 2. SERVER-SIDE: Trace citations to sources using direct parquet access
        traced_sources = await asyncio.to_thread(
            self._trace_citations_server_side,
            citation_ids,
            workspace_path,
            reranked_sources
        )

        if not traced_sources:
            logger.warning("No sources traced from citations")
            return answer, reranked_sources

        logger.info(f"Traced {len(traced_sources)} source documents from citations")

        # 3. Format answer with Harvard citations
        formatted_answer = await self._format_with_harvard_citations(
            answer, traced_sources, collection_id, reranked_sources
        )

        return formatted_answer, traced_sources

    def _trace_citations_server_side(
        self,
        citation_ids: Dict[str, set],
        workspace_path: str,
        reranked_sources: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        SERVER-SIDE: Trace citations to source documents using direct parquet file access.

        Args:
            citation_ids: Parsed citation IDs from answer
            workspace_path: Path to GraphRAG output directory
            reranked_sources: Sources from search (used to prioritize relevant documents)

        Returns:
            List of source documents with page numbers, sorted by relevance
        """
        import pandas as pd
        import os

        # Helper to safely load parquet files
        def load_parquet_safe(filename):
            path = os.path.join(workspace_path, filename)
            if os.path.exists(path):
                try:
                    return pd.read_parquet(path)
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            return None

        # Load required parquet files
        communities_df = load_parquet_safe("communities.parquet")
        entities_df = load_parquet_safe("entities.parquet")
        text_units_df = load_parquet_safe("text_units.parquet")
        documents_df = load_parquet_safe("documents.parquet")
        logger.info(f"Loaded parquet files: communities={communities_df is not None}, entities={entities_df is not None}, text_units={text_units_df is not None}, documents={documents_df is not None}")

        if communities_df is None or entities_df is None:
            logger.error(f"Required parquet files not found in {workspace_path}")
            logger.error(f"  communities.parquet exists: {communities_df is not None}")
            logger.error(f"  entities.parquet exists: {entities_df is not None}")
            return []

        if text_units_df is None:
            logger.error(f"text_units.parquet not found in {workspace_path}")
            return []

        if documents_df is None:
            logger.warning(f"documents.parquet not found in {workspace_path}, will skip document mapping")
            # Can still return partial results without document details

        # Trace: Report IDs → Community IDs → Entity IDs → Text Unit IDs → Chunks → Documents
        # OPTIMIZED: Use batch lookups instead of row-by-row iteration

        # Step 1: Collect all entity IDs from communities
        all_entity_ids = set()
        report_ids = citation_ids.get("report_ids", [])

        if report_ids:
            # Batch lookup: Find all communities at once
            comm_mask = communities_df["community"].isin(report_ids)
            for entity_list in communities_df[comm_mask]["entity_ids"]:
                if entity_list is not None and len(entity_list) > 0:
                    all_entity_ids.update(entity_list)

        # Add directly cited entities
        direct_entity_ids = citation_ids.get("entity_ids", [])
        if direct_entity_ids:
            all_entity_ids.update(direct_entity_ids)

        if not all_entity_ids:
            logger.warning(f"No entities found for {len(report_ids)} reports")
            return []

        logger.info(f"Found {len(all_entity_ids)} entities from citations, extracting text units...")

        # Step 2: Batch lookup: Get text units from all entities at once
        text_unit_ids = set()
        entity_mask = entities_df["id"].isin(all_entity_ids)
        for tu_list in entities_df[entity_mask]["text_unit_ids"]:
            if tu_list is not None and len(tu_list) > 0:
                text_unit_ids.update(tu_list)

        if not text_unit_ids:
            logger.warning(f"No text units found for {len(all_entity_ids)} entities")
            return []

        logger.info(f"Found {len(text_unit_ids)} text units, mapping to chunks...")

        # Step 3: Batch lookup: Map text units to chunk UUIDs
        # Note: text_units.document_ids contains FileIntel chunk UUIDs (GraphRAG's "document" = FileIntel's "chunk")
        chunk_uuids = set()
        if text_units_df is not None:
            # Batch lookup: Filter all text units at once
            tu_mask = text_units_df["id"].isin(text_unit_ids)
            for doc_id_list in text_units_df[tu_mask]["document_ids"]:
                if doc_id_list is not None and len(doc_id_list) > 0:
                    chunk_uuids.update(doc_id_list)

        if not chunk_uuids:
            logger.warning(f"No chunk UUIDs found from {len(text_unit_ids)} text units")
            return []

        logger.info(f"Mapped to {len(chunk_uuids)} chunks, grouping by document...")

        if documents_df is None:
            logger.warning("documents.parquet not available, skipping document grouping")
            return []

        # Step 4: Batch lookup - Group chunks by document
        # Convert chunk UUIDs to strings for matching
        chunk_uuid_strs = [str(uuid) for uuid in chunk_uuids]

        # Batch filter: Get all matching documents at once
        doc_mask = documents_df["id"].isin(chunk_uuid_strs)
        matched_docs = documents_df[doc_mask]

        # Group by document title
        doc_chunks = {}
        for _, row in matched_docs.iterrows():
            doc_title = row.get("title", "Unknown")
            chunk_id = row.get("id")
            if doc_title not in doc_chunks:
                doc_chunks[doc_title] = {"chunk_uuids": [], "pages": set()}
            doc_chunks[doc_title]["chunk_uuids"].append(chunk_id)

        logger.info(f"Grouped {len(chunk_uuids)} chunks into {len(doc_chunks)} documents")

        # Extract relevant document names from reranked sources (these are the ACTUAL sources used in the answer)
        relevant_doc_names = set()
        if reranked_sources:
            for source in reranked_sources:
                doc_name = source.get("document_name") or source.get("title")
                if doc_name:
                    relevant_doc_names.add(doc_name)

        # Sort documents by relevance:
        # 1. Documents that appear in reranked_sources (actually used in answer) - if available
        # 2. Documents with most cited chunks (fallback when GraphRAG doesn't return sources)
        def doc_relevance_score(item):
            doc_name, info = item
            in_reranked = 1 if (relevant_doc_names and doc_name in relevant_doc_names) else 0
            chunk_count = len(info["chunk_uuids"])
            return (in_reranked, chunk_count)  # Sort by (reranked first, then chunk count)

        sorted_docs = sorted(doc_chunks.items(), key=doc_relevance_score, reverse=True)

        # Limit to top 10 most relevant documents to avoid slow storage queries
        top_docs = sorted_docs[:10]
        if relevant_doc_names:
            logger.info(f"Selected top {len(top_docs)} documents from {len(doc_chunks)} total (prioritized by reranked sources: {len(relevant_doc_names)})")
        else:
            logger.info(f"Selected top {len(top_docs)} documents from {len(doc_chunks)} total (prioritized by chunk frequency - no reranked sources available)")

        # Get page numbers from storage
        sources = []
        for doc_title, info in top_docs:
            pages = set()
            document_id = None

            # Query storage for chunk metadata
            for chunk_uuid in info["chunk_uuids"][:10]:  # Limit to first 10 chunks
                try:
                    chunk = self.storage.get_chunk_by_id(str(chunk_uuid))
                    if chunk and chunk.chunk_metadata:
                        page = chunk.chunk_metadata.get("page_number")
                        if page is not None:
                            pages.add(page)
                        if not document_id:
                            document_id = chunk.document_id
                except Exception as e:
                    logger.debug(f"Could not get chunk {chunk_uuid}: {e}")
                    continue

            sources.append({
                "document_name": doc_title,
                "document_id": str(document_id) if document_id else None,
                "pages": pages,
                "chunk_uuids": [str(c) for c in info["chunk_uuids"]],
                "chunk_count": len(info["chunk_uuids"])
            })

        logger.info(f"Citation tracing complete: found {len(sources)} source documents with page numbers")
        return sources

    def _parse_citation_ids(self, answer_text: str) -> Dict[str, set]:
        """Parse inline GraphRAG citations and extract specific IDs."""
        import re

        citation_pattern = r'\[Data: (Reports|Entities|Relationships) \(([0-9, ]+)\)\]'
        citations = re.findall(citation_pattern, answer_text)

        parsed = {
            "report_ids": set(),
            "entity_ids": set(),
            "relationship_ids": set()
        }

        for cit_type, ids_str in citations:
            ids = [int(id.strip()) for id in ids_str.split(',')]

            if cit_type == "Reports":
                parsed["report_ids"].update(ids)
            elif cit_type == "Entities":
                parsed["entity_ids"].update(ids)
            elif cit_type == "Relationships":
                parsed["relationship_ids"].update(ids)

        return parsed

    async def _format_with_harvard_citations(
        self, answer_text: str, sources: List[Dict], collection_id: str, reranked_sources: List[Dict]
    ) -> str:
        """Format answer by replacing inline citations with Harvard-style references using semantic matching."""
        import re

        # Find all citation contexts (sentence + citation)
        citation_pattern = r'([^.!?]*\[Data: (?:Reports|Entities|Relationships) \([0-9, ]+)\][^.!?]*[.!?])'
        citation_contexts = list(re.finditer(citation_pattern, answer_text))

        if not citation_contexts:
            return answer_text

        logger.info(f"Formatting {len(citation_contexts)} citations with semantic matching")

        if not sources:
            logger.warning("No sources available for citation matching")
            return answer_text

        # Extract unique citation contexts for semantic matching
        unique_contexts = {}  # {citation_marker: context_text}

        for match in citation_contexts:
            full_context = match.group(1)

            # Extract the citation marker
            marker_match = re.search(r'\[Data: (?:Reports|Entities|Relationships) \([0-9, ]+)\]', full_context)
            if not marker_match:
                continue

            citation_marker = marker_match.group(0)

            if citation_marker not in unique_contexts:
                # Get context text without citation for semantic matching
                context_text = full_context.replace(citation_marker, '').strip()
                unique_contexts[citation_marker] = context_text

        # Semantically match each citation to the most relevant source
        citation_mappings = await self._match_citations_to_sources(
            unique_contexts, sources, reranked_sources
        )

        # Replace citations using simple string replacement (avoid regex issues with parentheses)
        result = answer_text
        for marker, harvard_citation in citation_mappings.items():
            # Use simple string replacement instead of regex to avoid parenthesis issues
            replacement = f' ({harvard_citation})'
            # Replace with optional leading whitespace handling
            result = result.replace(f' {marker}', replacement)  # Space before marker
            result = result.replace(marker, replacement)  # No space before marker
            logger.info(f"Replaced citation '{marker[:50]}...' with '{replacement}'")

        return result

    async def _match_citations_to_sources(
        self,
        unique_contexts: Dict[str, str],
        sources: List[Dict],
        reranked_sources: List[Dict]
    ) -> Dict[str, str]:
        """
        Semantically match each citation context to the most relevant source document.

        Args:
            unique_contexts: {citation_marker: context_text}
            sources: List of traced source documents
            reranked_sources: Reranked sources for fallback scoring

        Returns:
            {citation_marker: harvard_citation_string}
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        citation_mappings = {}

        if len(sources) == 1:
            # Only one source - use it for all citations
            harvard_citation = self._build_harvard_citation(sources[0], reranked_sources)
            for marker in unique_contexts.keys():
                citation_mappings[marker] = harvard_citation
            logger.info(f"Single source: using same citation for all {len(unique_contexts)} contexts")
            return citation_mappings

        try:
            # Get embeddings for all citation contexts
            context_texts = list(unique_contexts.values())
            logger.info(f"Embedding {len(context_texts)} citation contexts for semantic matching...")

            context_embeddings = await asyncio.to_thread(
                self.embedding_provider.embed_batch,
                context_texts
            )

            # Get representative text for each source (first chunk or document name)
            source_texts = []
            for source in sources:
                # Try to get actual chunk text if available
                doc_name = source.get("document_name", "Unknown")
                # Use document name as fallback (not ideal but works)
                source_texts.append(doc_name)

            logger.info(f"Embedding {len(source_texts)} source documents...")
            source_embeddings = await asyncio.to_thread(
                self.embedding_provider.embed_batch,
                source_texts
            )

            # Compute similarity matrix: contexts x sources
            similarities = cosine_similarity(context_embeddings, source_embeddings)

            # Match each citation to its most similar source
            for idx, (marker, context_text) in enumerate(unique_contexts.items()):
                # Find most similar source
                best_source_idx = np.argmax(similarities[idx])
                best_similarity = similarities[idx][best_source_idx]

                best_source = sources[best_source_idx]
                harvard_citation = self._build_harvard_citation(best_source, reranked_sources)

                citation_mappings[marker] = harvard_citation

                logger.info(
                    f"Matched citation '{context_text[:50]}...' → "
                    f"{best_source.get('document_name', 'Unknown')[:30]} "
                    f"(similarity: {best_similarity:.3f})"
                )

            return citation_mappings

        except Exception as e:
            # Fallback: If semantic matching fails, use frequency-based assignment
            logger.warning(f"Semantic matching failed: {e}. Using fallback (first source for all)")

            harvard_citation = self._build_harvard_citation(sources[0], reranked_sources)
            for marker in unique_contexts.keys():
                citation_mappings[marker] = harvard_citation

            return citation_mappings

    def _build_harvard_citation(self, source: Dict, reranked_sources: List[Dict]) -> str:
        """Build Harvard-style citation from source metadata."""
        # Extract document info
        document_name = source.get("document_name") or source.get("title") or "Unknown"
        pages = source.get("pages", set())

        logger.debug(f"Building citation for: {document_name}, pages: {pages}")

        # Check if this source has reranking score
        reranked_score = None
        if reranked_sources:
            # Find matching source in reranked list
            for rs in reranked_sources:
                if rs.get("document_name") == document_name:
                    reranked_score = rs.get("reranked_score")
                    break

        # Format page numbers
        page_str = ""
        if pages:
            pages_list = sorted(list(pages))[:3]  # First 3 pages
            if len(pages_list) == 1:
                page_str = f", p. {pages_list[0]}"
            else:
                page_str = f", pp. {', '.join(map(str, pages_list))}"

        # Add reranking score if available and enabled
        score_str = ""
        if reranked_score is not None and self.settings.rag.reranking.enabled:
            score_str = f", relevance: {reranked_score:.2f}"

        # Simplified citation format (can be enhanced with author/year parsing)
        return f"{document_name}{page_str}{score_str}"
