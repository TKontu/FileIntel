"""This module defines the GraphRAG service for direct API interaction."""

import os
import pandas as pd
import asyncio
import logging
import shutil
import threading
from typing import List, Dict, Any
from pathlib import Path
from fileintel.storage.models import DocumentChunk
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.rag.graph_rag.adapters.data_adapter import GraphRAGDataAdapter
from fileintel.core.config import Settings, get_config
from fileintel.rag.graph_rag.adapters.config_adapter import GraphRAGConfigAdapter
from fileintel.rag.reranker_service import RerankerService
from .parquet_loader import ParquetLoader
from .dataframe_cache import GraphRAGDataFrameCache
from .._graphrag_imports import global_search, local_search, build_index, GraphRagConfig
from fileintel.prompt_management import AnswerFormatManager

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

        # Initialize embedding provider for semantic citation matching
        from fileintel.llm_integration.embedding_provider import OpenAIEmbeddingProvider
        self.embedding_provider = OpenAIEmbeddingProvider(settings=settings)

        # Initialize reranker service if enabled for graph results
        self.reranker = None
        if settings.rag.reranking.enabled and settings.rag.reranking.rerank_graph_results:
            try:
                self.reranker = RerankerService(settings)
                logger.info("RerankerService initialized for GraphRAG")
            except Exception as e:
                logger.warning(f"Failed to initialize RerankerService: {e}. Continuing without reranking.")

        # Initialize answer format manager and LLM provider for format application
        try:
            # Try to get prompts directory from environment or use relative path
            prompts_dir = os.getenv('FILEINTEL_PROMPTS_DIR')
            if prompts_dir:
                formats_dir = Path(prompts_dir) / "examples"
            else:
                # Fallback to relative path (5 levels up: services -> graph_rag -> rag -> fileintel -> src -> project_root)
                project_root = Path(__file__).parent.parent.parent.parent.parent.parent
                formats_dir = project_root / "prompts" / "examples"

            if formats_dir.exists():
                self.format_manager = AnswerFormatManager(formats_dir)
                logger.info(f"AnswerFormatManager initialized for GraphRAG: {formats_dir}")
            else:
                logger.warning(f"Format templates directory not found: {formats_dir}")
                self.format_manager = None
        except Exception as e:
            logger.warning(f"Failed to initialize AnswerFormatManager: {e}")
            self.format_manager = None

        # Initialize LLM provider for answer reformatting (lazy - only when needed)
        self._llm_provider = None
        self._llm_provider_lock = threading.Lock()

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

    async def query(
        self,
        query: str,
        collection_id: str,
        search_type: str = "global",
        answer_format: str = "default",
        include_cited_chunks: bool = False
    ) -> Dict[str, Any]:
        """
        Standard query interface with citation tracing. Routes to global or local search.

        Args:
            query: The search query
            collection_id: Collection to search
            search_type: "global" for global search, "local" for local search (default: "global")
            answer_format: Answer format template name (default: "default")
            include_cited_chunks: Include full text content of cited chunks in response (default: False)

        Returns:
            Dict with answer (formatted with citations), sources, confidence, and metadata
            If include_cited_chunks=True, also includes 'cited_chunks' field with full chunk content
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
            context = raw_response.get("context", {})
            logger.info(f"GraphRAG context type: {type(context)}, keys: {context.keys() if isinstance(context, dict) else 'N/A'}")
        else:
            answer = getattr(raw_response, "response", str(raw_response))
            context = {}

        # ========================================================================
        # PRIORITY 1: Citation Tracing (Primary Method)
        # ========================================================================
        # Try citation tracing first - traces specific citations in the answer
        # to specific source documents. Most precise method.

        formatted_answer = answer
        traced_sources = []
        cited_chunks = []
        citation_tracing_succeeded = False

        try:
            # Get workspace path for this collection
            index_info = await asyncio.to_thread(
                self.storage.get_graphrag_index_info, collection_id
            )

            if index_info and index_info.get("index_path"):
                workspace_path = index_info["index_path"]
                logger.info("🎯 Using citation tracing (PRIMARY method)")

                # Citation tracing doesn't need context sources - it builds sources from citations
                formatted_answer, traced_sources, cited_chunks = await self._trace_and_format_citations(
                    answer, collection_id, workspace_path, [], include_cited_chunks
                )

                if traced_sources:
                    citation_tracing_succeeded = True
                    logger.info(f"✅ Citation tracing succeeded: {len(traced_sources)} sources traced from answer citations")
                else:
                    logger.warning("⚠️ Citation tracing returned no sources - will use fallback")
            else:
                logger.info(f"ℹ️ Citation tracing unavailable: No index_path found for collection {collection_id}")

        except Exception as e:
            import traceback
            logger.error(f"❌ Citation tracing failed: {e}")
            logger.debug(f"Citation tracing traceback: {traceback.format_exc()}")

        # ========================================================================
        # PRIORITY 2: Context Sources (Fallback Method)
        # ========================================================================
        # If citation tracing didn't work, fall back to sources from GraphRAG context

        if not citation_tracing_succeeded:
            logger.info("🔄 Using context sources (FALLBACK method)")

            # Extract sources from context
            # PRIORITY ORDER: sources > reports > data
            # - sources: Actual source documents (best fallback)
            # - reports: Community summaries (metadata only, not actual documents)
            # - data: Generic fallback field
            context_sources = []

            if isinstance(context, dict):
                # Try common field names in priority order
                for key in ["sources", "reports", "data"]:
                    if key in context:
                        potential_sources = context[key]
                        # Convert DataFrame to list of dicts if needed
                        import pandas as pd
                        if isinstance(potential_sources, pd.DataFrame):
                            context_sources = potential_sources.to_dict('records')
                            logger.info(f"Extracted {len(context_sources)} sources from context['{key}'] (DataFrame)")
                            break
                        elif potential_sources and len(potential_sources) > 0:
                            context_sources = potential_sources
                            logger.info(f"Extracted {len(context_sources)} sources from context['{key}']")
                            break
            else:
                # Context might be a SearchResult object
                context_sources = getattr(context, "sources", []) or getattr(context, "reports", []) or getattr(context, "data", []) or []

            # Rerank context sources if enabled
            context_sources = await self._rerank_sources_if_enabled(query, context_sources)

            # Use context sources as fallback
            traced_sources = context_sources

            if traced_sources:
                logger.info(f"✅ Using {len(traced_sources)} context sources as fallback")
            else:
                logger.warning("⚠️ No sources available from context or citation tracing")

        # Calculate confidence based on result quality
        confidence = self._calculate_confidence(raw_response, traced_sources)

        # Apply answer format if requested (and format manager available)
        final_answer = formatted_answer
        if answer_format != "default" and self.format_manager is not None:
            try:
                final_answer = await self._apply_answer_format(formatted_answer, answer_format)
            except Exception as e:
                logger.warning(f"Failed to apply answer format '{answer_format}': {e}. Using default format.")
                # Keep formatted_answer as fallback

        result = {
            "answer": final_answer,          # Formatted with Harvard citations + answer format
            "raw_answer": answer,            # Original with [Data: Reports (X)]
            "sources": traced_sources,
            "confidence": confidence,
            "metadata": {"search_type": search_type, "answer_format": answer_format},
        }

        # Add cited chunks if requested and available
        if include_cited_chunks and cited_chunks:
            result["cited_chunks"] = cited_chunks
            logger.info(f"Included {len(cited_chunks)} cited chunks in response")

        return result

    async def _apply_answer_format(self, answer: str, answer_format: str) -> str:
        """
        Apply answer format template to GraphRAG response via LLM reformatting.

        Args:
            answer: Original answer with citations
            answer_format: Format template name

        Returns:
            Reformatted answer with citations preserved
        """
        # Get format template
        try:
            format_template = self.format_manager.get_format_template(answer_format)
        except (ValueError, IOError) as e:
            logger.warning(f"Failed to load format template '{answer_format}': {e}")
            return answer  # Return original if template unavailable

        # Lazy-initialize LLM provider with thread safety (double-checked locking)
        if self._llm_provider is None:
            with self._llm_provider_lock:
                # Double-check after acquiring lock
                if self._llm_provider is None:
                    from fileintel.llm_integration.unified_provider import UnifiedLLMProvider
                    self._llm_provider = UnifiedLLMProvider(self.settings, self.storage)

        # Build reformatting prompt (preserving citations is critical)
        reformat_prompt = f"""Reformat the following answer according to the specified format instructions.

CRITICAL: You MUST preserve ALL citations exactly as they appear in the original answer. Do not modify, remove, or change citation formats.

Format Instructions:
{format_template}

Original Answer (with citations):
{answer}

Reformatted Answer (with all citations preserved):"""

        # Use LLM to reformat (run in thread to avoid blocking)
        try:
            response = await asyncio.to_thread(
                self._llm_provider.generate_response,
                prompt=reformat_prompt,
                max_tokens=1000,
                temperature=0.1
            )

            if hasattr(response, "content"):
                return response.content
            elif isinstance(response, dict) and "content" in response:
                return response["content"]
            else:
                return str(response)

        except Exception as e:
            logger.error(f"LLM reformatting failed: {e}")
            return answer  # Fallback to original

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

        logger.info(f"🔍 Starting local_search() call for query: '{query[:50]}...'")
        logger.info(f"📊 DataFrames loaded - entities: {len(dataframes['entities'])}, communities: {len(dataframes['communities'])}, text_units: {len(dataframes.get('text_units', []))}")

        # Call local_search directly - it's async but contains blocking sync operations
        # The Celery task now uses asyncio.run() which creates a fresh event loop,
        # so the blocking operations won't cause event loop conflicts
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

        logger.info(f"✅ local_search() completed successfully")

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

    async def _fetch_chunk_contents(
        self,
        chunk_uuids: List[str],
        collection_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch full text content for chunk UUIDs from PostgreSQL.

        Args:
            chunk_uuids: List of chunk UUIDs to fetch
            collection_id: Collection ID

        Returns:
            List of dicts with {chunk_id, text, document_title, page, metadata}
            Full chunk text is included without truncation
        """
        if not chunk_uuids:
            return []

        try:
            # Fetch chunks from PostgreSQL using individual fetch (no bulk method available)
            # Batch the requests to avoid overwhelming storage
            chunk_data = []
            batch_size = 100

            for i in range(0, len(chunk_uuids), batch_size):
                batch_uuids = chunk_uuids[i:i+batch_size]

                for chunk_uuid in batch_uuids:
                    try:
                        # Fetch chunk using sync storage method in thread
                        chunk = await asyncio.to_thread(
                            self.storage.get_chunk_by_id,
                            chunk_uuid
                        )

                        if chunk:
                            # Extract chunk data with safe attribute access
                            # DocumentChunk model has: id, chunk_text, chunk_metadata, document (relationship)
                            # Try to get document title from related document if available
                            doc_title = "Unknown"
                            if hasattr(chunk, 'document') and chunk.document:
                                doc_title = getattr(chunk.document, 'original_filename', 'Unknown')

                            # Page number might be in chunk_metadata
                            chunk_meta = getattr(chunk, 'chunk_metadata', {}) or {}
                            page_num = chunk_meta.get('page_number') or chunk_meta.get('page')

                            chunk_data.append({
                                "chunk_id": getattr(chunk, 'id', chunk_uuid),
                                "text": getattr(chunk, 'chunk_text', ''),  # FULL text, no truncation
                                "document_title": doc_title,
                                "page": page_num,
                                "metadata": chunk_meta
                            })
                    except Exception as e:
                        logger.debug(f"Failed to fetch chunk {chunk_uuid}: {e}")
                        continue

            logger.info(f"Fetched {len(chunk_data)} / {len(chunk_uuids)} cited chunks with full content")
            return chunk_data

        except Exception as e:
            logger.error(f"Failed to fetch chunk contents: {e}")
            return []

    async def _trace_and_format_citations(
        self, answer: str, collection_id: str, workspace_path: str, reranked_sources: List[Dict[str, Any]],
        include_cited_chunks: bool = False
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        SERVER-SIDE: Trace GraphRAG citations to source documents and format with Harvard citations.

        Args:
            answer: Raw GraphRAG answer with inline citations like [Data: Reports (6826)]
            collection_id: Collection ID
            workspace_path: Path to GraphRAG output directory (e.g., /data/.../output)
            reranked_sources: Sources that have been reranked (if enabled)
            include_cited_chunks: If True, fetch and return full chunk content

        Returns:
            Tuple of (formatted_answer, traced_sources, cited_chunks)
        """
        import re
        import os
        import pandas as pd

        # 1. Parse individual citations with their contexts
        citation_contexts = self._parse_citation_contexts(answer)

        if not citation_contexts:
            # No citations to trace
            return answer, reranked_sources, []

        logger.info(f"Tracing {len(citation_contexts)} individual citations")

        # 2. SERVER-SIDE: Trace each citation individually to maintain citation→chunk mapping
        citation_mappings, all_sources, cited_chunks = await asyncio.to_thread(
            self._trace_citations_individually,
            citation_contexts,
            workspace_path,
            reranked_sources,
            include_cited_chunks,
            collection_id
        )

        if not citation_mappings:
            logger.warning("Citation tracing failed or returned no mappings - returning raw answer without citations")
            # Return raw answer with GraphRAG's original inline citations and reranked sources
            return answer, reranked_sources, []

        logger.info(f"Traced {len(citation_mappings)} citations to {len(all_sources)} unique sources")

        # 3. Format answer with Harvard citations using the specific mappings
        # Also pass citation_contexts to remove low-quality/failed citations
        formatted_answer = self._apply_citation_mappings(answer, citation_mappings, citation_contexts)

        return formatted_answer, all_sources, cited_chunks

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
                "pages": sorted(list(pages)) if pages else [],  # Convert set to sorted list for JSON serialization
                "chunk_uuids": [str(c) for c in info["chunk_uuids"]],
                "chunk_count": len(info["chunk_uuids"])
            })

        logger.info(f"Citation tracing complete: found {len(sources)} source documents with page numbers")
        return sources

    def _parse_citation_contexts(self, answer_text: str) -> List[Dict[str, Any]]:
        """
        Parse individual citations with their context text.
        Each citation gets a UNIQUE marker based on its position to avoid replacement conflicts.
        Supports both simple and compound citations:
        - Simple: [Data: Reports (123)]
        - Compound: [Data: Reports (123); Entities (456, 789)]

        Returns:
            List of dicts with {marker, original_marker, types_data, context_text, start, end}
            types_data = [{type: "Reports", ids: [123]}, {type: "Entities", ids: [456, 789]}]
        """
        import re

        # Match entire citation block (simple or compound)
        # Pattern matches: [Data: TYPE (IDs); TYPE (IDs); ...]
        citation_pattern = r'\[Data: ([^\]]+)\]'

        citation_contexts = []
        for match in re.finditer(citation_pattern, answer_text):
            original_marker = match.group(0)
            inner_content = match.group(1)  # Everything between [Data: and ]

            # Parse individual type-ID pairs separated by semicolons
            # Each part is like "Reports (123, 456)" or "Entities (789)"
            type_pattern = r'(Reports|Entities|Relationships|Sources)\s*\(([0-9, ]+)\)'
            types_data = []

            for type_match in re.finditer(type_pattern, inner_content):
                cit_type = type_match.group(1)
                ids_str = type_match.group(2)
                ids = [int(id.strip()) for id in ids_str.split(',')]
                types_data.append({
                    "type": cit_type,
                    "ids": ids
                })

            if not types_data:
                # Skip if we couldn't parse any citation types
                logger.warning(f"Could not parse citation: {original_marker}")
                continue

            # Extract surrounding context (sentence containing the citation)
            start = match.start()
            end = match.end()

            # Find sentence boundaries
            sentence_start = answer_text.rfind('.', 0, start) + 1
            sentence_end = answer_text.find('.', end)
            if sentence_end == -1:
                sentence_end = len(answer_text)
            else:
                sentence_end += 1

            context_text = answer_text[sentence_start:sentence_end].strip()
            # Remove the citation marker from context for semantic matching
            context_text_clean = context_text.replace(original_marker, '').strip()

            # Create UNIQUE marker with position to avoid replacement conflicts
            unique_marker = f"{original_marker}__POS_{start}__"

            citation_contexts.append({
                "marker": unique_marker,  # Unique marker for replacement
                "original_marker": original_marker,  # Original for finding in text
                "types_data": types_data,  # List of {type, ids} dicts
                "context_text": context_text_clean,
                "start": start,  # Position in text
                "end": end
            })

        return citation_contexts

    def _trace_citations_individually(
        self,
        citation_contexts: List[Dict[str, Any]],
        workspace_path: str,
        reranked_sources: List[Dict[str, Any]],
        include_cited_chunks: bool = False,
        collection_id: str = None
    ) -> tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Trace each citation individually to maintain citation→chunk→page mapping.
        OPTIMIZED for massive collections with batch operations.

        Args:
            citation_contexts: List of citation contexts to trace
            workspace_path: Path to GraphRAG workspace
            reranked_sources: List of reranked sources
            include_cited_chunks: If True, fetch and return full chunk content
            collection_id: Collection ID (required if include_cited_chunks=True)

        Returns:
            Tuple of (citation_mappings, all_sources, cited_chunks)
            - citation_mappings: {marker: harvard_citation}
            - all_sources: List of unique source documents referenced
            - cited_chunks: List of chunk content dicts (empty if include_cited_chunks=False)
        """
        import pandas as pd
        import os
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Helper to safely load parquet files
        def load_parquet_safe(filename):
            path = os.path.join(workspace_path, filename)
            if os.path.exists(path):
                try:
                    return pd.read_parquet(path)
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            return None

        # Load required parquet files once (cached in memory for all citations)
        communities_df = load_parquet_safe("communities.parquet")
        entities_df = load_parquet_safe("entities.parquet")
        relationships_df = load_parquet_safe("relationships.parquet")
        text_units_df = load_parquet_safe("text_units.parquet")
        documents_df = load_parquet_safe("documents.parquet")

        if communities_df is None or entities_df is None or text_units_df is None or documents_df is None:
            logger.error(f"Required parquet files not found in {workspace_path}")
            return {}, []

        # CRITICAL: Ensure ID columns are consistent types for lookups
        # GraphRAG citations use int short_ids (human_readable_id), not the UUID id field
        # SAFETY: Don't modify DataFrames - convert lookup values instead to avoid SIGSEGV
        # Log types for debugging
        logger.debug(f"Entity human_readable_id column type: {entities_df['human_readable_id'].dtype}")
        logger.debug(f"Community column type: {communities_df['community'].dtype if 'community' in communities_df.columns else 'N/A'}")
        logger.debug(f"Text unit ID column type: {text_units_df['id'].dtype if 'id' in text_units_df.columns else 'N/A'}")
        if relationships_df is not None:
            logger.debug(f"Relationship human_readable_id column type: {relationships_df['human_readable_id'].dtype if 'human_readable_id' in relationships_df.columns else 'N/A'}")

        # OPTIMIZATION 1a: Calculate information density for all text units
        # Density = number of entities + relationships (semantic richness indicator)
        logger.info("Calculating information density for text units...")
        try:
            text_units_df['info_density'] = text_units_df.apply(
                lambda x: (len(x['entity_ids']) if x['entity_ids'] is not None and hasattr(x['entity_ids'], '__len__') else 0) +
                         (len(x['relationship_ids']) if x['relationship_ids'] is not None and hasattr(x['relationship_ids'], '__len__') else 0),
                axis=1
            )
            logger.info(f"Density stats - Mean: {text_units_df['info_density'].mean():.2f}, Max: {text_units_df['info_density'].max()}, Min: {text_units_df['info_density'].min()}")
        except Exception as e:
            logger.warning(f"Failed to calculate info_density: {e}. Using uniform density.")
            text_units_df['info_density'] = 1

        # OPTIMIZATION 1b: Collect text units and apply density-based filtering
        all_relevant_text_units = set()
        citation_to_text_units = {}  # {marker: [text_unit_ids]}
        citation_to_chunks = {}  # {marker: {doc_title: [chunk_uuids]}}

        for citation in citation_contexts:
            marker = citation["marker"]
            types_data = citation["types_data"]  # List of {type, ids} dicts

            # Step 1: Collect text unit IDs from ALL types in this citation (handles compound citations)
            text_unit_ids = set()
            citation_summary = []  # For logging: ["Reports (3 IDs)", "Entities (5 IDs)"]

            for type_data in types_data:
                cit_type = type_data["type"]
                ids = type_data["ids"]
                citation_summary.append(f"{cit_type} ({len(ids)} IDs)")

                if cit_type == "Sources":
                    # Sources citations contain text unit IDs directly (no lookup needed)
                    # [Data: Sources (8137, 43990)] → text_unit_ids = {8137, 43990}
                    text_unit_ids.update(ids)
                    logger.debug(f"Citation {marker} [{cit_type}]: {len(ids)} source IDs (text units) added directly")

                elif cit_type == "Relationships":
                    # Direct lookup of relationships by human_readable_id
                    if relationships_df is not None:
                        rel_mask = relationships_df["human_readable_id"].isin(ids)
                        matched_rels = relationships_df[rel_mask]
                        logger.debug(f"Citation {marker} [{cit_type}]: {len(ids)} relationship IDs → {len(matched_rels)} matched")

                        for tu_list in matched_rels["text_unit_ids"]:
                            if tu_list is not None and len(tu_list) > 0:
                                text_unit_ids.update(tu_list)
                    else:
                        logger.warning(f"Citation {marker} [{cit_type}]: Relationships DataFrame not loaded")

                else:
                    # For Reports and Entities, get entity IDs first, then text units
                    entity_short_ids = set()  # Changed name to clarify these are human_readable_ids (integers)

                    if cit_type == "Reports":
                        # CRITICAL: Communities.entity_ids contains UUID strings (entity.id field)
                        # But citations contain integers (entity.human_readable_id field)
                        # We need to convert UUIDs → human_readable_ids
                        comm_mask = communities_df["community"].isin(ids)
                        entity_uuids = set()
                        for entity_list in communities_df[comm_mask]["entity_ids"]:
                            if entity_list is not None and len(entity_list) > 0:
                                entity_uuids.update(entity_list)

                        if entity_uuids:
                            # Convert entity UUIDs to human_readable_ids by looking them up
                            uuid_mask = entities_df["id"].isin(entity_uuids)
                            matched_by_uuid = entities_df[uuid_mask]
                            entity_short_ids = set(matched_by_uuid["human_readable_id"].dropna().astype(int).tolist())
                            logger.debug(f"Citation {marker} [{cit_type}]: {len(entity_uuids)} entity UUIDs from communities → {len(entity_short_ids)} human_readable_ids")

                    elif cit_type == "Entities":
                        # Entity citations already contain human_readable_ids (integers)
                        entity_short_ids.update(ids)

                    if entity_short_ids:
                        # Step 2: Get text units for these specific entities
                        # CRITICAL: Citations use short_id (human_readable_id), not id!
                        # GraphRAG shows short_id to LLM in context, so citations contain short_id values
                        entity_mask = entities_df["human_readable_id"].isin(entity_short_ids)
                        matched_entities = entities_df[entity_mask]
                        logger.debug(f"Citation {marker} [{cit_type}]: {len(entity_short_ids)} entity IDs (short_id) → {len(matched_entities)} matched in DataFrame")

                        if len(matched_entities) == 0:
                            logger.warning(f"Citation {marker} [{cit_type}]: None of the {len(entity_short_ids)} entity short_ids found in entities DataFrame")
                            logger.warning(f"  Citation short_ids: {sorted(list(entity_short_ids))[:10]}")
                            logger.warning(f"  DataFrame human_readable_id range: {entities_df['human_readable_id'].min()} - {entities_df['human_readable_id'].max()}")
                            logger.warning(f"  DataFrame entity count: {len(entities_df)}")
                            logger.warning(f"  Sample DataFrame human_readable_ids: {sorted(entities_df['human_readable_id'].dropna().head(10).tolist())}")
                        else:
                            for tu_list in matched_entities["text_unit_ids"]:
                                if tu_list is not None and len(tu_list) > 0:
                                    text_unit_ids.update(tu_list)
                    else:
                        logger.debug(f"Citation {marker} [{cit_type}]: No entity IDs found")

            # Check if we found any text units from ALL types in this citation
            if not text_unit_ids:
                logger.warning(f"Citation {marker}: No text_unit_ids found for citation with types: {', '.join(citation_summary)}")
                continue

            logger.debug(f"Citation {marker} ({', '.join(citation_summary)}): Collected {len(text_unit_ids)} text units total")

            citation_to_text_units[marker] = text_unit_ids
            all_relevant_text_units.update(text_unit_ids)

        logger.info(f"Collected {len(all_relevant_text_units)} unique text units across {len(citation_contexts)} citations")

        # OPTIMIZATION 1c: Smart chunk selection using CITATION-SPECIFIC density filtering
        # For each citation, select its most dense text units, then collect chunks
        # This ensures each citation gets relevant chunks, not globally dense ones
        MAX_CHUNKS_PER_CITATION = 5000  # Reasonable limit per citation

        chunk_candidates = set()
        chunk_density_scores = {}  # Track density for tiebreaking
        chunk_frequency = {}  # Track how often each chunk appears

        # Get text units with density scores
        relevant_tus = text_units_df[text_units_df["id"].isin(all_relevant_text_units)]

        # Process each citation separately to ensure relevance
        for marker, text_unit_ids in citation_to_text_units.items():
            # Get text units for THIS SPECIFIC citation
            citation_tus = relevant_tus[relevant_tus["id"].isin(text_unit_ids)]

            # If this citation has too many text units, take the most dense ones
            if len(citation_tus) > 1000:
                logger.info(f"Citation {marker}: {len(citation_tus)} text units, filtering to top 1000 by density")
                citation_tus = citation_tus.nlargest(1000, 'info_density')

            # Collect chunks from this citation's text units
            for _, tu in citation_tus.iterrows():
                doc_ids = tu.get("document_ids")
                density = tu.get("info_density", 0)
                if doc_ids is not None and len(doc_ids) > 0:
                    for chunk_id in doc_ids:
                        chunk_id_str = str(chunk_id)
                        chunk_candidates.add(chunk_id_str)
                        chunk_frequency[chunk_id_str] = chunk_frequency.get(chunk_id_str, 0) + 1
                        # Track max density for this chunk
                        chunk_density_scores[chunk_id_str] = max(
                            chunk_density_scores.get(chunk_id_str, 0), density
                        )

        logger.info(f"Citation-specific filtering produced {len(chunk_candidates)} candidate chunks")

        # If still too many total chunks, prioritize by:
        # 1. Frequency (chunks appearing in multiple citations)
        # 2. Density (chunks from high-density text units)
        MAX_TOTAL_CHUNKS = 10000
        if len(chunk_candidates) > MAX_TOTAL_CHUNKS:
            logger.info(f"Too many chunks ({len(chunk_candidates)}). Prioritizing by frequency + density...")
            # Score = frequency * 100 + density (frequency is primary, density is tiebreaker)
            chunk_scores = {
                chunk_id: chunk_frequency[chunk_id] * 100 + chunk_density_scores.get(chunk_id, 0)
                for chunk_id in chunk_candidates
            }
            sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
            chunk_candidates = set([c[0] for c in sorted_chunks[:MAX_TOTAL_CHUNKS]])
            logger.info(f"Reduced to {len(chunk_candidates)} chunks (top by frequency + density)")

        # Map chunks to documents for citation grouping
        chunk_uuid_strs = list(chunk_candidates)
        doc_mask = documents_df["id"].isin(chunk_uuid_strs)
        matched_docs = documents_df[doc_mask]

        # Build chunk-to-document mapping
        chunk_to_doc_title = {}
        for _, row in matched_docs.iterrows():
            chunk_id = str(row.get("id"))
            doc_title = row.get("title", "Unknown")
            chunk_to_doc_title[chunk_id] = doc_title

        # Group chunks by citation and document
        for marker, text_unit_ids in citation_to_text_units.items():
            # Get chunks for this citation's text units
            citation_text_units = text_units_df[text_units_df["id"].isin(text_unit_ids)]
            citation_chunks = set()

            for _, tu in citation_text_units.iterrows():
                doc_ids = tu.get("document_ids")
                if doc_ids is not None and len(doc_ids) > 0:
                    for chunk_id in doc_ids:
                        chunk_id_str = str(chunk_id)
                        if chunk_id_str in chunk_candidates:  # Only include if in our filtered set
                            citation_chunks.add(chunk_id_str)

            # Group by document
            doc_chunks = {}
            for chunk_id in citation_chunks:
                doc_title = chunk_to_doc_title.get(chunk_id, "Unknown")
                if doc_title not in doc_chunks:
                    doc_chunks[doc_title] = []
                doc_chunks[doc_title].append(chunk_id)

            if doc_chunks:
                citation_to_chunks[marker] = doc_chunks

        logger.info(f"Smart chunk selection: {len(chunk_candidates)} chunks for embedding (density + frequency filtered)")

        # OPTIMIZATION 2: Batch fetch all selected chunks at once
        chunk_cache = self._batch_fetch_chunks(list(chunk_candidates))
        logger.info(f"Batch fetched {len(chunk_cache)} chunks from storage")

        # OPTIMIZATION 3a: Batch embed all citation contexts at once
        context_texts = [c["context_text"] for c in citation_contexts]
        try:
            context_embeddings = self.embedding_provider.get_embeddings(context_texts)
            logger.info(f"Batch embedded {len(context_embeddings)} citation contexts")
        except Exception as e:
            logger.error(f"Batch embedding failed for citation tracing: {e}")
            logger.error("Cannot perform semantic citation matching - returning answer without detailed citations")
            # Return empty mappings - caller will handle this gracefully
            return {}, []

        # OPTIMIZATION 3b: Use PRE-COMPUTED embeddings from PostgreSQL
        # Chunks already have embeddings stored - no need to re-embed!
        all_chunk_embeddings = []
        chunk_uuid_list = []
        chunk_uuid_to_index = {}
        chunks_to_embed = []  # Fallback for chunks without embeddings
        chunks_to_embed_indices = []

        logger.info("Extracting pre-computed embeddings from PostgreSQL chunks...")
        for chunk_uuid, chunk in chunk_cache.items():
            if chunk:
                chunk_uuid_list.append(chunk_uuid)
                idx = len(chunk_uuid_list) - 1
                chunk_uuid_to_index[chunk_uuid] = idx

                if chunk.embedding is not None and len(chunk.embedding) > 0:
                    # Use pre-computed embedding (convert from pgvector to list if needed)
                    embedding = chunk.embedding
                    if not isinstance(embedding, list):
                        embedding = list(embedding)  # Convert numpy array or pgvector to list
                    all_chunk_embeddings.append(embedding)
                else:
                    # Mark for embedding (rare case - chunk without embedding)
                    all_chunk_embeddings.append(None)  # Placeholder
                    chunks_to_embed.append(chunk.chunk_text[:500] if chunk.chunk_text else "")
                    chunks_to_embed_indices.append(idx)

        # Embed any chunks that don't have pre-computed embeddings (fallback)
        if chunks_to_embed:
            logger.warning(f"{len(chunks_to_embed)} chunks missing embeddings - computing now...")
            try:
                new_embeddings = self.embedding_provider.get_embeddings(chunks_to_embed)
                for i, embedding in enumerate(new_embeddings):
                    all_chunk_embeddings[chunks_to_embed_indices[i]] = embedding
            except Exception as e:
                logger.error(f"Failed to embed {len(chunks_to_embed)} chunks: {e}")
                # Remove chunks without embeddings from consideration
                for idx in reversed(chunks_to_embed_indices):
                    all_chunk_embeddings[idx] = [0.0] * 1024  # Zero vector fallback

        logger.info(f"Using embeddings for {len(all_chunk_embeddings)} chunks ({len(chunk_cache) - len(chunks_to_embed)} pre-computed, {len(chunks_to_embed)} newly computed)")

        # Process each citation with cached data
        citation_mappings = {}
        all_sources_dict = {}  # {doc_name: source_dict} to deduplicate

        for idx, citation in enumerate(citation_contexts):
            marker = citation["marker"]
            context_text = citation["context_text"]

            if marker not in citation_to_chunks:
                logger.warning(f"No chunks found for citation {marker}")
                continue

            doc_chunks = citation_to_chunks[marker]
            logger.info(f"Citation {marker} references {len(doc_chunks)} documents")

            # NEW APPROACH: Find most relevant chunks GLOBALLY across all documents in the community
            # This is more accurate than picking a document first
            try:
                context_embedding = context_embeddings[idx]

                # Collect ALL chunks across ALL documents referenced by this citation
                all_chunk_uuids = []
                for chunk_list in doc_chunks.values():
                    all_chunk_uuids.extend(chunk_list)

                logger.debug(f"Searching across {len(all_chunk_uuids)} total chunks for citation {marker}")

                # Find globally most relevant chunks (not limited to one document)
                # Use pre-computed embeddings for efficiency
                relevant_chunks_with_scores = self._find_most_relevant_chunks_with_precomputed_embeddings(
                    context_embedding,
                    all_chunk_uuids,
                    chunk_uuid_to_index,
                    all_chunk_embeddings,
                    top_k=10
                )

                if not relevant_chunks_with_scores:
                    logger.error(f"No relevant chunks found for citation {marker}")
                    continue

                # Group selected chunks by document and create sources
                sources_by_doc = {}
                for chunk_uuid, similarity in relevant_chunks_with_scores:
                    # Find which document this chunk belongs to
                    chunk_doc_title = None
                    for doc_title, doc_chunk_list in doc_chunks.items():
                        if chunk_uuid in doc_chunk_list:
                            chunk_doc_title = doc_title
                            break

                    if chunk_doc_title:
                        if chunk_doc_title not in sources_by_doc:
                            sources_by_doc[chunk_doc_title] = {
                                "chunk_uuids": [],
                                "similarities": []
                            }
                        sources_by_doc[chunk_doc_title]["chunk_uuids"].append(chunk_uuid)
                        sources_by_doc[chunk_doc_title]["similarities"].append(similarity)

                # Edge case: no chunks mapped to documents (shouldn't happen but safety check)
                if not sources_by_doc:
                    logger.error(f"No chunks mapped to documents for citation {marker}")
                    continue

                # Use the document with the most relevant chunks (or highest avg similarity)
                best_doc_title = max(
                    sources_by_doc.keys(),
                    key=lambda d: (len(sources_by_doc[d]["chunk_uuids"]),
                                   sum(sources_by_doc[d]["similarities"]))
                )
                best_chunk_uuids = sources_by_doc[best_doc_title]["chunk_uuids"]
                best_similarity = sum(sources_by_doc[best_doc_title]["similarities"]) / len(sources_by_doc[best_doc_title]["similarities"])

                # QUALITY THRESHOLD: Only include citations with reasonable similarity
                # With citation-specific filtering, we can be more lenient (0.55+)
                # GraphRAG already filtered to relevant entities, so moderate similarity is acceptable
                # Lowered from 0.65 to 0.55 to include more relevant citations
                SIMILARITY_THRESHOLD = 0.55
                if best_similarity < SIMILARITY_THRESHOLD:
                    logger.warning(f"Citation {marker} has low similarity ({best_similarity:.3f} < {SIMILARITY_THRESHOLD}) - excluding citation")
                    # Don't add to citation_mappings - marker will be removed from text
                    continue

                # Create source from the globally most relevant chunks
                best_source = self._create_source_from_chunks_cached(
                    best_doc_title, best_chunk_uuids, reranked_sources, chunk_cache
                )

                logger.info(f"Global chunk search: {best_doc_title} (avg similarity: {best_similarity:.3f}, {len(best_chunk_uuids)} chunks, pages: {best_source.get('pages', [])})")

            except Exception as e:
                logger.error(f"Global chunk search failed for citation {marker}: {e}")
                continue

            if not best_source:
                logger.warning(f"No best source found for citation {marker}")
                continue

            # Fetch document metadata once for both Harvard and in-text citations
            document_id = best_source.get("document_id")
            document_metadata = {}
            if document_id:
                try:
                    doc = self.storage.get_document(document_id)
                    if doc and doc.document_metadata:
                        document_metadata = doc.document_metadata
                        logger.debug(f"Fetched metadata for {best_doc_title}: {document_metadata.keys()}")
                except Exception as e:
                    logger.debug(f"Could not fetch metadata for document {document_id}: {e}")

            # Build Harvard citation for this specific source
            harvard_citation = self._build_harvard_citation(best_source, reranked_sources)
            citation_mappings[marker] = harvard_citation

            # Enrich source with citation fields for proper display (matches Vector RAG format)
            from fileintel.citation import format_in_text_citation

            # Build chunk-like dict for in-text citation WITH METADATA for proper Harvard format
            chunk_data_for_citation = {
                "document_id": document_id,
                "document_metadata": document_metadata,  # Include metadata for Harvard-style citations
                "original_filename": best_doc_title,
                "chunk_metadata": {"pages": best_source.get("pages", [])}
            }
            in_text_citation = format_in_text_citation(chunk_data_for_citation)

            # Add citation fields to source
            best_source["citation"] = harvard_citation
            best_source["in_text_citation"] = in_text_citation
            best_source["similarity_score"] = best_similarity
            best_source["relevance_score"] = best_similarity  # CLI compatibility
            best_source["filename"] = best_doc_title  # CLI/API compatibility

            # Add to all_sources (deduplicate by document name)
            doc_name = best_source.get("document_name")
            if doc_name not in all_sources_dict:
                all_sources_dict[doc_name] = best_source
            else:
                # Merge pages and chunks, keep highest similarity score
                existing = all_sources_dict[doc_name]
                merged_pages = sorted(list(set(existing.get("pages", []) + best_source.get("pages", []))))[:5]
                existing["pages"] = merged_pages
                existing["chunk_uuids"] = list(set(existing.get("chunk_uuids", []) + best_source.get("chunk_uuids", [])))
                existing["chunk_count"] = len(existing["chunk_uuids"])
                # Update similarity to highest score
                existing["similarity_score"] = max(existing.get("similarity_score", 0), best_similarity)
                existing["relevance_score"] = existing["similarity_score"]

                # Regenerate in_text_citation with merged pages (reuse fetched metadata)
                chunk_data_for_merged = {
                    "document_id": existing.get("document_id"),
                    "document_metadata": document_metadata,  # Reuse metadata fetched above
                    "original_filename": doc_name,
                    "chunk_metadata": {"pages": merged_pages}
                }
                existing["in_text_citation"] = format_in_text_citation(chunk_data_for_merged)

            logger.info(f"Matched citation {marker} → {doc_name} (pages: {best_source.get('pages', [])}, similarity: {best_similarity:.3f})")

        all_sources = list(all_sources_dict.values())

        # Collect cited chunks if requested
        cited_chunks = []
        if include_cited_chunks and collection_id:
            # Collect all chunk UUIDs that were actually cited
            all_cited_chunk_uuids = set()
            for source in all_sources:
                chunk_uuids = source.get("chunk_uuids", [])
                all_cited_chunk_uuids.update(chunk_uuids)

            if all_cited_chunk_uuids:
                logger.info(f"Fetching full content for {len(all_cited_chunk_uuids)} cited chunks...")
                # Fetch chunk contents - need to use asyncio.run since this is a sync method
                # but _fetch_chunk_contents is async
                try:
                    import asyncio
                    # Check if we're already in an event loop
                    # This should NOT happen since we're called via asyncio.to_thread()
                    # which isolates us from the parent loop
                    try:
                        loop = asyncio.get_running_loop()
                        # Unexpected: We're in an event loop
                        logger.error(
                            f"UNEXPECTED: Already in event loop while fetching cited chunks. "
                            f"This indicates asyncio.to_thread() isolation failed. "
                            f"Cannot fetch {len(all_cited_chunk_uuids)} chunks - returning empty."
                        )
                        # Return empty rather than risk deadlock
                        cited_chunks = []
                    except RuntimeError:
                        # Expected: No event loop - safe to use asyncio.run()
                        cited_chunks = asyncio.run(
                            self._fetch_chunk_contents(list(all_cited_chunk_uuids), collection_id)
                        )
                        logger.info(f"Successfully fetched content for {len(cited_chunks)} cited chunks")
                except Exception as e:
                    logger.error(f"Failed to fetch cited chunk contents: {e}", exc_info=True)
                    cited_chunks = []

        return citation_mappings, all_sources, cited_chunks

    def _batch_fetch_chunks(self, chunk_uuids: List[str]) -> Dict[str, Any]:
        """
        Batch fetch chunks from storage to minimize database queries.

        Returns:
            Dict mapping chunk_uuid → chunk object
        """
        chunk_cache = {}

        # Batch query chunks (limit to 100 at a time to avoid overwhelming DB)
        batch_size = 100
        for i in range(0, len(chunk_uuids), batch_size):
            batch = chunk_uuids[i:i+batch_size]
            for chunk_uuid in batch:
                try:
                    chunk = self.storage.get_chunk_by_id(chunk_uuid)
                    if chunk:
                        chunk_cache[chunk_uuid] = chunk
                except Exception as e:
                    logger.debug(f"Could not fetch chunk {chunk_uuid}: {e}")
                    continue

        return chunk_cache

    def _find_most_relevant_chunks_with_precomputed_embeddings(
        self,
        context_embedding: List[float],
        chunk_uuids: List[str],
        chunk_uuid_to_index: Dict[str, int],
        all_chunk_embeddings: List[List[float]],
        top_k: int = 10
    ) -> List[tuple]:
        """
        Find the most semantically relevant chunks using PRE-COMPUTED embeddings.
        This is much more efficient than re-embedding chunks for each citation.

        Args:
            context_embedding: Embedding of the citation context
            chunk_uuids: All chunk UUIDs to search
            chunk_uuid_to_index: Mapping from chunk_uuid to index in all_chunk_embeddings
            all_chunk_embeddings: Pre-computed embeddings for ALL cached chunks
            top_k: Number of top chunks to return

        Returns:
            List of tuples: [(chunk_uuid, similarity_score), ...]
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Filter to only chunks we have embeddings for
        valid_uuids = []
        valid_indices = []

        for chunk_uuid in chunk_uuids:
            if str(chunk_uuid) in chunk_uuid_to_index:
                valid_uuids.append(chunk_uuid)
                valid_indices.append(chunk_uuid_to_index[str(chunk_uuid)])

        if not valid_uuids:
            logger.warning("No valid chunks with embeddings found")
            return []

        if len(valid_uuids) <= top_k:
            # Fewer chunks than top_k, return all with computed similarity
            chunk_embeddings = [all_chunk_embeddings[i] for i in valid_indices]
            similarities = cosine_similarity([context_embedding], chunk_embeddings)[0]
            return [(valid_uuids[i], float(similarities[i])) for i in range(len(valid_uuids))]

        # Get embeddings for these chunks (no API call needed - already computed!)
        chunk_embeddings = [all_chunk_embeddings[i] for i in valid_indices]

        # Compute similarity to context
        similarities = cosine_similarity([context_embedding], chunk_embeddings)[0]

        # Get top_k most similar chunks with their scores
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
        results = [(valid_uuids[i], float(similarities[i])) for i in top_indices]

        logger.debug(f"Selected {len(results)} most relevant chunks (similarity range: {similarities[top_indices[-1]]:.3f} - {similarities[top_indices[0]]:.3f})")
        return results

    def _find_most_relevant_chunks_global(
        self,
        context_embedding: List[float],
        chunk_uuids: List[str],
        chunk_cache: Dict[str, Any],
        top_k: int = 10
    ) -> List[tuple]:
        """
        Find the most semantically relevant chunks globally across documents.

        Args:
            context_embedding: Embedding of the citation context
            chunk_uuids: All chunk UUIDs to search (from multiple documents)
            chunk_cache: Pre-fetched chunks
            top_k: Number of top chunks to return

        Returns:
            List of tuples: [(chunk_uuid, similarity_score), ...]
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Collect chunk texts for embedding
        chunk_texts = []
        valid_uuids = []

        for chunk_uuid in chunk_uuids:
            chunk = chunk_cache.get(str(chunk_uuid))
            if chunk and chunk.chunk_text:
                chunk_texts.append(chunk.chunk_text[:500])  # First 500 chars
                valid_uuids.append(chunk_uuid)

        if not chunk_texts:
            logger.warning("No valid chunk texts found for global search")
            return []

        if len(chunk_texts) <= top_k:
            # Fewer chunks than top_k, return all with placeholder similarity
            return [(uuid, 1.0) for uuid in valid_uuids]

        # Embed all chunks
        try:
            chunk_embeddings = self.embedding_provider.get_embeddings(chunk_texts)

            # Compute similarity to context
            similarities = cosine_similarity([context_embedding], chunk_embeddings)[0]

            # Get top_k most similar chunks with their scores
            top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
            results = [(valid_uuids[i], float(similarities[i])) for i in top_indices]

            logger.debug(f"Selected {len(results)} most relevant chunks globally (similarity range: {similarities[top_indices[-1]]:.3f} - {similarities[top_indices[0]]:.3f})")
            return results

        except Exception as e:
            logger.error(f"Failed to find most relevant chunks globally: {e}")
            return []

    def _create_source_from_chunks_cached(
        self,
        doc_title: str,
        chunk_uuids: List[str],
        reranked_sources: List[Dict[str, Any]],
        chunk_cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a source dict from specific chunks using cached chunk data.
        This avoids repeated storage queries.
        """
        pages = set()
        document_id = None

        # Get page numbers from these SPECIFIC chunks using cache
        for chunk_uuid in chunk_uuids[:20]:  # Limit to 20 chunks for performance
            chunk = chunk_cache.get(str(chunk_uuid))
            if chunk and chunk.chunk_metadata:
                # PostgreSQL stores pages as a LIST in chunk_metadata["pages"]
                chunk_pages = chunk.chunk_metadata.get("pages")
                if chunk_pages:
                    if isinstance(chunk_pages, list):
                        pages.update(chunk_pages)
                    else:
                        pages.add(chunk_pages)  # Handle single page number
                else:
                    # Fallback: check for singular "page_number" field
                    page_num = chunk.chunk_metadata.get("page_number")
                    if page_num is not None:
                        pages.add(page_num)
                    else:
                        logger.debug(f"Chunk {chunk_uuid}: No pages found in metadata. Keys: {list(chunk.chunk_metadata.keys())}")

                if not document_id:
                    document_id = chunk.document_id
            else:
                logger.debug(f"Chunk {chunk_uuid}: No chunk or no metadata found")

        return {
            "document_name": doc_title,
            "document_id": str(document_id) if document_id else None,
            "pages": sorted(list(pages))[:5],  # First 5 pages only
            "chunk_uuids": [str(c) for c in chunk_uuids],
            "chunk_count": len(chunk_uuids)
        }

    def _create_source_from_chunks(
        self,
        doc_title: str,
        chunk_uuids: List[str],
        reranked_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a source dict from specific chunks, including page numbers from those chunks.
        (DEPRECATED: Use _create_source_from_chunks_cached with chunk cache for better performance)
        """
        pages = set()
        document_id = None

        # Get page numbers from these SPECIFIC chunks (not just first 10)
        for chunk_uuid in chunk_uuids[:20]:  # Limit to 20 chunks for performance
            try:
                chunk = self.storage.get_chunk_by_id(str(chunk_uuid))
                if chunk and chunk.chunk_metadata:
                    # PostgreSQL stores pages as a LIST in chunk_metadata["pages"]
                    chunk_pages = chunk.chunk_metadata.get("pages")
                    if chunk_pages:
                        if isinstance(chunk_pages, list):
                            pages.update(chunk_pages)
                        else:
                            pages.add(chunk_pages)  # Handle single page number
                    else:
                        # Fallback: check for singular "page_number" field
                        page_num = chunk.chunk_metadata.get("page_number")
                        if page_num is not None:
                            pages.add(page_num)

                    if not document_id:
                        document_id = chunk.document_id
            except Exception as e:
                logger.debug(f"Could not get chunk {chunk_uuid}: {e}")
                continue

        return {
            "document_name": doc_title,
            "document_id": str(document_id) if document_id else None,
            "pages": sorted(list(pages))[:5],  # First 5 pages only
            "chunk_uuids": [str(c) for c in chunk_uuids],
            "chunk_count": len(chunk_uuids)
        }

    def _apply_citation_mappings(
        self,
        answer_text: str,
        citation_mappings: Dict[str, str],
        citation_contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Replace citation markers with Harvard-style references using position-based approach.
        REMOVES markers that weren't mapped (below similarity threshold or failed processing).
        Replaces from END to START to avoid position shifts.

        Args:
            answer_text: Original answer with [Data: ...] markers
            citation_mappings: {unique_marker: harvard_citation}
            citation_contexts: All parsed citation contexts (to identify unmapped markers)

        Returns:
            Formatted answer with Harvard citations and unmapped markers removed
        """
        import re

        # Build list of all operations (replacements and removals)
        operations = []

        # Add replacements for mapped citations
        for unique_marker, harvard_citation in citation_mappings.items():
            match = re.match(r'(.+)__POS_(\d+)__', unique_marker)
            if match:
                original_marker = match.group(1)
                position = int(match.group(2))
                operations.append((position, original_marker, f' ({harvard_citation})', 'replace'))
            else:
                logger.warning(f"Could not parse unique marker: {unique_marker}")

        # Add removals for unmapped citations (below threshold or failed)
        mapped_markers = set(citation_mappings.keys())
        for citation_context in citation_contexts:
            unique_marker = citation_context["marker"]
            if unique_marker not in mapped_markers:
                # This citation was excluded (low similarity or error)
                original_marker = citation_context["original_marker"]
                position = citation_context["start"]
                operations.append((position, original_marker, '', 'remove'))
                logger.info(f"Removing low-quality citation marker at position {position}: {original_marker}")

        # Sort by position (descending) to process from end to start
        operations.sort(key=lambda x: x[0], reverse=True)

        result = answer_text
        for position, original_marker, replacement, operation_type in operations:
            marker_len = len(original_marker)
            if result[position:position+marker_len] == original_marker:
                # Replace/remove this specific occurrence
                result = result[:position] + replacement + result[position+marker_len:]
                if operation_type == 'replace':
                    logger.debug(f"Replaced '{original_marker}' at position {position} with '{replacement}'")
                else:
                    logger.debug(f"Removed '{original_marker}' at position {position}")
            else:
                logger.warning(f"Marker mismatch at position {position}: expected '{original_marker}', found '{result[position:position+marker_len]}'")

        return result

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
        citation_pattern = r'([^.!?]*\[Data: (?:Reports|Entities|Relationships) \([0-9, ]+\)\][^.!?]*[.!?])'
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
            marker_match = re.search(r'\[Data: (?:Reports|Entities|Relationships) \([0-9, ]+\)\]', full_context)
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

            # Get representative text for each source (fetch actual chunk content)
            source_texts = []
            for source in sources:
                # Fetch actual chunk content from first chunk UUID for semantic matching
                chunk_uuids = source.get("chunk_uuids", [])
                chunk_text = None

                if chunk_uuids:
                    try:
                        # Get first chunk's content as representative text
                        chunk = await asyncio.to_thread(
                            self.storage.get_chunk_by_id,
                            str(chunk_uuids[0])
                        )
                        if chunk and chunk.chunk_text:
                            # Use first 500 chars for efficient embedding
                            chunk_text = chunk.chunk_text[:500]
                    except Exception as e:
                        logger.debug(f"Could not fetch chunk {chunk_uuids[0]}: {e}")

                # Fallback to document name if chunk fetch fails
                if not chunk_text:
                    chunk_text = source.get("document_name", "Unknown")
                    logger.debug(f"Using document name as fallback for {chunk_text}")

                source_texts.append(chunk_text)

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
        """Build Harvard-style citation from document metadata using CitationFormatter."""
        from fileintel.citation import format_in_text_citation

        document_id = source.get("document_id")
        document_name = source.get("document_name") or source.get("title") or "Unknown"
        pages = source.get("pages", [])

        logger.debug(f"Building citation for: {document_name} (ID: {document_id}), pages: {pages}")

        # Fetch document metadata for proper Harvard citation
        document_metadata = {}

        if document_id:
            try:
                doc = self.storage.get_document(document_id)
                if doc and doc.document_metadata:
                    document_metadata = doc.document_metadata
                    logger.debug(f"Document metadata fetched: {document_metadata.keys()}")
            except Exception as e:
                logger.debug(f"Could not fetch document metadata for {document_id}: {e}")

        # Prepare chunk-like dict for CitationFormatter (same format as vector RAG)
        chunk_data = {
            "document_id": document_id,
            "document_metadata": document_metadata,
            "original_filename": document_name,
            "filename": document_name,
            "chunk_metadata": {
                "pages": pages  # CitationFormatter handles list of pages
            }
        }

        # Use the same citation formatter as vector RAG for consistency
        try:
            citation = format_in_text_citation(chunk_data)
            # Strip outer parentheses - they will be added by the caller
            # format_in_text_citation returns "(Author, Year)" but caller adds parentheses
            if citation.startswith("(") and citation.endswith(")"):
                citation = citation[1:-1]
            return citation
        except Exception as e:
            logger.warning(f"CitationFormatter failed: {e}, using fallback")
            # Fallback: simple format
            if pages:
                pages_list = sorted(list(pages))[:5]
                if len(pages_list) == 1:
                    page_str = f", p. {pages_list[0]}"
                else:
                    page_str = f", pp. {', '.join(map(str, pages_list))}"
                return f"{document_name}{page_str}"
            return document_name
