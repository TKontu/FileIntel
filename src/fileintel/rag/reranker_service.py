"""
Reranker Service for improving RAG result relevance.

Uses BAAI/bge-reranker-v2-m3 or similar models to rerank
initial retrieval results based on semantic relevance.
"""

from typing import List, Dict, Any, Optional
import logging
import time
from dataclasses import dataclass

from fileintel.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Container for reranked result with original data."""
    original_index: int  # Index in original results
    reranked_score: float  # New relevance score
    original_score: float  # Original similarity/relevance score
    chunk_data: Dict[str, Any]  # Original chunk/passage data
    score_delta: float  # Change from original score


class RerankerService:
    """
    Service for reranking RAG retrieval results.

    Supports:
    - Normal rerankers (bge-reranker-v2-m3, bge-reranker-large)
    - LLM-based rerankers (bge-reranker-v2-gemma)
    - Layerwise rerankers (bge-reranker-v2-minicpm-layerwise)

    Features:
    - Singleton model caching
    - Batch processing
    - GPU/CPU support
    - Score normalization
    - Performance tracking
    """

    _instance = None  # Singleton instance
    _model = None  # Cached model
    _model_name = None  # Track loaded model

    def __new__(cls, config: Settings):
        """Singleton pattern - one model instance per process."""
        if not config.rag.reranking.cache_model:
            # Don't use singleton if caching disabled
            return super().__new__(cls)

        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Settings):
        """Initialize reranker service."""
        self.config = config
        self.rerank_config = config.rag.reranking

        # Performance metrics
        self._total_reranks = 0
        self._total_latency_ms = 0
        self._cache_hits = 0

        # Load model if enabled
        if self.rerank_config.enabled and self._should_load_model():
            self._load_model()

    def _should_load_model(self) -> bool:
        """Check if model needs to be loaded."""
        # If not caching, always load
        if not self.rerank_config.cache_model:
            return True

        # If cached but different model, reload
        if self._model_name != self.rerank_config.model_name:
            return True

        # If model not loaded yet
        if self._model is None:
            return True

        return False

    def _load_model(self):
        """Load reranker model (lazy loading with caching)."""
        try:
            from FlagEmbedding import FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker

            model_name = self.rerank_config.model_name
            model_type = self.rerank_config.model_type
            use_fp16 = self.rerank_config.use_fp16

            logger.info(f"Loading reranker model: {model_name} (type: {model_type})")
            start_time = time.time()

            if model_type == "normal":
                self._model = FlagReranker(model_name, use_fp16=use_fp16)
            elif model_type == "llm":
                self._model = FlagLLMReranker(model_name, use_fp16=use_fp16)
            elif model_type == "layerwise":
                self._model = LayerWiseFlagLLMReranker(model_name, use_fp16=use_fp16)
            else:
                raise ValueError(f"Invalid model_type: {model_type}")

            self._model_name = model_name

            load_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Reranker model loaded in {load_time_ms}ms")

        except ImportError as e:
            logger.error(
                f"Failed to import FlagEmbedding. Install with: pip install FlagEmbedding\n"
                f"Error: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        passage_text_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages by relevance to query.

        Args:
            query: User query
            passages: List of passage dicts (must have 'content' or specified text key)
            top_k: Number of top results to return (default: config.final_top_k)
            passage_text_key: Key in passage dict containing text

        Returns:
            List of reranked passages (sorted by relevance, top_k entries)
        """
        if not self.rerank_config.enabled:
            logger.debug("Reranking disabled, returning original results")
            return passages[:top_k] if top_k else passages

        if not passages:
            return []

        top_k = top_k or self.rerank_config.final_top_k

        start_time = time.time()

        try:
            # Ensure model is loaded
            if self._model is None:
                self._load_model()

            # Extract text from passages
            passage_texts = []
            for p in passages:
                text = p.get(passage_text_key, "")
                if not text:
                    logger.warning(f"Passage missing '{passage_text_key}' field, skipping")
                    continue
                passage_texts.append(text)

            if not passage_texts:
                logger.warning("No valid passages to rerank")
                return passages[:top_k]

            # Prepare query-passage pairs
            pairs = [[query, text] for text in passage_texts]

            # Compute reranking scores
            logger.debug(f"Reranking {len(pairs)} passages for query: '{query[:50]}...'")

            if self.rerank_config.model_type == "layerwise":
                # Layerwise models need cutoff_layers parameter
                scores = self._model.compute_score(
                    pairs,
                    normalize=self.rerank_config.normalize_scores,
                    cutoff_layers=[28]  # Can make this configurable
                )
            else:
                # Normal and LLM rerankers
                scores = self._model.compute_score(
                    pairs,
                    normalize=self.rerank_config.normalize_scores
                )

            # Handle single score vs list of scores
            if not isinstance(scores, list):
                scores = [scores]

            # Create reranked results
            reranked = []
            for idx, (passage, score) in enumerate(zip(passages[:len(scores)], scores)):
                original_score = passage.get("similarity_score", 0.0) or passage.get("relevance_score", 0.0)

                reranked.append(RerankedResult(
                    original_index=idx,
                    reranked_score=float(score),
                    original_score=float(original_score),
                    chunk_data=passage,
                    score_delta=float(score) - float(original_score)
                ))

            # Sort by reranked score (descending)
            reranked.sort(key=lambda x: x.reranked_score, reverse=True)

            # Filter by minimum score threshold if set
            if self.rerank_config.min_score_threshold is not None:
                reranked = [
                    r for r in reranked
                    if r.reranked_score >= self.rerank_config.min_score_threshold
                ]

            # Take top_k
            reranked = reranked[:top_k]

            # Reconstruct passage dicts with reranked scores
            result = []
            for r in reranked:
                passage = r.chunk_data.copy()
                passage["reranked_score"] = r.reranked_score
                passage["original_score"] = r.original_score
                passage["original_rank"] = r.original_index + 1
                passage["reranked"] = True
                result.append(passage)

            # Performance tracking
            latency_ms = int((time.time() - start_time) * 1000)
            self._total_reranks += 1
            self._total_latency_ms += latency_ms

            avg_latency = self._total_latency_ms / self._total_reranks

            logger.info(
                f"Reranked {len(passages)} → {len(result)} passages "
                f"(latency: {latency_ms}ms, avg: {avg_latency:.1f}ms)"
            )

            # Log significant reorderings
            self._log_reordering_changes(reranked)

            return result

        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original results")
            return passages[:top_k]

    def _log_reordering_changes(self, reranked: List[RerankedResult]):
        """Log significant changes in result ordering."""
        if not reranked:
            return

        # Find biggest jumps in ranking
        max_jump = 0
        max_jump_idx = 0

        for i, result in enumerate(reranked):
            jump = result.original_index - i
            if abs(jump) > abs(max_jump):
                max_jump = jump
                max_jump_idx = i

        if abs(max_jump) > 3:  # Log if moved more than 3 positions
            logger.info(
                f"Significant reranking: Result originally at position {reranked[max_jump_idx].original_index + 1} "
                f"moved to position {max_jump_idx + 1} "
                f"(score: {reranked[max_jump_idx].original_score:.3f} → {reranked[max_jump_idx].reranked_score:.3f})"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get reranker performance statistics."""
        avg_latency = (
            self._total_latency_ms / self._total_reranks
            if self._total_reranks > 0
            else 0
        )

        return {
            "enabled": self.rerank_config.enabled,
            "model_name": self._model_name,
            "model_loaded": self._model is not None,
            "total_reranks": self._total_reranks,
            "total_latency_ms": self._total_latency_ms,
            "average_latency_ms": avg_latency,
            "cache_hits": self._cache_hits,
        }
