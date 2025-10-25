"""
Reranker Service for improving RAG result relevance.

Uses remote vLLM/OpenAI-compatible API to rerank initial retrieval
results based on semantic relevance using models like BAAI/bge-reranker-v2-m3.
"""

from typing import List, Dict, Any, Optional
import logging
import time
import requests
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
    Service for reranking RAG retrieval results using remote API.

    Connects to vLLM or OpenAI-compatible reranking API to rerank
    initial retrieval results based on semantic relevance.

    Features:
    - HTTP API integration with vLLM/OpenAI servers
    - Batch processing
    - Score normalization
    - Performance tracking
    - Graceful error handling
    """

    def __init__(self, config: Settings):
        """Initialize reranker service with API settings."""
        self.config = config
        self.rerank_config = config.rag.reranking

        # API configuration
        self.base_url = self.rerank_config.base_url.rstrip('/')
        self.api_key = self.rerank_config.api_key
        self.timeout = self.rerank_config.timeout
        self.model_name = self.rerank_config.model_name

        # Performance metrics
        self._total_reranks = 0
        self._total_latency_ms = 0
        self._api_errors = 0

        logger.info(
            f"RerankerService initialized with API: {self.base_url}, "
            f"model: {self.model_name}"
        )

    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        passage_text_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages by relevance to query using API.

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
            # Extract text from passages
            passage_texts = []
            valid_passages = []
            for p in passages:
                text = p.get(passage_text_key, "")
                if not text:
                    logger.warning(f"Passage missing '{passage_text_key}' field, skipping")
                    continue
                passage_texts.append(text)
                valid_passages.append(p)

            if not passage_texts:
                logger.warning("No valid passages to rerank")
                return passages[:top_k]

            logger.debug(f"Reranking {len(passage_texts)} passages for query: '{query[:50]}...'")

            # Call reranking API
            scores = self._call_rerank_api(query, passage_texts)

            # Validate scores match passages
            if len(scores) != len(valid_passages):
                logger.warning(
                    f"API returned {len(scores)} scores for {len(valid_passages)} documents. "
                    f"Using min length."
                )
                # Truncate to minimum length to avoid index errors
                min_len = min(len(scores), len(valid_passages))
                scores = scores[:min_len]
                valid_passages = valid_passages[:min_len]

            # Create reranked results
            reranked = []
            for idx, (passage, score) in enumerate(zip(valid_passages, scores)):
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
                f"Reranked {len(valid_passages)} → {len(result)} passages "
                f"(latency: {latency_ms}ms, avg: {avg_latency:.1f}ms)"
            )

            # Log significant reorderings
            self._log_reordering_changes(reranked)

            return result

        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original results")
            self._api_errors += 1
            return passages[:top_k]

    def _call_rerank_api(self, query: str, documents: List[str]) -> List[float]:
        """
        Call the reranking API to get relevance scores.

        Args:
            query: Search query
            documents: List of document texts to rerank

        Returns:
            List of relevance scores (same order as documents)

        Raises:
            Exception: If API call fails
        """
        # Validate inputs
        if not documents:
            logger.warning("Empty documents list passed to reranker API")
            return []

        if not query or not query.strip():
            logger.warning("Empty query passed to reranker API")
            return [0.0] * len(documents)  # Return zero scores

        url = f"{self.base_url}/rerank"

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "return_documents": False,  # We already have the documents
            "top_n": len(documents),  # Return all scores
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            response.raise_for_status()

            data = response.json()

            # Parse response - handle different API formats
            # Standard format: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
            if "results" in data:
                # Sort by index to ensure correct order
                results = sorted(data["results"], key=lambda x: x.get("index", 0))
                scores = [r.get("relevance_score", r.get("score", 0.0)) for r in results]
            # Alternative format: {"scores": [0.95, 0.82, ...]}
            elif "scores" in data:
                scores = data["scores"]
            else:
                raise ValueError(f"Unexpected API response format: {data.keys()}")

            # Note: normalize_scores config is deprecated for API mode
            # vLLM API returns already-normalized scores, so we don't apply
            # sigmoid normalization here (would double-normalize)

            if not scores:
                logger.warning("API returned empty scores list")
                return []

            return scores

        except requests.exceptions.Timeout:
            logger.error(f"Reranking API timeout after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Reranking API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Reranking API error: {e}")
            raise

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
            "api_url": self.base_url,
            "model_name": self.model_name,
            "total_reranks": self._total_reranks,
            "total_latency_ms": self._total_latency_ms,
            "average_latency_ms": avg_latency,
            "api_errors": self._api_errors,
        }
