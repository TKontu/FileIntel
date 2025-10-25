from typing import Any, Dict, Optional
from fileintel.core.config import Settings
import logging
import hashlib
import time
import json

logger = logging.getLogger(__name__)

# Constants for query classification
DEFAULT_CONFIDENCE = 0.5
KEYWORD_MATCH_CONFIDENCE = 0.8
HYBRID_KEYWORD_CONFIDENCE = 0.7

# Default keyword configurations
DEFAULT_GRAPH_KEYWORDS = [
    "entity",
    "entities",
    "relationship",
    "relationships",
    "connection",
    "connections",
    "network",
    "graph",
    "connected",
    "relates",
    "related",
    "link",
    "links",
    "overview",
    "summary",
    "summarize",
    "theme",
    "themes",
    "thematic",
    "big picture",
    "global",
    "overall",
    "compare",
    "comparison",
    "contrast",
]

DEFAULT_VECTOR_KEYWORDS = [
    "specific",
    "detail",
    "details",
    "exact",
    "precise",
    "find",
    "search",
    "look for",
    "locate",
    "where",
    "when",
    "how",
    "what",
    "document",
    "paragraph",
    "sentence",
    "quote",
    "citation",
    "reference",
    "fact",
    "facts",
]

DEFAULT_HYBRID_KEYWORDS = [
    "analyze and find",
    "find and analyze",
    "summarize and show",
    "show and summarize",
    "both",
    "also",
    "additionally",
    "furthermore",
    "as well as",
    "and then",
    "complex",
    "multi-part",
    "comprehensive",
    "detailed analysis",
]


class QueryClassifier:
    def __init__(
        self,
        config: Settings,
        graph_keywords: list = None,
        vector_keywords: list = None,
        hybrid_keywords: list = None,
    ):
        """
        Initialize QueryClassifier with configurable keywords.

        Keyword priority (first non-None/non-empty value used):
        1. Explicit parameter passed to __init__
        2. Keywords from config.rag.graph_keywords (YAML config)
        3. DEFAULT_GRAPH_KEYWORDS (hardcoded fallback)

        Args:
            config: Settings object with RAG configuration
            graph_keywords: Optional list of keywords to override config
            vector_keywords: Optional list of keywords to override config
            hybrid_keywords: Optional list of keywords to override config
        """
        self.config = config

        # Set configurable keywords with fallbacks to config or defaults
        # The 'or' operator handles None/empty lists from config
        self.graph_keywords = graph_keywords or getattr(
            config.rag, "graph_keywords", DEFAULT_GRAPH_KEYWORDS
        ) or DEFAULT_GRAPH_KEYWORDS
        self.vector_keywords = vector_keywords or getattr(
            config.rag, "vector_keywords", DEFAULT_VECTOR_KEYWORDS
        ) or DEFAULT_VECTOR_KEYWORDS
        self.hybrid_keywords = hybrid_keywords or getattr(
            config.rag, "hybrid_keywords", DEFAULT_HYBRID_KEYWORDS
        ) or DEFAULT_HYBRID_KEYWORDS

        # Classification cache: {query_hash: (result, timestamp)}
        self._classification_cache: Dict[str, tuple] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent cache keys.

        - Lowercase
        - Strip whitespace
        - Remove extra spaces

        Args:
            query: Raw user query

        Returns:
            Normalized query string
        """
        return " ".join(query.lower().strip().split())

    def _get_cache_key(self, query: str) -> str:
        """
        Generate cache key from query using SHA-256 hash.

        Args:
            query: Normalized query string

        Returns:
            Hex string of query hash
        """
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _cache_get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get classification result from cache if not expired.

        Args:
            query: User query

        Returns:
            Cached classification result or None if miss/expired
        """
        if not self.config.rag.classification_cache_enabled:
            return None

        cache_key = self._get_cache_key(query)

        if cache_key in self._classification_cache:
            result, timestamp = self._classification_cache[cache_key]
            age = time.time() - timestamp

            # Check if still valid (within TTL)
            if age < self.config.rag.classification_cache_ttl:
                self._cache_hits += 1
                logger.debug(f"Cache HIT for query (age: {age:.1f}s)")
                return result
            else:
                # Expired, remove from cache
                del self._classification_cache[cache_key]
                logger.debug(f"Cache EXPIRED for query (age: {age:.1f}s)")

        self._cache_misses += 1
        logger.debug("Cache MISS for query")
        return None

    def _cache_set(self, query: str, result: Dict[str, Any]) -> None:
        """
        Store classification result in cache.

        Args:
            query: User query
            result: Classification result to cache
        """
        if not self.config.rag.classification_cache_enabled:
            return

        cache_key = self._get_cache_key(query)
        self._classification_cache[cache_key] = (result, time.time())

        # Log cache stats periodically (every 100 operations)
        total_ops = self._cache_hits + self._cache_misses
        if total_ops > 0 and total_ops % 100 == 0:
            hit_rate = (self._cache_hits / total_ops) * 100
            logger.info(
                f"Classification cache stats: {self._cache_hits} hits, "
                f"{self._cache_misses} misses, hit rate: {hit_rate:.1f}%, "
                f"size: {len(self._classification_cache)} entries"
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0.0

        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate_percent": hit_rate,
            "cache_size": len(self._classification_cache),
            "enabled": self.config.rag.classification_cache_enabled,
            "ttl_seconds": self.config.rag.classification_cache_ttl,
        }

    def _llm_classify(self, query: str) -> Dict[str, Any]:
        """
        Classify query using LLM semantic understanding.

        Uses structured prompt to analyze query intent and select optimal RAG strategy.
        Falls back to keyword classification on errors.

        Args:
            query: User query to classify

        Returns:
            Classification result dict with type, confidence, reasoning

        Raises:
            Exception: On LLM errors (caller should catch and fallback)
        """
        start_time = time.time()

        try:
            # Import LLM provider
            from fileintel.llm_integration.unified_provider import UnifiedLLMProvider

            # Create classification prompt
            prompt = f"""Analyze this user query and determine the best RAG search strategy.

QUERY: "{query}"

AVAILABLE STRATEGIES:

1. VECTOR - Semantic similarity search for factual information
   Best for: Definitions, specific facts, "what is X", direct information lookup
   Examples:
   - "What is quantum computing?"
   - "Define photosynthesis"
   - "Tell me about the history of Rome"
   - "List the features of Python"

2. GRAPH - Relationship and entity analysis using knowledge graphs
   Best for: Connections, relationships, comparisons, network analysis
   Examples:
   - "How are X and Y related?"
   - "Show connections between entities"
   - "Compare X and Y"
   - "What relationships exist in this network?"

3. HYBRID - Combined approach using both methods
   Best for: Complex queries needing both factual details AND relationships
   Examples:
   - "Compare X and Y and provide detailed information about each"
   - "Analyze the relationships and give me specific facts"
   - "What are the connections and what do the documents say?"

INSTRUCTIONS:
- Analyze the query intent and choose the SINGLE best strategy
- Consider what the user is actually asking for, not just keywords
- Return ONLY valid JSON, no explanations before or after

Respond in this exact JSON format:
{{
  "type": "VECTOR|GRAPH|HYBRID",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of why this strategy was chosen"
}}"""

            # Call LLM with timeout
            llm_provider = UnifiedLLMProvider(self.config)

            # Use timeout from config
            timeout = self.config.rag.classification_timeout_seconds

            response = llm_provider.generate_response(
                prompt=prompt,
                model=self.config.rag.classification_model,
                max_tokens=self.config.rag.classification_max_tokens,
                temperature=self.config.rag.classification_temperature,
                timeout=timeout,
            )

            # Parse JSON response
            result_text = response.content.strip()

            # Try to extract JSON if response has extra text
            if "{" in result_text and "}" in result_text:
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                result_text = result_text[json_start:json_end]

            result = json.loads(result_text)

            # Validate response structure
            if "type" not in result or "confidence" not in result:
                raise ValueError("LLM response missing required fields (type, confidence)")

            # Normalize type to uppercase
            result["type"] = result["type"].upper()

            # Validate type
            valid_types = {"VECTOR", "GRAPH", "HYBRID"}
            if result["type"] not in valid_types:
                raise ValueError(f"Invalid type '{result['type']}', must be one of {valid_types}")

            # Ensure confidence is float between 0 and 1
            result["confidence"] = float(result["confidence"])
            if not (0.0 <= result["confidence"] <= 1.0):
                result["confidence"] = 0.8  # Default to high confidence

            # Add metadata
            latency_ms = int((time.time() - start_time) * 1000)
            result["method"] = "llm"
            result["latency_ms"] = latency_ms

            logger.info(
                f"LLM classification: {result['type']} "
                f"(confidence: {result['confidence']:.2f}, "
                f"latency: {latency_ms}ms)"
            )

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            raise
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            raise

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify query using configured method with intelligent fallback.

        Classification flow:
        1. Check cache → return if hit
        2. Try primary method (llm/keyword/hybrid)
        3. Fallback to keyword if LLM fails (hybrid mode)
        4. Cache result
        5. Return classification

        Args:
            query: User query to classify

        Returns:
            Classification dict with type, confidence, reasoning, method, latency_ms
        """
        # Check cache first
        cached_result = self._cache_get(query)
        if cached_result is not None:
            cached_result["cached"] = True
            return cached_result

        # Get classification method from config
        method = self.config.rag.classification_method.lower()

        result = None
        fallback_used = False

        try:
            if method == "llm":
                # LLM only - no fallback
                result = self._llm_classify(query)

            elif method == "keyword":
                # Keyword only - fast and deterministic
                result = self._keyword_classify(query)

            elif method == "hybrid":
                # Try LLM first, fallback to keywords
                try:
                    result = self._llm_classify(query)
                except Exception as e:
                    logger.info(f"LLM classification failed, falling back to keywords: {e}")
                    result = self._keyword_classify(query)
                    result["fallback_reason"] = str(e)
                    fallback_used = True

            else:
                logger.warning(
                    f"Invalid classification method '{method}', using keyword fallback"
                )
                result = self._keyword_classify(query)

        except Exception as e:
            # Final safety fallback
            logger.error(f"Classification failed completely: {e}, using keyword fallback")
            result = self._keyword_classify(query)
            result["fallback_reason"] = str(e)
            fallback_used = True

        # Add metadata
        result["cached"] = False
        if fallback_used:
            result["fallback_used"] = True

        # Cache the result
        self._cache_set(query, result)

        return result

    def _keyword_classify(self, query: str) -> Dict[str, Any]:
        """
        Keyword-based classification using configurable keywords.

        Fast, deterministic, zero-cost classification based on keyword matching.

        Args:
            query: User query to classify

        Returns:
            Classification dict with type, confidence, reasoning, method
        """
        start_time = time.time()
        query_lower = query.lower()

        # Check for hybrid patterns first (most specific)
        if any(keyword in query_lower for keyword in self.hybrid_keywords):
            result = {
                "type": "HYBRID",
                "confidence": HYBRID_KEYWORD_CONFIDENCE,
                "reasoning": "Matched HYBRID keywords indicating complex multi-part query.",
            }
        # Check for graph patterns (relationship/entity analysis)
        elif any(keyword in query_lower for keyword in self.graph_keywords):
            result = {
                "type": "GRAPH",
                "confidence": KEYWORD_MATCH_CONFIDENCE,
                "reasoning": "Matched GRAPH keywords indicating relationship/entity analysis.",
            }
        # Check for vector patterns (specific search/facts)
        elif any(keyword in query_lower for keyword in self.vector_keywords):
            result = {
                "type": "VECTOR",
                "confidence": KEYWORD_MATCH_CONFIDENCE,
                "reasoning": "Matched VECTOR keywords indicating specific search/factual query.",
            }
        # Default to vector search for general queries
        else:
            result = {
                "type": "VECTOR",
                "confidence": DEFAULT_CONFIDENCE,
                "reasoning": "No specific keywords matched, defaulting to VECTOR search.",
            }

        # Add metadata
        latency_ms = int((time.time() - start_time) * 1000)
        result["method"] = "keyword"
        result["latency_ms"] = latency_ms

        logger.debug(
            f"Keyword classification: {result['type']} "
            f"(confidence: {result['confidence']:.2f}, "
            f"latency: {latency_ms}ms)"
        )

        return result
