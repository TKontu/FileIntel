from typing import Any, Dict
from fileintel.core.config import Settings
import logging

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
        self.config = config

        # Set configurable keywords with fallbacks to config or defaults
        self.graph_keywords = graph_keywords or getattr(
            config.rag, "graph_keywords", DEFAULT_GRAPH_KEYWORDS
        )
        self.vector_keywords = vector_keywords or getattr(
            config.rag, "vector_keywords", DEFAULT_VECTOR_KEYWORDS
        )
        self.hybrid_keywords = hybrid_keywords or getattr(
            config.rag, "hybrid_keywords", DEFAULT_HYBRID_KEYWORDS
        )

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifies the query using keyword-based classification.
        Fast, reliable, and covers the main use cases effectively.
        """
        return self._keyword_classify(query)

    def _keyword_classify(self, query: str) -> Dict[str, Any]:
        """Keyword-based classification using configurable keywords."""
        query_lower = query.lower()

        # Check for hybrid patterns first (most specific)
        if any(keyword in query_lower for keyword in self.hybrid_keywords):
            return {
                "type": "HYBRID",
                "confidence": HYBRID_KEYWORD_CONFIDENCE,
                "reasoning": "Matched HYBRID keywords indicating complex multi-part query.",
            }

        # Check for graph patterns (relationship/entity analysis)
        if any(keyword in query_lower for keyword in self.graph_keywords):
            return {
                "type": "GRAPH",
                "confidence": KEYWORD_MATCH_CONFIDENCE,
                "reasoning": "Matched GRAPH keywords indicating relationship/entity analysis.",
            }

        # Check for vector patterns (specific search/facts)
        if any(keyword in query_lower for keyword in self.vector_keywords):
            return {
                "type": "VECTOR",
                "confidence": KEYWORD_MATCH_CONFIDENCE,
                "reasoning": "Matched VECTOR keywords indicating specific search/factual query.",
            }

        # Default to vector search for general queries
        return {
            "type": "VECTOR",
            "confidence": DEFAULT_CONFIDENCE,
            "reasoning": "No specific keywords matched, defaulting to VECTOR search.",
        }
