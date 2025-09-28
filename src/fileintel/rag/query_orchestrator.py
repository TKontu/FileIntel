from typing import Any, Dict, List, Tuple, Optional
from fileintel.rag.vector_rag.services.vector_rag_service import VectorRAGService
from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService
from fileintel.rag.query_classifier import QueryClassifier
from fileintel.core.config import RAGSettings
from fileintel.rag.models import DirectQueryResponse
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class QueryOrchestrator:
    def __init__(
        self,
        vector_rag_service: VectorRAGService,
        graphrag_service: GraphRAGService,
        query_classifier: QueryClassifier,
        config: RAGSettings,
    ):
        self.vector_rag_service = vector_rag_service
        self.graphrag_service = graphrag_service
        self.query_classifier = query_classifier
        self.config = config
        self._routing_explanation = "No routing decision made yet."

    async def route_query(
        self, query: str, collection_id: str, routing_override: str = "auto"
    ) -> DirectQueryResponse:
        if routing_override != "auto":
            query_type = QueryType(routing_override)
            self._routing_explanation = (
                f"Routing overridden to '{routing_override}' by user."
            )
        else:
            query_type = self.classify_query_type(query)

        if query_type == QueryType.VECTOR:
            response = self.vector_rag_service.query(query, collection_id)
        elif query_type == QueryType.GRAPH:
            # GraphRAG service must remain async due to protected GraphRAG library
            response = await self.graphrag_service.query(query, collection_id)
        elif query_type == QueryType.HYBRID:
            # Execute both vector and graph searches, then combine results
            response = await self._execute_hybrid_query(query, collection_id)
        else:
            raise ValueError(f"Unknown query type: {query_type}")

        return DirectQueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            query_type=query_type.value,
            routing_explanation=self.get_routing_explanation(),
        )

    def classify_query_type(self, query: str) -> QueryType:
        classification = self.query_classifier.classify(query)
        query_type = QueryType(classification["type"].lower())
        self._routing_explanation = (
            f"Query classified as {classification['type']} "
            f"(confidence: {classification['confidence']:.2f}). "
            f"Reasoning: {classification['reasoning']}"
        )
        return query_type

    def get_routing_explanation(self) -> str:
        return self._routing_explanation

    async def _execute_hybrid_query(
        self, query: str, collection_id: str
    ) -> Dict[str, Any]:
        """
        Execute hybrid query combining vector and graph search results.

        Args:
            query: The user's question
            collection_id: Collection to search in

        Returns:
            Combined response with ranked sources from both search methods
        """
        try:
            # Execute vector and graph searches in parallel (where possible)
            vector_response = self.vector_rag_service.query(query, collection_id)
            graph_response = await self.graphrag_service.query(query, collection_id)

            # Combine and rank sources from both methods
            combined_sources = self._combine_and_rank_sources(
                vector_response.get("sources", []),
                graph_response.get("sources", []),
                query,
            )

            # Generate hybrid answer using best sources from both methods
            hybrid_answer = self._generate_hybrid_answer(
                query,
                vector_response.get("answer", ""),
                graph_response.get("answer", ""),
                combined_sources,
            )

            self._routing_explanation += (
                f" Hybrid search executed both vector and graph methods. "
                f"Vector found {len(vector_response.get('sources', []))} sources, "
                f"Graph found {len(graph_response.get('sources', []))} sources. "
                f"Combined and ranked {len(combined_sources)} total sources."
            )

            return {
                "answer": hybrid_answer,
                "sources": combined_sources,
                "confidence": max(
                    vector_response.get("confidence", 0.0),
                    graph_response.get("confidence", 0.0),
                ),
                "metadata": {
                    "vector_sources": len(vector_response.get("sources", [])),
                    "graph_sources": len(graph_response.get("sources", [])),
                    "combined_sources": len(combined_sources),
                },
            }

        except Exception as e:
            logger.error(f"Error in hybrid query execution: {e}")
            # Fallback to vector search if hybrid fails
            self._routing_explanation += (
                f" Hybrid search failed ({str(e)}), falling back to vector search."
            )
            return self.vector_rag_service.query(query, collection_id)

    def _combine_and_rank_sources(
        self, vector_sources: List[Dict], graph_sources: List[Dict], query: str
    ) -> List[Dict]:
        """
        Combine and rank sources from vector and graph searches.

        Args:
            vector_sources: Sources from vector similarity search
            graph_sources: Sources from graph-based search
            query: Original query for relevance scoring

        Returns:
            Combined list of sources ranked by relevance
        """
        combined_sources = []
        seen_chunks = set()  # Track chunks to avoid duplicates

        # Process vector sources (add source type metadata)
        for source in vector_sources:
            chunk_id = source.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                source_with_type = source.copy()
                source_with_type["search_method"] = "vector"
                source_with_type["rank_score"] = source.get("similarity_score", 0.0)
                combined_sources.append(source_with_type)
                seen_chunks.add(chunk_id)

        # Process graph sources (add source type metadata)
        for source in graph_sources:
            chunk_id = source.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                source_with_type = source.copy()
                source_with_type["search_method"] = "graph"
                # Graph sources may not have similarity scores, estimate based on position
                source_with_type["rank_score"] = source.get(
                    "relevance_score", 0.7
                )  # Default relevance
                combined_sources.append(source_with_type)
                seen_chunks.add(chunk_id)
            elif chunk_id in seen_chunks:
                # If same chunk found by both methods, boost its score
                for existing_source in combined_sources:
                    if existing_source.get("chunk_id") == chunk_id:
                        existing_source[
                            "search_method"
                        ] = "vector+graph"  # Mark as found by both
                        existing_source["rank_score"] = min(
                            1.0, existing_source["rank_score"] + 0.2
                        )  # Boost score
                        break

        # Sort by rank score (highest first)
        combined_sources.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)

        # Limit to top sources based on configuration
        max_sources = getattr(self.config, "max_hybrid_sources", 8)
        return combined_sources[:max_sources]

    def _generate_hybrid_answer(
        self,
        query: str,
        vector_answer: str,
        graph_answer: str,
        combined_sources: List[Dict],
    ) -> str:
        """
        Generate a hybrid answer combining insights from vector and graph searches.

        Args:
            query: Original query
            vector_answer: Answer from vector search
            graph_answer: Answer from graph search
            combined_sources: Combined and ranked sources

        Returns:
            Synthesized answer incorporating both search methods
        """
        # Simple synthesis logic - in production this could use LLM for better combination
        if not vector_answer and not graph_answer:
            return "No relevant information found to answer the query."

        if not vector_answer:
            return graph_answer

        if not graph_answer:
            return vector_answer

        # Both answers exist - try to synthesize
        if len(vector_answer) > len(graph_answer):
            primary_answer = vector_answer
            secondary_insight = graph_answer
        else:
            primary_answer = graph_answer
            secondary_insight = vector_answer

        # If answers are very similar, just return the longer one
        common_words = set(vector_answer.lower().split()) & set(
            graph_answer.lower().split()
        )
        if (
            len(common_words)
            > min(len(vector_answer.split()), len(graph_answer.split())) * 0.6
        ):
            return primary_answer

        # Otherwise combine both perspectives
        return f"{primary_answer}\n\nAdditional context: {secondary_insight}"
