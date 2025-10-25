from typing import Any, Dict, List, Tuple, Optional
from fileintel.rag.vector_rag.services.vector_rag_service import VectorRAGService
from fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService
from fileintel.rag.query_classifier import QueryClassifier
from fileintel.rag.reranker_service import RerankerService
from fileintel.core.config import RAGSettings, Settings
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
        settings: Optional[Settings] = None,
    ):
        self.vector_rag_service = vector_rag_service
        self.graphrag_service = graphrag_service
        self.query_classifier = query_classifier
        self.config = config
        self._routing_explanation = "No routing decision made yet."

        # Initialize reranker service if enabled for hybrid results
        self.reranker = None
        if settings and settings.rag.reranking.enabled and settings.rag.reranking.rerank_hybrid_results:
            try:
                self.reranker = RerankerService(settings)
                logger.info("RerankerService initialized for hybrid query orchestration")
            except Exception as e:
                logger.warning(f"Failed to initialize RerankerService: {e}. Continuing without reranking.")

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

        # Enhanced routing explanation with two-tier awareness
        explanation_parts = [
            f"Query classified as {classification['type']} "
            f"(confidence: {classification['confidence']:.2f}). "
            f"Reasoning: {classification['reasoning']}"
        ]

        # Add two-tier chunking context if enabled
        if getattr(self.config, 'enable_two_tier_chunking', False):
            if query_type == QueryType.VECTOR:
                explanation_parts.append("Using optimized 300-token vector chunks for semantic retrieval.")
            elif query_type == QueryType.GRAPH:
                explanation_parts.append("Using deduplicated 1500-token graph chunks for relationship extraction.")
            elif query_type == QueryType.HYBRID:
                explanation_parts.append("Using both vector chunks (retrieval) and graph chunks (analysis) for comprehensive results.")

        self._routing_explanation = " ".join(explanation_parts)
        return query_type

    def get_routing_explanation(self) -> str:
        return self._routing_explanation

    def get_chunk_type_for_query(self, query_type: QueryType) -> str:
        """Determine which chunk type to use based on query type and two-tier settings."""
        if not getattr(self.config, 'enable_two_tier_chunking', False):
            return 'traditional'

        chunk_type_mapping = {
            QueryType.VECTOR: 'vector',
            QueryType.GRAPH: 'graph',
            QueryType.HYBRID: 'both'
        }

        return chunk_type_mapping.get(query_type, 'vector')

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

        # Rerank combined sources if reranker is enabled
        if self.reranker and combined_sources:
            try:
                # Convert sources to reranker format
                passages = []
                for source in combined_sources:
                    text = source.get("text", "")
                    if text:
                        passages.append({
                            "content": text,
                            "similarity_score": source.get("rank_score", 0.0),
                            **source  # Include all other fields
                        })

                if passages:
                    # Get top sources limit
                    max_sources = getattr(self.config, "max_hybrid_sources", 8)

                    # Rerank passages
                    reranked_passages = self.reranker.rerank(
                        query=query,
                        passages=passages,
                        top_k=max_sources,
                        passage_text_key="content"
                    )

                    # Convert back to source format
                    combined_sources = []
                    for passage in reranked_passages:
                        source = passage.copy()
                        source["text"] = source.pop("content")
                        source["rank_score"] = source["reranked_score"]
                        source["original_rank_score"] = source.get("original_score", 0.0)
                        combined_sources.append(source)

                    logger.info(f"Reranked {len(passages)} hybrid sources → {len(combined_sources)} final sources")
                    return combined_sources

            except Exception as e:
                logger.error(f"Hybrid source reranking failed: {e}. Using fallback ranking.")
                # Fall through to original sorting logic

        # Sort by rank score (highest first) - fallback if no reranking
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
