"""
Hybrid query tasks for combining Vector RAG and GraphRAG results.

This module provides tasks for executing hybrid queries that combine
results from both Vector RAG (semantic search) and GraphRAG (relationship analysis).
"""

import logging
from typing import Dict, Any
from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="fileintel.tasks.combine_hybrid_results")
def combine_hybrid_results(
    self,
    vector_result: Dict[str, Any],
    graph_result: Dict[str, Any],
    query: str,
    collection_id: str,
    answer_format: str = "default"
) -> Dict[str, Any]:
    """
    Combine results from Vector RAG and GraphRAG for hybrid queries.

    Strategy:
    1. Extract answers and sources from both results
    2. Merge sources (deduplicate by document ID)
    3. Use LLM to synthesize combined answer
    4. Return unified response with metadata

    Args:
        vector_result: Result dict from Vector RAG query
        graph_result: Result dict from GraphRAG query
        query: Original user query
        collection_id: Collection ID
        answer_format: Answer format specification

    Returns:
        Combined result dict with unified answer and merged sources

    Example:
        >>> vector_res = {"answer": "X is...", "sources": [...]}
        >>> graph_res = {"answer": "X relates to Y...", "sources": [...]}
        >>> combined = combine_hybrid_results(vector_res, graph_res, "What is X?", "coll-1")
        >>> combined["answer"]  # Synthesized answer combining both
    """
    try:
        from fileintel.core.config import get_config
        from fileintel.llm_integration.unified_provider import UnifiedLLMProvider
        from fileintel.storage.postgresql_storage import PostgreSQLStorage

        logger.info(f"Combining hybrid results for query: '{query}' (collection: {collection_id})")

        # Extract answers from both sources
        vector_answer = vector_result.get("answer", "")
        graph_answer = graph_result.get("answer", "")

        if not vector_answer and not graph_answer:
            logger.warning("Both vector and graph results are empty")
            return {
                "answer": "No results found from either search method.",
                "sources": [],
                "query_type": "hybrid",
                "error": "Empty results from both vector and graph searches"
            }

        # Extract and merge sources
        vector_sources = vector_result.get("sources", [])
        graph_sources = graph_result.get("sources", [])

        # Deduplicate sources by document ID or filename
        seen_docs = set()
        merged_sources = []

        for source in vector_sources + graph_sources:
            # Try multiple ID fields for deduplication
            doc_id = (
                source.get("document_id") or
                source.get("filename") or
                source.get("citation") or
                source.get("text", "")[:50]  # Fallback to text snippet
            )

            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                merged_sources.append(source)
            elif not doc_id:
                # Include sources without identifiable IDs (shouldn't happen, but be safe)
                merged_sources.append(source)

        logger.info(
            f"Source merging: vector={len(vector_sources)}, "
            f"graph={len(graph_sources)}, merged={len(merged_sources)}"
        )

        # Synthesize combined answer using LLM
        config = get_config()
        storage = PostgreSQLStorage(config)
        llm_provider = UnifiedLLMProvider(config, storage)

        # Build synthesis prompt
        synthesis_prompt = f"""You are synthesizing results from two complementary search methods for the same query.

QUERY: "{query}"

VECTOR SEARCH RESULT (factual/semantic similarity search):
{vector_answer if vector_answer else "[No vector search results]"}

GRAPH SEARCH RESULT (relationship/entity/knowledge graph analysis):
{graph_answer if graph_answer else "[No graph search results]"}

INSTRUCTIONS:
1. Synthesize a comprehensive answer that combines insights from BOTH searches
2. Prioritize factual accuracy from vector search for specific details
3. Add relationship context and broader insights from graph search
4. Preserve ALL citations from both sources exactly as they appear
5. Remove redundant or duplicate information
6. Present a coherent, unified response that leverages the strengths of both methods
7. If one search returned no results, rely on the other while noting the limitation

CRITICAL: You MUST preserve all citations exactly as they appear in the source answers.

Synthesized Answer:"""

        logger.debug(f"Synthesizing hybrid answer with LLM (format: {answer_format})")

        try:
            response = llm_provider.generate_response(
                prompt=synthesis_prompt,
                max_tokens=800,
                temperature=0.2  # Low temperature for consistent synthesis
            )

            synthesized_answer = response.content if hasattr(response, 'content') else str(response)

        except Exception as llm_error:
            logger.error(f"LLM synthesis failed: {llm_error}")
            # Fallback: concatenate both answers with clear separation
            synthesized_answer = ""
            if vector_answer:
                synthesized_answer += f"**Vector Search Results:**\n{vector_answer}\n\n"
            if graph_answer:
                synthesized_answer += f"**Graph Search Results:**\n{graph_answer}"

            if not synthesized_answer:
                synthesized_answer = "Unable to synthesize results from both search methods."

        # Build combined result
        combined_result = {
            "answer": synthesized_answer,
            "sources": merged_sources[:15],  # Limit to top 15 sources to avoid overwhelming
            "query_type": "hybrid",
            "metadata": {
                "vector_answer_length": len(vector_answer),
                "graph_answer_length": len(graph_answer),
                "source_count": {
                    "vector": len(vector_sources),
                    "graph": len(graph_sources),
                    "merged": len(merged_sources),
                    "returned": min(len(merged_sources), 15)
                },
                "synthesis_method": "llm"
            }
        }

        logger.info(
            f"Hybrid synthesis complete: {len(merged_sources)} sources merged, "
            f"answer length: {len(synthesized_answer)} chars"
        )

        return combined_result

    except Exception as e:
        logger.error(f"Hybrid result combination failed: {e}", exc_info=True)

        # Fallback: return the result that has content, preferring graph (usually more comprehensive)
        fallback_answer = ""
        fallback_sources = []

        if graph_result and graph_result.get("answer"):
            fallback_answer = graph_result.get("answer", "")
            fallback_sources = graph_result.get("sources", [])
            fallback_type = "graph"
        elif vector_result and vector_result.get("answer"):
            fallback_answer = vector_result.get("answer", "")
            fallback_sources = vector_result.get("sources", [])
            fallback_type = "vector"
        else:
            fallback_answer = f"Error combining hybrid results: {str(e)}"
            fallback_type = "error"

        logger.warning(
            f"Falling back to {fallback_type} result only due to combination error"
        )

        return {
            "answer": fallback_answer,
            "sources": fallback_sources,
            "query_type": "hybrid_fallback",
            "error": str(e),
            "fallback_type": fallback_type
        }
