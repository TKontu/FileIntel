from typing import Any, Dict, List, Optional
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.rag.reranker_service import RerankerService
import logging

logger = logging.getLogger(__name__)


class VectorRAGService:
    def __init__(self, config, storage: PostgreSQLStorage):
        """
        Initialize VectorRAGService with configuration and storage.

        Args:
            config: Configuration object with RAG and LLM settings
            storage: PostgreSQL storage instance
        """
        self.config = config
        self.storage = storage

        # Initialize embedding provider
        from fileintel.llm_integration.embedding_provider import OpenAIEmbeddingProvider

        self.embedding_provider = OpenAIEmbeddingProvider(settings=config)

        # Initialize LLM provider
        from fileintel.llm_integration.unified_provider import UnifiedLLMProvider

        self.llm_provider = UnifiedLLMProvider(config, storage)

        # Initialize reranker service if enabled
        self.reranker = None
        if config.rag.reranking.enabled and config.rag.reranking.rerank_vector_results:
            try:
                self.reranker = RerankerService(config)
                logger.info("RerankerService initialized for VectorRAG")
            except Exception as e:
                logger.warning(f"Failed to initialize RerankerService: {e}. Continuing without reranking.")

    def query(
        self,
        query: str,
        collection_id: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
        similarity_metric: str = "cosine",
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Perform vector similarity search and generate answer.

        Args:
            query: The user's question
            collection_id: Collection to search in
            document_id: Optional specific document to search within
            top_k: Number of similar chunks to retrieve
            similarity_metric: 'cosine', 'l2', or 'inner_product'
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            Dict containing answer, sources, and metadata
        """
        try:
            # Validate collection exists (direct sync call)
            collection = self.storage.get_collection(collection_id)
            if not collection:
                logger.warning(f"Collection '{collection_id}' not found")
                return {
                    "answer": f"Collection '{collection_id}' not found. Please verify the collection ID and try again.",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": {
                        "error": "collection_not_found",
                        "query": query,
                        "collection_id": collection_id,
                    },
                }

            # Generate embedding for the query
            try:
                query_embedding = self.embedding_provider.get_embeddings([query])[0]
            except Exception as e:
                logger.error(f"Failed to generate embedding for query: {e}")
                return {
                    "answer": f"Failed to process query due to embedding generation error: {str(e)}",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": {
                        "error": "embedding_generation_failed",
                        "query": query,
                        "collection_id": collection_id,
                        "error_details": str(e),
                    },
                }

            # Determine retrieval limit (more if reranking enabled)
            retrieval_limit = top_k
            if self.reranker is not None:
                retrieval_limit = self.config.rag.reranking.initial_retrieval_k
                logger.debug(f"Reranking enabled: retrieving {retrieval_limit} chunks for reranking to {top_k}")

            # Retrieve similar chunks using enhanced storage methods
            if document_id:
                # Search within specific document
                similar_chunks = self.storage.find_relevant_chunks_in_document(
                    document_id=document_id,
                    query_embedding=query_embedding,
                    limit=retrieval_limit,
                    similarity_threshold=min_similarity,
                )
            else:
                # Search within entire collection
                similar_chunks = self.storage.find_relevant_chunks_in_collection(
                    collection_id=collection_id,
                    query_embedding=query_embedding,
                    limit=retrieval_limit,
                    similarity_threshold=min_similarity,
                )

            # Rerank results if reranker is enabled
            if self.reranker is not None and similar_chunks:
                try:
                    # Convert chunks to format expected by reranker
                    passages = []
                    for chunk in similar_chunks:
                        passages.append({
                            "content": chunk["text"],
                            "similarity_score": chunk["similarity"],
                            **chunk  # Include all other fields
                        })

                    # Rerank passages
                    reranked_passages = self.reranker.rerank(
                        query=query,
                        passages=passages,
                        top_k=top_k,
                        passage_text_key="content"
                    )

                    # Convert back to chunk format
                    similar_chunks = []
                    for passage in reranked_passages:
                        chunk = passage.copy()
                        chunk["text"] = chunk.pop("content")
                        chunk["similarity"] = chunk["reranked_score"]
                        similar_chunks.append(chunk)

                    logger.info(f"Reranked {len(passages)} → {len(similar_chunks)} chunks for query")
                except Exception as e:
                    logger.error(f"Reranking failed: {e}. Using original vector search results.")
                    # Keep original similar_chunks, just truncate to top_k
                    similar_chunks = similar_chunks[:top_k]
            elif similar_chunks and self.reranker is None:
                # No reranking, just truncate to top_k
                similar_chunks = similar_chunks[:top_k]

            if not similar_chunks:
                return {
                    "answer": f"No relevant information found in collection '{collection_id}' for query: '{query}'",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": {"query": query, "collection_id": collection_id},
                }

            # Generate answer based on retrieved chunks with query type classification
            query_type = self._classify_query_type(query)
            answer = self._generate_answer(query, similar_chunks, query_type)

            # Format sources using enhanced citation formatting
            sources = []
            for chunk in similar_chunks:
                # Get enhanced citation for the source
                try:
                    from fileintel.citation import format_source_reference, format_in_text_citation
                    citation = format_source_reference(chunk)
                    in_text_citation = format_in_text_citation(chunk)
                except ImportError:
                    # Fallback to original filename
                    citation = chunk["original_filename"]
                    in_text_citation = f"({chunk['original_filename']})"

                source_data = {
                    "document_id": chunk["document_id"],
                    "chunk_id": chunk["chunk_id"],
                    "filename": chunk["original_filename"],  # Keep original for compatibility
                    "citation": citation,  # Enhanced citation format
                    "in_text_citation": in_text_citation,  # For Harvard style in-text references
                    "text": chunk["text"][:200] + "..."
                    if len(chunk["text"]) > 200
                    else chunk["text"],
                    "similarity_score": chunk["similarity"],
                    "relevance_score": chunk["similarity"],  # For CLI compatibility
                    "distance": 1 - chunk["similarity"],  # Convert similarity to distance
                    "position": chunk.get("chunk_index", chunk.get("position", 0)),
                    "chunk_metadata": chunk.get("metadata", chunk.get("chunk_metadata", {})),
                    "document_metadata": chunk.get("document_metadata", {}),  # Include document metadata
                }
                sources.append(source_data)

            # Calculate confidence based on similarity scores
            confidence = self._calculate_confidence(similar_chunks)

            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "metadata": {
                    "query": query,
                    "collection_id": collection_id,
                    "chunks_retrieved": len(similar_chunks),
                },
            }

        except Exception as e:
            logger.error(f"Error in vector RAG query: {e}")
            return {
                "answer": f"Error occurred while processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "error": str(e),
                    "query": query,
                    "collection_id": collection_id,
                },
            }

    def _generate_answer(
        self, query: str, chunks: List[Dict[str, Any]], query_type: str = "general"
    ) -> str:
        """
        Generate answer based on retrieved chunks using specialized RAG response generation.
        """
        if not chunks:
            return "No relevant information found to answer the query."

        # If no LLM provider, fall back to simple context concatenation
        if not self.llm_provider:
            context_parts = [
                f"[Source {i+1}]: {chunk['text']}" for i, chunk in enumerate(chunks[:5])
            ]
            context = "\n\n".join(context_parts)
            return f"Based on the available documents:\n\n{context[:800]}..."

        # Use specialized RAG response generation
        try:
            response = self.llm_provider.generate_rag_response(
                query=query,
                context_chunks=chunks,
                query_type=query_type,
                max_tokens=600,
                temperature=0.1,
            )

            if hasattr(response, "content"):
                return response.content
            elif isinstance(response, dict) and "content" in response:
                return response["content"]
            else:
                logger.warning(
                    "Unexpected LLM response format, falling back to context"
                )
                context_parts = [
                    f"[Source {i+1}]: {chunk['text']}"
                    for i, chunk in enumerate(chunks[:5])
                ]
                context = "\n\n".join(context_parts)
                return f"Based on the available documents:\n\n{context[:800]}..."

        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            # Fallback to simple context presentation
            context_parts = [
                f"[Source {i+1}]: {chunk['text']}" for i, chunk in enumerate(chunks[:5])
            ]
            context = "\n\n".join(context_parts)
            return f"Based on the available documents:\n\n{context[:800]}..."

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type for specialized RAG prompting.

        Args:
            query: User's question

        Returns:
            Query type classification ('factual', 'analytical', 'summarization', 'comparison', 'general')
        """
        query_lower = query.lower()

        # Factual questions - who, what, when, where, how many, specific facts
        if any(
            word in query_lower
            for word in [
                "who",
                "what",
                "when",
                "where",
                "how many",
                "which",
                "specific",
                "exactly",
            ]
        ):
            return "factual"

        # Analytical questions - why, how, analysis, relationship, impact
        elif any(
            word in query_lower
            for word in [
                "why",
                "how",
                "analyze",
                "explain",
                "relationship",
                "impact",
                "cause",
                "effect",
                "implications",
            ]
        ):
            return "analytical"

        # Summarization questions - summarize, overview, main points
        elif any(
            word in query_lower
            for word in [
                "summarize",
                "summary",
                "overview",
                "main points",
                "key points",
                "outline",
            ]
        ):
            return "summarization"

        # Comparison questions - compare, contrast, difference, similar
        elif any(
            word in query_lower
            for word in [
                "compare",
                "contrast",
                "difference",
                "differences",
                "similar",
                "similarities",
                "versus",
                "vs",
            ]
        ):
            return "comparison"

        # Default to general
        else:
            return "general"

    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on similarity scores.
        """
        if not chunks:
            return 0.0

        # Average similarity score as confidence
        similarities = [chunk.get("similarity", 0.0) for chunk in chunks]
        return sum(similarities) / len(similarities) if similarities else 0.0
