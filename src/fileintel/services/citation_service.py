"""
Citation Generation Service.

Provides citation generation for text segments using vector similarity search
to find source documents and generate proper Harvard-style citations.
"""

import logging
from typing import Dict, Any, Optional, List
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.core.config import Settings
from fileintel.citation.citation_formatter import CitationFormatter

logger = logging.getLogger(__name__)


class CitationGenerationService:
    """Service for generating citations from text segments using vector search."""

    def __init__(
        self,
        config: Settings,
        storage: PostgreSQLStorage
    ):
        """
        Initialize Citation Generation Service.

        Args:
            config: Configuration settings
            storage: PostgreSQL storage instance
        """
        self.config = config
        self.storage = storage
        self.citation_formatter = CitationFormatter(style="harvard")

        # Lazy-initialize vector service (avoids circular dependencies)
        self._vector_service = None

        # Get citation settings with safe defaults
        self.min_similarity = getattr(
            getattr(config, 'citation', None),
            'min_similarity',
            0.7
        )
        self.default_top_k = getattr(
            getattr(config, 'citation', None),
            'default_top_k',
            5
        )

        # Confidence thresholds
        citation_config = getattr(config, 'citation', None)
        if citation_config and hasattr(citation_config, 'confidence_thresholds'):
            self.confidence_thresholds = citation_config.confidence_thresholds
        else:
            self.confidence_thresholds = {
                "high": 0.85,
                "medium": 0.70,
                "low": 0.0
            }

        self.max_excerpt_length = getattr(
            getattr(config, 'citation', None),
            'max_excerpt_length',
            300
        )

    @property
    def vector_service(self):
        """Lazy-initialize VectorRAGService to avoid circular imports."""
        if self._vector_service is None:
            from fileintel.rag.vector_rag.services.vector_rag_service import VectorRAGService
            self._vector_service = VectorRAGService(self.config, self.storage)
        return self._vector_service

    def generate_citation(
        self,
        text_segment: str,
        collection_id: str,
        document_id: Optional[str] = None,
        min_similarity: Optional[float] = None,
        include_llm_analysis: bool = False,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate citation for a text segment.

        Uses vector similarity search to find the most relevant source document,
        then generates proper Harvard-style citations.

        Args:
            text_segment: The text that needs citation (10-5000 chars)
            collection_id: Collection to search in
            document_id: Optional specific document to search within
            min_similarity: Minimum similarity threshold (0.0-1.0)
            include_llm_analysis: Use LLM for relevance analysis
            top_k: Number of candidates to retrieve

        Returns:
            Dict containing:
            - citation: {in_text, full, style}
            - source: {document_id, chunk_id, similarity_score, text_excerpt, ...}
            - confidence: "high"|"medium"|"low"
            - relevance_note: str (if include_llm_analysis)
            - warning: str (if applicable)

        Raises:
            ValueError: If input is invalid
            RuntimeError: If no source found above threshold
        """
        # Validate input
        if not text_segment or not text_segment.strip():
            raise ValueError("Text segment cannot be empty")

        text_segment = text_segment.strip()

        if len(text_segment) < 10:
            raise ValueError("Text segment must be at least 10 characters")

        if len(text_segment) > 5000:
            raise ValueError("Text segment must not exceed 5000 characters")

        # Use provided values or defaults
        similarity_threshold = min_similarity if min_similarity is not None else self.min_similarity
        k = top_k if top_k is not None else self.default_top_k

        # Validate similarity threshold
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0")

        logger.info(
            f"Generating citation for text segment (length: {len(text_segment)}, "
            f"collection: {collection_id}, threshold: {similarity_threshold})"
        )

        try:
            # Find best matching source
            best_match = self._find_best_match(
                text_segment=text_segment,
                collection_id=collection_id,
                document_id=document_id,
                min_similarity=similarity_threshold,
                top_k=k
            )

            if not best_match:
                # No match found above threshold
                raise RuntimeError(
                    f"No source found above similarity threshold of {similarity_threshold}"
                )

            # Extract chunk data
            chunk = best_match["chunk"]
            similarity_score = best_match["similarity_score"]

            # Validate metadata availability
            has_metadata = self._validate_source_metadata(chunk)

            # Generate citations
            citations = self._generate_citations_from_chunk(chunk)

            # Determine confidence level
            confidence = self._determine_confidence(similarity_score, has_metadata)

            # Build source details
            source_details = self._build_source_details(chunk, similarity_score)

            # Build response
            response = {
                "citation": citations,
                "source": source_details,
                "confidence": confidence
            }

            # Add warning if metadata is incomplete
            if not has_metadata:
                response["warning"] = (
                    "Source found but metadata unavailable. "
                    "Citation based on filename."
                )

            # Optional: LLM-enhanced relevance analysis
            if include_llm_analysis:
                try:
                    relevance_note = self._enhance_with_llm(
                        text_segment, chunk, citations
                    )
                    response["relevance_note"] = relevance_note
                except Exception as e:
                    logger.warning(f"LLM analysis failed: {e}")
                    response["relevance_note"] = None

            logger.info(
                f"Citation generated successfully (similarity: {similarity_score:.3f}, "
                f"confidence: {confidence})"
            )

            return response

        except RuntimeError:
            # Re-raise runtime errors (no match found)
            raise
        except Exception as e:
            logger.error(f"Citation generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Citation generation failed: {str(e)}")

    def _find_best_match(
        self,
        text_segment: str,
        collection_id: str,
        document_id: Optional[str],
        min_similarity: float,
        top_k: int
    ) -> Optional[Dict[str, Any]]:
        """
        Find best matching source using vector similarity.

        Args:
            text_segment: Text to match
            collection_id: Collection to search
            document_id: Optional specific document
            min_similarity: Minimum similarity threshold
            top_k: Number of candidates to retrieve

        Returns:
            Dict with 'chunk' and 'similarity_score' if found, None otherwise
        """
        try:
            # Use VectorRAGService to perform similarity search
            # We use the query method but extract only the sources
            result = self.vector_service.query(
                query=text_segment,
                collection_id=collection_id,
                document_id=document_id,
                top_k=top_k,
                min_similarity=min_similarity,
                answer_format="default"  # We don't need the generated answer
            )

            sources = result.get("sources", [])

            if not sources:
                logger.info(f"No sources found above similarity threshold {min_similarity}")
                return None

            # Get the best match (highest similarity)
            best_source = sources[0]
            similarity_score = best_source.get(
                "similarity_score",
                best_source.get("relevance_score", 0.0)
            )

            logger.info(
                f"Found best match: similarity={similarity_score:.3f}, "
                f"document={best_source.get('document_id', 'unknown')}"
            )

            return {
                "chunk": best_source,
                "similarity_score": similarity_score
            }

        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return None

    def _validate_source_metadata(self, chunk: Dict[str, Any]) -> bool:
        """
        Check if source has sufficient metadata for proper citation.

        Args:
            chunk: Chunk data with metadata

        Returns:
            True if has sufficient metadata for citation
        """
        document_metadata = chunk.get("document_metadata", {})

        # Check if we have LLM-extracted metadata
        if not document_metadata.get("llm_extracted", False):
            return False

        # Check for essential fields
        has_authors = bool(document_metadata.get("authors") or
                          document_metadata.get("author_surnames"))
        has_title = bool(document_metadata.get("title"))
        has_year = bool(document_metadata.get("publication_date"))

        return has_authors or has_title

    def _generate_citations_from_chunk(
        self,
        chunk: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate both in-text and full citations from chunk.

        Args:
            chunk: Chunk data with metadata

        Returns:
            Dict with 'in_text', 'full', and 'style' keys
        """
        in_text = self.citation_formatter.format_in_text_citation(chunk)
        full = self.citation_formatter.format_full_citation(chunk)

        return {
            "in_text": in_text,
            "full": full,
            "style": "harvard"
        }

    def _determine_confidence(
        self,
        similarity_score: float,
        has_metadata: bool
    ) -> str:
        """
        Determine confidence level based on match quality.

        Args:
            similarity_score: Similarity score (0.0-1.0)
            has_metadata: Whether source has complete metadata

        Returns:
            "high", "medium", or "low"
        """
        # High confidence: high similarity AND full metadata
        if similarity_score >= self.confidence_thresholds["high"] and has_metadata:
            return "high"

        # Medium confidence: good similarity OR has metadata
        if similarity_score >= self.confidence_thresholds["medium"] or has_metadata:
            return "medium"

        # Low confidence: low similarity and no metadata
        return "low"

    def _build_source_details(
        self,
        chunk: Dict[str, Any],
        similarity_score: float
    ) -> Dict[str, Any]:
        """
        Build source details dictionary for response.

        Args:
            chunk: Chunk data
            similarity_score: Similarity score

        Returns:
            Dict with source details
        """
        # Extract text excerpt (truncate if needed)
        text = chunk.get("text", chunk.get("chunk_text", ""))
        if len(text) > self.max_excerpt_length:
            text = text[:self.max_excerpt_length] + "..."

        # Extract page numbers
        chunk_metadata = chunk.get("metadata", chunk.get("chunk_metadata", {}))
        page_numbers = self._extract_page_numbers(chunk_metadata)

        # Build details
        details = {
            "document_id": chunk.get("document_id", ""),
            "chunk_id": chunk.get("chunk_id", chunk.get("id", "")),
            "similarity_score": similarity_score,
            "text_excerpt": text,
            "page_numbers": page_numbers,
            "filename": chunk.get("original_filename", chunk.get("filename", "Unknown")),
            "metadata": chunk.get("document_metadata", {})
        }

        return details

    def _extract_page_numbers(self, chunk_metadata: Dict[str, Any]) -> List[int]:
        """
        Extract page numbers from chunk metadata.

        Args:
            chunk_metadata: Chunk metadata dict

        Returns:
            List of page numbers
        """
        page_numbers = []

        # Try different field names
        page_number = chunk_metadata.get("page_number")
        if page_number:
            if isinstance(page_number, int):
                page_numbers = [page_number]
            elif isinstance(page_number, str) and page_number.isdigit():
                page_numbers = [int(page_number)]

        # Try page_range
        if not page_numbers:
            page_range = chunk_metadata.get("page_range")
            if page_range:
                if isinstance(page_range, str):
                    if "-" in page_range:
                        # Handle ranges like "45-47"
                        try:
                            start, end = page_range.split("-")
                            page_numbers = list(range(int(start), int(end) + 1))
                        except ValueError:
                            pass
                    elif page_range.isdigit():
                        page_numbers = [int(page_range)]

        # Try pages list
        if not page_numbers:
            pages = chunk_metadata.get("pages")
            if pages and isinstance(pages, list):
                page_numbers = [p for p in pages if isinstance(p, int)]

        return page_numbers

    def _enhance_with_llm(
        self,
        text_segment: str,
        chunk: Dict[str, Any],
        citations: Dict[str, str]
    ) -> str:
        """
        Use LLM to generate relevance note explaining why this source supports the text.

        Args:
            text_segment: Original text needing citation
            chunk: Matched source chunk
            citations: Generated citations

        Returns:
            Relevance note explaining the connection
        """
        try:
            from fileintel.llm_integration.unified_provider import UnifiedLLMProvider

            llm_provider = UnifiedLLMProvider(self.config, self.storage)

            # Build prompt
            source_text = chunk.get("text", chunk.get("chunk_text", ""))[:500]
            document_metadata = chunk.get("document_metadata", {})
            title = document_metadata.get("title", "Unknown")

            prompt = f"""You are a citation expert. Analyze the relevance between the original text and the source.

**Original Text Being Cited:**
"{text_segment}"

**Source Text Excerpt:**
"{source_text}"

**Source Title:** {title}

**Generated Citation:** {citations["in_text"]}

Provide a brief (1-2 sentence) explanation of why this source is relevant and how it supports the original text.

Relevance explanation:"""

            # Use fast model for analysis
            model = getattr(
                getattr(self.config, 'citation', None),
                'llm_analysis_model',
                'gemma3-4B'
            )

            response = llm_provider.generate_response(
                prompt=prompt,
                model=model,
                max_tokens=150,
                temperature=0.3
            )

            relevance_note = response.content if hasattr(response, 'content') else str(response)
            return relevance_note.strip()

        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return None
