"""
Unit tests for CitationGenerationService.

Tests the citation generation service including vector search integration,
metadata validation, confidence scoring, and citation formatting.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from fileintel.services.citation_service import CitationGenerationService


class TestCitationGenerationService:
    """Test suite for CitationGenerationService."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with citation settings."""
        config = Mock()

        # Citation settings
        citation_config = Mock()
        citation_config.min_similarity = 0.7
        citation_config.default_top_k = 5
        citation_config.max_excerpt_length = 300
        citation_config.confidence_thresholds = {
            "high": 0.85,
            "medium": 0.70,
            "low": 0.0
        }
        config.citation = citation_config

        return config

    @pytest.fixture
    def mock_storage(self):
        """Mock PostgreSQL storage."""
        return Mock()

    @pytest.fixture
    def citation_service(self, mock_config, mock_storage):
        """Create CitationGenerationService instance."""
        return CitationGenerationService(mock_config, mock_storage)

    @pytest.fixture
    def complete_chunk(self) -> Dict[str, Any]:
        """Sample chunk with complete metadata."""
        return {
            "chunk_id": "chunk-123",
            "document_id": "doc-456",
            "text": "Machine learning models learn patterns from data through iterative training processes.",
            "similarity_score": 0.92,
            "document_metadata": {
                "title": "Introduction to Machine Learning",
                "authors": ["John Smith", "Jane Doe"],
                "author_surnames": ["Smith", "Doe"],
                "publication_date": "2023-05-15",
                "publisher": "Academic Press",
                "llm_extracted": True
            },
            "metadata": {
                "page_number": 45
            },
            "original_filename": "ml_intro.pdf"
        }

    @pytest.fixture
    def incomplete_chunk(self) -> Dict[str, Any]:
        """Sample chunk with incomplete metadata (no LLM extraction)."""
        return {
            "chunk_id": "chunk-789",
            "document_id": "doc-012",
            "text": "Neural networks consist of interconnected layers of neurons.",
            "similarity_score": 0.75,
            "document_metadata": {},
            "metadata": {},
            "original_filename": "neural_networks.pdf"
        }

    def test_input_validation_empty_text(self, citation_service):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text segment cannot be empty"):
            citation_service.generate_citation(
                text_segment="   ",
                collection_id="collection-123"
            )

    def test_input_validation_too_short(self, citation_service):
        """Test that text shorter than 10 chars raises ValueError."""
        with pytest.raises(ValueError, match="at least 10 characters"):
            citation_service.generate_citation(
                text_segment="Too short",
                collection_id="collection-123"
            )

    def test_input_validation_too_long(self, citation_service):
        """Test that text longer than 5000 chars raises ValueError."""
        long_text = "a" * 5001
        with pytest.raises(ValueError, match="must not exceed 5000 characters"):
            citation_service.generate_citation(
                text_segment=long_text,
                collection_id="collection-123"
            )

    def test_input_validation_similarity_range(self, citation_service):
        """Test that invalid similarity threshold raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            citation_service.generate_citation(
                text_segment="Valid text segment here",
                collection_id="collection-123",
                min_similarity=1.5  # Invalid
            )

    def test_successful_citation_generation_high_confidence(
        self, citation_service, complete_chunk, mock_config
    ):
        """Test successful citation generation with high confidence."""
        # Mock vector search to return complete chunk
        with patch.object(citation_service, '_find_best_match') as mock_find:
            mock_find.return_value = {
                "chunk": complete_chunk,
                "similarity_score": 0.92
            }

            result = citation_service.generate_citation(
                text_segment="Machine learning models learn from data",
                collection_id="collection-123"
            )

            # Verify structure
            assert "citation" in result
            assert "source" in result
            assert "confidence" in result

            # Verify citation format
            citation = result["citation"]
            assert citation["style"] == "harvard"
            assert "Smith" in citation["in_text"]
            assert "2023" in citation["in_text"]
            assert "p. 45" in citation["in_text"]

            # Verify confidence
            assert result["confidence"] == "high"  # 0.92 >= 0.85 AND has metadata

            # Verify source details
            source = result["source"]
            assert source["document_id"] == "doc-456"
            assert source["similarity_score"] == 0.92
            assert source["page_numbers"] == [45]

    def test_successful_citation_generation_medium_confidence(
        self, citation_service, incomplete_chunk
    ):
        """Test citation generation with medium confidence (no metadata but good similarity)."""
        # Mock vector search to return incomplete chunk with good similarity
        with patch.object(citation_service, '_find_best_match') as mock_find:
            mock_find.return_value = {
                "chunk": incomplete_chunk,
                "similarity_score": 0.75
            }

            result = citation_service.generate_citation(
                text_segment="Neural networks have layers",
                collection_id="collection-123"
            )

            # Verify confidence
            assert result["confidence"] == "medium"  # 0.75 >= 0.70

            # Verify warning about incomplete metadata
            assert "warning" in result
            assert "metadata unavailable" in result["warning"].lower()

            # Verify fallback citation uses filename
            citation = result["citation"]
            assert "neural_networks" in citation["in_text"].lower()

    def test_successful_citation_generation_low_confidence(
        self, citation_service, incomplete_chunk
    ):
        """Test citation generation with low confidence (low similarity, no metadata)."""
        # Mock vector search to return incomplete chunk with low similarity
        incomplete_chunk["similarity_score"] = 0.65

        with patch.object(citation_service, '_find_best_match') as mock_find:
            mock_find.return_value = {
                "chunk": incomplete_chunk,
                "similarity_score": 0.65
            }

            result = citation_service.generate_citation(
                text_segment="Some text segment",
                collection_id="collection-123"
            )

            # Verify confidence
            assert result["confidence"] == "low"  # <0.70 AND no metadata

    def test_no_source_found_above_threshold(self, citation_service):
        """Test that RuntimeError is raised when no source found."""
        # Mock vector search to return None
        with patch.object(citation_service, '_find_best_match') as mock_find:
            mock_find.return_value = None

            with pytest.raises(RuntimeError, match="No source found above similarity threshold"):
                citation_service.generate_citation(
                    text_segment="Text with no matching source",
                    collection_id="collection-123"
                )

    def test_custom_min_similarity(self, citation_service, complete_chunk):
        """Test that custom min_similarity is passed to vector search."""
        with patch.object(citation_service, '_find_best_match') as mock_find:
            mock_find.return_value = {
                "chunk": complete_chunk,
                "similarity_score": 0.92
            }

            citation_service.generate_citation(
                text_segment="Test text segment here",
                collection_id="collection-123",
                min_similarity=0.85
            )

            # Verify min_similarity was passed
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs["min_similarity"] == 0.85

    def test_custom_top_k(self, citation_service, complete_chunk):
        """Test that custom top_k is passed to vector search."""
        with patch.object(citation_service, '_find_best_match') as mock_find:
            mock_find.return_value = {
                "chunk": complete_chunk,
                "similarity_score": 0.92
            }

            citation_service.generate_citation(
                text_segment="Test text segment here",
                collection_id="collection-123",
                top_k=10
            )

            # Verify top_k was passed
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs["top_k"] == 10

    def test_document_id_filtering(self, citation_service, complete_chunk):
        """Test that document_id parameter is passed for filtering."""
        with patch.object(citation_service, '_find_best_match') as mock_find:
            mock_find.return_value = {
                "chunk": complete_chunk,
                "similarity_score": 0.92
            }

            citation_service.generate_citation(
                text_segment="Test text segment here",
                collection_id="collection-123",
                document_id="specific-doc"
            )

            # Verify document_id was passed
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs["document_id"] == "specific-doc"

    def test_validate_source_metadata_complete(self, citation_service, complete_chunk):
        """Test metadata validation with complete metadata."""
        result = citation_service._validate_source_metadata(complete_chunk)
        assert result is True  # Has LLM extracted metadata with authors

    def test_validate_source_metadata_incomplete(self, citation_service, incomplete_chunk):
        """Test metadata validation with incomplete metadata."""
        result = citation_service._validate_source_metadata(incomplete_chunk)
        assert result is False  # No LLM extracted metadata

    def test_validate_source_metadata_has_title_only(self, citation_service):
        """Test metadata validation accepts title without authors."""
        chunk = {
            "document_metadata": {
                "title": "Some Document",
                "llm_extracted": True
            }
        }
        result = citation_service._validate_source_metadata(chunk)
        assert result is True  # Has title

    def test_confidence_determination_high(self, citation_service):
        """Test high confidence determination (high similarity + metadata)."""
        confidence = citation_service._determine_confidence(
            similarity_score=0.90,
            has_metadata=True
        )
        assert confidence == "high"

    def test_confidence_determination_medium_good_similarity(self, citation_service):
        """Test medium confidence (good similarity, no metadata)."""
        confidence = citation_service._determine_confidence(
            similarity_score=0.80,
            has_metadata=False
        )
        assert confidence == "medium"

    def test_confidence_determination_medium_has_metadata(self, citation_service):
        """Test medium confidence (low similarity, has metadata)."""
        confidence = citation_service._determine_confidence(
            similarity_score=0.60,
            has_metadata=True
        )
        assert confidence == "medium"

    def test_confidence_determination_low(self, citation_service):
        """Test low confidence (low similarity, no metadata)."""
        confidence = citation_service._determine_confidence(
            similarity_score=0.60,
            has_metadata=False
        )
        assert confidence == "low"

    def test_extract_page_numbers_single_page(self, citation_service):
        """Test page number extraction from page_number field."""
        chunk_metadata = {"page_number": 42}
        result = citation_service._extract_page_numbers(chunk_metadata)
        assert result == [42]

    def test_extract_page_numbers_range(self, citation_service):
        """Test page number extraction from page_range field."""
        chunk_metadata = {"page_range": "45-47"}
        result = citation_service._extract_page_numbers(chunk_metadata)
        assert result == [45, 46, 47]

    def test_extract_page_numbers_list(self, citation_service):
        """Test page number extraction from pages list."""
        chunk_metadata = {"pages": [10, 11, 12]}
        result = citation_service._extract_page_numbers(chunk_metadata)
        assert result == [10, 11, 12]

    def test_extract_page_numbers_none(self, citation_service):
        """Test page number extraction with no page info."""
        chunk_metadata = {}
        result = citation_service._extract_page_numbers(chunk_metadata)
        assert result == []

    def test_text_excerpt_truncation(self, citation_service, complete_chunk):
        """Test that long text excerpts are truncated."""
        # Create chunk with very long text
        long_chunk = complete_chunk.copy()
        long_chunk["text"] = "a" * 500  # Longer than max_excerpt_length (300)

        with patch.object(citation_service, '_find_best_match') as mock_find:
            mock_find.return_value = {
                "chunk": long_chunk,
                "similarity_score": 0.92
            }

            result = citation_service.generate_citation(
                text_segment="Test text segment here",
                collection_id="collection-123"
            )

            # Verify excerpt is truncated
            excerpt = result["source"]["text_excerpt"]
            assert len(excerpt) <= 303  # 300 + "..."
            assert excerpt.endswith("...")

    def test_llm_analysis_disabled_by_default(self, citation_service, complete_chunk):
        """Test that LLM analysis is not included by default."""
        with patch.object(citation_service, '_find_best_match') as mock_find:
            mock_find.return_value = {
                "chunk": complete_chunk,
                "similarity_score": 0.92
            }

            result = citation_service.generate_citation(
                text_segment="Test text segment here",
                collection_id="collection-123",
                include_llm_analysis=False
            )

            # Verify no relevance_note
            assert "relevance_note" not in result

    def test_llm_analysis_enabled(self, citation_service, complete_chunk, mock_config):
        """Test that LLM analysis is included when requested."""
        with patch.object(citation_service, '_find_best_match') as mock_find, \
             patch.object(citation_service, '_enhance_with_llm') as mock_enhance:

            mock_find.return_value = {
                "chunk": complete_chunk,
                "similarity_score": 0.92
            }
            mock_enhance.return_value = "This source is relevant because..."

            result = citation_service.generate_citation(
                text_segment="Test text segment here",
                collection_id="collection-123",
                include_llm_analysis=True
            )

            # Verify relevance_note is included
            assert "relevance_note" in result
            assert result["relevance_note"] == "This source is relevant because..."

    def test_llm_analysis_failure_graceful(self, citation_service, complete_chunk):
        """Test that LLM analysis failure doesn't break citation generation."""
        with patch.object(citation_service, '_find_best_match') as mock_find, \
             patch.object(citation_service, '_enhance_with_llm') as mock_enhance:

            mock_find.return_value = {
                "chunk": complete_chunk,
                "similarity_score": 0.92
            }
            mock_enhance.side_effect = Exception("LLM unavailable")

            result = citation_service.generate_citation(
                text_segment="Test text segment here",
                collection_id="collection-123",
                include_llm_analysis=True
            )

            # Verify citation still generated successfully
            assert "citation" in result
            # Verify relevance_note is None (graceful failure)
            assert result["relevance_note"] is None

    def test_vector_service_lazy_initialization(self, citation_service):
        """Test that vector service is lazily initialized."""
        # Initially None
        assert citation_service._vector_service is None

        # Access property (VectorRAGService is imported inside the property)
        with patch('fileintel.rag.vector_rag.services.vector_rag_service.VectorRAGService') as mock_vector_class:
            mock_vector_instance = Mock()
            mock_vector_class.return_value = mock_vector_instance

            service = citation_service.vector_service

            # Verify initialized
            assert service == mock_vector_instance
            mock_vector_class.assert_called_once()

            # Accessing again doesn't re-initialize
            service2 = citation_service.vector_service
            assert service2 == mock_vector_instance
            assert mock_vector_class.call_count == 1  # Still only called once


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
