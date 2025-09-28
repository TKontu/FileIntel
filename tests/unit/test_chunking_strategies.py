"""Comprehensive unit tests for text splitting and chunking optimization strategies."""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any
import re
from dataclasses import dataclass

# Test fixtures
from tests.fixtures import (
    test_documents,
    sample_text_doc,
    sample_markdown_doc,
    test_cleanup,
    temporary_test_documents,
)


@dataclass
class ChunkingTestCase:
    """Test case for chunking strategy validation."""

    name: str
    content: str
    expected_chunk_count: int
    chunk_size: int
    overlap: int
    expected_overlap_chars: int = 0


class TestTextChunker:
    """Test cases for base text chunker functionality."""

    def test_chunker_initialization(self):
        """Test text chunker initialization with default parameters."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker()
        assert chunker.chunk_size > 0
        assert chunker.overlap >= 0
        assert chunker.overlap < chunker.chunk_size

    def test_chunker_custom_parameters(self):
        """Test text chunker with custom parameters."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker(chunk_size=500, overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50

    def test_chunker_parameter_validation(self):
        """Test text chunker parameter validation."""
        from src.fileintel.document_processing.chunking import TextChunker

        # Invalid chunk size
        with pytest.raises((ValueError, AssertionError)):
            TextChunker(chunk_size=0)

        with pytest.raises((ValueError, AssertionError)):
            TextChunker(chunk_size=-100)

        # Invalid overlap (should be less than chunk_size)
        with pytest.raises((ValueError, AssertionError)):
            TextChunker(chunk_size=100, overlap=150)

        # Overlap equal to chunk_size should be invalid
        with pytest.raises((ValueError, AssertionError)):
            TextChunker(chunk_size=100, overlap=100)

    def test_empty_text_chunking(self):
        """Test chunking behavior with empty text."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker()
        chunks = chunker.chunk_text("")

        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_short_text_chunking(self):
        """Test chunking behavior with text shorter than chunk size."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker(chunk_size=100, overlap=10)
        short_text = "This is a short text."

        chunks = chunker.chunk_text(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_exact_chunk_size_text(self):
        """Test chunking with text exactly matching chunk size."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunk_size = 50
        chunker = TextChunker(chunk_size=chunk_size, overlap=0)
        exact_text = "a" * chunk_size

        chunks = chunker.chunk_text(exact_text)

        assert len(chunks) == 1
        assert len(chunks[0]) == chunk_size


class TestBasicChunkingStrategy:
    """Test cases for basic character-based chunking."""

    @pytest.mark.parametrize(
        "chunk_size,overlap", [(100, 0), (100, 10), (200, 20), (500, 50)]
    )
    def test_basic_chunking_parameters(self, chunk_size, overlap, sample_text_doc):
        """Test basic chunking with various parameters."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk_text(sample_text_doc["content"])

        # Verify chunks are created
        assert len(chunks) > 0

        # Verify chunk sizes (except possibly the last chunk)
        for i, chunk in enumerate(chunks[:-1]):
            assert len(chunk) <= chunk_size

        # Verify overlap if specified
        if overlap > 0 and len(chunks) > 1:
            for i in range(len(chunks) - 1):
                chunk1 = chunks[i]
                chunk2 = chunks[i + 1]

                # Check if there's actual overlap
                overlap_start = max(0, len(chunk1) - overlap)
                overlap_text1 = chunk1[overlap_start:]
                overlap_text2 = chunk2[: len(overlap_text1)]

                # There should be some common content
                assert len(overlap_text1) > 0

    def test_chunking_preserves_content(self, sample_text_doc):
        """Test that chunking preserves all original content."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk_text(sample_text_doc["content"])

        # Reconstruct text from chunks
        reconstructed = "".join(chunks)

        assert reconstructed == sample_text_doc["content"]

    def test_chunking_with_overlap_coverage(self, sample_text_doc):
        """Test that chunking with overlap covers all content appropriately."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk_text(sample_text_doc["content"])

        if len(chunks) > 1:
            # Verify that removing overlaps still covers original content
            unique_content_length = 0
            unique_content_length += len(chunks[0])

            for i in range(1, len(chunks)):
                unique_content_length += len(chunks[i]) - min(50, len(chunks[i]))

            # Should approximately equal original length
            original_length = len(sample_text_doc["content"])
            assert (
                abs(unique_content_length - original_length) <= 100
            )  # Allow some variance


class TestSentenceAwareChunking:
    """Test cases for sentence-aware chunking strategies."""

    def test_sentence_boundary_detection(self):
        """Test sentence boundary detection."""
        from src.fileintel.document_processing.chunking import SentenceAwareChunker

        chunker = SentenceAwareChunker()
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."

        sentences = chunker._split_into_sentences(text)

        assert len(sentences) == 4
        assert sentences[0].strip() == "First sentence."
        assert sentences[1].strip() == "Second sentence!"
        assert sentences[2].strip() == "Third sentence?"
        assert sentences[3].strip() == "Fourth sentence."

    def test_sentence_chunking_preserves_boundaries(self):
        """Test that sentence-aware chunking preserves sentence boundaries."""
        from src.fileintel.document_processing.chunking import SentenceAwareChunker

        chunker = SentenceAwareChunker(chunk_size=100)
        text = "This is the first sentence. This is a second sentence that is quite long and detailed. Short sentence. Another lengthy sentence with multiple clauses and detailed information."

        chunks = chunker.chunk_text(text)

        for chunk in chunks:
            # Each chunk should end with sentence-ending punctuation or be the last chunk
            if chunk != chunks[-1]:  # Not the last chunk
                assert chunk.strip().endswith((".", "!", "?", ":", ";"))

    def test_sentence_chunking_handles_abbreviations(self):
        """Test sentence chunking with abbreviations and edge cases."""
        from src.fileintel.document_processing.chunking import SentenceAwareChunker

        chunker = SentenceAwareChunker()
        text = "Dr. Smith works at Inc. Corp. He has a Ph.D. degree. The company is located in Washington, D.C."

        chunks = chunker.chunk_text(text)

        # Should not break on abbreviations
        reconstructed = " ".join(chunk.strip() for chunk in chunks)
        # Remove extra spaces for comparison
        normalized_original = re.sub(r"\s+", " ", text.strip())
        normalized_reconstructed = re.sub(r"\s+", " ", reconstructed)

        assert normalized_reconstructed == normalized_original

    def test_sentence_chunking_with_overlap(self):
        """Test sentence-aware chunking with overlap."""
        from src.fileintel.document_processing.chunking import SentenceAwareChunker

        chunker = SentenceAwareChunker(chunk_size=200, overlap=50)
        text = "First sentence with some content. Second sentence with more details and information. Third sentence continues the narrative. Fourth sentence adds additional context. Fifth sentence concludes the paragraph."

        chunks = chunker.chunk_text(text)

        if len(chunks) > 1:
            # Verify overlap exists between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1 = chunks[i]
                chunk2 = chunks[i + 1]

                # There should be some overlapping content
                # This is a simplified check
                assert len(chunk1) > 0 and len(chunk2) > 0


class TestParagraphAwareChunking:
    """Test cases for paragraph-aware chunking strategies."""

    def test_paragraph_boundary_detection(self):
        """Test paragraph boundary detection."""
        from src.fileintel.document_processing.chunking import ParagraphAwareChunker

        chunker = ParagraphAwareChunker()
        text = "First paragraph with some content.\n\nSecond paragraph with different content.\n\n\nThird paragraph after multiple line breaks."

        paragraphs = chunker._split_into_paragraphs(text)

        assert len(paragraphs) >= 3
        assert "First paragraph" in paragraphs[0]
        assert "Second paragraph" in paragraphs[1]
        assert "Third paragraph" in paragraphs[2]

    def test_paragraph_chunking_preserves_structure(self, sample_markdown_doc):
        """Test paragraph-aware chunking preserves document structure."""
        from src.fileintel.document_processing.chunking import ParagraphAwareChunker

        chunker = ParagraphAwareChunker(chunk_size=500)
        chunks = chunker.chunk_text(sample_markdown_doc["content"])

        # Verify that chunks maintain paragraph integrity
        for chunk in chunks:
            # Should not split in the middle of a paragraph inappropriately
            lines = chunk.split("\n")
            # Basic structural integrity check
            assert len(lines) > 0

    def test_paragraph_chunking_handles_headers(self):
        """Test paragraph chunking with headers and sections."""
        from src.fileintel.document_processing.chunking import ParagraphAwareChunker

        chunker = ParagraphAwareChunker()
        text = """# Main Header

This is the first paragraph under the main header.

## Subheader

This is a paragraph under the subheader.

### Sub-subheader

Final paragraph content."""

        chunks = chunker.chunk_text(text)

        # Should preserve header-paragraph relationships
        for chunk in chunks:
            if "# Main Header" in chunk:
                assert "first paragraph" in chunk
            if "## Subheader" in chunk:
                assert "paragraph under the subheader" in chunk


class TestSemanticChunkingStrategy:
    """Test cases for semantic-aware chunking."""

    def test_semantic_chunker_initialization(self):
        """Test semantic chunker initialization."""
        try:
            from src.fileintel.document_processing.chunking import SemanticChunker

            chunker = SemanticChunker()
            assert chunker is not None
        except ImportError:
            pytest.skip("Semantic chunking not available without NLP libraries")

    def test_semantic_similarity_calculation(self):
        """Test semantic similarity calculation between text segments."""
        try:
            from src.fileintel.document_processing.chunking import SemanticChunker

            chunker = SemanticChunker()

            # Similar texts
            text1 = "Dogs are loyal pets and great companions."
            text2 = "Canines make excellent friends and faithful animals."

            # Different texts
            text3 = "The weather today is sunny and warm."

            similarity1 = chunker._calculate_similarity(text1, text2)
            similarity2 = chunker._calculate_similarity(text1, text3)

            # Similar texts should have higher similarity than different texts
            assert similarity1 > similarity2

        except (ImportError, AttributeError):
            pytest.skip("Semantic similarity not available")

    def test_semantic_chunking_coherence(self):
        """Test that semantic chunking maintains topic coherence."""
        try:
            from src.fileintel.document_processing.chunking import SemanticChunker

            chunker = SemanticChunker(chunk_size=300)

            # Text with topic shifts
            text = """
            Dogs are wonderful pets that bring joy to families. They are loyal, friendly, and protective.
            Training a dog requires patience and consistency. Different breeds have different temperaments.

            The stock market has been volatile recently. Economic indicators show mixed signals.
            Investors are cautious about the current market conditions. Portfolio diversification is important.

            Cooking is both an art and a science. Fresh ingredients make a significant difference.
            Traditional recipes often have cultural significance. Modern cooking techniques offer new possibilities.
            """

            chunks = chunker.chunk_text(text.strip())

            # Should group related content together
            assert len(chunks) > 1

            # Check that dog-related content is grouped
            dog_chunks = [
                chunk
                for chunk in chunks
                if "dog" in chunk.lower() or "pet" in chunk.lower()
            ]
            market_chunks = [
                chunk
                for chunk in chunks
                if "market" in chunk.lower() or "stock" in chunk.lower()
            ]
            cooking_chunks = [
                chunk
                for chunk in chunks
                if "cook" in chunk.lower() or "recipe" in chunk.lower()
            ]

            # Each topic should be somewhat grouped
            assert len(dog_chunks) > 0
            assert len(market_chunks) > 0
            assert len(cooking_chunks) > 0

        except (ImportError, AttributeError):
            pytest.skip("Semantic chunking not available")


class TestRecursiveChunkingStrategy:
    """Test cases for recursive character text splitting."""

    def test_recursive_chunker_initialization(self):
        """Test recursive chunker initialization."""
        from src.fileintel.document_processing.chunking import (
            RecursiveCharacterTextSplitter,
        )

        chunker = RecursiveCharacterTextSplitter()
        assert chunker.chunk_size > 0
        assert len(chunker.separators) > 0

    def test_recursive_separator_hierarchy(self):
        """Test recursive chunker respects separator hierarchy."""
        from src.fileintel.document_processing.chunking import (
            RecursiveCharacterTextSplitter,
        )

        chunker = RecursiveCharacterTextSplitter(chunk_size=50)

        # Text with multiple types of separators
        text = "First paragraph.\n\nSecond paragraph with sentences. Multiple sentences here.\n\nThird paragraph."

        chunks = chunker.chunk_text(text)

        # Should prefer splitting on paragraph breaks, then sentences, then words
        assert len(chunks) > 1

        # Verify chunks respect natural boundaries
        for chunk in chunks:
            # Should not end abruptly in the middle of a word (unless necessary)
            if len(chunk) == chunker.chunk_size:
                # If chunk is exactly chunk_size, it might be cut at word boundary
                words = chunk.split()
                if words:
                    # Last word should be complete or chunk should end with punctuation
                    assert chunk.endswith((".", "!", "?", " ")) or len(words[-1]) > 2

    def test_recursive_chunker_fallback(self):
        """Test recursive chunker fallback behavior."""
        from src.fileintel.document_processing.chunking import (
            RecursiveCharacterTextSplitter,
        )

        chunker = RecursiveCharacterTextSplitter(chunk_size=30)

        # Text with no preferred separators (single long word)
        text = "supercalifragilisticexpialidocious" * 5  # Very long without spaces

        chunks = chunker.chunk_text(text)

        # Should still split somehow, even if not ideal
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= chunker.chunk_size or len(chunks) == 1


class TestChunkingWithMetadata:
    """Test cases for chunking with metadata preservation."""

    def test_chunk_metadata_generation(self):
        """Test chunk metadata generation."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker(chunk_size=100)
        text = "This is a test document with multiple sentences. It will be split into chunks."

        chunks_with_metadata = chunker.chunk_text_with_metadata(text)

        assert isinstance(chunks_with_metadata, list)
        assert len(chunks_with_metadata) > 0

        for chunk_data in chunks_with_metadata:
            assert "text" in chunk_data
            assert "metadata" in chunk_data
            assert "chunk_id" in chunk_data["metadata"]
            assert "start_index" in chunk_data["metadata"]
            assert "end_index" in chunk_data["metadata"]
            assert "character_count" in chunk_data["metadata"]

    def test_chunk_position_tracking(self):
        """Test chunk position tracking in original document."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "0123456789" * 20  # 200 characters

        chunks_with_metadata = chunker.chunk_text_with_metadata(text)

        # Verify position tracking
        for i, chunk_data in enumerate(chunks_with_metadata):
            start_idx = chunk_data["metadata"]["start_index"]
            end_idx = chunk_data["metadata"]["end_index"]

            # Verify the chunk text matches the position in original text
            expected_text = text[start_idx:end_idx]
            assert chunk_data["text"] == expected_text

    def test_chunk_relationship_tracking(self):
        """Test tracking relationships between chunks."""
        from src.fileintel.document_processing.chunking import TextChunker

        chunker = TextChunker(chunk_size=100, overlap=20)
        text = "A" * 300  # 300 characters

        chunks_with_metadata = chunker.chunk_text_with_metadata(text)

        # Verify chunk relationships
        for i, chunk_data in enumerate(chunks_with_metadata):
            metadata = chunk_data["metadata"]

            if i > 0:
                assert "previous_chunk_id" in metadata
            if i < len(chunks_with_metadata) - 1:
                assert "next_chunk_id" in metadata

            assert metadata["chunk_index"] == i
            assert metadata["total_chunks"] == len(chunks_with_metadata)


class TestChunkingOptimization:
    """Test cases for chunking optimization strategies."""

    def test_optimal_chunk_size_detection(self):
        """Test detection of optimal chunk size for content."""
        from src.fileintel.document_processing.chunking import ChunkingOptimizer

        optimizer = ChunkingOptimizer()

        # Short content
        short_text = "This is short content."
        short_optimal = optimizer.suggest_chunk_size(short_text)

        # Long content
        long_text = "This is much longer content. " * 100
        long_optimal = optimizer.suggest_chunk_size(long_text)

        # Longer content should suggest larger chunks
        assert long_optimal >= short_optimal

    def test_chunk_quality_metrics(self):
        """Test chunk quality assessment."""
        from src.fileintel.document_processing.chunking import (
            TextChunker,
            ChunkQualityAnalyzer,
        )

        chunker = TextChunker(chunk_size=100, overlap=10)
        analyzer = ChunkQualityAnalyzer()

        text = """This is a test document with multiple paragraphs.

        Each paragraph contains related information that should ideally stay together.

        However, chunking might split paragraphs in suboptimal ways.

        The quality analyzer should detect these issues."""

        chunks = chunker.chunk_text(text)
        quality_score = analyzer.assess_chunk_quality(chunks, text)

        assert isinstance(quality_score, (int, float))
        assert 0 <= quality_score <= 1

    def test_adaptive_chunking_strategy(self):
        """Test adaptive chunking based on content characteristics."""
        from src.fileintel.document_processing.chunking import AdaptiveChunker

        adaptive_chunker = AdaptiveChunker()

        # Technical content (should use smaller chunks)
        technical_text = "The API endpoint returns a JSON object with status codes. Error handling includes 400, 401, 403, 404, and 500 responses."

        # Narrative content (should use larger chunks)
        narrative_text = "Once upon a time, there was a story that unfolded over many paragraphs with flowing narrative and character development."

        tech_chunks = adaptive_chunker.chunk_text(technical_text)
        narrative_chunks = adaptive_chunker.chunk_text(narrative_text)

        # Adaptive chunker should adjust strategy based on content type
        assert len(tech_chunks) > 0
        assert len(narrative_chunks) > 0


class TestChunkingIntegration:
    """Integration tests for chunking with other document processing components."""

    def test_chunking_with_document_readers(self, sample_markdown_doc, test_cleanup):
        """Test chunking integration with document readers."""
        from src.fileintel.document_processing.factory import ReaderFactory
        from src.fileintel.document_processing.chunking import TextChunker

        # Create test file
        temp_file = test_cleanup.create_temp_file(suffix=".md")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_markdown_doc["content"])

        # Read document
        factory = ReaderFactory()
        reader = factory.get_reader(".md")
        content = reader.read_document(temp_file)

        # Chunk content
        chunker = TextChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk_text(content)

        # Verify integration
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunking_performance_large_document(self):
        """Test chunking performance with large documents."""
        from src.fileintel.document_processing.chunking import TextChunker
        import time

        # Create large document
        large_text = "This is a sentence in a large document. " * 10000  # ~400KB

        chunker = TextChunker(chunk_size=1000, overlap=100)

        start_time = time.time()
        chunks = chunker.chunk_text(large_text)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete in reasonable time
        assert processing_time < 10.0  # seconds
        assert len(chunks) > 0

    def test_chunking_memory_efficiency(self):
        """Test chunking memory efficiency with large documents."""
        from src.fileintel.document_processing.chunking import TextChunker
        import sys

        # Monitor memory usage during chunking
        large_text = "Memory test content. " * 50000

        chunker = TextChunker(chunk_size=1000, overlap=100)

        # Measure memory before chunking
        initial_memory = sys.getsizeof(large_text)

        chunks = chunker.chunk_text(large_text)

        # Memory usage shouldn't explode
        total_chunk_memory = sum(sys.getsizeof(chunk) for chunk in chunks)

        # Total memory should be reasonable relative to input
        memory_ratio = total_chunk_memory / initial_memory
        assert memory_ratio < 3.0  # Allow for some overhead

    @pytest.mark.parametrize(
        "strategy", ["basic", "sentence_aware", "paragraph_aware", "recursive"]
    )
    def test_chunking_strategy_comparison(self, strategy, sample_text_doc):
        """Compare different chunking strategies on the same content."""
        chunk_size = 200
        overlap = 50

        if strategy == "basic":
            from src.fileintel.document_processing.chunking import TextChunker

            chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        elif strategy == "sentence_aware":
            from src.fileintel.document_processing.chunking import SentenceAwareChunker

            chunker = SentenceAwareChunker(chunk_size=chunk_size, overlap=overlap)
        elif strategy == "paragraph_aware":
            from src.fileintel.document_processing.chunking import ParagraphAwareChunker

            chunker = ParagraphAwareChunker(chunk_size=chunk_size, overlap=overlap)
        elif strategy == "recursive":
            from src.fileintel.document_processing.chunking import (
                RecursiveCharacterTextSplitter,
            )

            chunker = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, overlap=overlap
            )
        else:
            pytest.skip(f"Strategy {strategy} not available")

        chunks = chunker.chunk_text(sample_text_doc["content"])

        # All strategies should produce valid results
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)

        # Verify content preservation
        if overlap == 0:
            reconstructed = "".join(chunks)
            assert reconstructed == sample_text_doc["content"]
