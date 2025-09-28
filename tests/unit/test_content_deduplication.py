"""Comprehensive unit tests for hash-based duplicate detection and content deduplication."""

import pytest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Set, Tuple, Optional
from datetime import datetime, timedelta
import json
import time

# Test fixtures
from tests.fixtures import (
    test_documents,
    sample_text_doc,
    sample_markdown_doc,
    test_document_files,
    test_cleanup,
    temporary_test_documents,
)


class TestContentHashGeneration:
    """Test cases for content hash generation and algorithms."""

    def test_sha256_hash_generation(self):
        """Test SHA-256 hash generation for content."""
        from src.fileintel.document_processing.deduplication import ContentHasher

        hasher = ContentHasher(algorithm="sha256")
        content = "This is test content for hashing."

        hash1 = hasher.generate_hash(content)
        hash2 = hasher.generate_hash(content)

        # Same content should produce identical hashes
        assert hash1 == hash2

        # Hash should be correct length for SHA-256
        assert len(hash1) == 64  # 32 bytes * 2 hex chars

        # Hash should be deterministic
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert hash1 == expected_hash

    def test_md5_hash_generation(self):
        """Test MD5 hash generation for content."""
        from src.fileintel.document_processing.deduplication import ContentHasher

        hasher = ContentHasher(algorithm="md5")
        content = "Test content for MD5 hashing."

        hash1 = hasher.generate_hash(content)
        hash2 = hasher.generate_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 32  # 16 bytes * 2 hex chars

        expected_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        assert hash1 == expected_hash

    def test_different_algorithms(self):
        """Test different hashing algorithms produce different results."""
        from src.fileintel.document_processing.deduplication import ContentHasher

        content = "Same content, different algorithms."

        sha256_hasher = ContentHasher(algorithm="sha256")
        md5_hasher = ContentHasher(algorithm="md5")

        sha256_hash = sha256_hasher.generate_hash(content)
        md5_hash = md5_hasher.generate_hash(content)

        assert sha256_hash != md5_hash
        assert len(sha256_hash) > len(md5_hash)

    def test_empty_content_hashing(self):
        """Test hashing of empty content."""
        from src.fileintel.document_processing.deduplication import ContentHasher

        hasher = ContentHasher()
        empty_hash = hasher.generate_hash("")

        assert isinstance(empty_hash, str)
        assert len(empty_hash) > 0

        # Should be consistent
        assert empty_hash == hasher.generate_hash("")

    def test_unicode_content_hashing(self):
        """Test hashing of Unicode content."""
        from src.fileintel.document_processing.deduplication import ContentHasher

        hasher = ContentHasher()

        unicode_content = "Test with Unicode: áéíóú ñç 中文 🎉"
        hash1 = hasher.generate_hash(unicode_content)
        hash2 = hasher.generate_hash(unicode_content)

        assert hash1 == hash2
        assert len(hash1) > 0

    def test_large_content_hashing(self):
        """Test hashing of large content blocks."""
        from src.fileintel.document_processing.deduplication import ContentHasher

        hasher = ContentHasher()

        # Create large content (1MB)
        large_content = "Large content block. " * 50000

        start_time = time.time()
        content_hash = hasher.generate_hash(large_content)
        end_time = time.time()

        # Should complete in reasonable time
        assert (end_time - start_time) < 5.0
        assert len(content_hash) > 0

        # Should be consistent
        assert content_hash == hasher.generate_hash(large_content)


class TestFileHashGeneration:
    """Test cases for file-based hash generation."""

    def test_file_content_hashing(self, sample_text_doc, test_cleanup):
        """Test generating hash directly from file content."""
        from src.fileintel.document_processing.deduplication import FileHasher

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_text_doc["content"])

        hasher = FileHasher()
        file_hash = hasher.generate_file_hash(temp_file)

        # Should match content hash
        expected_hash = hashlib.sha256(
            sample_text_doc["content"].encode("utf-8")
        ).hexdigest()
        assert file_hash == expected_hash

    def test_binary_file_hashing(self, test_cleanup):
        """Test hashing of binary files."""
        from src.fileintel.document_processing.deduplication import FileHasher
        from tests.fixtures.test_documents import TestDocumentFixtures

        fixtures = TestDocumentFixtures()
        pdf_content = fixtures.create_test_pdf_content()

        temp_file = test_cleanup.create_temp_file(suffix=".pdf")
        with open(temp_file, "wb") as f:
            f.write(pdf_content)

        hasher = FileHasher()
        file_hash = hasher.generate_file_hash(temp_file)

        # Should match binary content hash
        expected_hash = hashlib.sha256(pdf_content).hexdigest()
        assert file_hash == expected_hash

    def test_nonexistent_file_handling(self):
        """Test handling of non-existent files."""
        from src.fileintel.document_processing.deduplication import FileHasher

        hasher = FileHasher()

        with pytest.raises((FileNotFoundError, IOError, ValueError)):
            hasher.generate_file_hash(Path("/nonexistent/file.txt"))

    def test_file_modification_detection(self, test_cleanup):
        """Test detection of file modifications through hash changes."""
        from src.fileintel.document_processing.deduplication import FileHasher

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        hasher = FileHasher()

        # Write initial content
        initial_content = "Initial content"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(initial_content)

        initial_hash = hasher.generate_file_hash(temp_file)

        # Modify content
        modified_content = "Modified content"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(modified_content)

        modified_hash = hasher.generate_file_hash(temp_file)

        # Hashes should be different
        assert initial_hash != modified_hash

    def test_file_permissions_handling(self, test_cleanup):
        """Test handling of files with different permissions."""
        from src.fileintel.document_processing.deduplication import FileHasher

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("Content with permissions test")

        hasher = FileHasher()

        # Should work with normal permissions
        normal_hash = hasher.generate_file_hash(temp_file)
        assert len(normal_hash) > 0

        # Test with different permissions (if supported by OS)
        try:
            import os

            os.chmod(temp_file, 0o444)  # Read-only
            readonly_hash = hasher.generate_file_hash(temp_file)

            # Hash should be the same regardless of permissions
            assert normal_hash == readonly_hash

        except (OSError, AttributeError):
            # Permission changes might not be supported
            pass


class TestDuplicateDetection:
    """Test cases for duplicate content detection."""

    def test_exact_duplicate_detection(self):
        """Test detection of exact duplicate content."""
        from src.fileintel.document_processing.deduplication import DuplicateDetector

        detector = DuplicateDetector()

        content1 = "This is the same content."
        content2 = "This is the same content."
        content3 = "This is different content."

        # Add content to detector
        doc1_hash = detector.add_content("doc1", content1)
        doc2_hash = detector.add_content("doc2", content2)
        doc3_hash = detector.add_content("doc3", content3)

        # Check for duplicates
        duplicates = detector.find_duplicates()

        # Should detect doc1 and doc2 as duplicates
        assert len(duplicates) > 0
        duplicate_group = None

        for group in duplicates:
            if "doc1" in group and "doc2" in group:
                duplicate_group = group
                break

        assert duplicate_group is not None
        assert "doc3" not in duplicate_group

    def test_near_duplicate_detection(self):
        """Test detection of near-duplicate content."""
        from src.fileintel.document_processing.deduplication import (
            NearDuplicateDetector,
        )

        detector = NearDuplicateDetector(similarity_threshold=0.8)

        content1 = "The quick brown fox jumps over the lazy dog."
        content2 = "The quick brown fox jumps over the lazy dog!"  # Minor difference
        content3 = "Completely different text about something else entirely."

        detector.add_content("doc1", content1)
        detector.add_content("doc2", content2)
        detector.add_content("doc3", content3)

        near_duplicates = detector.find_near_duplicates()

        # Should detect doc1 and doc2 as near-duplicates
        assert len(near_duplicates) > 0

        # Find the near-duplicate group containing doc1 and doc2
        found_group = False
        for group in near_duplicates:
            if "doc1" in group and "doc2" in group:
                found_group = True
                assert "doc3" not in group
                break

        assert found_group

    def test_shingling_based_detection(self):
        """Test shingling-based duplicate detection."""
        from src.fileintel.document_processing.deduplication import ShinglingDetector

        detector = ShinglingDetector(shingle_size=3)

        content1 = "The quick brown fox jumps over the lazy dog."
        content2 = (
            "The quick brown fox jumps over a lazy dog."  # Similar with shingling
        )
        content3 = "Purple elephants dance in the moonlight gracefully."

        detector.add_content("doc1", content1)
        detector.add_content("doc2", content2)
        detector.add_content("doc3", content3)

        similar_docs = detector.find_similar_documents(similarity_threshold=0.5)

        # Should find doc1 and doc2 as similar
        assert len(similar_docs) > 0

        for doc_id, similar_list in similar_docs.items():
            if doc_id == "doc1":
                assert "doc2" in similar_list
                assert "doc3" not in similar_list

    def test_minhash_based_detection(self):
        """Test MinHash-based duplicate detection for large-scale deduplication."""
        from src.fileintel.document_processing.deduplication import MinHashDetector

        detector = MinHashDetector(num_perm=128)

        documents = {
            "doc1": "The quick brown fox jumps over the lazy dog. This is a test document.",
            "doc2": "The quick brown fox jumps over the lazy dog. This is a test document.",  # Exact duplicate
            "doc3": "The quick brown fox leaps over the lazy dog. This is a test document.",  # Similar
            "doc4": "Completely different content about cats and their behaviors in nature.",  # Different
            "doc5": "Another unique document with different content and structure entirely.",
        }

        for doc_id, content in documents.items():
            detector.add_content(doc_id, content)

        # Find duplicates with high threshold
        exact_duplicates = detector.find_duplicates(threshold=0.95)
        assert len(exact_duplicates) > 0

        # Find similar documents with lower threshold
        similar_docs = detector.find_duplicates(threshold=0.7)
        assert len(similar_docs) >= len(exact_duplicates)


class TestDocumentDeduplication:
    """Test cases for document-level deduplication."""

    def test_document_fingerprinting(self, test_cleanup):
        """Test document fingerprinting for deduplication."""
        from src.fileintel.document_processing.deduplication import (
            DocumentFingerprinter,
        )

        fingerprinter = DocumentFingerprinter()

        # Create test documents
        doc1_content = "Document 1 content with some unique text."
        doc2_content = "Document 1 content with some unique text."  # Identical
        doc3_content = "Document 2 with completely different content."

        doc1_file = test_cleanup.create_temp_file(suffix=".txt")
        doc2_file = test_cleanup.create_temp_file(suffix=".txt")
        doc3_file = test_cleanup.create_temp_file(suffix=".txt")

        with open(doc1_file, "w") as f:
            f.write(doc1_content)
        with open(doc2_file, "w") as f:
            f.write(doc2_content)
        with open(doc3_file, "w") as f:
            f.write(doc3_content)

        # Generate fingerprints
        fp1 = fingerprinter.generate_fingerprint(doc1_file)
        fp2 = fingerprinter.generate_fingerprint(doc2_file)
        fp3 = fingerprinter.generate_fingerprint(doc3_file)

        # Identical documents should have identical fingerprints
        assert fp1 == fp2
        assert fp1 != fp3

    def test_document_similarity_scoring(self):
        """Test document similarity scoring."""
        from src.fileintel.document_processing.deduplication import (
            DocumentSimilarityScorer,
        )

        scorer = DocumentSimilarityScorer()

        doc1 = "The weather today is sunny and warm with clear skies."
        doc2 = "Today's weather is sunny and warm with clear skies."  # Very similar
        doc3 = (
            "The weather yesterday was rainy and cold with clouds."  # Somewhat similar
        )
        doc4 = "Programming languages have different syntax and features."  # Different

        # Calculate similarity scores
        score_1_2 = scorer.calculate_similarity(doc1, doc2)
        score_1_3 = scorer.calculate_similarity(doc1, doc3)
        score_1_4 = scorer.calculate_similarity(doc1, doc4)

        # Similarity should decrease as expected
        assert score_1_2 > score_1_3
        assert score_1_3 > score_1_4
        assert 0 <= score_1_4 <= 1

    def test_batch_duplicate_detection(self, test_cleanup):
        """Test batch processing for duplicate detection."""
        from src.fileintel.document_processing.deduplication import (
            BatchDuplicateDetector,
        )

        detector = BatchDuplicateDetector()

        # Create multiple test files
        documents = [
            ("doc1.txt", "First unique document content."),
            ("doc2.txt", "First unique document content."),  # Duplicate of doc1
            ("doc3.txt", "Second unique document content."),
            ("doc4.txt", "Third unique document content."),
            ("doc5.txt", "Second unique document content."),  # Duplicate of doc3
        ]

        file_paths = []
        for filename, content in documents:
            temp_file = test_cleanup.create_temp_file(suffix=f"_{filename}")
            with open(temp_file, "w") as f:
                f.write(content)
            file_paths.append(temp_file)

        # Detect duplicates in batch
        duplicate_groups = detector.find_duplicates_in_batch(file_paths)

        assert len(duplicate_groups) == 2  # Two groups of duplicates

        # Verify correct grouping
        found_groups = 0
        for group in duplicate_groups:
            if len(group) == 2:  # Each group should have 2 files
                found_groups += 1

        assert found_groups == 2

    def test_incremental_duplicate_detection(self):
        """Test incremental duplicate detection as new documents arrive."""
        from src.fileintel.document_processing.deduplication import (
            IncrementalDuplicateDetector,
        )

        detector = IncrementalDuplicateDetector()

        # Add documents incrementally
        doc1_id = "doc1"
        doc1_content = "Original document content."
        detector.add_document(doc1_id, doc1_content)

        # Check no duplicates initially
        duplicates = detector.get_duplicates()
        assert len(duplicates) == 0

        # Add a duplicate
        doc2_id = "doc2"
        doc2_content = "Original document content."  # Same as doc1
        is_duplicate = detector.add_document(doc2_id, doc2_content)

        assert is_duplicate is True

        # Check duplicates found
        duplicates = detector.get_duplicates()
        assert len(duplicates) > 0

        # Add a unique document
        doc3_id = "doc3"
        doc3_content = "Completely different document content."
        is_duplicate = detector.add_document(doc3_id, doc3_content)

        assert is_duplicate is False


class TestDeduplicationStrategies:
    """Test cases for different deduplication strategies."""

    def test_exact_match_strategy(self):
        """Test exact match deduplication strategy."""
        from src.fileintel.document_processing.deduplication import (
            ExactMatchDeduplicator,
        )

        deduplicator = ExactMatchDeduplicator()

        documents = {
            "doc1": "Exact same content.",
            "doc2": "Exact same content.",
            "doc3": "Different content.",
            "doc4": "Exact same content.",
        }

        for doc_id, content in documents.items():
            deduplicator.add_document(doc_id, content)

        unique_docs = deduplicator.get_unique_documents()

        # Should have only 2 unique documents
        assert len(unique_docs) == 2

        duplicate_groups = deduplicator.get_duplicate_groups()
        assert len(duplicate_groups) == 1  # One group with 3 duplicates

        # The group should contain doc1, doc2, and doc4
        duplicate_group = duplicate_groups[0]
        assert len(duplicate_group) == 3
        assert "doc1" in duplicate_group
        assert "doc2" in duplicate_group
        assert "doc4" in duplicate_group

    def test_fuzzy_match_strategy(self):
        """Test fuzzy match deduplication strategy."""
        from src.fileintel.document_processing.deduplication import (
            FuzzyMatchDeduplicator,
        )

        deduplicator = FuzzyMatchDeduplicator(similarity_threshold=0.8)

        documents = {
            "doc1": "The quick brown fox jumps over the lazy dog.",
            "doc2": "The quick brown fox jumps over the lazy dog!",  # Very similar
            "doc3": "A quick brown fox jumps over a lazy dog.",  # Similar
            "doc4": "Purple elephants dance in the moonlight.",  # Different
        }

        for doc_id, content in documents.items():
            deduplicator.add_document(doc_id, content)

        similar_groups = deduplicator.get_similar_groups()

        # Should find at least one group of similar documents
        assert len(similar_groups) > 0

        # Check that doc1, doc2, and doc3 are grouped together
        large_group = max(similar_groups, key=len)
        assert len(large_group) >= 2

    def test_content_based_strategy(self):
        """Test content-based deduplication strategy."""
        from src.fileintel.document_processing.deduplication import (
            ContentBasedDeduplicator,
        )

        deduplicator = ContentBasedDeduplicator()

        # Documents with same content but different metadata
        documents = [
            {"id": "doc1", "content": "Same content", "metadata": {"title": "Title 1"}},
            {"id": "doc2", "content": "Same content", "metadata": {"title": "Title 2"}},
            {
                "id": "doc3",
                "content": "Different content",
                "metadata": {"title": "Title 3"},
            },
        ]

        for doc in documents:
            deduplicator.add_document(doc["id"], doc["content"], doc["metadata"])

        # Should detect content duplicates despite different metadata
        content_duplicates = deduplicator.find_content_duplicates()
        assert len(content_duplicates) > 0

        # Should preserve unique content
        unique_content = deduplicator.get_unique_content()
        assert len(unique_content) == 2  # Two unique content pieces

    def test_metadata_aware_strategy(self):
        """Test metadata-aware deduplication strategy."""
        from src.fileintel.document_processing.deduplication import (
            MetadataAwareDeduplicator,
        )

        deduplicator = MetadataAwareDeduplicator()

        documents = [
            {
                "id": "doc1",
                "content": "Same content",
                "filename": "file1.txt",
                "author": "Author A",
            },
            {
                "id": "doc2",
                "content": "Same content",
                "filename": "file1.txt",
                "author": "Author A",
            },  # Complete duplicate
            {
                "id": "doc3",
                "content": "Same content",
                "filename": "file2.txt",
                "author": "Author B",
            },  # Same content, different metadata
        ]

        for doc in documents:
            deduplicator.add_document(
                doc["id"],
                doc["content"],
                {"filename": doc["filename"], "author": doc["author"]},
            )

        # Should distinguish between complete duplicates and content-only duplicates
        complete_duplicates = deduplicator.find_complete_duplicates()
        content_duplicates = deduplicator.find_content_duplicates()

        assert len(complete_duplicates) > 0  # doc1 and doc2
        assert len(content_duplicates) > len(
            complete_duplicates
        )  # All three have same content


class TestDeduplicationPerformance:
    """Test cases for deduplication performance and scalability."""

    def test_large_scale_deduplication(self):
        """Test deduplication performance with large number of documents."""
        from src.fileintel.document_processing.deduplication import ScalableDeduplicator
        import time

        deduplicator = ScalableDeduplicator()

        # Create large number of documents with some duplicates
        num_docs = 1000
        documents = {}

        for i in range(num_docs):
            if i % 10 == 0:  # Every 10th document is a duplicate of the first
                content = "Duplicate content that appears multiple times."
            else:
                content = (
                    f"Unique content for document number {i} with specific details."
                )

            documents[f"doc_{i}"] = content

        # Measure deduplication time
        start_time = time.time()

        for doc_id, content in documents.items():
            deduplicator.add_document(doc_id, content)

        duplicates = deduplicator.find_duplicates()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete in reasonable time
        assert processing_time < 10.0  # seconds

        # Should find the expected duplicates
        assert len(duplicates) > 0

        # Should find approximately 100 duplicate documents (every 10th)
        total_duplicates = sum(len(group) for group in duplicates)
        assert 90 <= total_duplicates <= 110  # Allow some variance

    def test_memory_efficient_deduplication(self):
        """Test memory-efficient deduplication for large documents."""
        from src.fileintel.document_processing.deduplication import (
            MemoryEfficientDeduplicator,
        )
        import sys

        deduplicator = MemoryEfficientDeduplicator()

        # Create documents with varying sizes
        small_doc = "Small document content."
        medium_doc = "Medium document content. " * 1000
        large_doc = "Large document content. " * 10000

        documents = {
            "small": small_doc,
            "medium": medium_doc,
            "large": large_doc,
            "small_dup": small_doc,  # Duplicate of small
            "medium_dup": medium_doc,  # Duplicate of medium
        }

        # Monitor memory usage
        initial_memory = sys.getsizeof(deduplicator)

        for doc_id, content in documents.items():
            deduplicator.add_document(doc_id, content)

        final_memory = sys.getsizeof(deduplicator)
        duplicates = deduplicator.find_duplicates()

        # Memory usage should be reasonable
        memory_growth = final_memory - initial_memory
        total_content_size = sum(
            sys.getsizeof(content) for content in documents.values()
        )

        # Memory growth should be much less than total content size
        assert memory_growth < total_content_size * 0.5

        # Should still find duplicates correctly
        assert len(duplicates) == 2  # Two duplicate pairs

    def test_streaming_deduplication(self):
        """Test streaming deduplication for continuous document processing."""
        from src.fileintel.document_processing.deduplication import (
            StreamingDeduplicator,
        )

        deduplicator = StreamingDeduplicator()

        # Simulate streaming documents
        document_stream = [
            ("doc1", "First document content."),
            ("doc2", "Second document content."),
            ("doc3", "First document content."),  # Duplicate of doc1
            ("doc4", "Third document content."),
            ("doc5", "Second document content."),  # Duplicate of doc2
        ]

        duplicates_found = []

        for doc_id, content in document_stream:
            is_duplicate, duplicate_of = deduplicator.process_document(doc_id, content)

            if is_duplicate:
                duplicates_found.append((doc_id, duplicate_of))

        # Should find two duplicates
        assert len(duplicates_found) == 2
        assert ("doc3", "doc1") in duplicates_found
        assert ("doc5", "doc2") in duplicates_found


class TestDeduplicationIntegration:
    """Integration tests for deduplication with other FileIntel components."""

    def test_deduplication_with_document_processing(self, test_document_files):
        """Test deduplication integration with document processing pipeline."""
        from src.fileintel.document_processing.factory import ReaderFactory
        from src.fileintel.document_processing.deduplication import (
            DocumentDeduplicationPipeline,
        )

        factory = ReaderFactory()
        pipeline = DocumentDeduplicationPipeline()

        processed_documents = []

        # Process all test documents
        for file_path in test_document_files.glob("*"):
            if file_path.is_file():
                try:
                    reader = factory.get_reader(file_path.suffix)
                    if reader:
                        content = reader.read_document(file_path)

                        doc_info = {
                            "id": f"doc_{file_path.name}",
                            "filename": file_path.name,
                            "content": content,
                            "file_path": str(file_path),
                        }

                        processed_documents.append(doc_info)

                except Exception:
                    continue

        # Add documents to deduplication pipeline
        for doc in processed_documents:
            pipeline.add_document(
                doc["id"],
                doc["content"],
                {"filename": doc["filename"], "file_path": doc["file_path"]},
            )

        # Find duplicates
        duplicates = pipeline.find_duplicates()

        # Verify pipeline worked
        assert isinstance(duplicates, list)

    def test_deduplication_with_storage_layer(self):
        """Test deduplication integration with storage layer."""
        from src.fileintel.document_processing.deduplication import (
            StorageAwareDeduplicator,
        )

        # Mock storage interface
        mock_storage = Mock()
        mock_storage.get_document_by_hash.return_value = None  # No existing documents

        deduplicator = StorageAwareDeduplicator(storage=mock_storage)

        # Add documents
        doc1_content = "Document content for storage test."
        doc2_content = "Document content for storage test."  # Duplicate

        result1 = deduplicator.add_document("doc1", doc1_content)
        result2 = deduplicator.add_document("doc2", doc2_content)

        # First document should be added, second should be marked as duplicate
        assert result1.is_duplicate is False
        assert result2.is_duplicate is True

        # Storage should be queried for existing hashes
        assert mock_storage.get_document_by_hash.called

    def test_deduplication_configuration(self):
        """Test deduplication with different configuration options."""
        from src.fileintel.document_processing.deduplication import (
            ConfigurableDeduplicator,
        )

        # Test with strict configuration
        strict_config = {
            "algorithm": "sha256",
            "similarity_threshold": 0.95,
            "enable_fuzzy_matching": False,
            "min_document_size": 10,
        }

        strict_deduplicator = ConfigurableDeduplicator(strict_config)

        # Test with lenient configuration
        lenient_config = {
            "algorithm": "md5",
            "similarity_threshold": 0.7,
            "enable_fuzzy_matching": True,
            "min_document_size": 1,
        }

        lenient_deduplicator = ConfigurableDeduplicator(lenient_config)

        test_docs = {
            "doc1": "The quick brown fox jumps over the lazy dog.",
            "doc2": "The quick brown fox jumps over the lazy dog!",  # Minor difference
        }

        # Add to both deduplicators
        for doc_id, content in test_docs.items():
            strict_deduplicator.add_document(doc_id, content)
            lenient_deduplicator.add_document(doc_id, content)

        strict_duplicates = strict_deduplicator.find_duplicates()
        lenient_duplicates = lenient_deduplicator.find_duplicates()

        # Lenient should find more similarities
        assert len(lenient_duplicates) >= len(strict_duplicates)

    def test_deduplication_reporting(self):
        """Test deduplication reporting and statistics."""
        from src.fileintel.document_processing.deduplication import (
            DeduplicationReporter,
        )

        reporter = DeduplicationReporter()

        # Add sample deduplication results
        reporter.record_duplicate_group(["doc1", "doc2", "doc3"])  # Group of 3
        reporter.record_duplicate_group(["doc4", "doc5"])  # Group of 2
        reporter.record_unique_document("doc6")
        reporter.record_unique_document("doc7")

        # Generate report
        report = reporter.generate_report()

        assert report["total_documents"] == 7
        assert report["unique_documents"] == 2
        assert report["duplicate_documents"] == 5
        assert report["duplicate_groups"] == 2
        assert report["largest_duplicate_group"] == 3
        assert report["deduplication_ratio"] > 0

    def test_real_time_deduplication_monitoring(self):
        """Test real-time monitoring of deduplication process."""
        from src.fileintel.document_processing.deduplication import (
            MonitoredDeduplicator,
        )

        # Mock monitoring callback
        monitor_events = []

        def monitoring_callback(event_type, data):
            monitor_events.append({"type": event_type, "data": data})

        deduplicator = MonitoredDeduplicator(monitor_callback=monitoring_callback)

        # Add documents
        deduplicator.add_document("doc1", "First document.")
        deduplicator.add_document("doc2", "First document.")  # Duplicate
        deduplicator.add_document("doc3", "Second document.")

        # Check monitoring events were generated
        assert len(monitor_events) >= 3

        event_types = [event["type"] for event in monitor_events]
        assert "document_added" in event_types
        assert "duplicate_detected" in event_types
