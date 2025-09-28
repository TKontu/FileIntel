"""Comprehensive unit tests for document metadata processing and extraction."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import hashlib
import json

# Test fixtures
from tests.fixtures import (
    test_documents,
    sample_text_doc,
    sample_markdown_doc,
    test_document_files,
    test_cleanup,
    temporary_test_documents,
)


class TestBaseMetadataExtractor:
    """Test cases for base metadata extractor functionality."""

    def test_metadata_extractor_initialization(self):
        """Test metadata extractor initialization."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        extractor = MetadataExtractor()
        assert extractor is not None

    def test_metadata_extractor_interface(self):
        """Test metadata extractor defines required interface."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        extractor = MetadataExtractor()

        # Should have required methods
        assert hasattr(extractor, "extract_basic_metadata")
        assert hasattr(extractor, "extract_content_metadata")
        assert hasattr(extractor, "extract_file_metadata")

    def test_file_not_found_handling(self):
        """Test metadata extractor handles missing files gracefully."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        extractor = MetadataExtractor()

        with pytest.raises((FileNotFoundError, ValueError, IOError)):
            extractor.extract_basic_metadata(Path("/nonexistent/file.txt"))


class TestBasicMetadataExtraction:
    """Test cases for basic file metadata extraction."""

    def test_basic_file_metadata(self, sample_text_doc, test_cleanup):
        """Test extraction of basic file metadata."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        # Create test file
        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_text_doc["content"])

        extractor = MetadataExtractor()
        metadata = extractor.extract_basic_metadata(temp_file)

        # Verify basic metadata fields
        expected_fields = [
            "filename",
            "file_extension",
            "file_size",
            "created_at",
            "modified_at",
            "mime_type",
        ]

        for field in expected_fields:
            assert field in metadata, f"Missing field: {field}"

        # Verify values
        assert metadata["filename"] == temp_file.name
        assert metadata["file_extension"] == ".txt"
        assert metadata["file_size"] > 0
        assert metadata["mime_type"] in ["text/plain", "text/x-plain"]

    def test_file_timestamps(self, test_cleanup):
        """Test extraction of file timestamp metadata."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w") as f:
            f.write("test content")

        extractor = MetadataExtractor()
        metadata = extractor.extract_basic_metadata(temp_file)

        # Check timestamp fields
        assert "created_at" in metadata or "ctime" in metadata
        assert "modified_at" in metadata or "mtime" in metadata

        # Timestamps should be reasonable (recent)
        if "modified_at" in metadata:
            if isinstance(metadata["modified_at"], (int, float)):
                # Unix timestamp
                assert metadata["modified_at"] > 1609459200  # Jan 1, 2021
            elif isinstance(metadata["modified_at"], datetime):
                assert metadata["modified_at"].year >= 2021

    def test_file_size_accuracy(self, test_cleanup):
        """Test accuracy of file size metadata."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        content = "Test content with specific length"
        expected_size = len(content.encode("utf-8"))

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(content)

        extractor = MetadataExtractor()
        metadata = extractor.extract_basic_metadata(temp_file)

        assert metadata["file_size"] == expected_size

    def test_mime_type_detection(self, test_cleanup):
        """Test MIME type detection for various file types."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        extractor = MetadataExtractor()

        test_cases = [
            (".txt", "test content", "text/plain"),
            (".json", '{"key": "value"}', "application/json"),
            (".html", "<html><body>test</body></html>", "text/html"),
            (".xml", '<?xml version="1.0"?><root></root>', "application/xml"),
        ]

        for extension, content, expected_mime_prefix in test_cases:
            temp_file = test_cleanup.create_temp_file(suffix=extension)
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)

            metadata = extractor.extract_basic_metadata(temp_file)

            # MIME type detection might vary by system
            detected_mime = metadata.get("mime_type", "").lower()
            expected_prefix = expected_mime_prefix.split("/")[0]

            assert expected_prefix in detected_mime or detected_mime.startswith(
                expected_mime_prefix
            )


class TestContentMetadataExtraction:
    """Test cases for content-based metadata extraction."""

    def test_text_content_analysis(self, sample_text_doc, test_cleanup):
        """Test text content analysis and metadata extraction."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_text_doc["content"])

        extractor = MetadataExtractor()
        metadata = extractor.extract_content_metadata(
            temp_file, sample_text_doc["content"]
        )

        # Verify content metadata fields
        expected_fields = [
            "character_count",
            "word_count",
            "line_count",
            "paragraph_count",
            "language",
            "encoding",
        ]

        for field in expected_fields:
            if field in metadata:  # Some fields might be optional
                assert metadata[field] is not None

        # Verify counts
        if "character_count" in metadata:
            assert metadata["character_count"] == len(sample_text_doc["content"])

        if "word_count" in metadata:
            expected_words = len(sample_text_doc["content"].split())
            assert (
                abs(metadata["word_count"] - expected_words) <= 5
            )  # Allow small variance

    def test_language_detection(self, test_cleanup):
        """Test language detection from content."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        extractor = MetadataExtractor()

        language_tests = [
            ("This is English text content.", "en"),
            ("Esto es contenido en español.", "es"),
            ("Ceci est du contenu en français.", "fr"),
            ("Das ist deutscher Inhalt.", "de"),
        ]

        for content, expected_lang in language_tests:
            temp_file = test_cleanup.create_temp_file(suffix=".txt")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)

            try:
                metadata = extractor.extract_content_metadata(temp_file, content)

                if "language" in metadata:
                    detected_lang = metadata["language"]
                    # Language detection might not be exact, so check prefix
                    assert (
                        detected_lang.startswith(expected_lang)
                        or expected_lang in detected_lang
                    )

            except (ImportError, AttributeError):
                # Language detection might not be available
                pytest.skip("Language detection not available")

    def test_encoding_detection(self, test_cleanup):
        """Test text encoding detection."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        extractor = MetadataExtractor()

        # Test different encodings
        content = "Text with special characters: áéíóú ñç"
        encodings = ["utf-8", "latin-1", "ascii"]

        for encoding in encodings:
            try:
                temp_file = test_cleanup.create_temp_file(suffix=".txt")
                with open(temp_file, "w", encoding=encoding, errors="ignore") as f:
                    f.write(content)

                # Read back with detected encoding
                with open(temp_file, "rb") as f:
                    raw_content = f.read()

                detected_content = raw_content.decode(encoding, errors="ignore")
                metadata = extractor.extract_content_metadata(
                    temp_file, detected_content
                )

                if "encoding" in metadata:
                    # Encoding detection might be approximate
                    assert isinstance(metadata["encoding"], str)
                    assert len(metadata["encoding"]) > 0

            except (UnicodeDecodeError, UnicodeEncodeError):
                # Some encoding combinations might not work
                continue

    def test_structural_analysis(self, sample_markdown_doc, test_cleanup):
        """Test structural analysis of documents."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        temp_file = test_cleanup.create_temp_file(suffix=".md")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_markdown_doc["content"])

        extractor = MetadataExtractor()
        metadata = extractor.extract_content_metadata(
            temp_file, sample_markdown_doc["content"]
        )

        # Check for structural elements
        structural_fields = [
            "headers_count",
            "links_count",
            "code_blocks_count",
            "tables_count",
            "lists_count",
        ]

        for field in structural_fields:
            if field in metadata:
                assert isinstance(metadata[field], int)
                assert metadata[field] >= 0

    def test_content_hash_generation(self, sample_text_doc, test_cleanup):
        """Test content hash generation for deduplication."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_text_doc["content"])

        extractor = MetadataExtractor()
        metadata = extractor.extract_content_metadata(
            temp_file, sample_text_doc["content"]
        )

        # Check hash generation
        if "content_hash" in metadata:
            expected_hash = hashlib.sha256(
                sample_text_doc["content"].encode("utf-8")
            ).hexdigest()
            assert metadata["content_hash"] == expected_hash

        if "content_md5" in metadata:
            expected_md5 = hashlib.md5(
                sample_text_doc["content"].encode("utf-8")
            ).hexdigest()
            assert metadata["content_md5"] == expected_md5


class TestFormatSpecificMetadata:
    """Test cases for format-specific metadata extraction."""

    def test_pdf_metadata_extraction(self, test_cleanup):
        """Test PDF-specific metadata extraction."""
        from src.fileintel.document_processing.metadata_extractor import (
            PDFMetadataExtractor,
        )
        from tests.fixtures.test_documents import TestDocumentFixtures

        fixtures = TestDocumentFixtures()
        pdf_content = fixtures.create_test_pdf_content()

        temp_file = test_cleanup.create_temp_file(suffix=".pdf")
        with open(temp_file, "wb") as f:
            f.write(pdf_content)

        extractor = PDFMetadataExtractor()

        try:
            metadata = extractor.extract_pdf_metadata(temp_file)

            # Expected PDF metadata fields
            pdf_fields = [
                "page_count",
                "pdf_version",
                "title",
                "author",
                "creator",
                "producer",
            ]

            for field in pdf_fields:
                if field in metadata:
                    assert metadata[field] is not None

        except (ImportError, AttributeError):
            pytest.skip("PDF metadata extraction not available")

    def test_docx_metadata_extraction(self, test_cleanup):
        """Test DOCX-specific metadata extraction."""
        from src.fileintel.document_processing.metadata_extractor import (
            DOCXMetadataExtractor,
        )

        # This would typically require a real DOCX file
        # For now, test the interface
        try:
            extractor = DOCXMetadataExtractor()
            assert hasattr(extractor, "extract_docx_metadata")

        except ImportError:
            pytest.skip("DOCX metadata extraction not available")

    def test_image_metadata_extraction(self, test_cleanup):
        """Test image metadata extraction (if supported)."""
        from src.fileintel.document_processing.metadata_extractor import (
            ImageMetadataExtractor,
        )

        # Create minimal image data (placeholder)
        minimal_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"

        temp_file = test_cleanup.create_temp_file(suffix=".png")
        with open(temp_file, "wb") as f:
            f.write(minimal_png)

        try:
            extractor = ImageMetadataExtractor()
            metadata = extractor.extract_image_metadata(temp_file)

            # Expected image metadata
            image_fields = ["width", "height", "format", "mode", "has_transparency"]

            for field in image_fields:
                if field in metadata:
                    assert metadata[field] is not None

        except (ImportError, AttributeError):
            pytest.skip("Image metadata extraction not available")

    def test_json_metadata_extraction(self, test_cleanup):
        """Test JSON document metadata extraction."""
        from src.fileintel.document_processing.metadata_extractor import (
            JSONMetadataExtractor,
        )

        json_data = {
            "title": "Test JSON Document",
            "version": "1.0",
            "data": {"items": [1, 2, 3], "nested": {"key": "value"}},
            "metadata": {"created": "2024-01-01", "author": "Test"},
        }

        temp_file = test_cleanup.create_temp_file(suffix=".json")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        extractor = JSONMetadataExtractor()
        metadata = extractor.extract_json_metadata(temp_file)

        # JSON-specific metadata
        json_fields = [
            "json_structure",
            "top_level_keys",
            "nested_depth",
            "array_count",
            "object_count",
        ]

        for field in json_fields:
            if field in metadata:
                assert metadata[field] is not None

        # Verify structure analysis
        if "top_level_keys" in metadata:
            assert "title" in metadata["top_level_keys"]
            assert "data" in metadata["top_level_keys"]


class TestMetadataValidation:
    """Test cases for metadata validation and sanitization."""

    def test_metadata_schema_validation(self):
        """Test metadata schema validation."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataValidator,
        )

        validator = MetadataValidator()

        # Valid metadata
        valid_metadata = {
            "filename": "test.txt",
            "file_size": 1024,
            "mime_type": "text/plain",
            "created_at": datetime.now(),
            "character_count": 500,
        }

        assert validator.validate_metadata(valid_metadata) is True

        # Invalid metadata
        invalid_metadata = {
            "filename": "",  # Empty filename
            "file_size": -1,  # Negative size
            "mime_type": None,  # None value
        }

        assert validator.validate_metadata(invalid_metadata) is False

    def test_metadata_sanitization(self):
        """Test metadata sanitization and cleanup."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataSanitizer,
        )

        sanitizer = MetadataSanitizer()

        # Metadata with potentially problematic values
        raw_metadata = {
            "filename": "../../../etc/passwd",  # Path traversal
            "title": '<script>alert("xss")</script>',  # XSS attempt
            "author": "Normal Author",
            "file_size": "1024",  # String instead of int
            "tags": ["tag1", "tag2", "", None],  # Mixed valid/invalid
            "nested": {"key": "value", "empty": ""},
        }

        clean_metadata = sanitizer.sanitize_metadata(raw_metadata)

        # Verify sanitization
        assert "../" not in clean_metadata["filename"]
        assert "<script>" not in clean_metadata["title"]
        assert isinstance(clean_metadata["file_size"], int)
        assert "" not in clean_metadata["tags"]
        assert None not in clean_metadata["tags"]

    def test_metadata_type_coercion(self):
        """Test metadata type coercion and normalization."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        extractor = MetadataExtractor()

        # Test data with mixed types
        raw_data = {
            "file_size": "1024",  # String number
            "page_count": 5.0,  # Float integer
            "created_at": "2024-01-01T10:00:00Z",  # ISO string
            "tags": "tag1,tag2,tag3",  # Comma-separated string
            "is_encrypted": "true",  # String boolean
        }

        normalized = extractor.normalize_metadata_types(raw_data)

        assert isinstance(normalized["file_size"], int)
        assert isinstance(normalized["page_count"], int)
        assert isinstance(normalized["created_at"], (datetime, str))
        assert isinstance(normalized["tags"], list)
        assert isinstance(normalized["is_encrypted"], bool)


class TestMetadataAggregation:
    """Test cases for metadata aggregation and merging."""

    def test_metadata_merging(self):
        """Test merging metadata from multiple sources."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataAggregator,
        )

        aggregator = MetadataAggregator()

        # Basic file metadata
        file_metadata = {
            "filename": "document.pdf",
            "file_size": 2048,
            "mime_type": "application/pdf",
        }

        # Content metadata
        content_metadata = {
            "character_count": 5000,
            "word_count": 800,
            "language": "en",
        }

        # Format-specific metadata
        format_metadata = {
            "page_count": 10,
            "pdf_version": "1.4",
            "author": "Test Author",
        }

        merged = aggregator.merge_metadata(
            [file_metadata, content_metadata, format_metadata]
        )

        # All fields should be present
        all_fields = (
            set(file_metadata.keys())
            | set(content_metadata.keys())
            | set(format_metadata.keys())
        )
        assert all(field in merged for field in all_fields)

    def test_metadata_conflict_resolution(self):
        """Test handling of conflicting metadata values."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataAggregator,
        )

        aggregator = MetadataAggregator()

        # Conflicting metadata sources
        source1 = {"title": "Title from Source 1", "author": "Author 1"}
        source2 = {"title": "Title from Source 2", "description": "Description"}

        merged = aggregator.merge_metadata(
            [source1, source2], conflict_resolution="priority"
        )

        # Should prefer first source for conflicts
        assert merged["title"] == "Title from Source 1"
        assert merged["author"] == "Author 1"
        assert merged["description"] == "Description"

    def test_metadata_priority_system(self):
        """Test metadata priority system for conflict resolution."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataAggregator,
        )

        aggregator = MetadataAggregator()

        # Define priority levels
        low_priority = {"title": "Default Title", "priority": 1}
        high_priority = {"title": "Important Title", "priority": 3}
        medium_priority = {"title": "Medium Title", "priority": 2}

        merged = aggregator.merge_metadata_with_priority(
            [(low_priority, 1), (high_priority, 3), (medium_priority, 2)]
        )

        # Should use highest priority value
        assert merged["title"] == "Important Title"


class TestMetadataStorage:
    """Test cases for metadata storage and retrieval."""

    def test_metadata_serialization(self):
        """Test metadata serialization for storage."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataSerializer,
        )

        serializer = MetadataSerializer()

        metadata = {
            "filename": "test.txt",
            "created_at": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            "file_size": 1024,
            "tags": ["tag1", "tag2"],
            "nested": {"key": "value", "number": 42},
        }

        # Serialize to JSON
        json_data = serializer.to_json(metadata)
        assert isinstance(json_data, str)

        # Deserialize back
        restored_metadata = serializer.from_json(json_data)
        assert restored_metadata["filename"] == metadata["filename"]
        assert restored_metadata["file_size"] == metadata["file_size"]

    def test_metadata_database_storage(self):
        """Test metadata storage in database format."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataDBAdapter,
        )

        adapter = MetadataDBAdapter()

        metadata = {
            "document_id": "doc-123",
            "filename": "test.pdf",
            "file_size": 2048,
            "extraction_timestamp": datetime.utcnow(),
            "metadata_fields": {
                "title": "Test Document",
                "author": "Test Author",
                "page_count": 5,
            },
        }

        # Convert to database format
        db_format = adapter.to_database_format(metadata)

        assert "document_id" in db_format
        assert "metadata_json" in db_format or "metadata_fields" in db_format
        assert isinstance(db_format["extraction_timestamp"], (datetime, str))


class TestMetadataIntegration:
    """Integration tests for metadata extraction pipeline."""

    def test_end_to_end_metadata_extraction(self, test_document_files):
        """Test end-to-end metadata extraction process."""
        from src.fileintel.document_processing.factory import ReaderFactory
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        factory = ReaderFactory()
        extractor = MetadataExtractor()

        processed_files = []

        for file_path in test_document_files.glob("*"):
            if file_path.is_file():
                try:
                    # Extract basic metadata
                    basic_metadata = extractor.extract_basic_metadata(file_path)

                    # Read content
                    reader = factory.get_reader(file_path.suffix)
                    if reader:
                        content = reader.read_document(file_path)

                        # Extract content metadata
                        content_metadata = extractor.extract_content_metadata(
                            file_path, content
                        )

                        # Merge metadata
                        full_metadata = {**basic_metadata, **content_metadata}

                        processed_files.append(
                            {
                                "filename": file_path.name,
                                "metadata_fields": len(full_metadata),
                                "has_content_hash": "content_hash" in full_metadata,
                            }
                        )

                except Exception as e:
                    # Some files might not be processable
                    continue

        assert len(processed_files) > 0

    def test_metadata_consistency_across_runs(self, sample_text_doc, test_cleanup):
        """Test metadata consistency across multiple extraction runs."""
        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_text_doc["content"])

        extractor = MetadataExtractor()

        # Extract metadata multiple times
        metadata1 = extractor.extract_basic_metadata(temp_file)
        metadata2 = extractor.extract_basic_metadata(temp_file)

        # Non-time-dependent fields should be identical
        time_dependent_fields = {"accessed_at", "processed_at", "extraction_timestamp"}

        for key in metadata1:
            if key not in time_dependent_fields:
                assert (
                    metadata1[key] == metadata2[key]
                ), f"Inconsistent metadata for {key}"

    def test_metadata_performance(self, test_cleanup):
        """Test metadata extraction performance."""
        import time

        from src.fileintel.document_processing.metadata_extractor import (
            MetadataExtractor,
        )

        # Create larger test file
        large_content = "This is test content. " * 10000
        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(large_content)

        extractor = MetadataExtractor()

        start_time = time.time()
        metadata = extractor.extract_basic_metadata(temp_file)
        content_metadata = extractor.extract_content_metadata(temp_file, large_content)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete in reasonable time
        assert processing_time < 5.0  # seconds
        assert len(metadata) > 0
        assert len(content_metadata) > 0
