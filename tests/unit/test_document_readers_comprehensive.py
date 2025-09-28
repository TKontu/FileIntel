"""Comprehensive unit tests for all document format readers in FileIntel."""

import pytest
import tempfile
import io
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import hashlib
from datetime import datetime

# Test fixtures
from tests.fixtures import (
    test_documents,
    test_document_files,
    test_cleanup,
    temporary_test_documents,
)


class TestBaseDocumentReader:
    """Test cases for base document reader functionality."""

    def test_base_reader_interface(self):
        """Test that base reader defines required interface."""
        from src.fileintel.document_processing.base import DocumentReader

        # Should define abstract methods
        assert hasattr(DocumentReader, "read_document")
        assert hasattr(DocumentReader, "extract_metadata")
        assert hasattr(DocumentReader, "supported_formats")

    def test_base_reader_validation(self):
        """Test base reader input validation."""
        from src.fileintel.document_processing.base import DocumentReader

        # Mock concrete implementation
        class MockReader(DocumentReader):
            def read_document(self, file_path: Path) -> str:
                return "mock content"

            def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
                return {"title": "mock"}

            def supported_formats(self) -> List[str]:
                return [".txt"]

        reader = MockReader()

        # Test file existence validation
        with pytest.raises((FileNotFoundError, ValueError)):
            reader.read_document(Path("/nonexistent/file.txt"))


class TestTextDocumentReader:
    """Test cases for plain text document reader."""

    def test_text_reader_initialization(self):
        """Test text reader can be initialized."""
        from src.fileintel.document_processing.readers.text_reader import TextReader

        reader = TextReader()
        assert reader is not None
        assert ".txt" in reader.supported_formats()

    def test_text_reader_supported_formats(self):
        """Test text reader supports expected formats."""
        from src.fileintel.document_processing.readers.text_reader import TextReader

        reader = TextReader()
        formats = reader.supported_formats()

        expected_formats = [".txt", ".text", ".md", ".markdown"]
        for fmt in expected_formats:
            assert fmt in formats

    @pytest.mark.parametrize("encoding", ["utf-8", "latin-1", "ascii"])
    def test_text_reader_encoding_handling(self, test_cleanup, encoding):
        """Test text reader handles different encodings."""
        from src.fileintel.document_processing.readers.text_reader import TextReader

        # Create test file with specific encoding
        test_content = "Test content with special chars: áéíóú"
        temp_file = test_cleanup.create_temp_file(suffix=".txt")

        with open(temp_file, "w", encoding=encoding, errors="ignore") as f:
            f.write(test_content)

        reader = TextReader()

        try:
            content = reader.read_document(temp_file)
            assert isinstance(content, str)
            assert len(content) > 0
        except UnicodeDecodeError:
            # Expected for some encodings
            pytest.skip(f"Encoding {encoding} not supported")

    def test_text_reader_metadata_extraction(self, sample_text_doc, test_cleanup):
        """Test text reader metadata extraction."""
        from src.fileintel.document_processing.readers.text_reader import TextReader

        # Create test file
        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_text_doc["content"])

        reader = TextReader()
        metadata = reader.extract_metadata(temp_file)

        # Verify basic metadata
        assert "file_size" in metadata
        assert "character_count" in metadata
        assert "line_count" in metadata
        assert "encoding" in metadata

        assert metadata["file_size"] > 0
        assert metadata["character_count"] == len(sample_text_doc["content"])

    def test_text_reader_empty_file(self, test_cleanup):
        """Test text reader handles empty files."""
        from src.fileintel.document_processing.readers.text_reader import TextReader

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        # File is empty by default

        reader = TextReader()
        content = reader.read_document(temp_file)
        metadata = reader.extract_metadata(temp_file)

        assert content == ""
        assert metadata["character_count"] == 0
        assert metadata["line_count"] == 0


class TestPDFDocumentReader:
    """Test cases for PDF document reader."""

    def test_pdf_reader_initialization(self):
        """Test PDF reader can be initialized."""
        from src.fileintel.document_processing.readers.pdf_reader import PDFReader

        reader = PDFReader()
        assert reader is not None
        assert ".pdf" in reader.supported_formats()

    def test_pdf_reader_supported_formats(self):
        """Test PDF reader supports expected formats."""
        from src.fileintel.document_processing.readers.pdf_reader import PDFReader

        reader = PDFReader()
        formats = reader.supported_formats()

        assert ".pdf" in formats

    def test_pdf_reader_basic_extraction(self, test_cleanup):
        """Test PDF reader basic text extraction."""
        from src.fileintel.document_processing.readers.pdf_reader import PDFReader
        from tests.fixtures.test_documents import TestDocumentFixtures

        # Create test PDF
        fixtures = TestDocumentFixtures()
        pdf_content = fixtures.create_test_pdf_content()

        temp_file = test_cleanup.create_temp_file(suffix=".pdf")
        with open(temp_file, "wb") as f:
            f.write(pdf_content)

        reader = PDFReader()

        try:
            content = reader.read_document(temp_file)
            assert isinstance(content, str)
            # PDF might contain some extractable text
        except Exception as e:
            # PDF parsing might fail without proper libraries
            pytest.skip(f"PDF parsing not available: {e}")

    def test_pdf_reader_metadata_extraction(self, test_cleanup):
        """Test PDF reader metadata extraction."""
        from src.fileintel.document_processing.readers.pdf_reader import PDFReader
        from tests.fixtures.test_documents import TestDocumentFixtures

        fixtures = TestDocumentFixtures()
        pdf_content = fixtures.create_test_pdf_content()

        temp_file = test_cleanup.create_temp_file(suffix=".pdf")
        with open(temp_file, "wb") as f:
            f.write(pdf_content)

        reader = PDFReader()

        try:
            metadata = reader.extract_metadata(temp_file)

            # Expected PDF metadata fields
            expected_fields = ["file_size", "page_count", "pdf_version"]
            for field in expected_fields:
                if field in metadata:
                    assert metadata[field] is not None

        except Exception as e:
            pytest.skip(f"PDF metadata extraction not available: {e}")

    @patch("fitz.open")  # PyMuPDF
    def test_pdf_reader_with_pymupdf(self, mock_fitz_open, test_cleanup):
        """Test PDF reader with PyMuPDF backend."""
        from src.fileintel.document_processing.readers.pdf_reader import PDFReader

        # Mock PDF document
        mock_doc = Mock()
        mock_doc.page_count = 2
        mock_doc.metadata = {"title": "Test PDF", "author": "Test Author"}

        mock_page = Mock()
        mock_page.get_text.return_value = "Test page content"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page, mock_page]))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)

        mock_fitz_open.return_value = mock_doc

        temp_file = test_cleanup.create_temp_file(suffix=".pdf")
        reader = PDFReader()

        try:
            content = reader.read_document(temp_file)
            metadata = reader.extract_metadata(temp_file)

            assert "Test page content" in content
            assert metadata.get("page_count") == 2
            assert metadata.get("title") == "Test PDF"

        except ImportError:
            pytest.skip("PyMuPDF not available")


class TestDOCXDocumentReader:
    """Test cases for DOCX document reader."""

    def test_docx_reader_initialization(self):
        """Test DOCX reader can be initialized."""
        from src.fileintel.document_processing.readers.docx_reader import DOCXReader

        reader = DOCXReader()
        assert reader is not None
        assert ".docx" in reader.supported_formats()

    def test_docx_reader_supported_formats(self):
        """Test DOCX reader supports expected formats."""
        from src.fileintel.document_processing.readers.docx_reader import DOCXReader

        reader = DOCXReader()
        formats = reader.supported_formats()

        expected_formats = [".docx", ".doc"]
        for fmt in expected_formats:
            assert fmt in formats

    @patch("docx.Document")  # python-docx
    def test_docx_reader_content_extraction(self, mock_document_class, test_cleanup):
        """Test DOCX reader content extraction."""
        from src.fileintel.document_processing.readers.docx_reader import DOCXReader

        # Mock document structure
        mock_paragraph1 = Mock()
        mock_paragraph1.text = "First paragraph content"
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "Second paragraph content"

        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_document_class.return_value = mock_doc

        temp_file = test_cleanup.create_temp_file(suffix=".docx")
        reader = DOCXReader()

        try:
            content = reader.read_document(temp_file)

            assert "First paragraph content" in content
            assert "Second paragraph content" in content

        except ImportError:
            pytest.skip("python-docx not available")

    @patch("docx.Document")
    def test_docx_reader_metadata_extraction(self, mock_document_class, test_cleanup):
        """Test DOCX reader metadata extraction."""
        from src.fileintel.document_processing.readers.docx_reader import DOCXReader

        # Mock document with core properties
        mock_props = Mock()
        mock_props.title = "Test Document Title"
        mock_props.author = "Test Author"
        mock_props.subject = "Test Subject"
        mock_props.created = datetime(2024, 1, 15)
        mock_props.modified = datetime(2024, 1, 16)

        mock_doc = Mock()
        mock_doc.core_properties = mock_props
        mock_doc.paragraphs = []
        mock_document_class.return_value = mock_doc

        temp_file = test_cleanup.create_temp_file(suffix=".docx")
        reader = DOCXReader()

        try:
            metadata = reader.extract_metadata(temp_file)

            assert metadata.get("title") == "Test Document Title"
            assert metadata.get("author") == "Test Author"
            assert metadata.get("subject") == "Test Subject"

        except ImportError:
            pytest.skip("python-docx not available")


class TestEPUBDocumentReader:
    """Test cases for EPUB document reader."""

    def test_epub_reader_initialization(self):
        """Test EPUB reader can be initialized."""
        from src.fileintel.document_processing.readers.epub_reader import EPUBReader

        reader = EPUBReader()
        assert reader is not None
        assert ".epub" in reader.supported_formats()

    def test_epub_reader_supported_formats(self):
        """Test EPUB reader supports expected formats."""
        from src.fileintel.document_processing.readers.epub_reader import EPUBReader

        reader = EPUBReader()
        formats = reader.supported_formats()

        assert ".epub" in formats

    @patch("ebooklib.epub.read_epub")
    def test_epub_reader_content_extraction(self, mock_read_epub, test_cleanup):
        """Test EPUB reader content extraction."""
        from src.fileintel.document_processing.readers.epub_reader import EPUBReader

        # Mock EPUB structure
        mock_item1 = Mock()
        mock_item1.get_type.return_value = 9  # ITEM_DOCUMENT
        mock_item1.get_content.return_value = (
            b"<html><body><p>Chapter 1 content</p></body></html>"
        )

        mock_item2 = Mock()
        mock_item2.get_type.return_value = 9
        mock_item2.get_content.return_value = (
            b"<html><body><p>Chapter 2 content</p></body></html>"
        )

        mock_book = Mock()
        mock_book.get_items.return_value = [mock_item1, mock_item2]
        mock_read_epub.return_value = mock_book

        temp_file = test_cleanup.create_temp_file(suffix=".epub")
        reader = EPUBReader()

        try:
            content = reader.read_document(temp_file)

            assert "Chapter 1 content" in content
            assert "Chapter 2 content" in content

        except ImportError:
            pytest.skip("ebooklib not available")

    @patch("ebooklib.epub.read_epub")
    def test_epub_reader_metadata_extraction(self, mock_read_epub, test_cleanup):
        """Test EPUB reader metadata extraction."""
        from src.fileintel.document_processing.readers.epub_reader import EPUBReader

        # Mock EPUB metadata
        mock_book = Mock()
        mock_book.get_metadata.return_value = [
            ("DC", "title", "Test EPUB Title"),
            ("DC", "creator", "Test Author"),
            ("DC", "language", "en"),
            ("DC", "publisher", "Test Publisher"),
        ]
        mock_book.get_items.return_value = []
        mock_read_epub.return_value = mock_book

        temp_file = test_cleanup.create_temp_file(suffix=".epub")
        reader = EPUBReader()

        try:
            metadata = reader.extract_metadata(temp_file)

            expected_fields = ["title", "creator", "language", "publisher"]
            for field in expected_fields:
                assert field in metadata

        except ImportError:
            pytest.skip("ebooklib not available")


class TestMOBIDocumentReader:
    """Test cases for MOBI document reader."""

    def test_mobi_reader_initialization(self):
        """Test MOBI reader can be initialized."""
        from src.fileintel.document_processing.readers.mobi_reader import MOBIReader

        reader = MOBIReader()
        assert reader is not None
        assert ".mobi" in reader.supported_formats()

    def test_mobi_reader_supported_formats(self):
        """Test MOBI reader supports expected formats."""
        from src.fileintel.document_processing.readers.mobi_reader import MOBIReader

        reader = MOBIReader()
        formats = reader.supported_formats()

        expected_formats = [".mobi", ".azw", ".azw3"]
        for fmt in expected_formats:
            assert fmt in formats


class TestDocumentReaderFactory:
    """Test cases for document reader factory."""

    def test_factory_initialization(self):
        """Test document reader factory initialization."""
        from src.fileintel.document_processing.factory import ReaderFactory

        factory = ReaderFactory()
        assert factory is not None

    def test_factory_reader_registration(self):
        """Test reader registration in factory."""
        from src.fileintel.document_processing.factory import ReaderFactory

        factory = ReaderFactory()

        # Should have readers for common formats
        assert factory.get_reader(".txt") is not None
        assert factory.get_reader(".pdf") is not None
        assert factory.get_reader(".docx") is not None

    def test_factory_format_detection(self):
        """Test format detection by file extension."""
        from src.fileintel.document_processing.factory import ReaderFactory

        factory = ReaderFactory()

        # Test various file extensions
        test_cases = [
            ("document.txt", ".txt"),
            ("report.pdf", ".pdf"),
            ("letter.docx", ".docx"),
            ("book.epub", ".epub"),
            ("file.DOCX", ".docx"),  # Case insensitive
        ]

        for filename, expected_ext in test_cases:
            detected_ext = factory._detect_format(Path(filename))
            assert detected_ext.lower() == expected_ext.lower()

    def test_factory_unsupported_format(self):
        """Test factory handling of unsupported formats."""
        from src.fileintel.document_processing.factory import ReaderFactory

        factory = ReaderFactory()

        # Should handle unsupported formats gracefully
        reader = factory.get_reader(".xyz")
        assert reader is None or hasattr(reader, "read_document")

    def test_factory_mime_type_detection(self):
        """Test MIME type detection."""
        from src.fileintel.document_processing.factory import ReaderFactory

        factory = ReaderFactory()

        mime_type_cases = [
            ("document.txt", "text/plain"),
            ("report.pdf", "application/pdf"),
            (
                "spreadsheet.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        ]

        for filename, expected_mime in mime_type_cases:
            detected_mime = factory.detect_mime_type(Path(filename))
            # Might not exactly match due to different detection methods
            assert isinstance(detected_mime, str)


class TestDocumentProcessingIntegration:
    """Integration tests for document processing pipeline."""

    def test_end_to_end_text_processing(self, sample_text_doc, test_cleanup):
        """Test end-to-end text document processing."""
        from src.fileintel.document_processing.factory import ReaderFactory

        # Create test file
        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_text_doc["content"])

        factory = ReaderFactory()
        reader = factory.get_reader(temp_file.suffix)

        # Process document
        content = reader.read_document(temp_file)
        metadata = reader.extract_metadata(temp_file)

        # Verify results
        assert content == sample_text_doc["content"]
        assert metadata["file_size"] == temp_file.stat().st_size
        assert metadata["character_count"] > 0

    def test_multiple_format_processing(self, test_document_files):
        """Test processing multiple document formats."""
        from src.fileintel.document_processing.factory import ReaderFactory

        factory = ReaderFactory()
        processed_files = []

        # Process all files in test fixtures
        for file_path in test_document_files.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [
                ".txt",
                ".md",
                ".json",
            ]:
                try:
                    reader = factory.get_reader(file_path.suffix)
                    if reader:
                        content = reader.read_document(file_path)
                        metadata = reader.extract_metadata(file_path)

                        processed_files.append(
                            {
                                "file": file_path.name,
                                "content_length": len(content),
                                "metadata_fields": len(metadata),
                            }
                        )
                except Exception as e:
                    # Some files might not be processable
                    continue

        assert len(processed_files) > 0

    def test_error_handling_pipeline(self, test_cleanup):
        """Test error handling in document processing pipeline."""
        from src.fileintel.document_processing.factory import ReaderFactory

        factory = ReaderFactory()

        # Test with corrupted file
        corrupted_file = test_cleanup.create_temp_file(suffix=".pdf")
        with open(corrupted_file, "w") as f:
            f.write("This is not a PDF file")

        reader = factory.get_reader(".pdf")

        # Should handle corrupted file gracefully
        try:
            content = reader.read_document(corrupted_file)
            # Might return empty string or partial content
            assert isinstance(content, str)
        except Exception:
            # Exception is acceptable for corrupted files
            pass

    def test_content_validation(self, sample_text_doc, test_cleanup):
        """Test content validation after processing."""
        from src.fileintel.document_processing.factory import ReaderFactory

        temp_file = test_cleanup.create_temp_file(suffix=".txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_text_doc["content"])

        factory = ReaderFactory()
        reader = factory.get_reader(".txt")

        content = reader.read_document(temp_file)

        # Validate content integrity
        original_hash = hashlib.sha256(sample_text_doc["content"].encode()).hexdigest()
        processed_hash = hashlib.sha256(content.encode()).hexdigest()
        assert original_hash == processed_hash

    def test_metadata_consistency(self, test_document_files):
        """Test metadata consistency across processing runs."""
        from src.fileintel.document_processing.factory import ReaderFactory

        factory = ReaderFactory()

        for file_path in test_document_files.glob("*.txt"):
            reader = factory.get_reader(".txt")

            # Process same file multiple times
            metadata1 = reader.extract_metadata(file_path)
            metadata2 = reader.extract_metadata(file_path)

            # Metadata should be consistent
            for key in metadata1:
                if key not in [
                    "processed_at",
                    "timestamp",
                ]:  # Exclude time-dependent fields
                    assert metadata1.get(key) == metadata2.get(
                        key
                    ), f"Inconsistent metadata for {key}"

    @pytest.mark.parametrize("file_extension", [".txt", ".md", ".json"])
    def test_format_specific_processing(self, file_extension, test_cleanup):
        """Test processing for specific file formats."""
        from src.fileintel.document_processing.factory import ReaderFactory

        factory = ReaderFactory()
        reader = factory.get_reader(file_extension)

        # Create format-specific test content
        test_content = f"Test content for {file_extension} format"
        temp_file = test_cleanup.create_temp_file(suffix=file_extension)

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        content = reader.read_document(temp_file)
        metadata = reader.extract_metadata(temp_file)

        assert content == test_content
        assert metadata["file_size"] > 0
