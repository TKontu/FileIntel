from typing import Type
import logging
from .base import FileReader
from .processors.epub_processor import EPUBReader
from .processors.mobi_processor import MOBIReader
from .processors.text_processor import TextReader
from .processors.unified_pdf_processor import UnifiedPDFProcessor
from .type_detector import DocumentTypeDetector

logger = logging.getLogger(__name__)


class ReaderFactory:
    def __init__(self):
        self.detector = DocumentTypeDetector()
        self.readers = {
            "application/epub+zip": EPUBReader,
            "application/x-mobipocket-ebook": MOBIReader,
            "application/pdf": UnifiedPDFProcessor,
            "text/plain": TextReader,
            "text/markdown": TextReader,
        }

    def get_reader(self, file_path) -> FileReader:
        """Get the appropriate reader for a given file."""
        mime_type = self.detector.detect_mime_type(file_path)
        logger.info(f"Detected MIME type '{mime_type}' for file {file_path.name}.")
        reader_class = self.readers.get(mime_type)
        if not reader_class:
            logger.error(f"No reader found for MIME type: {mime_type}")
            raise ValueError(f"No reader found for MIME type: {mime_type}")
        logger.info(f"Selected reader: {reader_class.__name__}")
        return reader_class()
