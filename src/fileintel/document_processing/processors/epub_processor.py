import ebooklib
from ebooklib import epub
from pathlib import Path
from typing import List, Tuple, Dict, Any
from bs4 import BeautifulSoup
import logging

# FileReader import removed - no longer using abstract base
from ..elements import DocumentElement, TextElement

# Removed custom exception import - using built-in exceptions
from .traditional_pdf import (
    validate_file_for_processing,
    DocumentProcessingError,
    FileCorruptionError,
)

logger = logging.getLogger(__name__)


class EPUBReader:
    def read(
        self, file_path: Path, adapter: logging.LoggerAdapter = None
    ) -> Tuple[List[DocumentElement], Dict[str, Any]]:
        """
        Reads the text content from an EPUB file, returning a TextElement
        for each chapter/document within the EPUB.
        """
        log = adapter or logger

        # Comprehensive file validation
        validate_file_for_processing(file_path, ".epub")

        elements = []
        doc_metadata = {}
        try:
            book = epub.read_epub(file_path)
            log.info(f"Processing EPUB file: {file_path.name}")

            # Extract EPUB metadata
            try:
                doc_metadata = {
                    "title": book.get_metadata("DC", "title")[0][0]
                    if book.get_metadata("DC", "title")
                    else None,
                    "authors": [
                        author[0] for author in book.get_metadata("DC", "creator")
                    ]
                    if book.get_metadata("DC", "creator")
                    else [],
                    "publisher": book.get_metadata("DC", "publisher")[0][0]
                    if book.get_metadata("DC", "publisher")
                    else None,
                    "publication_date": book.get_metadata("DC", "date")[0][0]
                    if book.get_metadata("DC", "date")
                    else None,
                    "language": book.get_metadata("DC", "language")[0][0]
                    if book.get_metadata("DC", "language")
                    else None,
                    "identifier": book.get_metadata("DC", "identifier")[0][0]
                    if book.get_metadata("DC", "identifier")
                    else None,
                }
                # Remove None values
                doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}
                log.info(f"Extracted EPUB metadata: {doc_metadata}")
            except Exception as meta_error:
                log.warning(f"Could not extract EPUB metadata: {meta_error}")
                doc_metadata = {}
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                content = item.get_content()
                soup = BeautifulSoup(content, "html.parser")
                text = soup.get_text()
                if text and text.strip():
                    # Use the item's file name as a proxy for chapter name
                    metadata = {"source": str(file_path), "chapter": item.get_name()}
                    elements.append(TextElement(text=text, metadata=metadata))
            log.info(f"Successfully processed EPUB file: {file_path.name}")
            return elements, doc_metadata
        except Exception as e:
            log.error(f"Error processing EPUB file {file_path}: {e}", exc_info=True)
            raise DocumentProcessingError(
                f"Error processing EPUB file {file_path}: {e}"
            )
