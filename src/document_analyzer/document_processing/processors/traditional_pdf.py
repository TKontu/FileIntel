import pdfplumber
from pathlib import Path
from typing import List
import logging
from ..base import FileReader
from ..elements import DocumentElement, TextElement
from ...core.exceptions import DocumentProcessingException

logger = logging.getLogger(__name__)


class TraditionalPDFProcessor(FileReader):
    def read(
        self, file_path: Path, adapter: logging.LoggerAdapter = None
    ) -> List[DocumentElement]:
        """
        Reads the text content from a PDF file using pdfplumber,
        returning a TextElement for each page.
        """
        log = adapter or logger
        if not file_path.exists():
            raise DocumentProcessingException(f"File not found: {file_path}")

        elements = []
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                log.info(
                    f"Starting text extraction for {file_path.name} ({total_pages} pages)."
                )
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        metadata = {"source": str(file_path), "page_number": i + 1}
                        elements.append(TextElement(text=page_text, metadata=metadata))
                    log.info(
                        f"Extracted text from page {i + 1}/{total_pages} of {file_path.name}."
                    )
            return elements
        except Exception as e:
            raise DocumentProcessingException(
                f"Error processing PDF file {file_path}: {e}"
            )
