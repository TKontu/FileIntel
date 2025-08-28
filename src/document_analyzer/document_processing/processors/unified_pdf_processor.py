from pathlib import Path
from typing import List, Tuple, Dict, Any
import pdfplumber
import logging

from ..base import FileReader
from ..elements import DocumentElement, TextElement
from .traditional_pdf import TraditionalPDFProcessor
from .ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)


class UnifiedPDFProcessor(FileReader):
    """
    A unified processor for PDF documents that triages and delegates
    to the appropriate specialized processor.
    """

    def __init__(self, text_threshold: int = 100):
        """
        Initializes the processor.

        Args:
            text_threshold: The minimum number of characters that must be
                            extracted from a PDF to be considered text-based.
        """
        self.text_threshold = text_threshold

    def read(
        self, file_path: Path, adapter: logging.LoggerAdapter = None
    ) -> Tuple[List[DocumentElement], Dict[str, Any]]:
        """
        Reads a PDF file, determines its type, and uses the best
        processor to extract its content.

        Args:
            file_path: The path to the PDF file.
            adapter: A logger adapter for contextual logging.

        Returns:
            A list of DocumentElement objects extracted from the PDF.
        """
        log = adapter or logger
        try:
            return self._process_pdf(file_path, log)
        except Exception as e:
            log.error(f"Error processing PDF {file_path}: {e}", exc_info=True)
            # Re-raise the exception to be handled by the caller
            raise e

    def _process_pdf(
        self, file_path: Path, log: logging.LoggerAdapter
    ) -> List[DocumentElement]:
        with pdfplumber.open(file_path) as pdf:
            text_sample = ""
            for page in pdf.pages[:3]:
                page_text = page.extract_text()
                if page_text:
                    text_sample += page_text
                if len(text_sample) > self.text_threshold:
                    break

            if len(text_sample) > self.text_threshold:
                log.info("PDF identified as text-based. Using TraditionalPDFProcessor.")
                processor = TraditionalPDFProcessor()
                return processor.read(file_path, log)
            else:
                log.info("PDF identified as image-based. Delegating to OCRProcessor.")
                processor = OCRProcessor()
                return processor.read(file_path, log)
