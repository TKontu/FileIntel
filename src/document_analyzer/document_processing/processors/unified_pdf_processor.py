from pathlib import Path
from typing import List
import pdfplumber

from ..base import FileReader
from ..elements import DocumentElement, TextElement
from .traditional_pdf import TraditionalPDFProcessor
from .ocr_processor import OCRProcessor

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

    def read(self, file_path: Path) -> List[DocumentElement]:
        """
        Reads a PDF file, determines its type, and uses the best
        processor to extract its content.

        Args:
            file_path: The path to the PDF file.

        Returns:
            A list of DocumentElement objects extracted from the PDF.
        """
        try:
            return self._process_pdf(file_path)
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            # Re-raise the exception to be handled by the caller
            raise e

    def _process_pdf(self, file_path: Path) -> List[DocumentElement]:
        with pdfplumber.open(file_path) as pdf:
            text_sample = ""
            for page in pdf.pages[:3]:
                page_text = page.extract_text()
                if page_text:
                    text_sample += page_text
                if len(text_sample) > self.text_threshold:
                    break

            if len(text_sample) > self.text_threshold:
                print("PDF identified as text-based. Using TraditionalPDFProcessor.")
                processor = TraditionalPDFProcessor()
                return processor.read(file_path)
            else:
                print("PDF identified as image-based. Delegating to OCRProcessor.")
                processor = OCRProcessor()
                return processor.read(file_path)

