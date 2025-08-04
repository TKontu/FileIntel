import pdfplumber
from pathlib import Path
from typing import List
from ..base import FileReader
from ..elements import DocumentElement, TextElement
from ...core.exceptions import DocumentProcessingException

class TraditionalPDFProcessor(FileReader):
    def read(self, file_path: Path) -> List[DocumentElement]:
        """
        Reads the text content from a PDF file using pdfplumber,
        returning a TextElement for each page.
        """
        if not file_path.exists():
            raise DocumentProcessingException(f"File not found: {file_path}")

        elements = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        metadata = {"source": str(file_path), "page_number": i + 1}
                        elements.append(TextElement(text=page_text, metadata=metadata))
            return elements
        except Exception as e:
            raise DocumentProcessingException(f"Error processing PDF file {file_path}: {e}")

