import pdfplumber
from pathlib import Path
from document_analyzer.document_processing.base import FileReader
from document_analyzer.core.exceptions import DocumentProcessingException

class TraditionalPDFProcessor(FileReader):
    def read(self, file_path: Path) -> str:
        """
        Reads the text content from a PDF file using pdfplumber.
        """
        if not file_path.exists():
            raise DocumentProcessingException(f"File not found: {file_path}")

        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            raise DocumentProcessingException(f"Error processing PDF file {file_path}: {e}")

