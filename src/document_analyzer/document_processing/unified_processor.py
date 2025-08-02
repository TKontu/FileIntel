from pathlib import Path
from .processors.traditional_pdf import TraditionalPDFProcessor
from .processors.epub_processor import EPUBProcessor
from .processors.mobi_processor import MOBIProcessor
from .preprocessor import TextPreprocessor
from ..core.exceptions import DocumentProcessingException

class UnifiedDocumentProcessor:
    def __init__(self):
        self.pdf_processor = TraditionalPDFProcessor()
        self.epub_processor = EPUBProcessor()
        self.mobi_processor = MOBIProcessor()
        self.preprocessor = TextPreprocessor()

    def process(self, file_path: str) -> list[str]:
        """
        Processes a document, returning a list of text chunks.
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentProcessingException(f"File not found: {file_path}")

        extension = path.suffix.lower()
        content = ""

        if extension == ".pdf":
            content = self.pdf_processor.read(path)
        elif extension == ".epub":
            content = self.epub_processor.read(path)
        elif extension == ".mobi":
            content = self.mobi_processor.read(path)
        else:
            raise DocumentProcessingException(f"Unsupported file type: {extension}")

        return self.preprocessor.preprocess(content)
