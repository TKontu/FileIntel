import ebooklib
from ebooklib import epub
from pathlib import Path
from document_analyzer.document_processing.base import FileReader
from document_analyzer.core.exceptions import DocumentProcessingException
from bs4 import BeautifulSoup

class EPUBProcessor(FileReader):
    def read(self, file_path: Path) -> str:
        """
        Reads the text content from an EPUB file using ebooklib.
        """
        if not file_path.exists():
            raise DocumentProcessingException(f"File not found: {file_path}")

        try:
            book = epub.read_epub(file_path)
            content = ""
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), "html.parser")
                content += soup.get_text() + "\n"
            return content
        except Exception as e:
            raise DocumentProcessingException(f"Error processing EPUB file {file_path}: {e}")

