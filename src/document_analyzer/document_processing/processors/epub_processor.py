import ebooklib
from ebooklib import epub
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup

from ..base import FileReader
from ..elements import DocumentElement, TextElement
from ...core.exceptions import DocumentProcessingException

class EPUBReader(FileReader):
    def read(self, file_path: Path) -> List[DocumentElement]:
        """
        Reads the text content from an EPUB file, returning a TextElement
        for each chapter/document within the EPUB.
        """
        if not file_path.exists():
            raise DocumentProcessingException(f"File not found: {file_path}")

        elements = []
        try:
            book = epub.read_epub(file_path)
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                content = item.get_content()
                soup = BeautifulSoup(content, "html.parser")
                text = soup.get_text()
                if text and text.strip():
                    # Use the item's file name as a proxy for chapter name
                    metadata = {"source": str(file_path), "chapter": item.get_name()}
                    elements.append(TextElement(text=text, metadata=metadata))
            return elements
        except Exception as e:
            raise DocumentProcessingException(f"Error processing EPUB file {file_path}: {e}")

