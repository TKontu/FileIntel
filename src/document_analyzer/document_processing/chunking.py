from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..core.config import settings


class TextChunker:
    def __init__(self):
        self.chunk_size = settings.rag.chunk_size
        self.chunk_overlap = settings.rag.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def chunk_text(self, text: str) -> List[str]:
        """Chunks the given text into smaller pieces."""
        return self._splitter.split_text(text)

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunks a list of documents."""
        return self._splitter.split_documents(documents)
