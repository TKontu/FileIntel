from pathlib import Path
from typing import List
from ..base import FileReader
from ..elements import TextElement, DocumentElement

class TextReader(FileReader):
    """A reader for plain text and Markdown files."""

    def read(self, file_path: Path) -> List[DocumentElement]:
        """Read the content of a text or markdown file and return it as a list containing a single TextElement."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return [TextElement(text=text, metadata={"source": str(file_path)})]
        except Exception as e:
            # In a real application, you'd have more specific error handling
            # and logging here.
            print(f"Error reading file {file_path}: {e}")
            return []
