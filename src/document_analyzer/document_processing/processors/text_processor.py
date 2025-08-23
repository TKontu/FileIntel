from pathlib import Path
from typing import List
import logging
from ..base import FileReader
from ..elements import TextElement, DocumentElement

logger = logging.getLogger(__name__)


from typing import Tuple, Dict, Any


class TextReader(FileReader):
    """A reader for plain text and Markdown files."""

    def read(
        self, file_path: Path, adapter: logging.LoggerAdapter = None
    ) -> Tuple[List[DocumentElement], Dict[str, Any]]:
        """Read the content of a text or markdown file and return it as a list containing a single TextElement."""
        log = adapter or logger
        try:
            log.info(f"Reading text file: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            log.info(f"Successfully read text file: {file_path.name}")
            elements = [TextElement(text=text, metadata={"source": str(file_path)})]
            return elements, {}  # No metadata for plain text
        except Exception as e:
            log.error(f"Error reading file {file_path}: {e}", exc_info=True)
            return [], {}
