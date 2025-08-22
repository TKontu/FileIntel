from abc import ABC
from typing import Any, Dict, List


class DocumentElement(ABC):
    """A base class for all elements extracted from a document."""

    def __init__(self, metadata: Dict[str, Any] = None):
        self.metadata = metadata or {}


class TextElement(DocumentElement):
    """Represents a block of text."""

    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        super().__init__(metadata)
        self.text = text


class TableElement(DocumentElement):
    """Represents a table."""

    def __init__(self, rows: List[List[str]], metadata: Dict[str, Any] = None):
        super().__init__(metadata)
        self.rows = rows


class ImageElement(DocumentElement):
    """Represents an image."""

    def __init__(
        self, image_data: bytes, mime_type: str, metadata: Dict[str, Any] = None
    ):
        super().__init__(metadata)
        self.image_data = image_data
        self.mime_type = mime_type
