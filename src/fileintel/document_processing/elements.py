from abc import ABC
from typing import Any, Dict


class DocumentElement(ABC):
    """A base class for all elements extracted from a document."""

    def __init__(self, metadata: Dict[str, Any] = None):
        self.metadata = metadata or {}


class TextElement(DocumentElement):
    """Represents a block of text."""

    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        super().__init__(metadata)
        self.text = text
