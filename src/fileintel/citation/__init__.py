"""
Citation formatting utilities for FileIntel.

Provides proper academic citation formatting using extracted document metadata.
"""

from .citation_formatter import CitationFormatter, format_source_reference, format_in_text_citation, format_full_citation

__all__ = [
    "CitationFormatter",
    "format_source_reference",
    "format_in_text_citation",
    "format_full_citation"
]