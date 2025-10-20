"""
Citation formatting utilities for query results.

Provides Harvard-style citation formatting using extracted document metadata,
with graceful fallback to filename when metadata is not available.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CitationFormatter:
    """Formats document references using metadata for proper academic citations."""

    def __init__(self, style: str = "harvard"):
        """
        Initialize citation formatter.

        Args:
            style: Citation style (currently only 'harvard' supported)
        """
        self.style = style

    def format_source_reference(self, chunk: Dict[str, Any]) -> str:
        """
        Format a source reference for display in query results.

        Args:
            chunk: Chunk data containing document metadata and file information

        Returns:
            Formatted citation string (Author, Year, Title) or filename fallback
        """
        try:
            document_metadata = chunk.get("document_metadata", {})

            # Check if we have metadata for proper citation
            if self._has_citation_metadata(document_metadata):
                return self._format_harvard_citation(document_metadata)
            else:
                # Fallback to filename
                return self._format_filename_fallback(chunk)

        except Exception as e:
            logger.warning(f"Error formatting citation: {e}")
            return self._format_filename_fallback(chunk)

    def format_in_text_citation(self, chunk: Dict[str, Any]) -> str:
        """
        Format an in-text citation for Harvard style with page number.

        Args:
            chunk: Chunk data containing document metadata and chunk metadata

        Returns:
            Formatted in-text citation like (Author, Year, p. 15) or (Author, Year) if no page
        """
        try:
            document_metadata = chunk.get("document_metadata", {})
            chunk_metadata = chunk.get("metadata", chunk.get("chunk_metadata", {}))
            page_number = chunk_metadata.get("page_number")

            if self._has_citation_metadata(document_metadata):
                author_surname = self._extract_author_surname(document_metadata)
                year = self._extract_year(document_metadata)

                # Build citation with page number if available
                if author_surname and year and page_number:
                    return f"({author_surname}, {year}, p. {page_number})"
                elif author_surname and year:
                    return f"({author_surname}, {year})"
                elif author_surname:
                    return f"({author_surname})"

            # Fallback to simplified filename
            filename = chunk.get("original_filename", chunk.get("filename", "Unknown"))
            # Remove extension for cleaner in-text citation
            if "." in filename:
                filename = filename.rsplit(".", 1)[0]
            return f"({filename})"

        except Exception as e:
            logger.warning(f"Error formatting in-text citation: {e}")
            filename = chunk.get("original_filename", chunk.get("filename", "Unknown"))
            return f"({filename})"

    def format_full_citation(self, chunk: Dict[str, Any]) -> str:
        """
        Format a complete Harvard-style citation.

        Args:
            chunk: Chunk data containing document metadata

        Returns:
            Full formatted citation with all available details
        """
        try:
            document_metadata = chunk.get("document_metadata", {})

            if self._has_citation_metadata(document_metadata):
                return self._format_complete_harvard_citation(document_metadata)
            else:
                return self._format_filename_fallback(chunk, include_details=True)

        except Exception as e:
            logger.warning(f"Error formatting full citation: {e}")
            return self._format_filename_fallback(chunk, include_details=True)

    def _has_citation_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata contains sufficient information for proper citation."""
        return (
            metadata and
            (metadata.get("authors") or metadata.get("title")) and
            metadata.get("llm_extracted", False)
        )

    def _format_harvard_citation(self, metadata: Dict[str, Any]) -> str:
        """Format basic Harvard citation: Author, Year, Title."""
        parts = []

        # Author(s)
        author = self._extract_primary_author(metadata)
        if author:
            parts.append(author)

        # Year
        year = self._extract_year(metadata)
        if year:
            parts.append(f"({year})")

        # Title
        title = metadata.get("title", "").strip()
        if title:
            # Clean up title formatting
            title = title.strip('"').strip("'")
            parts.append(f"'{title}'")

        return ", ".join(parts) if parts else "Unknown Source"

    def _format_complete_harvard_citation(self, metadata: Dict[str, Any]) -> str:
        """Format complete Harvard citation with all available details."""
        citation_parts = []

        # Author(s)
        author = self._format_authors_full(metadata)
        if author:
            citation_parts.append(author)

        # Year
        year = self._extract_year(metadata)
        if year:
            citation_parts.append(f"({year})")

        # Title
        title = metadata.get("title", "").strip()
        if title:
            title = title.strip('"').strip("'")
            citation_parts.append(f"'{title}'")

        # Publisher
        publisher = metadata.get("publisher", "").strip()
        if publisher:
            citation_parts.append(publisher)

        # DOI or URL
        doi = metadata.get("doi", "").strip()
        source_url = metadata.get("source_url", "").strip()

        if doi:
            citation_parts.append(f"doi:{doi}")
        elif source_url:
            citation_parts.append(f"Available at: {source_url}")

        return ". ".join(citation_parts) + "." if citation_parts else "Unknown Source."

    def _format_filename_fallback(self, chunk: Dict[str, Any], include_details: bool = False) -> str:
        """Format fallback citation using filename when metadata is unavailable."""
        filename = chunk.get("original_filename", chunk.get("filename", "Unknown Source"))

        if include_details:
            # Include document ID for reference
            doc_id = chunk.get("document_id", "")
            if doc_id:
                return f"{filename} (Document ID: {doc_id})"

        return filename

    def _extract_primary_author(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract primary author for citation."""
        authors = metadata.get("authors", "")
        if not authors:
            return None

        if isinstance(authors, list):
            # Take first author
            return authors[0] if authors else None
        elif isinstance(authors, str):
            # Split and take first author
            author_list = [a.strip() for a in authors.split(",")]
            return author_list[0] if author_list else None

        return str(authors) if authors else None

    def _extract_author_surname(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract surname only from primary author for in-text citations.

        Handles formats:
        - "LastName, FirstName" → "LastName"
        - "FirstName LastName" → "LastName"
        - "FirstName MiddleName LastName" → "LastName"
        """
        full_name = self._extract_primary_author(metadata)
        if not full_name:
            return None

        # Handle "LastName, FirstName" format
        if ',' in full_name:
            return full_name.split(',')[0].strip()

        # Handle "FirstName LastName" format - take last word
        name_parts = full_name.strip().split()
        if name_parts:
            return name_parts[-1]

        return full_name

    def _format_authors_full(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Format all authors for complete citation."""
        authors = metadata.get("authors", "")
        if not authors:
            return None

        if isinstance(authors, list):
            if len(authors) == 1:
                return authors[0]
            elif len(authors) == 2:
                return f"{authors[0]} and {authors[1]}"
            else:
                return f"{authors[0]} et al."
        elif isinstance(authors, str):
            # Clean up author string
            authors = authors.strip()
            if "," in authors:
                author_list = [a.strip() for a in authors.split(",")]
                if len(author_list) == 1:
                    return author_list[0]
                elif len(author_list) == 2:
                    return f"{author_list[0]} and {author_list[1]}"
                else:
                    return f"{author_list[0]} et al."
            return authors

        return str(authors) if authors else None

    def _extract_year(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract publication year from metadata."""
        pub_date = metadata.get("publication_date", "")
        if not pub_date:
            return None

        # Try to extract year from various date formats
        if isinstance(pub_date, str):
            pub_date = pub_date.strip()
            # Handle common date formats: YYYY, YYYY-MM-DD, DD/MM/YYYY, etc.
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', pub_date)
            if year_match:
                return year_match.group(0)

        return str(pub_date) if pub_date else None


# Module-level convenience functions
_formatter = CitationFormatter()

def format_source_reference(chunk: Dict[str, Any]) -> str:
    """Format a source reference using the default formatter."""
    return _formatter.format_source_reference(chunk)

def format_in_text_citation(chunk: Dict[str, Any]) -> str:
    """Format an in-text citation using the default formatter."""
    return _formatter.format_in_text_citation(chunk)

def format_full_citation(chunk: Dict[str, Any]) -> str:
    """Format a complete citation using the default formatter."""
    return _formatter.format_full_citation(chunk)