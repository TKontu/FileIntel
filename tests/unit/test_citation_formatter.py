"""
Unit tests for citation formatter.

Tests the CitationFormatter class for proper Harvard-style citation formatting
with and without page numbers.
"""

import pytest
from fileintel.citation import CitationFormatter, format_in_text_citation


class TestCitationFormatter:
    """Test suite for CitationFormatter class."""

    def test_format_in_text_citation_with_page(self):
        """Test in-text citation includes page number when available."""
        chunk = {
            "document_metadata": {
                "authors": ["Robert Cooper"],
                "publication_date": "2018-03-07",
                "title": "Agile-Stage-Gate for Manufacturers",
                "llm_extracted": True
            },
            "metadata": {
                "page_number": 17
            }
        }

        formatter = CitationFormatter()
        result = formatter.format_in_text_citation(chunk)

        assert result == "(Cooper, 2018, p. 17)"  # Surname only

    def test_format_in_text_citation_without_page(self):
        """Test in-text citation works without page number (graceful degradation)."""
        chunk = {
            "document_metadata": {
                "authors": ["Robert Cooper"],
                "publication_date": "2018-03-07",
                "title": "Agile-Stage-Gate for Manufacturers",
                "llm_extracted": True
            },
            "metadata": {}  # No page number
        }

        formatter = CitationFormatter()
        result = formatter.format_in_text_citation(chunk)

        assert result == "(Cooper, 2018)"  # Surname only

    def test_format_in_text_citation_multiple_authors_with_page(self):
        """Test citation with multiple authors includes page number (uses first author's surname)."""
        chunk = {
            "document_metadata": {
                "authors": ["Robert Cooper", "Anita Sommer"],
                "publication_date": "2018-03-07",
                "llm_extracted": True
            },
            "metadata": {
                "page_number": 23
            }
        }

        formatter = CitationFormatter()
        result = formatter.format_in_text_citation(chunk)

        assert result == "(Cooper, 2018, p. 23)"  # First author's surname only

    def test_format_in_text_citation_chunk_metadata_alias(self):
        """Test that chunk_metadata field alias works for page number."""
        chunk = {
            "document_metadata": {
                "authors": ["John Smith"],
                "publication_date": "2020-01-01",
                "llm_extracted": True
            },
            "chunk_metadata": {  # Alternative field name
                "page_number": 42
            }
        }

        formatter = CitationFormatter()
        result = formatter.format_in_text_citation(chunk)

        assert result == "(Smith, 2020, p. 42)"  # Surname from comma format

    def test_format_in_text_citation_no_metadata(self):
        """Test fallback to filename when no document metadata available."""
        chunk = {
            "original_filename": "research_paper.pdf",
            "metadata": {
                "page_number": 10
            }
        }

        formatter = CitationFormatter()
        result = formatter.format_in_text_citation(chunk)

        # Should fall back to filename (no page in fallback)
        assert result == "(research_paper)"

    def test_format_in_text_citation_module_function(self):
        """Test module-level convenience function."""
        chunk = {
            "document_metadata": {
                "authors": ["Jane Doe"],
                "publication_date": "2019-05-15",
                "llm_extracted": True
            },
            "metadata": {
                "page_number": 5
            }
        }

        result = format_in_text_citation(chunk)

        assert result == "(Doe, 2019, p. 5)"  # Surname extraction works

    def test_format_in_text_citation_page_number_zero(self):
        """Test that page number 0 is not included (falsy value)."""
        chunk = {
            "document_metadata": {
                "authors": ["Test Author"],
                "publication_date": "2021",
                "llm_extracted": True
            },
            "metadata": {
                "page_number": 0
            }
        }

        formatter = CitationFormatter()
        result = formatter.format_in_text_citation(chunk)

        # Page 0 should NOT be included (falsy value in Python)
        # This is correct - page 0 doesn't make sense for citations
        assert result == "(Author, 2021)"  # Surname only

    def test_format_in_text_citation_author_name_extraction(self):
        """Test surname extraction from full name with multiple parts."""
        chunk = {
            "document_metadata": {
                "authors": ["John David Smith"],
                "publication_date": "2022",
                "llm_extracted": True
            },
            "metadata": {
                "page_number": 15
            }
        }

        formatter = CitationFormatter()
        result = formatter.format_in_text_citation(chunk)

        # Should extract last name "Smith" from "John David Smith"
        assert result == "(Smith, 2022, p. 15)"
