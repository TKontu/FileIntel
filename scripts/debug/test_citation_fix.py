#!/usr/bin/env python3
"""
Test script to verify citation formatting improvements.

This tests:
1. _build_harvard_citation with specific_page parameter
2. Proper author/year extraction
3. Single page citation instead of ranges
"""

from src.fileintel.cli.graphrag import _build_harvard_citation


def test_citation_with_specific_page():
    """Test that specific_page parameter produces single page citation."""

    source = {
        "document": "AgileNPD_01_21.pdf",
        "page_numbers": "pp. 1-10, 12-25, 27, 29, 31-34, 36-48, 51-52",  # Aggregated pages
        "chunk_count": 70,
        "metadata": {
            "author_surnames": ["Cooper", "Smith"],
            "publication_year": "2021",
            "title": "Agile NPD Best Practices"
        }
    }

    # Test with specific page (should override page_numbers)
    citation = _build_harvard_citation(source, specific_page=16)
    print(f"Citation with specific page: {citation}")

    # Should produce: "Cooper & Smith, 2021, p. 16"
    # NOT: "Cooper & Smith, 2021, pp. 1-10, 12-25, ..."
    assert "p. 16" in citation, f"Expected 'p. 16' but got: {citation}"
    assert "pp. 1-10" not in citation, f"Should not include aggregated pages: {citation}"
    assert "Cooper" in citation, f"Expected author surname: {citation}"
    assert "2021" in citation, f"Expected year: {citation}"

    print("✓ Specific page citation works correctly")


def test_citation_without_specific_page():
    """Test fallback to aggregated page_numbers when specific_page is None."""

    source = {
        "document": "Test.pdf",
        "page_numbers": "pp. 5-7",
        "metadata": {
            "author_surnames": ["Jones"],
            "publication_year": "2020"
        }
    }

    citation = _build_harvard_citation(source, specific_page=None)
    print(f"Citation without specific page: {citation}")

    # Should fallback to page_numbers
    assert "pp. 5-7" in citation, f"Expected aggregated pages: {citation}"
    assert "Jones, 2020" in citation, f"Expected author and year: {citation}"

    print("✓ Fallback to aggregated pages works correctly")


def test_citation_with_multiple_authors():
    """Test et al. formatting for multiple authors."""

    source = {
        "document": "Research.pdf",
        "page_numbers": None,
        "metadata": {
            "author_surnames": ["Smith", "Jones", "Brown"],
            "publication_year": "2019"
        }
    }

    citation = _build_harvard_citation(source, specific_page=42)
    print(f"Citation with multiple authors: {citation}")

    # Should produce: "Smith et al., 2019, p. 42"
    assert "Smith et al." in citation, f"Expected 'et al.' format: {citation}"
    assert "2019" in citation, f"Expected year: {citation}"
    assert "p. 42" in citation, f"Expected specific page: {citation}"

    print("✓ Multiple authors (et al.) formatting works correctly")


def test_citation_without_metadata():
    """Test fallback when no author/year metadata available."""

    source = {
        "document": "AgileNPD_01_21.pdf",
        "page_numbers": "pp. 10-15",
        "metadata": {}
    }

    citation = _build_harvard_citation(source, specific_page=12)
    print(f"Citation without metadata: {citation}")

    # Should fallback to filename without .pdf
    assert "AgileNPD_01_21" in citation, f"Expected filename: {citation}"
    assert ".pdf" not in citation, f"Should strip .pdf extension: {citation}"
    assert "p. 12" in citation, f"Expected specific page: {citation}"

    print("✓ Fallback to filename works correctly")


def test_single_author():
    """Test single author formatting."""

    source = {
        "document": "Cooper.pdf",
        "metadata": {
            "author_surnames": ["Cooper"],
            "publication_year": "2023"
        }
    }

    citation = _build_harvard_citation(source, specific_page=5)
    print(f"Citation with single author: {citation}")

    # Should produce: "Cooper, 2023, p. 5" (no et al.)
    assert "Cooper, 2023" in citation, f"Expected single author format: {citation}"
    assert "et al." not in citation, f"Should not use et al. for single author: {citation}"
    assert "p. 5" in citation, f"Expected specific page: {citation}"

    print("✓ Single author formatting works correctly")


if __name__ == "__main__":
    print("Testing citation formatting improvements...\n")

    test_citation_with_specific_page()
    print()

    test_citation_without_specific_page()
    print()

    test_citation_with_multiple_authors()
    print()

    test_citation_without_metadata()
    print()

    test_single_author()
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nKey improvements verified:")
    print("  1. Specific page citations instead of ranges")
    print("  2. Proper author/year extraction from metadata")
    print("  3. Harvard-compliant formatting (Author, Year, p. X)")
    print("  4. Fallback handling for missing metadata")
