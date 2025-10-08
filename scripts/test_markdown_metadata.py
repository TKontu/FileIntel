#!/usr/bin/env python3
"""
Test script for markdown metadata enhancement in MinerU integration.

Validates the complete flow:
1. MinerU extraction with markdown headers
2. Header mapping to pages
3. Metadata flow to TextElements
4. Metadata propagation to chunks
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_header_extraction():
    """Test markdown header extraction from sample markdown."""
    from fileintel.document_processing.processors.mineru_selfhosted import MinerUSelfHostedProcessor

    # Sample markdown content
    markdown_content = """# Introduction
This is the introduction section.

## Background
Some background information here.

### Methodology
Details about the methodology.

# Results
This section contains results.

## Analysis
Analysis of the results.
"""

    processor = MinerUSelfHostedProcessor()
    headers = processor._extract_markdown_headers(markdown_content)

    print("=" * 80)
    print("TEST 1: Markdown Header Extraction")
    print("=" * 80)
    print(f"Extracted {len(headers)} headers:")
    for header in headers:
        indent = "  " * (header['level'] - 1)
        print(f"{indent}[H{header['level']}] {header['text']} (line {header['line_number']})")

    # Validate
    assert len(headers) == 5, f"Expected 5 headers, got {len(headers)}"
    assert headers[0]['text'] == "Introduction", f"Expected 'Introduction', got '{headers[0]['text']}'"
    assert headers[0]['level'] == 1, f"Expected level 1, got {headers[0]['level']}"

    print("\n✓ Header extraction test passed!\n")
    return True


def test_header_to_page_mapping():
    """Test mapping headers to pages using content_list.json structure."""
    from fileintel.document_processing.processors.mineru_selfhosted import MinerUSelfHostedProcessor

    markdown_content = """# Chapter 1
Content for chapter 1.

## Section 1.1
More content here.

# Chapter 2
Content for chapter 2.
"""

    # Simulate content_list.json structure
    content_list = [
        {'page_idx': 0, 'text': '# Chapter 1', 'type': 'text'},
        {'page_idx': 0, 'text': 'Content for chapter 1.', 'type': 'text'},
        {'page_idx': 1, 'text': '## Section 1.1', 'type': 'text'},
        {'page_idx': 1, 'text': 'More content here.', 'type': 'text'},
        {'page_idx': 2, 'text': '# Chapter 2', 'type': 'text'},
        {'page_idx': 2, 'text': 'Content for chapter 2.', 'type': 'text'},
    ]

    processor = MinerUSelfHostedProcessor()
    headers = processor._extract_markdown_headers(markdown_content)
    headers_by_page = processor._map_headers_to_pages(headers, markdown_content, content_list)

    print("=" * 80)
    print("TEST 2: Header to Page Mapping")
    print("=" * 80)
    print(f"Mapped headers across {len(headers_by_page)} pages:")
    for page_idx in sorted(headers_by_page.keys()):
        print(f"\nPage {page_idx}:")
        for header in headers_by_page[page_idx]:
            print(f"  - [{header['type']}] {header['text']}")

    # Validate
    assert 0 in headers_by_page, "Page 0 should have headers"
    assert 1 in headers_by_page, "Page 1 should have headers"
    assert 2 in headers_by_page, "Page 2 should have headers"

    print("\n✓ Header mapping test passed!\n")
    return True


def test_chunk_metadata_flow():
    """Test that metadata flows through to chunk storage format."""
    from fileintel.tasks.document_tasks import clean_and_chunk_text

    # Sample text with page mappings including header metadata
    text = "This is sentence one on page 1. This is sentence two on page 1. This is sentence three on page 2."

    page_mappings = [
        {
            'start_pos': 0,
            'end_pos': 60,
            'page_number': 1,
            'extraction_method': 'mineru_selfhosted_json',
            'section_title': 'Introduction',
            'section_path': 'Document > Introduction',
            'markdown_headers': [
                {'level': 1, 'text': 'Introduction', 'type': 'h1'}
            ]
        },
        {
            'start_pos': 61,
            'end_pos': len(text),
            'page_number': 2,
            'extraction_method': 'mineru_selfhosted_json',
            'section_title': 'Methods',
            'section_path': 'Document > Methods',
            'markdown_headers': [
                {'level': 1, 'text': 'Methods', 'type': 'h1'}
            ]
        }
    ]

    chunks = clean_and_chunk_text(text, page_mappings=page_mappings)

    print("=" * 80)
    print("TEST 3: Chunk Metadata Flow")
    print("=" * 80)
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        metadata = chunk.get('metadata', {})
        print(f"\nChunk {i}:")
        print(f"  Text: {chunk['text'][:50]}...")
        print(f"  Pages: {metadata.get('pages', [])}")
        print(f"  Section: {metadata.get('section_title', 'N/A')}")
        print(f"  Path: {metadata.get('section_path', 'N/A')}")
        print(f"  Headers: {len(metadata.get('markdown_headers', []))} headers")
        print(f"  Extraction: {metadata.get('extraction_methods', [])}")

    # Validate metadata preservation
    has_section_title = any('section_title' in c.get('metadata', {}) for c in chunks)
    has_headers = any('markdown_headers' in c.get('metadata', {}) for c in chunks)
    has_extraction = any('extraction_methods' in c.get('metadata', {}) for c in chunks)

    print("\n" + "=" * 80)
    print("Metadata Preservation Check:")
    print(f"  ✓ Section titles: {'Yes' if has_section_title else 'No'}")
    print(f"  ✓ Markdown headers: {'Yes' if has_headers else 'No'}")
    print(f"  ✓ Extraction methods: {'Yes' if has_extraction else 'No'}")

    if has_section_title and has_headers and has_extraction:
        print("\n✓ Chunk metadata flow test passed!\n")
        return True
    else:
        print("\n✗ Some metadata not preserved in chunks!\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MARKDOWN METADATA ENHANCEMENT TEST SUITE")
    print("=" * 80 + "\n")

    tests = [
        ("Header Extraction", test_header_extraction),
        ("Header to Page Mapping", test_header_to_page_mapping),
        ("Chunk Metadata Flow", test_chunk_metadata_flow),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 80 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
