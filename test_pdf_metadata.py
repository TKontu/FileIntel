#!/usr/bin/env python3
"""
Test script to examine what metadata is available from PDF files.
"""

import sys
from pathlib import Path
import pdfplumber

def analyze_pdf_metadata(pdf_path):
    """Analyze what metadata and page information is available."""
    print(f"📄 Analyzing PDF: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Document-level metadata
            print(f"\n📋 Document Metadata:")
            metadata = pdf.metadata or {}
            for key, value in metadata.items():
                print(f"  {key}: {value}")

            print(f"\n📊 Basic Info:")
            print(f"  Total pages: {len(pdf.pages)}")
            print(f"  PDF version: {getattr(pdf, 'pdf_version', 'Unknown')}")

            # Analyze first few pages for available metadata
            print(f"\n📖 Page-level Analysis:")
            for i, page in enumerate(pdf.pages[:3]):  # First 3 pages
                print(f"\n  Page {i+1} (physical):")

                # Page object attributes
                page_attrs = [attr for attr in dir(page) if not attr.startswith('_')]
                print(f"    Available attributes: {', '.join(page_attrs[:10])}...")

                # Page dimensions and properties
                print(f"    Dimensions: {page.width} x {page.height}")
                print(f"    Rotation: {getattr(page, 'rotation', 'Unknown')}")

                # Check if page has any numbering information
                full_text = page.extract_text() or ""
                print(f"    Text length: {len(full_text)} chars")

                # Look for page numbers in various locations
                # Top region (header)
                if hasattr(page, 'within_bbox'):
                    try:
                        header = page.within_bbox((0, 0, page.width, 50)).extract_text() or ""
                        footer = page.within_bbox((0, page.height-50, page.width, page.height)).extract_text() or ""

                        print(f"    Header text: '{header.strip()}'")
                        print(f"    Footer text: '{footer.strip()}'")
                    except Exception as e:
                        print(f"    Header/footer extraction failed: {e}")

                # Check for any page-specific metadata
                if hasattr(page, 'attrs'):
                    print(f"    Page attrs: {page.attrs}")

                # Look for text objects with position info
                if hasattr(page, 'chars'):
                    chars = page.chars[:5]  # First 5 characters
                    if chars:
                        print(f"    Sample char positions: {[(c.get('text'), c.get('x0'), c.get('y0')) for c in chars]}")

                print(f"    First 100 chars: '{full_text[:100]}'")

            # Check if PDF has any structural information
            print(f"\n🔍 Structural Information:")

            # Bookmarks/Outline
            try:
                if hasattr(pdf, 'outline'):
                    outline = pdf.outline
                    print(f"  Bookmarks/Outline: {len(outline) if outline else 0} entries")
                    if outline:
                        for i, bookmark in enumerate(outline[:3]):
                            print(f"    {i+1}: {bookmark}")
                else:
                    print("  No outline/bookmark information available")
            except Exception as e:
                print(f"  Outline extraction failed: {e}")

            # Page labels (if any)
            try:
                if hasattr(pdf, 'pages') and hasattr(pdf.pages[0], 'page_number'):
                    print("  Page has page_number attribute")
                else:
                    print("  No page_number attribute found")
            except:
                pass

    except Exception as e:
        print(f"❌ Error analyzing PDF: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Look for test PDF files
    test_files = [
        "./uploads/e4169caa-0d6c-4760-85b4-87bc4fc55789.pdf",
        "./uploads/4eb15427-ffa2-462d-bd22-481d342c63be.pdf",
        "./uploads/Fifth Discipline_ The Art and Practice of the Learning Organization, The - Peter M. Senge.pdf",
    ]

    for pdf_path in test_files:
        if Path(pdf_path).exists():
            analyze_pdf_metadata(pdf_path)
            break
    else:
        print("No test PDF files found")

if __name__ == "__main__":
    main()