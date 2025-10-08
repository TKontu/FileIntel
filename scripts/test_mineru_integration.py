#!/usr/bin/env python3
"""Test MinerU integration with a sample PDF."""

import sys
import os
sys.path.append('src')

from pathlib import Path
from fileintel.document_processing.processors.mineru_selfhosted import MinerUSelfHostedProcessor
from fileintel.core.config import get_config
import logging

logging.basicConfig(level=logging.INFO)

def test_mineru_processing():
    """Test MinerU processor with sample PDF."""
    # Check for test PDF
    test_pdf = Path("test_document.pdf")
    if not test_pdf.exists():
        print(f"Error: Test PDF not found: {test_pdf}")
        return False

    # Check configuration
    config = get_config()
    mineru_config = config.document_processing.mineru

    if not mineru_config.base_url:
        print("Error: MinerU base_url not configured")
        return False

    print(f"Testing with {mineru_config.api_type} MinerU API at {mineru_config.base_url}")

    try:
        # Test processor
        processor = MinerUSelfHostedProcessor()
        elements, metadata = processor.read(test_pdf)

        print(f"✓ Processing successful!")
        print(f"  Elements: {len(elements)}")
        print(f"  Content length: {len(elements[0].text) if elements else 0}")
        print(f"  Metadata keys: {list(metadata.keys())}")

        return True

    except Exception as e:
        print(f"✗ Processing failed: {e}")
        return False

if __name__ == "__main__":
    success = test_mineru_processing()
    sys.exit(0 if success else 1)