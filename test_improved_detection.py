#!/usr/bin/env python3
"""
Test improved TOC/bibliography detection for false positives.

This script validates that the improved detection functions:
1. Correctly detect synthetic TOC/bibliography examples
2. Do NOT incorrectly flag legitimate prose chunks (false positives)
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fileintel.core.config import get_config
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.document_processing.element_detection import is_toc_or_lof, is_bibliography_section

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_on_synthetic_examples():
    """Test detection on synthetic TOC/bibliography examples."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Validate detection on synthetic TOC/bibliography examples")
    logger.info("="*80)

    # Synthetic test cases based on actual problematic chunks
    test_cases = [
        {
            "name": "TOC with dot leaders (inline)",
            "text": "1.1 Structure of the PMBOK® Guide ....... 1 1.2 Relationship Between the PMBOK® Guide and Business Documents ....... 2 1.3 Changes to the Sixth Edition ....... 3 2.1 Project Life Cycle ....... 19 2.2 Project Phases ....... 20 2.3 Tailoring Considerations ....... 22",
            "expected": "toc"
        },
        {
            "name": "Bibliography with numbered URLs",
            "text": "[16] http://www.iso.org/iso/catalogue_detail?csnumber=50003 [17] http://www.iso.org/iso/home/standards/iso31000.htm [18] http://www.pmi.org/learning/library/risk-analysis-project-management-7070 [19] http://www.sei.cmu.edu/reports/10tr045.pdf [20] http://www.computer.org/portal/web/swebok",
            "expected": "bibliography"
        },
        {
            "name": "Regular prose (should NOT detect)",
            "text": "Project management involves planning, organizing, and controlling resources. Section 2.1 discusses project phases. The project life cycle typically has 4-6 phases. According to Smith (2020), best practices include stakeholder engagement. For more details, see http://www.pmi.org/standards.",
            "expected": None
        },
        {
            "name": "Technical content with references (should NOT detect)",
            "text": "The ISO 9001 standard (http://www.iso.org) defines quality management. Studies [1][2][3] show effectiveness. Implementation involves documentation, training, and auditing. See Chapter 3.2.1 for details. Organizations typically achieve certification within 12-18 months.",
            "expected": None
        }
    ]

    correct = 0
    incorrect = 0

    for test in test_cases:
        is_toc, toc_type = is_toc_or_lof(test["text"])
        is_bib = is_bibliography_section(test["text"])

        detected = is_toc or is_bib
        detection_type = toc_type if is_toc else ("bibliography" if is_bib else None)

        if test["expected"] == detection_type or (test["expected"] is None and not detected):
            correct += 1
            logger.info(f"✓ {test['name']}: Correct (expected={test['expected']}, got={detection_type})")
        else:
            incorrect += 1
            logger.warning(f"✗ {test['name']}: Incorrect (expected={test['expected']}, got={detection_type})")
            preview = test["text"][:200]
            logger.warning(f"  Preview: {preview}...")

    logger.info(f"\nResults: {correct}/{len(test_cases)} correct, {incorrect} incorrect")

    if correct == len(test_cases):
        logger.info("✓ SUCCESS: All synthetic tests passed!")
    else:
        logger.warning(f"⚠ PARTIAL: {incorrect} synthetic tests failed")

    return correct, incorrect


def test_false_positives_on_prose(storage: PostgreSQLStorage, sample_size: int = 100):
    """Test that legitimate prose chunks are NOT incorrectly flagged."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Check for false positives on legitimate prose chunks")
    logger.info("="*80)

    # Fetch random prose chunks from thesis_sources
    from fileintel.storage.models import DocumentChunk, Collection
    from sqlalchemy import func

    # Get collection ID
    collection = storage.document_storage.db.query(Collection).filter(Collection.name == 'thesis_sources').first()
    if not collection:
        logger.error("Collection 'thesis_sources' not found!")
        return 0, 0

    # Get random prose chunks (assume chunks without explicit semantic_type are prose)
    prose_chunks = storage.document_storage.db.query(DocumentChunk).filter(
        DocumentChunk.collection_id == collection.id,
        DocumentChunk.chunk_text != None,
        func.length(DocumentChunk.chunk_text) > 500  # Only substantive chunks
    ).order_by(func.random()).limit(sample_size).all()

    logger.info(f"Testing {len(prose_chunks)} random chunks from 'thesis_sources'...")

    false_positive_count = 0
    true_negative_count = 0

    for i, chunk in enumerate(prose_chunks, 1):
        # Test detection
        is_toc, toc_type = is_toc_or_lof(chunk.chunk_text)
        is_bib = is_bibliography_section(chunk.chunk_text)

        detected = is_toc or is_bib

        if detected:
            false_positive_count += 1
            detection_type = toc_type if is_toc else "bibliography"
            logger.warning(f"⚠ FALSE POSITIVE #{false_positive_count}: Chunk {chunk.id[:8]} flagged as {detection_type}")
            logger.warning(f"  Length: {len(chunk.chunk_text)} chars")
            preview = chunk.chunk_text[:400].replace('\n', ' ')
            logger.warning(f"  Preview: {preview}...")

            # Show detection details for debugging
            dot_count = chunk.chunk_text[:2000].count('...')
            url_count = chunk.chunk_text.count('http://') + chunk.chunk_text.count('https://')
            logger.warning(f"  Metrics: dots={dot_count}, urls={url_count}")
        else:
            true_negative_count += 1
            if i <= 5:  # Show first 5 correct negatives
                preview = chunk.chunk_text[:100].replace('\n', ' ')
                logger.debug(f"✓ Chunk {i}: Correctly NOT flagged. Preview: {preview}...")

    logger.info(f"\nResults: {true_negative_count}/{len(prose_chunks)} correctly NOT flagged")
    logger.info(f"False positives: {false_positive_count}/{len(prose_chunks)} ({false_positive_count/len(prose_chunks)*100:.1f}%)")

    if false_positive_count == 0:
        logger.info("✓ SUCCESS: No false positives detected!")
    elif false_positive_count <= 2:
        logger.warning(f"⚠ ACCEPTABLE: Only {false_positive_count} false positives (< 2%)")
    else:
        logger.error(f"✗ FAILURE: {false_positive_count} false positives (> 2%) - needs adjustment")

    return true_negative_count, false_positive_count


def main():
    """Run all validation tests."""
    logger.info("="*80)
    logger.info("IMPROVED DETECTION VALIDATION TEST")
    logger.info("="*80)

    # Test 1: Synthetic examples should be detected correctly
    correct_synthetic, incorrect_synthetic = test_on_synthetic_examples()

    # Test 2: Random prose chunks should NOT be flagged
    config = get_config()
    storage = PostgreSQLStorage(config)
    true_negatives, false_positives = test_false_positives_on_prose(storage, sample_size=100)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Test 1 (Synthetic examples): {correct_synthetic}/4 correct, {incorrect_synthetic} incorrect")
    logger.info(f"Test 2 (False positive check): {false_positives}/100 false positives ({false_positives}%)")

    # Overall verdict
    if correct_synthetic >= 3 and false_positives <= 2:
        logger.info("\n✓ VALIDATION PASSED: Improved detection is working well!")
        logger.info("  - Correctly detects TOC/bibliography patterns")
        logger.info("  - Very low false positive rate (< 2%)")
        return 0
    else:
        logger.warning("\n⚠ VALIDATION NEEDS REVIEW:")
        if correct_synthetic < 3:
            logger.warning(f"  - Only {correct_synthetic}/4 synthetic tests passed (target: 3+)")
        if false_positives > 2:
            logger.warning(f"  - {false_positives} false positives (target: ≤ 2)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
