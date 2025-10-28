"""
Backfill content_fingerprint for existing documents.

This script:
1. Finds all documents without content_fingerprint
2. Reads file from disk
3. Calculates fingerprint from content
4. Updates database record

Safe to run multiple times (idempotent).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
load_dotenv()

from fileintel.storage.document_storage import DocumentStorage
from fileintel.utils.fingerprint import generate_content_fingerprint
from fileintel.core.config import get_config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_fingerprints(dry_run: bool = False):
    """
    Backfill content_fingerprint for documents that don't have one.

    Args:
        dry_run: If True, don't commit changes to database
    """
    config = get_config()
    storage = DocumentStorage(config)

    # Get all documents
    logger.info("Fetching documents from database...")
    from fileintel.storage.models import Document
    all_documents = storage.db.query(Document).all()

    logger.info(f"Found {len(all_documents)} total documents")

    # Filter documents without fingerprint
    documents_to_update = [
        doc for doc in all_documents
        if not doc.content_fingerprint
    ]

    logger.info(f"Documents without fingerprint: {len(documents_to_update)}")

    if not documents_to_update:
        logger.info("All documents already have fingerprints. Nothing to do.")
        return

    if dry_run:
        logger.info("DRY RUN MODE - No changes will be committed")

    # Track statistics
    stats = {
        'updated': 0,
        'skipped_missing_file': 0,
        'skipped_no_path': 0,
        'errors': 0
    }

    for i, doc in enumerate(documents_to_update, 1):
        doc_id = doc.id
        original_filename = doc.original_filename

        logger.info(f"\n[{i}/{len(documents_to_update)}] Processing: {original_filename} ({doc_id})")

        # Get file path from metadata
        file_path_str = doc.document_metadata.get('file_path') if doc.document_metadata else None

        if not file_path_str:
            logger.warning(f"  No file_path in metadata, skipping")
            stats['skipped_no_path'] += 1
            continue

        file_path = Path(file_path_str)

        # Check if file exists
        if not file_path.exists():
            logger.warning(f"  File not found: {file_path}")
            stats['skipped_missing_file'] += 1
            continue

        try:
            # Calculate fingerprint from file content
            logger.info(f"  Reading file: {file_path}")
            fingerprint = generate_content_fingerprint(file_path)
            logger.info(f"  Calculated fingerprint: {fingerprint}")

            # Check if fingerprint already exists (duplicate detection)
            existing_doc = storage.get_document_by_fingerprint(fingerprint)
            if existing_doc and existing_doc.id != doc_id:
                logger.warning(
                    f"  DUPLICATE DETECTED: Fingerprint {fingerprint} already exists "
                    f"for document {existing_doc.id} ({existing_doc.original_filename})"
                )
                # Still update this document's fingerprint for consistency
                # but log the duplicate for manual review

            if not dry_run:
                # Update document
                doc.content_fingerprint = fingerprint
                storage.base._safe_commit()
                logger.info(f"  ✓ Updated document with fingerprint")
            else:
                logger.info(f"  [DRY RUN] Would update with fingerprint: {fingerprint}")

            stats['updated'] += 1

        except Exception as e:
            logger.error(f"  ✗ Error processing document {doc_id}: {e}")
            stats['errors'] += 1
            # Continue with next document

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total documents processed: {len(documents_to_update)}")
    logger.info(f"  ✓ Updated:               {stats['updated']}")
    logger.info(f"  ⊘ Skipped (no file path): {stats['skipped_no_path']}")
    logger.info(f"  ⊘ Skipped (file missing): {stats['skipped_missing_file']}")
    logger.info(f"  ✗ Errors:                {stats['errors']}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("\nThis was a DRY RUN. No changes were committed to the database.")
        logger.info("Run without --dry-run to apply changes.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Backfill content_fingerprint for existing documents'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (no database changes)'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CONTENT FINGERPRINT BACKFILL SCRIPT")
    logger.info("=" * 60)

    try:
        backfill_fingerprints(dry_run=args.dry_run)
        logger.info("\n✓ Script completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n✗ Script failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
