#!/usr/bin/env python3
"""
Cleanup duplicate documents in a collection.

This script identifies and removes duplicate documents based on content_hash,
keeping the oldest copy of each duplicate.

Usage:
    # Dry run (show what would be deleted)
    poetry run python scripts/cleanup_duplicates.py <collection_name_or_id>

    # Actually delete duplicates
    poetry run python scripts/cleanup_duplicates.py <collection_name_or_id> --execute

Example:
    poetry run python scripts/cleanup_duplicates.py thesis_sources
    poetry run python scripts/cleanup_duplicates.py thesis_sources --execute
"""

import sys
import logging
from typing import List, Tuple
from sqlalchemy import func

# Add parent directory to path for imports
sys.path.insert(0, '/home/appuser/app')

from src.fileintel.storage.postgresql_storage import PostgreSQLStorage
from src.fileintel.core.config import get_config
from src.fileintel.storage.models import Document

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def find_duplicates(storage: PostgreSQLStorage, collection_id: str) -> List[Tuple]:
    """
    Find all duplicate documents in a collection.

    Returns list of tuples: (content_hash, count, [doc_ids])
    """
    # Find content_hashes that appear multiple times in this collection
    duplicates = (
        storage.db.query(
            Document.content_hash,
            func.count(Document.id).label('count'),
            func.array_agg(Document.id).label('doc_ids'),
            func.array_agg(Document.original_filename).label('filenames'),
            func.array_agg(Document.created_at).label('created_ats')
        )
        .filter(Document.collection_id == collection_id)
        .group_by(Document.content_hash)
        .having(func.count(Document.id) > 1)
        .all()
    )

    return duplicates


def cleanup_duplicates(
    storage: PostgreSQLStorage,
    collection_id: str,
    dry_run: bool = True
) -> Tuple[int, int]:
    """
    Remove duplicate documents, keeping the oldest one.

    Args:
        storage: PostgreSQL storage instance
        collection_id: Collection to clean up
        dry_run: If True, only show what would be deleted

    Returns:
        Tuple of (total_duplicates, total_removed)
    """
    duplicates = find_duplicates(storage, collection_id)

    if not duplicates:
        logger.info("No duplicates found in this collection!")
        return 0, 0

    total_duplicate_groups = len(duplicates)
    total_documents_to_remove = 0

    logger.info(f"\nFound {total_duplicate_groups} duplicate groups:")
    logger.info("=" * 80)

    for content_hash, count, doc_ids, filenames, created_ats in duplicates:
        logger.info(f"\nDuplicate group: {content_hash[:16]}... ({count} copies)")

        # Get all documents with this hash
        docs = []
        for i, doc_id in enumerate(doc_ids):
            docs.append({
                'id': doc_id,
                'filename': filenames[i],
                'created_at': created_ats[i]
            })

        # Sort by created_at, keep oldest
        docs.sort(key=lambda d: d['created_at'])
        keep = docs[0]
        remove = docs[1:]

        logger.info(f"  ✓ KEEP:   {keep['id']}")
        logger.info(f"           File: {keep['filename']}")
        logger.info(f"           Created: {keep['created_at']}")

        for doc in remove:
            logger.info(f"  ✗ REMOVE: {doc['id']}")
            logger.info(f"           File: {doc['filename']}")
            logger.info(f"           Created: {doc['created_at']}")
            total_documents_to_remove += 1

            if not dry_run:
                try:
                    # Delete document (cascades to chunks via relationship)
                    document = storage.get_document(doc['id'])
                    if document:
                        storage.db.delete(document)
                        storage.document_storage.base._safe_commit()
                        logger.info(f"           ✓ Deleted")
                    else:
                        logger.warning(f"           ⚠ Document not found")
                except Exception as e:
                    logger.error(f"           ✗ Error deleting: {e}")
                    storage.db.rollback()

    logger.info("\n" + "=" * 80)
    logger.info(f"Summary:")
    logger.info(f"  Duplicate groups: {total_duplicate_groups}")
    logger.info(f"  Documents to remove: {total_documents_to_remove}")

    if dry_run:
        logger.info(f"\n⚠  DRY RUN - No changes made")
        logger.info(f"   Use --execute to actually delete duplicates")
    else:
        logger.info(f"\n✓ Cleanup complete!")

    return total_duplicate_groups, total_documents_to_remove


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    collection_identifier = sys.argv[1]
    dry_run = "--execute" not in sys.argv

    try:
        # Load config and create storage
        config = get_config()
        storage = PostgreSQLStorage(config)

        # Get collection
        collection = storage.get_collection_by_name(collection_identifier)
        if not collection:
            # Try by ID
            collection = storage.get_collection(collection_identifier)

        if not collection:
            logger.error(f"Collection '{collection_identifier}' not found")
            sys.exit(1)

        logger.info(f"Collection: {collection.name} ({collection.id})")

        if dry_run:
            logger.info("=" * 80)
            logger.info("DRY RUN MODE - No changes will be made")
            logger.info("Use --execute to actually delete duplicates")
            logger.info("=" * 80)

        # Find and clean duplicates
        cleanup_duplicates(storage, collection.id, dry_run=dry_run)

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
