#!/usr/bin/env python3
"""
Reset GraphRAG index status to 'building' to enable resume.

This allows the checkpoint system to detect incomplete workflows
and resume from where it left off, without re-doing completed work.

Usage:
    python reset_graphrag_status.py <collection_id>

Example:
    python reset_graphrag_status.py 6525aacb-55b1-4a88-aaaa-a4211d03beba
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.core.config import Settings


def reset_status(collection_id: str):
    """Reset GraphRAG index status to 'building' to enable resume."""

    print(f"Connecting to database...")
    settings = Settings()
    storage = PostgreSQLStorage(settings)

    # Get current status
    index_info = storage.get_graphrag_index_info(collection_id)

    if not index_info:
        print(f"❌ Error: No GraphRAG index found for collection {collection_id}")
        return False

    current_status = index_info.get("index_status", "unknown")
    index_path = index_info.get("index_path", "unknown")

    print(f"\nCurrent status:")
    print(f"  Collection ID: {collection_id}")
    print(f"  Status: {current_status}")
    print(f"  Index path: {index_path}")
    print(f"  Entities: {index_info.get('entities_count', 0)}")
    print(f"  Communities: {index_info.get('communities_count', 0)}")

    if current_status == "building":
        print(f"\n✓ Status is already 'building' - no change needed")
        return True

    # Reset status to 'building'
    print(f"\nResetting status to 'building'...")
    storage.update_graphrag_index_status(collection_id, "building")

    # Verify
    updated_info = storage.get_graphrag_index_info(collection_id)
    new_status = updated_info.get("index_status", "unknown")

    if new_status == "building":
        print(f"✓ Status successfully reset to 'building'")
        print(f"\nYou can now run:")
        print(f"  fileintel graphrag index thesis_sources")
        print(f"\nThe checkpoint system will:")
        print(f"  • Detect incomplete 'generate_text_embeddings' workflow")
        print(f"  • Resume from batch 167/294")
        print(f"  • Complete remaining ~63,616 text unit embeddings")
        print(f"  • Keep all existing entities, communities, and summaries")
        return True
    else:
        print(f"❌ Error: Status is '{new_status}', expected 'building'")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reset_graphrag_status.py <collection_id>")
        print("\nExample:")
        print("  python reset_graphrag_status.py 6525aacb-55b1-4a88-aaaa-a4211d03beba")
        sys.exit(1)

    collection_id = sys.argv[1]
    success = reset_status(collection_id)
    sys.exit(0 if success else 1)
