#!/usr/bin/env python3
"""Fix GraphRAG index status on remote server via storage layer"""

import os
import sys

# Set environment to point to remote server
os.environ['DB_HOST'] = '192.168.0.136'
os.environ['DB_PORT'] = '5432'
os.environ['DB_USER'] = 'user'
os.environ['DB_PASSWORD'] = 'password'
os.environ['DB_NAME'] = 'fileintel'

from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.config import Config

COLLECTION_ID = '6525aacb-55b1-4a88-aaaa-a4211d03beba'

def main():
    print("Connecting to remote database at 192.168.0.136...")

    # Create storage connection
    config = Config()
    storage = PostgreSQLStorage(config)

    try:
        # Get current status
        index_info = storage.get_graphrag_index_info(COLLECTION_ID)
        if not index_info:
            print(f"❌ No GraphRAG index found for collection {COLLECTION_ID}")
            return 1

        print(f"\nCurrent status: {index_info.get('index_status')}")
        print(f"Documents: {index_info.get('documents_count')}")
        print(f"Entities: {index_info.get('entities_count')}")
        print(f"Communities: {index_info.get('communities_count')}")

        # Update status
        print(f"\n Updating status to 'building'...")
        success = storage.update_graphrag_index_status(COLLECTION_ID, 'building')

        if success:
            # Verify update
            updated_info = storage.get_graphrag_index_info(COLLECTION_ID)
            print(f"\n✓ Status updated to: {updated_info.get('index_status')}")
            print("\nYou can now run: fileintel graphrag index thesis_sources")
            return 0
        else:
            print("❌ Failed to update status")
            return 1

    finally:
        storage.close()

if __name__ == '__main__':
    sys.exit(main())
