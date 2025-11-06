#!/usr/bin/env python3
"""
Backfill GraphRAG entities and communities into PostgreSQL database.

This script reads existing GraphRAG parquet files and populates the database
tables for fast API display.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fileintel.storage.graphrag_storage import GraphRAGStorage
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.core.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_value(value):
    """Clean values for database storage."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return int(value) if isinstance(value, np.integer) else float(value)
    if isinstance(value, list):
        return value
    return str(value)


def backfill_collection(graphrag_storage, base_storage, collection_id):
    """
    Backfill GraphRAG data for a single collection.

    Args:
        graphrag_storage: GraphRAGStorage instance
        base_storage: BaseStorage instance
        collection_id: Collection ID to backfill

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing collection: {collection_id}")
        logger.info(f"{'='*60}")

        # Get collection info
        collection = base_storage.get_collection(collection_id)
        if not collection:
            logger.error(f"  Collection not found: {collection_id}")
            return False

        logger.info(f"  Collection name: {collection.name if collection.name else 'Unknown'}")

        # Check if GraphRAG index exists
        index_info = graphrag_storage.get_graphrag_index_info(collection_id)
        if not index_info:
            logger.info(f"  No GraphRAG index found - skipping")
            return False

        workspace = index_info.get('index_path')
        if not workspace:
            logger.error(f"  Index exists but no path configured")
            return False

        logger.info(f"  Index path: {workspace}")

        # Check if workspace exists
        if not os.path.exists(workspace):
            logger.warning(f"  Workspace directory not found: {workspace}")
            return False

        # Check if data already exists in database
        existing_entities = graphrag_storage.get_graphrag_entities(collection_id)
        existing_communities = graphrag_storage.get_graphrag_communities(collection_id)

        if existing_entities or existing_communities:
            logger.info(f"  Database already has data:")
            logger.info(f"    - Entities: {len(existing_entities)}")
            logger.info(f"    - Communities: {len(existing_communities)}")
            response = input("  Overwrite existing data? (y/N): ").strip().lower()
            if response != 'y':
                logger.info(f"  Skipping collection")
                return False

        # Load and backfill entities
        entities_file = os.path.join(workspace, "entities.parquet")
        if os.path.exists(entities_file):
            logger.info(f"  Loading entities from: {entities_file}")
            try:
                entities_df = pd.read_parquet(entities_file)
                logger.info(f"    Loaded {len(entities_df)} entities from parquet")

                # Convert to list of dicts
                entities_data = entities_df.to_dict('records')

                # Clean values
                entities_clean = [
                    {k: clean_value(v) for k, v in entity.items()}
                    for entity in entities_data
                ]

                # Save to database (this includes filtering)
                graphrag_storage.save_graphrag_entities(collection_id, entities_clean)
                logger.info(f"    ✓ Saved entities to database")
                logger.info(f"    Note: Count may differ from parquet due to entity filtering")

            except Exception as e:
                logger.error(f"    Error processing entities: {e}")
                return False
        else:
            logger.warning(f"  Entities file not found: {entities_file}")

        # Load and backfill communities - merge structure and content files for complete data
        communities_structure_file = os.path.join(workspace, "communities.parquet")
        communities_content_file = os.path.join(workspace, "community_reports.parquet")

        if os.path.exists(communities_structure_file) and os.path.exists(communities_content_file):
            logger.info(f"  Loading communities from both structure and content files")
            try:
                # Load both files
                communities_df = pd.read_parquet(communities_structure_file)  # Has entity_ids
                reports_df = pd.read_parquet(communities_content_file)  # Has summary, findings

                logger.info(f"    Loaded {len(communities_df)} community structures")
                logger.info(f"    Loaded {len(reports_df)} community reports")

                # Merge on community and level for complete data
                merged_df = pd.merge(
                    communities_df,
                    reports_df[['community', 'level', 'summary', 'full_content', 'findings', 'rank', 'rating_explanation']],
                    on=['community', 'level'],
                    how='left'
                )

                logger.info(f"    Merged into {len(merged_df)} complete community records")

                # Convert to list of dicts
                communities_data = merged_df.to_dict('records')

                # Clean values
                communities_clean = [
                    {k: clean_value(v) for k, v in community.items()}
                    for community in communities_data
                ]

                # Save to database (this includes filtering)
                graphrag_storage.save_graphrag_communities(collection_id, communities_clean)
                logger.info(f"    ✓ Saved communities with full data (structure + summaries + entity_ids)")
                logger.info(f"    Note: Count may differ from parquet due to community filtering")

            except Exception as e:
                logger.error(f"    Error processing communities: {e}")
                import traceback
                traceback.print_exc()
                return False
        elif os.path.exists(communities_structure_file):
            logger.warning(f"  Only structure file found - summaries will be empty")
            logger.warning(f"  Missing: {communities_content_file}")
            try:
                communities_df = pd.read_parquet(communities_structure_file)
                communities_data = communities_df.to_dict('records')
                communities_clean = [
                    {k: clean_value(v) for k, v in community.items()}
                    for community in communities_data
                ]
                graphrag_storage.save_graphrag_communities(collection_id, communities_clean)
                logger.info(f"    ✓ Saved {len(communities_clean)} communities (structure only - no summaries)")
            except Exception as e:
                logger.error(f"    Error processing communities: {e}")
                return False
        else:
            logger.warning(f"  Communities files not found:")
            logger.warning(f"    {communities_structure_file}")
            logger.warning(f"    {communities_content_file}")

        # Verify what was saved
        final_entities = graphrag_storage.get_graphrag_entities(collection_id)
        final_communities = graphrag_storage.get_graphrag_communities(collection_id)

        logger.info(f"  ✓ Backfill complete:")
        logger.info(f"    - Entities in database: {len(final_entities)}")
        logger.info(f"    - Communities in database: {len(final_communities)}")

        return True

    except Exception as e:
        logger.error(f"  Error backfilling collection {collection_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main backfill process."""
    try:
        logger.info("Starting GraphRAG database backfill")
        logger.info("="*60)

        # Initialize storage
        config = get_config()
        base_storage = PostgreSQLStorage(config)
        graphrag_storage = GraphRAGStorage(config)

        # Get all collections
        collections = base_storage.get_all_collections()
        logger.info(f"Found {len(collections)} total collections")

        # Filter to collections with GraphRAG indexes
        collections_with_indexes = []
        for collection in collections:
            collection_id = collection.id
            index_info = graphrag_storage.get_graphrag_index_info(collection_id)
            if index_info:
                collections_with_indexes.append(collection)

        logger.info(f"Found {len(collections_with_indexes)} collections with GraphRAG indexes")

        if not collections_with_indexes:
            logger.info("No collections to backfill. Exiting.")
            return

        # Process each collection
        success_count = 0
        skip_count = 0
        fail_count = 0

        for collection in collections_with_indexes:
            collection_id = collection.id
            result = backfill_collection(graphrag_storage, base_storage, collection_id)

            if result:
                success_count += 1
            elif result is False:
                fail_count += 1
            else:
                skip_count += 1

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Backfill Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Collections processed: {len(collections_with_indexes)}")
        logger.info(f"  - Successful: {success_count}")
        logger.info(f"  - Failed: {fail_count}")
        logger.info(f"  - Skipped: {skip_count}")

        # Close storage
        base_storage.close()
        graphrag_storage.close()

        logger.info("\nBackfill complete!")

    except Exception as e:
        logger.error(f"Fatal error during backfill: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
