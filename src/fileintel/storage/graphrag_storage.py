"""
GraphRAG storage operations for entities, communities, and relationships.

Handles GraphRAG-specific storage operations separated from
core document storage functionality.
"""

import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy import text, and_
from .base_storage import BaseStorageInfrastructure
from .models import GraphRAGIndex, GraphRAGEntity, GraphRAGCommunity

logger = logging.getLogger(__name__)


class GraphRAGStorage:
    """
    Handles GraphRAG-specific storage operations.

    Manages GraphRAG indexes, entities, communities, and relationships
    separated from basic document storage.
    """

    def __init__(self, config_or_session):
        """Initialize with shared infrastructure."""
        self.base = BaseStorageInfrastructure(config_or_session)
        self.db = self.base.db

    # GraphRAG Index Operations
    def save_graphrag_index_info(
        self,
        collection_id: str,
        index_path: str,
        documents_count: int = 0,
        entities_count: int = 0,
        communities_count: int = 0,
    ) -> None:
        """Save GraphRAG index information."""
        try:
            # Check if index info already exists
            existing = self.get_graphrag_index_info(collection_id)

            if existing:
                # Update existing
                index_info = (
                    self.db.query(GraphRAGIndex)
                    .filter(GraphRAGIndex.collection_id == collection_id)
                    .first()
                )

                if index_info:
                    index_info.index_path = index_path
                    index_info.documents_count = documents_count
                    index_info.entities_count = entities_count
                    index_info.communities_count = communities_count
            else:
                # Create new
                index_info = GraphRAGIndex(
                    id=str(uuid.uuid4()),
                    collection_id=collection_id,
                    index_path=index_path,
                    documents_count=documents_count,
                    entities_count=entities_count,
                    communities_count=communities_count,
                )
                self.db.add(index_info)

            self.base._safe_commit()
            logger.info(f"Saved GraphRAG index info for collection {collection_id}")

        except Exception as e:
            logger.error(f"Error saving GraphRAG index info: {e}")
            self.base._handle_session_error(e)

    def get_graphrag_index_info(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get GraphRAG index information for a collection."""
        try:
            index_info = (
                self.db.query(GraphRAGIndex)
                .filter(GraphRAGIndex.collection_id == collection_id)
                .first()
            )

            if index_info:
                return {
                    "collection_id": index_info.collection_id,
                    "index_path": index_info.index_path,
                    "index_status": index_info.index_status,  # Include status for resume logic
                    "documents_count": index_info.documents_count,
                    "entities_count": index_info.entities_count,
                    "communities_count": index_info.communities_count,
                    "created_at": index_info.created_at,
                    "updated_at": index_info.updated_at,
                }

            return None

        except Exception as e:
            logger.error(f"Error getting GraphRAG index info: {e}")
            return None

    def update_graphrag_index_status(self, collection_id: str, status: str) -> bool:
        """
        Update GraphRAG index status.

        Args:
            collection_id: Collection ID
            status: New status (building, ready, failed, updating)

        Returns:
            True if successful, False otherwise
        """
        try:
            index_info = (
                self.db.query(GraphRAGIndex)
                .filter(GraphRAGIndex.collection_id == collection_id)
                .first()
            )

            if index_info:
                index_info.index_status = status
                self.base._safe_commit()
                logger.info(
                    f"Updated GraphRAG index status to '{status}' for collection {collection_id}"
                )
                return True
            else:
                logger.warning(
                    f"Cannot update status: No GraphRAG index found for collection {collection_id}"
                )
                return False

        except Exception as e:
            logger.error(f"Error updating GraphRAG index status: {e}")
            self.base._handle_session_error(e)
            return False

    def remove_graphrag_index_info(self, collection_id: str) -> bool:
        """Remove GraphRAG index information for a collection."""
        try:
            index_info = (
                self.db.query(GraphRAGIndex)
                .filter(GraphRAGIndex.collection_id == collection_id)
                .first()
            )

            if index_info:
                self.db.delete(index_info)
                self.base._safe_commit()
                logger.info(
                    f"Removed GraphRAG index info for collection {collection_id}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error removing GraphRAG index info: {e}")
            self.base._handle_session_error(e)
            return False

    # GraphRAG Entity Operations
    def save_graphrag_entities(self, collection_id: str, entities: List[dict]) -> None:
        """Save GraphRAG entities for a collection."""
        try:
            # Filter out contaminated entities before saving
            from fileintel.utils.entity_filter import filter_entities
            filtered_entities = filter_entities(entities)

            # Clear existing entities for this collection
            self.db.query(GraphRAGEntity).filter(
                GraphRAGEntity.collection_id == collection_id
            ).delete()

            # Add new filtered entities
            for entity_data in filtered_entities:
                entity = GraphRAGEntity(
                    id=str(uuid.uuid4()),
                    collection_id=collection_id,
                    entity_name=self.base._clean_text(entity_data.get("title", "")),  # GraphRAG uses "title" not "name"
                    entity_type=entity_data.get("type"),
                    description=self.base._clean_text(
                        entity_data.get("description", "")
                    ),
                    importance_score=entity_data.get("degree", 0),  # GraphRAG uses "degree" not "rank"
                    entity_metadata=entity_data,  # Store full data as JSON
                )
                self.db.add(entity)

            self.base._safe_commit()
            logger.info(
                f"Saved {len(filtered_entities)} GraphRAG entities for collection {collection_id} "
                f"(filtered from {len(entities)} total)"
            )

        except Exception as e:
            logger.error(f"Error saving GraphRAG entities: {e}")
            self.base._handle_session_error(e)

    def get_graphrag_entities(
        self, collection_id: str, limit: Optional[int] = None, offset: Optional[int] = 0
    ) -> List[dict]:
        """
        Get GraphRAG entities for a collection with pagination support.

        Args:
            collection_id: Collection ID
            limit: Maximum number of entities to return (None for all)
            offset: Number of entities to skip (default: 0)

        Returns:
            List of entity dictionaries ordered by importance_score descending
        """
        try:
            query = (
                self.db.query(GraphRAGEntity)
                .filter(GraphRAGEntity.collection_id == collection_id)
                .order_by(GraphRAGEntity.importance_score.desc())
            )

            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            entities = query.all()

            entity_list = []
            for entity in entities:
                entity_dict = {
                    "id": entity.id,
                    "name": entity.entity_name,
                    "type": entity.entity_type,
                    "description": entity.description,
                    "importance_score": entity.importance_score,
                    "collection_id": entity.collection_id,
                }

                # Add full entity metadata if available
                if entity.entity_metadata:
                    entity_dict.update(entity.entity_metadata)

                entity_list.append(self.base._clean_result_data(entity_dict))

            logger.info(
                f"Retrieved {len(entity_list)} GraphRAG entities for collection {collection_id}"
            )
            return entity_list

        except Exception as e:
            logger.error(f"Error getting GraphRAG entities: {e}")
            return []

    def _get_graphrag_entities_from_db(
        self, collection_id: str, entity_ids: List[str] = None
    ) -> List[dict]:
        """Get specific GraphRAG entities from database."""
        try:
            query = self.db.query(GraphRAGEntity).filter(
                GraphRAGEntity.collection_id == collection_id
            )

            if entity_ids:
                query = query.filter(GraphRAGEntity.entity_id.in_(entity_ids))

            entities = query.all()

            entity_list = []
            for entity in entities:
                entity_dict = {
                    "id": entity.entity_id,
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "source_id": entity.source_id,
                }

                if entity.entity_data:
                    entity_dict.update(entity.entity_data)

                entity_list.append(entity_dict)

            return entity_list

        except Exception as e:
            logger.error(f"Error getting entities from database: {e}")
            return []

    # GraphRAG Community Operations
    def save_graphrag_communities(
        self, collection_id: str, communities: List[dict]
    ) -> None:
        """Save GraphRAG communities for a collection."""
        try:
            # Filter out communities containing contaminated entities
            from fileintel.utils.entity_filter import filter_communities
            filtered_communities = filter_communities(communities)

            # Clear existing communities for this collection
            self.db.query(GraphRAGCommunity).filter(
                GraphRAGCommunity.collection_id == collection_id
            ).delete()

            # Add new filtered communities
            for community_data in filtered_communities:
                community = GraphRAGCommunity(
                    id=str(uuid.uuid4()),
                    collection_id=collection_id,
                    community_id=community_data.get("community"),  # GraphRAG uses "community" field
                    level=community_data.get("level", 0),
                    title=self.base._clean_text(community_data.get("title", "")),
                    summary=self.base._clean_text(community_data.get("summary", "")),  # May not exist in GraphRAG
                    entities=community_data.get("entity_ids", []),  # GraphRAG uses "entity_ids"
                    size=community_data.get("size", 0),
                )
                self.db.add(community)

            self.base._safe_commit()
            logger.info(
                f"Saved {len(filtered_communities)} GraphRAG communities for collection {collection_id} "
                f"(filtered from {len(communities)} total)"
            )

        except Exception as e:
            logger.error(f"Error saving GraphRAG communities: {e}")
            self.base._handle_session_error(e)

    def get_graphrag_communities(
        self, collection_id: str, limit: Optional[int] = None, offset: Optional[int] = 0
    ) -> List[dict]:
        """
        Get GraphRAG communities for a collection with pagination support.

        Args:
            collection_id: Collection ID
            limit: Maximum number of communities to return (None for all)
            offset: Number of communities to skip (default: 0)

        Returns:
            List of community dictionaries ordered by size descending
        """
        try:
            query = (
                self.db.query(GraphRAGCommunity)
                .filter(GraphRAGCommunity.collection_id == collection_id)
                .order_by(GraphRAGCommunity.size.desc())
            )

            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            communities = query.all()

            community_list = []
            for community in communities:
                community_dict = {
                    "id": community.id,
                    "community_id": community.community_id,
                    "level": community.level,
                    "title": community.title,
                    "summary": community.summary,
                    "entities": community.entities or [],
                    "size": community.size,
                    "collection_id": community.collection_id,
                }

                community_list.append(self.base._clean_result_data(community_dict))

            logger.info(
                f"Retrieved {len(community_list)} GraphRAG communities for collection {collection_id}"
            )
            return community_list

        except Exception as e:
            logger.error(f"Error getting GraphRAG communities: {e}")
            return []

    def get_graphrag_community_by_id(
        self, collection_id: str, community_id: int
    ) -> Optional[dict]:
        """
        Get a specific GraphRAG community by its ID.

        Args:
            collection_id: Collection ID
            community_id: Community ID (integer)

        Returns:
            Community dictionary or None if not found
        """
        try:
            community = (
                self.db.query(GraphRAGCommunity)
                .filter(
                    GraphRAGCommunity.collection_id == collection_id,
                    GraphRAGCommunity.community_id == community_id
                )
                .first()
            )

            if not community:
                return None

            community_dict = {
                "id": community.id,
                "community_id": community.community_id,
                "level": community.level,
                "title": community.title,
                "summary": community.summary,
                "entities": community.entities or [],
                "size": community.size,
                "collection_id": community.collection_id,
            }

            return self.base._clean_result_data(community_dict)

        except Exception as e:
            logger.error(f"Error getting community by ID: {e}")
            return None

    def _get_graphrag_communities_from_db(self, collection_id: str) -> List[dict]:
        """Get GraphRAG communities from database."""
        try:
            communities = (
                self.db.query(GraphRAGCommunity)
                .filter(GraphRAGCommunity.collection_id == collection_id)
                .order_by(GraphRAGCommunity.size.desc())
                .all()
            )

            community_list = []
            for community in communities:
                community_dict = {
                    "id": community.id,
                    "community_id": community.community_id,
                    "level": community.level,
                    "title": community.title,
                    "summary": community.summary,
                    "entities": community.entities or [],
                    "size": community.size,
                    "collection_id": community.collection_id,
                }

                community_list.append(community_dict)

            return community_list

        except Exception as e:
            logger.error(f"Error getting communities from database: {e}")
            return []

    def get_graphrag_relationships(self, collection_id: str) -> List[dict]:
        """
        Get GraphRAG relationships for a collection.

        Note: This extracts relationships from entity data since there's
        no separate relationships table in the current schema.
        """
        try:
            entities = self.get_graphrag_entities(collection_id)
            relationships = []

            for entity in entities:
                # Extract relationships from entity data
                if isinstance(entity.get("entity_data"), dict):
                    entity_relationships = entity["entity_data"].get(
                        "relationships", []
                    )
                    for rel in entity_relationships:
                        relationship = {
                            "source": entity.get("id"),
                            "target": rel.get("target"),
                            "relationship": rel.get("relationship", ""),
                            "description": self.base._clean_text(
                                rel.get("description", "")
                            ),
                            "weight": rel.get("weight", 1.0),
                            "collection_id": collection_id,
                        }
                        relationships.append(self.base._clean_result_data(relationship))

            logger.info(
                f"Retrieved {len(relationships)} GraphRAG relationships for collection {collection_id}"
            )
            return relationships

        except Exception as e:
            logger.error(f"Error getting GraphRAG relationships: {e}")
            return []

    def close(self):
        """Close the storage connection."""
        self.base.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
