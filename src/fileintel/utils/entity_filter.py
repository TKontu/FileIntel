"""
Entity filtering utilities for GraphRAG.

Filters out contaminated entities from GraphRAG extraction results.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class EntityFilter:
    """Filters entities based on contamination patterns."""

    def __init__(self):
        """Initialize entity filter with contamination patterns."""
        # Patterns to filter out
        self.contamination_patterns = [
            r'^EXAMPLE_.*',  # Entities starting with EXAMPLE_
            r'^Example.*',   # Entities starting with Example
            r'^example.*',   # Entities starting with example (lowercase)
            r'^STORY$',      # Generic "STORY" entity
            r'^STORY_STARTER$',  # Story starter entity
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.contamination_patterns]

    def is_contaminated_entity(self, entity_name: str) -> bool:
        """
        Check if an entity name matches contamination patterns.

        Args:
            entity_name: Name of the entity to check

        Returns:
            True if entity is contaminated and should be filtered out
        """
        if not entity_name or not isinstance(entity_name, str):
            return True  # Filter out empty or invalid names

        entity_name = entity_name.strip()

        # Check against contamination patterns
        for pattern in self.compiled_patterns:
            if pattern.match(entity_name):
                logger.debug(f"Filtering contaminated entity: {entity_name}")
                return True

        return False

    def filter_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of entities to remove contaminated ones.

        Args:
            entities: List of entity dictionaries

        Returns:
            Filtered list of entities with contamination removed
        """
        if not entities:
            return entities

        original_count = len(entities)
        filtered_entities = []

        for entity in entities:
            # Extract entity name - handle different formats
            entity_name = self._extract_entity_name(entity)

            if not self.is_contaminated_entity(entity_name):
                filtered_entities.append(entity)
            else:
                logger.debug(f"Filtered out contaminated entity: {entity_name}")

        filtered_count = len(filtered_entities)
        removed_count = original_count - filtered_count

        if removed_count > 0:
            logger.info(f"Entity filter removed {removed_count} contaminated entities out of {original_count}")

        return filtered_entities

    def filter_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter relationships to remove those involving contaminated entities.

        Args:
            relationships: List of relationship dictionaries

        Returns:
            Filtered list of relationships
        """
        if not relationships:
            return relationships

        original_count = len(relationships)
        filtered_relationships = []

        for relationship in relationships:
            source_entity = relationship.get("source", "")
            target_entity = relationship.get("target", "")

            # Filter out relationships involving contaminated entities
            if (not self.is_contaminated_entity(source_entity) and
                not self.is_contaminated_entity(target_entity)):
                filtered_relationships.append(relationship)
            else:
                logger.debug(f"Filtered out relationship involving contaminated entities: {source_entity} -> {target_entity}")

        filtered_count = len(filtered_relationships)
        removed_count = original_count - filtered_count

        if removed_count > 0:
            logger.info(f"Relationship filter removed {removed_count} contaminated relationships out of {original_count}")

        return filtered_relationships

    def filter_communities(self, communities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter communities to remove those containing contaminated entities.

        Args:
            communities: List of community dictionaries

        Returns:
            Filtered list of communities
        """
        if not communities:
            return communities

        original_count = len(communities)
        filtered_communities = []

        for community in communities:
            # Get entity list from community
            entity_ids = community.get("entity_ids", [])
            entities = community.get("entities", [])

            # Check both possible entity fields
            all_entities = entity_ids + entities

            # Check if any entity in the community is contaminated
            has_contaminated_entity = False
            for entity_name in all_entities:
                if self.is_contaminated_entity(str(entity_name)):
                    has_contaminated_entity = True
                    break

            if not has_contaminated_entity:
                filtered_communities.append(community)
            else:
                community_title = community.get("title", "Unknown")
                logger.debug(f"Filtered out community containing contaminated entities: {community_title}")

        filtered_count = len(filtered_communities)
        removed_count = original_count - filtered_count

        if removed_count > 0:
            logger.info(f"Community filter removed {removed_count} contaminated communities out of {original_count}")

        return filtered_communities

    def _extract_entity_name(self, entity: Dict[str, Any]) -> str:
        """
        Extract entity name from entity dictionary, handling various formats.

        Args:
            entity: Entity dictionary

        Returns:
            Entity name string
        """
        # Try different possible keys for entity name
        possible_keys = ["name", "entity_name", "title", "id"]

        for key in possible_keys:
            if key in entity and entity[key]:
                return str(entity[key]).strip()

        # Fallback: convert entire entity to string and try to extract name
        entity_str = str(entity)
        logger.warning(f"Could not extract entity name from: {entity_str[:100]}...")
        return ""


# Module-level convenience functions
_filter = EntityFilter()

def filter_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter entities using the default filter."""
    return _filter.filter_entities(entities)

def filter_relationships(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter relationships using the default filter."""
    return _filter.filter_relationships(relationships)

def filter_communities(communities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter communities using the default filter."""
    return _filter.filter_communities(communities)

def is_contaminated_entity(entity_name: str) -> bool:
    """Check if entity name is contaminated using the default filter."""
    return _filter.is_contaminated_entity(entity_name)