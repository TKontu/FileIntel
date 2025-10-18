"""
Document structure storage operations.

Handles storage and retrieval of extracted document structures
(TOC, LOF, LOT, headers) from Phase 2 filtering.
"""

import logging
from typing import List, Dict, Any, Optional
from .base_storage import BaseStorageInfrastructure
from .models import DocumentStructure

logger = logging.getLogger(__name__)


class DocumentStructureStorage:
    """
    Handles document structure storage operations.

    Stores and retrieves extracted structures (TOC, LOF, headers)
    that are filtered out during RAG processing but preserved
    for navigation and querying.
    """

    def __init__(self, config_or_session):
        """Initialize with shared infrastructure."""
        self.base = BaseStorageInfrastructure(config_or_session)
        self.db = self.base.db

    def store_document_structure(
        self,
        document_id: str,
        structure_type: str,
        data: Dict[str, Any]
    ) -> DocumentStructure:
        """
        Store extracted structure for a document.

        Args:
            document_id: ID of the document
            structure_type: Type of structure ('toc', 'lof', 'lot', 'headers')
            data: Structured data (format depends on type)

        Returns:
            Created DocumentStructure instance

        Example data formats:
            TOC: {"entries": [{"section": "1.1", "title": "Intro", "page": 5}, ...]}
            LOF: {"entries": [{"figure": "Figure 1", "title": "Diagram", "page": 8}, ...]}
            Headers: {"hierarchy": [{"level": 1, "text": "Chapter 1", "page": 1}, ...]}
        """
        try:
            # Validate structure type
            valid_types = ['toc', 'lof', 'lot', 'headers']
            if structure_type not in valid_types:
                raise ValueError(f"Invalid structure_type: {structure_type}. Must be one of {valid_types}")

            # Validate document_id
            document_id = self.base._validate_input_security(document_id, "document_id")

            import uuid
            structure_id = str(uuid.uuid4())

            structure = DocumentStructure(
                id=structure_id,
                document_id=document_id,
                structure_type=structure_type,
                data=data
            )

            self.db.add(structure)
            self.base._safe_commit()

            logger.info(f"Stored {structure_type} structure for document {document_id}")
            return structure

        except Exception as e:
            logger.error(f"Error storing document structure: {e}")
            self.base._handle_session_error(e)
            raise

    def get_document_structure(
        self,
        document_id: str,
        structure_type: Optional[str] = None
    ) -> List[DocumentStructure]:
        """
        Get structure(s) for a document.

        Args:
            document_id: ID of the document
            structure_type: Optional filter by type ('toc', 'lof', 'lot', 'headers')

        Returns:
            List of DocumentStructure instances
        """
        try:
            query = self.db.query(DocumentStructure).filter(
                DocumentStructure.document_id == document_id
            )

            if structure_type:
                query = query.filter(DocumentStructure.structure_type == structure_type)

            return query.all()

        except Exception as e:
            logger.error(f"Error retrieving document structure: {e}")
            return []

    def get_structure_by_id(self, structure_id: str) -> Optional[DocumentStructure]:
        """Get a specific structure by ID."""
        try:
            return self.db.query(DocumentStructure).filter(
                DocumentStructure.id == structure_id
            ).first()
        except Exception as e:
            logger.error(f"Error retrieving structure by ID: {e}")
            return None

    def delete_document_structures(self, document_id: str) -> bool:
        """
        Delete all structures for a document.

        Args:
            document_id: ID of the document

        Returns:
            True if successful, False otherwise
        """
        try:
            deleted_count = self.db.query(DocumentStructure).filter(
                DocumentStructure.document_id == document_id
            ).delete()

            self.base._safe_commit()
            logger.info(f"Deleted {deleted_count} structure(s) for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document structures: {e}")
            self.base._handle_session_error(e)
            return False

    def get_toc_entries(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get parsed TOC entries for a document.

        Returns:
            List of TOC entries [{"section": "1.1", "title": "...", "page": 5}, ...]
        """
        structures = self.get_document_structure(document_id, structure_type='toc')
        if not structures:
            return []

        # TOC structure has format: {"entries": [...]}
        return structures[0].data.get('entries', [])

    def get_figure_list(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get list of figures (LOF) for a document.

        Returns:
            List of figure entries [{"figure": "Figure 1", "title": "...", "page": 8}, ...]
        """
        structures = self.get_document_structure(document_id, structure_type='lof')
        if not structures:
            return []

        return structures[0].data.get('entries', [])

    def get_table_list(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get list of tables (LOT) for a document.

        Returns:
            List of table entries [{"table": "Table 1", "title": "...", "page": 10}, ...]
        """
        structures = self.get_document_structure(document_id, structure_type='lot')
        if not structures:
            return []

        return structures[0].data.get('entries', [])

    def get_header_hierarchy(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get header hierarchy for a document.

        Returns:
            List of headers [{"level": 1, "text": "Chapter 1", "page": 1}, ...]
        """
        structures = self.get_document_structure(document_id, structure_type='headers')
        if not structures:
            return []

        return structures[0].data.get('hierarchy', [])

    def search_toc_by_keyword(
        self,
        document_id: str,
        keyword: str
    ) -> List[Dict[str, Any]]:
        """
        Search TOC entries by keyword.

        Args:
            document_id: ID of the document
            keyword: Keyword to search for in section titles

        Returns:
            List of matching TOC entries
        """
        toc_entries = self.get_toc_entries(document_id)
        keyword_lower = keyword.lower()

        return [
            entry for entry in toc_entries
            if keyword_lower in entry.get('title', '').lower()
        ]

    def get_structure_statistics(self, document_id: str) -> Dict[str, Any]:
        """
        Get statistics about document structure.

        Returns:
            Dict with counts of different structure types
        """
        structures = self.get_document_structure(document_id)

        stats = {
            'toc_count': 0,
            'lof_count': 0,
            'lot_count': 0,
            'headers_count': 0,
            'total_structures': len(structures)
        }

        for struct in structures:
            if struct.structure_type == 'toc':
                stats['toc_count'] = len(struct.data.get('entries', []))
            elif struct.structure_type == 'lof':
                stats['lof_count'] = len(struct.data.get('entries', []))
            elif struct.structure_type == 'lot':
                stats['lot_count'] = len(struct.data.get('entries', []))
            elif struct.structure_type == 'headers':
                stats['headers_count'] = len(struct.data.get('hierarchy', []))

        return stats
