"""
API service layer.

Business logic services extracted from route handlers to improve
maintainability and follow SOLID principles.
"""

from .collections_service import CollectionsService
from .task_service import TaskService

# Import utility functions from the standalone services.py file
try:
    from ..services import get_collection_by_identifier
    __all__ = ["CollectionsService", "TaskService", "get_collection_by_identifier"]
except ImportError:
    # If standalone services.py is not available, define a minimal version
    from typing import Optional
    from ...storage.base import DocumentStorageInterface
    from ...storage.models import Collection

    async def get_collection_by_identifier(
        storage: DocumentStorageInterface, identifier: str
    ) -> Optional[Collection]:
        """Get collection by ID or name - fallback implementation."""
        try:
            import uuid
            uuid.UUID(identifier)
            return storage.get_collection(identifier)
        except ValueError:
            return storage.get_collection_by_name(identifier)

    __all__ = ["CollectionsService", "TaskService", "get_collection_by_identifier"]
