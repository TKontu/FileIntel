"""
Comprehensive tests for collection lifecycle operations including
creation, modification, deletion, permissions, and relationships.
"""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from src.fileintel.storage.postgresql_storage import PostgreSQLStorage
from src.fileintel.storage.models import Collection, Document, Job, GraphRAGIndex


class TestCollectionCreation:
    """Test collection creation and validation."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_create_collection_with_valid_data(self, storage, mock_session):
        """Test successful collection creation with all fields."""
        collection_data = {
            "name": "Research Papers",
            "description": "Collection of academic research papers",
            "created_by": "user-123",
            "metadata": {"category": "academic", "visibility": "private"},
            "tags": ["research", "academic", "papers"],
        }

        mock_collection = Mock(spec=Collection)
        mock_collection.id = str(uuid.uuid4())
        mock_collection.name = collection_data["name"]
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None

        with patch(
            "src.fileintel.storage.postgresql_storage.Collection"
        ) as mock_collection_class:
            mock_collection_class.return_value = mock_collection
            result = storage.create_collection(**collection_data)

            assert result == mock_collection
            mock_session.add.assert_called_once_with(mock_collection)
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once_with(mock_collection)

    def test_create_collection_with_minimal_data(self, storage, mock_session):
        """Test collection creation with only required fields."""
        collection_data = {"name": "Minimal Collection", "created_by": "user-456"}

        mock_collection = Mock(spec=Collection)
        mock_collection.id = str(uuid.uuid4())
        mock_collection.description = None
        mock_collection.metadata = {}
        mock_collection.tags = []

        with patch(
            "src.fileintel.storage.postgresql_storage.Collection"
        ) as mock_collection_class:
            mock_collection_class.return_value = mock_collection
            result = storage.create_collection(**collection_data)

            assert result == mock_collection
            mock_session.add.assert_called_once()

    def test_create_collection_with_duplicate_name_same_user(
        self, storage, mock_session
    ):
        """Test collection creation fails with duplicate name for same user."""
        collection_data = {"name": "Existing Collection", "created_by": "user-123"}

        mock_session.add.side_effect = IntegrityError("Duplicate key", None, None)

        with patch("src.fileintel.storage.postgresql_storage.Collection"):
            with pytest.raises(IntegrityError):
                storage.create_collection(**collection_data)

    def test_create_collection_with_same_name_different_users(
        self, storage, mock_session
    ):
        """Test collection creation succeeds with same name for different users."""
        collection_data = {"name": "Common Name", "created_by": "user-789"}

        mock_collection = Mock(spec=Collection)
        mock_collection.id = str(uuid.uuid4())

        with patch(
            "src.fileintel.storage.postgresql_storage.Collection"
        ) as mock_collection_class:
            mock_collection_class.return_value = mock_collection
            result = storage.create_collection(**collection_data)

            assert result == mock_collection
            mock_session.add.assert_called_once()

    def test_create_collection_with_invalid_metadata(self, storage, mock_session):
        """Test collection creation with non-serializable metadata."""
        collection_data = {
            "name": "Invalid Metadata Collection",
            "created_by": "user-123",
            "metadata": {"invalid": object()},  # Non-JSON serializable
        }

        with patch("json.dumps") as mock_json:
            mock_json.side_effect = TypeError("Not JSON serializable")

            with patch("src.fileintel.storage.postgresql_storage.Collection"):
                with pytest.raises(TypeError):
                    storage.create_collection(**collection_data)

    def test_create_collection_validates_name_length(self, storage, mock_session):
        """Test collection creation validates name length limits."""
        collection_data = {"name": "x" * 256, "created_by": "user-123"}  # Too long name

        mock_session.add.side_effect = IntegrityError("Value too long", None, None)

        with patch("src.fileintel.storage.postgresql_storage.Collection"):
            with pytest.raises(IntegrityError):
                storage.create_collection(**collection_data)


class TestCollectionRetrieval:
    """Test collection retrieval and querying operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_get_collection_by_id_exists(self, storage, mock_session):
        """Test retrieving existing collection by ID."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_collection.id = collection_id
        mock_collection.name = "Test Collection"

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.get_collection_by_id(collection_id)

        assert result == mock_collection
        mock_session.query.assert_called_once_with(Collection)

    def test_get_collection_by_id_not_exists(self, storage, mock_session):
        """Test retrieving non-existent collection returns None."""
        collection_id = str(uuid.uuid4())
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = storage.get_collection_by_id(collection_id)

        assert result is None

    def test_get_collection_by_name_and_user(self, storage, mock_session):
        """Test retrieving collection by name and user ID."""
        collection_name = "My Collection"
        user_id = "user-123"
        mock_collection = Mock(spec=Collection)
        mock_collection.name = collection_name
        mock_collection.created_by = user_id

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.get_collection_by_name_and_user(collection_name, user_id)

        assert result == mock_collection
        mock_session.query.assert_called_once_with(Collection)

    def test_get_collections_by_user_with_pagination(self, storage, mock_session):
        """Test retrieving user collections with pagination."""
        user_id = "user-123"
        page = 2
        page_size = 10

        mock_collections = [Mock(spec=Collection) for _ in range(page_size)]
        mock_query = mock_session.query.return_value.filter.return_value
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = (
            mock_collections
        )
        mock_query.count.return_value = 25  # Total collections

        result = storage.get_collections_by_user_paginated(user_id, page, page_size)

        assert len(result["collections"]) == page_size
        assert result["total_count"] == 25
        assert result["page"] == page
        assert result["page_size"] == page_size
        assert result["total_pages"] == 3

    def test_get_collections_by_tags(self, storage, mock_session):
        """Test retrieving collections by tags."""
        tags = ["research", "academic"]
        mock_collections = [Mock(spec=Collection), Mock(spec=Collection)]

        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_collections
        )

        result = storage.get_collections_by_tags(tags)

        assert result == mock_collections
        mock_session.query.assert_called_once_with(Collection)

    def test_search_collections_by_text(self, storage, mock_session):
        """Test full-text search in collections."""
        search_query = "machine learning"
        user_id = "user-123"
        mock_collections = [Mock(spec=Collection), Mock(spec=Collection)]

        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_collections
        )

        result = storage.search_collections_by_text(search_query, user_id)

        assert result == mock_collections
        mock_session.query.assert_called_once_with(Collection)

    def test_get_collection_with_statistics(self, storage, mock_session):
        """Test retrieving collection with document/job statistics."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_collection.id = collection_id

        # Mock related counts
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )
        mock_session.query.return_value.filter.return_value.count.side_effect = [
            15,
            3,
            1,
        ]  # docs, jobs, indexes

        result = storage.get_collection_with_statistics(collection_id)

        assert result["collection"] == mock_collection
        assert result["document_count"] == 15
        assert result["job_count"] == 3
        assert result["graphrag_index_count"] == 1

    def test_get_recently_updated_collections(self, storage, mock_session):
        """Test retrieving recently updated collections."""
        user_id = "user-123"
        limit = 5
        mock_collections = [Mock(spec=Collection) for _ in range(limit)]

        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            mock_collections
        )

        result = storage.get_recently_updated_collections(user_id, limit)

        assert len(result) == limit
        mock_session.query.assert_called_once_with(Collection)


class TestCollectionModification:
    """Test collection update and modification operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_update_collection_name(self, storage, mock_session):
        """Test updating collection name."""
        collection_id = str(uuid.uuid4())
        new_name = "Updated Collection Name"

        mock_collection = Mock(spec=Collection)
        mock_collection.name = "Old Name"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.update_collection_name(collection_id, new_name)

        assert result is True
        assert mock_collection.name == new_name
        mock_session.commit.assert_called_once()

    def test_update_collection_description(self, storage, mock_session):
        """Test updating collection description."""
        collection_id = str(uuid.uuid4())
        new_description = "Updated description with more details"

        mock_collection = Mock(spec=Collection)
        mock_collection.description = "Old description"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.update_collection_description(collection_id, new_description)

        assert result is True
        assert mock_collection.description == new_description
        mock_session.commit.assert_called_once()

    def test_update_collection_metadata(self, storage, mock_session):
        """Test updating collection metadata."""
        collection_id = str(uuid.uuid4())
        new_metadata = {"category": "updated", "version": "2.0", "priority": "high"}

        mock_collection = Mock(spec=Collection)
        mock_collection.metadata = {"category": "original", "version": "1.0"}
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.update_collection_metadata(collection_id, new_metadata)

        assert result is True
        assert mock_collection.metadata == new_metadata
        mock_session.commit.assert_called_once()

    def test_update_collection_tags(self, storage, mock_session):
        """Test updating collection tags."""
        collection_id = str(uuid.uuid4())
        new_tags = ["updated", "tags", "list"]

        mock_collection = Mock(spec=Collection)
        mock_collection.tags = ["old", "tags"]
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.update_collection_tags(collection_id, new_tags)

        assert result is True
        assert mock_collection.tags == new_tags
        mock_session.commit.assert_called_once()

    def test_add_tags_to_collection(self, storage, mock_session):
        """Test adding tags to existing collection tags."""
        collection_id = str(uuid.uuid4())
        additional_tags = ["new", "additional"]

        mock_collection = Mock(spec=Collection)
        mock_collection.tags = ["existing", "tags"]
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.add_tags_to_collection(collection_id, additional_tags)

        assert result is True
        expected_tags = ["existing", "tags", "new", "additional"]
        assert set(mock_collection.tags) == set(expected_tags)
        mock_session.commit.assert_called_once()

    def test_remove_tags_from_collection(self, storage, mock_session):
        """Test removing tags from collection."""
        collection_id = str(uuid.uuid4())
        tags_to_remove = ["old", "unwanted"]

        mock_collection = Mock(spec=Collection)
        mock_collection.tags = ["old", "keep", "unwanted", "preserve"]
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.remove_tags_from_collection(collection_id, tags_to_remove)

        assert result is True
        expected_tags = ["keep", "preserve"]
        assert set(mock_collection.tags) == set(expected_tags)
        mock_session.commit.assert_called_once()

    def test_update_collection_not_found(self, storage, mock_session):
        """Test updating non-existent collection returns False."""
        collection_id = str(uuid.uuid4())
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = storage.update_collection_name(collection_id, "New Name")

        assert result is False
        mock_session.commit.assert_not_called()

    def test_update_collection_with_duplicate_name(self, storage, mock_session):
        """Test updating collection name to duplicate raises error."""
        collection_id = str(uuid.uuid4())
        duplicate_name = "Existing Name"

        mock_collection = Mock(spec=Collection)
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )
        mock_session.commit.side_effect = IntegrityError("Duplicate key", None, None)

        with pytest.raises(IntegrityError):
            storage.update_collection_name(collection_id, duplicate_name)

        mock_session.rollback.assert_called_once()


class TestCollectionDeletion:
    """Test collection deletion and cleanup operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_delete_empty_collection(self, storage, mock_session):
        """Test deleting collection with no documents or jobs."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_collection.documents = []
        mock_collection.jobs = []

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.delete_collection(collection_id)

        assert result is True
        mock_session.delete.assert_called_once_with(mock_collection)
        mock_session.commit.assert_called_once()

    def test_delete_collection_with_documents_cascade(self, storage, mock_session):
        """Test deleting collection with cascade to documents."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_documents = [Mock(spec=Document), Mock(spec=Document)]
        mock_collection.documents = mock_documents

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.delete_collection_cascade(collection_id)

        assert result is True
        mock_session.delete.assert_called_once_with(mock_collection)
        mock_session.commit.assert_called_once()

    def test_delete_collection_with_active_jobs_fails(self, storage, mock_session):
        """Test deleting collection with active jobs fails."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_jobs = [Mock(spec=Job, status="running"), Mock(spec=Job, status="pending")]
        mock_collection.jobs = mock_jobs

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.delete_collection(collection_id, allow_with_active_jobs=False)

        assert result is False
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()

    def test_delete_collection_with_graphrag_index(self, storage, mock_session):
        """Test deleting collection cleans up GraphRAG index."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_graphrag_index = Mock(spec=GraphRAGIndex)
        mock_graphrag_index.index_path = "/data/graphrag/collection-123"

        mock_session.query.return_value.filter.return_value.first.side_effect = [
            mock_collection,
            mock_graphrag_index,
        ]

        with patch("os.path.exists") as mock_exists:
            with patch("shutil.rmtree") as mock_rmtree:
                mock_exists.return_value = True

                result = storage.delete_collection_with_cleanup(collection_id)

                assert result is True
                mock_rmtree.assert_called_once_with("/data/graphrag/collection-123")
                mock_session.delete.assert_called_with(mock_collection)

    def test_delete_collection_not_found(self, storage, mock_session):
        """Test deleting non-existent collection returns False."""
        collection_id = str(uuid.uuid4())
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = storage.delete_collection(collection_id)

        assert result is False
        mock_session.delete.assert_not_called()

    def test_soft_delete_collection(self, storage, mock_session):
        """Test soft deletion of collection (marking as deleted)."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_collection.deleted_at = None
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.soft_delete_collection(collection_id)

        assert result is True
        assert mock_collection.deleted_at is not None
        mock_session.commit.assert_called_once()

    def test_restore_soft_deleted_collection(self, storage, mock_session):
        """Test restoring soft-deleted collection."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_collection.deleted_at = datetime.utcnow()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.restore_collection(collection_id)

        assert result is True
        assert mock_collection.deleted_at is None
        mock_session.commit.assert_called_once()


class TestCollectionPermissions:
    """Test collection permission and sharing operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_check_collection_access_owner(self, storage, mock_session):
        """Test collection access check for owner."""
        collection_id = str(uuid.uuid4())
        user_id = "user-123"

        mock_collection = Mock(spec=Collection)
        mock_collection.created_by = user_id
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.check_collection_access(collection_id, user_id)

        assert result is True

    def test_check_collection_access_shared_user(self, storage, mock_session):
        """Test collection access check for shared user."""
        collection_id = str(uuid.uuid4())
        user_id = "user-456"

        mock_collection = Mock(spec=Collection)
        mock_collection.created_by = "user-123"  # Different owner
        mock_collection.shared_with = [user_id]
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.check_collection_access(collection_id, user_id)

        assert result is True

    def test_check_collection_access_denied(self, storage, mock_session):
        """Test collection access denied for unauthorized user."""
        collection_id = str(uuid.uuid4())
        user_id = "user-789"

        mock_collection = Mock(spec=Collection)
        mock_collection.created_by = "user-123"
        mock_collection.shared_with = ["user-456"]  # Different users
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.check_collection_access(collection_id, user_id)

        assert result is False

    def test_share_collection_with_user(self, storage, mock_session):
        """Test sharing collection with another user."""
        collection_id = str(uuid.uuid4())
        owner_id = "user-123"
        share_with_id = "user-456"

        mock_collection = Mock(spec=Collection)
        mock_collection.created_by = owner_id
        mock_collection.shared_with = []
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.share_collection(collection_id, owner_id, share_with_id)

        assert result is True
        assert share_with_id in mock_collection.shared_with
        mock_session.commit.assert_called_once()

    def test_unshare_collection_from_user(self, storage, mock_session):
        """Test removing user access from shared collection."""
        collection_id = str(uuid.uuid4())
        owner_id = "user-123"
        unshare_from_id = "user-456"

        mock_collection = Mock(spec=Collection)
        mock_collection.created_by = owner_id
        mock_collection.shared_with = [unshare_from_id, "user-789"]
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.unshare_collection(collection_id, owner_id, unshare_from_id)

        assert result is True
        assert unshare_from_id not in mock_collection.shared_with
        assert "user-789" in mock_collection.shared_with
        mock_session.commit.assert_called_once()

    def test_get_shared_collections(self, storage, mock_session):
        """Test retrieving collections shared with user."""
        user_id = "user-456"
        mock_collections = [Mock(spec=Collection), Mock(spec=Collection)]

        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_collections
        )

        result = storage.get_shared_collections(user_id)

        assert result == mock_collections
        mock_session.query.assert_called_once_with(Collection)

    def test_get_collection_sharing_info(self, storage, mock_session):
        """Test getting collection sharing information."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_collection.created_by = "user-123"
        mock_collection.shared_with = ["user-456", "user-789"]
        mock_collection.sharing_permissions = {"user-456": "read", "user-789": "write"}

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.get_collection_sharing_info(collection_id)

        assert result["owner"] == "user-123"
        assert len(result["shared_with"]) == 2
        assert result["permissions"] == {"user-456": "read", "user-789": "write"}


class TestCollectionRelationships:
    """Test collection relationships with documents, jobs, and indexes."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_add_document_to_collection(self, storage, mock_session):
        """Test adding document to collection."""
        collection_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())

        mock_collection = Mock(spec=Collection)
        mock_document = Mock(spec=Document)
        mock_document.collection_id = None

        mock_session.query.return_value.filter.return_value.first.side_effect = [
            mock_collection,
            mock_document,
        ]

        result = storage.add_document_to_collection(collection_id, document_id)

        assert result is True
        assert mock_document.collection_id == collection_id
        mock_session.commit.assert_called_once()

    def test_remove_document_from_collection(self, storage, mock_session):
        """Test removing document from collection."""
        collection_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())

        mock_document = Mock(spec=Document)
        mock_document.collection_id = collection_id
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_document
        )

        result = storage.remove_document_from_collection(document_id)

        assert result is True
        assert mock_document.collection_id is None
        mock_session.commit.assert_called_once()

    def test_get_collection_document_count(self, storage, mock_session):
        """Test getting document count for collection."""
        collection_id = str(uuid.uuid4())
        expected_count = 42

        mock_session.query.return_value.filter.return_value.count.return_value = (
            expected_count
        )

        result = storage.get_collection_document_count(collection_id)

        assert result == expected_count
        mock_session.query.assert_called_once_with(Document)

    def test_get_collection_jobs(self, storage, mock_session):
        """Test getting jobs associated with collection."""
        collection_id = str(uuid.uuid4())
        mock_jobs = [Mock(spec=Job), Mock(spec=Job), Mock(spec=Job)]

        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            mock_jobs
        )

        result = storage.get_collection_jobs(collection_id)

        assert result == mock_jobs
        mock_session.query.assert_called_once_with(Job)

    def test_get_collection_active_jobs(self, storage, mock_session):
        """Test getting active jobs for collection."""
        collection_id = str(uuid.uuid4())
        active_statuses = ["pending", "running", "retry_pending"]
        mock_active_jobs = [Mock(spec=Job), Mock(spec=Job)]

        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_active_jobs
        )

        result = storage.get_collection_active_jobs(collection_id)

        assert result == mock_active_jobs
        mock_session.query.assert_called_once_with(Job)

    def test_cancel_collection_jobs(self, storage, mock_session):
        """Test cancelling all jobs for collection."""
        collection_id = str(uuid.uuid4())
        mock_jobs = [
            Mock(spec=Job, status="pending"),
            Mock(spec=Job, status="running"),
            Mock(spec=Job, status="retry_pending"),
        ]

        mock_session.query.return_value.filter.return_value.all.return_value = mock_jobs

        result = storage.cancel_collection_jobs(collection_id)

        assert result == len(mock_jobs)
        for job in mock_jobs:
            assert job.status == "cancelled"
        mock_session.commit.assert_called_once()


class TestCollectionErrorHandling:
    """Test error handling in collection operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_database_error_during_creation(self, storage, mock_session):
        """Test handling database errors during collection creation."""
        collection_data = {"name": "Test Collection", "created_by": "user-123"}

        mock_session.add.side_effect = SQLAlchemyError("Database connection lost")

        with patch("src.fileintel.storage.postgresql_storage.Collection"):
            with pytest.raises(SQLAlchemyError):
                storage.create_collection(**collection_data)

    def test_transaction_rollback_on_error(self, storage, mock_session):
        """Test transaction rollback on error."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )
        mock_session.commit.side_effect = SQLAlchemyError("Commit failed")

        with pytest.raises(SQLAlchemyError):
            storage.update_collection_name(collection_id, "New Name")

        mock_session.rollback.assert_called_once()

    def test_concurrent_modification_handling(self, storage, mock_session):
        """Test handling concurrent collection modifications."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_collection.version = 1

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )
        mock_session.commit.side_effect = SQLAlchemyError(
            "Row was updated by another transaction"
        )

        with pytest.raises(SQLAlchemyError):
            storage.update_collection_with_version_check(
                collection_id, "New Name", expected_version=1
            )

        mock_session.rollback.assert_called_once()

    def test_cleanup_on_partial_failure(self, storage, mock_session):
        """Test cleanup of resources on partial operation failure."""
        collection_data = {
            "name": "Partial Failure Collection",
            "created_by": "user-123",
        }

        mock_collection = Mock(spec=Collection)
        mock_session.add.return_value = None
        mock_session.commit.side_effect = SQLAlchemyError("Partial failure")
        mock_session.rollback.return_value = None

        with patch(
            "src.fileintel.storage.postgresql_storage.Collection"
        ) as mock_collection_class:
            mock_collection_class.return_value = mock_collection

            with pytest.raises(SQLAlchemyError):
                storage.create_collection(**collection_data)

            mock_session.rollback.assert_called_once()
