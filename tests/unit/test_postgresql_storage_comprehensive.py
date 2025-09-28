"""
Comprehensive tests for PostgreSQL storage operations including CRUD,
relationships, indexes, and edge cases.
"""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from src.fileintel.storage.postgresql_storage import PostgreSQLStorage
from src.fileintel.storage.models import (
    Job,
    Result,
    Document,
    DocumentChunk,
    Collection,
    GraphRAGIndex,
    GraphRAGEntity,
    GraphRAGCommunity,
    GraphRAGRelationship,
    DeadLetterJob,
    CircuitBreakerState,
)


class TestPostgreSQLStorageBasicOperations:
    """Test basic CRUD operations for all models."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_create_job_success(self, storage, mock_session):
        """Test successful job creation with all fields."""
        job_data = {
            "job_type": "indexing",
            "job_data": {"document_id": "doc-123"},
            "collection_id": str(uuid.uuid4()),
            "priority": 5,
            "retry_count": 0,
            "max_retries": 3,
            "retry_delay": 60,
        }

        mock_job = Mock(spec=Job)
        mock_job.id = "job-123"
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None

        with patch("src.fileintel.storage.postgresql_storage.Job") as mock_job_class:
            mock_job_class.return_value = mock_job
            result = storage.create_job(**job_data)

            assert result == mock_job
            mock_session.add.assert_called_once_with(mock_job)
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once_with(mock_job)

    def test_create_job_with_invalid_priority(self, storage, mock_session):
        """Test job creation with invalid priority values."""
        job_data = {
            "job_type": "indexing",
            "job_data": {"document_id": "doc-123"},
            "priority": -1,  # Invalid priority
        }

        mock_session.add.side_effect = IntegrityError("Invalid priority", None, None)

        with patch("src.fileintel.storage.postgresql_storage.Job"):
            with pytest.raises(IntegrityError):
                storage.create_job(**job_data)

    def test_get_job_by_id_exists(self, storage, mock_session):
        """Test retrieving existing job by ID."""
        job_id = "job-123"
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.get_job_by_id(job_id)

        assert result == mock_job
        mock_session.query.assert_called_once_with(Job)

    def test_get_job_by_id_not_exists(self, storage, mock_session):
        """Test retrieving non-existent job returns None."""
        job_id = "nonexistent-job"
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = storage.get_job_by_id(job_id)

        assert result is None

    def test_update_job_status_success(self, storage, mock_session):
        """Test successful job status update."""
        job_id = "job-123"
        new_status = "completed"

        mock_job = Mock(spec=Job)
        mock_job.status = "running"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.update_job_status(job_id, new_status)

        assert result is True
        assert mock_job.status == new_status
        mock_session.commit.assert_called_once()

    def test_update_job_status_not_found(self, storage, mock_session):
        """Test job status update for non-existent job."""
        job_id = "nonexistent-job"
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = storage.update_job_status(job_id, "completed")

        assert result is False
        mock_session.commit.assert_not_called()

    def test_delete_job_success(self, storage, mock_session):
        """Test successful job deletion."""
        job_id = "job-123"
        mock_job = Mock(spec=Job)
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.delete_job(job_id)

        assert result is True
        mock_session.delete.assert_called_once_with(mock_job)
        mock_session.commit.assert_called_once()

    def test_delete_job_not_found(self, storage, mock_session):
        """Test deletion of non-existent job."""
        job_id = "nonexistent-job"
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = storage.delete_job(job_id)

        assert result is False
        mock_session.delete.assert_not_called()


class TestPostgreSQLStorageCollections:
    """Test collection management operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_create_collection_success(self, storage, mock_session):
        """Test successful collection creation."""
        collection_data = {
            "name": "Test Collection",
            "description": "Test description",
            "created_by": "user-123",
        }

        mock_collection = Mock(spec=Collection)
        mock_collection.id = str(uuid.uuid4())
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

    def test_create_collection_duplicate_name(self, storage, mock_session):
        """Test collection creation with duplicate name."""
        collection_data = {
            "name": "Existing Collection",
            "description": "Test description",
            "created_by": "user-123",
        }

        mock_session.add.side_effect = IntegrityError("Duplicate name", None, None)

        with patch("src.fileintel.storage.postgresql_storage.Collection"):
            with pytest.raises(IntegrityError):
                storage.create_collection(**collection_data)

    def test_get_collections_by_user(self, storage, mock_session):
        """Test retrieving collections for a specific user."""
        user_id = "user-123"
        mock_collections = [Mock(spec=Collection), Mock(spec=Collection)]
        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_collections
        )

        result = storage.get_collections_by_user(user_id)

        assert result == mock_collections
        mock_session.query.assert_called_once_with(Collection)

    def test_get_collection_with_documents(self, storage, mock_session):
        """Test retrieving collection with associated documents."""
        collection_id = str(uuid.uuid4())
        mock_collection = Mock(spec=Collection)
        mock_documents = [Mock(spec=Document), Mock(spec=Document)]
        mock_collection.documents = mock_documents

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.get_collection_with_documents(collection_id)

        assert result == mock_collection
        assert len(result.documents) == 2

    def test_update_collection_metadata(self, storage, mock_session):
        """Test updating collection metadata."""
        collection_id = str(uuid.uuid4())
        new_metadata = {"updated": True, "version": "2.0"}

        mock_collection = Mock(spec=Collection)
        mock_collection.metadata = {"version": "1.0"}
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_collection
        )

        result = storage.update_collection_metadata(collection_id, new_metadata)

        assert result is True
        assert mock_collection.metadata == new_metadata
        mock_session.commit.assert_called_once()

    def test_delete_collection_with_documents(self, storage, mock_session):
        """Test collection deletion with cascade to documents."""
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


class TestPostgreSQLStorageDocuments:
    """Test document storage and retrieval operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_create_document_with_chunks(self, storage, mock_session):
        """Test document creation with associated chunks."""
        document_data = {
            "filename": "test.pdf",
            "content_hash": "abc123",
            "collection_id": str(uuid.uuid4()),
            "file_size": 1024,
            "mime_type": "application/pdf",
            "metadata": {"pages": 10},
        }

        chunks_data = [
            {"content": "Chunk 1", "chunk_index": 0, "metadata": {}},
            {"content": "Chunk 2", "chunk_index": 1, "metadata": {}},
        ]

        mock_document = Mock(spec=Document)
        mock_document.id = str(uuid.uuid4())
        mock_chunks = [Mock(spec=DocumentChunk), Mock(spec=DocumentChunk)]

        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None

        with patch(
            "src.fileintel.storage.postgresql_storage.Document"
        ) as mock_doc_class:
            with patch(
                "src.fileintel.storage.postgresql_storage.DocumentChunk"
            ) as mock_chunk_class:
                mock_doc_class.return_value = mock_document
                mock_chunk_class.side_effect = mock_chunks

                result = storage.create_document_with_chunks(document_data, chunks_data)

                assert result == mock_document
                assert mock_session.add.call_count == 3  # 1 document + 2 chunks
                mock_session.commit.assert_called_once()

    def test_get_document_by_hash(self, storage, mock_session):
        """Test retrieving document by content hash."""
        content_hash = "abc123"
        mock_document = Mock(spec=Document)
        mock_document.content_hash = content_hash

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_document
        )

        result = storage.get_document_by_hash(content_hash)

        assert result == mock_document
        mock_session.query.assert_called_once_with(Document)

    def test_get_documents_by_collection_paginated(self, storage, mock_session):
        """Test paginated document retrieval for collection."""
        collection_id = str(uuid.uuid4())
        page = 1
        page_size = 10

        mock_documents = [Mock(spec=Document) for _ in range(page_size)]
        mock_query = mock_session.query.return_value.filter.return_value
        mock_query.offset.return_value.limit.return_value.all.return_value = (
            mock_documents
        )
        mock_query.count.return_value = 25  # Total count

        result = storage.get_documents_by_collection_paginated(
            collection_id, page, page_size
        )

        assert len(result["documents"]) == page_size
        assert result["total_count"] == 25
        assert result["page"] == page
        assert result["page_size"] == page_size
        assert result["total_pages"] == 3

    def test_update_document_processing_status(self, storage, mock_session):
        """Test updating document processing status."""
        document_id = str(uuid.uuid4())
        new_status = "processed"
        processing_result = {"chunks_created": 5, "embeddings_generated": True}

        mock_document = Mock(spec=Document)
        mock_document.processing_status = "processing"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_document
        )

        result = storage.update_document_processing_status(
            document_id, new_status, processing_result
        )

        assert result is True
        assert mock_document.processing_status == new_status
        assert mock_document.processing_result == processing_result
        mock_session.commit.assert_called_once()

    def test_find_duplicate_documents(self, storage, mock_session):
        """Test finding documents with duplicate content hashes."""
        mock_duplicates = [
            Mock(spec=Document, content_hash="hash1", id="doc1"),
            Mock(spec=Document, content_hash="hash1", id="doc2"),
            Mock(spec=Document, content_hash="hash2", id="doc3"),
            Mock(spec=Document, content_hash="hash2", id="doc4"),
        ]

        mock_session.query.return_value.group_by.return_value.having.return_value.all.return_value = (
            mock_duplicates
        )

        result = storage.find_duplicate_documents()

        assert len(result) == 4
        mock_session.query.assert_called_once_with(Document)


class TestPostgreSQLStorageJobQueue:
    """Test job queue operations and edge cases."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_get_next_jobs_with_priority(self, storage, mock_session):
        """Test job retrieval with priority ordering."""
        batch_size = 5
        mock_jobs = [Mock(spec=Job) for _ in range(batch_size)]

        mock_query = mock_session.query.return_value
        mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            mock_jobs
        )

        result = storage.get_next_jobs(batch_size)

        assert len(result) == batch_size
        mock_session.query.assert_called_once_with(Job)

    def test_get_next_jobs_with_retry_logic(self, storage, mock_session):
        """Test job retrieval including retry pending jobs."""
        batch_size = 3
        current_time = datetime.utcnow()

        # Mock jobs including retry pending jobs ready for retry
        mock_jobs = [
            Mock(spec=Job, status="pending"),
            Mock(
                spec=Job,
                status="retry_pending",
                next_retry_at=current_time - timedelta(minutes=1),
            ),
            Mock(
                spec=Job,
                status="retry_pending",
                next_retry_at=current_time + timedelta(minutes=1),
            ),  # Not ready
        ]

        # Should return first 2 jobs (pending + ready retry)
        mock_query = mock_session.query.return_value
        mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_jobs[
            :2
        ]

        result = storage.get_next_jobs(batch_size)

        assert len(result) == 2

    def test_get_jobs_by_status_and_type(self, storage, mock_session):
        """Test filtering jobs by status and type."""
        status = "failed"
        job_type = "indexing"

        mock_jobs = [Mock(spec=Job, status=status, job_type=job_type)]
        mock_session.query.return_value.filter.return_value.all.return_value = mock_jobs

        result = storage.get_jobs_by_status_and_type(status, job_type)

        assert len(result) == 1
        assert result[0].status == status
        assert result[0].job_type == job_type

    def test_update_job_with_result(self, storage, mock_session):
        """Test updating job with processing result."""
        job_id = "job-123"
        status = "completed"
        result_data = {"documents_processed": 5, "chunks_created": 25}

        mock_job = Mock(spec=Job)
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.update_job_with_result(job_id, status, result_data)

        assert result is True
        assert mock_job.status == status
        assert mock_job.result == result_data
        mock_session.commit.assert_called_once()

    def test_get_job_statistics(self, storage, mock_session):
        """Test retrieving job queue statistics."""
        # Mock count queries for different statuses
        mock_session.query.return_value.filter.return_value.count.side_effect = [
            10,
            5,
            2,
            1,
        ]

        result = storage.get_job_statistics()

        expected_stats = {
            "pending": 10,
            "running": 5,
            "completed": 2,
            "failed": 1,
            "total": 18,
        }

        assert result == expected_stats
        assert mock_session.query.call_count == 4


class TestPostgreSQLStorageGraphRAG:
    """Test GraphRAG-related storage operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_create_graphrag_index(self, storage, mock_session):
        """Test GraphRAG index creation."""
        index_data = {
            "collection_id": str(uuid.uuid4()),
            "status": "indexing",
            "index_path": "/data/graphrag/collection-123",
            "config": {"model": "gpt-4", "chunk_size": 512},
        }

        mock_index = Mock(spec=GraphRAGIndex)
        mock_index.id = 123
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None

        with patch(
            "src.fileintel.storage.postgresql_storage.GraphRAGIndex"
        ) as mock_index_class:
            mock_index_class.return_value = mock_index
            result = storage.create_graphrag_index(**index_data)

            assert result == mock_index
            mock_session.add.assert_called_once_with(mock_index)
            mock_session.commit.assert_called_once()

    def test_get_graphrag_index_by_collection(self, storage, mock_session):
        """Test retrieving GraphRAG index by collection ID."""
        collection_id = str(uuid.uuid4())
        mock_index = Mock(spec=GraphRAGIndex)
        mock_index.collection_id = collection_id

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_index
        )

        result = storage.get_graphrag_index_by_collection(collection_id)

        assert result == mock_index
        mock_session.query.assert_called_once_with(GraphRAGIndex)

    def test_update_graphrag_index_status(self, storage, mock_session):
        """Test updating GraphRAG index status."""
        index_id = 123
        new_status = "ready"

        mock_index = Mock(spec=GraphRAGIndex)
        mock_index.status = "indexing"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_index
        )

        result = storage.update_graphrag_index_status(index_id, new_status)

        assert result is True
        assert mock_index.status == new_status
        mock_session.commit.assert_called_once()

    def test_create_graphrag_entities_batch(self, storage, mock_session):
        """Test batch creation of GraphRAG entities."""
        collection_id = str(uuid.uuid4())
        entities_data = [
            {
                "entity_name": "Person A",
                "entity_type": "PERSON",
                "description": "A person",
            },
            {
                "entity_name": "Company B",
                "entity_type": "ORGANIZATION",
                "description": "A company",
            },
        ]

        mock_entities = [Mock(spec=GraphRAGEntity), Mock(spec=GraphRAGEntity)]
        mock_session.add_all.return_value = None
        mock_session.commit.return_value = None

        with patch(
            "src.fileintel.storage.postgresql_storage.GraphRAGEntity"
        ) as mock_entity_class:
            mock_entity_class.side_effect = mock_entities
            result = storage.create_graphrag_entities_batch(
                collection_id, entities_data
            )

            assert result == len(entities_data)
            mock_session.add_all.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_get_graphrag_entities_by_collection(self, storage, mock_session):
        """Test retrieving GraphRAG entities for a collection."""
        collection_id = str(uuid.uuid4())
        mock_entities = [Mock(spec=GraphRAGEntity), Mock(spec=GraphRAGEntity)]

        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_entities
        )

        result = storage.get_graphrag_entities_by_collection(collection_id)

        assert result == mock_entities
        mock_session.query.assert_called_once_with(GraphRAGEntity)

    def test_get_graphrag_communities_by_level(self, storage, mock_session):
        """Test retrieving GraphRAG communities by level."""
        collection_id = str(uuid.uuid4())
        level = 2
        mock_communities = [Mock(spec=GraphRAGCommunity), Mock(spec=GraphRAGCommunity)]

        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_communities
        )

        result = storage.get_graphrag_communities_by_level(collection_id, level)

        assert result == mock_communities
        mock_session.query.assert_called_once_with(GraphRAGCommunity)


class TestPostgreSQLStorageErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_database_connection_error(self, storage, mock_session):
        """Test handling database connection errors."""
        mock_session.query.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(SQLAlchemyError):
            storage.get_job_by_id("job-123")

    def test_transaction_rollback_on_error(self, storage, mock_session):
        """Test transaction rollback on error."""
        mock_session.commit.side_effect = SQLAlchemyError("Commit failed")
        mock_session.rollback.return_value = None

        job_data = {"job_type": "indexing", "job_data": {"document_id": "doc-123"}}

        with patch("src.fileintel.storage.postgresql_storage.Job"):
            with pytest.raises(SQLAlchemyError):
                storage.create_job(**job_data)

            mock_session.rollback.assert_called_once()

    def test_concurrent_job_update_conflict(self, storage, mock_session):
        """Test handling concurrent job updates."""
        job_id = "job-123"

        # Simulate optimistic locking failure
        mock_session.commit.side_effect = [
            None,
            SQLAlchemyError("Row was updated by another transaction"),
        ]
        mock_job = Mock(spec=Job)
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        # First update succeeds
        result1 = storage.update_job_status(job_id, "running")
        assert result1 is True

        # Second update fails due to concurrent modification
        with pytest.raises(SQLAlchemyError):
            storage.update_job_status(job_id, "completed")

    def test_invalid_json_data_handling(self, storage, mock_session):
        """Test handling of invalid JSON data."""
        with patch("json.dumps") as mock_json:
            mock_json.side_effect = TypeError("Not JSON serializable")

            job_data = {
                "job_type": "indexing",
                "job_data": {"invalid": object()},  # Non-serializable object
            }

            with patch("src.fileintel.storage.postgresql_storage.Job"):
                with pytest.raises(TypeError):
                    storage.create_job(**job_data)

    def test_cleanup_orphaned_resources(self, storage, mock_session):
        """Test cleanup of orphaned resources."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)

        # Mock orphaned jobs
        mock_orphaned_jobs = [Mock(spec=Job), Mock(spec=Job)]
        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_orphaned_jobs
        )

        result = storage.cleanup_orphaned_jobs(cutoff_time)

        assert result == len(mock_orphaned_jobs)
        assert mock_session.delete.call_count == len(mock_orphaned_jobs)
        mock_session.commit.assert_called_once()


class TestPostgreSQLStoragePerformance:
    """Test performance-related storage operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_bulk_document_insertion(self, storage, mock_session):
        """Test bulk insertion of documents for performance."""
        documents_data = [
            {
                "filename": f"doc{i}.pdf",
                "content_hash": f"hash{i}",
                "collection_id": str(uuid.uuid4()),
            }
            for i in range(100)
        ]

        mock_documents = [Mock(spec=Document) for _ in range(100)]
        mock_session.bulk_insert_mappings.return_value = None
        mock_session.commit.return_value = None

        with patch(
            "src.fileintel.storage.postgresql_storage.Document"
        ) as mock_doc_class:
            mock_doc_class.side_effect = mock_documents
            result = storage.bulk_insert_documents(documents_data)

            assert result == len(documents_data)
            mock_session.bulk_insert_mappings.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_batch_job_status_update(self, storage, mock_session):
        """Test batch updating of job statuses."""
        job_ids = ["job-1", "job-2", "job-3"]
        new_status = "completed"

        mock_session.query.return_value.filter.return_value.update.return_value = len(
            job_ids
        )
        mock_session.commit.return_value = None

        result = storage.batch_update_job_status(job_ids, new_status)

        assert result == len(job_ids)
        mock_session.commit.assert_called_once()

    def test_query_optimization_with_indexes(self, storage, mock_session):
        """Test that queries use proper indexes."""
        # Test that complex queries are optimized
        collection_id = str(uuid.uuid4())

        mock_session.query.return_value.filter.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = (
            []
        )

        storage.get_recent_documents_with_chunks(collection_id, limit=10)

        # Verify query was constructed (actual optimization would be tested in integration tests)
        mock_session.query.assert_called_once()

    def test_connection_pool_efficiency(self, storage, mock_session):
        """Test efficient use of connection pool."""
        # Test that long-running operations don't hold connections unnecessarily
        with patch.object(storage, "_execute_with_retry") as mock_execute:
            mock_execute.return_value = []

            storage.get_job_statistics()

            # Verify connection was released properly
            mock_execute.assert_called_once()
