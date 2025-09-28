"""
Comprehensive tests for job queue operations including job lifecycle,
priority management, retry mechanisms, and edge case handling.
"""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from src.fileintel.storage.postgresql_storage import PostgreSQLStorage
from src.fileintel.storage.models import Job, DeadLetterJob, Collection


class TestJobCreationAndValidation:
    """Test job creation and validation logic."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_create_job_with_all_fields(self, storage, mock_session):
        """Test creating job with all optional fields specified."""
        job_data = {
            "job_type": "indexing",
            "job_data": {"document_id": "doc-123", "chunking_strategy": "sentence"},
            "collection_id": str(uuid.uuid4()),
            "priority": 8,
            "retry_count": 0,
            "max_retries": 5,
            "retry_delay": 120,
            "scheduled_at": datetime.utcnow() + timedelta(minutes=30),
        }

        mock_job = Mock(spec=Job)
        mock_job.id = str(uuid.uuid4())
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

    def test_create_job_with_minimal_fields(self, storage, mock_session):
        """Test creating job with only required fields."""
        job_data = {
            "job_type": "analysis",
            "job_data": {"query": "summarize documents"},
        }

        mock_job = Mock(spec=Job)
        mock_job.id = str(uuid.uuid4())
        mock_job.priority = 5  # Default priority
        mock_job.max_retries = 3  # Default max retries

        with patch("src.fileintel.storage.postgresql_storage.Job") as mock_job_class:
            mock_job_class.return_value = mock_job
            result = storage.create_job(**job_data)

            assert result == mock_job
            mock_session.add.assert_called_once()

    def test_create_job_with_invalid_priority(self, storage, mock_session):
        """Test job creation with invalid priority values."""
        job_data = {
            "job_type": "indexing",
            "job_data": {"document_id": "doc-123"},
            "priority": 15,  # Invalid priority (> 10)
        }

        mock_session.add.side_effect = IntegrityError("Invalid priority", None, None)

        with patch("src.fileintel.storage.postgresql_storage.Job"):
            with pytest.raises(IntegrityError):
                storage.create_job(**job_data)

    def test_create_job_with_invalid_json_data(self, storage, mock_session):
        """Test job creation with non-serializable job data."""
        job_data = {
            "job_type": "indexing",
            "job_data": {"callback": lambda x: x},  # Non-serializable
        }

        with patch("json.dumps") as mock_json:
            mock_json.side_effect = TypeError("Not JSON serializable")

            with patch("src.fileintel.storage.postgresql_storage.Job"):
                with pytest.raises(TypeError):
                    storage.create_job(**job_data)

    def test_validate_job_type_supported(self, storage, mock_session):
        """Test validation of supported job types."""
        supported_types = ["indexing", "analysis", "graphrag", "citation_search"]

        for job_type in supported_types:
            job_data = {"job_type": job_type, "job_data": {"test": "data"}}

            mock_job = Mock(spec=Job)
            with patch(
                "src.fileintel.storage.postgresql_storage.Job"
            ) as mock_job_class:
                mock_job_class.return_value = mock_job
                result = storage.create_job(**job_data)
                assert result == mock_job

    def test_validate_job_type_unsupported(self, storage, mock_session):
        """Test rejection of unsupported job types."""
        job_data = {"job_type": "unsupported_type", "job_data": {"test": "data"}}

        with pytest.raises(ValueError) as exc_info:
            storage.validate_and_create_job(**job_data)

        assert "Unsupported job type" in str(exc_info.value)

    def test_create_scheduled_job_future_execution(self, storage, mock_session):
        """Test creating job scheduled for future execution."""
        future_time = datetime.utcnow() + timedelta(hours=2)
        job_data = {
            "job_type": "indexing",
            "job_data": {"document_id": "doc-123"},
            "scheduled_at": future_time,
        }

        mock_job = Mock(spec=Job)
        mock_job.status = "scheduled"
        mock_job.scheduled_at = future_time

        with patch("src.fileintel.storage.postgresql_storage.Job") as mock_job_class:
            mock_job_class.return_value = mock_job
            result = storage.create_job(**job_data)

            assert result == mock_job
            assert result.status == "scheduled"


class TestJobRetrieval:
    """Test job retrieval and querying operations."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_get_next_jobs_priority_order(self, storage, mock_session):
        """Test retrieving next jobs in priority order."""
        batch_size = 3
        mock_jobs = [
            Mock(spec=Job, priority=10, status="pending"),
            Mock(spec=Job, priority=8, status="pending"),
            Mock(spec=Job, priority=5, status="pending"),
        ]

        mock_query = mock_session.query.return_value
        mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            mock_jobs
        )

        result = storage.get_next_jobs(batch_size)

        assert len(result) == batch_size
        assert result[0].priority >= result[1].priority >= result[2].priority
        mock_session.query.assert_called_once_with(Job)

    def test_get_next_jobs_include_retry_ready(self, storage, mock_session):
        """Test that ready-for-retry jobs are included in next jobs."""
        batch_size = 2
        current_time = datetime.utcnow()

        mock_jobs = [
            Mock(spec=Job, status="pending", priority=8),
            Mock(
                spec=Job,
                status="retry_pending",
                priority=9,
                next_retry_at=current_time - timedelta(minutes=1),
            ),
        ]

        mock_query = mock_session.query.return_value
        mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            mock_jobs
        )

        result = storage.get_next_jobs(batch_size)

        assert len(result) == batch_size
        # Should include both pending and ready retry jobs
        statuses = {job.status for job in result}
        assert "pending" in statuses
        assert "retry_pending" in statuses

    def test_get_next_jobs_exclude_future_retry(self, storage, mock_session):
        """Test that jobs with future retry times are excluded."""
        batch_size = 3
        future_time = datetime.utcnow() + timedelta(minutes=30)

        mock_jobs = [
            Mock(spec=Job, status="pending", priority=8),
            Mock(spec=Job, status="pending", priority=7),
        ]

        # Should not include job with future retry time
        mock_query = mock_session.query.return_value
        mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            mock_jobs
        )

        result = storage.get_next_jobs(batch_size)

        assert len(result) == 2
        for job in result:
            assert job.status == "pending"

    def test_get_jobs_by_status(self, storage, mock_session):
        """Test retrieving jobs by status."""
        status = "running"
        mock_jobs = [Mock(spec=Job, status=status) for _ in range(5)]

        mock_session.query.return_value.filter.return_value.all.return_value = mock_jobs

        result = storage.get_jobs_by_status(status)

        assert len(result) == 5
        for job in result:
            assert job.status == status

    def test_get_jobs_by_collection_and_type(self, storage, mock_session):
        """Test retrieving jobs filtered by collection and type."""
        collection_id = str(uuid.uuid4())
        job_type = "indexing"
        mock_jobs = [Mock(spec=Job, collection_id=collection_id, job_type=job_type)]

        mock_session.query.return_value.filter.return_value.all.return_value = mock_jobs

        result = storage.get_jobs_by_collection_and_type(collection_id, job_type)

        assert len(result) == 1
        assert result[0].collection_id == collection_id
        assert result[0].job_type == job_type

    def test_get_stale_jobs(self, storage, mock_session):
        """Test retrieving jobs that have been running too long."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        mock_stale_jobs = [
            Mock(
                spec=Job,
                status="running",
                started_at=cutoff_time - timedelta(minutes=30),
            ),
            Mock(
                spec=Job,
                status="running",
                started_at=cutoff_time - timedelta(minutes=10),
            ),
        ]

        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_stale_jobs
        )

        result = storage.get_stale_jobs(cutoff_time)

        assert len(result) == 2
        for job in result:
            assert job.status == "running"

    def test_get_jobs_by_worker_id(self, storage, mock_session):
        """Test retrieving jobs assigned to specific worker."""
        worker_id = "worker-123"
        mock_jobs = [Mock(spec=Job, worker_id=worker_id, status="running")]

        mock_session.query.return_value.filter.return_value.all.return_value = mock_jobs

        result = storage.get_jobs_by_worker(worker_id)

        assert len(result) == 1
        assert result[0].worker_id == worker_id

    def test_get_job_history_for_collection(self, storage, mock_session):
        """Test retrieving job history for a collection."""
        collection_id = str(uuid.uuid4())
        limit = 10
        mock_jobs = [Mock(spec=Job, collection_id=collection_id) for _ in range(limit)]

        mock_query = mock_session.query.return_value
        mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            mock_jobs
        )

        result = storage.get_job_history_for_collection(collection_id, limit)

        assert len(result) == limit
        for job in result:
            assert job.collection_id == collection_id


class TestJobStatusManagement:
    """Test job status transitions and updates."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_update_job_status_simple(self, storage, mock_session):
        """Test simple job status update."""
        job_id = str(uuid.uuid4())
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

    def test_update_job_status_with_result(self, storage, mock_session):
        """Test job status update with result data."""
        job_id = str(uuid.uuid4())
        status = "completed"
        result_data = {
            "documents_processed": 10,
            "chunks_created": 50,
            "processing_time": 125.5,
        }

        mock_job = Mock(spec=Job)
        mock_job.status = "running"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.update_job_with_result(job_id, status, result_data)

        assert result is True
        assert mock_job.status == status
        assert mock_job.result == result_data
        mock_session.commit.assert_called_once()

    def test_update_job_status_with_worker_assignment(self, storage, mock_session):
        """Test job status update with worker assignment."""
        job_id = str(uuid.uuid4())
        worker_id = "worker-456"
        status = "running"

        mock_job = Mock(spec=Job)
        mock_job.status = "pending"
        mock_job.worker_id = None
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.assign_job_to_worker(job_id, worker_id, status)

        assert result is True
        assert mock_job.status == status
        assert mock_job.worker_id == worker_id
        assert mock_job.started_at is not None
        mock_session.commit.assert_called_once()

    def test_update_job_status_invalid_transition(self, storage, mock_session):
        """Test invalid job status transitions are rejected."""
        job_id = str(uuid.uuid4())
        invalid_status = "pending"  # Can't go back to pending from completed

        mock_job = Mock(spec=Job)
        mock_job.status = "completed"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        with pytest.raises(ValueError) as exc_info:
            storage.update_job_status_with_validation(job_id, invalid_status)

        assert "Invalid status transition" in str(exc_info.value)

    def test_batch_update_job_status(self, storage, mock_session):
        """Test batch updating multiple job statuses."""
        job_ids = [str(uuid.uuid4()) for _ in range(5)]
        new_status = "cancelled"

        mock_session.query.return_value.filter.return_value.update.return_value = len(
            job_ids
        )
        mock_session.commit.return_value = None

        result = storage.batch_update_job_status(job_ids, new_status)

        assert result == len(job_ids)
        mock_session.commit.assert_called_once()

    def test_cancel_job_with_cleanup(self, storage, mock_session):
        """Test job cancellation with resource cleanup."""
        job_id = str(uuid.uuid4())
        cancellation_reason = "User requested cancellation"

        mock_job = Mock(spec=Job)
        mock_job.status = "running"
        mock_job.worker_id = "worker-123"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.cancel_job_with_cleanup(job_id, cancellation_reason)

        assert result is True
        assert mock_job.status == "cancelled"
        assert mock_job.cancellation_reason == cancellation_reason
        assert mock_job.cancelled_at is not None
        mock_session.commit.assert_called_once()

    def test_mark_job_as_failed_with_error(self, storage, mock_session):
        """Test marking job as failed with error details."""
        job_id = str(uuid.uuid4())
        error_message = "Database connection timeout"
        error_traceback = "Traceback (most recent call last)..."

        mock_job = Mock(spec=Job)
        mock_job.status = "running"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.mark_job_as_failed(job_id, error_message, error_traceback)

        assert result is True
        assert mock_job.status == "failed"
        assert mock_job.error_message == error_message
        assert mock_job.error_traceback == error_traceback
        assert mock_job.failed_at is not None
        mock_session.commit.assert_called_once()


class TestJobRetryMechanisms:
    """Test job retry logic and exponential backoff."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_schedule_job_retry_within_limits(self, storage, mock_session):
        """Test scheduling job retry when within retry limits."""
        job_id = str(uuid.uuid4())
        base_delay = 60

        mock_job = Mock(spec=Job)
        mock_job.retry_count = 1
        mock_job.max_retries = 3
        mock_job.retry_delay = base_delay
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        with patch("random.uniform", return_value=1.0):  # No jitter for testing
            result = storage.schedule_job_retry(job_id)

            assert result is True
            assert mock_job.status == "retry_pending"
            assert mock_job.retry_count == 2
            # Exponential backoff: base_delay * (2 ** retry_count)
            expected_delay = base_delay * (2**2)
            assert mock_job.next_retry_at is not None
            mock_session.commit.assert_called_once()

    def test_schedule_job_retry_exceeds_limits(self, storage, mock_session):
        """Test job retry fails when retry limits exceeded."""
        job_id = str(uuid.uuid4())

        mock_job = Mock(spec=Job)
        mock_job.retry_count = 3
        mock_job.max_retries = 3
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.schedule_job_retry(job_id)

        assert result is False
        # Job should remain in current status (not moved to retry_pending)
        assert mock_job.status != "retry_pending"
        mock_session.commit.assert_not_called()

    def test_calculate_retry_delay_exponential_backoff(self, storage, mock_session):
        """Test exponential backoff calculation with jitter."""
        base_delay = 30
        max_delay = 3600

        # Test different retry counts
        retry_delays = []
        for retry_count in range(5):
            with patch("random.uniform", return_value=1.0):  # No jitter
                delay = storage.calculate_retry_delay(
                    base_delay, retry_count, max_delay
                )
                retry_delays.append(delay)

        # Should follow exponential pattern
        assert retry_delays[0] == base_delay * (2**0)  # 30
        assert retry_delays[1] == base_delay * (2**1)  # 60
        assert retry_delays[2] == base_delay * (2**2)  # 120
        assert retry_delays[3] == base_delay * (2**3)  # 240

        # Should not exceed max delay
        assert all(delay <= max_delay for delay in retry_delays)

    def test_retry_delay_with_jitter(self, storage, mock_session):
        """Test retry delay calculation includes jitter."""
        base_delay = 60
        retry_count = 2
        max_delay = 3600

        with patch("random.uniform", return_value=1.2):  # 20% increase
            delay = storage.calculate_retry_delay(base_delay, retry_count, max_delay)

            base_exponential_delay = base_delay * (2**retry_count)
            expected_delay = base_exponential_delay * 1.2
            assert delay == expected_delay

    def test_get_jobs_ready_for_retry(self, storage, mock_session):
        """Test retrieving jobs ready for retry."""
        current_time = datetime.utcnow()
        ready_jobs = [
            Mock(
                spec=Job,
                status="retry_pending",
                next_retry_at=current_time - timedelta(minutes=1),
            ),
            Mock(
                spec=Job,
                status="retry_pending",
                next_retry_at=current_time - timedelta(seconds=30),
            ),
        ]

        mock_session.query.return_value.filter.return_value.all.return_value = (
            ready_jobs
        )

        result = storage.get_jobs_ready_for_retry()

        assert len(result) == 2
        for job in result:
            assert job.status == "retry_pending"
            assert job.next_retry_at <= current_time

    def test_reset_job_for_retry(self, storage, mock_session):
        """Test resetting job state for retry attempt."""
        job_id = str(uuid.uuid4())

        mock_job = Mock(spec=Job)
        mock_job.status = "retry_pending"
        mock_job.worker_id = "old-worker-123"
        mock_job.error_message = "Previous error"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = storage.reset_job_for_retry(job_id)

        assert result is True
        assert mock_job.status == "pending"
        assert mock_job.worker_id is None
        assert mock_job.error_message is None
        assert mock_job.next_retry_at is None
        assert mock_job.last_retry_at is not None
        mock_session.commit.assert_called_once()


class TestDeadLetterQueue:
    """Test dead letter queue operations for failed jobs."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_move_job_to_dead_letter_queue(self, storage, mock_session):
        """Test moving exhausted job to dead letter queue."""
        job_id = str(uuid.uuid4())
        failure_reason = "Exceeded maximum retry attempts"

        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.job_type = "indexing"
        mock_job.job_data = {"document_id": "doc-123"}
        mock_job.retry_count = 3
        mock_job.max_retries = 3
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        mock_dead_letter_job = Mock(spec=DeadLetterJob)
        mock_session.add.return_value = None
        mock_session.commit.return_value = None

        with patch(
            "src.fileintel.storage.postgresql_storage.DeadLetterJob"
        ) as mock_dlj_class:
            mock_dlj_class.return_value = mock_dead_letter_job
            result = storage.move_job_to_dead_letter_queue(job_id, failure_reason)

            assert result is True
            mock_session.add.assert_called_once_with(mock_dead_letter_job)
            mock_session.delete.assert_called_once_with(mock_job)
            mock_session.commit.assert_called()

    def test_get_dead_letter_jobs(self, storage, mock_session):
        """Test retrieving dead letter queue jobs."""
        limit = 20
        mock_dead_jobs = [Mock(spec=DeadLetterJob) for _ in range(limit)]

        mock_query = mock_session.query.return_value
        mock_query.order_by.return_value.limit.return_value.all.return_value = (
            mock_dead_jobs
        )

        result = storage.get_dead_letter_jobs(limit)

        assert len(result) == limit
        mock_session.query.assert_called_once_with(DeadLetterJob)

    def test_get_dead_letter_jobs_by_type(self, storage, mock_session):
        """Test retrieving dead letter jobs filtered by job type."""
        job_type = "indexing"
        mock_dead_jobs = [Mock(spec=DeadLetterJob, job_type=job_type)]

        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            mock_dead_jobs
        )

        result = storage.get_dead_letter_jobs_by_type(job_type)

        assert len(result) == 1
        assert result[0].job_type == job_type

    def test_retry_dead_letter_job(self, storage, mock_session):
        """Test retrying job from dead letter queue."""
        dead_job_id = 123

        mock_dead_job = Mock(spec=DeadLetterJob)
        mock_dead_job.job_type = "indexing"
        mock_dead_job.job_data = {"document_id": "doc-123"}
        mock_dead_job.original_job_id = str(uuid.uuid4())
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_dead_job
        )

        mock_new_job = Mock(spec=Job)
        mock_session.add.return_value = None
        mock_session.commit.return_value = None

        with patch("src.fileintel.storage.postgresql_storage.Job") as mock_job_class:
            mock_job_class.return_value = mock_new_job
            result = storage.retry_dead_letter_job(dead_job_id)

            assert result == mock_new_job
            mock_session.add.assert_called_once_with(mock_new_job)
            mock_session.delete.assert_called_once_with(mock_dead_job)
            mock_session.commit.assert_called()

    def test_cleanup_old_dead_letter_jobs(self, storage, mock_session):
        """Test cleanup of old dead letter queue entries."""
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        deleted_count = 15

        mock_session.query.return_value.filter.return_value.delete.return_value = (
            deleted_count
        )
        mock_session.commit.return_value = None

        result = storage.cleanup_old_dead_letter_jobs(cutoff_date)

        assert result == deleted_count
        mock_session.query.assert_called_once_with(DeadLetterJob)
        mock_session.commit.assert_called_once()

    def test_get_dead_letter_job_statistics(self, storage, mock_session):
        """Test getting dead letter queue statistics."""
        # Mock counts by job type
        mock_session.query.return_value.group_by.return_value.all.return_value = [
            ("indexing", 10),
            ("analysis", 5),
            ("graphrag", 2),
        ]

        result = storage.get_dead_letter_job_statistics()

        expected_stats = {
            "total_jobs": 17,
            "by_job_type": {"indexing": 10, "analysis": 5, "graphrag": 2},
        }

        assert result == expected_stats
        mock_session.query.assert_called_once_with(DeadLetterJob)


class TestJobQueueMetrics:
    """Test job queue metrics and monitoring."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_get_job_queue_statistics(self, storage, mock_session):
        """Test comprehensive job queue statistics."""
        # Mock different count queries
        count_responses = [
            25,
            10,
            5,
            3,
            2,
            1,
            8,
        ]  # pending, running, completed, failed, cancelled, retry_pending, scheduled
        mock_session.query.return_value.filter.return_value.count.side_effect = (
            count_responses
        )

        result = storage.get_job_queue_statistics()

        expected_stats = {
            "pending": 25,
            "running": 10,
            "completed": 5,
            "failed": 3,
            "cancelled": 2,
            "retry_pending": 1,
            "scheduled": 8,
            "total": sum(count_responses),
            "active": 25 + 10 + 1 + 8,  # pending + running + retry_pending + scheduled
        }

        assert result == expected_stats

    def test_get_job_processing_metrics(self, storage, mock_session):
        """Test job processing performance metrics."""
        # Mock average processing times by job type
        mock_session.query.return_value.group_by.return_value.all.return_value = [
            ("indexing", 125.5, 50),
            ("analysis", 45.2, 20),
            ("graphrag", 300.8, 5),
        ]

        result = storage.get_job_processing_metrics()

        expected_metrics = {
            "average_processing_time_by_type": {
                "indexing": 125.5,
                "analysis": 45.2,
                "graphrag": 300.8,
            },
            "job_count_by_type": {"indexing": 50, "analysis": 20, "graphrag": 5},
        }

        assert result == expected_metrics

    def test_get_job_failure_rate(self, storage, mock_session):
        """Test calculating job failure rates."""
        # Mock success and failure counts
        mock_session.query.return_value.filter.return_value.count.side_effect = [
            80,
            20,
        ]  # success, failed

        result = storage.get_job_failure_rate()

        expected_rate = {
            "total_jobs": 100,
            "successful_jobs": 80,
            "failed_jobs": 20,
            "failure_rate": 0.2,
            "success_rate": 0.8,
        }

        assert result == expected_rate

    def test_get_worker_job_distribution(self, storage, mock_session):
        """Test getting job distribution across workers."""
        mock_session.query.return_value.group_by.return_value.all.return_value = [
            ("worker-1", 15),
            ("worker-2", 12),
            ("worker-3", 8),
            (None, 5),  # Unassigned jobs
        ]

        result = storage.get_worker_job_distribution()

        expected_distribution = {
            "worker-1": 15,
            "worker-2": 12,
            "worker-3": 8,
            "unassigned": 5,
        }

        assert result == expected_distribution

    def test_get_job_queue_depth_over_time(self, storage, mock_session):
        """Test getting job queue depth trends."""
        # Mock hourly job counts for last 24 hours
        mock_data = [
            (datetime.utcnow() - timedelta(hours=i), 20 - i) for i in range(24)
        ]
        mock_session.query.return_value.group_by.return_value.order_by.return_value.all.return_value = (
            mock_data
        )

        result = storage.get_job_queue_depth_trends(hours=24)

        assert len(result) == 24
        assert all("timestamp" in entry and "queue_depth" in entry for entry in result)


class TestJobQueueEdgeCases:
    """Test edge cases and error conditions in job queue management."""

    @pytest.fixture
    def mock_session(self):
        return Mock(spec=Session)

    @pytest.fixture
    def storage(self, mock_session):
        return PostgreSQLStorage(mock_session)

    def test_concurrent_job_assignment_conflict(self, storage, mock_session):
        """Test handling concurrent job assignment conflicts."""
        job_id = str(uuid.uuid4())
        worker_id = "worker-123"

        mock_job = Mock(spec=Job)
        mock_job.status = "pending"
        mock_job.worker_id = None
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        # Simulate optimistic locking failure
        mock_session.commit.side_effect = SQLAlchemyError("Concurrent modification")

        with pytest.raises(SQLAlchemyError):
            storage.assign_job_to_worker(job_id, worker_id, "running")

        mock_session.rollback.assert_called_once()

    def test_job_cleanup_orphaned_locks(self, storage, mock_session):
        """Test cleanup of jobs with orphaned worker locks."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)
        mock_orphaned_jobs = [
            Mock(spec=Job, status="running", worker_id="dead-worker-1"),
            Mock(spec=Job, status="running", worker_id="dead-worker-2"),
        ]

        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_orphaned_jobs
        )

        result = storage.cleanup_orphaned_job_locks(cutoff_time)

        assert result == len(mock_orphaned_jobs)
        for job in mock_orphaned_jobs:
            assert job.status == "pending"
            assert job.worker_id is None
        mock_session.commit.assert_called_once()

    def test_job_queue_at_capacity_limits(self, storage, mock_session):
        """Test job queue behavior at capacity limits."""
        max_pending_jobs = 1000
        current_pending = 1000

        mock_session.query.return_value.filter.return_value.count.return_value = (
            current_pending
        )

        job_data = {"job_type": "indexing", "job_data": {"document_id": "doc-123"}}

        with pytest.raises(ValueError) as exc_info:
            storage.create_job_with_capacity_check(max_pending_jobs, **job_data)

        assert "Queue at capacity" in str(exc_info.value)

    def test_massive_job_batch_operations(self, storage, mock_session):
        """Test handling of large batch operations."""
        job_ids = [str(uuid.uuid4()) for _ in range(10000)]
        batch_size = 1000

        # Mock batch processing
        total_updated = 0
        for i in range(0, len(job_ids), batch_size):
            batch_count = min(batch_size, len(job_ids) - i)
            total_updated += batch_count

        mock_session.query.return_value.filter.return_value.update.side_effect = [
            batch_size for _ in range(len(job_ids) // batch_size)
        ]

        result = storage.batch_update_job_status_chunked(
            job_ids, "cancelled", batch_size
        )

        assert result == len(job_ids)
        # Should commit after each batch
        assert mock_session.commit.call_count == len(job_ids) // batch_size

    def test_job_data_corruption_handling(self, storage, mock_session):
        """Test handling of corrupted job data."""
        job_id = str(uuid.uuid4())

        mock_job = Mock(spec=Job)
        mock_job.job_data = "corrupted_non_json_string"
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        with patch("json.loads") as mock_json:
            mock_json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

            result = storage.get_job_data_safely(job_id)

            assert result is None  # Should return None for corrupted data

    def test_database_constraint_violation_recovery(self, storage, mock_session):
        """Test recovery from database constraint violations."""
        job_data = {
            "job_type": "indexing",
            "job_data": {"document_id": "doc-123"},
            "collection_id": "nonexistent-collection",  # Foreign key violation
        }

        mock_session.add.side_effect = IntegrityError(
            "Foreign key violation", None, None
        )
        mock_session.rollback.return_value = None

        with patch("src.fileintel.storage.postgresql_storage.Job"):
            with pytest.raises(IntegrityError):
                storage.create_job(**job_data)

            mock_session.rollback.assert_called_once()
