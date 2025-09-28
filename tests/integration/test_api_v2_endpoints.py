"""
Integration tests for v2 API endpoints.

Tests the new Celery task-based API endpoints that replace the old job-based system.
"""
import pytest
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime

from src.fileintel.api.main import app
from src.fileintel.storage.models import Collection, Document


@pytest.mark.integration
@pytest.mark.v2_api
class TestV2APIEndpoints:
    """Test v2 API endpoints integration."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_storage(self):
        """Mock storage for API tests."""
        storage = Mock()
        storage.create_collection = Mock()
        storage.get_collection = Mock()
        storage.get_all_collections = Mock(return_value=[])
        storage.delete_collection = Mock()
        storage.upload_document = Mock()
        storage.get_documents_in_collection = Mock(return_value=[])
        return storage

    @pytest.fixture
    def sample_collection(self):
        """Sample collection for testing."""
        return Collection(
            id=str(uuid.uuid4()),
            name="test_collection",
            description="Test collection for API testing",
            created_at=datetime.now(),
        )

    @pytest.fixture
    def api_headers(self):
        """Standard API headers with authentication."""
        return {
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
        }

    @patch("src.fileintel.api.dependencies.get_storage")
    def test_create_collection_v2(
        self, mock_get_storage, client, mock_storage, api_headers
    ):
        """Test collection creation via v2 API."""
        # Setup mock
        mock_get_storage.return_value = mock_storage
        mock_storage.create_collection.return_value = Collection(
            id="test_id", name="test_collection", description="Test description"
        )

        # Make request
        response = client.post(
            "/api/v2/collections",
            json={"name": "test_collection", "description": "Test description"},
            headers=api_headers,
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "test_collection"

        # Verify storage was called
        mock_storage.create_collection.assert_called_once()

    @patch("src.fileintel.api.dependencies.get_storage")
    def test_list_collections_v2(
        self, mock_get_storage, client, mock_storage, api_headers, sample_collection
    ):
        """Test listing collections via v2 API."""
        # Setup mock
        mock_get_storage.return_value = mock_storage
        mock_storage.get_all_collections.return_value = [sample_collection]

        # Make request
        response = client.get("/api/v2/collections", headers=api_headers)

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 1
        assert data["data"][0]["name"] == "test_collection"

    @patch("src.fileintel.api.dependencies.get_storage")
    @patch(
        "src.fileintel.tasks.workflow_tasks.complete_collection_analysis.apply_async"
    )
    def test_submit_collection_processing_task(
        self,
        mock_task,
        mock_get_storage,
        client,
        mock_storage,
        api_headers,
        sample_collection,
    ):
        """Test submitting collection processing task via v2 API."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_storage.get_collection.return_value = sample_collection

        mock_result = Mock()
        mock_result.id = "task_123"
        mock_result.state = "PENDING"
        mock_task.return_value = mock_result

        # Make request
        response = client.post(
            f"/api/v2/collections/{sample_collection.id}/process",
            json={
                "task_type": "complete_analysis",
                "file_paths": ["/test/file1.pdf", "/test/file2.pdf"],
                "build_graph": True,
                "generate_embeddings": True,
            },
            headers=api_headers,
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == "task_123"
        assert data["data"]["task_state"] == "PENDING"

        # Verify task was submitted
        mock_task.assert_called_once()

    @patch("src.fileintel.api.dependencies.get_storage")
    def test_upload_document_v2(
        self, mock_get_storage, client, mock_storage, api_headers, sample_collection
    ):
        """Test document upload via v2 API."""
        # Setup mock
        mock_get_storage.return_value = mock_storage
        mock_storage.get_collection.return_value = sample_collection
        mock_storage.upload_document.return_value = Document(
            id="doc_123",
            collection_id=sample_collection.id,
            filename="test.pdf",
            original_filename="test.pdf",
            content_hash="abc123",
            file_size=1024,
            mime_type="application/pdf",
        )

        # Create test file
        test_file_content = b"test file content"

        # Make request
        response = client.post(
            f"/api/v2/collections/{sample_collection.id}/documents",
            files={"file": ("test.pdf", test_file_content, "application/pdf")},
            headers={
                "Authorization": "Bearer test_api_key"
            },  # multipart form needs different headers
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["filename"] == "test.pdf"

        # Verify storage was called
        mock_storage.upload_document.assert_called_once()

    @patch("src.fileintel.celery_config.get_task_status")
    def test_get_task_status_v2(self, mock_get_task_status, client, api_headers):
        """Test getting task status via v2 API."""
        # Setup mock
        mock_get_task_status.return_value = {
            "state": "SUCCESS",
            "result": {"status": "completed", "processed_documents": 5},
            "progress": {"current": 10, "total": 10, "description": "Completed"},
        }

        # Make request
        response = client.get("/api/v2/tasks/task_123", headers=api_headers)

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["task_state"] == "SUCCESS"
        assert data["data"]["result"]["status"] == "completed"

    @patch("src.fileintel.celery_config.cancel_task")
    def test_cancel_task_v2(self, mock_cancel_task, client, api_headers):
        """Test canceling task via v2 API."""
        # Setup mock
        mock_cancel_task.return_value = True

        # Make request
        response = client.post(
            "/api/v2/tasks/task_123/cancel", json={"force": False}, headers=api_headers
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["cancelled"] is True

        # Verify cancel was called
        mock_cancel_task.assert_called_once_with("task_123", terminate=False)

    @patch("src.fileintel.celery_config.get_active_tasks")
    def test_list_active_tasks_v2(self, mock_get_active_tasks, client, api_headers):
        """Test listing active tasks via v2 API."""
        # Setup mock
        mock_get_active_tasks.return_value = {
            "worker1": [
                {
                    "id": "task_123",
                    "name": "process_document",
                    "args": ["file.pdf"],
                    "kwargs": {"collection_id": "coll_123"},
                }
            ]
        }

        # Make request
        response = client.get("/api/v2/tasks/active", headers=api_headers)

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "worker1" in data["data"]
        assert len(data["data"]["worker1"]) == 1

    @patch("src.fileintel.celery_config.get_worker_stats")
    def test_get_task_metrics_v2(self, mock_get_worker_stats, client, api_headers):
        """Test getting task metrics via v2 API."""
        # Setup mock
        mock_get_worker_stats.return_value = {
            "total_workers": 2,
            "active_workers": 1,
            "total_active_tasks": 3,
            "total_pending_tasks": 5,
            "workers": {
                "worker1": {
                    "status": "online",
                    "active_tasks": 2,
                    "completed_tasks": 150,
                }
            },
        }

        # Make request
        response = client.get("/api/v2/tasks/metrics", headers=api_headers)

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["worker_count"] == 2
        assert data["data"]["active_tasks"] == 3

    @patch("src.fileintel.api.dependencies.get_storage")
    @patch("src.fileintel.tasks.document_tasks.process_document.apply_async")
    def test_submit_generic_task_v2(
        self, mock_task, mock_get_storage, client, mock_storage, api_headers
    ):
        """Test submitting generic task via v2 API."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage

        mock_result = Mock()
        mock_result.id = "task_456"
        mock_result.state = "PENDING"
        mock_task.return_value = mock_result

        # Make request
        response = client.post(
            "/api/v2/tasks/submit",
            json={
                "task_name": "process_document",
                "args": ["/test/document.pdf"],
                "kwargs": {"document_id": "doc_123", "collection_id": "coll_123"},
                "queue": "document_processing",
            },
            headers=api_headers,
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == "task_456"

    def test_api_authentication_required(self, client):
        """Test that API endpoints require authentication."""
        # Request without auth header
        response = client.get("/api/v2/collections")
        assert response.status_code == 401

        # Request with invalid auth
        response = client.get(
            "/api/v2/collections", headers={"Authorization": "Bearer invalid_key"}
        )
        assert response.status_code == 401

    @patch("src.fileintel.api.dependencies.get_storage")
    def test_error_handling_v2(self, mock_get_storage, client, api_headers):
        """Test error handling in v2 API endpoints."""
        # Setup mock to raise exception
        mock_get_storage.side_effect = Exception("Database connection failed")

        # Make request
        response = client.get("/api/v2/collections", headers=api_headers)

        # Verify error response
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    @patch("src.fileintel.api.dependencies.get_storage")
    def test_collection_not_found_v2(
        self, mock_get_storage, client, mock_storage, api_headers
    ):
        """Test handling of non-existent collections in v2 API."""
        # Setup mock
        mock_get_storage.return_value = mock_storage
        mock_storage.get_collection.return_value = None

        # Make request
        response = client.get("/api/v2/collections/nonexistent_id", headers=api_headers)

        # Verify 404 response
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @patch("src.fileintel.api.dependencies.get_storage")
    def test_websocket_connection_v2(self, mock_get_storage, client, mock_storage):
        """Test WebSocket connection for task monitoring."""
        # Setup mock
        mock_get_storage.return_value = mock_storage

        # Test WebSocket endpoint exists
        # Note: Full WebSocket testing would require more complex setup
        # This just verifies the endpoint is available
        with client.websocket_connect("/api/v2/ws/tasks") as websocket:
            # Should connect successfully (authentication handled in WebSocket)
            assert websocket is not None
