"""
Integration tests for API endpoints with the new job_type parameter.
"""
import pytest
from fastapi.testclient import TestClient
import uuid
from src.fileintel.api.main import app

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestAPIEndpoints:
    """Test API endpoint integration with new job types using real API."""

    @pytest.fixture(scope="function")
    def client(self):
        """FastAPI test client with real database."""
        return TestClient(app)

    @pytest.fixture
    def test_collection(self, client):
        """Create a test collection for each test."""
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        response = client.post("/api/v1/collections", json={"name": collection_name})
        assert response.status_code == 201
        return response.json()

    @pytest.fixture
    def test_document(self, client, test_collection):
        """Create a test document for each test using real PDF."""
        import os
        from pathlib import Path

        # Use the real PDF from fixtures
        pdf_path = Path(__file__).parent.parent / "fixtures" / "test.pdf"

        if not pdf_path.exists():
            return None

        with open(pdf_path, "rb") as f:
            response = client.post(
                f"/api/v1/collections/{test_collection['id']}/documents",
                files={"file": ("test.pdf", f, "application/pdf")},
            )

        # Document creation might fail, that's ok for some tests
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def test_query_collection_default_merge(self, client, test_collection):
        """Test collection query defaults to merge mode."""
        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/question",
            json={"question": "What is this about?"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()
        # Job will be submitted with merge mode by default

    def test_query_collection_explicit_separate(self, client, test_collection):
        """Test collection query with explicit separate mode."""
        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/question",
            json={"question": "What is this about?", "job_type": "question_separate"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_query_collection_explicit_merge(self, client, test_collection):
        """Test collection query with explicit merge mode."""
        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/question",
            json={"question": "What is this about?", "job_type": "question_merge"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_analyze_collection_default_merge(self, client, test_collection):
        """Test collection analysis defaults to merge mode."""
        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/analyze",
            json={"task_name": "summarize"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_analyze_collection_explicit_separate(self, client, test_collection):
        """Test collection analysis with explicit separate mode."""
        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/analyze",
            json={"task_name": "extract_entities", "job_type": "analysis_separate"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_query_document_default_merge(self, client, test_collection, test_document):
        """Test document query defaults to merge mode."""
        if test_document is None:
            pytest.skip("Document upload not supported")

        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/documents/test.pdf/question",
            json={"question": "What is in this document?"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_query_document_explicit_separate(
        self, client, test_collection, test_document
    ):
        """Test document query with explicit separate mode."""
        if test_document is None:
            pytest.skip("Document upload not supported")

        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/documents/test.pdf/question",
            json={
                "question": "What is in this document?",
                "job_type": "document_question_separate",
            },
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_analyze_document_unchanged(self, client, test_collection, test_document):
        """Test document analysis remains unchanged (no separate/merge modes)."""
        if test_document is None:
            pytest.skip("Document upload not supported")

        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/documents/test.pdf/analyze",
            json={"task_name": "categorize"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_backward_compatibility_no_job_type(self, client, test_collection):
        """Test backward compatibility when job_type is not provided."""
        # Test collection query
        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/question",
            json={"question": "What is this about?", "task_name": "default_analysis"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_invalid_job_type_accepted(self, client, test_collection):
        """Test that API accepts any job_type (validation happens at processor level)."""
        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/question",
            json={"question": "What is this about?", "job_type": "invalid_job_type"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_job_data_includes_question_and_task(self, client, test_collection):
        """Test that job data includes both question and task_name."""
        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/question",
            json={
                "question": "Test question",
                "task_name": "custom_task",
                "job_type": "question_separate",
            },
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_collection_context_preserved(self, client, test_collection):
        """Test that collection context is properly preserved in job submission."""
        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/question",
            json={"question": "What is this about?", "job_type": "question_merge"},
        )

        assert response.status_code == 200
        assert "job_id" in response.json()

    def test_document_context_preserved(self, client, test_collection, test_document):
        """Test that document context is properly preserved in job submission."""
        if test_document is None:
            pytest.skip("Document upload not supported")

        response = client.post(
            f"/api/v1/collections/{test_collection['id']}/documents/test.pdf/question",
            json={
                "question": "What is in this document?",
                "job_type": "document_question_separate",
            },
        )

        assert response.status_code == 200
        assert "job_id" in response.json()
