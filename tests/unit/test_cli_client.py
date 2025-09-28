"""
Test the CLI client with new separate/merge methods.
"""
import pytest
from unittest.mock import Mock, patch
import requests
from src.fileintel.cli.client import FileIntelAPI


class TestFileIntelAPI:
    """Test CLI client functionality with new job types."""

    @pytest.fixture
    def api_client(self):
        """FileIntel API client."""
        return FileIntelAPI("http://localhost:8000/api/v1")

    @pytest.fixture
    def mock_response(self):
        """Mock successful HTTP response."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"job_id": "test-job-123"}
        return response

    def test_query_collection_separate(self, api_client, mock_response):
        """Test collection query with separate mode."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.query_collection_separate(
                "collection-123", "What is this about?"
            )

            # Verify request was made correctly
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0] == (
                "POST",
                "http://localhost:8000/api/v1/collections/collection-123/question",
            )
            assert call_args[1]["json"]["question"] == "What is this about?"
            assert call_args[1]["json"]["job_type"] == "question_separate"
            assert result == {"job_id": "test-job-123"}

    def test_query_collection_merge(self, api_client, mock_response):
        """Test collection query with merge mode."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.query_collection_merge(
                "collection-123", "What is this about?"
            )

            # Verify request was made correctly
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0] == (
                "POST",
                "http://localhost:8000/api/v1/collections/collection-123/question",
            )
            assert call_args[1]["json"]["question"] == "What is this about?"
            assert call_args[1]["json"]["job_type"] == "question_merge"
            assert result == {"job_id": "test-job-123"}

    def test_query_collection_backward_compatibility_merge(
        self, api_client, mock_response
    ):
        """Test backward compatible query_collection defaults to merge mode."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.query_collection(
                "collection-123", "What is this about?"
            )

            # Verify request was made correctly (defaults to merge)
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[1]["json"]["question"] == "What is this about?"
            assert call_args[1]["json"]["job_type"] == "question_merge"

    def test_query_collection_backward_compatibility_separate(
        self, api_client, mock_response
    ):
        """Test backward compatible query_collection with merge_mode=False."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.query_collection(
                "collection-123", "What is this about?", merge_mode=False
            )

            # Verify request was made correctly (separate mode)
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[1]["json"]["question"] == "What is this about?"
            assert call_args[1]["json"]["job_type"] == "question_separate"

    def test_analyze_collection_separate(self, api_client, mock_response):
        """Test collection analysis with separate mode."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.analyze_collection_separate(
                "collection-123", "summarize"
            )

            # Verify request was made correctly
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0] == (
                "POST",
                "http://localhost:8000/api/v1/collections/collection-123/analyze",
            )
            assert call_args[1]["json"]["task_name"] == "summarize"
            assert call_args[1]["json"]["job_type"] == "analysis_separate"
            assert result == {"job_id": "test-job-123"}

    def test_analyze_collection_merge(self, api_client, mock_response):
        """Test collection analysis with merge mode."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.analyze_collection_merge(
                "collection-123", "extract_entities"
            )

            # Verify request was made correctly
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0] == (
                "POST",
                "http://localhost:8000/api/v1/collections/collection-123/analyze",
            )
            assert call_args[1]["json"]["task_name"] == "extract_entities"
            assert call_args[1]["json"]["job_type"] == "analysis_merge"
            assert result == {"job_id": "test-job-123"}

    def test_analyze_collection_backward_compatibility(self, api_client, mock_response):
        """Test backward compatible analyze_collection defaults to merge mode."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.analyze_collection("collection-123", "categorize")

            # Verify request was made correctly (defaults to merge)
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[1]["json"]["task_name"] == "categorize"
            assert call_args[1]["json"]["job_type"] == "analysis_merge"

    def test_query_document_separate(self, api_client, mock_response):
        """Test document query with separate mode."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.query_document_separate(
                "collection-123", "doc-456", "What does this document say?"
            )

            # Verify request was made correctly
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0] == (
                "POST",
                "http://localhost:8000/api/v1/collections/collection-123/documents/doc-456/question",
            )
            assert call_args[1]["json"]["question"] == "What does this document say?"
            assert call_args[1]["json"]["job_type"] == "document_question_separate"

    def test_query_document_merge(self, api_client, mock_response):
        """Test document query with merge mode."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.query_document_merge(
                "collection-123", "doc-456", "What does this document say?"
            )

            # Verify request was made correctly
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0] == (
                "POST",
                "http://localhost:8000/api/v1/collections/collection-123/documents/doc-456/question",
            )
            assert call_args[1]["json"]["question"] == "What does this document say?"
            assert call_args[1]["json"]["job_type"] == "document_question_merge"

    def test_query_document_backward_compatibility(self, api_client, mock_response):
        """Test backward compatible query_document defaults to merge mode."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.query_document(
                "collection-123", "doc-456", "What does this document say?"
            )

            # Verify request was made correctly (defaults to merge)
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[1]["json"]["question"] == "What does this document say?"
            assert call_args[1]["json"]["job_type"] == "document_question_merge"

    def test_analyze_document_unchanged(self, api_client, mock_response):
        """Test document analysis remains unchanged (no job_type parameter)."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            result = api_client.analyze_document(
                "collection-123", "doc-456", "classify"
            )

            # Verify request was made correctly (no job_type)
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0] == (
                "POST",
                "http://localhost:8000/api/v1/collections/collection-123/documents/doc-456/analyze",
            )
            assert call_args[1]["json"]["task_name"] == "classify"
            assert (
                "job_type" not in call_args[1]["json"]
            )  # No job_type for document analysis

    def test_error_handling(self, api_client):
        """Test error handling in API client."""
        with patch("requests.request") as mock_request:
            mock_request.side_effect = requests.exceptions.RequestException(
                "Network error"
            )

            with pytest.raises(requests.exceptions.RequestException):
                api_client.query_collection_separate("collection-123", "What is this?")

    def test_base_url_configuration(self):
        """Test API client with different base URLs."""
        client = FileIntelAPI("https://example.com/api/v2")
        assert client.base_url == "https://example.com/api/v2"

        # Test with environment variable
        with patch.dict(
            "os.environ", {"FILEINTEL_API_BASE_URL": "https://prod.example.com/api/v1"}
        ):
            from src.fileintel.cli.client import API_BASE_URL

            # Note: This would need to reload the module to test properly

    def test_request_json_structure(self, api_client, mock_response):
        """Test that all requests include proper JSON structure."""
        with patch("requests.request", return_value=mock_response) as mock_request:
            # Test query separate
            api_client.query_collection_separate("coll-123", "question")
            call_args = mock_request.call_args
            json_data = call_args[1]["json"]

            assert "question" in json_data
            assert "task_name" in json_data  # Should have default
            assert "job_type" in json_data
            assert json_data["job_type"] == "question_separate"

            mock_request.reset_mock()

            # Test analysis merge
            api_client.analyze_collection_merge("coll-123", "custom_task")
            call_args = mock_request.call_args
            json_data = call_args[1]["json"]

            assert "task_name" in json_data
            assert "job_type" in json_data
            assert json_data["task_name"] == "custom_task"
            assert json_data["job_type"] == "analysis_merge"
