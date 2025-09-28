"""
Integration tests for GraphRAG Celery task integration.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.fileintel.tasks.graphrag_tasks import (
    build_graph_index,
    global_search_task,
    local_search_task,
)
from src.fileintel.rag.graph_rag.services.graphrag_service import GraphRAGService


@pytest.mark.integration
@pytest.mark.graphrag
@pytest.mark.celery
class TestGraphRAGTaskIntegration:
    """Test GraphRAG Celery task integration."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage for GraphRAG tasks."""
        storage = Mock()
        storage.get_documents_in_collection = Mock(return_value=[])
        storage.get_all_chunks_for_document = Mock(return_value=[])
        storage.save_graphrag_index_info = Mock()
        storage.get_graphrag_index_info = Mock(
            return_value={"index_path": "/test/path"}
        )
        return storage

    @pytest.fixture
    def sample_graph_documents(self):
        """Sample documents for GraphRAG indexing."""
        return [
            {
                "document_id": "doc1",
                "content": "This is a test document about artificial intelligence.",
            },
            {
                "document_id": "doc2",
                "content": "This document discusses machine learning algorithms.",
            },
        ]

    @patch("src.fileintel.tasks.graphrag_tasks.get_storage")
    @patch("src.fileintel.tasks.graphrag_tasks.GraphRAGService")
    def test_build_graph_index_task(
        self,
        mock_graphrag_service,
        mock_get_storage,
        mock_storage,
        sample_graph_documents,
    ):
        """Test GraphRAG index building task."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_service_instance = Mock(spec=GraphRAGService)
        mock_service_instance.build_index = AsyncMock(return_value="/test/workspace")
        mock_graphrag_service.return_value = mock_service_instance

        # Execute task
        result = build_graph_index(sample_graph_documents, "test_collection")

        # Verify GraphRAG service was called
        mock_graphrag_service.assert_called_once()
        mock_service_instance.build_index.assert_called_once()

        # Verify result structure
        assert result["status"] == "completed"
        assert result["collection_id"] == "test_collection"
        assert "workspace_path" in result

    @patch("src.fileintel.tasks.graphrag_tasks.get_storage")
    @patch("src.fileintel.tasks.graphrag_tasks.GraphRAGService")
    def test_global_search_task(
        self, mock_graphrag_service, mock_get_storage, mock_storage
    ):
        """Test GraphRAG global search task."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_service_instance = Mock(spec=GraphRAGService)
        mock_service_instance.global_search = AsyncMock(
            return_value={"answer": "Test answer", "sources": [], "confidence": 0.8}
        )
        mock_graphrag_service.return_value = mock_service_instance

        # Execute task
        result = global_search_task("What is AI?", "test_collection")

        # Verify GraphRAG service was called
        mock_service_instance.global_search.assert_called_once_with(
            query="What is AI?", collection_id="test_collection"
        )

        # Verify result structure
        assert result["status"] == "completed"
        assert result["query"] == "What is AI?"
        assert result["collection_id"] == "test_collection"
        assert "answer" in result

    @patch("src.fileintel.tasks.graphrag_tasks.get_storage")
    @patch("src.fileintel.tasks.graphrag_tasks.GraphRAGService")
    def test_local_search_task(
        self, mock_graphrag_service, mock_get_storage, mock_storage
    ):
        """Test GraphRAG local search task."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_service_instance = Mock(spec=GraphRAGService)
        mock_service_instance.local_search = AsyncMock(
            return_value={
                "answer": "Local search answer",
                "sources": ["doc1"],
                "confidence": 0.9,
            }
        )
        mock_graphrag_service.return_value = mock_service_instance

        # Execute task
        result = local_search_task(
            "What is machine learning?", "test_collection", "community1"
        )

        # Verify GraphRAG service was called
        mock_service_instance.local_search.assert_called_once_with(
            query="What is machine learning?",
            collection_id="test_collection",
            community="community1",
        )

        # Verify result structure
        assert result["status"] == "completed"
        assert result["search_type"] == "local"
        assert "answer" in result

    @patch("src.fileintel.tasks.graphrag_tasks.get_storage")
    def test_task_error_handling(self, mock_get_storage, mock_storage):
        """Test error handling in GraphRAG tasks."""
        # Setup mock to raise exception
        mock_get_storage.side_effect = Exception("Storage connection failed")

        # Execute task - should handle error gracefully
        result = build_graph_index([], "test_collection")

        # Verify error was handled
        assert result["status"] == "failed"
        assert "error" in result
        assert "Storage connection failed" in result["error"]

    def test_task_input_validation(self):
        """Test input validation in GraphRAG tasks."""
        # Test with invalid inputs
        result = build_graph_index(None, "test_collection")
        assert result["status"] == "failed"
        assert "error" in result

        result = global_search_task("", "test_collection")
        assert result["status"] == "failed"
        assert "error" in result

    @patch("src.fileintel.tasks.graphrag_tasks.get_storage")
    @patch("src.fileintel.tasks.graphrag_tasks.GraphRAGService")
    def test_async_task_execution(
        self, mock_graphrag_service, mock_get_storage, mock_storage
    ):
        """Test that GraphRAG tasks handle async operations correctly."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_service_instance = Mock(spec=GraphRAGService)

        # Mock async method that takes time
        async def slow_build_index(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate work
            return "/test/workspace"

        mock_service_instance.build_index = slow_build_index
        mock_graphrag_service.return_value = mock_service_instance

        # Execute task
        result = build_graph_index(
            [{"document_id": "test", "content": "test"}], "test_collection"
        )

        # Should complete successfully despite async operations
        assert result["status"] == "completed"
        assert "workspace_path" in result
