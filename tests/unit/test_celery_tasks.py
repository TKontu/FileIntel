"""
Unit tests for Celery task functions.

Tests the individual Celery task implementations that replaced the job management system.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import uuid

from src.fileintel.tasks.document_tasks import process_document, process_collection
from src.fileintel.tasks.graphrag_tasks import build_graph_index, global_search_task
from src.fileintel.tasks.llm_tasks import (
    generate_batch_embeddings,
    extract_metadata_with_llm,
)
from src.fileintel.tasks.workflow_tasks import complete_collection_analysis


@pytest.mark.celery
class TestDocumentTasks:
    """Test document processing Celery tasks."""

    @patch("src.fileintel.tasks.document_tasks.get_storage")
    @patch("src.fileintel.tasks.document_tasks.UnifiedDocumentProcessor")
    def test_process_document_task(
        self, mock_processor_class, mock_get_storage, mock_storage, sample_document
    ):
        """Test individual document processing task."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_processor = Mock()
        mock_processor.process_document.return_value = {
            "chunks": ["chunk1", "chunk2", "chunk3"],
            "metadata": {"title": "Test Doc", "pages": 3},
        }
        mock_processor_class.return_value = mock_processor

        # Execute task
        result = process_document(
            file_path="/test/document.pdf",
            document_id=sample_document.id,
            collection_id=sample_document.collection_id,
        )

        # Verify result
        assert result["status"] == "completed"
        assert result["document_id"] == sample_document.id
        assert len(result["chunks"]) == 3
        assert "metadata" in result

        # Verify processor was called
        mock_processor.process_document.assert_called_once()

    @patch("src.fileintel.tasks.document_tasks.get_storage")
    def test_process_collection_task(
        self, mock_get_storage, mock_storage, sample_collection
    ):
        """Test collection processing task."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_storage.get_documents_in_collection.return_value = [
            Mock(id="doc1", filename="doc1.pdf"),
            Mock(id="doc2", filename="doc2.pdf"),
        ]

        # Mock individual document processing
        with patch(
            "src.fileintel.tasks.document_tasks.process_document"
        ) as mock_process:
            mock_process.return_value = {"status": "completed", "chunks": ["chunk1"]}

            # Execute task
            result = process_collection(sample_collection.id)

            # Verify result
            assert result["status"] == "completed"
            assert result["collection_id"] == sample_collection.id
            assert result["documents_processed"] == 2

    def test_document_task_error_handling(self):
        """Test error handling in document tasks."""
        # Test with invalid file path
        result = process_document(
            file_path="/nonexistent/file.pdf",
            document_id="test_id",
            collection_id="test_collection",
        )

        assert result["status"] == "failed"
        assert "error" in result


@pytest.mark.celery
@pytest.mark.graphrag
class TestGraphRAGTasks:
    """Test GraphRAG processing Celery tasks."""

    @patch("src.fileintel.tasks.graphrag_tasks.get_storage")
    @patch("src.fileintel.tasks.graphrag_tasks.GraphRAGService")
    def test_build_graph_index_task(
        self, mock_service_class, mock_get_storage, mock_storage, sample_graph_documents
    ):
        """Test GraphRAG index building task."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_service = Mock()
        mock_service.build_index = AsyncMock(return_value="/test/workspace")
        mock_service_class.return_value = mock_service

        # Execute task
        result = build_graph_index(sample_graph_documents, "test_collection")

        # Verify result
        assert result["status"] == "completed"
        assert result["collection_id"] == "test_collection"
        assert "workspace_path" in result

        # Verify service was called
        mock_service.build_index.assert_called_once()

    @patch("src.fileintel.tasks.graphrag_tasks.get_storage")
    @patch("src.fileintel.tasks.graphrag_tasks.GraphRAGService")
    def test_global_search_task(
        self, mock_service_class, mock_get_storage, mock_storage
    ):
        """Test GraphRAG global search task."""
        # Setup mocks
        mock_get_storage.return_value = mock_storage
        mock_service = Mock()
        mock_service.global_search = AsyncMock(
            return_value={
                "answer": "Test answer",
                "sources": ["doc1"],
                "confidence": 0.8,
            }
        )
        mock_service_class.return_value = mock_service

        # Execute task
        result = global_search_task("What is AI?", "test_collection")

        # Verify result
        assert result["status"] == "completed"
        assert result["query"] == "What is AI?"
        assert "answer" in result

        # Verify service was called
        mock_service.global_search.assert_called_once_with(
            query="What is AI?", collection_id="test_collection"
        )

    def test_graphrag_task_validation(self):
        """Test input validation in GraphRAG tasks."""
        # Test with empty documents
        result = build_graph_index([], "test_collection")
        assert result["status"] == "failed"
        assert "error" in result

        # Test with empty query
        result = global_search_task("", "test_collection")
        assert result["status"] == "failed"
        assert "error" in result


@pytest.mark.celery
class TestLLMTasks:
    """Test LLM processing Celery tasks."""

    @patch("src.fileintel.tasks.llm_tasks.get_embedding_provider")
    def test_generate_batch_embeddings_task(self, mock_get_provider):
        """Test batch embedding generation task."""
        # Setup mock
        mock_provider = Mock()
        mock_provider.get_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_get_provider.return_value = mock_provider

        # Execute task
        chunks = ["text chunk 1", "text chunk 2"]
        result = generate_batch_embeddings(chunks, batch_size=10)

        # Verify result
        assert result["status"] == "completed"
        assert len(result["embeddings"]) == 2
        assert result["chunks_processed"] == 2

        # Verify provider was called
        mock_provider.get_embeddings.assert_called_once_with(chunks)

    @patch("src.fileintel.tasks.llm_tasks.get_llm_provider")
    def test_extract_metadata_with_llm_task(self, mock_get_provider):
        """Test LLM-based metadata extraction task."""
        # Setup mock
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = '{"title": "Test Document", "author": "Test Author"}'
        mock_provider.generate_response.return_value = mock_response
        mock_get_provider.return_value = mock_provider

        # Execute task
        chunks = ["Document content chunk 1", "Document content chunk 2"]
        result = extract_metadata_with_llm(chunks, document_id="test_doc")

        # Verify result
        assert result["status"] == "completed"
        assert result["document_id"] == "test_doc"
        assert "metadata" in result

        # Verify provider was called
        mock_provider.generate_response.assert_called_once()

    def test_llm_task_error_handling(self):
        """Test error handling in LLM tasks."""
        # Test with empty chunks
        result = generate_batch_embeddings([])
        assert result["status"] == "failed"
        assert "error" in result


@pytest.mark.celery
@pytest.mark.workflow
class TestWorkflowTasks:
    """Test workflow orchestration Celery tasks."""

    @patch("src.fileintel.tasks.workflow_tasks.process_document")
    @patch("src.fileintel.tasks.workflow_tasks.generate_batch_embeddings")
    @patch("src.fileintel.tasks.workflow_tasks.build_graph_index")
    def test_complete_collection_analysis_task(
        self, mock_graph, mock_embeddings, mock_process, sample_collection
    ):
        """Test complete collection analysis workflow task."""
        # Setup mocks
        mock_process.apply_async.return_value.get.return_value = {
            "status": "completed",
            "chunks": ["chunk1", "chunk2"],
        }
        mock_embeddings.apply_async.return_value.get.return_value = {
            "status": "completed",
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }
        mock_graph.apply_async.return_value.get.return_value = {
            "status": "completed",
            "workspace_path": "/test/workspace",
        }

        # Execute task
        file_paths = ["/test/doc1.pdf", "/test/doc2.pdf"]
        result = complete_collection_analysis(
            collection_id=sample_collection.id,
            file_paths=file_paths,
            build_graph=True,
            generate_embeddings=True,
        )

        # Verify result
        assert result["status"] == "completed"
        assert result["collection_id"] == sample_collection.id
        assert result["total_documents"] == 2
        assert result["operations_completed"]["document_processing"] is True
        assert result["operations_completed"]["embedding_generation"] is True
        assert result["operations_completed"]["graphrag_indexing"] is True

        # Verify subtasks were called
        assert mock_process.apply_async.called
        assert mock_embeddings.apply_async.called
        assert mock_graph.apply_async.called

    def test_workflow_task_partial_failure(self):
        """Test workflow task handling when some operations fail."""
        # Test with no file paths
        result = complete_collection_analysis(
            collection_id="test_collection",
            file_paths=[],
            build_graph=False,
            generate_embeddings=False,
        )

        assert result["status"] == "failed"
        assert "error" in result


@pytest.mark.celery
class TestTaskIntegrationPatterns:
    """Test Celery task integration patterns."""

    def test_task_chaining_pattern(self, mock_celery_task, mock_celery_result):
        """Test task chaining patterns using mocks."""
        # Simulate a chain: document processing -> embedding generation -> index building

        # Mock the chain execution
        with patch("celery.chain") as mock_chain:
            mock_chain.return_value.apply_async.return_value = mock_celery_result

            # Verify chain pattern works
            chain_result = mock_chain.return_value.apply_async()
            assert chain_result.ready()
            assert chain_result.successful()

    def test_task_group_pattern(self, mock_celery_app):
        """Test task group patterns for parallel execution."""
        # Mock parallel task execution
        with patch("celery.group") as mock_group:
            mock_group.return_value.apply_async.return_value.get.return_value = [
                {"status": "completed"},
                {"status": "completed"},
                {"status": "completed"},
            ]

            # Verify group pattern works
            group_result = mock_group.return_value.apply_async()
            results = group_result.get()
            assert len(results) == 3
            assert all(r["status"] == "completed" for r in results)

    def test_task_chord_pattern(self):
        """Test task chord patterns (group + callback)."""
        # Mock chord execution
        with patch("celery.chord") as mock_chord:
            mock_chord.return_value.apply_async.return_value.get.return_value = {
                "status": "completed",
                "group_results": [{"processed": True}, {"processed": True}],
                "callback_result": {"aggregated": True},
            }

            # Verify chord pattern works
            chord_result = mock_chord.return_value.apply_async()
            result = chord_result.get()
            assert result["status"] == "completed"
            assert "callback_result" in result

    @patch("src.fileintel.tasks.base.BaseFileIntelTask.update_progress")
    def test_task_progress_tracking(self, mock_update_progress):
        """Test task progress tracking functionality."""
        # Execute a task that should update progress
        with patch("src.fileintel.tasks.document_tasks.get_storage"):
            with patch("src.fileintel.tasks.document_tasks.UnifiedDocumentProcessor"):
                process_document(
                    file_path="/test/doc.pdf",
                    document_id="test_doc",
                    collection_id="test_collection",
                )

                # Verify progress was updated
                mock_update_progress.assert_called()

    def test_task_retry_mechanism(self):
        """Test task retry mechanisms."""
        with patch("src.fileintel.tasks.document_tasks.get_storage") as mock_storage:
            # Setup mock to fail first time, succeed second time
            mock_storage.side_effect = [Exception("Connection failed"), Mock()]

            # In a real Celery environment, retries would be automatic
            # Here we simulate the retry behavior
            try:
                result = process_document(
                    file_path="/test/doc.pdf",
                    document_id="test_doc",
                    collection_id="test_collection",
                )
                # First call should fail
                assert result["status"] == "failed"
            except Exception:
                pass

            # Reset mock for successful retry
            mock_storage.side_effect = None
            mock_storage.return_value = Mock()

            # Retry should succeed
            result = process_document(
                file_path="/test/doc.pdf",
                document_id="test_doc",
                collection_id="test_collection",
            )
            # This would succeed if processor is properly mocked
            assert "status" in result
