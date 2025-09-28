"""Integration tests for GraphRAG batch processing workflow."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from fileintel.rag.graph_rag.workflows.graphrag_workflow import GraphRAGWorkflow
from fileintel.storage.models import DocumentChunk, Document, Collection
from fileintel.core.config import Settings


@dataclass
class MockJob:
    """Mock job for testing."""

    id: str
    job_type: str
    data: dict


@dataclass
class MockDocument:
    """Mock document for testing."""

    id: str
    title: str
    created_at = None


@dataclass
class MockChunk:
    """Mock chunk for testing."""

    id: str
    chunk_text: str
    document_id: str
    document: MockDocument


@pytest.fixture
def mock_storage():
    """Mock storage for testing."""
    storage = Mock()
    storage.get_documents_in_collection.return_value = [
        MockDocument(id="doc1", title="Test Document 1"),
        MockDocument(id="doc2", title="Test Document 2"),
    ]
    storage.get_all_chunks_for_document.return_value = [
        MockChunk(
            id="chunk1",
            chunk_text="This is test content 1.",
            document_id="doc1",
            document=MockDocument(id="doc1", title="Test Doc 1"),
        ),
        MockChunk(
            id="chunk2",
            chunk_text="This is test content 2.",
            document_id="doc1",
            document=MockDocument(id="doc1", title="Test Doc 1"),
        ),
    ]
    storage.save_result.return_value = None
    storage.save_graphrag_index_info.return_value = None
    return storage


@pytest.fixture
def mock_graphrag_service():
    """Mock GraphRAG service for testing."""
    service = Mock()
    service.build_index = AsyncMock(return_value="/path/to/index")
    return service


@pytest.fixture
def mock_batch_processor():
    """Mock batch processor for testing."""
    processor = Mock()
    processor.extract_entities_batch = AsyncMock()
    processor.get_batch_metrics = Mock()
    return processor


@pytest.fixture
def mock_adapter():
    """Mock logger adapter for testing."""
    adapter = Mock()
    adapter.info = Mock()
    adapter.warning = Mock()
    adapter.error = Mock()
    return adapter


@pytest.mark.asyncio
async def test_graphrag_workflow_with_async_disabled(mock_storage):
    """Test GraphRAG workflow when async processing is disabled."""
    with patch(
        "fileintel.rag.graph_rag.workflows.graphrag_workflow.settings"
    ) as mock_settings:
        mock_settings.rag.graph_rag.async_processing.enabled = False

        workflow = GraphRAGWorkflow(mock_storage)

        assert workflow.batch_processor is None
        assert workflow.graphrag_service is not None


@pytest.mark.asyncio
async def test_graphrag_workflow_with_async_enabled(mock_storage):
    """Test GraphRAG workflow when async processing is enabled."""
    with patch(
        "fileintel.rag.graph_rag.workflows.graphrag_workflow.settings"
    ) as mock_settings:
        mock_settings.rag.graph_rag.async_processing.enabled = True

        with patch(
            "fileintel.rag.graph_rag.workflows.graphrag_workflow.OpenAIProvider"
        ) as mock_provider_class:
            mock_provider = Mock()
            mock_provider_class.return_value = mock_provider

            with patch(
                "fileintel.rag.graph_rag.workflows.graphrag_workflow.GraphRAGBatchProcessor"
            ) as mock_processor_class:
                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor

                workflow = GraphRAGWorkflow(mock_storage)

                assert workflow.batch_processor is not None
                mock_provider_class.assert_called_once()
                mock_processor_class.assert_called_once_with(
                    mock_provider, mock_settings
                )


@pytest.mark.asyncio
async def test_process_chunks_with_batching_success(mock_storage, mock_adapter):
    """Test successful batch processing of chunks."""
    with patch(
        "fileintel.rag.graph_rag.workflows.graphrag_workflow.settings"
    ) as mock_settings:
        mock_settings.rag.graph_rag.async_processing.enabled = True

        with patch(
            "fileintel.rag.graph_rag.workflows.graphrag_workflow.OpenAIProvider"
        ):
            with patch(
                "fileintel.rag.graph_rag.workflows.graphrag_workflow.GraphRAGBatchProcessor"
            ) as mock_processor_class:
                # Mock successful batch result
                from fileintel.rag.graph_rag.services.batch_processor import BatchResult

                batch_result = BatchResult(
                    success=True,
                    results=["result1", "result2"],
                    failed_chunks=[],
                    processing_time=2.5,
                )

                mock_processor = Mock()
                mock_processor.extract_entities_batch = AsyncMock(
                    return_value=batch_result
                )
                mock_processor.get_batch_metrics.return_value = {
                    "total_chunks": 2,
                    "successful_chunks": 2,
                    "processing_time": 2.5,
                    "chunks_per_second": 0.8,
                }
                mock_processor_class.return_value = mock_processor

                workflow = GraphRAGWorkflow(mock_storage)

                # Test chunks
                chunks = [
                    MockChunk(
                        id="1",
                        chunk_text="Text 1",
                        document_id="doc1",
                        document=MockDocument(id="doc1", title="Test"),
                    ),
                    MockChunk(
                        id="2",
                        chunk_text="Text 2",
                        document_id="doc1",
                        document=MockDocument(id="doc1", title="Test"),
                    ),
                ]

                await workflow._process_chunks_with_batching(chunks, mock_adapter)

                # Verify batch processor was called
                mock_processor.extract_entities_batch.assert_called_once()
                mock_processor.get_batch_metrics.assert_called_once()

                # Verify adapter was called with success message
                mock_adapter.info.assert_called()


@pytest.mark.asyncio
async def test_process_chunks_with_batching_failure(mock_storage, mock_adapter):
    """Test batch processing with failure."""
    with patch(
        "fileintel.rag.graph_rag.workflows.graphrag_workflow.settings"
    ) as mock_settings:
        mock_settings.rag.graph_rag.async_processing.enabled = True

        with patch(
            "fileintel.rag.graph_rag.workflows.graphrag_workflow.OpenAIProvider"
        ):
            with patch(
                "fileintel.rag.graph_rag.workflows.graphrag_workflow.GraphRAGBatchProcessor"
            ) as mock_processor_class:
                # Mock failed batch result
                from fileintel.rag.graph_rag.services.batch_processor import BatchResult

                batch_result = BatchResult(
                    success=False,
                    results=[],
                    failed_chunks=[0, 1],
                    processing_time=1.0,
                    error_message="Processing failed",
                )

                mock_processor = Mock()
                mock_processor.extract_entities_batch = AsyncMock(
                    return_value=batch_result
                )
                mock_processor.get_batch_metrics.return_value = {
                    "total_chunks": 2,
                    "successful_chunks": 0,
                    "failed_chunks": 2,
                }
                mock_processor_class.return_value = mock_processor

                workflow = GraphRAGWorkflow(mock_storage)

                chunks = [
                    MockChunk(
                        id="1",
                        chunk_text="Text 1",
                        document_id="doc1",
                        document=MockDocument(id="doc1", title="Test"),
                    ),
                    MockChunk(
                        id="2",
                        chunk_text="Text 2",
                        document_id="doc1",
                        document=MockDocument(id="doc1", title="Test"),
                    ),
                ]

                await workflow._process_chunks_with_batching(chunks, mock_adapter)

                # Verify error was logged
                mock_adapter.error.assert_called()


@pytest.mark.asyncio
async def test_process_chunks_with_batching_no_processor(mock_storage, mock_adapter):
    """Test batch processing when no processor is available."""
    with patch(
        "fileintel.rag.graph_rag.workflows.graphrag_workflow.settings"
    ) as mock_settings:
        mock_settings.rag.graph_rag.async_processing.enabled = False

        workflow = GraphRAGWorkflow(mock_storage)

        chunks = [
            MockChunk(
                id="1",
                chunk_text="Text 1",
                document_id="doc1",
                document=MockDocument(id="doc1", title="Test"),
            )
        ]

        await workflow._process_chunks_with_batching(chunks, mock_adapter)

        # Should warn that batch processor is not available
        mock_adapter.warning.assert_called_with(
            "Batch processor not available, skipping batch processing"
        )


@pytest.mark.asyncio
async def test_process_chunks_with_batching_exception(mock_storage, mock_adapter):
    """Test batch processing with exception handling."""
    with patch(
        "fileintel.rag.graph_rag.workflows.graphrag_workflow.settings"
    ) as mock_settings:
        mock_settings.rag.graph_rag.async_processing.enabled = True

        with patch(
            "fileintel.rag.graph_rag.workflows.graphrag_workflow.OpenAIProvider"
        ):
            with patch(
                "fileintel.rag.graph_rag.workflows.graphrag_workflow.GraphRAGBatchProcessor"
            ) as mock_processor_class:
                mock_processor = Mock()
                mock_processor.extract_entities_batch = AsyncMock(
                    side_effect=Exception("Batch processing error")
                )
                mock_processor_class.return_value = mock_processor

                workflow = GraphRAGWorkflow(mock_storage)

                chunks = [
                    MockChunk(
                        id="1",
                        chunk_text="Text 1",
                        document_id="doc1",
                        document=MockDocument(id="doc1", title="Test"),
                    )
                ]

                # Should not raise exception, just log error
                await workflow._process_chunks_with_batching(chunks, mock_adapter)

                # Verify error was logged
                mock_adapter.error.assert_called()


@pytest.mark.asyncio
async def test_graphrag_index_collection_with_batching(mock_storage, mock_adapter):
    """Test complete GraphRAG indexing job with batch processing."""
    with patch(
        "fileintel.rag.graph_rag.workflows.graphrag_workflow.settings"
    ) as mock_settings:
        mock_settings.rag.graph_rag.async_processing.enabled = True

        with patch(
            "fileintel.rag.graph_rag.workflows.graphrag_workflow.OpenAIProvider"
        ):
            with patch(
                "fileintel.rag.graph_rag.workflows.graphrag_workflow.GraphRAGBatchProcessor"
            ) as mock_processor_class:
                # Mock successful batch processing
                from fileintel.rag.graph_rag.services.batch_processor import BatchResult

                batch_result = BatchResult(
                    success=True,
                    results=["result1", "result2"],
                    failed_chunks=[],
                    processing_time=3.0,
                )

                mock_processor = Mock()
                mock_processor.extract_entities_batch = AsyncMock(
                    return_value=batch_result
                )
                mock_processor.get_batch_metrics.return_value = {"speedup": 2.5}
                mock_processor_class.return_value = mock_processor

                workflow = GraphRAGWorkflow(mock_storage)

                # Mock the GraphRAG service
                workflow.graphrag_service.build_index = AsyncMock(
                    return_value="/test/index/path"
                )

                # Create test job
                job = MockJob(
                    id="job123",
                    job_type="graphrag_index_collection",
                    data={"collection_id": "collection123"},
                )

                await workflow.process_graphrag_index_collection_job(job, mock_adapter)

                # Verify batch processing was called
                mock_processor.extract_entities_batch.assert_called_once()

                # Verify GraphRAG indexing was called
                workflow.graphrag_service.build_index.assert_called_once()

                # Verify result was saved
                mock_storage.save_result.assert_called_once()
                result_call = mock_storage.save_result.call_args[0]
                assert result_call[0] == "job123"  # job_id
                result_data = result_call[1]
                assert result_data["status"] == "success"
                assert result_data["index_path"] == "/test/index/path"


@pytest.mark.asyncio
async def test_graphrag_index_collection_empty_collection(mock_storage, mock_adapter):
    """Test GraphRAG indexing with empty collection."""
    mock_storage.get_documents_in_collection.return_value = []

    with patch(
        "fileintel.rag.graph_rag.workflows.graphrag_workflow.settings"
    ) as mock_settings:
        mock_settings.rag.graph_rag.async_processing.enabled = True

        workflow = GraphRAGWorkflow(mock_storage)

        job = MockJob(
            id="job456",
            job_type="graphrag_index_collection",
            data={"collection_id": "empty_collection"},
        )

        await workflow.process_graphrag_index_collection_job(job, mock_adapter)

        # Should warn about empty collection
        mock_adapter.warning.assert_called()

        # Should save error result
        mock_storage.save_result.assert_called_once()
        result_call = mock_storage.save_result.call_args[0]
        result_data = result_call[1]
        assert result_data["status"] == "error"
        assert "empty" in result_data["error"].lower()


@pytest.mark.asyncio
async def test_batch_processing_metrics_integration():
    """Test that batch processing metrics are properly collected."""
    from fileintel.rag.graph_rag.services.batch_processor import (
        GraphRAGBatchProcessor,
        BatchResult,
    )

    # Mock LLM provider
    mock_provider = Mock()
    mock_provider.generate_response = AsyncMock(
        return_value={
            "content": '{"entities": ["test"], "relationships": []}',
            "processing_time": 1.0,
        }
    )

    # Mock settings
    class MockAsyncSettings:
        enabled = True
        batch_size = 2
        max_concurrent_requests = 4
        batch_timeout = 30
        fallback_to_sequential = True

    class MockGraphRAGSettings:
        async_processing = MockAsyncSettings()

    class MockRAGSettings:
        graph_rag = MockGraphRAGSettings()

    class MockSettings:
        rag = MockRAGSettings()

    settings = MockSettings()
    processor = GraphRAGBatchProcessor(mock_provider, settings)

    text_chunks = ["Text 1", "Text 2", "Text 3"]

    result = await processor.extract_entities_batch(text_chunks)

    assert isinstance(result, BatchResult)

    # Test metrics calculation
    metrics = processor.get_batch_metrics(result, len(text_chunks))

    assert "total_chunks" in metrics
    assert "successful_chunks" in metrics
    assert "processing_time" in metrics
    assert "chunks_per_second" in metrics
    assert "success_rate" in metrics
    assert "async_enabled" in metrics

    assert metrics["total_chunks"] == 3
    assert metrics["async_enabled"] is True
