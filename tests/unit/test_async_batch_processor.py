"""Tests for GraphRAG async batch processor."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from fileintel.rag.graph_rag.services.batch_processor import (
    GraphRAGBatchProcessor,
    BatchResult,
    AsyncProcessingSettings,
)
from fileintel.core.config import Settings
from fileintel.llm_integration.openai_provider import OpenAIProvider


@dataclass
class MockSettings:
    """Mock settings for testing."""

    class MockRAGSettings:
        class MockGraphRAGSettings:
            async_processing = AsyncProcessingSettings(
                enabled=True,
                batch_size=4,
                max_concurrent_requests=8,
                batch_timeout=30,
                fallback_to_sequential=True,
            )

        graph_rag = MockGraphRAGSettings()

    rag = MockRAGSettings()


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    provider = Mock(spec=OpenAIProvider)
    provider.generate_response = AsyncMock()
    return provider


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return MockSettings()


@pytest.fixture
def batch_processor(mock_llm_provider, mock_settings):
    """Create batch processor for testing."""
    return GraphRAGBatchProcessor(mock_llm_provider, mock_settings)


@pytest.mark.asyncio
async def test_extract_entities_batch_success(batch_processor, mock_llm_provider):
    """Test successful batch processing."""
    # Mock LLM responses
    mock_llm_provider.generate_response.return_value = {
        "content": '{"entities": ["entity1", "entity2"], "relationships": []}',
        "processing_time": 1.0,
    }

    text_chunks = ["Text chunk 1", "Text chunk 2", "Text chunk 3"]

    result = await batch_processor.extract_entities_batch(text_chunks)

    assert result.success is True
    assert len(result.results) == 3
    assert len(result.failed_chunks) == 0
    assert result.processing_time > 0
    assert mock_llm_provider.generate_response.call_count == 3


@pytest.mark.asyncio
async def test_extract_entities_batch_with_failures(batch_processor, mock_llm_provider):
    """Test batch processing with some failures."""

    # Mock mixed responses (some succeed, some fail)
    async def mock_response(prompt, **kwargs):
        if "chunk 2" in prompt:
            raise Exception("LLM error")
        return {
            "content": '{"entities": ["entity1"], "relationships": []}',
            "processing_time": 1.0,
        }

    mock_llm_provider.generate_response.side_effect = mock_response

    text_chunks = ["Text chunk 1", "Text chunk 2", "Text chunk 3"]

    result = await batch_processor.extract_entities_batch(text_chunks)

    assert result.success is False  # Not all chunks succeeded
    assert len(result.results) == 2  # Two successful
    assert len(result.failed_chunks) > 0  # Some failed


@pytest.mark.asyncio
async def test_extract_entities_batch_timeout(batch_processor, mock_llm_provider):
    """Test batch processing timeout handling."""

    # Mock slow LLM response
    async def slow_response(prompt, **kwargs):
        await asyncio.sleep(35)  # Longer than batch_timeout (30s)
        return {"content": "response", "processing_time": 35.0}

    mock_llm_provider.generate_response.side_effect = slow_response

    text_chunks = ["Text chunk 1"]

    result = await batch_processor.extract_entities_batch(text_chunks, batch_size=1)

    assert result.success is False
    assert "timeout" in result.error_message.lower() or result.failed_chunks == [0]


@pytest.mark.asyncio
async def test_extract_entities_batch_disabled(batch_processor, mock_llm_provider):
    """Test behavior when async processing is disabled."""
    # Disable async processing
    batch_processor.async_config.enabled = False

    mock_llm_provider.generate_response.return_value = {
        "content": '{"entities": ["entity1"], "relationships": []}',
        "processing_time": 1.0,
    }

    text_chunks = ["Text chunk 1", "Text chunk 2"]

    result = await batch_processor.extract_entities_batch(text_chunks)

    assert result.success is True
    assert len(result.results) == 2
    # Should process sequentially
    assert mock_llm_provider.generate_response.call_count == 2


@pytest.mark.asyncio
async def test_extract_entities_batch_fallback_to_sequential(
    batch_processor, mock_llm_provider
):
    """Test fallback to sequential processing on batch failure."""
    call_count = 0

    async def mock_response(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:  # First batch attempt fails
            raise Exception("Batch processing error")
        # Sequential fallback succeeds
        return {
            "content": '{"entities": ["entity1"], "relationships": []}',
            "processing_time": 1.0,
        }

    mock_llm_provider.generate_response.side_effect = mock_response

    text_chunks = ["Text chunk 1", "Text chunk 2"]

    result = await batch_processor.extract_entities_batch(text_chunks)

    # Should succeed via fallback
    assert result.success is True
    assert len(result.results) == 2


@pytest.mark.asyncio
async def test_batch_metrics_calculation(batch_processor):
    """Test batch metrics calculation."""
    result = BatchResult(
        success=True,
        results=["result1", "result2", "result3"],
        failed_chunks=[],
        processing_time=5.0,
    )

    metrics = batch_processor.get_batch_metrics(result, chunk_count=3)

    assert metrics["total_chunks"] == 3
    assert metrics["successful_chunks"] == 3
    assert metrics["failed_chunks"] == 0
    assert metrics["processing_time"] == 5.0
    assert metrics["chunks_per_second"] == 0.6  # 3 chunks / 5 seconds
    assert metrics["success_rate"] == 1.0
    assert metrics["async_enabled"] is True


@pytest.mark.asyncio
async def test_custom_batch_size(batch_processor, mock_llm_provider):
    """Test using custom batch size."""
    mock_llm_provider.generate_response.return_value = {
        "content": '{"entities": ["entity1"], "relationships": []}',
        "processing_time": 1.0,
    }

    text_chunks = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

    # Use batch size of 2
    result = await batch_processor.extract_entities_batch(text_chunks, batch_size=2)

    assert result.success is True
    assert len(result.results) == 5
    # Should process in 3 batches: 2, 2, 1
    assert mock_llm_provider.generate_response.call_count == 5


def test_async_processing_settings_validation():
    """Test validation of async processing settings."""
    # Valid settings
    settings = AsyncProcessingSettings(
        enabled=True,
        batch_size=4,
        max_concurrent_requests=8,
        batch_timeout=30,
        fallback_to_sequential=True,
    )
    assert settings.batch_size == 4
    assert settings.max_concurrent_requests == 8

    # Test constraints
    with pytest.raises(ValueError):
        AsyncProcessingSettings(batch_size=0)  # Below minimum

    with pytest.raises(ValueError):
        AsyncProcessingSettings(batch_size=10)  # Above maximum

    with pytest.raises(ValueError):
        AsyncProcessingSettings(batch_timeout=5)  # Below minimum

    with pytest.raises(ValueError):
        AsyncProcessingSettings(batch_timeout=200)  # Above maximum


@pytest.mark.asyncio
async def test_sequential_processing_method(batch_processor, mock_llm_provider):
    """Test the sequential processing fallback method."""
    mock_llm_provider.generate_response.return_value = {
        "content": '{"entities": ["entity1"], "relationships": []}',
        "processing_time": 1.0,
    }

    text_chunks = ["Text chunk 1", "Text chunk 2"]

    # Call sequential method directly
    result = await batch_processor._process_sequential(text_chunks)

    assert result.success is True
    assert len(result.results) == 2
    assert len(result.failed_chunks) == 0
    assert result.processing_time > 0


@pytest.mark.asyncio
async def test_extract_entities_single_method(batch_processor, mock_llm_provider):
    """Test single entity extraction method."""
    expected_response = {
        "content": '{"entities": ["entity1", "entity2"], "relationships": []}',
        "processing_time": 1.5,
    }
    mock_llm_provider.generate_response.return_value = expected_response

    chunk_text = "This is a test chunk with entities."

    result = await batch_processor._extract_entities_single(chunk_text)

    assert result["text"] == chunk_text
    assert result["entities"] == expected_response["content"]
    assert result["processing_time"] == expected_response["processing_time"]

    # Verify the prompt was formatted correctly
    mock_llm_provider.generate_response.assert_called_once()
    call_args = mock_llm_provider.generate_response.call_args
    assert chunk_text in call_args[1]["prompt"]
    assert "Extract entities and relationships" in call_args[1]["prompt"]
