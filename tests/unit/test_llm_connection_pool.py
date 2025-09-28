"""Tests for LLM Connection Pool."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import httpx

from fileintel.llm_integration.connection_pool import (
    LLMConnectionPool,
    PooledConnection,
    ConnectionState,
    ConnectionMetrics,
)
from fileintel.core.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""

    class MockAsyncProcessing:
        max_concurrent_requests = 4
        batch_timeout = 30

    class MockGraphRAG:
        async_processing = MockAsyncProcessing()

    class MockRAG:
        graph_rag = MockGraphRAG()

    class MockOpenAI:
        base_url = "http://localhost:9003/v1"
        api_key = "test_key"

    class MockLLM:
        openai = MockOpenAI()

    class MockSettings:
        rag = MockRAG()
        llm = MockLLM()

    return MockSettings()


@pytest.fixture
def connection_pool(mock_settings):
    """Create connection pool for testing."""
    return LLMConnectionPool(mock_settings)


@pytest.mark.asyncio
async def test_connection_pool_initialization(connection_pool):
    """Test connection pool initialization."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock successful health check
        mock_client.get.return_value.status_code = 200

        await connection_pool.initialize()

        assert connection_pool.is_initialized is True
        assert len(connection_pool.connections) == 4  # max_concurrent_requests
        assert connection_pool.health_check_task is not None
        assert not connection_pool.health_check_task.done()


@pytest.mark.asyncio
async def test_connection_creation(connection_pool):
    """Test individual connection creation."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock successful health check
        mock_client.get.return_value.status_code = 200

        connection = await connection_pool._create_connection("test_conn")

        assert connection is not None
        assert connection.connection_id == "test_conn"
        assert connection.state == ConnectionState.HEALTHY
        assert isinstance(connection.metrics, ConnectionMetrics)


@pytest.mark.asyncio
async def test_connection_health_check(connection_pool):
    """Test connection health checking."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Test successful health check
        mock_client.get.return_value.status_code = 200
        is_healthy = await connection_pool._test_connection_health(mock_client)
        assert is_healthy is True

        # Test failed health check (fallback to completion endpoint)
        mock_client.get.side_effect = Exception("Health endpoint not found")
        mock_client.post.return_value.status_code = 200
        is_healthy = await connection_pool._test_connection_health(mock_client)
        assert is_healthy is True

        # Test completely failed health check
        mock_client.post.side_effect = Exception("Connection failed")
        is_healthy = await connection_pool._test_connection_health(mock_client)
        assert is_healthy is False


@pytest.mark.asyncio
async def test_get_and_return_connection(connection_pool):
    """Test getting and returning connections."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        await connection_pool.initialize()

        # Get a connection
        connection = await connection_pool.get_connection()
        assert connection is not None
        assert isinstance(connection, PooledConnection)

        # Queue should have one less connection
        initial_queue_size = connection_pool.connection_queue.qsize()

        # Return the connection
        await connection_pool.return_connection(connection)

        # Queue should be back to original size (for healthy connections)
        if connection.state == ConnectionState.HEALTHY:
            assert connection_pool.connection_queue.qsize() == initial_queue_size + 1


@pytest.mark.asyncio
async def test_execute_request_success(connection_pool):
    """Test successful request execution."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        await connection_pool.initialize()

        # Mock request function
        async def mock_request_func(client, data):
            return {"result": "success", "data": data}

        result = await connection_pool.execute_request(mock_request_func, "test_data")

        assert result["result"] == "success"
        assert result["data"] == "test_data"


@pytest.mark.asyncio
async def test_execute_request_failure(connection_pool):
    """Test request execution with failure."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        await connection_pool.initialize()

        # Mock failing request function
        async def failing_request_func(client, data):
            raise Exception("Request failed")

        with pytest.raises(Exception, match="Request failed"):
            await connection_pool.execute_request(failing_request_func, "test_data")


@pytest.mark.asyncio
async def test_connection_timeout(connection_pool):
    """Test connection timeout handling."""
    # Mock empty queue to trigger timeout
    connection_pool.connection_queue = asyncio.Queue()

    with pytest.raises(Exception):
        # Should timeout since queue is empty
        connection = await asyncio.wait_for(
            connection_pool.get_connection(), timeout=1.0
        )


@pytest.mark.asyncio
async def test_unhealthy_connection_replacement(connection_pool):
    """Test replacement of unhealthy connections."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # First call: healthy, second call: unhealthy, third call: healthy (replacement)
        health_responses = [200, 200, 500, 200]  # Fourth for replacement
        mock_client.get.side_effect = [
            Mock(status_code=code) for code in health_responses
        ]

        await connection_pool.initialize()

        # Get a connection and mark it as unhealthy
        connection = await connection_pool.get_connection()
        connection.state = ConnectionState.UNHEALTHY

        initial_conn_id = connection.connection_id
        initial_queue_size = connection_pool.connection_queue.qsize()

        # Return the unhealthy connection
        await connection_pool.return_connection(connection)

        # Should have created a replacement
        # Queue size should be maintained
        assert connection_pool.connection_queue.qsize() >= initial_queue_size


@pytest.mark.asyncio
async def test_pool_metrics(connection_pool):
    """Test pool metrics collection."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        await connection_pool.initialize()

        metrics = await connection_pool.get_pool_metrics()

        assert "total_connections" in metrics
        assert "healthy_connections" in metrics
        assert "unhealthy_connections" in metrics
        assert "queue_size" in metrics
        assert "success_rate" in metrics
        assert "pool_initialized" in metrics
        assert metrics["pool_initialized"] is True


@pytest.mark.asyncio
async def test_connection_metrics_tracking(connection_pool):
    """Test that connection metrics are properly tracked."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        await connection_pool.initialize()

        # Get a connection
        connection = await connection_pool.get_connection()

        # Simulate successful request
        async def mock_successful_request(client):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"status": "success"}

        result = await connection_pool.execute_request(mock_successful_request)

        # Check that metrics were updated
        assert connection.metrics.total_requests > 0
        assert connection.metrics.successful_requests > 0
        assert connection.metrics.average_response_time > 0
        assert connection.metrics.last_request_time is not None


@pytest.mark.asyncio
async def test_health_check_loop():
    """Test the periodic health check loop."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Create a pool with short health check interval for testing
        pool = LLMConnectionPool()
        pool.health_check_interval = 0.1  # 100ms for testing

        # Mock health check responses
        mock_client.get.return_value.status_code = 200

        await pool.initialize()

        # Wait for at least one health check cycle
        await asyncio.sleep(0.2)

        # Verify health check was called
        assert mock_client.get.call_count > 0

        await pool.close()


@pytest.mark.asyncio
async def test_pool_cleanup(connection_pool):
    """Test proper cleanup of connection pool."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        await connection_pool.initialize()

        initial_connections = len(connection_pool.connections)
        assert initial_connections > 0

        await connection_pool.close()

        assert connection_pool.is_initialized is False
        assert connection_pool.health_check_task.cancelled()
        # Verify all clients were closed
        assert mock_client.aclose.call_count == initial_connections


@pytest.mark.asyncio
async def test_context_manager(mock_settings):
    """Test connection pool as async context manager."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        async with LLMConnectionPool(mock_settings) as pool:
            assert pool.is_initialized is True
            assert len(pool.connections) > 0

        # Should be automatically closed
        assert pool.is_initialized is False


def test_connection_metrics_initialization():
    """Test ConnectionMetrics initialization."""
    metrics = ConnectionMetrics()

    assert metrics.total_requests == 0
    assert metrics.successful_requests == 0
    assert metrics.failed_requests == 0
    assert metrics.average_response_time == 0.0
    assert metrics.last_request_time is None
    assert metrics.last_health_check is None


def test_pooled_connection_initialization():
    """Test PooledConnection initialization."""
    mock_client = Mock()
    current_time = time.time()

    connection = PooledConnection(
        connection_id="test_id",
        client=mock_client,
        state=ConnectionState.HEALTHY,
        metrics=ConnectionMetrics(),
        created_at=current_time,
        last_used=current_time,
    )

    assert connection.connection_id == "test_id"
    assert connection.client == mock_client
    assert connection.state == ConnectionState.HEALTHY
    assert isinstance(connection.metrics, ConnectionMetrics)
    assert connection.created_at == current_time
    assert connection.last_used == current_time
