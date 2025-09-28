"""Advanced unit tests for LLM connection pool functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass

from src.fileintel.llm_integration.connection_pool import (
    ConnectionPool,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreaker,
    PooledConnection,
    ConnectionStats,
)
from src.fileintel.core.config import get_config


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.llm = Mock()
    config.llm.openai = Mock()
    config.llm.openai.api_key = "test-key"
    config.llm.openai.base_url = "https://api.openai.com/v1"
    config.llm.openai.max_connections = 10
    config.llm.openai.max_retries = 3
    config.llm.openai.timeout = 30
    config.llm.openai.rate_limit_rpm = 3000
    config.llm.openai.rate_limit_tpm = 150000
    config.llm.circuit_breaker = Mock()
    config.llm.circuit_breaker.enabled = True
    config.llm.circuit_breaker.failure_threshold = 5
    config.llm.circuit_breaker.recovery_timeout = 60
    config.llm.circuit_breaker.half_open_max_calls = 3
    return config


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = AsyncMock()
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def connection_pool(mock_config):
    """Create a connection pool for testing."""
    with patch(
        "src.fileintel.llm_integration.connection_pool.get_config",
        return_value=mock_config,
    ):
        pool = ConnectionPool()
        yield pool


@pytest.fixture
def circuit_breaker_config():
    """Create circuit breaker configuration for testing."""
    return CircuitBreakerConfig(
        failure_threshold=5, recovery_timeout=60, half_open_max_calls=3
    )


class TestCircuitBreaker:
    """Test cases for circuit breaker functionality."""

    def test_circuit_breaker_initialization(self, circuit_breaker_config):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(circuit_breaker_config)

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
        assert cb.half_open_calls == 0

    def test_circuit_breaker_success_recording(self, circuit_breaker_config):
        """Test recording successful calls."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Record multiple successes
        cb.record_success()
        cb.record_success()

        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_failure_recording(self, circuit_breaker_config):
        """Test recording failed calls."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Record failures below threshold
        for i in range(4):
            cb.record_failure()
            assert cb.state == CircuitBreakerState.CLOSED

        # Record failure that triggers circuit open
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 5

    def test_circuit_breaker_call_allowed_closed_state(self, circuit_breaker_config):
        """Test call allowed in closed state."""
        cb = CircuitBreaker(circuit_breaker_config)
        assert cb.call_allowed() is True

    def test_circuit_breaker_call_blocked_open_state(self, circuit_breaker_config):
        """Test call blocked in open state."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Trigger circuit open
        for _ in range(5):
            cb.record_failure()

        assert cb.call_allowed() is False

    def test_circuit_breaker_half_open_transition(self, circuit_breaker_config):
        """Test transition from open to half-open state."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Trigger circuit open
        for _ in range(5):
            cb.record_failure()

        # Simulate time passage beyond recovery timeout
        cb.last_failure_time = datetime.utcnow() - timedelta(seconds=70)

        # First call should transition to half-open
        assert cb.call_allowed() is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_half_open_success_closes_circuit(
        self, circuit_breaker_config
    ):
        """Test successful call in half-open state closes circuit."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Get to half-open state
        for _ in range(5):
            cb.record_failure()
        cb.last_failure_time = datetime.utcnow() - timedelta(seconds=70)
        cb.call_allowed()  # Transition to half-open

        # Record success in half-open state
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_half_open_failure_reopens_circuit(
        self, circuit_breaker_config
    ):
        """Test failure in half-open state reopens circuit."""
        cb = CircuitBreaker(circuit_breaker_config)

        # Get to half-open state
        for _ in range(5):
            cb.record_failure()
        cb.last_failure_time = datetime.utcnow() - timedelta(seconds=70)
        cb.call_allowed()  # Transition to half-open

        # Record failure in half-open state
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


class TestConnectionPool:
    """Test cases for connection pool functionality."""

    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self, connection_pool):
        """Test connection pool initialization."""
        assert connection_pool.max_connections == 10
        assert connection_pool.active_connections == 0
        assert len(connection_pool.connection_stats) == 0

    @pytest.mark.asyncio
    async def test_get_connection_creates_new_when_available(
        self, connection_pool, mock_openai_client
    ):
        """Test getting connection creates new one when under limit."""
        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            async with connection_pool.get_connection() as conn:
                assert conn is not None
                assert connection_pool.active_connections == 1

        assert connection_pool.active_connections == 0

    @pytest.mark.asyncio
    async def test_connection_pool_respects_max_connections(
        self, connection_pool, mock_openai_client
    ):
        """Test connection pool respects maximum connection limit."""
        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            # Fill pool to capacity
            connections = []
            for i in range(10):
                conn_ctx = connection_pool.get_connection()
                conn = await conn_ctx.__aenter__()
                connections.append((conn_ctx, conn))

            assert connection_pool.active_connections == 10

            # Attempt to get one more connection should timeout
            with pytest.raises(asyncio.TimeoutError):
                async with asyncio.timeout(0.1):
                    async with connection_pool.get_connection():
                        pass

            # Release connections
            for conn_ctx, conn in connections:
                await conn_ctx.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_calls_when_open(
        self, connection_pool, mock_openai_client
    ):
        """Test circuit breaker blocks calls when open."""
        # Force circuit breaker to open state
        connection_pool.circuit_breaker.state = CircuitBreakerState.OPEN

        with pytest.raises(Exception, match="Circuit breaker is open"):
            async with connection_pool.get_connection():
                pass

    @pytest.mark.asyncio
    async def test_connection_stats_tracking(self, connection_pool, mock_openai_client):
        """Test connection statistics tracking."""
        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            async with connection_pool.get_connection() as conn:
                # Simulate some usage
                await asyncio.sleep(0.01)

        # Check stats were recorded
        stats = connection_pool.get_connection_stats()
        assert stats["total_connections_created"] >= 1
        assert stats["active_connections"] == 0

    @pytest.mark.asyncio
    async def test_connection_retry_mechanism(
        self, connection_pool, mock_openai_client
    ):
        """Test connection retry mechanism on failure."""
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("Network error"),
            Exception("API error"),
            {"choices": [{"message": {"content": "Success"}}]},
        ]

        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            async with connection_pool.get_connection() as conn:
                # The connection should eventually succeed after retries
                result = await conn.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                )
                assert result["choices"][0]["message"]["content"] == "Success"

    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self, connection_pool, mock_openai_client):
        """Test rate limiting enforcement."""
        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            # Simulate rapid requests
            start_time = time.time()

            tasks = []
            for _ in range(5):
                task = asyncio.create_task(self._make_llm_call(connection_pool))
                tasks.append(task)

            await asyncio.gather(*tasks)

            elapsed_time = time.time() - start_time
            # Should take some time due to rate limiting
            assert elapsed_time >= 0.001  # At least some delay

    async def _make_llm_call(self, connection_pool):
        """Helper method to make an LLM call."""
        async with connection_pool.get_connection() as conn:
            return await conn.chat.completions.create(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": "test"}]
            )

    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self, connection_pool, mock_openai_client):
        """Test connection pool cleanup."""
        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            # Create some connections
            async with connection_pool.get_connection():
                pass
            async with connection_pool.get_connection():
                pass

        # Call cleanup
        await connection_pool.cleanup()

        # Verify cleanup occurred
        stats = connection_pool.get_connection_stats()
        assert stats["active_connections"] == 0

    def test_connection_stats_comprehensive(self, connection_pool):
        """Test comprehensive connection statistics."""
        stats = connection_pool.get_connection_stats()

        expected_keys = [
            "active_connections",
            "total_connections_created",
            "failed_connections",
            "circuit_breaker_state",
            "circuit_breaker_failure_count",
            "rate_limit_hits",
            "average_response_time",
        ]

        for key in expected_keys:
            assert key in stats

    @pytest.mark.asyncio
    async def test_concurrent_connection_requests(
        self, connection_pool, mock_openai_client
    ):
        """Test handling concurrent connection requests."""
        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            # Create multiple concurrent requests
            tasks = [self._make_llm_call(connection_pool) for _ in range(5)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed (no exceptions)
            for result in results:
                assert not isinstance(result, Exception)

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(
        self, connection_pool, mock_openai_client
    ):
        """Test connection timeout handling."""

        # Mock a slow response
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(2)
            return {"choices": [{"message": {"content": "slow response"}}]}

        mock_openai_client.chat.completions.create.side_effect = slow_response

        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            with pytest.raises(asyncio.TimeoutError):
                async with asyncio.timeout(0.5):
                    async with connection_pool.get_connection() as conn:
                        await conn.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "test"}],
                        )

    @pytest.mark.asyncio
    async def test_connection_health_monitoring(
        self, connection_pool, mock_openai_client
    ):
        """Test connection health monitoring."""
        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            # Simulate some failures
            mock_openai_client.chat.completions.create.side_effect = Exception(
                "Health check failed"
            )

            try:
                async with connection_pool.get_connection() as conn:
                    await conn.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "health check"}],
                    )
            except Exception:
                pass

            # Check that failure was recorded
            stats = connection_pool.get_connection_stats()
            assert stats["failed_connections"] > 0


class TestPooledConnection:
    """Test cases for individual pooled connections."""

    @pytest.mark.asyncio
    async def test_pooled_connection_context_manager(self, mock_openai_client):
        """Test pooled connection as context manager."""
        conn = PooledConnection(mock_openai_client, connection_id="test-1")

        async with conn:
            # Should be able to use the connection
            assert conn.client == mock_openai_client
            assert conn.is_active is True

        # Should be cleaned up after context
        assert conn.is_active is False

    @pytest.mark.asyncio
    async def test_pooled_connection_stats_tracking(self, mock_openai_client):
        """Test pooled connection statistics tracking."""
        conn = PooledConnection(mock_openai_client, connection_id="test-1")

        # Simulate usage
        async with conn:
            await asyncio.sleep(0.01)

        stats = conn.get_stats()
        assert stats.total_requests >= 0
        assert stats.total_response_time >= 0
        assert stats.created_at is not None

    def test_connection_stats_dataclass(self):
        """Test ConnectionStats dataclass."""
        stats = ConnectionStats(
            connection_id="test-1",
            created_at=datetime.utcnow(),
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
            total_response_time=1.5,
            last_used=datetime.utcnow(),
        )

        assert stats.success_rate == 0.8
        assert stats.average_response_time == 0.15


@pytest.mark.integration
class TestConnectionPoolIntegration:
    """Integration tests for connection pool with real scenarios."""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self, connection_pool, mock_openai_client):
        """Test full workflow simulation with multiple operations."""
        mock_responses = [
            {"choices": [{"message": {"content": f"Response {i}"}}]} for i in range(10)
        ]
        mock_openai_client.chat.completions.create.side_effect = mock_responses

        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            # Simulate multiple concurrent operations
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    self._simulate_document_processing(connection_pool, i)
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all operations completed successfully
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == 10

    async def _simulate_document_processing(self, connection_pool, doc_id: int):
        """Simulate processing a document with LLM."""
        async with connection_pool.get_connection() as conn:
            # Simulate analysis request
            response = await conn.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Analyze document {doc_id}"}],
            )
            return response

    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self, connection_pool, mock_openai_client):
        """Test error recovery in realistic scenario."""
        # Simulate intermittent failures followed by recovery
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception(f"Temporary error {call_count}")
            return {"choices": [{"message": {"content": "Recovered"}}]}

        mock_openai_client.chat.completions.create.side_effect = side_effect

        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            # Should eventually succeed after retries
            async with connection_pool.get_connection() as conn:
                result = await conn.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                )
                assert result["choices"][0]["message"]["content"] == "Recovered"
