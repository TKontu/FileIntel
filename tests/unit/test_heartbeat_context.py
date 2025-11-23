"""Unit tests for heartbeat context mechanism."""

import asyncio
import pytest
from graphrag.index.run.heartbeat_context import (
    has_heartbeat_enabled,
    set_heartbeat_callback,
    signal_heartbeat,
)


class TestHeartbeatContext:
    """Test the heartbeat context mechanism."""

    def test_default_state(self):
        """Test that heartbeat is disabled by default."""
        assert not has_heartbeat_enabled()

    def test_enable_heartbeat(self):
        """Test enabling heartbeat callback."""
        called = []

        def callback():
            called.append(True)

        set_heartbeat_callback(callback)
        assert has_heartbeat_enabled()

        signal_heartbeat("test")
        assert len(called) == 1

        # Cleanup
        set_heartbeat_callback(None)
        assert not has_heartbeat_enabled()

    def test_signal_without_callback(self):
        """Test that signal_heartbeat is safe when no callback is set."""
        set_heartbeat_callback(None)
        # Should not raise
        signal_heartbeat("test")
        signal_heartbeat()

    def test_multiple_signals(self):
        """Test multiple heartbeat signals."""
        count = [0]

        def callback():
            count[0] += 1

        set_heartbeat_callback(callback)

        for i in range(10):
            signal_heartbeat(f"signal {i}")

        assert count[0] == 10

        # Cleanup
        set_heartbeat_callback(None)

    def test_callback_exception_handling(self):
        """Test that callback exceptions don't crash the caller."""
        def failing_callback():
            raise ValueError("Test error")

        set_heartbeat_callback(failing_callback)

        # Should not raise - exception is caught and logged
        signal_heartbeat("test")

        # Cleanup
        set_heartbeat_callback(None)

    def test_description_parameter(self):
        """Test heartbeat with and without description."""
        called = []

        def callback():
            called.append(True)

        set_heartbeat_callback(callback)

        signal_heartbeat()  # No description
        signal_heartbeat("with description")

        assert len(called) == 2

        # Cleanup
        set_heartbeat_callback(None)


@pytest.mark.asyncio
class TestHeartbeatContextAsync:
    """Test heartbeat context in async scenarios."""

    async def test_context_isolation(self):
        """Test that heartbeat callbacks are isolated per async context."""
        results = {"task1": 0, "task2": 0}

        async def task1():
            def callback():
                results["task1"] += 1

            set_heartbeat_callback(callback)
            for _ in range(3):
                signal_heartbeat()
                await asyncio.sleep(0.01)
            set_heartbeat_callback(None)

        async def task2():
            def callback():
                results["task2"] += 1

            set_heartbeat_callback(callback)
            for _ in range(5):
                signal_heartbeat()
                await asyncio.sleep(0.01)
            set_heartbeat_callback(None)

        # Run both tasks concurrently
        await asyncio.gather(task1(), task2())

        # Each task should have its own count
        assert results["task1"] == 3
        assert results["task2"] == 5

    async def test_llm_style_heartbeats(self):
        """Test heartbeat pattern similar to LLM calls."""
        heartbeat_count = [0]

        def callback():
            heartbeat_count[0] += 1

        set_heartbeat_callback(callback)

        # Simulate LLM call pattern
        async def mock_llm_call():
            signal_heartbeat("LLM request starting")
            await asyncio.sleep(0.05)  # Simulate network call
            signal_heartbeat("LLM response received")

        # Make multiple calls
        for _ in range(3):
            await mock_llm_call()

        # Should have 2 heartbeats per call (before + after)
        assert heartbeat_count[0] == 6

        # Cleanup
        set_heartbeat_callback(None)

    async def test_nested_operations(self):
        """Test heartbeats in nested async operations."""
        operations = []

        def callback():
            operations.append(True)

        set_heartbeat_callback(callback)

        async def inner_operation():
            signal_heartbeat("inner start")
            await asyncio.sleep(0.01)
            signal_heartbeat("inner end")

        async def outer_operation():
            signal_heartbeat("outer start")
            await inner_operation()
            signal_heartbeat("outer end")

        await outer_operation()

        # Should have 4 heartbeats total
        assert len(operations) == 4

        # Cleanup
        set_heartbeat_callback(None)
