"""Unit tests for activity-based timeout mechanism."""

import asyncio
import pytest
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.index.run.activity_timeout import (
    ActivityTimeoutWrapper,
    run_with_activity_timeout,
)
from graphrag.logger.progress import Progress


class TestActivityTimeoutWrapper:
    """Test the ActivityTimeoutWrapper class."""

    def test_initialization(self):
        """Test wrapper initialization."""
        callbacks = NoopWorkflowCallbacks()
        wrapper = ActivityTimeoutWrapper(callbacks, inactivity_timeout=300)

        assert wrapper._inactivity_timeout == 300
        assert wrapper.activity_count == 0
        assert wrapper.seconds_since_last_activity() >= 0

    def test_activity_tracking(self):
        """Test that activity is tracked correctly."""
        callbacks = NoopWorkflowCallbacks()
        wrapper = ActivityTimeoutWrapper(callbacks, inactivity_timeout=300)

        # Initial state
        assert wrapper.activity_count == 0

        # Call various callbacks
        wrapper.pipeline_start([])
        assert wrapper.activity_count == 1

        wrapper.workflow_start("test", None)
        assert wrapper.activity_count == 2

        wrapper.progress(Progress(total_items=10, completed_items=5, description="test"))
        assert wrapper.activity_count == 3

        wrapper.workflow_end("test", None)
        assert wrapper.activity_count == 4

    def test_timer_reset(self):
        """Test that activity resets the timer."""
        callbacks = NoopWorkflowCallbacks()
        wrapper = ActivityTimeoutWrapper(callbacks, inactivity_timeout=300)

        # Initial timer
        initial_time = wrapper._last_activity_time

        # Wait a bit
        import time
        time.sleep(0.1)

        # Report progress
        wrapper.progress(Progress(total_items=10, completed_items=5))

        # Timer should be reset
        assert wrapper._last_activity_time > initial_time
        assert wrapper.seconds_since_last_activity() < 0.2


@pytest.mark.asyncio
class TestRunWithActivityTimeout:
    """Test the run_with_activity_timeout function."""

    async def test_successful_completion(self):
        """Test that a task completes successfully without timeout."""
        callbacks = NoopWorkflowCallbacks()
        wrapper = ActivityTimeoutWrapper(callbacks, inactivity_timeout=2.0)

        async def quick_task():
            wrapper.progress(Progress(total_items=1, completed_items=0))
            await asyncio.sleep(0.1)
            wrapper.progress(Progress(total_items=1, completed_items=1))
            return "success"

        result = await run_with_activity_timeout(
            quick_task(),
            wrapper,
            workflow_name="test",
            check_interval=0.5
        )

        assert result == "success"
        assert wrapper.activity_count >= 2

    async def test_timeout_on_inactivity(self):
        """Test that timeout triggers when task is inactive."""
        callbacks = NoopWorkflowCallbacks()
        wrapper = ActivityTimeoutWrapper(callbacks, inactivity_timeout=1.0)

        async def stuck_task():
            # Report initial progress
            wrapper.progress(Progress(total_items=1, completed_items=0))
            # Then hang without progress
            await asyncio.sleep(3.0)
            return "should not reach here"

        with pytest.raises(TimeoutError) as exc_info:
            await run_with_activity_timeout(
                stuck_task(),
                wrapper,
                workflow_name="test_workflow",
                check_interval=0.2
            )

        assert "test_workflow" in str(exc_info.value)
        assert "inactivity timeout" in str(exc_info.value)

    async def test_no_timeout_with_regular_progress(self):
        """Test that timeout doesn't trigger when task reports regular progress."""
        callbacks = NoopWorkflowCallbacks()
        wrapper = ActivityTimeoutWrapper(callbacks, inactivity_timeout=1.0)

        async def progressing_task():
            # Simulate a long task that reports progress regularly
            for i in range(10):
                wrapper.progress(Progress(total_items=10, completed_items=i))
                await asyncio.sleep(0.3)  # Each step takes 0.3s
            return "completed"

        # Total time: 3 seconds, but no single gap is > 1 second
        result = await run_with_activity_timeout(
            progressing_task(),
            wrapper,
            workflow_name="test",
            check_interval=0.2
        )

        assert result == "completed"
        assert wrapper.activity_count >= 10

    async def test_no_timeout_when_disabled(self):
        """Test that tasks run without timeout when it's disabled."""
        callbacks = NoopWorkflowCallbacks()
        wrapper = ActivityTimeoutWrapper(callbacks, inactivity_timeout=None)

        async def slow_task():
            # No progress reported, but timeout is disabled
            await asyncio.sleep(1.0)
            return "completed anyway"

        result = await run_with_activity_timeout(
            slow_task(),
            wrapper,
            workflow_name="test",
            check_interval=0.2
        )

        assert result == "completed anyway"


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    async def test_graphrag_like_workflow(self):
        """Test a scenario similar to actual GraphRAG workflow."""
        callbacks = NoopWorkflowCallbacks()
        wrapper = ActivityTimeoutWrapper(callbacks, inactivity_timeout=2.0)

        async def simulated_workflow():
            # Simulate extracting entities (makes progress)
            for i in range(5):
                wrapper.progress(Progress(
                    total_items=5,
                    completed_items=i,
                    description="Extracting entities"
                ))
                await asyncio.sleep(0.2)

            # Simulate a brief pause (less than timeout)
            await asyncio.sleep(0.5)

            # Simulate embedding (makes progress)
            for i in range(3):
                wrapper.progress(Progress(
                    total_items=3,
                    completed_items=i,
                    description="Generating embeddings"
                ))
                await asyncio.sleep(0.3)

            return "workflow complete"

        result = await run_with_activity_timeout(
            simulated_workflow(),
            wrapper,
            workflow_name="extract_graph",
            check_interval=0.3
        )

        assert result == "workflow complete"
        # Should have at least 8 progress updates (5 + 3)
        assert wrapper.activity_count >= 8
