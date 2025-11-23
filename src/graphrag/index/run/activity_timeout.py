# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Activity-based timeout wrapper for GraphRAG workflows.

This module provides a heartbeat-based timeout mechanism that only triggers
when a workflow is truly stuck (no progress for N seconds), rather than
using a global timeout that can expire even when the workflow is making progress.

Heartbeats are signaled from multiple levels:
1. Workflow-level: workflow_start, workflow_end callbacks
2. Progress-level: progress() callbacks (entity processed, embedding completed, etc.)
3. LLM-level: Before/after each LLM request (injected via heartbeat_context)
"""

import asyncio
import logging
import time
from collections.abc import Awaitable
from typing import TypeVar

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.run.heartbeat_context import set_heartbeat_callback
from graphrag.logger.progress import Progress

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ActivityTimeoutWrapper:
    """Wraps workflow callbacks to track activity and enable heartbeat-based timeouts.

    This wrapper monitors progress callbacks and resets the timeout timer whenever
    progress is made. This prevents timeouts on legitimately long-running tasks
    that are making steady progress.

    Example:
        >>> original_callbacks = context.callbacks
        >>> wrapper = ActivityTimeoutWrapper(original_callbacks, inactivity_timeout=300)
        >>> context.callbacks = wrapper
        >>> result = await run_with_activity_timeout(
        ...     workflow_function(config, context),
        ...     wrapper,
        ...     workflow_name="extract_graph"
        ... )
    """

    def __init__(
        self,
        wrapped_callbacks: WorkflowCallbacks,
        inactivity_timeout: float | None = None
    ):
        """Initialize the activity timeout wrapper.

        Args:
            wrapped_callbacks: The original workflow callbacks to wrap
            inactivity_timeout: Seconds of inactivity before timeout (None = no timeout)
        """
        self._wrapped = wrapped_callbacks
        self._inactivity_timeout = inactivity_timeout
        self._last_activity_time = time.time()
        self._activity_count = 0

    def reset_activity_timer(self) -> None:
        """Reset the activity timer (called on progress)."""
        self._last_activity_time = time.time()
        self._activity_count += 1

    def seconds_since_last_activity(self) -> float:
        """Get seconds elapsed since last activity."""
        return time.time() - self._last_activity_time

    @property
    def activity_count(self) -> int:
        """Get total number of activity signals received."""
        return self._activity_count

    # Delegate all callback methods to the wrapped implementation

    def pipeline_start(self, names: list[str]) -> None:
        """Execute this callback to signal when the entire pipeline starts."""
        self.reset_activity_timer()
        self._wrapped.pipeline_start(names)

    def pipeline_end(self, results: list) -> None:
        """Execute this callback to signal when the entire pipeline ends."""
        self.reset_activity_timer()
        self._wrapped.pipeline_end(results)

    def workflow_start(self, name: str, instance: object) -> None:
        """Execute this callback when a workflow starts."""
        self.reset_activity_timer()
        self._wrapped.workflow_start(name, instance)

    def workflow_end(self, name: str, instance: object) -> None:
        """Execute this callback when a workflow ends."""
        self.reset_activity_timer()
        self._wrapped.workflow_end(name, instance)

    def progress(self, progress: Progress) -> None:
        """Handle when progress occurs (resets activity timer)."""
        self.reset_activity_timer()
        self._wrapped.progress(progress)


async def run_with_activity_timeout(
    task: Awaitable[T],
    activity_wrapper: ActivityTimeoutWrapper,
    workflow_name: str = "workflow",
    check_interval: float = 10.0,
) -> T:
    """Run an async task with activity-based timeout monitoring.

    This function monitors the task and raises TimeoutError if no progress
    is made for longer than the configured inactivity timeout. The timeout
    is reset every time progress is reported through the callbacks OR when
    signal_heartbeat() is called from anywhere in the call stack (e.g., LLM layer).

    Args:
        task: The async task to run (workflow execution)
        activity_wrapper: The activity timeout wrapper tracking progress
        workflow_name: Name of the workflow for error messages
        check_interval: How often to check for inactivity (seconds)

    Returns:
        The result of the completed task

    Raises:
        TimeoutError: If the task shows no activity for longer than inactivity_timeout
    """
    if activity_wrapper._inactivity_timeout is None:
        # No timeout configured, just run the task normally
        # Still register heartbeat callback so LLM calls can signal activity
        set_heartbeat_callback(activity_wrapper.reset_activity_timer)
        try:
            return await task
        finally:
            set_heartbeat_callback(None)

    # Register the heartbeat callback so it can be called from anywhere
    # (e.g., from deep within LLM calls, embedding generation, etc.)
    set_heartbeat_callback(activity_wrapper.reset_activity_timer)

    # Create the task
    task_future = asyncio.create_task(task)

    # Monitor activity in a loop
    try:
        while not task_future.done():
            # Check if we've exceeded the inactivity timeout
            inactive_seconds = activity_wrapper.seconds_since_last_activity()

            if inactive_seconds > activity_wrapper._inactivity_timeout:
                # Task has been inactive too long
                task_future.cancel()
                raise TimeoutError(
                    f"Workflow '{workflow_name}' exceeded inactivity timeout "
                    f"({activity_wrapper._inactivity_timeout}s with no progress). "
                    f"Last activity was {inactive_seconds:.1f}s ago after "
                    f"{activity_wrapper.activity_count} progress updates."
                )

            # Wait a bit before checking again
            try:
                await asyncio.wait_for(
                    asyncio.shield(task_future),
                    timeout=check_interval
                )
                # Task completed successfully
                break
            except asyncio.TimeoutError:
                # Check interval expired, loop continues to check activity
                continue

        # Task completed, return result
        return await task_future

    except asyncio.CancelledError:
        # Task was cancelled (likely due to inactivity timeout)
        task_future.cancel()
        raise
    finally:
        # Always clean up the heartbeat callback
        set_heartbeat_callback(None)
