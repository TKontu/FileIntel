# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Global heartbeat context for activity timeout tracking.

This module provides a thread-safe way to signal activity (heartbeats) from
anywhere in the call stack, including deep within LLM calls and async operations.
The heartbeat signals are used by the activity timeout system to detect stuck workflows.
"""

import contextvars
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Context variable to hold the heartbeat callback
# This allows us to signal activity from anywhere in the async call stack
_heartbeat_callback: contextvars.ContextVar[Callable[[], None] | None] = (
    contextvars.ContextVar("heartbeat_callback", default=None)
)


def set_heartbeat_callback(callback: Callable[[], None] | None) -> None:
    """Set the heartbeat callback for the current async context.

    Args:
        callback: Function to call to signal activity (resets timeout timer)
                  Pass None to disable heartbeats
    """
    _heartbeat_callback.set(callback)


def signal_heartbeat(description: str = "") -> None:
    """Signal activity to reset the timeout timer.

    This can be called from anywhere in the call stack to indicate that
    progress is being made. It's a no-op if no heartbeat callback is set.

    This function is designed to be fail-safe - any errors are caught and logged
    but don't propagate to the caller. This ensures that heartbeat failures never
    crash LLM calls or other operations.

    Args:
        description: Optional description of the activity for debugging
    """
    try:
        callback = _heartbeat_callback.get()
        if callback is not None:
            if description:
                logger.debug(f"Heartbeat: {description}")
            callback()
    except Exception as e:
        # Never let heartbeat failures crash the calling code
        # Just log and continue - worst case is we timeout when we shouldn't
        logger.warning(f"Heartbeat signal failed (non-fatal): {e}")


def has_heartbeat_enabled() -> bool:
    """Check if heartbeat tracking is enabled in the current context.

    Returns:
        True if a heartbeat callback is set, False otherwise
    """
    return _heartbeat_callback.get() is not None
