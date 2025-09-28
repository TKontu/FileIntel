"""
Base Celery task class with common error handling and logging.

Provides a foundation for all FileIntel tasks with standardized:
- Error handling and retry logic
- Logging configuration
- Progress tracking
- Resource monitoring
"""

import logging
import time
from typing import Any, Dict, Optional
from celery import Task
from celery.exceptions import Retry
from fileintel.core.config import get_config

logger = logging.getLogger(__name__)


# Lazy import to avoid circular dependency during migrations
def get_celery_app():
    """Get Celery app instance, importing only when needed."""
    from fileintel.celery_config import app

    return app


class BaseFileIntelTask(Task):
    """
    Base task class for all FileIntel operations.

    Provides standardized error handling, logging, and progress tracking
    for distributed task execution across worker processes.
    """

    # Task metadata
    abstract = True

    # Retry configuration
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 60}
    retry_backoff = True
    retry_jitter = True

    def __init__(self):
        self.config = get_config()

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called when the task succeeds."""
        logger.info(f"Task {self.name}[{task_id}] completed successfully")

    def on_failure(
        self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo
    ) -> None:
        """Called when the task fails."""
        logger.error(f"Task {self.name}[{task_id}] failed: {exc}")

    def on_retry(
        self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo
    ) -> None:
        """Called when the task is retried."""
        logger.warning(f"Task {self.name}[{task_id}] retrying due to: {exc}")

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update task progress information."""
        progress = {
            "current": current,
            "total": total,
            "percentage": round((current / total) * 100, 2) if total > 0 else 0,
            "message": message,
            "timestamp": time.time(),
        }
        self.update_state(state="PROGRESS", meta=progress)

    def log_execution_time(self, start_time: float, operation: str) -> None:
        """Log execution time for performance monitoring."""
        execution_time = time.time() - start_time
        logger.info(f"{operation} completed in {execution_time:.2f} seconds")

    def validate_input(self, required_fields: list, **kwargs) -> None:
        """Validate required input fields."""
        from fileintel.core.validation import validate_required_fields

        validate_required_fields(kwargs, required_fields)


# Bind the base task to the Celery app (only when Celery is available)
try:
    app = get_celery_app()
    app.Task = BaseFileIntelTask
except ImportError:
    # Celery not available during migrations or in development
    pass
