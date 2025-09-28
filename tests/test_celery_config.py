"""
Test configuration for Celery tasks.

Provides test-specific Celery configuration to isolate task execution during testing.
"""

from celery import Celery
from unittest.mock import MagicMock

# Test Celery app configuration
test_celery_app = Celery("fileintel_test")
test_celery_app.conf.update(
    task_always_eager=True,  # Execute tasks synchronously during testing
    task_eager_propagates=True,  # Propagate exceptions immediately
    broker_url="memory://",  # Use in-memory broker for testing
    result_backend="cache+memory://",  # Use in-memory result backend
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


def mock_celery_task_decorator(*args, **kwargs):
    """Mock Celery task decorator for testing."""

    def decorator(func):
        # Add mock attributes that Celery tasks normally have
        func.apply_async = MagicMock()
        func.delay = MagicMock()
        func.s = MagicMock()  # For signature creation
        func.si = MagicMock()  # For immutable signature creation
        return func

    return decorator


def setup_test_celery_app():
    """Setup test Celery app for testing."""
    return test_celery_app


def mock_task_result(task_id=None, state="SUCCESS", result=None):
    """Create a mock task result for testing."""
    mock_result = MagicMock()
    mock_result.id = task_id or "test-task-id"
    mock_result.state = state
    mock_result.result = result or {"status": "completed"}
    mock_result.info = {"current": 10, "total": 10, "description": "Completed"}
    mock_result.ready.return_value = state in ["SUCCESS", "FAILURE", "REVOKED"]
    mock_result.successful.return_value = state == "SUCCESS"
    mock_result.failed.return_value = state == "FAILURE"
    mock_result.get.return_value = result or {"status": "completed"}
    return mock_result


def mock_task_submission(task_func, *args, **kwargs):
    """Mock task submission for testing."""
    # Execute the task function directly during testing
    result = task_func(*args, **kwargs)
    return mock_task_result(result=result)
