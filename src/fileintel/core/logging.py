import logging
import sys
import json
from logging.handlers import RotatingFileHandler

# Constants for logging configuration
DEFAULT_LOG_PATH = "logs/fileintel.log"
DEFAULT_CELERY_LOG_PATH = "logs/celery.log"
BYTES_PER_MB = 1024 * 1024


def mb_to_bytes(megabytes: int) -> int:
    """Convert megabytes to bytes."""
    return megabytes * BYTES_PER_MB


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Celery task fields
        task_fields = ["task_id", "task_name", "worker_id", "queue", "retries"]

        for field in task_fields:
            if hasattr(record, field):
                log_record[field] = getattr(record, field)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def create_file_handler(
    file_path: str, formatter: JsonFormatter, max_bytes: int, backup_count: int
) -> RotatingFileHandler:
    """Create a rotating file handler with the given configuration."""
    handler = RotatingFileHandler(
        file_path, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(formatter)
    return handler


def setup_root_logger(settings) -> tuple[str, JsonFormatter]:
    """Setup root logger with console and file handlers."""
    root_logger = logging.getLogger()
    log_level = settings.logging.level.upper()
    root_logger.setLevel(log_level)

    # Preserve uvicorn's handlers - only remove handlers we created previously
    # This ensures uvicorn's startup/shutdown messages continue to appear
    handlers_to_remove = []
    for handler in root_logger.handlers[:]:
        # Only remove our own JsonFormatter handlers from previous setup calls
        # Keep uvicorn's default handlers
        if isinstance(getattr(handler, 'formatter', None), JsonFormatter):
            handlers_to_remove.append(handler)

    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = JsonFormatter()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler
    log_file_path = settings.paths.logs
    max_bytes = mb_to_bytes(settings.logging.max_file_size_mb)
    backup_count = settings.logging.backup_count
    file_handler = create_file_handler(
        log_file_path, formatter, max_bytes, backup_count
    )
    root_logger.addHandler(file_handler)

    # Apply component-specific log levels
    if hasattr(settings.logging, 'component_levels'):
        for component, level in settings.logging.component_levels.items():
            component_logger = logging.getLogger(component)
            component_logger.setLevel(level.upper())

    return log_level, formatter


def test_logging_setup():
    """Test that logging is working correctly."""
    test_logger = logging.getLogger("fileintel.core.logging")
    test_logger.info("Logging system initialized successfully")


def setup_logging(settings):
    """
    Setup complete logging configuration.

    Args:
        settings: Configuration object (required for dependency injection).
    """

    # Setup root logger
    log_level, formatter = setup_root_logger(settings)

    # Configure Celery-specific loggers (imported from celery_config to avoid circular dependency)
    from ..celery_config import setup_celery_logging

    setup_celery_logging(log_level, formatter, settings)

    # Test that logging is working
    test_logging_setup()


# Celery logging setup moved to celery_config.py to avoid circular dependencies


# get_job_logger removed - job concept replaced by Celery tasks
# Use get_task_logger instead for Celery task logging


def get_task_logger(
    name: str, task_id: str, task_name: str, queue: str = None, worker_id: str = None
) -> logging.Logger:
    """Get a logger adapter for Celery tasks with task-specific context."""
    logger = logging.getLogger(name)
    adapter = logging.LoggerAdapter(
        logger,
        {
            "task_id": task_id,
            "task_name": task_name,
            "queue": queue,
            "worker_id": worker_id,
        },
    )
    return adapter
