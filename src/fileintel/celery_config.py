"""
Celery application configuration for FileIntel distributed task processing.

This module provides optimal Celery configuration for multicore utilization
and distributed processing of document analysis, RAG operations, and LLM tasks.
"""

from celery import Celery
from kombu import Queue
from fileintel.core.config import get_config
import logging

# Create Celery application instance
app = Celery("fileintel")


def configure_celery_app(config=None):
    """Configure Celery application with provided or default configuration."""
    if config is None:
        config = get_config()

    celery_settings = config.celery

    # Configure Celery with optimal settings for multicore utilization
    app.conf.update(
        # Broker and result backend configuration
        broker_url=celery_settings.broker_url,
        result_backend=celery_settings.result_backend,
        # Serialization settings
        task_serializer=celery_settings.task_serializer,
        accept_content=celery_settings.accept_content,
        result_serializer=celery_settings.result_serializer,
        # Timezone configuration
        timezone=celery_settings.timezone,
        enable_utc=celery_settings.enable_utc,
        # Worker optimization for multicore utilization
        worker_concurrency=celery_settings.worker_concurrency,
        worker_prefetch_multiplier=celery_settings.worker_prefetch_multiplier,
        task_acks_late=celery_settings.task_acks_late,
        worker_max_tasks_per_child=celery_settings.worker_max_tasks_per_child,
        # Task routing for optimal resource allocation
        task_routes=celery_settings.task_routes,
        # Queue configuration for different task types
        task_default_queue="default",
        task_queues=(
            Queue("default", routing_key="default"),
            Queue("document_processing", routing_key="document_processing"),
            Queue("rag_processing", routing_key="rag_processing"),
            Queue("llm_processing", routing_key="llm_processing"),
            Queue("indexing", routing_key="indexing"),
        ),
        # Result settings
        result_expires=3600,  # Results expire after 1 hour
        result_persistent=True,
        # Task execution settings
        task_compression="gzip",
        result_compression="gzip",
        # Monitoring and debugging
        worker_send_task_events=True,
        task_send_sent_event=True,
        # Flower compatibility settings
        worker_send_task_event=True,
        event_queue_expires=60,
        event_queue_ttl=5,
        # Security settings
        task_reject_on_worker_lost=True,
        task_ignore_result=False,
        # Performance optimizations
        worker_disable_rate_limits=True,
        task_always_eager=False,  # False for production, True for testing
        # Connection pool optimization
        broker_pool_limit=20,
        broker_connection_timeout=10,
        broker_connection_retry_on_startup=True,
        # Advanced worker settings for production
        worker_hijack_root_logger=False,
        worker_log_color=False,
        worker_enable_remote_control=True,
        worker_enable_heartbeats=True,
        # Task retry and error handling
        task_default_retry_delay=60,  # 1 minute default retry delay
        task_max_retries=3,
        # Priority and routing optimizations
        task_inherit_parent_priority=True,
        task_default_priority=5,  # Medium priority
        # Memory and cleanup settings
        worker_max_memory_per_child=500000,  # 500MB per child process (increased for heavy processing)
        task_soft_time_limit=1800,  # 30 minutes soft limit
        task_time_limit=3600,  # 1 hour hard limit
        # Result backend optimizations
        result_backend_max_retries=3,
        result_backend_retry_on_timeout=True,
        # Monitoring and health check settings
        worker_state_db="/tmp/celery_worker_state.db",
        worker_pool_restarts=True,
        # Security enhancements
        worker_redirect_stdouts_level="INFO",
        worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    )


def setup_celery_logging(log_level: str, formatter, settings):
    """Configure Celery-specific logging."""
    from fileintel.core.logging import create_file_handler, mb_to_bytes

    # Configure Celery task logger
    celery_logger = logging.getLogger("celery.task")
    celery_logger.setLevel(log_level)

    # Configure Celery worker logger
    worker_logger = logging.getLogger("celery.worker")
    worker_logger.setLevel(log_level)

    # Configure our custom task loggers
    task_logger = logging.getLogger("fileintel.tasks")
    task_logger.setLevel(log_level)

    # Add file handler for Celery logs
    celery_log_file = settings.paths.celery_logs
    max_bytes = mb_to_bytes(settings.logging.max_file_size_mb)
    backup_count = settings.logging.backup_count
    celery_file_handler = create_file_handler(
        celery_log_file, formatter, max_bytes, backup_count
    )

    # Add handlers to Celery loggers
    for logger in [celery_logger, worker_logger, task_logger]:
        if not logger.handlers:  # Avoid duplicate handlers
            logger.addHandler(celery_file_handler)


# Configure Celery app with default configuration at module load
# This ensures the app is ready for use while still allowing reconfiguration
configure_celery_app()

# Auto-discover tasks from all installed apps
app.autodiscover_tasks(["fileintel.tasks"])


def get_celery_app() -> Celery:
    """Get the configured Celery application instance."""
    return app


# Task result tracking utilities
def get_task_status(task_id: str) -> dict:
    """Get the status of a specific task."""
    result = app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "state": result.state,
        "result": result.result,
        "traceback": result.traceback,
        "info": result.info,
    }


def cancel_task(task_id: str, terminate: bool = False) -> bool:
    """Cancel a task by ID."""
    try:
        app.control.revoke(task_id, terminate=terminate)
        return True
    except Exception:
        return False


def get_active_tasks() -> dict:
    """Get currently active tasks from all workers."""
    try:
        inspect = app.control.inspect()
        return inspect.active() or {}
    except Exception:
        return {}


def get_worker_stats() -> dict:
    """Get statistics from all workers."""
    try:
        inspect = app.control.inspect()
        return inspect.stats() or {}
    except Exception:
        return {}


def get_queue_lengths() -> dict:
    """Get current queue lengths."""
    try:
        with app.connection() as conn:
            queues = {}
            for queue_name in [
                "default",
                "document_processing",
                "llm_processing",
                "memory_intensive",
                "io_bound",
            ]:
                try:
                    queue = conn.SimpleQueue(queue_name)
                    queues[queue_name] = queue.qsize()
                    queue.close()
                except Exception:
                    queues[queue_name] = 0
            return queues
    except Exception:
        return {}


def purge_queue(queue_name: str) -> int:
    """Purge all messages from a queue. Returns number of messages purged."""
    try:
        return app.control.purge()
    except Exception:
        return 0


def broadcast_task_revoke(task_id: str, terminate: bool = False) -> dict:
    """Broadcast task revocation to all workers."""
    try:
        return app.control.revoke(task_id, terminate=terminate)
    except Exception:
        return {}


# Worker management utilities for Flower integration
def enable_worker_management():
    """Enable additional worker management features for Flower."""
    # Ensure workers can be properly inspected
    try:
        inspect = app.control.inspect()

        # Test basic inspection capabilities
        registered_tasks = inspect.registered()
        active_tasks = inspect.active()

        return True
    except Exception:
        return False


# Production monitoring hooks
@app.task(bind=True)
def health_check_task(self):
    """Health check task for monitoring system status."""
    import psutil
    import time

    return {
        "timestamp": time.time(),
        "worker_id": self.request.id,
        "hostname": self.request.hostname,
        "memory_percent": psutil.virtual_memory().percent,
        "cpu_percent": psutil.cpu_percent(interval=1),
        "disk_usage": psutil.disk_usage("/").percent,
        "status": "healthy",
    }


# Signal handlers for production monitoring
from celery.signals import task_success, task_failure, task_retry


@task_success.connect
def task_success_handler(
    sender=None, task_id=None, result=None, retries=None, einfo=None, **kwargs
):
    """Handle successful task completion for monitoring."""
    # Log success for monitoring systems
    pass


@task_failure.connect
def task_failure_handler(
    sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs
):
    """Handle task failures for monitoring and alerting."""
    # Log failure for monitoring systems
    pass


@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kwargs):
    """Handle task retries for monitoring."""
    # Log retry for monitoring systems
    pass


# Celery monitoring functions
def get_celery_app():
    """Get the configured Celery application instance."""
    return app


def get_worker_stats():
    """Get worker statistics from Celery."""
    try:
        inspect = app.control.inspect()
        stats = inspect.stats()
        if not stats:
            return {}

        # Aggregate stats from all workers
        total_stats = {
            "total_workers": len(stats),
            "total_tasks_executed": 0,
            "total_tasks_active": 0,
            "workers": {},
        }

        for worker_name, worker_stats in stats.items():
            total_stats["workers"][worker_name] = worker_stats
            if "total" in worker_stats:
                total_stats["total_tasks_executed"] += worker_stats["total"].get(
                    "total", 0
                )
            if "len" in worker_stats.get("pool", {}):
                total_stats["total_tasks_active"] += worker_stats["pool"]["len"]

        return total_stats
    except Exception as e:
        logger.warning(f"Failed to get worker stats: {e}")
        return {}


def get_active_tasks():
    """Get currently active tasks from all workers."""
    try:
        inspect = app.control.inspect()
        active_tasks = inspect.active()
        if not active_tasks:
            return {}

        # Flatten tasks from all workers
        all_tasks = {}
        for worker_name, tasks in active_tasks.items():
            for task in tasks:
                task_id = task.get("id")
                if task_id:
                    all_tasks[task_id] = {
                        "id": task_id,
                        "name": task.get("name"),
                        "args": task.get("args", []),
                        "kwargs": task.get("kwargs", {}),
                        "worker": worker_name,
                        "time_start": task.get("time_start"),
                        "delivery_info": task.get("delivery_info", {}),
                    }

        return all_tasks
    except Exception as e:
        logger.warning(f"Failed to get active tasks: {e}")
        return {}


def get_task_status(task_id: str):
    """Get detailed status of a specific task."""
    try:
        from celery.result import AsyncResult

        result = AsyncResult(task_id, app=app)

        # Get basic status
        status_info = {
            "id": task_id,
            "status": result.status,
            "result": None,
            "traceback": None,
            "worker": None,
            "timestamp": None,
        }

        # Add result if task is completed
        if result.successful():
            status_info["result"] = result.result
        elif result.failed():
            status_info["result"] = str(result.result) if result.result else None
            status_info["traceback"] = result.traceback

        # Try to find worker info from active tasks
        active_tasks = get_active_tasks()
        if task_id in active_tasks:
            task_info = active_tasks[task_id]
            status_info["worker"] = task_info.get("worker")
            status_info["timestamp"] = task_info.get("time_start")
            status_info["name"] = task_info.get("name")
            status_info["args"] = task_info.get("args", [])
            status_info["kwargs"] = task_info.get("kwargs", {})

        return status_info
    except Exception as e:
        logger.warning(f"Failed to get task status for {task_id}: {e}")
        return None


def cancel_task(task_id: str, terminate: bool = False):
    """Cancel a running task."""
    try:
        # Revoke the task
        app.control.revoke(task_id, terminate=terminate)

        # Verify cancellation by checking if task is still active
        active_tasks = get_active_tasks()
        is_still_active = task_id in active_tasks

        return not is_still_active
    except Exception as e:
        logger.warning(f"Failed to cancel task {task_id}: {e}")
        return False
