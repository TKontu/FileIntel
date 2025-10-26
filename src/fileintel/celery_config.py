"""
Celery application configuration for FileIntel distributed task processing.

This module provides optimal Celery configuration for multicore utilization
and distributed processing of document analysis, RAG operations, and LLM tasks.
"""

from celery import Celery
from kombu import Queue
from fileintel.core.config import get_config
import logging

logger = logging.getLogger(__name__)

# Create Celery application instance
app = Celery("fileintel")

# Shared storage configuration for Celery tasks
_shared_engine = None
_shared_session_factory = None
_storage_lock = None

def get_shared_storage():
    """
    Get a new storage instance with shared connection pool.

    This creates a new storage instance for each task while sharing
    the underlying connection pool across all workers.

    Usage in tasks:
        storage = get_shared_storage()
        try:
            # Use storage
            result = storage.some_operation()
        finally:
            storage.close()
    """
    import threading
    global _shared_engine, _shared_session_factory, _storage_lock

    # Initialize lock if not exists
    if _storage_lock is None:
        _storage_lock = threading.Lock()

    # Thread-safe initialization using double-checked locking pattern
    if _shared_engine is None:
        with _storage_lock:
            # Check again after acquiring lock
            if _shared_engine is None:
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker

                config = get_config()
                database_url = config.storage.connection_string

                # Create shared engine with pool settings from config
                _shared_engine = create_engine(
                    database_url,
                    pool_pre_ping=True,  # Test connections before use
                    pool_size=config.storage.pool_size,
                    max_overflow=config.storage.max_overflow,
                    pool_recycle=3600,  # Recycle connections every hour
                    pool_timeout=config.storage.pool_timeout,
                    echo=False,  # Disable SQL logging in production
                )

                # Create session factory
                _shared_session_factory = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=_shared_engine
                )

    # Create new session for this task
    from fileintel.storage.postgresql_storage import PostgreSQLStorage
    session = _shared_session_factory()
    return PostgreSQLStorage(session)


def get_storage_context():
    """
    Get a context manager for storage that automatically handles session cleanup.

    Usage in tasks:
        with get_storage_context() as storage:
            # Use storage
            result = storage.some_operation()
        # Session is automatically closed
    """
    from contextlib import contextmanager

    @contextmanager
    def storage_context():
        storage = get_shared_storage()
        try:
            yield storage
        finally:
            storage.close()

    return storage_context()


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
            Queue("default", routing_key="default"),                            # Catch-all
            Queue("document_processing", routing_key="document_processing"),    # File processing, chunking
            Queue("embedding_processing", routing_key="embedding_processing"),  # High-throughput embedding generation
            Queue("llm_processing", routing_key="llm_processing"),              # Text generation, summarization
            Queue("rag_processing", routing_key="rag_processing"),              # Vector queries, lightweight ops
            Queue("graphrag_indexing", routing_key="graphrag_indexing"),        # Heavy GraphRAG index building
            Queue("graphrag_queries", routing_key="graphrag_queries"),          # GraphRAG query operations
        ),
        # Result settings
        result_expires=3600,  # Results expire after 1 hour
        result_persistent=True,
        # Task execution settings
        task_compression="gzip",
        result_compression="gzip",
        # Task state tracking - critical for detecting stale tasks
        task_track_started=True,  # Track when tasks actually start (not just queued)
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
        worker_max_memory_per_child=4000000,  # 4GB per child process (handles GraphRAG heavy operations)
        task_soft_time_limit=celery_settings.task_soft_time_limit,  # Configurable from YAML
        task_time_limit=celery_settings.task_time_limit,  # Configurable from YAML
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
                "embedding_processing",
                "llm_processing",
                "rag_processing",
                "graphrag_indexing",
                "graphrag_queries",
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


# Signal handlers for production monitoring and task tracking
from celery.signals import (
    task_success,
    task_failure,
    task_retry,
    worker_ready,
    celeryd_after_setup,
    task_prerun,
    task_postrun,
)
from datetime import datetime, timedelta


def _get_task_registry_session():
    """Get a database session for task registry operations."""
    from fileintel.storage.models import SessionLocal

    return SessionLocal()


def _safe_serialize(data):
    """
    Safely serialize task arguments for JSONB storage.

    Handles non-serializable objects by converting to string representation.
    """
    if data is None:
        return None

    import json
    try:
        # Test if data is JSON-serializable
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        # Fall back to string representation for non-serializable types
        try:
            return str(data)[:1000]  # Limit size to prevent bloat
        except Exception:
            return "<unserializable>"


@task_prerun.connect
def task_started_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Track task start in database registry."""
    if not task_id:
        logger.error("task_prerun called with empty task_id - skipping registry")
        return

    try:
        from fileintel.storage.models import CeleryTaskRegistry
        import os

        session = _get_task_registry_session()
        try:
            # Get worker info
            worker_id = task.request.hostname if task and hasattr(task, 'request') else 'unknown'
            worker_pid = os.getpid()

            # Create or update task registry entry
            task_entry = session.query(CeleryTaskRegistry).filter_by(task_id=task_id).first()

            if task_entry:
                # Update existing entry
                task_entry.status = 'STARTED'
                task_entry.started_at = datetime.utcnow()
                task_entry.worker_id = worker_id
                task_entry.worker_pid = worker_pid
                task_entry.last_heartbeat = datetime.utcnow()
            else:
                # Create new entry
                task_entry = CeleryTaskRegistry(
                    task_id=task_id,
                    task_name=sender.name if sender else 'unknown',
                    worker_id=worker_id,
                    worker_pid=worker_pid,
                    status='STARTED',
                    started_at=datetime.utcnow(),
                    last_heartbeat=datetime.utcnow(),
                    args=_safe_serialize(args),
                    kwargs=_safe_serialize(kwargs),
                )
                session.add(task_entry)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error tracking task start for {task_id}: {e}")
        # Don't fail the task due to tracking issues


@task_success.connect
def task_success_handler(
    sender=None, task_id=None, result=None, retries=None, einfo=None, **kwargs
):
    """Handle successful task completion and update registry."""
    try:
        from fileintel.storage.models import CeleryTaskRegistry

        session = _get_task_registry_session()
        try:
            task_entry = session.query(CeleryTaskRegistry).filter_by(task_id=task_id).first()
            if task_entry:
                task_entry.status = 'SUCCESS'
                task_entry.completed_at = datetime.utcnow()
                task_entry.result = {'success': True, 'result': str(result)[:1000]}  # Limit size
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error updating task success for {task_id}: {e}")


@task_failure.connect
def task_failure_handler(
    sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs
):
    """Handle task failures and update registry."""
    try:
        from fileintel.storage.models import CeleryTaskRegistry

        session = _get_task_registry_session()
        try:
            task_entry = session.query(CeleryTaskRegistry).filter_by(task_id=task_id).first()
            if task_entry:
                task_entry.status = 'FAILURE'
                task_entry.completed_at = datetime.utcnow()
                task_entry.result = {'error': str(exception)[:1000]}  # Limit size
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error updating task failure for {task_id}: {e}")


@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kwargs):
    """Handle task retries and update registry."""
    try:
        from fileintel.storage.models import CeleryTaskRegistry

        session = _get_task_registry_session()
        try:
            task_entry = session.query(CeleryTaskRegistry).filter_by(task_id=task_id).first()
            if task_entry:
                task_entry.status = 'RETRY'
                task_entry.last_heartbeat = datetime.utcnow()
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error updating task retry for {task_id}: {e}")


@worker_ready.connect
def cleanup_stale_tasks(sender=None, **kwargs):
    """
    Clean up stale tasks on worker startup.

    When workers are forcibly terminated (e.g., docker-compose down),
    tasks can be left in STARTED state in the database. This handler:
    1. Finds tasks in STARTED state from workers that are no longer alive
    2. Revokes those tasks so they can be retried or cleaned up

    NOTE: This handler has been optimized to run quickly and avoid blocking worker startup.
    """
    import os
    import threading

    logger.info(f"Worker ready signal received (PID: {os.getpid()}, Thread: {threading.current_thread().name})")

    # Run cleanup in background thread to avoid blocking worker startup
    def background_cleanup():
        try:
            _do_stale_task_cleanup()
        except Exception as e:
            logger.error(f"Background stale task cleanup failed: {e}")

    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    cleanup_thread.start()
    logger.info("Stale task cleanup started in background thread")


def _do_stale_task_cleanup():
    """Actual cleanup logic - runs in background thread."""
    logger.info("Background stale task cleanup starting...")

    try:
        from fileintel.storage.models import CeleryTaskRegistry

        # Get currently active workers
        inspect = app.control.inspect()
        stats = inspect.stats()

        if not stats:
            logger.warning(
                "Cannot get worker stats from Celery - skipping stale task cleanup "
                "to avoid incorrectly revoking active tasks during worker startup"
            )
            return

        active_worker_ids = set(stats.keys())
        logger.info(f"Active workers: {active_worker_ids}")

        # Get currently executing tasks to avoid revoking them
        active_task_ids = set()
        active_tasks = inspect.active()
        if active_tasks:
            for worker_tasks in active_tasks.values():
                for task_dict in worker_tasks:
                    active_task_ids.add(task_dict['id'])
            logger.info(f"Found {len(active_task_ids)} currently active tasks")

        # Query database for tasks in STARTED or RETRY state
        session = _get_task_registry_session()
        try:
            stale_tasks = (
                session.query(CeleryTaskRegistry)
                .filter(CeleryTaskRegistry.status.in_(['STARTED', 'RETRY']))
                .all()
            )

            if not stale_tasks:
                logger.info("No tasks in STARTED/RETRY state found")
                return

            # Batch collect stale tasks to revoke (more efficient than one-by-one)
            stale_count = 0
            tasks_to_revoke = []

            for task_entry in stale_tasks:
                # Skip if task is currently active on any worker (prevents race condition)
                if task_entry.task_id in active_task_ids:
                    logger.debug(f"Task {task_entry.task_id} is currently active - skipping")
                    continue

                # Check if worker is still alive
                if task_entry.worker_id not in active_worker_ids:
                    tasks_to_revoke.append(task_entry)

            # Batch revoke and update database
            if tasks_to_revoke:
                logger.warning(
                    f"Found {len(tasks_to_revoke)} stale tasks from dead workers - batch revoking"
                )

                try:
                    # Batch revoke all stale tasks at once
                    for task_entry in tasks_to_revoke:
                        app.control.revoke(task_entry.task_id, terminate=False)

                        # Update in-memory (batch commit later)
                        task_entry.status = 'REVOKED'
                        task_entry.completed_at = datetime.utcnow()
                        task_entry.result = {
                            'error': f'Worker {task_entry.worker_id} died unexpectedly'
                        }
                        stale_count += 1

                    # Single commit for all updates
                    session.commit()
                    logger.info(f"Batch revoked {stale_count} stale tasks")

                except Exception as revoke_error:
                    session.rollback()
                    logger.error(f"Error during batch revoke: {revoke_error}")

            # Check for stuck tasks on alive workers (heartbeat monitoring)
            for task_entry in stale_tasks:
                if task_entry.task_id not in active_task_ids and task_entry.worker_id in active_worker_ids:
                    # Worker is alive but task might be stuck - check heartbeat
                    if task_entry.last_heartbeat:
                        time_since_heartbeat = datetime.utcnow() - task_entry.last_heartbeat
                        # If no heartbeat for 6 hours, consider it stale
                        if time_since_heartbeat > timedelta(hours=6):
                            logger.warning(
                                f"Task {task_entry.task_id} has no heartbeat for "
                                f"{time_since_heartbeat} - may be stuck"
                            )
                            # Don't auto-revoke - just log warning for manual investigation

            if stale_count > 0:
                logger.info(f"Cleanup complete: Revoked {stale_count} stale tasks from dead workers")
            else:
                logger.info("No stale tasks found requiring cleanup")

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error during stale task cleanup: {e}")
        # Don't fail worker startup due to cleanup issues

    logger.info("Stale task check complete")


@celeryd_after_setup.connect
def setup_worker_logging(sender=None, instance=None, **kwargs):
    """Configure enhanced logging after worker setup."""
    logger.info(f"Celery worker initialized: {sender}")
    logger.info("Task acks_late=True: Tasks will be requeued if worker dies")
    logger.info("Task reject_on_worker_lost=True: Tasks will fail if worker is lost")
    logger.info("Worker ready to process tasks")
