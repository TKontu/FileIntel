from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from fileintel.core.logging import setup_logging
from fileintel.core.config import get_config
from fileintel.storage.models import create_tables
from fileintel.api.routes import (
    collections_v2,
    tasks_v2,
    websocket_v2,
    query,
    graphrag_v2,
    metadata_v2,
    documents_v2,
)
from fileintel.rag.graph_rag.services.dataframe_cache import GraphRAGDataFrameCache
from fileintel.api.dependencies import get_storage
from fileintel.storage.simple_cache import get_cache

app = FastAPI(
    title="FileIntel API",
    description="API for File Intel.",
    version="0.1.0",
)

# API version constants
API_V1_PREFIX = "/api/v1"
API_V2_PREFIX = "/api/v2"


def configure_cors(app: FastAPI, config=None):
    """Configure CORS middleware with provided or default configuration."""
    if config is None:
        config = get_config()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Configure CORS with default configuration
configure_cors(app)

# V1 routes removed - migrated to task-based v2 API
# V2 Task-based API endpoints
app.include_router(collections_v2.router, prefix=API_V2_PREFIX, tags=["collections-v2"])
app.include_router(documents_v2.router, prefix=API_V2_PREFIX, tags=["documents-v2"])
app.include_router(tasks_v2.router, prefix=API_V2_PREFIX, tags=["tasks-v2"])
app.include_router(query.router, prefix=API_V2_PREFIX, tags=["query-v2"])
app.include_router(graphrag_v2.router, prefix=API_V2_PREFIX, tags=["graphrag-v2"])
app.include_router(metadata_v2.router, prefix=API_V2_PREFIX, tags=["metadata-v2"])
app.include_router(websocket_v2.router, prefix=API_V2_PREFIX, tags=["websocket-v2"])
# V1 websocket routes removed - functionality replaced by V2 task-based monitoring

try:
    from prometheus_client import Counter

    HAS_PROMETHEUS = True

    # Prometheus metrics for task monitoring (Celery-based)
    tasks_processed = Counter(
        "tasks_processed_total",
        "Total number of tasks processed",
        ["task_type", "status"],
    )

    tasks_retried = Counter(
        "tasks_retried_total",
        "Total number of task retries",
        ["task_type", "retry_count"],
    )
except ImportError:
    HAS_PROMETHEUS = False

    # Create mock objects that act like Counter but do nothing
    class MockCounter:
        def __init__(self, *args, **kwargs):
            self._value = type("MockValue", (), {"_value": 0})()

        def inc(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    tasks_processed = MockCounter()
    tasks_retried = MockCounter()


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    if not HAS_PROMETHEUS:
        return {
            "error": "Prometheus client not available",
            "tasks_processed": 0,
            "tasks_retried": 0,
        }
    return {
        "tasks_processed": tasks_processed._value._value,
        "tasks_retried": tasks_retried._value._value,
    }


@app.get(f"{API_V1_PREFIX}/metrics/summary")
def metrics_summary():
    """Detailed metrics summary endpoint."""
    if not HAS_PROMETHEUS:
        return {
            "status": "ok",
            "prometheus_available": False,
            "tasks_processed": 0,
            "tasks_retried": 0,
        }
    return {
        "status": "ok",
        "prometheus_available": True,
        "tasks_processed": tasks_processed._value._value,
        "tasks_retried": tasks_retried._value._value,
    }


@app.get(f"{API_V1_PREFIX}/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    cache = get_cache()
    return {"status": "ok", "redis_available": cache.ping()}


@app.delete(f"{API_V1_PREFIX}/cache/{{namespace}}")
async def clear_cache_namespace(namespace: str):
    """Clear a specific cache namespace."""
    from fileintel.storage.simple_cache import clear_cache_namespace

    cleared_count = clear_cache_namespace(namespace)
    return {
        "success": cleared_count > 0,
        "message": f"Cleared {cleared_count} keys in namespace: {namespace}",
        "timestamp": __import__("time").time(),
    }


@app.on_event("startup")
async def on_startup():
    import os
    import asyncio

    # ============================================================================
    # CRITICAL FIX: fnllm v0.4.1 Concurrency Bottleneck
    # ============================================================================
    # Issue: fnllm has a class-level Semaphore(1) in LimitContext.acquire_semaphore
    #        that serializes ALL limiter acquisitions across ALL requests globally.
    #        This defeats parallelism entirely despite our concurrent_requests=25 config.
    #
    # Root Cause: /fnllm/limiting/base.py:20
    #   class LimitContext:
    #       acquire_semaphore: ClassVar[Semaphore] = Semaphore()  # Defaults to 1!
    #
    # Impact: GraphRAG queries that should take 30-60 seconds timeout after 5 minutes
    #         because only 1 LLM request processes at a time instead of 8-25 in parallel.
    #
    # Safety: This fix is safe because:
    #   1. All requests share the same CompositeLimiter instance
    #   2. All acquire limiters in the same order (no circular dependencies)
    #   3. Individual limiters have their own synchronization primitives
    #
    # See: /docs/graphrag_concurrency_bottleneck_analysis.md for full analysis
    # ============================================================================
    try:
        from fnllm.limiting.base import LimitContext

        # Replace class-level semaphore to match vLLM capacity and config
        # Standard: 16 concurrent requests (matches VLLM_MAX_NUM_SEQS and max_concurrent_requests)
        LimitContext.acquire_semaphore = asyncio.Semaphore(25)

        logger = logging.getLogger(__name__)
        logger.info("✓ Applied fnllm concurrency fix: LimitContext.acquire_semaphore = Semaphore(25)")
    except ImportError:
        # fnllm not installed - GraphRAG not available, skip fix
        pass
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to apply fnllm concurrency fix: {e}")
    # ============================================================================

    config = get_config()
    setup_logging(config)

    # Only run migrations if this service is designated to handle them
    should_run_migrations = os.environ.get('RUN_MIGRATIONS', 'false').lower() == 'true'
    logger = logging.getLogger(__name__)

    if should_run_migrations:
        logger.info("Running database table creation and migrations...")
        create_tables()
    else:
        logger.info("Skipping database migrations - not designated migration runner")

    # Test cache availability
    cache = get_cache()
    if not cache.ping():
        logger.warning("Redis cache is not available")

    if config.rag.cache.enabled and config.rag.cache.warmup_collections:
        storage = get_storage()
        cache = GraphRAGDataFrameCache(config)
        await cache.warmup_cache(storage)


@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on application shutdown."""
    # No cleanup needed for simple cache
    pass


@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "timestamp": __import__("time").time(),
    }


@app.get("/health/database")
def database_health_check():
    """Database and migration status check."""
    import os

    # Check if this service handles migrations
    should_run_migrations = os.environ.get('RUN_MIGRATIONS', 'false').lower() == 'true'

    if not should_run_migrations:
        return {
            "status": "ok",
            "migration_status": "skipped",
            "message": "Migration management not enabled for this service",
            "timestamp": __import__("time").time(),
        }

    try:
        from fileintel.storage.postgresql_storage import PostgreSQLStorage
        import sys
        from pathlib import Path

        # Add scripts to path for migration manager
        scripts_path = Path(__file__).parent.parent.parent.parent / "scripts"
        if str(scripts_path) not in sys.path:
            sys.path.append(str(scripts_path))

        from fileintel.migration_manager import MigrationManager

        config = get_config()
        storage = PostgreSQLStorage(config)
        try:
            manager = MigrationManager(storage)

            migration_status = manager.get_migration_status()

            return {
                "status": "ok",
                "migration_status": migration_status.get("status", "unknown"),
                "current_version": migration_status.get("current_version"),
                "pending_migrations": migration_status.get("total_pending", 0),
                "timestamp": __import__("time").time(),
            }
        finally:
            storage.close()
    except Exception as e:
        return {
            "status": "error",
            "migration_status": "error",
            "error": str(e),
            "timestamp": __import__("time").time(),
        }


@app.get("/health/celery")
def celery_health_check():
    """Celery worker status check."""
    try:
        from fileintel.celery_config import get_worker_stats, get_active_tasks

        worker_stats = get_worker_stats()
        active_tasks = get_active_tasks()

        return {
            "status": "ok",
            "available": True,
            "workers": len(worker_stats) if worker_stats else 0,
            "active_tasks": sum(len(tasks) for tasks in active_tasks.values())
            if active_tasks
            else 0,
            "timestamp": __import__("time").time(),
        }
    except ImportError as e:
        return {
            "status": "error",
            "available": False,
            "error": f"Celery not available: {e}",
            "timestamp": __import__("time").time(),
        }
    except Exception as e:
        return {
            "status": "error",
            "available": False,
            "error": f"Celery error: {e}",
            "timestamp": __import__("time").time(),
        }
