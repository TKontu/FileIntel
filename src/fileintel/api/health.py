"""
Health check system for FileIntel API.

Monitors critical dependencies with timeouts to prevent cascading failures.
"""

import logging
import time
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """Health status for a single component."""

    def __init__(
        self,
        status: HealthStatus,
        message: str = "",
        response_time_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.status = status
        self.message = message
        self.response_time_ms = response_time_ms
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "status": self.status.value,
            "message": self.message,
        }
        if self.response_time_ms is not None:
            result["response_time_ms"] = round(self.response_time_ms, 2)
        if self.details:
            result["details"] = self.details
        return result


class HealthCheckService:
    """
    Service for checking health of all system components.

    Includes timeouts to prevent blocking on unhealthy dependencies.
    """

    def __init__(self):
        self.check_timeout = 5.0  # seconds
        self.cache_ttl = 10  # Cache health results for 10 seconds to prevent overload
        self._cache: Dict[str, tuple[ComponentHealth, datetime]] = {}

    def check_database(self, session) -> ComponentHealth:
        """Check PostgreSQL database connectivity."""
        cache_key = "database"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        start = time.time()
        try:
            # Simple query with timeout
            from sqlalchemy import text
            result = session.execute(text("SELECT 1")).fetchone()

            if result and result[0] == 1:
                response_time = (time.time() - start) * 1000
                health = ComponentHealth(
                    HealthStatus.HEALTHY,
                    "Database connection successful",
                    response_time
                )
            else:
                health = ComponentHealth(
                    HealthStatus.UNHEALTHY,
                    "Database query returned unexpected result"
                )
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            health = ComponentHealth(
                HealthStatus.UNHEALTHY,
                f"Database connection failed: {str(e)[:100]}"
            )

        self._set_cache(cache_key, health)
        return health

    def check_redis(self) -> ComponentHealth:
        """Check Redis connectivity with timeout."""
        cache_key = "redis"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        start = time.time()
        try:
            from fileintel.celery_config import get_celery_app
            app = get_celery_app()

            if not app:
                return ComponentHealth(
                    HealthStatus.UNHEALTHY,
                    "Celery app not available"
                )

            # Get Redis backend URL
            backend_url = app.conf.result_backend

            # Try to connect to Redis with timeout
            import redis
            from urllib.parse import urlparse

            parsed = urlparse(backend_url)
            r = redis.Redis(
                host=parsed.hostname or 'localhost',
                port=parsed.port or 6379,
                db=int(parsed.path.lstrip('/')) if parsed.path else 0,
                socket_connect_timeout=self.check_timeout,
                socket_timeout=self.check_timeout,
                decode_responses=True
            )

            # Ping test
            pong = r.ping()
            response_time = (time.time() - start) * 1000

            # Get Redis info for details
            info = r.info('server')

            if pong:
                health = ComponentHealth(
                    HealthStatus.HEALTHY,
                    "Redis connection successful",
                    response_time,
                    {
                        "redis_version": info.get('redis_version'),
                        "used_memory_human": info.get('used_memory_human')
                    }
                )
            else:
                health = ComponentHealth(
                    HealthStatus.UNHEALTHY,
                    "Redis ping failed"
                )
        except redis.TimeoutError:
            health = ComponentHealth(
                HealthStatus.UNHEALTHY,
                f"Redis connection timeout ({self.check_timeout}s)"
            )
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            health = ComponentHealth(
                HealthStatus.UNHEALTHY,
                f"Redis connection failed: {str(e)[:100]}"
            )

        self._set_cache(cache_key, health)
        return health

    def check_celery_workers(self) -> ComponentHealth:
        """Check if Celery workers are active."""
        cache_key = "celery_workers"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        start = time.time()
        try:
            from fileintel.celery_config import get_celery_app
            app = get_celery_app()

            if not app:
                return ComponentHealth(
                    HealthStatus.UNHEALTHY,
                    "Celery app not available"
                )

            # Get worker stats with timeout
            inspect = app.control.inspect(timeout=self.check_timeout)
            stats = inspect.stats()

            response_time = (time.time() - start) * 1000

            if stats:
                worker_count = len(stats)
                health = ComponentHealth(
                    HealthStatus.HEALTHY,
                    f"{worker_count} worker(s) active",
                    response_time,
                    {"workers": list(stats.keys())}
                )
            else:
                health = ComponentHealth(
                    HealthStatus.UNHEALTHY,
                    "No Celery workers responding"
                )
        except Exception as e:
            logger.error(f"Celery worker health check failed: {e}")
            health = ComponentHealth(
                HealthStatus.UNHEALTHY,
                f"Worker check failed: {str(e)[:100]}"
            )

        self._set_cache(cache_key, health)
        return health

    def check_vector_store(self) -> ComponentHealth:
        """Check LanceDB vector store accessibility."""
        cache_key = "vector_store"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        start = time.time()
        try:
            from fileintel.core.config import get_config
            import os

            config = get_config()
            db_path = config.lancedb.db_path

            # Check if directory exists and is accessible
            if os.path.exists(db_path) and os.access(db_path, os.R_OK | os.W_OK):
                response_time = (time.time() - start) * 1000

                # Get directory size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(db_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)

                size_mb = total_size / (1024 * 1024)

                health = ComponentHealth(
                    HealthStatus.HEALTHY,
                    "Vector store accessible",
                    response_time,
                    {"db_path": db_path, "size_mb": round(size_mb, 2)}
                )
            else:
                health = ComponentHealth(
                    HealthStatus.DEGRADED,
                    f"Vector store path not accessible: {db_path}"
                )
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            health = ComponentHealth(
                HealthStatus.DEGRADED,  # Degraded not unhealthy - API can work without it
                f"Vector store check failed: {str(e)[:100]}"
            )

        self._set_cache(cache_key, health)
        return health

    def check_all(self, session=None) -> Dict[str, Any]:
        """
        Check health of all components.

        Returns overall health status and individual component statuses.
        """
        components = {}

        # Check each component
        if session:
            components["database"] = self.check_database(session).to_dict()

        components["redis"] = self.check_redis().to_dict()
        components["celery_workers"] = self.check_celery_workers().to_dict()
        components["vector_store"] = self.check_vector_store().to_dict()

        # Determine overall status
        statuses = [comp["status"] for comp in components.values()]

        if all(s == "healthy" for s in statuses):
            overall_status = HealthStatus.HEALTHY
            overall_message = "All systems operational"
        elif any(s == "unhealthy" for s in statuses):
            unhealthy_components = [
                name for name, comp in components.items()
                if comp["status"] == "unhealthy"
            ]
            overall_status = HealthStatus.UNHEALTHY
            overall_message = f"Critical components unhealthy: {', '.join(unhealthy_components)}"
        else:
            degraded_components = [
                name for name, comp in components.items()
                if comp["status"] == "degraded"
            ]
            overall_status = HealthStatus.DEGRADED
            overall_message = f"Some components degraded: {', '.join(degraded_components)}"

        return {
            "status": overall_status.value,
            "message": overall_message,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components
        }

    def _get_cached(self, key: str) -> Optional[ComponentHealth]:
        """Get cached health result if still valid."""
        if key in self._cache:
            health, cached_at = self._cache[key]
            if datetime.utcnow() - cached_at < timedelta(seconds=self.cache_ttl):
                return health
        return None

    def _set_cache(self, key: str, health: ComponentHealth):
        """Cache health result."""
        self._cache[key] = (health, datetime.utcnow())


# Global health check service instance
_health_service = None


def get_health_service() -> HealthCheckService:
    """Get or create the health check service singleton."""
    global _health_service
    if _health_service is None:
        _health_service = HealthCheckService()
    return _health_service
