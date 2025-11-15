"""
Health check endpoints for API monitoring.

Provides comprehensive health status for all system components.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..health import get_health_service, HealthStatus
from ..database import get_db
from ..error_handlers import create_success_response
from ..models import ApiResponseV2

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=ApiResponseV2)
async def health_check(db: Session = Depends(get_db)) -> ApiResponseV2:
    """
    Comprehensive health check for all system components.

    Checks:
    - Database connectivity
    - Redis/Celery result backend
    - Celery workers
    - Vector store accessibility

    Returns HTTP 200 even if some components are degraded,
    but includes status details in response.
    """
    health_service = get_health_service()
    health_status = health_service.check_all(session=db)

    return create_success_response(health_status)


@router.get("/health/live", response_model=ApiResponseV2)
async def liveness_check() -> ApiResponseV2:
    """
    Kubernetes liveness probe - checks if API server is running.

    Returns 200 if the API process is alive.
    This should almost always succeed unless the server is completely down.
    """
    return create_success_response({
        "status": "alive",
        "message": "API server is running"
    })


@router.get("/health/ready", response_model=ApiResponseV2)
async def readiness_check(db: Session = Depends(get_db)) -> ApiResponseV2:
    """
    Kubernetes readiness probe - checks if API is ready to serve traffic.

    Checks critical dependencies (DB, Redis) and returns 503 if unhealthy.
    Use this for load balancer health checks.
    """
    from fastapi import HTTPException

    health_service = get_health_service()

    # Check only critical components
    db_health = health_service.check_database(db)
    redis_health = health_service.check_redis()
    celery_health = health_service.check_celery_workers()

    critical_components = {
        "database": db_health,
        "redis": redis_health,
        "celery_workers": celery_health
    }

    # If any critical component is unhealthy, return 503
    for name, health in critical_components.items():
        if health.status == HealthStatus.UNHEALTHY:
            raise HTTPException(
                status_code=503,
                detail=f"Service unavailable: {name} is unhealthy - {health.message}"
            )

    return create_success_response({
        "status": "ready",
        "message": "API is ready to serve traffic",
        "components": {
            name: health.to_dict()
            for name, health in critical_components.items()
        }
    })


@router.get("/health/metrics", response_model=ApiResponseV2)
async def health_metrics(db: Session = Depends(get_db)) -> ApiResponseV2:
    """
    Detailed health metrics for monitoring systems.

    Returns detailed information about all components including
    response times, versions, and resource usage.
    """
    health_service = get_health_service()
    metrics = health_service.check_all(session=db)

    # Add API-level metrics
    import psutil
    import os

    process = psutil.Process(os.getpid())

    metrics["api_metrics"] = {
        "memory_usage_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "cpu_percent": process.cpu_percent(interval=0.1),
        "open_files": len(process.open_files()),
        "threads": process.num_threads(),
    }

    # Add circuit breaker stats
    from ..circuit_breaker import get_all_circuit_breakers
    metrics["circuit_breakers"] = get_all_circuit_breakers()

    return create_success_response(metrics)
