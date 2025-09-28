"""
Tasks API v2 - Simplified using service layer.

Refactored to use TaskService for business logic,
following Single Responsibility Principle.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from fileintel.api.dependencies import get_api_key
from fileintel.api.error_handlers import api_error_handler, create_success_response
from fileintel.api.models import ApiResponseV2, TaskSubmissionRequest
from fileintel.api.services import TaskService

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_api_key)])


class BatchCancelRequest(BaseModel):
    task_ids: List[str]
    terminate: bool = False


def get_task_service() -> TaskService:
    """Dependency to get TaskService."""
    return TaskService()


@router.get("/tasks/{task_id}/status", response_model=ApiResponseV2)
@api_error_handler("get task status")
async def get_task_status(
    task_id: str, service: TaskService = Depends(get_task_service)
) -> ApiResponseV2:
    """Get comprehensive status of a specific task."""
    status = service.get_task_status(task_id)
    return create_success_response(status)


@router.get("/tasks", response_model=ApiResponseV2)
@api_error_handler("list tasks")
async def list_tasks(
    state: Optional[str] = Query(None, description="Filter tasks by state"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of tasks to return"
    ),
    offset: int = Query(0, ge=0, description="Number of tasks to skip"),
    service: TaskService = Depends(get_task_service),
) -> ApiResponseV2:
    """List tasks with optional filtering and pagination."""
    result = service.list_tasks(state_filter=state, limit=limit, offset=offset)
    return create_success_response(result)


@router.post("/tasks/{task_id}/cancel", response_model=ApiResponseV2)
@api_error_handler("cancel task")
async def cancel_task(
    task_id: str,
    terminate: bool = Query(
        False, description="Whether to terminate the task forcefully"
    ),
    service: TaskService = Depends(get_task_service),
) -> ApiResponseV2:
    """Cancel a specific task."""
    result = service.cancel_task(task_id, terminate=terminate)
    return create_success_response(result)


@router.get("/tasks/{task_id}/result", response_model=ApiResponseV2)
@api_error_handler("get task result")
async def get_task_result(
    task_id: str,
    timeout: Optional[float] = Query(None, description="Timeout in seconds"),
    service: TaskService = Depends(get_task_service),
) -> ApiResponseV2:
    """Get result of a completed task."""
    result = service.get_task_result(task_id, timeout=timeout)
    return create_success_response(result)


@router.get("/tasks/metrics", response_model=ApiResponseV2)
@api_error_handler("get task metrics")
async def get_task_metrics(
    service: TaskService = Depends(get_task_service),
) -> ApiResponseV2:
    """Get Celery worker metrics and task statistics."""
    metrics = service.get_worker_metrics()
    return create_success_response(metrics)


@router.get("/tasks/{task_id}/logs", response_model=ApiResponseV2)
@api_error_handler("get task logs")
async def get_task_logs(
    task_id: str,
    lines: int = Query(
        100, ge=1, le=10000, description="Number of log lines to return"
    ),
    service: TaskService = Depends(get_task_service),
) -> ApiResponseV2:
    """Get logs for a specific task."""
    # Note: This is a placeholder implementation
    # Actual log retrieval would depend on your logging setup
    logger.warning(f"Task logs requested for {task_id} but not implemented")

    return create_success_response(
        {
            "task_id": task_id,
            "logs": [],
            "message": "Task logging feature not yet implemented",
            "lines_requested": lines,
        }
    )


@router.post("/tasks/batch/cancel", response_model=ApiResponseV2)
@api_error_handler("batch cancel tasks")
async def batch_cancel_tasks(
    request: BatchCancelRequest, service: TaskService = Depends(get_task_service)
) -> ApiResponseV2:
    """Cancel multiple tasks in batch."""
    if not request.task_ids:
        raise HTTPException(status_code=400, detail="No task IDs provided")

    if len(request.task_ids) > 100:
        raise HTTPException(status_code=400, detail="Too many task IDs (max 100)")

    result = service.batch_cancel_tasks(request.task_ids, terminate=request.terminate)
    return create_success_response(result)


@router.post("/tasks/{task_id}/retry", response_model=ApiResponseV2)
@api_error_handler("retry task")
async def retry_task(
    task_id: str, service: TaskService = Depends(get_task_service)
) -> ApiResponseV2:
    """Retry a failed task."""
    result = service.retry_task(task_id)
    return create_success_response(result)


@router.post("/tasks/submit", response_model=ApiResponseV2)
@api_error_handler("submit task")
async def submit_task(
    request: TaskSubmissionRequest, service: TaskService = Depends(get_task_service)
) -> ApiResponseV2:
    """Submit a new task for processing."""
    # Note: This is a placeholder for general task submission
    # Specific task types should be submitted through their dedicated endpoints
    logger.warning("Generic task submission not implemented")

    return create_success_response(
        {
            "message": "Generic task submission not implemented",
            "suggestion": "Use specific endpoints like /collections/{id}/process for task submission",
        }
    )
