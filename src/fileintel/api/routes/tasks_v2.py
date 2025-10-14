"""
Tasks API v2 - Direct Celery task monitoring and control.

Provides direct access to Celery task status, cancellation, and monitoring.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from celery.result import AsyncResult
from celery import current_app

from ..dependencies import get_api_key
from ..error_handlers import celery_error_handler, create_success_response
from ..models import (
    TaskStatusResponse,
    TaskListResponse,
    TaskOperationRequest,
    TaskOperationResponse,
    TaskProgressInfo,
    TaskMetricsResponse,
    ApiResponseV2,
    TaskState,
    GenericTaskSubmissionRequest,
    TaskSubmissionResponse,
    BatchCancelRequest,
)

# Import Celery functions with error handling for worker availability
try:
    from fileintel.celery_config import (
        get_celery_app,
        get_task_status,
        cancel_task,
        get_active_tasks,
        get_worker_stats,
    )

    CELERY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Celery configuration not available: {e}")
    CELERY_AVAILABLE = False

    # Provide stub functions to prevent import errors
    def get_celery_app():
        return None

    def get_task_status(task_id):
        raise HTTPException(503, "Celery not available")

    def cancel_task(task_id, terminate=False):
        raise HTTPException(503, "Celery not available")

    def get_active_tasks():
        return {}

    def get_worker_stats():
        return {}


logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_api_key)])


def _map_celery_state_to_task_state(celery_state: str) -> TaskState:
    """Map Celery task states to our TaskState enum."""
    mapping = {
        "PENDING": TaskState.PENDING,
        "RECEIVED": TaskState.RECEIVED,
        "STARTED": TaskState.STARTED,
        "SUCCESS": TaskState.SUCCESS,
        "FAILURE": TaskState.FAILURE,
        "RETRY": TaskState.RETRY,
        "REVOKED": TaskState.REVOKED,
        "PROGRESS": TaskState.PROGRESS,
    }
    return mapping.get(celery_state, TaskState.PENDING)


def _format_task_result(result: Any) -> Optional[Dict[str, Any]]:
    """Format task result for API response."""
    if result is None:
        return None

    if isinstance(result, dict):
        return result

    try:
        # Try to convert to dict if possible
        return {"result": str(result)}
    except (TypeError, ValueError):
        return {"result": "Unable to format result"}


def _extract_progress_info(task_info: Dict[str, Any]) -> Optional[TaskProgressInfo]:
    """Extract progress information from Celery task info."""
    if task_info.get("state") == "PROGRESS" and "result" in task_info:
        progress_data = task_info["result"]
        if isinstance(progress_data, dict):
            return TaskProgressInfo(
                current=progress_data.get("current", 0),
                total=progress_data.get("total", 1),
                percentage=progress_data.get("percentage", 0.0),
                message=progress_data.get("message", ""),
                timestamp=progress_data.get("timestamp", datetime.utcnow().timestamp()),
            )
    return None


@router.get("/tasks/{task_id}/status", response_model=ApiResponseV2)
@celery_error_handler("get task status")
async def get_task_status_endpoint(task_id: str) -> ApiResponseV2:
    """
    Get the current status of a specific Celery task.

    Returns detailed information including progress, results, and error information.
    """
    # Get task status from Celery
    task_info = get_task_status(task_id)

    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Extract progress information
    progress = _extract_progress_info(task_info)

    # Format the response
    status_response = TaskStatusResponse(
        task_id=task_id,
        task_name=task_info.get("name", "unknown"),
        status=_map_celery_state_to_task_state(task_info["state"]),
        result=_format_task_result(task_info.get("result"))
        if task_info["state"] == "SUCCESS"
        else None,
        error=str(task_info.get("result")) if task_info["state"] == "FAILURE" else None,
        progress=progress,
        started_at=None,  # Celery doesn't provide this easily
        completed_at=None,  # Would need to be stored separately
        worker_id=task_info.get("worker_id"),
        retry_count=task_info.get("retries", 0),
    )

    return create_success_response(status_response.dict())


@router.get("/tasks", response_model=ApiResponseV2)
@celery_error_handler("list tasks")
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by task status"),
    limit: int = Query(20, description="Maximum number of tasks to return"),
    offset: int = Query(0, description="Number of tasks to skip"),
) -> ApiResponseV2:
    """
    List active and recent Celery tasks.

    Note: This provides limited functionality compared to full task history storage.
    For complete task history, consider implementing a task result backend.
    """
    # Get active tasks from Celery
    active_tasks_data = get_active_tasks()

    tasks = []
    task_count = 0

    # Process active tasks from all workers
    for worker_name, worker_tasks in active_tasks_data.items():
        if not worker_tasks:
            continue

        for task_info in worker_tasks:
            if status and task_info.get("name", "").split(".")[-1] != status:
                continue

            task_id = task_info.get("id", "unknown")
            task_response = TaskStatusResponse(
                task_id=task_id,
                task_name=task_info.get("name", "unknown"),
                status=TaskState.STARTED,  # Active tasks are running
                result=None,
                error=None,
                progress=None,
                started_at=None,
                completed_at=None,
                worker_id=worker_name,
                retry_count=0,
            )

            tasks.append(task_response)
            task_count += 1

    # Apply pagination
    paginated_tasks = tasks[offset : offset + limit]

    list_response = TaskListResponse(
        tasks=paginated_tasks, total=task_count, limit=limit, offset=offset
    )

    return create_success_response(list_response.dict())


@router.post("/tasks/{task_id}/cancel", response_model=ApiResponseV2)
@celery_error_handler("cancel task")
async def cancel_task_endpoint(
    task_id: str, request: TaskOperationRequest
) -> ApiResponseV2:
    """
    Cancel a running Celery task.

    Supports both soft cancellation (revoke) and hard termination.
    """
    # Check if task exists first
    task_info = get_task_status(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Check if task is in a cancellable state
    if task_info["state"] in ["SUCCESS", "FAILURE", "REVOKED"]:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} is already completed ({task_info['state']}) and cannot be cancelled",
        )

    # Cancel the task
    success = cancel_task(task_id)

    if request.terminate:
        # For terminate, we need to use Celery's control API
        app = get_celery_app()
        app.control.revoke(task_id, terminate=True)

    cancellation_response = TaskOperationResponse(
        task_id=task_id,
        success=success,
        message=f"Task {'terminated' if request.terminate else 'cancelled'} successfully"
        if success
        else "Cancellation failed",
        timestamp=datetime.utcnow(),
    )

    return create_success_response(cancellation_response.dict())


@router.get("/tasks/{task_id}/result", response_model=ApiResponseV2)
@celery_error_handler("get task result")
async def get_task_result(task_id: str) -> ApiResponseV2:
    """
    Get the result of a completed task.

    Returns the actual task result data directly (not wrapped).
    """
    import json
    from celery.result import AsyncResult

    async_result = AsyncResult(task_id)

    if not async_result.ready():
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed (state: {async_result.state})"
        )

    if async_result.failed():
        raise HTTPException(
            status_code=500,
            detail=f"Task failed: {str(async_result.result)}"
        )

    result = async_result.result

    # Validate result is serializable
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        logger.error(f"Task {task_id} result not serializable: {result}")
        raise HTTPException(
            status_code=500,
            detail="Task result cannot be serialized"
        )

    # Return actual task result directly (not wrapped)
    return create_success_response(result)


@router.get("/tasks/metrics", response_model=ApiResponseV2)
async def get_task_metrics() -> ApiResponseV2:
    """
    Get comprehensive metrics about the Celery task system.

    Provides insights into system performance and queue status.
    """
    try:
        # Check if Celery is available
        if not CELERY_AVAILABLE:
            return ApiResponseV2(
                success=False,
                error="Celery task system is not available",
                timestamp=datetime.utcnow(),
            )
        # Get worker statistics
        worker_stats = get_worker_stats()
        active_tasks_data = get_active_tasks()

        # Calculate metrics
        total_workers = len(worker_stats) if worker_stats else 0
        total_active_tasks = (
            sum(len(tasks) for tasks in active_tasks_data.values())
            if active_tasks_data
            else 0
        )

        # In a real implementation, these would come from a task result backend or monitoring system
        metrics = TaskMetricsResponse(
            active_tasks=total_active_tasks,
            pending_tasks=0,  # Would need to query broker
            completed_tasks=0,  # Would need persistent storage
            failed_tasks=0,  # Would need persistent storage
            average_task_duration=None,  # Would need historical data
            worker_count=total_workers,
            queue_lengths={},  # Would need to query broker
        )

        return ApiResponseV2(
            success=True, data=metrics.dict(), timestamp=datetime.utcnow()
        )

    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error(f"Error getting task metrics: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())


@router.get("/tasks/{task_id}/logs", response_model=ApiResponseV2)
async def get_task_logs(
    task_id: str, lines: int = Query(100, description="Number of log lines to return")
) -> ApiResponseV2:
    """
    Get logs for a specific task.

    Note: This is a placeholder. Real implementation would require
    centralized logging with task correlation.
    """
    try:
        # In a real implementation, this would query centralized logs
        # For now, return a placeholder response
        return ApiResponseV2(
            success=True,
            data={
                "task_id": task_id,
                "logs": [
                    f"Log entry for task {task_id} - this would contain actual log data",
                    "Real implementation requires centralized logging setup",
                ],
                "lines_returned": 2,
                "lines_requested": lines,
            },
            timestamp=datetime.utcnow(),
        )

    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error(f"Error getting task logs for {task_id}: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())


@router.post("/tasks/batch/cancel", response_model=ApiResponseV2)
async def cancel_batch_tasks(
    request: BatchCancelRequest
) -> ApiResponseV2:
    """
    Cancel multiple tasks in batch.

    Useful for cancelling all tasks in a workflow or batch operation.
    """
    try:
        results = []
        successful_cancellations = 0

        for task_id in request.task_ids:
            try:
                # Check if task exists and is cancellable
                task_info = get_task_status(task_id)
                if not task_info:
                    results.append(
                        {
                            "task_id": task_id,
                            "success": False,
                            "message": "Task not found",
                        }
                    )
                    continue

                if task_info["state"] in ["SUCCESS", "FAILURE", "REVOKED"]:
                    results.append(
                        {
                            "task_id": task_id,
                            "success": False,
                            "message": f"Task already completed ({task_info['state']})",
                        }
                    )
                    continue

                # Cancel the task
                success = cancel_task(task_id)

                if request.terminate:
                    app = get_celery_app()
                    app.control.revoke(task_id, terminate=True)

                if success:
                    successful_cancellations += 1

                results.append(
                    {
                        "task_id": task_id,
                        "success": success,
                        "message": f"Task {'terminated' if request.terminate else 'cancelled'} successfully"
                        if success
                        else "Cancellation failed",
                    }
                )

            except (ValueError, ConnectionError, RuntimeError) as e:
                results.append(
                    {"task_id": task_id, "success": False, "message": str(e)}
                )

        return ApiResponseV2(
            success=True,
            data={
                "total_tasks": len(request.task_ids),
                "successful_cancellations": successful_cancellations,
                "failed_cancellations": len(request.task_ids) - successful_cancellations,
                "results": results,
                "summary": {
                    "cancelled": successful_cancellations,
                    "already_completed": 0,  # Could be enhanced to track this
                    "errors": len(request.task_ids) - successful_cancellations,
                },
            },
            timestamp=datetime.utcnow(),
        )

    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error(f"Error cancelling batch tasks: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())


@router.post("/tasks/{task_id}/retry", response_model=ApiResponseV2)
async def retry_task(task_id: str) -> ApiResponseV2:
    """
    Retry a failed task.

    Creates a new task with the same parameters as the failed task.
    """
    try:
        task_info = get_task_status(task_id)

        if not task_info:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if task_info["state"] != "FAILURE":
            raise HTTPException(
                status_code=400,
                detail=f"Task {task_id} is not in failed state (current state: {task_info['state']})",
            )

        # In a real implementation, you'd need to store original task parameters
        # to be able to retry with the same parameters
        # For now, return a placeholder response

        return ApiResponseV2(
            success=False,
            error="Task retry not implemented - requires storing original task parameters",
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error(f"Error retrying task {task_id}: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())


@router.post("/tasks/submit", response_model=ApiResponseV2)
async def submit_task(request: GenericTaskSubmissionRequest) -> ApiResponseV2:
    """
    Submit a generic Celery task for execution.

    Allows submission of any registered Celery task with arguments.
    """
    try:
        # Check if Celery is available
        if not CELERY_AVAILABLE:
            return ApiResponseV2(
                success=False,
                error="Celery task system is not available",
                timestamp=datetime.utcnow(),
            )

        # Get the Celery app
        app = get_celery_app()
        if not app:
            return ApiResponseV2(
                success=False,
                error="Celery application not available",
                timestamp=datetime.utcnow(),
            )

        # Build task execution options
        task_options = {}
        if request.queue:
            task_options["queue"] = request.queue
        if request.countdown:
            task_options["countdown"] = request.countdown
        if request.eta:
            task_options["eta"] = request.eta

        # Submit the task
        try:
            result = app.send_task(
                request.task_name,
                args=request.args,
                kwargs=request.kwargs,
                **task_options,
            )

            task_response = TaskSubmissionResponse(
                task_id=result.id,
                task_type=request.task_name,
                status=TaskState.PENDING,
                submitted_at=datetime.utcnow(),
            )

            return ApiResponseV2(
                success=True, data=task_response.dict(), timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Failed to submit task {request.task_name}: {e}")
            return ApiResponseV2(
                success=False,
                error=f"Task submission failed: {str(e)}",
                timestamp=datetime.utcnow(),
            )

    except Exception as e:
        logger.error(f"Error in task submission endpoint: {e}")
        return ApiResponseV2(success=False, error=str(e), timestamp=datetime.utcnow())
