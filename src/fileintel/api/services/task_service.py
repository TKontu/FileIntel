"""
Task monitoring and management service.

Extracts Celery task management logic from API route handlers to improve
maintainability and follow the Single Responsibility Principle.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from celery.result import AsyncResult
from celery.exceptions import WorkerLostError, Retry

from fileintel.api.models import TaskState, TaskProgressInfo

logger = logging.getLogger(__name__)


class TaskService:
    """Service class for task monitoring and management."""

    def __init__(self):
        """Initialize task service."""
        pass

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get comprehensive task status information."""
        try:
            result = AsyncResult(task_id)

            # Map Celery state to our TaskState enum
            task_state = self._map_celery_state_to_task_state(result.state)

            # Get task info
            task_info = {
                "task_id": task_id,
                "state": task_state.value,
                "status": result.status,  # Raw Celery status
            }

            # Add result or error info
            if result.ready():
                if result.successful():
                    task_info["result"] = self._format_task_result(result.result)
                else:
                    task_info["error"] = (
                        str(result.info) if result.info else "Unknown error"
                    )

            # Add progress info if available
            if hasattr(result, "info") and isinstance(result.info, dict):
                progress = self._extract_progress_info(result.info)
                if progress:
                    task_info["progress"] = progress.dict()

            # Add timing information
            task_info.update(self._get_task_timing_info(result))

            return task_info

        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {e}")
            return {
                "task_id": task_id,
                "state": TaskState.FAILURE.value,
                "error": f"Failed to get task status: {str(e)}",
            }

    def list_tasks(
        self, state_filter: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        """List tasks with optional filtering."""
        try:
            from celery import current_app

            # Get active tasks
            inspect = current_app.control.inspect()

            # Collect tasks from different states
            all_tasks = []

            # Active tasks
            if not state_filter or state_filter.upper() in ["PENDING", "RUNNING"]:
                active_tasks = inspect.active()
                if active_tasks:
                    for worker, tasks in active_tasks.items():
                        for task in tasks:
                            all_tasks.append(
                                {
                                    "task_id": task.get("id"),
                                    "state": TaskState.RUNNING.value,
                                    "name": task.get("name"),
                                    "worker": worker,
                                    "args": task.get("args", []),
                                    "kwargs": task.get("kwargs", {}),
                                    "time_start": task.get("time_start"),
                                }
                            )

            # Scheduled tasks
            if not state_filter or state_filter.upper() == "PENDING":
                scheduled_tasks = inspect.scheduled()
                if scheduled_tasks:
                    for worker, tasks in scheduled_tasks.items():
                        for task in tasks:
                            all_tasks.append(
                                {
                                    "task_id": task.get("request", {}).get("id"),
                                    "state": TaskState.PENDING.value,
                                    "name": task.get("request", {}).get("task"),
                                    "worker": worker,
                                    "eta": task.get("eta"),
                                }
                            )

            # Apply pagination
            total_count = len(all_tasks)
            paginated_tasks = all_tasks[offset : offset + limit]

            return {
                "tasks": paginated_tasks,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count,
            }

        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return {
                "tasks": [],
                "total_count": 0,
                "error": f"Failed to list tasks: {str(e)}",
            }

    def cancel_task(self, task_id: str, terminate: bool = False) -> Dict[str, Any]:
        """Cancel a specific task."""
        try:
            from celery import current_app

            result = AsyncResult(task_id)

            if result.ready():
                return {
                    "task_id": task_id,
                    "status": "already_completed",
                    "message": "Task has already completed and cannot be cancelled",
                }

            # Revoke the task
            current_app.control.revoke(task_id, terminate=terminate)

            logger.info(f"Task {task_id} cancelled (terminate={terminate})")

            return {
                "task_id": task_id,
                "status": "cancelled",
                "terminate": terminate,
                "message": "Task cancellation requested",
            }

        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "error",
                "error": f"Failed to cancel task: {str(e)}",
            }

    def get_task_result(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get task result with optional timeout."""
        try:
            result = AsyncResult(task_id)

            if not result.ready():
                return {
                    "task_id": task_id,
                    "ready": False,
                    "state": self._map_celery_state_to_task_state(result.state).value,
                    "message": "Task is still running",
                }

            if result.successful():
                return {
                    "task_id": task_id,
                    "ready": True,
                    "successful": True,
                    "result": self._format_task_result(result.result),
                }
            else:
                return {
                    "task_id": task_id,
                    "ready": True,
                    "successful": False,
                    "error": str(result.info) if result.info else "Unknown error",
                }

        except Exception as e:
            logger.error(f"Error getting task result for {task_id}: {e}")
            return {"task_id": task_id, "error": f"Failed to get task result: {str(e)}"}

    def get_worker_metrics(self) -> Dict[str, Any]:
        """Get Celery worker metrics and statistics."""
        try:
            from celery import current_app

            inspect = current_app.control.inspect()

            # Get worker statistics
            stats = inspect.stats()
            active = inspect.active()
            scheduled = inspect.scheduled()
            reserved = inspect.reserved()

            worker_metrics = {}

            if stats:
                for worker_name, worker_stats in stats.items():
                    worker_metrics[worker_name] = {
                        "status": "online",
                        "pool": worker_stats.get("pool", {}),
                        "total_tasks": worker_stats.get("total", {}),
                        "rusage": worker_stats.get("rusage", {}),
                        "clock": worker_stats.get("clock"),
                        "active_tasks": len(active.get(worker_name, []))
                        if active
                        else 0,
                        "scheduled_tasks": len(scheduled.get(worker_name, []))
                        if scheduled
                        else 0,
                        "reserved_tasks": len(reserved.get(worker_name, []))
                        if reserved
                        else 0,
                    }

            # Calculate totals
            total_active = sum(len(tasks) for tasks in (active or {}).values())
            total_scheduled = sum(len(tasks) for tasks in (scheduled or {}).values())
            total_reserved = sum(len(tasks) for tasks in (reserved or {}).values())

            return {
                "workers": worker_metrics,
                "summary": {
                    "total_workers": len(worker_metrics),
                    "total_active_tasks": total_active,
                    "total_scheduled_tasks": total_scheduled,
                    "total_reserved_tasks": total_reserved,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting worker metrics: {e}")
            return {
                "error": f"Failed to get worker metrics: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            }

    def batch_cancel_tasks(
        self, task_ids: List[str], terminate: bool = False
    ) -> Dict[str, Any]:
        """Cancel multiple tasks in batch."""
        try:
            from celery import current_app

            results = {}

            for task_id in task_ids:
                try:
                    result = AsyncResult(task_id)

                    if result.ready():
                        results[task_id] = {
                            "status": "already_completed",
                            "message": "Task has already completed",
                        }
                    else:
                        current_app.control.revoke(task_id, terminate=terminate)
                        results[task_id] = {
                            "status": "cancelled",
                            "terminate": terminate,
                        }

                except Exception as e:
                    results[task_id] = {"status": "error", "error": str(e)}

            logger.info(
                f"Batch cancelled {len(task_ids)} tasks (terminate={terminate})"
            )

            return {
                "total_requested": len(task_ids),
                "results": results,
                "summary": self._summarize_batch_results(results),
            }

        except Exception as e:
            logger.error(f"Error in batch cancel: {e}")
            return {"error": f"Failed to batch cancel tasks: {str(e)}"}

    def retry_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """Retry a failed task."""
        try:
            result = AsyncResult(task_id)

            if not result.ready():
                return {
                    "task_id": task_id,
                    "status": "error",
                    "message": "Cannot retry a task that is still running",
                }

            if result.successful():
                return {
                    "task_id": task_id,
                    "status": "error",
                    "message": "Cannot retry a successful task",
                }

            # Get original task info
            task_name = getattr(result, "name", None)
            if not task_name:
                return {
                    "task_id": task_id,
                    "status": "error",
                    "message": "Cannot determine original task type for retry",
                }

            # Note: This is a placeholder - actual retry implementation
            # would need access to the original task function and arguments
            logger.warning(f"Task retry requested for {task_id} but not implemented")

            return {
                "task_id": task_id,
                "status": "not_implemented",
                "message": "Task retry feature is not yet implemented",
            }

        except Exception as e:
            logger.error(f"Error retrying task {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "error",
                "error": f"Failed to retry task: {str(e)}",
            }

    def _map_celery_state_to_task_state(self, celery_state: str) -> TaskState:
        """Map Celery state to our TaskState enum."""
        mapping = {
            "PENDING": TaskState.PENDING,
            "STARTED": TaskState.RUNNING,
            "RETRY": TaskState.RUNNING,
            "SUCCESS": TaskState.SUCCESS,
            "FAILURE": TaskState.FAILURE,
            "REVOKED": TaskState.CANCELLED,
        }
        return mapping.get(celery_state, TaskState.PENDING)

    def _format_task_result(self, result: Any) -> Optional[Dict[str, Any]]:
        """Format task result for API response."""
        if result is None:
            return None

        if isinstance(result, dict):
            return result

        if isinstance(result, (str, int, float, bool)):
            return {"value": result}

        # Try to convert to dict
        try:
            return {"result": str(result)}
        except Exception:
            return {"result": "Unable to serialize result"}

    def _extract_progress_info(
        self, task_info: Dict[str, Any]
    ) -> Optional[TaskProgressInfo]:
        """Extract progress information from task info."""
        try:
            if "progress" in task_info:
                progress_data = task_info["progress"]
                return TaskProgressInfo(
                    current=progress_data.get("current", 0),
                    total=progress_data.get("total", 0),
                    status=progress_data.get("status", "Processing..."),
                    stage=progress_data.get("stage"),
                    percentage=progress_data.get("percentage"),
                )
        except Exception as e:
            logger.debug(f"Could not extract progress info: {e}")

        return None

    def _get_task_timing_info(self, result: AsyncResult) -> Dict[str, Any]:
        """Extract timing information from task result."""
        timing_info = {}

        try:
            # This would need to be enhanced based on how timing info is stored
            # in your Celery setup
            if hasattr(result, "date_done") and result.date_done:
                timing_info["completed_at"] = result.date_done.isoformat()
        except Exception as e:
            logger.debug(f"Could not extract timing info: {e}")

        return timing_info

    def _summarize_batch_results(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Summarize batch operation results."""
        summary = {"cancelled": 0, "already_completed": 0, "errors": 0}

        for task_result in results.values():
            status = task_result.get("status", "error")
            if status == "cancelled":
                summary["cancelled"] += 1
            elif status == "already_completed":
                summary["already_completed"] += 1
            else:
                summary["errors"] += 1

        return summary
