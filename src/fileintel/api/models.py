"""
API v2 models for task-based operations.

These models support the new Celery-based task system replacing the custom job infrastructure.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class TaskState(str, Enum):
    """Celery task states"""

    PENDING = "PENDING"
    RECEIVED = "RECEIVED"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"
    PROGRESS = "PROGRESS"


class TaskSubmissionRequest(BaseModel):
    """Request model for submitting collection processing tasks."""

    collection_id: str = Field(..., description="Collection identifier")
    operation_type: str = Field(
        default="complete_analysis", description="Type of processing operation"
    )
    build_graph: bool = Field(
        default=True, description="Whether to build GraphRAG index"
    )
    extract_metadata: bool = Field(
        default=True, description="Whether to extract document metadata"
    )
    generate_embeddings: bool = Field(
        default=True, description="Whether to generate embeddings"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional task parameters"
    )


class GenericTaskSubmissionRequest(BaseModel):
    """Request model for submitting generic Celery tasks."""

    task_name: str = Field(..., description="Name of the Celery task to execute")
    args: List[Any] = Field(
        default_factory=list, description="Positional arguments for the task"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the task"
    )
    queue: Optional[str] = Field(None, description="Queue to submit task to")
    countdown: Optional[int] = Field(
        None, description="Delay before execution in seconds"
    )
    eta: Optional[datetime] = Field(
        None, description="Estimated time of arrival for execution"
    )


class DocumentProcessingRequest(BaseModel):
    """Request model for processing individual documents."""

    file_paths: List[str] = Field(..., description="List of file paths to process")
    document_ids: Optional[List[str]] = Field(
        None, description="Optional document identifiers"
    )
    chunk_size: Optional[int] = Field(
        None, description="Chunk size for text processing"
    )
    chunk_overlap: Optional[int] = Field(None, description="Overlap between chunks")


class TaskSubmissionResponse(BaseModel):
    """Response model for task submission."""

    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task submitted")
    status: TaskState = Field(..., description="Current task status")
    submitted_at: datetime = Field(..., description="Task submission timestamp")
    estimated_duration: Optional[int] = Field(
        None, description="Estimated duration in seconds"
    )
    collection_id: Optional[str] = Field(None, description="Associated collection ID")


class TaskProgressInfo(BaseModel):
    """Progress information for running tasks."""

    current: int = Field(..., description="Current progress value")
    total: int = Field(..., description="Total progress value")
    percentage: float = Field(..., description="Progress percentage")
    message: str = Field(default="", description="Progress message")
    timestamp: float = Field(..., description="Progress update timestamp")


class TaskStatusResponse(BaseModel):
    """Response model for task status queries."""

    task_id: str = Field(..., description="Task identifier")
    task_name: str = Field(..., description="Name of the task")
    status: TaskState = Field(..., description="Current task status")
    result: Optional[Dict[str, Any]] = Field(
        None, description="Task result if completed"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[TaskProgressInfo] = Field(
        None, description="Progress information"
    )
    started_at: Optional[datetime] = Field(None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Task completion timestamp"
    )
    worker_id: Optional[str] = Field(None, description="Worker that processed the task")
    retry_count: int = Field(default=0, description="Number of retry attempts")


class TaskListResponse(BaseModel):
    """Response model for listing tasks."""

    tasks: List[TaskStatusResponse] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks")
    limit: int = Field(..., description="Result limit")
    offset: int = Field(default=0, description="Result offset")


class TaskOperationRequest(BaseModel):
    """Generic request model for task operations (cancel, retry, etc)."""

    terminate: bool = Field(
        default=False, description="Whether to terminate running task"
    )
    reason: Optional[str] = Field(None, description="Operation reason")


class TaskOperationResponse(BaseModel):
    """Generic response model for task operations."""

    task_id: str = Field(..., description="Task identifier")
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Operation result message")
    timestamp: datetime = Field(..., description="Operation timestamp")


class BatchCancelRequest(BaseModel):
    """Request model for batch task cancellation."""

    task_ids: List[str] = Field(..., description="List of task IDs to cancel")
    terminate: bool = Field(default=False, description="Whether to terminate tasks")


class BatchTaskSubmissionRequest(BaseModel):
    """Request model for submitting multiple tasks."""

    tasks: List[TaskSubmissionRequest] = Field(
        ..., description="List of tasks to submit"
    )
    workflow_type: str = Field(
        default="parallel", description="Workflow execution type"
    )
    callback_url: Optional[str] = Field(
        None, description="Webhook URL for completion notification"
    )


class BatchTaskSubmissionResponse(BaseModel):
    """Response model for batch task submission."""

    batch_id: str = Field(..., description="Batch identifier")
    task_ids: List[str] = Field(..., description="Individual task identifiers")
    submitted_count: int = Field(..., description="Number of tasks submitted")
    failed_count: int = Field(default=0, description="Number of tasks that failed to submit")
    failures: List[dict] = Field(default_factory=list, description="Details of failed submissions")
    workflow_type: str = Field(..., description="Workflow execution type")
    estimated_duration: Optional[int] = Field(
        None, description="Estimated total duration"
    )


# Removed unused specialized request models - functionality covered by TaskSubmissionRequest with parameters


class WebSocketEventType(str, Enum):
    """WebSocket event types for task monitoring."""

    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRY = "task_retry"
    WORKER_ONLINE = "worker_online"
    WORKER_OFFLINE = "worker_offline"


class WebSocketTaskEvent(BaseModel):
    """WebSocket event model for task updates."""

    event_type: WebSocketEventType = Field(..., description="Event type")
    task_id: str = Field(..., description="Task identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    worker_id: Optional[str] = Field(None, description="Worker identifier")


class TaskMetricsResponse(BaseModel):
    """Response model for task metrics."""

    active_tasks: int = Field(..., description="Number of active tasks")
    pending_tasks: int = Field(..., description="Number of pending tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    average_task_duration: Optional[float] = Field(
        None, description="Average task duration in seconds"
    )
    worker_count: int = Field(..., description="Number of active workers")
    queue_lengths: Dict[str, int] = Field(
        default_factory=dict, description="Length of each queue"
    )


class ApiResponseV2(BaseModel):
    """Standard API response wrapper for v2 endpoints."""

    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        None, description="Response data"
    )
    error: Optional[str] = Field(None, description="Error message if unsuccessful")
    timestamp: datetime = Field(..., description="Response timestamp")
    api_version: str = Field(default="2.0", description="API version")


# Removed HealthCheckV2 - health checks handled by main app endpoints
