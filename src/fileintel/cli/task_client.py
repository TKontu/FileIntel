"""
Simplified FileIntel API client using v2 task-based endpoints.

Replaces job-based operations with Celery task operations for clean architecture.
"""

import os
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from .constants import (
    DEFAULT_TASK_TIMEOUT,
    DEFAULT_POLL_INTERVAL,
    JSON_INDENT,
    TASK_ID_DISPLAY_LENGTH,
    PROGRESS_BAR_TOTAL,
    DEFAULT_API_PORT,
)

# V2 API configuration
API_BASE_URL_V2 = os.getenv(
    "FILEINTEL_API_BASE_URL", f"http://localhost:{DEFAULT_API_PORT}/api/v2"
)


class TaskAPIClient:
    """
    Clean FileIntel API client focused on task-based operations.

    Uses v2 task endpoints for distributed processing with Celery.
    Eliminates complex job management in favor of simple task submission and monitoring.
    """

    def __init__(self, base_url_v2: str = API_BASE_URL_V2):
        self.base_url_v2 = base_url_v2
        self.console = Console()

    def _request(
        self, method: str, endpoint: str, base_url: str = None, **kwargs: Any
    ) -> Any:
        """Make API request with proper error handling."""
        url = f"{base_url or self.base_url_v2}/{endpoint}"

        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]API request failed:[/bold red] {e}")
            raise

    def _request_raw(
        self, method: str, endpoint: str, base_url: str = None, **kwargs: Any
    ) -> requests.Response:
        """Make raw API request for file operations."""
        url = f"{base_url or self.base_url_v2}/{endpoint}"

        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]API request failed:[/bold red] {e}")
            raise

    # Collection Operations (v2)
    def create_collection(self, name: str, description: str = None) -> Dict[str, Any]:
        """Create a new collection."""
        payload = {"name": name}
        if description:
            payload["description"] = description
        response = self._request("POST", "collections", json=payload)
        return response.get("data", response)

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections."""
        response = self._request("GET", "collections")
        return response.get("data", response)

    def get_collection(self, identifier: str) -> Dict[str, Any]:
        """Get collection by ID or name."""
        response = self._request("GET", f"collections/{identifier}")
        return response.get("data", response)

    def delete_collection(self, identifier: str) -> Dict[str, Any]:
        """Delete a collection."""
        response = self._request("DELETE", f"collections/{identifier}")
        return response.get("data", response)

    # Document Operations (v2)
    def upload_document(
        self, collection_identifier: str, file_path: str
    ) -> Dict[str, Any]:
        """Upload a single document to a collection."""
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = self._request(
                "POST", f"collections/{collection_identifier}/documents", files=files
            )
            return response.get("data", response)

    def upload_documents_batch(
        self,
        collection_identifier: str,
        file_paths: List[str],
        process_immediately: bool = True,
    ) -> Dict[str, Any]:
        """Upload multiple documents using v2 upload-and-process endpoint."""
        files = []
        for file_path in file_paths:
            files.append(
                ("files", (os.path.basename(file_path), open(file_path, "rb")))
            )

        try:
            data = {
                "process_immediately": str(process_immediately).lower(),
                "build_graph": "true",
                "extract_metadata": "true",
                "generate_embeddings": "true",
            }

            return self._request(
                "POST",
                f"collections/{collection_identifier}/upload-and-process",
                files=files,
                data=data,
            )
        finally:
            # Close file handles
            for _, (_, file_handle) in files:
                file_handle.close()

    # Task Operations (v2)
    def submit_collection_processing_task(
        self,
        collection_id: str,
        operation_type: str = "complete_analysis",
        build_graph: bool = True,
        extract_metadata: bool = True,
        generate_embeddings: bool = True,
        **parameters,
    ) -> Dict[str, Any]:
        """Submit a collection for processing using Celery tasks."""
        request_data = {
            "collection_id": collection_id,
            "operation_type": operation_type,
            "build_graph": build_graph,
            "extract_metadata": extract_metadata,
            "generate_embeddings": generate_embeddings,
            "parameters": parameters,
        }

        return self._request(
            "POST", f"collections/{collection_id}/process", json=request_data
        )

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the current status of a task."""
        return self._request("GET", f"tasks/{task_id}/status")

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Get the result of a completed task."""
        return self._request("GET", f"tasks/{task_id}/result")

    def cancel_task(
        self, task_id: str, terminate: bool = False, reason: str = None
    ) -> Dict[str, Any]:
        """Cancel a running task."""
        request_data = {"terminate": terminate, "reason": reason}
        return self._request("POST", f"tasks/{task_id}/cancel", json=request_data)

    def list_tasks(
        self, status: str = None, limit: int = 20, offset: int = 0
    ) -> Dict[str, Any]:
        """List active and recent tasks."""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._request("GET", "tasks", params=params)

    def get_task_metrics(self) -> Dict[str, Any]:
        """Get comprehensive task system metrics."""
        return self._request("GET", "tasks/metrics")

    def get_worker_metrics(self) -> Dict[str, Any]:
        """Get worker metrics (alias for get_task_metrics)."""
        return self.get_task_metrics()

    def _check_task_status(self, task_id: str) -> Dict[str, Any]:
        """Check task status and validate response."""
        status_response = self.get_task_status(task_id)

        if not status_response.get("success"):
            raise Exception(f"Task status check failed: {status_response.get('error')}")

        return status_response

    def _update_progress_display(
        self, progress, task_progress, task_id: str, task_data: Dict[str, Any]
    ):
        """Update progress bar with current task status."""
        task_status = task_data["status"]

        if task_data.get("progress"):
            progress_info = task_data["progress"]
            progress.update(
                task_progress,
                completed=progress_info.get("percentage", 0),
                description=f"Task {task_id[:TASK_ID_DISPLAY_LENGTH]}... - {progress_info.get('message', task_status)}",
            )
        else:
            progress.update(
                task_progress,
                description=f"Task {task_id[:TASK_ID_DISPLAY_LENGTH]}... - {task_status}",
            )

    def wait_for_task_completion(
        self,
        task_id: str,
        timeout: int = DEFAULT_TASK_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete with optional progress display.

        Args:
            task_id: Task identifier
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check status in seconds
            show_progress: Whether to show progress bar

        Returns:
            Final task status response
        """
        start_time = time.time()

        if show_progress:
            return self._wait_with_progress(task_id, timeout, poll_interval, start_time)
        else:
            return self._wait_simple(task_id, timeout, poll_interval, start_time)

    def _wait_with_progress(
        self, task_id: str, timeout: int, poll_interval: float, start_time: float
    ) -> Dict[str, Any]:
        """Wait for task completion with progress display."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )

        with progress:
            task_progress = progress.add_task(
                f"Task {task_id[:TASK_ID_DISPLAY_LENGTH]}...", total=PROGRESS_BAR_TOTAL
            )

            while time.time() - start_time < timeout:
                status_response = self._check_task_status(task_id)
                task_data = status_response["data"]
                task_status = task_data["status"]

                self._update_progress_display(
                    progress, task_progress, task_id, task_data
                )

                if task_status in ["SUCCESS", "FAILURE", "REVOKED"]:
                    progress.update(task_progress, completed=PROGRESS_BAR_TOTAL)
                    return status_response

                time.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

    def _wait_simple(
        self, task_id: str, timeout: int, poll_interval: float, start_time: float
    ) -> Dict[str, Any]:
        """Wait for task completion without progress display."""
        while time.time() - start_time < timeout:
            status_response = self._check_task_status(task_id)
            task_data = status_response["data"]
            task_status = task_data["status"]

            if task_status in ["SUCCESS", "FAILURE", "REVOKED"]:
                return status_response

            time.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

    # Query Operations (using task-based RAG processing)
    def query_collection(
        self, collection_identifier: str, question: str, search_type: str = "adaptive"
    ) -> Dict[str, Any]:
        """Query a collection using task-based RAG processing."""
        # Map search types to appropriate tasks
        task_map = {
            "adaptive": "adaptive_graphrag_query",
            "graph": "process_graph_rag_query",
            "vector": "process_vector_rag_query",
            "global": "query_graph_global",
            "local": "query_graph_local",
        }

        task_name = task_map.get(search_type, "adaptive_graphrag_query")

        # Submit as Celery task
        task_response = self._request(
            "POST",
            "tasks/submit",
            self.base_url_v2,
            json={"task_name": task_name, "args": [question, collection_identifier]},
        )

        # Wait for task completion and return result
        if task_response.get("success"):
            task_id = task_response["data"]["task_id"]
            return self.wait_for_task_completion(task_id)
        else:
            raise Exception(
                f"Failed to submit query task: {task_response.get('message', 'Unknown error')}"
            )

    def query_document(
        self, collection_identifier: str, document_identifier: str, question: str
    ) -> Dict[str, Any]:
        """Query a specific document via task-based vector RAG."""
        # Submit vector RAG task for document-specific query
        task_response = self._request(
            "POST",
            "tasks/submit",
            self.base_url_v2,
            json={
                "task_name": "process_vector_rag_query",
                "args": [question, collection_identifier],
                "kwargs": {"document_id": document_identifier},
            },
        )

        # Wait for task completion and return result
        if task_response.get("success"):
            task_id = task_response["data"]["task_id"]
            return self.wait_for_task_completion(task_id)
        else:
            raise Exception(
                f"Failed to submit document query task: {task_response.get('message', 'Unknown error')}"
            )

    # Collection Processing Operations
    def process_collection(
        self,
        collection_identifier: str,
        include_embeddings: bool = True,
        options: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Process a collection with document analysis and optional embeddings."""
        payload = {
            "collection_id": collection_identifier,
            "operation_type": "complete_analysis",
            "build_graph": True,
            "extract_metadata": True,
            "generate_embeddings": include_embeddings,
            "parameters": options or {},
        }
        response = self._request(
            "POST", f"collections/{collection_identifier}/process", json=payload
        )
        return response.get("data", response)

    def upload_and_process_document(
        self,
        collection_identifier: str,
        file_path: str,
        include_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """Upload a document and immediately start collection processing."""
        with open(file_path, "rb") as f:
            files = {"files": (os.path.basename(file_path), f)}
            data = {
                "process_immediately": "true",
                "build_graph": "true",
                "extract_metadata": "true",
                "generate_embeddings": str(include_embeddings).lower(),
            }
            response = self._request(
                "POST",
                f"collections/{collection_identifier}/upload-and-process",
                files=files,
                data=data,
            )
            return response.get("data", response)


def create_task_api_client() -> TaskAPIClient:
    """Create a task API client instance."""
    return TaskAPIClient()


# Utility functions
def format_task_status(status_data: Dict[str, Any]) -> str:
    """Format task status for display."""
    status = status_data.get("status", "UNKNOWN")

    if status_data.get("progress"):
        progress = status_data["progress"]
        percentage = progress.get("percentage", 0)
        message = progress.get("message", "")
        return f"{status} - {percentage:.1f}% {message}"

    return status


def format_task_duration(status_data: Dict[str, Any]) -> str:
    """Format task duration for display."""
    started_at = status_data.get("started_at")
    completed_at = status_data.get("completed_at")

    if started_at and completed_at:
        # Would need to parse datetime strings and calculate duration
        return "Completed"
    elif started_at:
        return "Running"
    else:
        return "Pending"
