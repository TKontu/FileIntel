"""
Consolidated task management CLI commands.

Streamlined task monitoring and management using shared CLI utilities.
Eliminates duplicate task handling patterns across modules.
"""

import typer
from typing import Optional, List

from .shared import cli_handler, display_entity_list, monitor_task_with_progress
from .constants import DEFAULT_TASK_LIMIT, DEFAULT_TASK_OFFSET, TASK_ID_DISPLAY_LENGTH

app = typer.Typer(help="Monitor and manage Celery tasks.")


@app.command("list")
def list_tasks(
    state: Optional[str] = typer.Option(
        None,
        "--state",
        "-s",
        help="Filter tasks by state (PENDING, RUNNING, SUCCESS, FAILURE).",
    ),
    limit: int = typer.Option(
        DEFAULT_TASK_LIMIT, "--limit", "-l", help="Maximum number of tasks to return."
    ),
    offset: int = typer.Option(
        DEFAULT_TASK_OFFSET, "--offset", "-o", help="Number of tasks to skip."
    ),
):
    """List tasks with optional filtering and pagination."""

    def _list_tasks(api):
        # Use the dedicated list_tasks method for proper parameter handling
        return api.list_tasks(status=state, limit=limit, offset=offset)

    response = cli_handler.handle_api_call(_list_tasks, "list tasks")
    tasks_data = response.get("data", response)

    if isinstance(tasks_data, dict) and "tasks" in tasks_data:
        tasks = tasks_data["tasks"]
        total = tasks_data.get("total_count", len(tasks))

        cli_handler.console.print(
            f"[bold blue]Tasks ({len(tasks)} of {total}):[/bold blue]"
        )

        if tasks:
            for task in tasks:
                # Display condensed task info
                task_id = task.get("task_id", "Unknown")[:TASK_ID_DISPLAY_LENGTH]
                state = task.get("state", "UNKNOWN")
                name = task.get("name", "Unknown Task")

                cli_handler.console.print(f"  {task_id} | {state:12} | {name}")

            # Show pagination info
            has_more = tasks_data.get("has_more", False)
            if has_more:
                next_offset = offset + limit
                cli_handler.console.print(
                    f"\n[dim]Use --offset {next_offset} to see more tasks[/dim]"
                )
        else:
            cli_handler.console.print("[yellow]No tasks found.[/yellow]")
    else:
        # Fallback for other response formats
        display_entity_list(
            tasks_data if isinstance(tasks_data, list) else [tasks_data], "tasks"
        )


@app.command("get")
def get_task(
    task_id: str = typer.Argument(..., help="The ID of the task to retrieve.")
):
    """Get detailed information about a specific task."""

    def _get_task(api):
        return api.get_task_status(task_id)

    task_data = cli_handler.handle_api_call(_get_task, "get task")

    # Check if API call was successful
    if not task_data.get("success", False):
        error_msg = task_data.get("error", "Unknown error occurred")
        cli_handler.display_error(f"Failed to get task status: {error_msg}")
        return

    cli_handler.display_json(task_data.get("data", {}), f"Task: {task_id}")


@app.command("cancel")
def cancel_task(
    task_id: str = typer.Argument(..., help="The ID of the task to cancel."),
    terminate: bool = typer.Option(
        False, "--terminate", "-t", help="Forcefully terminate the task."
    ),
):
    """Cancel a specific task."""

    def _cancel_task(api):
        # Use the dedicated cancel_task method for proper request handling
        return api.cancel_task(task_id, terminate=terminate)

    result = cli_handler.handle_api_call(_cancel_task, "cancel task")
    status = result.get("data", result).get("status", "unknown")

    if status == "cancelled":
        cli_handler.display_success(f"Task {task_id} cancelled successfully")
    elif status == "already_completed":
        cli_handler.console.print(
            f"[yellow]Task {task_id} has already completed[/yellow]"
        )
    else:
        cli_handler.display_error(f"Failed to cancel task: {status}")


@app.command("result")
def get_task_result(
    task_id: str = typer.Argument(..., help="The ID of the task to get result for."),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", "-t", help="Timeout in seconds to wait for result."
    ),
):
    """Get the result of a completed task."""

    def _get_result(api):
        # Use the dedicated get_task_result method (timeout not currently supported by API)
        return api.get_task_result(task_id)

    result_data = cli_handler.handle_api_call(_get_result, "get task result")

    # Check if API call was successful
    if not result_data.get("success", False):
        error_msg = result_data.get("error", "Unknown error occurred")
        cli_handler.display_error(f"Failed to get task result: {error_msg}")
        return

    task_result = result_data.get("data", {})

    if task_result and task_result.get("ready"):
        if task_result.get("successful"):
            cli_handler.display_success("Task completed successfully")
            if "result" in task_result:
                cli_handler.display_json(task_result["result"], "Task Result")
        else:
            error = task_result.get("error", "Unknown error")
            cli_handler.display_error(f"Task failed: {error}")
    else:
        cli_handler.console.print(f"[yellow]Task {task_id} is still running[/yellow]")
        cli_handler.console.print("Use 'fileintel tasks wait' to monitor progress")


@app.command("wait")
def wait_for_task(
    task_id: str = typer.Argument(..., help="The ID of the task to wait for."),
    timeout: Optional[int] = typer.Option(
        None, "--timeout", "-t", help="Maximum time to wait in seconds."
    ),
):
    """Wait for a task to complete and show progress."""
    try:
        monitor_task_with_progress(task_id, "Task execution")
    except KeyboardInterrupt:
        cli_handler.console.print(
            f"\n[yellow]Stopped monitoring task {task_id}[/yellow]"
        )


@app.command("metrics")
def get_task_metrics():
    """Get Celery worker metrics and task statistics."""

    def _get_metrics(api):
        return api.get_worker_metrics()

    metrics = cli_handler.handle_api_call(_get_metrics, "get task metrics")
    cli_handler.display_json(metrics.get("data", metrics), "Task System Metrics")


@app.command("batch-cancel")
def batch_cancel_tasks(
    task_ids: List[str] = typer.Argument(..., help="List of task IDs to cancel."),
    terminate: bool = typer.Option(
        False, "--terminate", "-t", help="Forcefully terminate the tasks."
    ),
):
    """Cancel multiple tasks in batch."""
    if len(task_ids) > 100:
        cli_handler.display_error("Too many task IDs (maximum 100 allowed)")
        raise typer.Exit(1)

    def _batch_cancel(api):
        payload = {"task_ids": task_ids, "terminate": terminate}
        return api._request("POST", "tasks/batch/cancel", json=payload)

    result = cli_handler.handle_api_call(_batch_cancel, "batch cancel tasks")
    batch_result = result.get("data", result)

    summary = batch_result.get("summary", {})
    cli_handler.console.print(f"[bold blue]Batch Cancel Results:[/bold blue]")
    cli_handler.console.print(f"  Cancelled: {summary.get('cancelled', 0)}")
    cli_handler.console.print(
        f"  Already completed: {summary.get('already_completed', 0)}"
    )
    cli_handler.console.print(f"  Errors: {summary.get('errors', 0)}")

    if summary.get("errors", 0) > 0:
        cli_handler.console.print(
            "\n[yellow]Some tasks could not be cancelled. Use 'get' command for details.[/yellow]"
        )


@app.command("system-status")
def system_status():
    """Check task management system status."""

    def _check_status(api):
        return api.get_worker_metrics()

    metrics = cli_handler.handle_api_call(_check_status, "check task system status")
    workers = metrics.get("data", metrics).get("workers", {})

    cli_handler.console.print(f"[bold blue]Task System Status:[/bold blue]")
    cli_handler.console.print(f"  Active Workers: {len(workers)}")

    for worker_name, worker_info in workers.items():
        status = worker_info.get("status", "unknown")
        active_tasks = worker_info.get("active_tasks", 0)
        cli_handler.console.print(
            f"  {worker_name}: {status} ({active_tasks} active tasks)"
        )

    if not workers:
        cli_handler.console.print("[yellow]No active workers found[/yellow]")
