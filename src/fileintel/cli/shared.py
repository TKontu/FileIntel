"""
Shared CLI utilities and common functionality.

Consolidates duplicate patterns across CLI modules to eliminate code bloat
and improve maintainability following DRY principles.
"""

import typer
from typing import Dict, Any, Optional, Callable
from rich.console import Console
from rich.json import JSON

from .task_client import (
    create_task_api_client,
    format_task_status,
    format_task_duration,
)
from .constants import CLI_ERROR, JSON_INDENT


class CLIHandler:
    """Shared CLI handler with common patterns."""

    def __init__(self):
        self.console = Console()
        self.api = None

    def get_api_client(self):
        """Get or create API client with error handling."""
        if not self.api:
            try:
                self.api = create_task_api_client()
            except Exception as e:
                self.console.print(
                    f"[bold red]Failed to connect to API:[/bold red] {e}"
                )
                raise typer.Exit(CLI_ERROR)
        return self.api

    def handle_api_call(
        self, operation: Callable, operation_name: str, *args, **kwargs
    ) -> Any:
        """Execute API call with standardized error handling."""
        try:
            api = self.get_api_client()
            return operation(api, *args, **kwargs)
        except Exception as e:
            self.console.print(f"[bold red]Failed to {operation_name}:[/bold red] {e}")
            raise typer.Exit(CLI_ERROR)

    def display_json(self, data: Dict[str, Any], title: Optional[str] = None):
        """Display JSON data with consistent formatting."""
        if title:
            self.console.print(f"[bold blue]{title}[/bold blue]")
        self.console.print(JSON.from_data(data, indent=JSON_INDENT))

    def display_success(self, message: str):
        """Display success message."""
        self.console.print(f"[bold green]✓[/bold green] {message}")

    def display_error(self, message: str):
        """Display error message."""
        self.console.print(f"[bold red]✗[/bold red] {message}")

    def display_task_info(self, task_data: Dict[str, Any]):
        """Display task information in standardized format."""
        if task_data is None:
            self.display_error("No task information returned from API")
            return
        task_id = task_data.get("task_id", "Unknown")
        self.console.print(f"Task ID: [bold]{task_id}[/bold]")
        if "status" in task_data or "processing_status" in task_data:
            status = task_data.get("status") or task_data.get("processing_status")
            status_text = (
                format_task_status({"status": status}) if status else "Unknown"
            )
            self.console.print(f"Status: {status_text}")
        if (
            "estimated_duration_seconds" in task_data
            or "estimated_duration" in task_data
        ):
            duration = task_data.get("estimated_duration_seconds") or task_data.get(
                "estimated_duration", 0
            )
            self.console.print(f"Estimated Duration: {duration}s")


# Global handler instance
cli_handler = CLIHandler()


def check_system_status(system_name: str, status_endpoint: str) -> None:
    """Generic status checker for different systems."""

    def _check_status(api):
        response = api._request("GET", status_endpoint)
        return response.get("data", response)

    status_data = cli_handler.handle_api_call(
        _check_status, f"check {system_name} status"
    )

    cli_handler.display_json(status_data, f"{system_name.title()} System Status")


def get_entity_by_identifier(
    entity_type: str, identifier: str, get_method: str
) -> Dict[str, Any]:
    """Generic entity retrieval (collection, document, etc.)."""

    def _get_entity(api):
        method = getattr(api, get_method)
        return method(identifier)

    return cli_handler.handle_api_call(_get_entity, f"get {entity_type}")


def display_entity_list(entities: list, entity_type: str, display_fields: list = None):
    """Display a list of entities (collections, tasks, etc.) in a standardized format."""
    if not entities:
        cli_handler.console.print(f"[yellow]No {entity_type} found.[/yellow]")
        return

    cli_handler.console.print(
        f"[bold blue]{entity_type.title()} ({len(entities)}):[/bold blue]"
    )

    for entity in entities:
        if display_fields:
            # Display only specified fields
            display_data = {field: entity.get(field, "N/A") for field in display_fields}
        else:
            # Display all data
            display_data = entity

        cli_handler.console.print(JSON.from_data(display_data, indent=JSON_INDENT))
        cli_handler.console.print()  # Add spacing between entities


def monitor_task_with_progress(task_id: str, task_description: str = "Processing"):
    """Monitor task progress with live updates."""
    api = cli_handler.get_api_client()

    try:
        # Use the existing monitoring functionality from task_client
        result = api.wait_for_task_completion(
            task_id, timeout=None, show_progress=True  # No timeout for CLI usage
        )

        if result.get("successful"):
            cli_handler.display_success(f"{task_description} completed successfully")
            if "result" in result:
                cli_handler.display_json(result["result"], "Task Result")
        else:
            error_msg = result.get("error", "Task failed")
            cli_handler.display_error(f"{task_description} failed: {error_msg}")
            raise typer.Exit(CLI_ERROR)

    except KeyboardInterrupt:
        cli_handler.console.print(
            "\n[yellow]Monitoring interrupted. Task continues running.[/yellow]"
        )
        cli_handler.console.print(
            f"Use 'fileintel tasks get {task_id}' to check status later."
        )
    except Exception as e:
        cli_handler.display_error(f"Failed to monitor task: {e}")
        raise typer.Exit(CLI_ERROR)


def validate_file_exists(file_path: str) -> None:
    """Validate that a file exists before processing."""
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        cli_handler.display_error(f"File not found: {file_path}")
        raise typer.Exit(CLI_ERROR)

    if not path.is_file():
        cli_handler.display_error(f"Path is not a file: {file_path}")
        raise typer.Exit(CLI_ERROR)


def validate_supported_format(file_path: str) -> None:
    """Validate file format is supported."""
    try:
        from fileintel.core.validation import SUPPORTED_FILE_FORMATS
        from pathlib import Path

        extension = Path(file_path).suffix.lower().lstrip(".")
        if extension not in SUPPORTED_FILE_FORMATS:
            supported = ", ".join(SUPPORTED_FILE_FORMATS)
            cli_handler.display_error(f"Unsupported file format: {extension}")
            cli_handler.console.print(f"Supported formats: {supported}")
            raise typer.Exit(CLI_ERROR)

    except ImportError:
        # Fallback if validators not available
        pass
