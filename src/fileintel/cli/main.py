"""
FileIntel CLI - Consolidated task-based interface.

Streamlined CLI implementation eliminating duplicate functionality
across modules while maintaining full feature set.
"""

import typer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from . import collections, documents, tasks, query, graphrag, metadata
from .shared import cli_handler

app = typer.Typer(
    name="fileintel",
    help="FileIntel CLI - Streamlined document processing and analysis.",
    add_completion=False,
)

# Add consolidated command groups
app.add_typer(
    collections.app,
    name="collections",
    help="Manage collections with task-based processing.",
)
app.add_typer(documents.app, name="documents", help="Manage documents.")
app.add_typer(tasks.app, name="tasks", help="Monitor and manage Celery tasks.")
app.add_typer(query.app, name="query", help="Query collections using RAG.")
app.add_typer(graphrag.app, name="graphrag", help="GraphRAG operations.")
app.add_typer(metadata.app, name="metadata", help="Extract and manage document metadata.")


@app.command("version")
def version():
    """Show CLI version information."""
    try:
        from fileintel import __version__
        version_str = __version__
    except ImportError:
        version_str = "unknown"

    cli_handler.console.print(
        f"[bold blue]FileIntel CLI v{version_str}[/bold blue]"
    )
    cli_handler.console.print("API: Celery-based task processing")
    cli_handler.console.print("Architecture: Streamlined modular design")
    cli_handler.console.print("CLI: Consolidated command structure")


@app.command("health")
def health_check():
    """Check API and task system health."""
    try:
        api = cli_handler.get_api_client()

        # Check v2 API health
        cli_handler.console.print("[blue]Checking API health...[/blue]")
        response = api._request("GET", "tasks/metrics")

        if response.get("success"):
            cli_handler.display_success("API is healthy")

            # Extract metrics information
            data = response.get("data", {})
            if isinstance(data, dict):
                # Display key metrics from TaskMetricsResponse
                worker_count = data.get("worker_count", 0)
                active_tasks = data.get("active_tasks", 0)
                pending_tasks = data.get("pending_tasks", 0)

                cli_handler.console.print(f"Active Workers: {worker_count}")
                cli_handler.console.print(f"Active Tasks: {active_tasks}")
                cli_handler.console.print(f"Pending Tasks: {pending_tasks}")
            else:
                cli_handler.console.print("Metrics data available")
        else:
            cli_handler.display_error("API health check failed")
            raise typer.Exit(1)

    except Exception as e:
        cli_handler.display_error(f"Health check failed: {e}")
        raise typer.Exit(1)


@app.command("status")
def overall_status():
    """Check overall system status across all components."""
    cli_handler.console.print("[bold blue]FileIntel System Status Check[/bold blue]")

    components = [
        ("API Health", "tasks/metrics"),
        ("Collections", "collections"),
        ("Query System", "query/status"),
        ("GraphRAG System", "graphrag/status"),
        ("Metadata System", "metadata/system-status"),
    ]

    for component_name, endpoint in components:
        try:
            api = cli_handler.get_api_client()
            response = api._request("GET", endpoint)

            if response.get(
                "success", True
            ):  # Some endpoints don't return success field
                cli_handler.console.print(
                    f"  ✓ {component_name}: [green]Healthy[/green]"
                )
            else:
                cli_handler.console.print(f"  ✗ {component_name}: [red]Unhealthy[/red]")

        except Exception as e:
            cli_handler.console.print(f"  ✗ {component_name}: [red]Failed ({e})[/red]")

    cli_handler.console.print(
        "\nUse specific 'system-status' commands for detailed information:"
    )
    cli_handler.console.print("  • fileintel collections system-status")
    cli_handler.console.print("  • fileintel tasks system-status")
    cli_handler.console.print("  • fileintel query system-status")
    cli_handler.console.print("  • fileintel graphrag system-status")


@app.command("quickstart")
def quickstart():
    """Show quick start guide for common operations."""
    cli_handler.console.print("[bold blue]FileIntel CLI Quick Start[/bold blue]")
    cli_handler.console.print()

    cli_handler.console.print("[bold yellow]1. Create a collection:[/bold yellow]")
    cli_handler.console.print(
        "  fileintel collections create 'My Documents' --description 'Sample collection'"
    )
    cli_handler.console.print()

    cli_handler.console.print("[bold yellow]2. Upload documents:[/bold yellow]")
    cli_handler.console.print(
        "  fileintel documents upload 'My Documents' /path/to/document.pdf"
    )
    cli_handler.console.print(
        "  fileintel documents batch-upload 'My Documents' /path/to/directory/"
    )
    cli_handler.console.print()

    cli_handler.console.print("[bold yellow]3. Process collection:[/bold yellow]")
    cli_handler.console.print("  fileintel collections process 'My Documents' --wait")
    cli_handler.console.print()

    cli_handler.console.print("[bold yellow]4. Query your documents:[/bold yellow]")
    cli_handler.console.print(
        "  fileintel query collection 'My Documents' 'What are the main topics?'"
    )
    cli_handler.console.print()

    cli_handler.console.print("[bold yellow]5. Monitor tasks:[/bold yellow]")
    cli_handler.console.print("  fileintel tasks list")
    cli_handler.console.print("  fileintel tasks get <task-id>")
    cli_handler.console.print()

    cli_handler.console.print(
        "[bold green]Tip:[/bold green] Use --help with any command for detailed options"
    )


if __name__ == "__main__":
    app()
