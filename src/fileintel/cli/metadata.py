"""
Metadata extraction CLI commands.

Provides commands for extracting and managing document metadata using LLM analysis.
"""

import typer
from typing import Optional
from pathlib import Path

from .shared import (
    cli_handler,
    check_system_status,
    monitor_task_with_progress,
    get_entity_by_identifier,
)

app = typer.Typer(help="Document metadata extraction operations.")


@app.command("extract")
def extract_document_metadata(
    document_id: str = typer.Argument(
        ..., help="The ID of the document to extract metadata for."
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for extraction to complete."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-extraction even if metadata exists."
    ),
):
    """Extract metadata from a document using LLM analysis."""

    def _extract_metadata(api):
        payload = {"document_id": document_id, "force_reextract": force}
        return api._request("POST", "metadata/extract", json=payload)

    result = cli_handler.handle_api_call(_extract_metadata, "extract document metadata")
    task_data = result.get("data", result)

    cli_handler.display_success("Metadata extraction started")
    cli_handler.display_task_info(task_data)

    if wait:
        task_id = task_data.get("task_id")
        if task_id:
            monitor_task_with_progress(task_id, "Metadata extraction")


@app.command("show")
def show_document_metadata(
    document_id: str = typer.Argument(
        ..., help="The ID of the document to show metadata for."
    ),
):
    """Show extracted metadata for a document."""

    def _get_metadata(api):
        return api._request("GET", f"metadata/document/{document_id}")

    metadata_data = cli_handler.handle_api_call(_get_metadata, "get document metadata")
    data = metadata_data.get("data", metadata_data)

    cli_handler.console.print(f"[bold blue]Metadata for '{data.get('filename', 'Unknown')}'[/bold blue]")
    cli_handler.console.print(f"Document ID: {data.get('document_id')}")

    has_extracted = data.get("has_extracted_metadata", False)
    if has_extracted:
        cli_handler.console.print("✓ [green]Has LLM-extracted metadata[/green]")
    else:
        cli_handler.console.print("○ [yellow]No LLM-extracted metadata[/yellow]")

    metadata = data.get("metadata", {})
    if metadata:
        cli_handler.console.print(f"\n[bold green]Metadata Fields ({len(metadata)}):[/bold green]")

        # Display canonical fields in a nice format
        canonical_fields = [
            "title", "authors", "publication_date", "publisher", "doi",
            "source_url", "language", "document_type", "keywords", "abstract"
        ]

        for field in canonical_fields:
            if field in metadata and metadata[field]:
                value = metadata[field]
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                elif len(str(value)) > 100:
                    value = str(value)[:100] + "..."

                cli_handler.console.print(f"  [cyan]{field.replace('_', ' ').title()}:[/cyan] {value}")

        # Show other fields
        other_fields = {k: v for k, v in metadata.items() if k not in canonical_fields and not k.startswith('_')}
        if other_fields:
            cli_handler.console.print(f"\n[bold]Other Fields:[/bold]")
            for field, value in other_fields.items():
                if value:
                    cli_handler.console.print(f"  [cyan]{field}:[/cyan] {value}")
    else:
        cli_handler.console.print("\n[yellow]No metadata found[/yellow]")


@app.command("extract-collection")
def extract_collection_metadata(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to extract metadata for."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-extraction for all documents."
    ),
):
    """Extract metadata for all documents in a collection."""

    def _extract_collection_metadata(api):
        params = {"force_reextract": force}
        return api._request("POST", f"metadata/collection/{collection_identifier}/extract-all", params=params)

    result = cli_handler.handle_api_call(_extract_collection_metadata, "extract collection metadata")
    data = result.get("data", result)

    collection_name = data.get("collection_name", collection_identifier)
    total_docs = data.get("total_documents", 0)
    processed_docs = data.get("documents_processed", 0)
    tasks = data.get("tasks_started", [])

    cli_handler.display_success(f"Metadata extraction started for collection '{collection_name}'")
    cli_handler.console.print(f"Total documents: {total_docs}")
    cli_handler.console.print(f"Documents processed: {processed_docs}")

    if tasks:
        cli_handler.console.print(f"\n[bold green]Started Tasks:[/bold green]")
        for task in tasks[:5]:  # Show first 5 tasks
            filename = task.get("filename", "Unknown")
            task_id = task.get("task_id", "Unknown")
            cli_handler.console.print(f"  • {filename} (Task: {task_id[:8]}...)")

        if len(tasks) > 5:
            cli_handler.console.print(f"  ... and {len(tasks) - 5} more tasks")


@app.command("status")
def collection_metadata_status(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to check status for."
    ),
):
    """Get metadata extraction status for a collection."""

    def _get_status(api):
        return api._request("GET", f"metadata/collection/{collection_identifier}/status")

    status_data = cli_handler.handle_api_call(_get_status, "get collection metadata status")
    data = status_data.get("data", status_data)

    collection_name = data.get("collection_name", collection_identifier)
    summary = data.get("summary", {})
    documents = data.get("documents", [])

    cli_handler.console.print(f"[bold blue]Metadata Status for '{collection_name}'[/bold blue]")

    total = summary.get("total_documents", 0)
    extracted = summary.get("with_extracted_metadata", 0)
    file_only = summary.get("with_file_metadata_only", 0)
    none = summary.get("without_metadata", 0)

    cli_handler.console.print(f"\n[bold green]Summary:[/bold green]")
    cli_handler.console.print(f"  Total documents: {total}")
    cli_handler.console.print(f"  ✓ With extracted metadata: [green]{extracted}[/green]")
    cli_handler.console.print(f"  ○ File metadata only: [yellow]{file_only}[/yellow]")
    cli_handler.console.print(f"  ✗ No metadata: [red]{none}[/red]")

    if total > 0:
        completion_rate = (extracted / total) * 100
        cli_handler.console.print(f"  Completion rate: {completion_rate:.1f}%")

    # Show document details
    if documents:
        cli_handler.console.print(f"\n[bold green]Document Details:[/bold green]")
        for doc in documents[:10]:  # Show first 10 documents
            filename = doc.get("filename", "Unknown")
            status = doc.get("status", "unknown")
            fields = doc.get("metadata_fields", 0)

            status_icon = {
                "extracted": "✓",
                "file_metadata_only": "○",
                "no_metadata": "✗"
            }.get(status, "?")

            status_color = {
                "extracted": "green",
                "file_metadata_only": "yellow",
                "no_metadata": "red"
            }.get(status, "white")

            cli_handler.console.print(f"  {status_icon} [{status_color}]{filename}[/{status_color}] ({fields} fields)")

        if len(documents) > 10:
            cli_handler.console.print(f"  ... and {len(documents) - 10} more documents")


@app.command("system-status")
def metadata_system_status():
    """Check metadata extraction system status."""
    check_system_status("Metadata Extraction", "metadata/system-status")


# Help command
@app.command("help")
def show_help():
    """Show detailed help for metadata commands."""
    cli_handler.console.print("[bold blue]Metadata Extraction Commands[/bold blue]")
    cli_handler.console.print()

    commands = [
        ("extract <document_id>", "Extract metadata from a specific document"),
        ("show <document_id>", "Display extracted metadata for a document"),
        ("extract-collection <collection>", "Extract metadata for all documents in collection"),
        ("status <collection>", "Show metadata extraction status for collection"),
        ("system-status", "Check metadata extraction system status"),
    ]

    for cmd, desc in commands:
        cli_handler.console.print(f"  [cyan]fileintel metadata {cmd}[/cyan]")
        cli_handler.console.print(f"    {desc}")
        cli_handler.console.print()

    cli_handler.console.print("[bold]Examples:[/bold]")
    cli_handler.console.print("  fileintel metadata extract abc123 --wait")
    cli_handler.console.print("  fileintel metadata show abc123")
    cli_handler.console.print("  fileintel metadata extract-collection 'my-collection' --force")
    cli_handler.console.print("  fileintel metadata status 'my-collection'")