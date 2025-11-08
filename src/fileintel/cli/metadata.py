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


def _normalize_value(value):
    """Normalize a value for metadata comparison.

    Handles:
    - None vs empty string
    - Lists with different ordering or formatting
    - Single-item list vs string (CSV export artifact)
    - String vs number comparison (e.g., "2023" vs 2023)
    - Boolean strings (e.g., "True" vs True)
    """
    if value is None or value == "":
        return None

    # Convert booleans to strings for comparison
    if isinstance(value, bool):
        return str(value)

    # Handle lists - sort for order-independent comparison
    if isinstance(value, list):
        # Single-item list should match the string value
        # This handles CSV export artifact: ["Wright"] exported as "Wright"
        if len(value) == 1:
            return str(value[0])
        # Multi-item lists: convert to sorted tuple for consistent comparison
        return tuple(sorted([str(item) for item in value]))

    # Convert to string for uniform comparison
    return str(value)


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
    chunks: int = typer.Option(
        3, "--chunks", "-c", help="Number of chunks to use for extraction (default: 3)."
    ),
):
    """Extract metadata from a document using LLM analysis."""

    def _extract_metadata(api):
        payload = {
            "document_id": document_id,
            "force_reextract": force,
            "max_chunks": chunks
        }
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
    chunks: int = typer.Option(
        3, "--chunks", "-c", help="Number of chunks to use for extraction (default: 3)."
    ),
):
    """Extract metadata for all documents in a collection."""

    def _extract_collection_metadata(api):
        params = {"force_reextract": force, "max_chunks": chunks}
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


@app.command("export-table")
def export_collection_table(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to export."
    ),
    format: str = typer.Option(
        "markdown", "--format", "-f", help="Export format: 'csv', 'markdown', or 'json'."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (prints to console if not specified)."
    ),
    fields: Optional[str] = typer.Option(
        None, "--fields", help="Comma-separated metadata fields to include (includes all if not specified)."
    ),
):
    """Export collection documents with metadata in table format.

    This command is useful for reviewing collection integrity, creating backups,
    or exporting metadata for analysis in external tools.

    Note: CSV format uses pipe (|) as the delimiter instead of comma, since
    metadata often contains commas in author lists, titles, etc.

    Examples:
        # Export as Markdown table to console
        fileintel metadata export-table thesis_sources

        # Export as CSV to file (pipe-delimited)
        fileintel metadata export-table thesis_sources --format csv --output metadata.csv

        # Export specific fields only
        fileintel metadata export-table thesis_sources --fields title,author,year --format csv

        # Export as JSON for programmatic use
        fileintel metadata export-table thesis_sources --format json --output backup.json
    """
    from .formatters import format_as_csv, format_as_markdown, format_as_json

    # Validate format
    format_lower = format.lower()
    if format_lower not in ["csv", "markdown", "json"]:
        cli_handler.display_error(f"Invalid format '{format}'. Use: csv, markdown, or json")
        raise typer.Exit(1)

    # Call API to get export data
    def _get_export_data(api):
        return api._request("GET", f"metadata/collection/{collection_identifier}/export")

    result = cli_handler.handle_api_call(_get_export_data, "export collection metadata")
    data = result.get("data", {})

    collection_name = data.get("collection_name", collection_identifier)
    total_docs = data.get("total_documents", 0)

    # Check if collection has documents
    if total_docs == 0:
        cli_handler.display_warning(f"Collection '{collection_name}' has no documents")
        raise typer.Exit(0)

    # Parse field selection
    selected_fields = None
    if fields:
        selected_fields = [f.strip() for f in fields.split(",")]

    # Format based on format option
    try:
        if format_lower == "csv":
            output_content = format_as_csv(data, selected_fields)
        elif format_lower == "markdown":
            output_content = format_as_markdown(data, selected_fields)
        elif format_lower == "json":
            output_content = format_as_json(data, selected_fields)
    except Exception as e:
        cli_handler.display_error(f"Failed to format export: {e}")
        raise typer.Exit(1)

    # Output to file or console
    if output:
        try:
            with open(output, "w", encoding="utf-8") as f:
                f.write(output_content)
            cli_handler.display_success(f"Exported {total_docs} documents to {output}")
        except IOError as e:
            cli_handler.display_error(f"Failed to write to {output}: {e}")
            raise typer.Exit(1)
    else:
        # Display to console
        cli_handler.console.print(f"\n[bold blue]Collection:[/bold blue] {collection_name}")
        cli_handler.console.print(f"[bold blue]Total Documents:[/bold blue] {total_docs}")
        if selected_fields:
            cli_handler.console.print(f"[bold blue]Selected Fields:[/bold blue] {', '.join(selected_fields)}")
        cli_handler.console.print()
        print(output_content)


@app.command("import-table")
def import_collection_table(
    file_path: str = typer.Argument(
        ..., help="Path to the CSV file to import."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without applying them."
    ),
    replace: bool = typer.Option(
        False, "--replace", help="Replace existing metadata entirely (default: merge)."
    ),
):
    """Import metadata from a CSV file to update documents.

    This command allows you to bulk update metadata by:
    1. Exporting metadata to CSV with export-table
    2. Editing the CSV file to fix issues
    3. Importing the corrected CSV back

    The CSV must have a 'document_id' column. Other columns (except filename,
    has_extracted_metadata) will be treated as metadata fields.

    By default, new metadata is MERGED with existing metadata. Use --replace
    to completely replace existing metadata instead.

    Examples:
        # Preview import without making changes
        fileintel metadata import-table metadata.csv --dry-run

        # Import and merge with existing metadata
        fileintel metadata import-table metadata.csv

        # Import and replace existing metadata entirely
        fileintel metadata import-table metadata.csv --replace
    """
    from .formatters import parse_csv_import
    from pathlib import Path

    # Validate file exists
    if not Path(file_path).exists():
        cli_handler.display_error(f"File not found: {file_path}")
        raise typer.Exit(1)

    # Parse CSV file
    cli_handler.console.print(f"[blue]Parsing CSV file:[/blue] {file_path}")
    try:
        updates = parse_csv_import(file_path)
    except ValueError as e:
        cli_handler.display_error(f"Failed to parse CSV: {e}")
        raise typer.Exit(1)

    if not updates:
        cli_handler.display_warning("No updates found in CSV file")
        raise typer.Exit(0)

    cli_handler.console.print(f"[green]Found {len(updates)} document(s) in CSV[/green]")

    # Fetch current metadata to show actual differences
    def _get_current_metadata(doc_id):
        """Fetch current metadata for a document."""
        try:
            result = cli_handler.api._request("GET", f"metadata/document/{doc_id}")
            return result.get("data", {}).get("metadata", {})
        except Exception:
            return {}

    # Analyze changes
    cli_handler.console.print("[blue]Comparing with current metadata...[/blue]")
    actual_changes = []
    no_changes = []

    for update in updates:
        doc_id = update["document_id"]
        new_metadata = update["metadata"]
        current_metadata = _get_current_metadata(doc_id)

        # Find fields that are actually different
        changed_fields = []
        for field, new_value in new_metadata.items():
            current_value = current_metadata.get(field)
            # Normalize for comparison (handle None vs empty string, etc.)
            normalized_new = _normalize_value(new_value)
            normalized_current = _normalize_value(current_value)

            if normalized_new != normalized_current:
                changed_fields.append(field)
                # Debug: Print first difference for troubleshooting
                if len(actual_changes) == 0 and len(changed_fields) == 1:
                    cli_handler.console.print(f"[dim]Debug - First difference found:[/dim]")
                    cli_handler.console.print(f"[dim]  Field: {field}[/dim]")
                    cli_handler.console.print(f"[dim]  CSV value: {repr(new_value)} (type: {type(new_value).__name__})[/dim]")
                    cli_handler.console.print(f"[dim]  DB value: {repr(current_value)} (type: {type(current_value).__name__})[/dim]")
                    cli_handler.console.print(f"[dim]  Normalized CSV: {repr(normalized_new)}[/dim]")
                    cli_handler.console.print(f"[dim]  Normalized DB: {repr(normalized_current)}[/dim]")

        if changed_fields:
            actual_changes.append({
                "document_id": doc_id,
                "changed_fields": changed_fields,
                "metadata": new_metadata
            })
        else:
            no_changes.append(doc_id)

    # Show preview
    if dry_run:
        cli_handler.console.print("\n[bold yellow]DRY RUN - No changes will be made[/bold yellow]\n")

    if actual_changes:
        cli_handler.console.print("[bold]Documents with changes:[/bold]")
        for i, change in enumerate(actual_changes[:5], 1):  # Show first 5
            doc_id = change["document_id"][:8] + "..."
            fields = ", ".join(change["changed_fields"])
            cli_handler.console.print(f"  {i}. {doc_id}: {fields}")

        if len(actual_changes) > 5:
            cli_handler.console.print(f"  ... and {len(actual_changes) - 5} more documents with changes")

    if no_changes:
        cli_handler.console.print(f"\n[dim]Documents with no changes: {len(no_changes)}[/dim]")

    cli_handler.console.print(f"\nMode: {'REPLACE' if replace else 'MERGE'}")
    cli_handler.console.print(f"Total: {len(actual_changes)} to update, {len(no_changes)} unchanged")

    if dry_run:
        cli_handler.console.print("\n[yellow]Dry run complete. Use without --dry-run to apply changes.[/yellow]")
        raise typer.Exit(0)

    # Skip if no changes
    if not actual_changes:
        cli_handler.console.print("\n[green]No changes detected. All metadata is already up to date.[/green]")
        raise typer.Exit(0)

    # Confirm action
    if not typer.confirm("\nProceed with import?"):
        cli_handler.console.print("Import cancelled")
        raise typer.Exit(0)

    # Call API to perform bulk update (only for documents with actual changes)
    def _bulk_update(api):
        payload = {
            "updates": [
                {"document_id": change["document_id"], "metadata": change["metadata"]}
                for change in actual_changes
            ],
            "replace": replace
        }
        return api._request("POST", "metadata/bulk-update", json=payload)

    result = cli_handler.handle_api_call(_bulk_update, "bulk update metadata")
    data = result.get("data", {})

    # Display results
    success_count = data.get("success_count", 0)
    failed_count = data.get("failed_count", 0)

    cli_handler.console.print(f"\n[bold green]✓ Successfully updated: {success_count} documents[/bold green]")

    if failed_count > 0:
        cli_handler.console.print(f"[bold red]✗ Failed: {failed_count} documents[/bold red]")

        failed_updates = data.get("failed_updates", [])
        if failed_updates:
            cli_handler.console.print("\n[red]Failed updates:[/red]")
            for fail in failed_updates[:10]:  # Show first 10 failures
                doc_id = fail.get("document_id", "unknown")
                error = fail.get("error", "unknown error")
                cli_handler.console.print(f"  • {doc_id}: {error}")

            if len(failed_updates) > 10:
                cli_handler.console.print(f"  ... and {len(failed_updates) - 10} more failures")


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
        ("export-table <collection>", "Export collection documents with metadata in table format"),
        ("import-table <file>", "Import metadata from CSV file to update documents"),
        ("system-status", "Check metadata extraction system status"),
    ]

    for cmd, desc in commands:
        cli_handler.console.print(f"  [cyan]fileintel metadata {cmd}[/cyan]")
        cli_handler.console.print(f"    {desc}")
        cli_handler.console.print()

    cli_handler.console.print("[bold]Examples:[/bold]")
    cli_handler.console.print("  # Extract metadata")
    cli_handler.console.print("  fileintel metadata extract abc123 --wait")
    cli_handler.console.print("  fileintel metadata extract abc123 --force --chunks 6")
    cli_handler.console.print()
    cli_handler.console.print("  # View metadata")
    cli_handler.console.print("  fileintel metadata show abc123")
    cli_handler.console.print("  fileintel metadata status 'my-collection'")
    cli_handler.console.print()
    cli_handler.console.print("  # Export/import workflow (fix metadata issues)")
    cli_handler.console.print("  fileintel metadata export-table thesis_sources --format csv --output metadata.csv")
    cli_handler.console.print("  # Edit metadata.csv in Excel/editor to fix issues")
    cli_handler.console.print("  fileintel metadata import-table metadata.csv --dry-run  # Preview changes")
    cli_handler.console.print("  fileintel metadata import-table metadata.csv            # Apply changes")
    cli_handler.console.print()
    cli_handler.console.print("  # Export for review")
    cli_handler.console.print("  fileintel metadata export-table thesis_sources")
    cli_handler.console.print("  fileintel metadata export-table thesis_sources --fields title,author,year")
    cli_handler.console.print()
    cli_handler.console.print("  # Batch operations")
    cli_handler.console.print("  fileintel metadata extract-collection 'my-collection' --force")
    cli_handler.console.print("  fileintel metadata extract-collection 'my-collection' --chunks 10")