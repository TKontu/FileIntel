"""
Consolidated document management CLI commands.

Streamlined document operations using shared CLI utilities.
Eliminates duplicate upload and status checking patterns.
"""

import typer
from typing import List
from pathlib import Path

from .shared import (
    cli_handler,
    check_system_status,
    validate_file_exists,
    validate_supported_format,
    monitor_task_with_progress,
)

app = typer.Typer(help="Manage documents.")


@app.command("upload")
def upload_document(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection."
    ),
    file_path: str = typer.Argument(..., help="Path to the document to upload."),
    process: bool = typer.Option(
        False, "--process", "-p", help="Immediately process after upload."
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for processing to complete."
    ),
):
    """Upload a single document to a collection."""
    # Validate file
    validate_file_exists(file_path)
    validate_supported_format(file_path)

    def _upload(api):
        return api.upload_document(collection_identifier, file_path)

    result = cli_handler.handle_api_call(_upload, "upload document")
    cli_handler.display_success(
        f"Document '{Path(file_path).name}' uploaded successfully"
    )
    cli_handler.display_json(result)

    if process:
        # Start collection processing after upload
        def _process(api):
            return api.process_collection(
                collection_identifier, include_embeddings=True
            )

        process_result = cli_handler.handle_api_call(_process, "process collection")
        cli_handler.display_success("Collection processing started")
        cli_handler.display_task_info(process_result)

        if wait:
            task_id = process_result.get("task_id")
            if task_id:
                monitor_task_with_progress(task_id, "Document processing")


@app.command("batch-upload")
def batch_upload_documents(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection."
    ),
    directory: str = typer.Argument(
        ..., help="Directory containing documents to upload."
    ),
    pattern: str = typer.Option(
        "*", "--pattern", "-p", help="File pattern to match (e.g., '*.pdf')."
    ),
    process: bool = typer.Option(
        True, "--process/--no-process", help="Process collection after upload."
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for processing to complete."
    ),
):
    """Upload multiple documents from a directory."""
    dir_path = Path(directory)

    if not dir_path.exists():
        cli_handler.display_error(f"Directory not found: {directory}")
        raise typer.Exit(1)

    if not dir_path.is_dir():
        cli_handler.display_error(f"Path is not a directory: {directory}")
        raise typer.Exit(1)

    # Find matching files
    files = list(dir_path.glob(pattern))
    if not files:
        cli_handler.display_error(
            f"No files found matching pattern '{pattern}' in {directory}"
        )
        raise typer.Exit(1)

    # Filter for supported formats
    try:
        from fileintel.core.validation import SUPPORTED_FILE_FORMATS

        supported_files = [
            f for f in files if f.suffix.lower().lstrip(".") in SUPPORTED_FILE_FORMATS
        ]
    except ImportError:
        supported_files = files  # Fallback if validators not available

    if not supported_files:
        cli_handler.display_error("No supported file formats found")
        raise typer.Exit(1)

    cli_handler.console.print(
        f"[blue]Uploading {len(supported_files)} documents...[/blue]"
    )

    # Upload files one by one (simpler than complex batch API)
    uploaded_count = 0
    for file_path in supported_files:
        try:

            def _upload(api):
                return api.upload_document(collection_identifier, str(file_path))

            result = cli_handler.handle_api_call(_upload, f"upload {file_path.name}")
            uploaded_count += 1
            cli_handler.console.print(f"  ✓ {file_path.name}")

        except Exception as e:
            cli_handler.console.print(f"  ✗ {file_path.name}: {e}")

    cli_handler.display_success(
        f"Uploaded {uploaded_count} of {len(supported_files)} documents"
    )

    if process and uploaded_count > 0:
        # Start collection processing after uploads
        def _process(api):
            return api.process_collection(
                collection_identifier, include_embeddings=True
            )

        process_result = cli_handler.handle_api_call(_process, "process collection")
        cli_handler.display_success("Collection processing started")
        cli_handler.display_task_info(process_result)

        if wait:
            task_id = process_result.get("task_id")
            if task_id:
                monitor_task_with_progress(task_id, "Batch document processing")


@app.command("list")
def list_documents(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection."
    )
):
    """List documents in a collection."""

    def _get_collection(api):
        return api.get_collection(collection_identifier)

    collection_data = cli_handler.handle_api_call(_get_collection, "get collection")
    documents = collection_data.get("documents", [])

    if documents:
        cli_handler.console.print(
            f"[bold blue]Documents in '{collection_identifier}' ({len(documents)}):[/bold blue]"
        )
        for doc in documents:
            doc_id = doc.get("id", "Unknown")  # Show full ID
            filename = doc.get("original_filename") or doc.get("filename", "Unknown")
            file_size = doc.get("file_size", 0)
            mime_type = doc.get("mime_type", "Unknown")

            # Format file size
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} bytes"

            cli_handler.console.print(
                f"  {doc_id} | {filename} | {size_str} | {mime_type}"
            )
    else:
        cli_handler.console.print(
            f"[yellow]No documents found in collection '{collection_identifier}'[/yellow]"
        )


@app.command("get")
def get_document(
    document_id: str = typer.Argument(..., help="The ID of the document to retrieve.")
):
    """Get detailed information about a specific document."""

    def _get_document(api):
        return api._request("GET", f"documents/{document_id}")

    document_data = cli_handler.handle_api_call(_get_document, "get document")
    cli_handler.display_json(
        document_data.get("data", document_data), f"Document: {document_id}"
    )


@app.command("delete")
def delete_document(
    document_id: str = typer.Argument(..., help="The ID of the document to delete."),
    confirm: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt."
    ),
):
    """Delete a document."""
    if not confirm:
        # Get document info first
        def _get_document(api):
            return api._request("GET", f"documents/{document_id}")

        doc_data = cli_handler.handle_api_call(_get_document, "get document info")
        filename = doc_data.get("filename", document_id)

        if not typer.confirm(f"Are you sure you want to delete document '{filename}'?"):
            cli_handler.console.print("Operation cancelled.")
            return

    def _delete(api):
        return api._request("DELETE", f"documents/{document_id}")

    result = cli_handler.handle_api_call(_delete, "delete document")
    cli_handler.display_success(f"Document '{document_id}' deleted successfully")


@app.command("chunks")
def view_chunks(
    document_id: str = typer.Argument(..., help="The ID of the document (full or prefix)."),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of chunks to display."),
    offset: int = typer.Option(0, "--offset", "-o", help="Number of chunks to skip."),
    show_text: bool = typer.Option(True, "--text/--no-text", help="Show chunk text content."),
    text_preview: int = typer.Option(100, "--preview", "-p", help="Max characters of text to show (0 for full)."),
):
    """View chunks and their metadata for a document."""

    def _get_chunks(api):
        return api._request("GET", f"documents/{document_id}/chunks", params={"limit": limit, "offset": offset})

    result = cli_handler.handle_api_call(_get_chunks, "get document chunks")

    if not result or "data" not in result:
        cli_handler.display_error("No chunk data returned")
        return

    data = result["data"]
    chunks = data.get("chunks", [])

    if not chunks:
        cli_handler.console.print(f"[yellow]No chunks found for document '{document_id}'[/yellow]")
        return

    cli_handler.console.print(
        f"[bold blue]Document {data.get('document_id', 'Unknown')} - "
        f"Chunks {offset + 1}-{offset + len(chunks)} of {data.get('total_chunks', 0)}[/bold blue]\n"
    )

    for i, chunk in enumerate(chunks):
        chunk_idx = chunk.get("chunk_index", offset + i)
        chunk_id = chunk.get("chunk_id", "Unknown")
        has_embedding = chunk.get("has_embedding", False)
        embedding_dims = chunk.get("embedding_dimensions")
        metadata = chunk.get("chunk_metadata", {})
        text = chunk.get("text", "")

        # Display chunk header
        cli_handler.console.print(f"[bold cyan]Chunk #{chunk_idx}[/bold cyan] [dim]({chunk_id[:12]}...)[/dim]")

        # Display embedding info
        embedding_status = f"✓ {embedding_dims}D" if has_embedding else "✗ No embedding"
        cli_handler.console.print(f"  Embedding: {embedding_status}")

        # Display metadata
        if metadata:
            cli_handler.console.print(f"  [bold]Metadata:[/bold]")
            for key, value in metadata.items():
                # Format the value nicely
                if isinstance(value, (dict, list)):
                    import json
                    value_str = json.dumps(value, indent=2)
                    cli_handler.console.print(f"    {key}:")
                    for line in value_str.split('\n'):
                        cli_handler.console.print(f"      {line}")
                else:
                    cli_handler.console.print(f"    {key}: {value}")
        else:
            cli_handler.console.print(f"  [dim]No metadata[/dim]")

        # Display text content
        if show_text:
            if text_preview > 0 and len(text) > text_preview:
                preview_text = text[:text_preview] + "..."
                cli_handler.console.print(f"  [bold]Text:[/bold] {preview_text}")
                cli_handler.console.print(f"  [dim](showing {text_preview} of {len(text)} characters)[/dim]")
            else:
                cli_handler.console.print(f"  [bold]Text:[/bold] {text}")

        cli_handler.console.print()  # Empty line between chunks


@app.command("system-status")
def system_status():
    """Check document management system status."""
    # Document management is handled through collections, so check collections status
    # This gives insight into the overall document storage and processing system
    check_system_status("document management", "collections")
