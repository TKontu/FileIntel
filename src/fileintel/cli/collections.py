"""
Consolidated collections CLI commands.

Streamlined collection management using shared CLI utilities.
Eliminates duplicate status checking and error handling patterns.
"""

import typer
from typing import Optional

from .shared import (
    cli_handler,
    check_system_status,
    get_entity_by_identifier,
    display_entity_list,
    monitor_task_with_progress,
)

app = typer.Typer(help="Manage collections with task-based processing.")


@app.command("create")
def create_collection(
    name: str = typer.Argument(..., help="The name of the collection to create."),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Optional description for the collection."
    ),
):
    """Create a new collection."""

    def _create(api):
        return api.create_collection(name, description)

    result = cli_handler.handle_api_call(_create, "create collection")
    cli_handler.display_success(f"Collection '{name}' created successfully")
    cli_handler.display_json(result)


@app.command("list")
def list_collections():
    """List all collections."""

    def _list(api):
        return api.list_collections()

    collections = cli_handler.handle_api_call(_list, "list collections")
    display_entity_list(
        collections, "collections", ["id", "name", "description", "status"]
    )


@app.command("get")
def get_collection(
    identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to retrieve."
    )
):
    """Get detailed information about a collection."""
    collection = get_entity_by_identifier("collection", identifier, "get_collection")
    cli_handler.display_json(collection, f"Collection: {identifier}")


@app.command("delete")
def delete_collection(
    identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to delete."
    ),
    confirm: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt."
    ),
):
    """Delete a collection."""
    if not confirm:
        collection = get_entity_by_identifier(
            "collection", identifier, "get_collection"
        )
        collection_name = collection.get("name", identifier)

        if not typer.confirm(
            f"Are you sure you want to delete collection '{collection_name}'?"
        ):
            cli_handler.console.print("Operation cancelled.")
            return

    def _delete(api):
        return api.delete_collection(identifier)

    result = cli_handler.handle_api_call(_delete, "delete collection")
    cli_handler.display_success(f"Collection '{identifier}' deleted successfully")


@app.command("process")
def process_collection(
    identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to process."
    ),
    embeddings: bool = typer.Option(
        True, "--embeddings/--no-embeddings", help="Include embedding generation."
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for processing to complete."
    ),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", help="Maximum number of worker processes."
    ),
):
    """Process a collection with document analysis and optional embeddings."""

    def _process(api):
        options = {}
        if max_workers is not None:
            options["max_workers"] = max_workers

        return api.process_collection(
            identifier,
            include_embeddings=embeddings,
            options=options if options else None,
        )

    result = cli_handler.handle_api_call(_process, "process collection")
    if result:
        cli_handler.display_success("Collection processing started")
        cli_handler.display_task_info(result)

        if wait:
            task_id = result.get("task_id")
            if task_id:
                monitor_task_with_progress(task_id, "Collection processing")
    else:
        cli_handler.display_error(
            "Process collection request failed - no response received"
        )


@app.command("status")
def collection_status(
    identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to check."
    )
):
    """Get detailed processing status of a collection."""

    def _get_status(api):
        return api._request("GET", f"collections/{identifier}/processing-status")

    status_data = cli_handler.handle_api_call(_get_status, "get collection status")
    cli_handler.display_json(
        status_data.get("data", status_data), f"Collection Status: {identifier}"
    )


@app.command("system-status")
def system_status():
    """Check collection management system status."""
    check_system_status("collection", "collections")


@app.command("upload-and-process")
def upload_and_process(
    identifier: str = typer.Argument(..., help="The name or ID of the collection."),
    file_path: str = typer.Argument(..., help="Path to the document to upload."),
    embeddings: bool = typer.Option(
        True, "--embeddings/--no-embeddings", help="Include embedding generation."
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for processing to complete."
    ),
):
    """Upload a document and immediately start collection processing."""
    from .shared import validate_file_exists, validate_supported_format

    # Validate file
    validate_file_exists(file_path)
    validate_supported_format(file_path)

    def _upload_and_process(api):
        return api.upload_and_process_document(identifier, file_path, embeddings)

    result = cli_handler.handle_api_call(
        _upload_and_process, "upload and process document"
    )
    if result:
        cli_handler.display_success("Document uploaded and processing started")
        cli_handler.display_task_info(result)

        if wait and result.get("task_id"):
            task_id = result["task_id"]
            monitor_task_with_progress(task_id, "Document processing")
    else:
        cli_handler.display_error("Upload and process request failed")
