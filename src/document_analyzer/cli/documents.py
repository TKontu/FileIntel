import typer
from rich.console import Console
from rich.table import Table
import json
import os
import mimetypes
from pathlib import Path
from typing import List

from .client import FileIntelAPI, print_json, print_table

app = typer.Typer()
api = FileIntelAPI()
console = Console()


@app.command(
    "upload",
    epilog="Example: fileintel documents upload my-collection /path/to/document.pdf",
)
def upload_document(
    collection: str = typer.Argument(
        ..., help="The name or ID of the collection to upload to."
    ),
    path: str = typer.Argument(..., help="The path to the document to upload."),
):
    """Uploads a document to a collection."""
    try:
        result = api.upload_document(collection, path)
        print_json(result)
    except Exception:
        raise typer.Exit(1)


@app.command(
    "add-folder",
    epilog="Example: fileintel documents add-folder my-collection /path/to/folder --recursive",
)
def add_folder_to_collection(
    collection: str = typer.Argument(
        ..., help="The name or ID of the collection to upload to."
    ),
    path: str = typer.Argument(
        ..., help="The path to the folder containing documents."
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Include files from subdirectories"
    ),
    extensions: str = typer.Option(
        "pdf,txt,md,docx,epub,mobi",
        "--extensions",
        help="Comma-separated list of file extensions to include (without dots)",
    ),
    max_files: int = typer.Option(
        999, "--max-files", help="Maximum number of files to process in batch"
    ),
):
    """Add all files from a folder to a collection."""
    try:
        folder_path = Path(path)

        if not folder_path.exists():
            console.print(f"[bold red]Error:[/bold red] Folder '{path}' does not exist")
            raise typer.Exit(1)

        if not folder_path.is_dir():
            console.print(f"[bold red]Error:[/bold red] '{path}' is not a directory")
            raise typer.Exit(1)

        # Parse extensions
        allowed_extensions = [ext.strip().lower() for ext in extensions.split(",")]

        # Find files
        console.print(f"[blue]Scanning folder:[/blue] {path}")
        if recursive:
            console.print("[blue]Including subdirectories[/blue]")

        files_to_upload = []

        if recursive:
            for file_path in folder_path.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower().lstrip(".") in allowed_extensions
                ):
                    files_to_upload.append(str(file_path))
        else:
            for file_path in folder_path.iterdir():
                if (
                    file_path.is_file()
                    and file_path.suffix.lower().lstrip(".") in allowed_extensions
                ):
                    files_to_upload.append(str(file_path))

        if not files_to_upload:
            console.print(
                f"[yellow]No files found with extensions: {', '.join(allowed_extensions)}[/yellow]"
            )
            raise typer.Exit(0)

        console.print(f"[green]Found {len(files_to_upload)} files[/green]")

        # Limit batch size
        if len(files_to_upload) > max_files:
            console.print(
                f"[yellow]Warning: Found {len(files_to_upload)} files, but max-files is set to {max_files}[/yellow]"
            )
            console.print(
                "[yellow]Only the first {max_files} files will be processed[/yellow]"
            )
            files_to_upload = files_to_upload[:max_files]

        # Show files to be uploaded
        if len(files_to_upload) <= 10:
            console.print("[blue]Files to upload:[/blue]")
            for file_path in files_to_upload:
                console.print(f"  • {os.path.basename(file_path)}")
        else:
            console.print(
                f"[blue]Sample files (showing first 5 of {len(files_to_upload)}):[/blue]"
            )
            for file_path in files_to_upload[:5]:
                console.print(f"  • {os.path.basename(file_path)}")
            console.print(f"  ... and {len(files_to_upload) - 5} more")

        # Confirm upload
        if not typer.confirm(
            f"\nUpload {len(files_to_upload)} files to collection '{collection}'?"
        ):
            console.print("[yellow]Upload cancelled[/yellow]")
            raise typer.Exit(0)

        console.print(f"[green]Uploading {len(files_to_upload)} files...[/green]")

        # Upload files in batch
        result = api.upload_documents_batch(collection, files_to_upload)

        console.print(f"[green]✅ {result.get('message')}[/green]")

        if result.get("job_id"):
            console.print(f"[blue]Job ID:[/blue] {result.get('job_id')}")
            console.print(
                "[blue]Use 'fileintel jobs status <job_id>' to check progress[/blue]"
            )

        if result.get("processed_files"):
            console.print(
                f"[green]Successfully queued: {len(result['processed_files'])} files[/green]"
            )

        if result.get("skipped_files"):
            console.print(
                f"[yellow]Skipped: {len(result['skipped_files'])} files[/yellow]"
            )
            for skipped in result["skipped_files"]:
                console.print(f"  • {skipped['filename']}: {skipped['reason']}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_documents(
    collection: str = typer.Argument(
        ..., help="The name or ID of the collection to list documents from."
    )
):
    """Lists all documents in a collection."""
    try:
        documents = api.list_documents(collection)
        print_table(f"Documents in '{collection}'", documents)
    except Exception:
        raise typer.Exit(1)


@app.command(
    "delete",
    epilog="Example: fileintel documents delete my-collection document.pdf",
)
def delete_document(
    collection: str = typer.Argument(
        ..., help="The name or ID of the collection to delete from."
    ),
    document: str = typer.Argument(
        ..., help="The filename or ID of the document to delete."
    ),
):
    """Deletes a document from a collection."""
    try:
        result = api.delete_document(collection, document)
        print_json(result)
    except Exception:
        raise typer.Exit(1)


@app.command(
    "details",
    epilog="Example: fileintel documents details doc-123-uuid",
)
def get_document_details(
    document_id: str = typer.Argument(
        ..., help="The UUID of the document to get details for."
    ),
    show_metadata: bool = typer.Option(
        True, "--metadata/--no-metadata", help="Show detailed metadata"
    ),
):
    """Get detailed information about a document including its metadata."""
    try:
        result = api.get_document_details(document_id)

        if show_metadata:
            # Display in a more readable format
            console.print(f"[bold blue]Document Details[/bold blue]")
            console.print(f"ID: {result.get('id')}")
            console.print(f"Collection ID: {result.get('collection_id')}")
            console.print(f"Filename: {result.get('filename')}")
            console.print(f"Original Filename: {result.get('original_filename')}")
            console.print(f"File Size: {result.get('file_size')} bytes")
            console.print(f"MIME Type: {result.get('mime_type')}")
            console.print(f"Created: {result.get('created_at')}")
            console.print(f"Updated: {result.get('updated_at')}")

            metadata = result.get("document_metadata", {})
            if metadata:
                console.print(f"\n[bold green]Metadata:[/bold green]")

                # Create a nice table for metadata
                table = Table(title="Document Metadata")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")

                for key, value in metadata.items():
                    # Skip internal debugging fields
                    if key.startswith("_"):
                        continue

                    # Format display values
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    elif isinstance(value, dict):
                        value = json.dumps(value, indent=2)

                    # Make field names more readable
                    display_key = key.replace("_", " ").title()
                    table.add_row(display_key, str(value))

                console.print(table)
            else:
                console.print("\n[yellow]No metadata available[/yellow]")
        else:
            print_json(result)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command(
    "update-metadata",
    epilog="Example: fileintel documents update-metadata doc-123-uuid --title 'New Title' --authors 'Author 1,Author 2'",
)
def update_document_metadata(
    document_id: str = typer.Argument(
        ..., help="The UUID of the document to update metadata for."
    ),
    title: str = typer.Option(None, "--title", help="Document title"),
    authors: str = typer.Option(
        None, "--authors", help="Comma-separated list of authors"
    ),
    publication_date: str = typer.Option(
        None, "--publication-date", help="Publication date (YYYY-MM-DD or YYYY)"
    ),
    publisher: str = typer.Option(None, "--publisher", help="Publisher name"),
    doi: str = typer.Option(None, "--doi", help="Digital Object Identifier"),
    source_url: str = typer.Option(None, "--source-url", help="Source URL"),
    language: str = typer.Option(None, "--language", help="Document language"),
    document_type: str = typer.Option(None, "--document-type", help="Type of document"),
    keywords: str = typer.Option(
        None, "--keywords", help="Comma-separated list of keywords"
    ),
    abstract: str = typer.Option(None, "--abstract", help="Document abstract"),
    harvard_citation: str = typer.Option(
        None, "--harvard-citation", help="Harvard-style citation"
    ),
):
    """Update metadata fields for a document."""
    try:
        # Build metadata update object
        metadata_update = {}

        if title:
            metadata_update["title"] = title
        if authors:
            metadata_update["authors"] = [
                author.strip() for author in authors.split(",")
            ]
        if publication_date:
            metadata_update["publication_date"] = publication_date
        if publisher:
            metadata_update["publisher"] = publisher
        if doi:
            metadata_update["doi"] = doi
        if source_url:
            metadata_update["source_url"] = source_url
        if language:
            metadata_update["language"] = language
        if document_type:
            metadata_update["document_type"] = document_type
        if keywords:
            metadata_update["keywords"] = [
                keyword.strip() for keyword in keywords.split(",")
            ]
        if abstract:
            metadata_update["abstract"] = abstract
        if harvard_citation:
            metadata_update["harvard_citation"] = harvard_citation

        if not metadata_update:
            console.print("[yellow]No metadata fields provided to update[/yellow]")
            raise typer.Exit(1)

        result = api.update_document_metadata(document_id, metadata_update)

        console.print(f"[green]✅ {result.get('message')}[/green]")
        console.print(f"Updated fields: {', '.join(result.get('updated_fields', []))}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
