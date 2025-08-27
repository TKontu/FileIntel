import typer
from rich.console import Console
from rich.table import Table
import json

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
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    elif isinstance(value, dict):
                        value = json.dumps(value, indent=2)
                    table.add_row(key, str(value))

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
