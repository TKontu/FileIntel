import typer
from rich.console import Console

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
