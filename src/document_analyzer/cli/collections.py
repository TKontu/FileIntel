import typer
from rich.console import Console

from .client import FileIntelAPI, print_json, print_table

app = typer.Typer()
api = FileIntelAPI()
console = Console()


@app.command("create")
def create_collection(
    name: str = typer.Argument(..., help="The name of the collection to create.")
):
    """Creates a new collection."""
    try:
        result = api.create_collection(name)
        print_json(result)
    except Exception:
        raise typer.Exit(1)


@app.command("list")
def list_collections():
    """Lists all available collections."""
    try:
        collections = api.list_collections()
        print_table("Collections", collections)
    except Exception:
        raise typer.Exit(1)


@app.command("get")
def get_collection(
    identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to retrieve."
    )
):
    """Gets details for a specific collection."""
    try:
        collection = api.get_collection(identifier)
        print_json(collection)
    except Exception:
        raise typer.Exit(1)


@app.command("delete")
def delete_collection(
    identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to delete."
    )
):
    """Deletes a collection."""
    try:
        result = api.delete_collection(identifier)
        print_json(result)
    except Exception:
        raise typer.Exit(1)
