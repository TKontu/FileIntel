import typer

from . import collections, documents, jobs
from .client import FileIntelAPI, print_json

app = typer.Typer(
    name="fileintel",
    help="A CLI for interacting with the FileIntel API.",
    add_completion=False,
)

api = FileIntelAPI()

app.add_typer(collections.app, name="collections", help="Manage collections.")
app.add_typer(documents.app, name="documents", help="Manage documents in collections.")
app.add_typer(jobs.app, name="jobs", help="Check the status and results of jobs.")


@app.command(epilog="Example: fileintel query my-collection 'What is the main topic?'")
def query(
    collection: str = typer.Argument(
        ..., help="The name or ID of the collection to query."
    ),
    question: str = typer.Argument(..., help="The question to ask the collection."),
):
    """Submits a query to a collection."""
    try:
        result = api.query_collection(collection, question)
        print_json(result)
    except Exception:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
