import typer
from rich.console import Console
from .client import FileIntelAPI, print_json

app = typer.Typer(help="Ask questions using RAG.")
api = FileIntelAPI()
console = Console()


@app.command("from-collection")
def query_collection(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to query."
    ),
    question: str = typer.Argument(..., help="The question to ask."),
):
    """Queries an entire collection with a specific question."""
    try:
        result = api.query_collection(collection_identifier, question)
        print_json(result)
    except Exception:
        raise typer.Exit(1)


@app.command("from-document")
def query_document(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection containing the document."
    ),
    document_identifier: str = typer.Argument(
        ..., help="The filename or ID of the document to query."
    ),
    question: str = typer.Argument(..., help="The question to ask."),
):
    """Queries a single document within a collection with a specific question."""
    try:
        result = api.query_document(
            collection_identifier, document_identifier, question
        )
        print_json(result)
    except Exception:
        raise typer.Exit(1)
