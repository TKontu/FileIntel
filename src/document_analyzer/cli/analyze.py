import typer
from rich.console import Console
from .client import FileIntelAPI, print_json

app = typer.Typer(help="Perform template-driven analysis.")
api = FileIntelAPI()
console = Console()


@app.command("from-collection")
def analyze_collection(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to analyze."
    ),
    task_name: str = typer.Option(
        "default_analysis", "--task-name", "-t", help="The analysis task to run."
    ),
):
    """Analyzes an entire collection using a prompt template."""
    try:
        result = api.analyze_collection(collection_identifier, task_name)
        print_json(result)
    except Exception:
        raise typer.Exit(1)


@app.command("from-document")
def analyze_document(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection containing the document."
    ),
    document_identifier: str = typer.Argument(
        ..., help="The filename or ID of the document to analyze."
    ),
    task_name: str = typer.Option(
        "default_analysis", "--task-name", "-t", help="The analysis task to run."
    ),
):
    """Analyzes a single document within a collection using a prompt template."""
    try:
        result = api.analyze_document(
            collection_identifier, document_identifier, task_name
        )
        print_json(result)
    except Exception:
        raise typer.Exit(1)
