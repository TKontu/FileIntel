"""
Consolidated query CLI commands.

Streamlined querying operations using shared CLI utilities.
"""

import typer
from typing import Optional

from .shared import cli_handler, check_system_status

app = typer.Typer(help="Query collections.")


@app.command("collection")
def query_collection(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to query."
    ),
    question: str = typer.Argument(
        ..., help="The question to ask about the collection."
    ),
    rag_type: Optional[str] = typer.Option(
        "auto", "--type", "-t", help="RAG type: 'vector', 'graph', or 'auto'."
    ),
):
    """Query a collection with a question using RAG."""

    def _query(api):
        # Map CLI rag_type to API search_type
        search_type_map = {"auto": "adaptive", "vector": "vector", "graph": "graph"}
        search_type = search_type_map.get(rag_type, "adaptive")

        payload = {"question": question, "search_type": search_type}
        return api._request(
            "POST", f"collections/{collection_identifier}/query", json=payload
        )

    result = cli_handler.handle_api_call(_query, "query collection")
    response_data = result.get("data", result)

    cli_handler.console.print(f"[bold blue]Query:[/bold blue] {question}")
    cli_handler.console.print(
        f"[bold blue]Collection:[/bold blue] {collection_identifier}"
    )
    cli_handler.console.print(
        f"[bold blue]RAG Type:[/bold blue] {response_data.get('rag_type', rag_type)}"
    )
    cli_handler.console.print()

    answer = response_data.get("answer", "No answer provided")
    cli_handler.console.print(f"[bold green]Answer:[/bold green] {answer}")

    # Show sources if available
    sources = response_data.get("sources", [])
    if sources:
        cli_handler.console.print(f"\n[bold blue]Sources ({len(sources)}):[/bold blue]")
        for i, source in enumerate(sources, 1):
            filename = source.get("filename", "Unknown")
            relevance = source.get("relevance_score", 0)
            cli_handler.console.print(f"  {i}. {filename} (relevance: {relevance:.3f})")


@app.command("document")
def query_document(
    collection_identifier: str = typer.Argument(
        ..., help="The collection ID containing the document."
    ),
    document_id: str = typer.Argument(..., help="The ID of the document to query."),
    question: str = typer.Argument(..., help="The question to ask about the document."),
):
    """Query a specific document with a question."""

    def _query_doc(api):
        payload = {
            "question": question,
            "search_type": "vector",  # Document queries use vector search
        }
        return api._request(
            "POST",
            f"collections/{collection_identifier}/documents/{document_id}/query",
            json=payload,
        )

    result = cli_handler.handle_api_call(_query_doc, "query document")
    response_data = result.get("data", result)

    cli_handler.console.print(f"[bold blue]Query:[/bold blue] {question}")
    cli_handler.console.print(
        f"[bold blue]Collection:[/bold blue] {collection_identifier}"
    )
    cli_handler.console.print(f"[bold blue]Document:[/bold blue] {document_id}")
    cli_handler.console.print()

    answer = response_data.get("answer", "No answer provided")
    cli_handler.console.print(f"[bold green]Answer:[/bold green] {answer}")

    # Show document context from sources if available
    sources = response_data.get("sources", [])
    if sources:
        cli_handler.console.print(f"\n[bold blue]Relevant Context ({len(sources)} chunks):[/bold blue]")
        for i, source in enumerate(sources, 1):
            text = source.get("text", "")
            similarity = source.get("similarity_score", 0)
            cli_handler.console.print(f"  {i}. {text} (similarity: {similarity:.3f})")


@app.command("system-status")
def system_status():
    """Check query system status and available operations."""
    check_system_status("query", "query/status")


@app.command("test")
def test_query_system(
    collection_identifier: str = typer.Argument(..., help="Collection to test with."),
    test_question: str = typer.Option(
        "What is this collection about?", "--question", "-q", help="Test question."
    ),
):
    """Test the query system with a simple question."""
    cli_handler.console.print(
        f"[blue]Testing query system with collection '{collection_identifier}'...[/blue]"
    )

    try:

        def _test_query(api):
            payload = {"question": test_question, "search_type": "adaptive"}
            return api._request(
                "POST", f"collections/{collection_identifier}/query", json=payload
            )

        result = cli_handler.handle_api_call(_test_query, "test query")
        response_data = result.get("data", result)

        cli_handler.display_success("Query system is working correctly")
        cli_handler.console.print(
            f"Test answer: {response_data.get('answer', 'No answer')}"
        )

    except Exception as e:
        cli_handler.display_error(f"Query system test failed: {e}")
        raise typer.Exit(1)
