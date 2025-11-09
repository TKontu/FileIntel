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
        "auto", "--type", "-t", help="RAG type: 'vector', 'graph', 'global', 'local', or 'auto'."
    ),
    answer_format: Optional[str] = typer.Option(
        "default",
        "--format",
        "-f",
        help="Answer format: 'default', 'single_paragraph', 'table', 'list', 'json', 'essay', or 'markdown'."
    ),
):
    """Query a collection with a question using RAG."""

    def _query(api):
        # Map CLI rag_type to API search_type
        search_type_map = {
            "auto": "adaptive",
            "vector": "vector",
            "graph": "graph",
            "global": "global",
            "local": "local"
        }
        search_type = search_type_map.get(rag_type, "adaptive")

        payload = {
            "question": question,
            "search_type": search_type,
            "answer_format": answer_format
        }
        return api._request(
            "POST", f"collections/{collection_identifier}/query", json=payload
        )

    result = cli_handler.handle_api_call(_query, "query collection")
    response_data = result.get("data", result)

    # Check if this is an async response (contains task_id)
    if "task_id" in response_data:
        task_id = response_data["task_id"]
        cli_handler.console.print(
            f"[blue]Query submitted as task {task_id[:8]}... Waiting for completion...[/blue]"
        )

        # Wait for task to complete
        api = cli_handler.get_api_client()
        task_result = api.wait_for_task_completion(task_id, show_progress=True)

        # Extract result from completed task
        task_data = task_result.get("data", {})
        if task_data.get("status") == "SUCCESS":
            response_data = task_data.get("result", {})
        else:
            cli_handler.console.print(
                f"[bold red]Query task failed:[/bold red] {task_data.get('error', 'Unknown error')}"
            )
            raise typer.Exit(1)

    cli_handler.console.print(f"[bold blue]Query:[/bold blue] {question}")
    cli_handler.console.print(
        f"[bold blue]Collection:[/bold blue] {collection_identifier}"
    )
    cli_handler.console.print(
        f"[bold blue]RAG Type:[/bold blue] {response_data.get('rag_type', response_data.get('query_type', rag_type))}"
    )
    cli_handler.console.print()

    answer = response_data.get("answer", "No answer provided")
    cli_handler.console.print(f"[bold green]Answer:[/bold green] {answer}")

    # Show sources if available
    sources = response_data.get("sources", [])
    if sources:
        cli_handler.console.print(f"\n[bold blue]Sources ({len(sources)}):[/bold blue]")
        for i, source in enumerate(sources, 1):
            # Use in-text citation (with page numbers) if available, otherwise fallback
            in_text = source.get("in_text_citation")
            full_citation = source.get("citation", source.get("filename", "Unknown"))
            relevance = source.get("similarity_score", source.get("relevance_score", 0))

            # Display format: "In-text citation" - Full reference (relevance)
            if in_text:
                cli_handler.console.print(f"  {i}. {in_text} - {full_citation} (relevance: {relevance:.3f})")
            else:
                cli_handler.console.print(f"  {i}. {full_citation} (relevance: {relevance:.3f})")


@app.command("document")
def query_document(
    collection_identifier: str = typer.Argument(
        ..., help="The collection ID containing the document."
    ),
    document_id: str = typer.Argument(..., help="The ID of the document to query."),
    question: str = typer.Argument(..., help="The question to ask about the document."),
    answer_format: Optional[str] = typer.Option(
        "default",
        "--format",
        "-f",
        help="Answer format: 'default', 'single_paragraph', 'table', 'list', 'json', 'essay', or 'markdown'."
    ),
):
    """Query a specific document with a question."""

    def _query_doc(api):
        payload = {
            "question": question,
            "search_type": "vector",  # Document queries use vector search
            "answer_format": answer_format
        }
        return api._request(
            "POST",
            f"collections/{collection_identifier}/documents/{document_id}/query",
            json=payload,
        )

    result = cli_handler.handle_api_call(_query_doc, "query document")
    response_data = result.get("data", result)

    # Check if this is an async response (contains task_id)
    if "task_id" in response_data:
        task_id = response_data["task_id"]
        cli_handler.console.print(
            f"[blue]Query submitted as task {task_id[:8]}... Waiting for completion...[/blue]"
        )

        # Wait for task to complete
        api = cli_handler.get_api_client()
        task_result = api.wait_for_task_completion(task_id, show_progress=True)

        # Extract result from completed task
        task_data = task_result.get("data", {})
        if task_data.get("status") == "SUCCESS":
            response_data = task_data.get("result", {})
        else:
            cli_handler.console.print(
                f"[bold red]Query task failed:[/bold red] {task_data.get('error', 'Unknown error')}"
            )
            raise typer.Exit(1)

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
    answer_format: Optional[str] = typer.Option(
        "default",
        "--format",
        "-f",
        help="Answer format: 'default', 'single_paragraph', 'table', 'list', 'json', 'essay', or 'markdown'."
    ),
):
    """Test the query system with a simple question."""
    cli_handler.console.print(
        f"[blue]Testing query system with collection '{collection_identifier}'...[/blue]"
    )

    try:

        def _test_query(api):
            payload = {
                "question": test_question,
                "search_type": "adaptive",
                "answer_format": answer_format
            }
            return api._request(
                "POST", f"collections/{collection_identifier}/query", json=payload
            )

        result = cli_handler.handle_api_call(_test_query, "test query")
        response_data = result.get("data", result)

        # Check if this is an async response (contains task_id)
        if "task_id" in response_data:
            task_id = response_data["task_id"]
            cli_handler.console.print(
                f"[blue]Test query submitted as task {task_id[:8]}... Waiting for completion...[/blue]"
            )

            # Wait for task to complete
            api = cli_handler.get_api_client()
            task_result = api.wait_for_task_completion(task_id, show_progress=True)

            # Extract result from completed task
            task_data = task_result.get("data", {})
            if task_data.get("status") == "SUCCESS":
                response_data = task_data.get("result", {})
            else:
                raise Exception(
                    f"Query task failed: {task_data.get('error', 'Unknown error')}"
                )

        cli_handler.display_success("Query system is working correctly")
        cli_handler.console.print(
            f"Test answer: {response_data.get('answer', 'No answer')}"
        )

    except Exception as e:
        cli_handler.display_error(f"Query system test failed: {e}")
        raise typer.Exit(1)
