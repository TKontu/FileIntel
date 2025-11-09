"""
Citation generation CLI commands.

Provides commands for generating Harvard-style citations for text segments
using vector similarity search.
"""

import typer
from typing import Optional
from rich.table import Table

from .shared import (
    cli_handler,
    get_entity_by_identifier,
)

app = typer.Typer(help="Citation generation operations.")


@app.command("collection")
def cite_collection(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to search for citations."
    ),
    text: str = typer.Argument(
        ..., help="Text segment that needs citation (10-5000 characters)."
    ),
    min_similarity: Optional[float] = typer.Option(
        None,
        "--min-similarity",
        "-s",
        help="Minimum similarity threshold (0.0-1.0, default from config).",
    ),
    document_id: Optional[str] = typer.Option(
        None,
        "--document",
        "-d",
        help="Restrict search to specific document ID.",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Number of candidate sources to retrieve (default from config).",
    ),
    with_analysis: bool = typer.Option(
        False,
        "--with-analysis",
        "-a",
        help="Include LLM-based relevance analysis (adds latency).",
    ),
    show_source: bool = typer.Option(
        True,
        "--show-source/--no-source",
        help="Show source details in output (default: True).",
    ),
):
    """
    Generate citation for a text segment.

    Searches the specified collection for the most similar source document
    and generates Harvard-style citations.

    Examples:

        # Basic citation
        fileintel cite collection my-collection "Machine learning models learn patterns from data"

        # With minimum similarity threshold
        fileintel cite collection papers "Deep learning requires large datasets" --min-similarity 0.8

        # Restrict to specific document
        fileintel cite collection library "Quantum entanglement" --document doc-123

        # Include LLM analysis
        fileintel cite collection thesis "Neural networks" --with-analysis

        # Copy just the in-text citation
        fileintel cite collection papers "Some text" --no-source
    """
    # Validate text length
    if len(text) < 10:
        cli_handler.display_error("Text segment must be at least 10 characters")
        raise typer.Exit(1)

    if len(text) > 5000:
        cli_handler.display_error("Text segment must not exceed 5000 characters")
        raise typer.Exit(1)

    # Get collection to resolve identifier
    def _get_collection(api):
        return get_entity_by_identifier(
            api,
            "collections",
            collection_identifier,
            "collection"
        )

    collection = cli_handler.handle_api_call(_get_collection, "get collection")
    collection_id = collection.get("id")

    # Build request payload
    payload = {
        "text_segment": text,
        "include_llm_analysis": with_analysis,
    }

    if document_id:
        payload["document_id"] = document_id
    if min_similarity is not None:
        payload["min_similarity"] = min_similarity
    if top_k is not None:
        payload["top_k"] = top_k

    # Generate citation
    def _generate_citation(api):
        return api._request(
            "POST",
            f"citations/collections/{collection_id}/generate-citation",
            json=payload
        )

    result = cli_handler.handle_api_call(_generate_citation, "generate citation")
    data = result.get("data", result)

    # Display results
    citation = data.get("citation", {})
    source = data.get("source", {})
    confidence = data.get("confidence", "unknown")
    warning = data.get("warning")
    relevance_note = data.get("relevance_note")

    # Display in-text citation prominently
    cli_handler.console.print(f"\n[bold green]In-Text Citation:[/bold green]")
    cli_handler.console.print(f"  {citation.get('in_text', 'N/A')}")

    # Display full citation
    cli_handler.console.print(f"\n[bold blue]Full Citation:[/bold blue]")
    full_citation = citation.get('full', 'N/A')
    # Wrap long citations
    if len(full_citation) > 100:
        # Split by '. ' to preserve sentence boundaries
        parts = full_citation.split('. ')
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                cli_handler.console.print(f"  {part}.")
            else:
                cli_handler.console.print(f"  {part}")
    else:
        cli_handler.console.print(f"  {full_citation}")

    # Display confidence
    confidence_color = {
        "high": "green",
        "medium": "yellow",
        "low": "red"
    }.get(confidence.lower(), "white")

    cli_handler.console.print(f"\n[bold]Confidence:[/bold] [{confidence_color}]{confidence.upper()}[/{confidence_color}]")

    # Display warning if present
    if warning:
        cli_handler.console.print(f"\n[yellow]⚠ Warning:[/yellow] {warning}")

    # Display source details if requested
    if show_source:
        cli_handler.console.print(f"\n[bold cyan]Source Details:[/bold cyan]")

        # Create table for source info
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Document ID", source.get("document_id", "N/A"))
        table.add_row("Filename", source.get("filename", "N/A"))
        table.add_row(
            "Similarity Score",
            f"{source.get('similarity_score', 0.0):.3f}"
        )

        # Show page numbers if available
        page_numbers = source.get("page_numbers", [])
        if page_numbers:
            pages_str = ", ".join(str(p) for p in page_numbers)
            table.add_row("Page Numbers", pages_str)

        cli_handler.console.print(table)

        # Show text excerpt
        excerpt = source.get("text_excerpt", "")
        if excerpt:
            cli_handler.console.print(f"\n[bold]Text Excerpt:[/bold]")
            # Wrap excerpt
            max_width = 100
            if len(excerpt) > max_width:
                for i in range(0, len(excerpt), max_width):
                    cli_handler.console.print(f"  {excerpt[i:i+max_width]}")
            else:
                cli_handler.console.print(f"  {excerpt}")

    # Display relevance analysis if present
    if relevance_note:
        cli_handler.console.print(f"\n[bold magenta]Relevance Analysis:[/bold magenta]")
        cli_handler.console.print(f"  {relevance_note}")

    cli_handler.display_success("Citation generated successfully")


@app.command("config")
def show_config():
    """
    Show current citation generation configuration.

    Displays the system's default settings for citation generation,
    including similarity thresholds, confidence levels, and LLM settings.
    """
    def _get_config(api):
        # Use a dummy collection (the config endpoint doesn't actually use it)
        return api._request("GET", "citations/collections/dummy/citation-config")

    result = cli_handler.handle_api_call(_get_config, "get citation configuration")
    config = result.get("data", result)

    cli_handler.console.print(f"\n[bold blue]Citation Generation Configuration[/bold blue]")

    # Create table
    table = Table(show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Minimum Similarity", str(config.get("min_similarity", "N/A")))
    table.add_row("Default Top K", str(config.get("default_top_k", "N/A")))
    table.add_row("Max Excerpt Length", str(config.get("max_excerpt_length", "N/A")))
    table.add_row(
        "LLM Analysis Enabled",
        str(config.get("enable_llm_analysis", "N/A"))
    )
    table.add_row(
        "LLM Analysis Model",
        config.get("llm_analysis_model", "N/A")
    )

    cli_handler.console.print(table)

    # Show confidence thresholds
    thresholds = config.get("confidence_thresholds", {})
    if thresholds:
        cli_handler.console.print(f"\n[bold]Confidence Thresholds:[/bold]")
        threshold_table = Table(show_header=True)
        threshold_table.add_column("Level", style="cyan")
        threshold_table.add_column("Threshold", style="green")

        for level, value in thresholds.items():
            threshold_table.add_row(level.upper(), f"{value:.2f}")

        cli_handler.console.print(threshold_table)
