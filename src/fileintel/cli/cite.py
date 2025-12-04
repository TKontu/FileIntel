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


def _handle_candidates_mode(
    collection_id: str,
    text: str,
    min_similarity: Optional[float],
    document_id: Optional[str],
    num_candidates: int,
    show_source: bool
):
    """
    Handle citation candidates mode - displays multiple candidates for user selection.
    """
    # Build request payload
    payload = {
        "text_segment": text,
        "num_candidates": num_candidates,
    }

    if document_id:
        payload["document_id"] = document_id
    if min_similarity is not None:
        payload["min_similarity"] = min_similarity

    # Generate citation candidates (async task)
    def _generate_candidates(api):
        return api._request(
            "POST",
            f"citations/collections/{collection_id}/citation-candidates",
            json=payload
        )

    result = cli_handler.handle_api_call(_generate_candidates, "submit citation candidates task")
    data = result.get("data", result)

    # Check if this is an async response (contains task_id)
    if "task_id" in data:
        task_id = data["task_id"]
        cli_handler.console.print(
            f"[blue]Citation candidates task submitted ({task_id[:8]}...). Waiting for completion...[/blue]\n"
        )

        # Wait for task to complete
        api = cli_handler.get_api_client()
        task_result = api.wait_for_task_completion(task_id, show_progress=True)

        # Extract result from completed task
        task_data = task_result.get("data", {})
        if task_data.get("status") == "SUCCESS":
            data = task_data.get("result", {})
        else:
            error_msg = task_data.get("error", "Unknown error")
            cli_handler.display_error(f"Citation candidates generation failed: {error_msg}")
            raise typer.Exit(1)

    # Display results
    candidates = data.get("candidates", [])
    total_found = data.get("total_found", 0)

    if not candidates:
        cli_handler.display_error("No citation candidates found above similarity threshold")
        raise typer.Exit(1)

    cli_handler.console.print(f"\n[bold blue]Found {total_found} matches, showing top {len(candidates)} candidates:[/bold blue]\n")

    # Display each candidate
    for i, candidate in enumerate(candidates, 1):
        citation = candidate.get("citation", {})
        source = candidate.get("source", {})
        confidence = candidate.get("confidence", "unknown")
        warning = candidate.get("warning")

        # Confidence color
        confidence_color = {
            "high": "green",
            "medium": "yellow",
            "low": "red"
        }.get(confidence.lower(), "white")

        # Header for candidate
        cli_handler.console.print(f"[bold cyan]━━━ Candidate {i} ━━━[/bold cyan]")

        # In-text citation
        cli_handler.console.print(f"[bold green]In-Text:[/bold green] {citation.get('in_text', 'N/A')}")

        # Full citation
        cli_handler.console.print(f"[bold blue]Full:[/bold blue] {citation.get('full', 'N/A')}")

        # Confidence and similarity
        similarity = source.get('similarity_score', 0.0)
        cli_handler.console.print(
            f"[bold]Confidence:[/bold] [{confidence_color}]{confidence.upper()}[/{confidence_color}] "
            f"(similarity: {similarity:.3f})"
        )

        # Warning if present
        if warning:
            cli_handler.console.print(f"[yellow]⚠ {warning}[/yellow]")

        # Source details if requested
        if show_source:
            cli_handler.console.print(f"\n[dim]Source: {source.get('filename', 'N/A')}[/dim]")

            # Page numbers if available
            page_numbers = source.get("page_numbers", [])
            if page_numbers:
                pages_str = ", ".join(str(p) for p in page_numbers)
                cli_handler.console.print(f"[dim]Pages: {pages_str}[/dim]")

            # Full text excerpt (not truncated)
            excerpt = source.get("text_excerpt", "")
            if excerpt:
                cli_handler.console.print(f"\n[bold]Reference Text:[/bold]")
                # Word wrap for readability
                cli_handler.console.print(f"[dim]{excerpt}[/dim]")

        cli_handler.console.print("")  # Blank line between candidates

    cli_handler.display_success(f"Generated {len(candidates)} citation candidates")


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
    candidates: Optional[int] = typer.Option(
        None,
        "--candidates",
        "-c",
        help="Show N citation candidates for selection instead of just the best match (1-10).",
    ),
):
    """
    Generate citation for a text segment.

    Searches the specified collection for the most similar source document
    and generates Harvard-style citations.

    Examples:

        # Basic citation (returns best match)
        fileintel cite collection my-collection "Machine learning models learn patterns from data"

        # Show 3 citation candidates for selection
        fileintel cite collection thesis "Neural networks" --candidates 3

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

    # Validate candidates option
    if candidates is not None:
        if candidates < 1 or candidates > 10:
            cli_handler.display_error("--candidates must be between 1 and 10")
            raise typer.Exit(1)

    # Get collection to resolve identifier
    collection = get_entity_by_identifier(
        "collection",
        collection_identifier,
        "get_collection"
    )
    collection_id = collection.get("id")

    # Branch based on whether candidates mode is requested
    if candidates is not None:
        # Use citation candidates endpoint
        _handle_candidates_mode(
            collection_id=collection_id,
            text=text,
            min_similarity=min_similarity,
            document_id=document_id,
            num_candidates=candidates,
            show_source=show_source
        )
        return

    # Standard single citation mode
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

    # Generate citation (async task)
    def _generate_citation(api):
        return api._request(
            "POST",
            f"citations/collections/{collection_id}/generate-citation",
            json=payload
        )

    result = cli_handler.handle_api_call(_generate_citation, "submit citation task")
    data = result.get("data", result)

    # Check if this is an async response (contains task_id)
    if "task_id" in data:
        task_id = data["task_id"]
        cli_handler.console.print(
            f"[blue]Citation task submitted ({task_id[:8]}...). Waiting for completion...[/blue]\n"
        )

        # Wait for task to complete
        api = cli_handler.get_api_client()
        task_result = api.wait_for_task_completion(task_id, show_progress=True)

        # Extract result from completed task
        task_data = task_result.get("data", {})
        if task_data.get("status") == "SUCCESS":
            data = task_data.get("result", {})
        else:
            error_msg = task_data.get("error", "Unknown error")
            cli_handler.display_error(f"Citation generation failed: {error_msg}")
            raise typer.Exit(1)

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
    table.add_row("Min Chunk Length", str(config.get("min_chunk_length", "N/A")))
    max_excerpt = config.get("max_excerpt_length", 0)
    table.add_row("Max Excerpt Length", "No limit" if max_excerpt == 0 else str(max_excerpt))
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


@app.command("inject")
def inject_citation(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to search for citations."
    ),
    text: str = typer.Argument(
        ..., help="Text segment to annotate with citation (10-10000 characters)."
    ),
    style: str = typer.Option(
        "footnote",
        "--style",
        "-s",
        help="Citation injection style: inline, footnote, endnote, or markdown_link."
    ),
    min_similarity: Optional[float] = typer.Option(
        None,
        "--min-similarity",
        "-m",
        help="Minimum similarity threshold (0.0-1.0, default from config)."
    ),
    document_id: Optional[str] = typer.Option(
        None,
        "--document",
        "-d",
        help="Restrict search to specific document ID."
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Number of candidate sources to retrieve (default from config)."
    ),
    full: bool = typer.Option(
        False,
        "--full",
        "-f",
        help="Include full citation text in footnote/endnote/markdown styles."
    ),
    show_source: bool = typer.Option(
        True,
        "--show-source/--no-source",
        help="Show source details in output (default: True)."
    ),
):
    """
    Inject citation into text segment with various styles.

    Searches the specified collection for the most similar source document
    and injects a Harvard-style citation directly into the text.

    **Injection Styles:**
    - **inline**: Appends citation at end: "Text. (Author, Year)"
    - **footnote**: Adds superscript: "Text.¹" + "[1] Full citation"
    - **endnote**: Adds endnote: "Text.[dn1]" + "[dn1] Full citation"
    - **markdown_link**: Creates link: "Text [(Author, Year)](#source)"

    Examples:

        # Basic citation injection (footnote style)
        fileintel cite inject my-collection "Machine learning models learn patterns from data"

        # Inline citation
        fileintel cite inject papers "Deep learning requires large datasets" --style inline

        # Footnote with full citation text
        fileintel cite inject thesis "Neural networks" --style footnote --full

        # Markdown link style
        fileintel cite inject docs "Quantum entanglement" --style markdown_link --full
    """
    # Validate text length
    if len(text) < 10:
        cli_handler.display_error("Text segment must be at least 10 characters")
        raise typer.Exit(1)

    if len(text) > 10000:
        cli_handler.display_error("Text segment must not exceed 10,000 characters")
        raise typer.Exit(1)

    # Validate injection style
    valid_styles = ['inline', 'footnote', 'endnote', 'markdown_link']
    if style not in valid_styles:
        cli_handler.display_error(f"Invalid style. Must be one of: {', '.join(valid_styles)}")
        raise typer.Exit(1)

    # Get collection to resolve identifier
    collection = get_entity_by_identifier(
        "collection",
        collection_identifier,
        "get_collection"
    )
    collection_id = collection.get("id")

    # Build request payload
    payload = {
        "text_segment": text,
        "insertion_style": style,
        "include_full_citation": full,
    }

    if document_id:
        payload["document_id"] = document_id
    if min_similarity is not None:
        payload["min_similarity"] = min_similarity
    if top_k is not None:
        payload["top_k"] = top_k

    # Inject citation (async task)
    def _inject_citation(api):
        return api._request(
            "POST",
            f"citations/collections/{collection_id}/inject-citation",
            json=payload
        )

    result = cli_handler.handle_api_call(_inject_citation, "submit citation injection task")
    data = result.get("data", result)

    # Check if this is an async response (contains task_id)
    if "task_id" in data:
        task_id = data["task_id"]
        cli_handler.console.print(
            f"[blue]Citation injection task submitted ({task_id[:8]}...). Waiting for completion...[/blue]\n"
        )

        # Wait for task to complete
        api = cli_handler.get_api_client()
        task_result = api.wait_for_task_completion(task_id, show_progress=True)

        # Extract result from completed task
        task_data = task_result.get("data", {})
        if task_data.get("status") == "SUCCESS":
            data = task_data.get("result", {})
        else:
            error_msg = task_data.get("error", "Unknown error")
            cli_handler.display_error(f"Citation injection failed: {error_msg}")
            raise typer.Exit(1)

    # Display annotated text prominently
    cli_handler.console.print(f"\n[bold green]✨ Annotated Text:[/bold green]")
    cli_handler.console.print(f"[white]{data.get('annotated_text', 'N/A')}[/white]")

    # Display citation details
    citation = data.get("citation", {})
    confidence = data.get("confidence", "unknown")

    cli_handler.console.print(f"\n[bold blue]Citation Details:[/bold blue]")
    cli_handler.console.print(f"  In-Text: {citation.get('in_text', 'N/A')}")
    cli_handler.console.print(f"  Full: {citation.get('full', 'N/A')}")
    cli_handler.console.print(f"  Style: {data.get('insertion_style', style)}")

    # Display confidence
    confidence_color = {
        "high": "green",
        "medium": "yellow",
        "low": "red"
    }.get(confidence.lower(), "white")

    cli_handler.console.print(f"\n[bold]Confidence:[/bold] [{confidence_color}]{confidence.upper()}[/{confidence_color}]")

    # Display character positions
    char_pos = data.get("character_positions", {})
    if char_pos:
        cli_handler.console.print(f"\n[dim]Citation inserted at characters {char_pos.get('start', 'N/A')}-{char_pos.get('end', 'N/A')}[/dim]")

    # Display source details if requested
    if show_source:
        source = data.get("source", {})
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

        cli_handler.console.print(table)

    # Display warning if present
    warning = data.get("warning")
    if warning:
        cli_handler.console.print(f"\n[yellow]⚠ Warning:[/yellow] {warning}")

    cli_handler.display_success("Citation injected successfully")
