"""
Plagiarism detection CLI commands.

Provides commands for analyzing documents for potential plagiarism
by comparing them against reference collections.
"""

import typer
from typing import Optional
from rich.table import Table
from rich.console import Console

from .shared import (
    cli_handler,
    get_entity_by_identifier,
)

app = typer.Typer(help="Plagiarism detection operations.")
console = Console()


@app.command("analyze")
def analyze_plagiarism(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the reference collection to search against."
    ),
    document_id: str = typer.Argument(
        ..., help="Document ID to analyze for plagiarism."
    ),
    min_similarity: float = typer.Option(
        0.7,
        "--min-similarity",
        "-s",
        help="Minimum similarity threshold to flag potential plagiarism (0.0-1.0).",
    ),
    chunk_overlap: float = typer.Option(
        0.3,
        "--chunk-overlap",
        "-c",
        help="Minimum fraction of chunks that must match to report a source (0.0-1.0).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed matched chunks for each source.",
    ),
):
    """
    Analyze a document for potential plagiarism.

    Compares the specified document against all sources in the reference collection
    to identify potential plagiarism. The document must already be uploaded and
    processed (with embeddings generated).

    **Typical Workflow:**
    1. Create collection for document to analyze (e.g., "student-paper")
    2. Upload document and process it (generates chunks + embeddings)
    3. Create/use collection with reference sources (e.g., "reference-papers")
    4. Run this command to analyze document against reference collection

    **Risk Levels:**
    - HIGH: ≥50% of document flagged as suspicious
    - MEDIUM: 20-49% flagged
    - LOW: 5-19% flagged
    - NONE: <5% flagged

    Examples:

        # Basic plagiarism analysis
        fileintel plagiarism analyze reference-papers doc-abc-123

        # More sensitive detection (lower threshold)
        fileintel plagiarism analyze papers doc-123 --min-similarity 0.6

        # Require higher overlap to report sources
        fileintel plagiarism analyze papers doc-123 --chunk-overlap 0.5

        # Show detailed matched chunks
        fileintel plagiarism analyze papers doc-123 --verbose
    """
    # Validate thresholds
    if not 0.0 <= min_similarity <= 1.0:
        cli_handler.display_error("min-similarity must be between 0.0 and 1.0")
        raise typer.Exit(1)

    if not 0.0 <= chunk_overlap <= 1.0:
        cli_handler.display_error("chunk-overlap must be between 0.0 and 1.0")
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
        "document_id": document_id,
        "min_similarity": min_similarity,
        "chunk_overlap_factor": chunk_overlap,
        "include_sources": verbose,  # Include detailed matches if verbose
        "group_by_source": True,
    }

    # Submit plagiarism analysis (async task)
    def _analyze_plagiarism(api):
        return api._request(
            "POST",
            f"collections/{collection_id}/analyze-plagiarism",
            json=payload
        )

    result = cli_handler.handle_api_call(_analyze_plagiarism, "submit plagiarism analysis task")
    data = result.get("data", result)

    # Check if this is an async response (contains task_id)
    if "task_id" in data:
        task_id = data["task_id"]
        cli_handler.console.print(
            f"[blue]Plagiarism analysis task submitted ({task_id[:8]}...). Waiting for completion...[/blue]\n"
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
            cli_handler.display_error(f"Plagiarism analysis failed: {error_msg}")
            raise typer.Exit(1)

    # Display results
    analyzed_filename = data.get("analyzed_filename", "Unknown")
    total_chunks = data.get("total_chunks", 0)
    flagged_chunks = data.get("flagged_chunks_count", 0)
    suspicious_pct = data.get("suspicious_percentage", 0.0)
    risk_level = data.get("overall_plagiarism_risk", "unknown").upper()
    matches = data.get("matches", [])

    # Display header
    cli_handler.console.print(f"\n[bold blue]📊 Plagiarism Analysis Report[/bold blue]")
    cli_handler.console.print(f"[dim]Document: {analyzed_filename}[/dim]")
    cli_handler.console.print(f"[dim]Reference Collection: {collection.get('name')}[/dim]\n")

    # Display overall statistics
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value")

    stats_table.add_row("Total Chunks Analyzed", str(total_chunks))
    stats_table.add_row("Flagged Chunks", str(flagged_chunks))
    stats_table.add_row("Suspicious Percentage", f"{suspicious_pct:.1f}%")

    # Color-code risk level
    risk_color = {
        "HIGH": "red",
        "MEDIUM": "yellow",
        "LOW": "green",
        "NONE": "green"
    }.get(risk_level, "white")

    stats_table.add_row(
        "Overall Risk",
        f"[{risk_color}]{risk_level}[/{risk_color}]"
    )

    cli_handler.console.print(stats_table)

    # Display warning based on risk level
    if risk_level == "HIGH":
        cli_handler.console.print(
            f"\n[bold red]⚠️  HIGH RISK: {suspicious_pct:.1f}% of document shows potential plagiarism![/bold red]"
        )
    elif risk_level == "MEDIUM":
        cli_handler.console.print(
            f"\n[bold yellow]⚠️  MEDIUM RISK: {suspicious_pct:.1f}% of document shows potential plagiarism[/bold yellow]"
        )
    elif risk_level == "LOW":
        cli_handler.console.print(
            f"\n[green]✓ LOW RISK: {suspicious_pct:.1f}% of document shows minor similarities[/green]"
        )
    else:
        cli_handler.console.print(
            f"\n[bold green]✓ CLEAN: No significant plagiarism detected[/bold green]"
        )

    # Display matched sources
    if matches:
        cli_handler.console.print(f"\n[bold cyan]Matching Sources ({len(matches)} found):[/bold cyan]")

        for i, match in enumerate(matches, 1):
            source_filename = match.get("source_filename", "Unknown")
            match_pct = match.get("match_percentage", 0.0)
            avg_similarity = match.get("average_similarity", 0.0)
            matched_chunks = match.get("matched_chunks", [])

            # Color-code match percentage
            if match_pct >= 50:
                match_color = "red"
            elif match_pct >= 20:
                match_color = "yellow"
            else:
                match_color = "green"

            cli_handler.console.print(
                f"\n[bold]{i}. {source_filename}[/bold]"
            )
            cli_handler.console.print(
                f"   Match: [{match_color}]{match_pct:.1f}%[/{match_color}] "
                f"| Avg Similarity: {avg_similarity:.3f}"
            )

            # Show detailed matched chunks if verbose
            if verbose and matched_chunks:
                cli_handler.console.print(f"   [dim]Top matching chunks:[/dim]")
                for j, chunk_match in enumerate(matched_chunks[:3], 1):  # Show top 3
                    similarity = chunk_match.get("similarity", 0.0)
                    analyzed_text = chunk_match.get("analyzed_chunk_text", "")
                    source_text = chunk_match.get("source_chunk_text", "")
                    source_page = chunk_match.get("source_page")

                    page_info = f" (p.{source_page})" if source_page else ""

                    cli_handler.console.print(
                        f"   [{j}] Similarity: {similarity:.3f}{page_info}"
                    )
                    cli_handler.console.print(
                        f"       Analyzed: [dim]{analyzed_text[:80]}...[/dim]"
                    )
                    cli_handler.console.print(
                        f"       Source: [dim]{source_text[:80]}...[/dim]"
                    )

    else:
        cli_handler.console.print(
            f"\n[green]✓ No matching sources found above threshold[/green]"
        )

    # Display summary
    if risk_level in ["HIGH", "MEDIUM"]:
        cli_handler.console.print(
            f"\n[yellow]⚠️  Review flagged sections carefully. "
            f"Ensure proper citations for similar content.[/yellow]"
        )

    cli_handler.display_success("Plagiarism analysis completed")
