"""
Consolidated GraphRAG CLI commands.

Streamlined GraphRAG operations using shared CLI utilities.
"""

import typer
import os
from typing import Optional
from pathlib import Path

from .shared import (
    cli_handler,
    check_system_status,
    monitor_task_with_progress,
    get_entity_by_identifier,
)

app = typer.Typer(help="GraphRAG operations.")


@app.command("index")
def index_collection(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to index."
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for indexing to complete."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-indexing even if index exists."
    ),
):
    """Create GraphRAG index for a collection."""

    def _index(api):
        payload = {"collection_id": collection_identifier, "force_rebuild": force}
        return api._request("POST", f"graphrag/index", json=payload)

    result = cli_handler.handle_api_call(_index, "index collection for GraphRAG")
    task_data = result.get("data", result)

    cli_handler.display_success("GraphRAG indexing started")
    cli_handler.display_task_info(task_data)

    if wait:
        task_id = task_data.get("task_id")
        if task_id:
            monitor_task_with_progress(task_id, "GraphRAG indexing")


@app.command("query")
def query_with_graphrag(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to query."
    ),
    question: str = typer.Argument(..., help="The question to ask using GraphRAG."),
    show_sources: bool = typer.Option(
        False, "--show-sources", "-s", help="Show source documents with page numbers."
    ),
):
    """Query a collection using GraphRAG for graph-based reasoning."""

    def _graphrag_query(api):
        payload = {"question": question, "search_type": "graph"}  # Force GraphRAG
        return api._request(
            "POST", f"collections/{collection_identifier}/query", json=payload
        )

    result = cli_handler.handle_api_call(_graphrag_query, "GraphRAG query")
    response_data = result.get("data", result)

    cli_handler.console.print(f"[bold blue]GraphRAG Query:[/bold blue] {question}")
    cli_handler.console.print(
        f"[bold blue]Collection:[/bold blue] {collection_identifier}"
    )
    cli_handler.console.print()

    # Handle both "answer" (from service wrapper) and "response" (from direct GraphRAG)
    answer = response_data.get("answer") or response_data.get("response", "No answer provided")

    # If answer is a dict (raw GraphRAG response), extract the response text
    if isinstance(answer, dict):
        answer = answer.get("response", str(answer))

    # Convert citations if --show-sources flag is set
    display_answer = answer
    if show_sources:
        converted_answer, sources = _display_source_documents(answer, collection_identifier, cli_handler)
        if converted_answer:
            display_answer = converted_answer

    cli_handler.console.print(f"[bold green]Answer:[/bold green]")

    # Display the answer (don't use Markdown renderer to avoid centered headers)
    # Instead, just print with proper formatting preserved
    cli_handler.console.print(display_answer)

    # Show context information if available (from GraphRAG response)
    context = response_data.get("context", {})

    # Show community reports if available
    if "reports" in context:
        reports = context["reports"]
        if reports is not None and len(reports) > 0:
            cli_handler.console.print(
                f"\n[bold blue]Community Reports Used ({len(reports)}):[/bold blue]"
            )
            # Display top 5 reports
            for i, (_, report) in enumerate(list(reports.head(5).iterrows()), 1):
                title = report.get("title", "Unknown Community")
                rank = report.get("rank", 0)
                cli_handler.console.print(
                    f"  {i}. {title} [dim](rank: {rank:.1f})[/dim]"
                )

            if len(reports) > 5:
                cli_handler.console.print(f"  [dim]... and {len(reports) - 5} more[/dim]")

    # Show GraphRAG-specific information (legacy format)
    entities = response_data.get("entities", [])
    if entities:
        cli_handler.console.print(
            f"\n[bold blue]Related Entities ({len(entities)}):[/bold blue]"
        )
        for entity in entities[:5]:  # Show top 5 entities
            name = entity.get("name", "Unknown")
            entity_type = entity.get("type", "Unknown")
            cli_handler.console.print(f"  • {name} ({entity_type})")

    communities = response_data.get("communities", [])
    if communities:
        cli_handler.console.print(
            f"\n[bold blue]Related Communities ({len(communities)}):[/bold blue]"
        )
        for community in communities[:3]:  # Show top 3 communities
            title = community.get("title", "Unknown")
            size = community.get("size", 0)
            level = community.get("level", 0)
            cli_handler.console.print(f"  • {title} (level: {level}, size: {size})")


@app.command("status")
def graphrag_status(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to check."
    )
):
    """Get GraphRAG index status for a collection."""

    def _get_status(api):
        return api._request("GET", f"graphrag/{collection_identifier}/status")

    status_data = cli_handler.handle_api_call(_get_status, "get GraphRAG status")
    cli_handler.display_json(
        status_data.get("data", status_data),
        f"GraphRAG Status: {collection_identifier}",
    )


@app.command("entities")
def list_entities(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection."
    ),
    limit: Optional[int] = typer.Option(
        20, "--limit", "-l", help="Maximum number of entities to show."
    ),
):
    """List GraphRAG entities for a collection."""

    def _get_entities(api):
        params = {"limit": limit} if limit else {}
        return api._request(
            "GET", f"graphrag/{collection_identifier}/entities", params=params
        )

    entities_data = cli_handler.handle_api_call(_get_entities, "get GraphRAG entities")
    entities = entities_data.get("data", entities_data)

    if isinstance(entities, list) and entities:
        cli_handler.console.print(
            f"[bold blue]GraphRAG Entities ({len(entities)}):[/bold blue]"
        )
        for entity in entities:
            name = entity.get("name", "Unknown")
            entity_type = entity.get("type", "Unknown")
            importance = entity.get("importance_score", 0)
            description = entity.get("description", "")[:100]  # Truncate

            cli_handler.console.print(
                f"[bold]{name}[/bold] ({entity_type}) - Score: {importance:.2f}"
            )
            if description:
                cli_handler.console.print(f"  {description}...")
            cli_handler.console.print()
    else:
        cli_handler.console.print(
            f"[yellow]No entities found for collection '{collection_identifier}'[/yellow]"
        )


@app.command("communities")
def list_communities(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection."
    ),
    limit: Optional[int] = typer.Option(
        10, "--limit", "-l", help="Maximum number of communities to show."
    ),
):
    """List GraphRAG communities for a collection."""

    def _get_communities(api):
        params = {"limit": limit} if limit else {}
        return api._request(
            "GET", f"graphrag/{collection_identifier}/communities", params=params
        )

    communities_data = cli_handler.handle_api_call(
        _get_communities, "get GraphRAG communities"
    )
    communities = communities_data.get("data", communities_data)

    if isinstance(communities, list) and communities:
        cli_handler.console.print(
            f"[bold blue]GraphRAG Communities ({len(communities)}):[/bold blue]"
        )
        for community in communities:
            title = community.get("title", "Unknown")
            level = community.get("level", 0)
            summary = community.get("summary", "")[:150]  # Truncate
            size = community.get("size", 0)
            community_id = community.get("community_id", "N/A")

            cli_handler.console.print(
                f"[bold]{title}[/bold] (ID: {community_id}, Level: {level}, Size: {size})"
            )
            if summary:
                cli_handler.console.print(f"  {summary}...")
            cli_handler.console.print()
    else:
        cli_handler.console.print(
            f"[yellow]No communities found for collection '{collection_identifier}'[/yellow]"
        )


@app.command("rebuild")
def rebuild_index(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection."
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for rebuild to complete."
    ),
    confirm: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt."
    ),
):
    """Rebuild GraphRAG index for a collection (removes existing index)."""
    if not confirm:
        if not typer.confirm(
            f"Are you sure you want to rebuild the GraphRAG index for '{collection_identifier}'?"
        ):
            cli_handler.console.print("Operation cancelled.")
            return

    # First remove existing index
    def _remove_index(api):
        return api._request("DELETE", f"graphrag/{collection_identifier}/index")

    try:
        cli_handler.handle_api_call(_remove_index, "remove existing GraphRAG index")
        cli_handler.console.print("[yellow]Existing index removed[/yellow]")
    except:
        # Index might not exist, continue with rebuild
        pass

    # Then create new index
    def _rebuild(api):
        payload = {"collection_id": collection_identifier, "force_rebuild": True}
        return api._request("POST", f"graphrag/index", json=payload)

    result = cli_handler.handle_api_call(_rebuild, "rebuild GraphRAG index")
    task_data = result.get("data", result)

    cli_handler.display_success("GraphRAG index rebuild started")
    cli_handler.display_task_info(task_data)

    if wait:
        task_id = task_data.get("task_id")
        if task_id:
            monitor_task_with_progress(task_id, "GraphRAG index rebuild")


@app.command("system-status")
def system_status():
    """Check GraphRAG system status."""
    check_system_status("GraphRAG", "graphrag/status")


@app.command("workspace")
def explore_workspace(
    collection_identifier: str = typer.Argument(
        ..., help="The name or ID of the collection."
    )
):
    """Explore GraphRAG workspace files and data for a collection."""

    def _get_collection_info(api):
        return api._request("GET", f"collections/{collection_identifier}")

    def _get_index_info(api):
        return api._request("GET", f"graphrag/{collection_identifier}/status")

    # Get collection and index information
    collection_data = cli_handler.handle_api_call(_get_collection_info, "get collection info")
    collection = collection_data.get("data", collection_data)

    index_data = cli_handler.handle_api_call(_get_index_info, "get GraphRAG index info")
    index_info = index_data.get("data", index_data)

    collection_id = collection.get("id")
    collection_name = collection.get("name", "Unknown")

    cli_handler.console.print(f"[bold blue]GraphRAG Workspace for '{collection_name}'[/bold blue]")
    cli_handler.console.print(f"Collection ID: {collection_id}")

    # Display index information
    if index_info and index_info.get("index_path"):
        index_path = index_info.get("index_path")
        cli_handler.console.print(f"Index Path: {index_path}")
        cli_handler.console.print(f"Status: {index_info.get('index_status', 'unknown')}")
        cli_handler.console.print(f"Documents: {index_info.get('documents_count', 0)}")
        cli_handler.console.print(f"Entities: {index_info.get('entities_count', 0)}")
        cli_handler.console.print(f"Communities: {index_info.get('communities_count', 0)}")

        # Check for workspace files
        workspace_path = Path(index_path).parent
        if workspace_path.exists():
            cli_handler.console.print(f"\n[bold green]Workspace Files:[/bold green]")

            # Look for common GraphRAG files
            output_dir = workspace_path / "output"
            if output_dir.exists():
                parquet_files = list(output_dir.glob("*.parquet"))
                if parquet_files:
                    cli_handler.console.print("\n[cyan]Parquet Data Files:[/cyan]")
                    for file in sorted(parquet_files):
                        size = file.stat().st_size / 1024  # KB
                        cli_handler.console.print(f"  {file.name} ({size:.1f} KB)")

                csv_files = list(output_dir.glob("*.csv"))
                if csv_files:
                    cli_handler.console.print("\n[cyan]CSV Files:[/cyan]")
                    for file in sorted(csv_files):
                        size = file.stat().st_size / 1024  # KB
                        cli_handler.console.print(f"  {file.name} ({size:.1f} KB)")

                other_files = [f for f in output_dir.iterdir() if f.is_file() and f.suffix not in ['.parquet', '.csv']]
                if other_files:
                    cli_handler.console.print("\n[cyan]Other Files:[/cyan]")
                    for file in sorted(other_files):
                        size = file.stat().st_size / 1024  # KB
                        cli_handler.console.print(f"  {file.name} ({size:.1f} KB)")
            else:
                cli_handler.console.print(f"[yellow]Output directory not found: {output_dir}[/yellow]")
        else:
            cli_handler.console.print(f"[yellow]Workspace directory not accessible: {workspace_path}[/yellow]")
    else:
        cli_handler.console.print("[yellow]No GraphRAG index found for this collection[/yellow]")

    # Display available commands
    cli_handler.console.print(f"\n[bold green]Available Commands:[/bold green]")
    cli_handler.console.print(f"  fileintel graphrag entities {collection_identifier}")
    cli_handler.console.print(f"  fileintel graphrag communities {collection_identifier}")
    cli_handler.console.print(f"  fileintel query graphrag-global '{collection_identifier}' 'your question'")
    cli_handler.console.print(f"  fileintel query graphrag-local '{collection_identifier}' 'your question'")


def _parse_citation_ids(answer_text):
    """Parse inline citations and extract specific IDs."""
    import re

    citation_pattern = r'\[Data: (Reports|Entities|Relationships) \(([0-9, ]+)\)\]'
    citations = re.findall(citation_pattern, answer_text)

    parsed = {
        "report_ids": set(),
        "entity_ids": set(),
        "relationship_ids": set()
    }

    for cit_type, ids_str in citations:
        ids = [int(x.strip()) for x in ids_str.split(',')]

        if cit_type == "Reports":
            parsed["report_ids"].update(ids)
        elif cit_type == "Entities":
            parsed["entity_ids"].update(ids)
        elif cit_type == "Relationships":
            parsed["relationship_ids"].update(ids)

    return parsed


def _convert_to_harvard_citations(answer_text, sources):
    """Convert GraphRAG numbered citations to Harvard-style citations.

    Args:
        answer_text: Text with [Data: Reports (5)] style citations
        sources: List of source documents with metadata

    Returns:
        Text with (Author, Year) style citations
    """
    import re

    if not sources:
        return answer_text

    # Build citations from document metadata
    citations = []
    for i, source in enumerate(sources[:10], 1):  # Top 10 sources
        metadata = source.get("metadata", {})
        page_numbers = source.get("page_numbers")  # Now formatted as "p. 45" or "pp. 16-17" etc.

        # Extract citation from metadata (authors, year, title)
        authors = metadata.get("authors", [])
        author_surnames = metadata.get("author_surnames", [])

        # Extract year from various metadata fields
        year = metadata.get("publication_year") or metadata.get("year")
        if not year and "publication_date" in metadata:
            # Extract year from publication_date (e.g., "2021-03" -> "2021")
            pub_date = str(metadata["publication_date"])
            year = pub_date[:4] if len(pub_date) >= 4 else None

        title = metadata.get("title", source.get("document", "Unknown"))

        # Build citation in Harvard format (Surname et al., Year)
        # Use author_surnames if available (LLM-extracted), otherwise use full author names
        citation_names = author_surnames if author_surnames else authors

        if citation_names and year:
            # Format: "Surname et al., Year"
            if len(citation_names) == 1:
                citation = f"{citation_names[0]}, {year}"
            elif len(citation_names) == 2:
                citation = f"{citation_names[0]} & {citation_names[1]}, {year}"
            else:
                citation = f"{citation_names[0]} et al., {year}"
        elif year:
            # Has year but no authors
            citation = f"{title}, {year}"
        else:
            # Fallback: use title
            citation = title.replace('.pdf', '')

        # Add page numbers if available (already formatted)
        if page_numbers:
            citation = f"{citation}, {page_numbers}"

        citations.append(citation)

    # Replace numbered citations with Harvard style
    # Simple approach: replace all [Data: ...] with first citation
    # More sophisticated: map specific IDs to specific sources
    if citations:
        primary_citation = citations[0]
        # Remove all [Data: ...] citations and replace with primary source
        result = re.sub(
            r'\s*\[Data: (?:Reports|Entities|Relationships) \([0-9, ]+\)\]',
            f' ({primary_citation})',
            answer_text
        )
        return result

    return answer_text


def _display_source_documents(answer_text, collection_identifier, cli_handler):
    """Display source documents traced from GraphRAG answer.

    Returns:
        tuple: (converted_answer_text, sources) or (answer_text, None) if tracing fails
    """

    # Phase 1: Parse citation IDs
    citation_ids = _parse_citation_ids(answer_text)

    if not citation_ids["report_ids"] and not citation_ids["entity_ids"]:
        cli_handler.console.print("\n[dim]Note: GraphRAG response contains no inline citations to trace[/dim]")
        return answer_text, None

    # Display citation summary
    cli_handler.console.print("\n[bold blue]GraphRAG Source References:[/bold blue]")
    if citation_ids["report_ids"]:
        cli_handler.console.print(f"  • Community Reports: {len(citation_ids['report_ids'])} referenced")
    if citation_ids["entity_ids"]:
        cli_handler.console.print(f"  • Entities: {len(citation_ids['entity_ids'])} referenced")
    if citation_ids["relationship_ids"]:
        cli_handler.console.print(f"  • Relationships: {len(citation_ids['relationship_ids'])} referenced")

    # Get workspace path
    def _get_index_info(api):
        return api._request("GET", f"graphrag/{collection_identifier}/status")

    try:
        index_result = cli_handler.handle_api_call(_get_index_info, "get GraphRAG index info")
        index_data = index_result.get("data", index_result)
        workspace_path = index_data.get("index_path")

        if not workspace_path:
            cli_handler.console.print("\n[yellow]No GraphRAG index found for source tracing[/yellow]")
            return answer_text, None

        # Transform Docker path to local filesystem path
        # API runs in Docker (/data/graphrag_indices/...) but CLI runs locally (./graphrag_indices/...)
        if workspace_path.startswith("/data/graphrag_indices/"):
            workspace_path = workspace_path.replace("/data/graphrag_indices/", "./graphrag_indices/graphrag_indices/")

        # Phase 2-5: Trace to source documents
        from fileintel.rag.graph_rag.utils.source_tracer import trace_citations_to_sources

        sources = trace_citations_to_sources(
            citation_ids,
            workspace_path,
            cli_handler.get_api_client()
        )

        # Convert to Harvard citations
        converted_answer = _convert_to_harvard_citations(answer_text, sources)

        # Display sources
        _display_sources(sources, cli_handler)

        return converted_answer, sources

    except Exception as e:
        cli_handler.console.print(f"\n[yellow]Could not trace sources: {e}[/yellow]")
        return answer_text, None


def _display_sources(sources, cli_handler):
    """Display source documents in CLI."""
    if not sources:
        cli_handler.console.print("\n[dim]No source documents found[/dim]")
        return

    cli_handler.console.print(f"\n[bold blue]Source Documents ({len(sources)}):[/bold blue]")

    for i, source in enumerate(sources[:10], 1):
        doc = source.get("document", "Unknown")
        page = source.get("page_number")
        chunk_count = source.get("chunk_count", 1)

        if page:
            cli_handler.console.print(f"\n  [{i}] {doc}, p. {page}")
        else:
            cli_handler.console.print(f"\n  [{i}] {doc}")

        if chunk_count > 1:
            cli_handler.console.print(f"      [dim]({chunk_count} chunks referenced)[/dim]")

    if len(sources) > 10:
        cli_handler.console.print(f"\n  [dim]... and {len(sources) - 10} more documents[/dim]")
