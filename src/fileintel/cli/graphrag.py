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

    cli_handler.console.print(f"[bold green]Answer:[/bold green]")

    # Display the answer (don't use Markdown renderer to avoid centered headers)
    # Instead, just print with proper formatting preserved
    cli_handler.console.print(answer)

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
