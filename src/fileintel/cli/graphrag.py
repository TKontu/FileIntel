"""
Consolidated GraphRAG CLI commands.

Streamlined GraphRAG operations using shared CLI utilities.
"""

import typer
from typing import Optional

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

    answer = response_data.get("answer", "No answer provided")
    cli_handler.console.print(f"[bold green]Answer:[/bold green] {answer}")

    # Show GraphRAG-specific information
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
            rank = community.get("rank", 0)
            cli_handler.console.print(f"  • {title} (rank: {rank})")


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
            rank = community.get("rank", 0)
            summary = community.get("summary", "")[:150]  # Truncate
            size = community.get("size", 0)

            cli_handler.console.print(
                f"[bold]{title}[/bold] (Rank: {rank}, Size: {size})"
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
