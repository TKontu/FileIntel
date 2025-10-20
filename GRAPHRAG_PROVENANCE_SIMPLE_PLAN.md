# GraphRAG Provenance - Simple Implementation

## Architecture Review Against Quality Standards

### ❌ Original Plan Issues
- **Unnecessary abstraction**: ProvenanceService used in only 1 place
- **Wrapper anti-pattern**: CLI wrapping API wrapping another API
- **Over-engineering**: New service + route + command for simple functionality
- **Tight coupling**: Service depends on GraphRAG internals anyway

### ✅ Simple Solution

**Add ONE optional parameter to existing GraphRAG query:**

```python
fileintel graphrag query test "what is agile" --show-sources
```

That's it. No new commands, no new services, no new API routes.

## Implementation

### 1. Modify Existing CLI Command

**File:** `src/fileintel/cli/graphrag.py`

```python
@app.command("query")
def query_with_graphrag(
    collection_identifier: str = typer.Argument(...),
    question: str = typer.Argument(...),
    show_sources: bool = typer.Option(  # NEW: Just add this flag
        False, "--show-sources", "-s", help="Show source documents with page numbers"
    ),
):
    """Query a collection using GraphRAG for graph-based reasoning."""

    # Existing code unchanged...
    result = cli_handler.handle_api_call(_graphrag_query, "GraphRAG query")
    response_data = result.get("data", result)

    # ... existing display code ...

    # NEW: Add source extraction if requested
    if show_sources:
        sources = _extract_sources_from_context(
            response_data,
            collection_identifier,
            cli_handler
        )
        _display_sources(sources, cli_handler)
```

### 2. Add Helper Function (in same file)

**Single responsibility:** Extract source chunks from GraphRAG context

```python
def _extract_sources_from_context(
    response_data: Dict[str, Any],
    collection_identifier: str,
    cli_handler
) -> List[Dict[str, Any]]:
    """
    Extract source documents from GraphRAG response context.

    Traces: Context → Text Units → FileIntel Chunks
    Returns: List of {document, page_number, text_preview}
    """
    context = response_data.get("context", {})

    # Get text unit IDs from context
    text_unit_ids = set()

    # From reports → communities → entities → text units
    if "reports" in context:
        reports_df = context["reports"]
        if isinstance(reports_df, pd.DataFrame) and not reports_df.empty:
            # Load GraphRAG parquet to trace sources
            index_info = cli_handler.get_api_client().get_graphrag_index_info(
                collection_identifier
            )
            workspace_path = index_info.get("index_path")

            if workspace_path:
                text_unit_ids = _get_text_units_from_reports(
                    reports_df,
                    workspace_path
                )

    # Look up chunks in PostgreSQL
    if not text_unit_ids:
        return []

    api = cli_handler.get_api_client()
    sources = []

    for unit_id in text_unit_ids:
        # Call existing API: GET /chunks/{chunk_id}
        try:
            chunk = api._request("GET", f"chunks/{unit_id}")
            chunk_data = chunk.get("data", chunk)

            sources.append({
                "document": chunk_data.get("document", {}).get("original_filename", "Unknown"),
                "page_number": chunk_data.get("chunk_metadata", {}).get("page_number"),
                "text_preview": chunk_data.get("chunk_text", "")[:150],
            })
        except:
            continue

    return _deduplicate_sources(sources)


def _get_text_units_from_reports(reports_df: pd.DataFrame, workspace_path: str) -> set:
    """Trace reports → communities → entities → text units."""
    import os
    import pandas as pd

    text_unit_ids = set()

    # Load necessary parquet files
    entities_path = os.path.join(workspace_path, "entities.parquet")
    communities_path = os.path.join(workspace_path, "communities.parquet")

    if not os.path.exists(entities_path) or not os.path.exists(communities_path):
        return text_unit_ids

    entities_df = pd.read_parquet(entities_path)
    communities_df = pd.read_parquet(communities_path)

    # Get community IDs from reports
    community_ids = reports_df["community"].tolist() if "community" in reports_df.columns else []

    # For each community, get entities
    for comm_id in community_ids:
        comm_row = communities_df[communities_df["id"] == comm_id]
        if not comm_row.empty:
            entity_ids = comm_row.iloc[0].get("entity_ids", [])

            # For each entity, get text units
            for entity_id in entity_ids:
                entity_row = entities_df[entities_df["id"] == entity_id]
                if not entity_row.empty:
                    unit_ids = entity_row.iloc[0].get("text_unit_ids", [])
                    text_unit_ids.update(unit_ids)

    return text_unit_ids


def _deduplicate_sources(sources: List[Dict]) -> List[Dict]:
    """Remove duplicate document+page combinations."""
    seen = set()
    unique = []

    for source in sources:
        key = (source["document"], source.get("page_number"))
        if key not in seen:
            seen.add(key)
            unique.append(source)

    return unique


def _display_sources(sources: List[Dict], cli_handler):
    """Display source documents in CLI."""
    if not sources:
        return

    cli_handler.console.print(f"\n[bold blue]Source Documents ({len(sources)}):[/bold blue]")

    for i, source in enumerate(sources[:10], 1):
        doc = source["document"]
        page = source.get("page_number")
        preview = source.get("text_preview", "")

        if page:
            cli_handler.console.print(f"  [{i}] {doc}, p. {page}")
        else:
            cli_handler.console.print(f"  [{i}] {doc}")

        if preview:
            cli_handler.console.print(f"      [dim]\"{preview}...\"[/dim]")

    if len(sources) > 10:
        cli_handler.console.print(f"  [dim]... and {len(sources) - 10} more[/dim]")
```

## Quality Assessment

### ✅ Meets Standards

1. **Single Responsibility**
   - `_extract_sources_from_context`: Extract sources
   - `_get_text_units_from_reports`: Trace parquet data
   - `_deduplicate_sources`: Remove duplicates
   - `_display_sources`: Display in CLI

2. **No Unnecessary Abstractions**
   - No new service class
   - No new API route
   - Helper functions in same file (used only here)

3. **Clear Separation**
   - Data access: Read parquet files, call API
   - Business logic: Trace sources
   - Presentation: Display sources

4. **Justified Existence**
   - Each function has a clear, single purpose
   - No wrappers around wrappers
   - All code is actually used

5. **Maintainability**
   - Simple to understand
   - Easy to debug
   - No complex dependencies

6. **Compatibility**
   - Uses existing API endpoints (GET /chunks/{id})
   - Uses existing GraphRAG parquet structure
   - Optional flag doesn't break existing usage

### Trade-offs

**Cons of simple approach:**
- Helper functions in CLI file (could argue they should be in a utility module)
- Reads parquet files directly from CLI (mixing concerns)

**Counter-arguments:**
- Functions are private (`_`) and used only in this file
- Parquet reading is unavoidable - GraphRAG stores data there
- Moving to separate module would be premature abstraction (only 1 use case)

## Alternative: Even Simpler

If reading parquet files from CLI feels wrong, we could:

**Option A: Add to existing GraphRAG service response**

Modify `graphrag_service.py` to optionally include sources:

```python
# graphrag_service.py
async def global_query(self, collection_id: str, query: str, include_sources: bool = False):
    raw_response = await self.global_search(query, collection_id)

    result = {
        "answer": raw_response.get("response"),
        "context": raw_response.get("context", {}),
    }

    if include_sources:
        # Extract text unit IDs from context
        text_unit_ids = self._extract_text_unit_ids(raw_response.get("context"))

        # Look up chunks
        sources = []
        for unit_id in text_unit_ids:
            chunk = self.storage.get_chunk_by_id(unit_id)
            if chunk:
                sources.append({
                    "document": chunk.document.original_filename,
                    "page_number": chunk.chunk_metadata.get("page_number"),
                })

        result["sources"] = sources

    return result
```

**Pros:**
- Cleaner separation (GraphRAG service handles GraphRAG data)
- CLI just displays what API returns
- Reusable if API users want sources too

**Cons:**
- Modifies GraphRAG service (you said don't touch it)
- Adds optional parameter that may rarely be used

## Recommendation

**Use the simple CLI-based approach** (first option) because:

1. ✅ Doesn't modify GraphRAG service
2. ✅ Minimal code additions
3. ✅ Clear, single-purpose functions
4. ✅ No unnecessary abstractions
5. ✅ Easy to test and debug

If we later need provenance in API/other places (2+ uses), **then** move logic to shared utility.

## Implementation Effort

- Add `--show-sources` flag: 5 minutes
- Write `_extract_sources_from_context`: 1 hour
- Write `_get_text_units_from_reports`: 1 hour
- Write helper functions: 30 minutes
- Test: 1 hour

**Total: ~3.5 hours**

Much simpler than the 7-hour over-engineered version.

---

## Answer to Your Question

**Will it work?** Yes, if we use the simple approach.

**Does it comply with quality standards?**

Original plan: ❌ No (unnecessary service, wrapper anti-pattern)

Simple approach: ✅ Yes (justified functions, no abstractions, clear responsibility)
