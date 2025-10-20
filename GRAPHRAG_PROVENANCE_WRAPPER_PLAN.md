# GraphRAG Provenance Wrapper - Implementation Plan

## Architecture

**Principle:** Don't modify GraphRAG internals. Add a wrapper that enriches responses with source provenance.

```
User Query
    ↓
New: graphrag query-with-sources  (CLI wrapper)
    ↓
Existing: GraphRAG query (unchanged)
    ↓
New: Provenance Enrichment Layer
    ↓
Response with Sources
```

## Implementation Approach

### 1. New CLI Command

Create a separate command that wraps the existing GraphRAG query:

```bash
# Existing (unchanged):
fileintel graphrag query test "what is agile"

# New (with provenance):
fileintel graphrag query-with-sources test "what is agile"
```

**File:** `src/fileintel/cli/graphrag.py`

```python
@app.command("query-with-sources")
def query_with_sources(
    collection_identifier: str = typer.Argument(...),
    question: str = typer.Argument(...),
    max_sources: int = typer.Option(10, "--max-sources", "-n", help="Max sources to show"),
):
    """Query using GraphRAG and show source document provenance."""

    # Step 1: Call existing GraphRAG query endpoint
    def _graphrag_query(api):
        payload = {"question": question, "search_type": "graph"}
        return api._request("POST", f"collections/{collection_identifier}/query", json=payload)

    result = cli_handler.handle_api_call(_graphrag_query, "GraphRAG query")
    response_data = result.get("data", result)

    # Step 2: Call new provenance enrichment endpoint
    def _get_provenance(api):
        payload = {
            "collection_id": collection_identifier,
            "graphrag_response": response_data
        }
        return api._request("POST", f"graphrag/provenance", json=payload)

    provenance_result = cli_handler.handle_api_call(_get_provenance, "get provenance")
    provenance_data = provenance_result.get("data", provenance_result)

    # Step 3: Display answer
    cli_handler.console.print(f"[bold blue]GraphRAG Query:[/bold blue] {question}")
    cli_handler.console.print(f"[bold blue]Collection:[/bold blue] {collection_identifier}\n")

    answer = response_data.get("answer") or response_data.get("response", "No answer")
    if isinstance(answer, dict):
        answer = answer.get("response", str(answer))

    cli_handler.console.print(f"[bold green]Answer:[/bold green]")
    cli_handler.console.print(answer)

    # Step 4: Display sources with provenance
    sources = provenance_data.get("sources", [])
    if sources:
        cli_handler.console.print(f"\n[bold blue]Source Documents ({len(sources)}):[/bold blue]")
        for i, source in enumerate(sources[:max_sources], 1):
            doc_name = source.get("document", "Unknown")
            page = source.get("page_number")
            entity = source.get("entity_name", "")
            chunk_preview = source.get("text_preview", "")[:100]

            if page:
                cli_handler.console.print(f"\n  [{i}] {doc_name}, p. {page}")
            else:
                cli_handler.console.print(f"\n  [{i}] {doc_name}")

            if entity:
                cli_handler.console.print(f"      [dim]Via entity: {entity}[/dim]")

            if chunk_preview:
                cli_handler.console.print(f"      [dim]\"{chunk_preview}...\"[/dim]")

        if len(sources) > max_sources:
            cli_handler.console.print(f"\n  [dim]... and {len(sources) - max_sources} more sources[/dim]")
```

### 2. New API Route

**File:** `src/fileintel/api/routes/graphrag.py` (new file)

```python
from fastapi import APIRouter, Depends
from typing import Dict, Any, List
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.api.dependencies import get_storage
from fileintel.api.utils import ApiResponseV2, create_success_response
from fileintel.core.config import get_config

router = APIRouter(prefix="/graphrag", tags=["graphrag"])


@router.post("/provenance", response_model=ApiResponseV2)
async def get_graphrag_provenance(
    request: Dict[str, Any],
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """
    Extract source provenance from a GraphRAG response.

    Takes a GraphRAG response and traces back through the knowledge graph
    to find the original source documents and pages that support the answer.
    """
    collection_id = request.get("collection_id")
    graphrag_response = request.get("graphrag_response", {})

    # Initialize provenance service
    from fileintel.rag.graph_rag.services.provenance_service import ProvenanceService

    config = get_config()
    provenance_service = ProvenanceService(storage, config)

    # Extract sources
    sources = await provenance_service.extract_sources(
        collection_id=collection_id,
        graphrag_response=graphrag_response,
        max_sources=request.get("max_sources", 50)
    )

    return create_success_response({
        "sources": sources,
        "total_sources": len(sources),
        "collection_id": collection_id
    })
```

### 3. New Provenance Service

**File:** `src/fileintel/rag/graph_rag/services/provenance_service.py` (new file)

```python
import asyncio
import os
from typing import Dict, Any, List
import pandas as pd
from fileintel.storage.postgresql_storage import PostgreSQLStorage


class ProvenanceService:
    """Service for extracting source provenance from GraphRAG responses."""

    def __init__(self, storage: PostgreSQLStorage, config):
        self.storage = storage
        self.config = config

    async def extract_sources(
        self,
        collection_id: str,
        graphrag_response: Dict[str, Any],
        max_sources: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Extract source documents from GraphRAG response.

        Traces: GraphRAG Context → Communities → Entities → Text Units → Source Chunks
        """

        # Step 1: Get GraphRAG workspace path
        index_info = await asyncio.to_thread(
            self.storage.get_graphrag_index_info, collection_id
        )

        if not index_info or not index_info.get("index_path"):
            return []

        workspace_path = index_info["index_path"]

        # Step 2: Load necessary parquet files
        dataframes = await self._load_parquet_files(workspace_path)

        # Step 3: Extract context from GraphRAG response
        context = graphrag_response.get("context", {})

        # Step 4: Collect source text unit IDs from context
        text_unit_ids = set()

        # From community reports
        if "reports" in context:
            reports_df = context["reports"]
            if isinstance(reports_df, pd.DataFrame) and not reports_df.empty:
                # Get communities used
                community_ids = reports_df["community"].tolist() if "community" in reports_df.columns else []

                # Get entities in these communities
                if "communities" in dataframes:
                    communities_df = dataframes["communities"]
                    for comm_id in community_ids:
                        comm_row = communities_df[communities_df["id"] == comm_id]
                        if not comm_row.empty:
                            entity_ids = comm_row.iloc[0].get("entity_ids", [])

                            # Get text units for these entities
                            if "entities" in dataframes:
                                entities_df = dataframes["entities"]
                                for entity_id in entity_ids:
                                    entity_row = entities_df[entities_df["id"] == entity_id]
                                    if not entity_row.empty:
                                        unit_ids = entity_row.iloc[0].get("text_unit_ids", [])
                                        text_unit_ids.update(unit_ids)

        # From entities (if provided directly)
        if "entities" in context:
            entities_list = context["entities"]
            if isinstance(entities_list, list):
                for entity in entities_list:
                    unit_ids = entity.get("text_unit_ids", [])
                    text_unit_ids.update(unit_ids)

        # Step 5: Look up source chunks from text units
        sources = await self._resolve_text_units_to_sources(
            text_unit_ids,
            dataframes.get("text_units"),
            collection_id
        )

        # Step 6: Deduplicate and limit
        sources = self._deduplicate_sources(sources)
        return sources[:max_sources]

    async def _load_parquet_files(self, workspace_path: str) -> Dict[str, pd.DataFrame]:
        """Load GraphRAG parquet files."""
        files_to_load = {
            "entities": "entities.parquet",
            "communities": "communities.parquet",
            "text_units": "text_units.parquet",
            "relationships": "relationships.parquet",
        }

        dataframes = {}
        for name, filename in files_to_load.items():
            path = os.path.join(workspace_path, filename)
            if os.path.exists(path):
                df = await asyncio.to_thread(pd.read_parquet, path)
                dataframes[name] = df

        return dataframes

    async def _resolve_text_units_to_sources(
        self,
        text_unit_ids: set,
        text_units_df: pd.DataFrame,
        collection_id: str
    ) -> List[Dict[str, Any]]:
        """Map text unit IDs back to FileIntel chunks."""

        if text_units_df is None or text_units_df.empty:
            return []

        sources = []

        for unit_id in text_unit_ids:
            # Find text unit in parquet
            unit_row = text_units_df[text_units_df["id"] == unit_id]

            if unit_row.empty:
                continue

            unit = unit_row.iloc[0]

            # The text unit ID should match FileIntel chunk UUID
            # Look up the chunk in PostgreSQL
            chunk = await asyncio.to_thread(
                self.storage.get_chunk_by_id, unit_id
            )

            if chunk:
                sources.append({
                    "chunk_id": str(chunk.id),
                    "document": chunk.document.original_filename,
                    "document_id": str(chunk.document_id),
                    "page_number": chunk.chunk_metadata.get("page_number") if chunk.chunk_metadata else None,
                    "text_preview": chunk.chunk_text[:200],
                    "entity_name": None,  # Could enhance to track which entity
                })

        return sources

    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources, keeping unique document+page combinations."""
        seen = set()
        unique_sources = []

        for source in sources:
            key = (source["document_id"], source.get("page_number"))
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)

        return unique_sources
```

### 4. Storage Method Addition

**File:** `src/fileintel/storage/postgresql_storage.py`

Add method to retrieve chunk by UUID:

```python
def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
    """Get a chunk by its UUID."""
    return self.session.query(DocumentChunk).filter(
        DocumentChunk.id == chunk_id
    ).first()
```

### 5. Register New Route

**File:** `src/fileintel/api/main.py`

```python
# Add import
from fileintel.api.routes import graphrag

# Register router
app.include_router(graphrag.router, prefix="/api/v2")
```

## Usage Examples

### CLI Usage

```bash
# Standard GraphRAG query (unchanged)
fileintel graphrag query test "what is agile"

# GraphRAG with source provenance
fileintel graphrag query-with-sources test "what is agile"

# Limit number of sources shown
fileintel graphrag query-with-sources test "what is agile" --max-sources 5
```

### Expected Output

```
GraphRAG Query: what is agile
Collection: test

Answer:
Agile methodologies are increasingly central to modern product development
across various sectors. They represent a significant shift away from
traditional waterfall approaches...

Source Documents (10):

  [1] Cooper, R. (2018). Agile-Stage-Gate Hybrids.pdf, p. 17
      Via entity: AGILE
      "Agile methodologies have become central to NPD processes..."

  [2] Cooper, R. (2018). Agile-Stage-Gate Hybrids.pdf, p. 18
      Via entity: NEW PRODUCT DEVELOPMENT
      "The integration of Agile into Stage-Gate systems..."

  [3] Schwaber, K. (2020). Scrum Guide.pdf, p. 5
      Via entity: SCRUM
      "Scrum is the most widely used Agile framework..."

  ... and 7 more sources
```

### API Usage

```bash
# Step 1: Query GraphRAG
curl -X POST '/api/v2/collections/test/query' \
  -H 'Content-Type: application/json' \
  -d '{"question": "what is agile", "search_type": "graph"}'

# Returns: {data: {answer: "...", context: {...}}}

# Step 2: Get provenance
curl -X POST '/api/v2/graphrag/provenance' \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_id": "test",
    "graphrag_response": {...},  # Response from step 1
    "max_sources": 10
  }'

# Returns: {data: {sources: [...], total_sources: 10}}
```

## Advantages of This Approach

1. **Non-invasive:** GraphRAG internals remain unchanged
2. **Optional:** Users can still use regular `graphrag query` without provenance
3. **Modular:** Provenance service is separate, can be reused
4. **API-first:** Both CLI and API users can access provenance
5. **Maintainable:** Updates to GraphRAG don't break provenance
6. **Extensible:** Easy to add more enrichments (verification, confidence scores, etc.)

## Implementation Effort

- **ProvenanceService:** 2-3 hours
- **API route:** 30 minutes
- **CLI command:** 1 hour
- **Storage method:** 15 minutes
- **Testing:** 1-2 hours

**Total:** ~5-7 hours

## Next Steps

1. Create `provenance_service.py` with source extraction logic
2. Add API route for `/graphrag/provenance`
3. Add CLI command `query-with-sources`
4. Test with actual GraphRAG index
5. (Optional) Add caching for performance

Would you like me to start implementing this?
