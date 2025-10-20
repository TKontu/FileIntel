# GraphRAG Source Tracing - Full Implementation Plan

## Current State (MVP)

✅ **Working**: `fileintel graphrag query test "question" --show-sources`
- Parses inline citations from GraphRAG answer: `[Data: Reports (5)]`
- Displays citation summary: "Community Reports: 7 referenced"
- Note: "Full source document tracing requires additional implementation"

## Goal

Enable complete source traceability from GraphRAG responses to specific document pages:

```bash
fileintel graphrag query test "what is agile" --show-sources

GraphRAG Source References:
  • Community Reports: 7 referenced

Source Documents (15):
  [1] Cooper, R. (2018). Agile-Stage-Gate Hybrids.pdf, p. 17
      "Agile methodologies have become central to NPD processes..."

  [2] Cooper, R. (2018). Agile-Stage-Gate Hybrids.pdf, p. 18
      "The integration of Agile into Stage-Gate systems..."

  [3] Schwaber, K. (2020). Scrum Guide.pdf, p. 5
      "Scrum is the most widely used Agile framework..."
```

## Implementation Strategy

### Phase 1: Citation ID Extraction (30 minutes)

**Goal**: Extract specific report/entity/relationship IDs from inline citations

**Current**:
```python
# Extracts: "7 reports referenced"
citations = re.findall(citation_pattern, answer_text)
```

**Enhanced**:
```python
# Extracts: report IDs [5, 3, 47, 10, ...]
def parse_citation_ids(answer_text):
    """Parse inline citations and extract specific IDs."""
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
```

**File**: `src/fileintel/cli/graphrag.py` - Update `_display_source_documents()`

---

### Phase 2: Report → Text Units Mapping (1 hour)

**Goal**: Load GraphRAG parquet files and trace report IDs to text_unit IDs

**Implementation**:

```python
def trace_reports_to_text_units(report_ids, workspace_path):
    """
    Trace report IDs to text unit IDs via communities and entities.

    Flow: report_ids → community_ids → entity_ids → text_unit_ids
    """
    import pandas as pd
    import os

    # Load parquet files
    reports_df = pd.read_parquet(os.path.join(workspace_path, "community_reports.parquet"))
    communities_df = pd.read_parquet(os.path.join(workspace_path, "communities.parquet"))
    entities_df = pd.read_parquet(os.path.join(workspace_path, "entities.parquet"))

    text_unit_ids = set()

    # Step 1: Get community IDs from report IDs
    community_ids = set()
    for report_id in report_ids:
        report_row = reports_df[reports_df['id'] == report_id]
        if not report_row.empty:
            community_ids.add(report_row.iloc[0]['community'])

    # Step 2: Get entity IDs from communities
    entity_ids = set()
    for community_id in community_ids:
        comm_row = communities_df[communities_df['id'] == community_id]
        if not comm_row.empty:
            ent_ids = comm_row.iloc[0].get('entity_ids', [])
            entity_ids.update(ent_ids)

    # Step 3: Get text unit IDs from entities
    for entity_id in entity_ids:
        entity_row = entities_df[entities_df['id'] == entity_id]
        if not entity_row.empty:
            tu_ids = entity_row.iloc[0].get('text_unit_ids', [])
            text_unit_ids.update(tu_ids)

    return text_unit_ids
```

**File**: `src/fileintel/rag/graph_rag/utils/source_tracer.py` - Add this function

---

### Phase 3: Text Units → Chunk UUIDs Mapping (30 minutes)

**Goal**: Map GraphRAG text_unit SHA512 IDs to FileIntel chunk UUIDs

**Implementation** (already exists, just use it):

```python
def map_text_units_to_chunks(text_unit_ids, workspace_path):
    """
    Map text_unit IDs (SHA512) to chunk UUIDs via text_units.parquet.

    Args:
        text_unit_ids: Set of GraphRAG text_unit IDs (SHA512 hashes)
        workspace_path: Path to GraphRAG workspace

    Returns:
        Set of FileIntel chunk UUIDs
    """
    import pandas as pd
    import os

    text_units_df = pd.read_parquet(os.path.join(workspace_path, "text_units.parquet"))

    chunk_uuids = set()
    for unit_id in text_unit_ids:
        tu_row = text_units_df[text_units_df['id'] == unit_id]
        if not tu_row.empty:
            doc_ids = tu_row.iloc[0].get('document_ids', [])
            chunk_uuids.update(doc_ids)

    return chunk_uuids
```

**File**: `src/fileintel/rag/graph_rag/utils/source_tracer.py` - Already exists

---

### Phase 4: Chunk UUID → Document Metadata (1 hour)

**Goal**: Query chunks via API to get document names and page numbers

**Current approach** (needs optimization):
```python
# Makes N API calls (slow for many chunks)
for chunk_uuid in chunk_uuids:
    chunk_data = api._request("GET", f"chunks/{chunk_uuid}")
```

**Optimized approach**:

**Option A: Batch API Endpoint** (Recommended)
```python
# New API endpoint: POST /api/v2/chunks/batch
# Request: {"chunk_ids": ["uuid1", "uuid2", ...]}
# Response: {"data": [chunk1, chunk2, ...]}

def get_chunks_batch(chunk_uuids, api):
    """Get multiple chunks in one API call."""
    return api._request("POST", "chunks/batch", json={"chunk_ids": list(chunk_uuids)})
```

**File**: `src/fileintel/api/routes/chunks.py` (NEW ENDPOINT)

```python
@router.post("/batch", response_model=ApiResponseV2)
async def get_chunks_batch(
    request: Dict[str, List[str]],
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """Get multiple chunks by IDs in a single request."""
    chunk_ids = request.get("chunk_ids", [])

    chunks = []
    for chunk_id in chunk_ids[:50]:  # Limit to 50 per request
        chunk = storage.get_chunk_by_id(chunk_id)
        if chunk:
            chunks.append({
                "id": str(chunk.id),
                "document": {
                    "original_filename": chunk.document.original_filename,
                },
                "chunk_metadata": chunk.chunk_metadata,
                "chunk_text": chunk.chunk_text[:200],  # Preview only
            })

    return create_success_response({"chunks": chunks, "count": len(chunks)})
```

**Option B: Direct Parquet Reading** (Faster, No API)
```python
# Read documents.parquet directly (already has chunk metadata)
def get_chunk_metadata_from_parquet(chunk_uuids, workspace_path):
    """Get chunk metadata directly from documents.parquet."""
    import pandas as pd
    import os

    docs_df = pd.read_parquet(os.path.join(workspace_path, "documents.parquet"))

    sources = []
    for chunk_uuid in chunk_uuids:
        doc_row = docs_df[docs_df['id'] == str(chunk_uuid)]
        if not doc_row.empty:
            sources.append({
                "document": doc_row.iloc[0].get('title', 'Unknown'),
                "text_preview": doc_row.iloc[0].get('text', '')[:200],
            })

    return sources
```

**Problem with Option B**: documents.parquet doesn't have page numbers

**Solution**: Hybrid approach
```python
def get_source_metadata(chunk_uuids, workspace_path, api):
    """
    Get source metadata from both parquet (fast) and API (complete).

    1. Get document names from documents.parquet (fast, no page numbers)
    2. For unique documents, get one chunk per document via API (slow, has page numbers)
    3. Merge results
    """
    import pandas as pd
    import os

    # Step 1: Get basic info from parquet
    docs_df = pd.read_parquet(os.path.join(workspace_path, "documents.parquet"))

    # Group chunks by document
    doc_chunks = {}
    for chunk_uuid in chunk_uuids:
        doc_row = docs_df[docs_df['id'] == str(chunk_uuid)]
        if not doc_row.empty:
            doc_title = doc_row.iloc[0].get('title', 'Unknown')
            if doc_title not in doc_chunks:
                doc_chunks[doc_title] = []
            doc_chunks[doc_title].append(str(chunk_uuid))

    # Step 2: Get page numbers from API (one chunk per document)
    sources = []
    for doc_title, chunk_list in doc_chunks.items():
        # Get first chunk from this document
        chunk_data = api._request("GET", f"chunks/{chunk_list[0]}")
        chunk_info = chunk_data.get("data", chunk_data)

        page = chunk_info.get("chunk_metadata", {}).get("page_number")

        sources.append({
            "document": doc_title,
            "page_number": page,
            "chunk_count": len(chunk_list),  # How many chunks from this doc
        })

    return sources
```

**File**: `src/fileintel/rag/graph_rag/utils/source_tracer.py`

---

### Phase 5: Display Formatting (30 minutes)

**Goal**: Pretty-print source documents with page numbers

**Implementation**:

```python
def display_sources(sources, cli_handler):
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
```

**File**: `src/fileintel/cli/graphrag.py` - Update `_display_source_documents()`

---

## Complete Implementation Flow

```python
def _display_source_documents(answer_text, collection_identifier, cli_handler):
    """Display source documents traced from GraphRAG answer."""

    # Phase 1: Parse citation IDs
    citation_ids = parse_citation_ids(answer_text)

    if not citation_ids["report_ids"]:
        cli_handler.console.print("\n[dim]No citations found[/dim]")
        return

    # Get workspace path
    index_info = api._request("GET", f"graphrag/{collection_identifier}/status")
    workspace_path = index_info.get("data", {}).get("index_path")

    # Phase 2: Trace reports → text units
    text_unit_ids = trace_reports_to_text_units(
        citation_ids["report_ids"],
        workspace_path
    )

    # Phase 3: Map text units → chunk UUIDs
    chunk_uuids = map_text_units_to_chunks(text_unit_ids, workspace_path)

    # Phase 4: Get chunk metadata
    sources = get_source_metadata(chunk_uuids, workspace_path, api)

    # Phase 5: Display
    display_sources(sources, cli_handler)
```

---

## Implementation Checklist

### Phase 1: Citation Parsing ✅ (Already done in MVP)
- [x] Parse inline citations
- [x] Count report/entity/relationship references
- [ ] Extract specific IDs

### Phase 2: Report Tracing
- [ ] Load community_reports.parquet
- [ ] Load communities.parquet
- [ ] Load entities.parquet
- [ ] Implement report → text_unit traversal

### Phase 3: Text Unit Mapping ✅ (Already done)
- [x] Load text_units.parquet
- [x] Map SHA512 IDs to chunk UUIDs

### Phase 4: Chunk Metadata
- [ ] Option A: Create batch API endpoint
- [ ] Option B: Read documents.parquet directly
- [ ] Implement hybrid approach (recommended)

### Phase 5: Display
- [ ] Format source list
- [ ] Show document names
- [ ] Show page numbers
- [ ] Show chunk counts

---

## Effort Estimates

| Phase | Task | Effort | Priority |
|-------|------|--------|----------|
| 1 | Extract citation IDs | 30 min | High |
| 2 | Report → text unit tracing | 1 hour | High |
| 3 | Text unit → chunk mapping | ✅ Done | - |
| 4a | Batch API endpoint | 1 hour | Medium |
| 4b | Hybrid metadata approach | 1.5 hours | High |
| 5 | Display formatting | 30 min | High |

**Total**: ~4.5 hours for complete implementation

---

## Testing Strategy

### Unit Tests
```python
# Test citation parsing
def test_parse_citation_ids():
    text = "Agile is central [Data: Reports (5, 3, 47)]"
    ids = parse_citation_ids(text)
    assert ids["report_ids"] == {5, 3, 47}

# Test report tracing
def test_trace_reports_to_text_units():
    report_ids = {5}
    text_unit_ids = trace_reports_to_text_units(report_ids, workspace_path)
    assert len(text_unit_ids) > 0
```

### Integration Tests
```bash
# Test end-to-end
fileintel graphrag query test "what is agile" --show-sources

# Expected output:
# Source Documents (15):
#   [1] Cooper (2018), p. 17
#   [2] Cooper (2018), p. 18
#   ...
```

---

## Alternative Approach: Server-Side Tracing

Instead of tracing in CLI, do it in the GraphRAG service:

**Pros**:
- Cleaner separation
- Can be used by API clients
- Reusable

**Cons**:
- Need to pass sources through JSON (no DataFrames)
- More complex service logic

**Implementation**:
```python
# In graphrag_service.py
async def global_query(self, collection_id: str, query: str, include_sources: bool = False):
    raw_response = await self.global_search(query, collection_id)

    result = {
        "answer": raw_response.get("response"),
        "sources": [],
    }

    if include_sources:
        # Extract sources on server side
        sources = self._trace_sources(raw_response.get("context"), collection_id)
        result["sources"] = sources

    return result
```

**Recommendation**: Start with CLI-side tracing (simpler), move to server-side if needed by API users.

---

## Summary

**Current MVP**: Shows citation counts
**Full Implementation**: Shows specific source documents with page numbers
**Effort**: ~4.5 hours total
**Approach**: CLI-side tracing using parquet files + API for metadata

**Next Step**: Implement Phase 1 (extract citation IDs) and Phase 2 (report tracing) first to prove the approach works.
