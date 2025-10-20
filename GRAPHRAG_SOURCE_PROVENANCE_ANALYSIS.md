# GraphRAG Source Provenance Pipeline Analysis

**Date**: 2025-10-20
**Analysis Type**: End-to-End Pipeline Investigation
**Status**: CRITICAL ISSUES FOUND - Pipeline Non-Functional

---

## Executive Summary

The GraphRAG source provenance pipeline has **3 critical bugs** that prevent it from functioning:

1. **CRITICAL**: Incorrect assumption about text_unit IDs (expects UUIDs, gets SHA256 hashes)
2. **CRITICAL**: Wrong workspace path (expects parquet files at `workspace/`, they're at `workspace/output/`)
3. **CRITICAL**: Incorrect data flow logic (tries to use text_unit IDs as chunk IDs)

**Impact**: The `--show-sources` flag will **always timeout or return empty results** due to these bugs.

**Root Cause**: Fundamental misunderstanding of GraphRAG's internal data model and file structure.

---

## Pipeline Architecture Overview

### Execution Flow

```
CLI: fileintel graphrag query <collection> "question" --show-sources
  |
  v
API: POST /collections/{collection}/query
  |
  v
GraphRAGService.global_query(collection_id, query)
  |
  v
graphrag.api.query.global_search(...)
  |
  v
Returns: {"response": str, "context": {"reports": DataFrame, ...}}
  |
  v
CLI: _display_source_documents(context, collection, cli_handler)
  |
  v
extract_source_chunks(context, workspace_path, storage)
  |
  v
[BUG TRIGGERS HERE - Pipeline fails]
```

### Key Components

1. **CLI Layer**: `/src/fileintel/cli/graphrag.py`
   - `query_with_graphrag()` - Command handler
   - `_display_source_documents()` - Source display logic

2. **API Layer**: `/src/fileintel/api/routes/query.py`
   - `query_collection()` - REST endpoint
   - `_process_graph_query()` - GraphRAG routing

3. **Service Layer**: `/src/fileintel/rag/graph_rag/services/graphrag_service.py`
   - `global_query()` - Wrapper for global search
   - `global_search()` - Executes GraphRAG query

4. **Tracer Utility**: `/src/fileintel/rag/graph_rag/utils/source_tracer.py`
   - `extract_source_chunks()` - Main tracing function
   - `_trace_reports_to_text_units()` - Graph traversal
   - `lookup_chunks()` - Database queries

5. **Storage Layer**: `/src/fileintel/storage/postgresql_storage.py`
   - `get_chunk_by_id()` - Chunk retrieval

---

## Detailed Component Analysis

### 1. GraphRAG Query Execution (WORKING)

**File**: `/src/fileintel/cli/graphrag.py` (lines 52-138)

**Flow**:
```python
# CLI makes API request
payload = {"question": question, "search_type": "graph"}
result = api._request("POST", f"collections/{collection_identifier}/query", json=payload)
response_data = result.get("data", result)
```

**What's Returned**:
```python
response_data = {
    "answer": str,           # The AI-generated response
    "sources": [],          # Empty (not populated by GraphRAG)
    "context": {            # GraphRAG internal context
        "reports": DataFrame,  # Community reports (pandas DataFrame)
        "entities": ...,
        "relationships": ...
    }
}
```

**Status**: ✅ WORKING - Query executes successfully, context is populated

---

### 2. Source Tracer Initialization (BUG #2)

**File**: `/src/fileintel/cli/graphrag.py` (lines 378-402)

**Bug Location**: Line 391
```python
workspace_path = index_data.get("index_path")
# BUG: This returns "/path/to/workspace"
# But parquet files are at "/path/to/workspace/output"
```

**What API Returns**:
```python
index_data = {
    "index_path": "/home/tuomo/code/fileintel/graphrag_indices/graphrag_indices/8bb30b16-817d-4572-9a45-903cbdf43086"
}
```

**Where Parquet Files Actually Are**:
```
/home/tuomo/code/fileintel/graphrag_indices/graphrag_indices/8bb30b16-817d-4572-9a45-903cbdf43086/
├── cache/
├── input/
├── logs/
└── output/  ← PARQUET FILES ARE HERE
    ├── entities.parquet
    ├── communities.parquet
    ├── community_reports.parquet
    ├── text_units.parquet
    ├── documents.parquet
    └── relationships.parquet
```

**Impact**: `_load_parquet_safe()` will fail to find files, return `None`, causing early exit

**Fix Required**: Append `/output` to workspace_path before passing to `extract_source_chunks()`

---

### 3. Source Tracer Data Flow (BUG #1 & #3)

**File**: `/src/fileintel/rag/graph_rag/utils/source_tracer.py`

#### Issue 3a: Incorrect Text Unit ID Assumption (BUG #1)

**Location**: Lines 43-88 (`get_text_unit_ids_from_context`)

**The Bug**:
```python
def get_text_unit_ids_from_context(...) -> Set[str]:
    """
    Extract text unit IDs from GraphRAG context.

    Returns:
        Set of text unit IDs (chunk UUIDs)  # ← WRONG ASSUMPTION!
    """
```

**Reality**:
- GraphRAG creates **SHA256 hashes** as text_unit IDs (128 characters)
- FileIntel chunk IDs are **UUIDs** (36 characters with hyphens)
- These are **completely different** and cannot be used interchangeably

**Actual Data**:
```python
# text_units.parquet
{
    "id": "5f6363b0d320fbabe1c69998ecdfe1b9aede14aedef2a09c0a0f024630aceb339d40a711b383765e8e93fde0ea7830eedbf5788cbe84d133e59f0b46aec25edd",  # SHA256 hash
    "document_ids": ["00a5bd2c-3e82-4ee5-8520-71e5ffcaeda7"]  # ← THIS is the chunk UUID!
}
```

#### Issue 3b: Wrong ID Used for Database Lookup (BUG #3)

**Location**: Lines 168-210 (`lookup_chunks`)

**The Bug**:
```python
def lookup_chunks(text_unit_ids: Set[str], storage) -> List[Dict[str, Any]]:
    for unit_id in text_unit_ids:
        chunk = storage.get_chunk_by_id(unit_id)  # ← WRONG!
        # unit_id is a SHA256 hash, not a chunk UUID
        # This will NEVER find a match in PostgreSQL
```

**What Should Happen**:
1. Get text_unit IDs (SHA256 hashes)
2. Load `text_units.parquet`
3. For each text_unit ID, get `document_ids` (these ARE chunk UUIDs)
4. Use those UUIDs to query PostgreSQL

---

### 4. Parquet File Structure Analysis

**GraphRAG Data Model**:

```
community_reports.parquet
├── id: UUID
├── community: int (links to communities.parquet)
├── title: str
├── summary: str
└── ... other fields

communities.parquet
├── id: UUID
├── community: int
├── entity_ids: ndarray[str]  (links to entities.parquet)
├── text_unit_ids: ndarray[str]  (SHA256 hashes)
└── ... other fields

entities.parquet
├── id: UUID
├── title: str
├── type: str
├── text_unit_ids: ndarray[str]  (SHA256 hashes, links to text_units.parquet)
└── ... other fields

text_units.parquet
├── id: str (SHA256 hash - 128 chars)  ← NOT a UUID!
├── text: str
├── document_ids: ndarray[str]  (chunk UUIDs - 36 chars)  ← THIS is what we need!
├── entity_ids: ndarray[str]
└── ... other fields

documents.parquet
├── id: str (chunk UUID - matches FileIntel chunk.id)  ← Original chunk IDs preserved here
├── title: str (document filename)
├── text: str (chunk text)
└── ... other fields
```

**Key Insight**:
- `documents.parquet` preserves FileIntel chunk UUIDs in the `id` field
- `text_units.parquet` creates new SHA256 hash IDs but stores original chunk UUIDs in `document_ids`
- **Correct mapping**: `text_unit.document_ids` → PostgreSQL `document_chunks.id`

---

### 5. Storage Integration (WORKING)

**File**: `/src/fileintel/storage/postgresql_storage.py` (lines 149-151)

```python
def get_chunk_by_id(self, chunk_id: str):
    """Get a single chunk by its UUID."""
    return self.document_storage.get_chunk_by_id(chunk_id)
```

**Status**: ✅ WORKING - Method exists and functions correctly

**Verified**:
- Method delegates to `DocumentStorage.get_chunk_by_id()`
- Query uses correct UUID format
- Returns `DocumentChunk` object with all metadata
- Includes `document` relationship (for filename) and `chunk_metadata` (for page_number)

---

## Critical Issues Identified

### BUG #1: Incorrect Text Unit ID Assumption ⚠️ CRITICAL

**File**: `/src/fileintel/rag/graph_rag/utils/source_tracer.py`
**Lines**: 43-88, 168-210
**Severity**: CRITICAL - Pipeline Cannot Function

**Problem**:
- Code assumes text_unit IDs are chunk UUIDs
- Reality: text_unit IDs are SHA256 hashes
- Chunk UUIDs are in `text_unit.document_ids`

**Impact**:
- `storage.get_chunk_by_id(unit_id)` will NEVER find chunks
- Pipeline returns empty results or times out trying to find non-existent chunks

**Evidence**:
```python
# What the code expects
text_unit_id = "e196c002-d755-44a1-a336-6bb850b36e57"  # UUID format

# What it actually gets
text_unit_id = "5f6363b0d320fbabe1c69998ecdfe1b9aede14aedef2a09c0a0f024630aceb339d40a711b383765e8e93fde0ea7830eedbf5788cbe84d133e59f0b46aec25edd"  # SHA256 hash
```

---

### BUG #2: Incorrect Workspace Path ⚠️ CRITICAL

**File**: `/src/fileintel/cli/graphrag.py`
**Line**: 391
**Severity**: CRITICAL - Parquet Files Not Found

**Problem**:
```python
workspace_path = index_data.get("index_path")
# Returns: "/path/to/workspace"
# Needs: "/path/to/workspace/output"

sources = extract_source_chunks(context, workspace_path, storage)
# extract_source_chunks expects parquet files at workspace_path/
# But files are at workspace_path/output/
```

**Impact**:
- `_load_parquet_safe()` cannot find parquet files
- Returns `None` for all parquet files
- Early exit from traversal logic
- No text_unit IDs extracted

**Evidence**:
```bash
$ ls /path/to/workspace/
cache/  input/  logs/  output/  # ← Parquet files in output/

$ ls /path/to/workspace/entities.parquet
# File not found

$ ls /path/to/workspace/output/entities.parquet
# File exists
```

---

### BUG #3: Wrong Data Flow Logic ⚠️ CRITICAL

**File**: `/src/fileintel/rag/graph_rag/utils/source_tracer.py`
**Lines**: 91-143, 168-210
**Severity**: CRITICAL - Logic Fundamentally Broken

**Problem**:
The traversal flow is correct (reports → communities → entities → text_units), but the final step is wrong:

```python
# Current flow:
1. Get community IDs from reports ✓
2. Get entity IDs from communities ✓
3. Get text_unit IDs from entities ✓
4. Use text_unit IDs to query PostgreSQL ✗ WRONG!

# Correct flow should be:
1. Get community IDs from reports
2. Get entity IDs from communities
3. Get text_unit IDs from entities (SHA256 hashes)
4. Load text_units.parquet
5. For each text_unit ID, get document_ids (chunk UUIDs)
6. Use chunk UUIDs to query PostgreSQL
```

**Current Code** (lines 183-206):
```python
def lookup_chunks(text_unit_ids: Set[str], storage) -> List[Dict[str, Any]]:
    sources = []
    for unit_id in text_unit_ids:
        chunk = storage.get_chunk_by_id(unit_id)  # ← WRONG: unit_id is SHA256 hash
        if chunk:
            # This will never execute because chunk is always None
            sources.append({...})
    return sources  # Always returns empty list
```

**What It Should Be**:
```python
def lookup_chunks(text_unit_ids: Set[str], workspace_path: str, storage) -> List[Dict[str, Any]]:
    sources = []

    # Load text_units.parquet to map SHA256 IDs to chunk UUIDs
    text_units_df = _load_parquet_safe(workspace_path, "text_units.parquet")
    if text_units_df is None:
        return sources

    # Get chunk UUIDs from text_units
    chunk_uuids = set()
    for unit_id in text_unit_ids:
        tu_row = text_units_df[text_units_df['id'] == unit_id]
        if not tu_row.empty:
            doc_ids = tu_row.iloc[0]['document_ids']
            chunk_uuids.update(doc_ids)

    # Now query PostgreSQL with actual chunk UUIDs
    for chunk_uuid in chunk_uuids:
        chunk = storage.get_chunk_by_id(chunk_uuid)
        if chunk:
            sources.append({...})

    return sources
```

---

## Integration Analysis

### API Endpoint Integration ✅ WORKING

**Endpoint**: `GET /graphrag/{collection_identifier}/status`
**File**: `/src/fileintel/api/routes/graphrag_v2.py` (lines 112-143)

**Returns**:
```python
{
    "success": true,
    "data": {
        "status": "indexed",
        "index_path": "/home/tuomo/code/fileintel/graphrag_indices/graphrag_indices/<collection_id>",
        "documents_count": 530,
        "entities_count": 4310,
        "communities_count": 718,
        "created_at": "...",
        "updated_at": "..."
    }
}
```

**Status**: ✅ Working correctly, returns expected data

---

### Context Format ✅ CORRECT

**Source**: GraphRAG's `global_search()` returns context via `data_adapter.convert_response()`

**Format**:
```python
context = {
    "reports": pandas.DataFrame,  # community_reports.parquet data
    # columns: id, community, title, summary, rank, ...
}
```

**Verification**:
- `context["reports"]` IS a pandas DataFrame ✓
- Has `community` column for linking ✓
- CLI code correctly accesses it with `.iterrows()` ✓

---

### PostgreSQL Storage ✅ WORKING

**Storage Initialization**: Lines 397-399 in `/src/fileintel/cli/graphrag.py`

```python
config = get_config()
storage = PostgreSQLStorage(config)
```

**Status**: ✅ Correct initialization, no circular import issues

**Chunk Retrieval**:
```python
chunk = storage.get_chunk_by_id(chunk_uuid)
# Returns DocumentChunk with:
#   - chunk.id (UUID)
#   - chunk.chunk_text (text content)
#   - chunk.chunk_metadata (dict with page_number, etc.)
#   - chunk.document (relationship to Document)
#   - chunk.document.original_filename (source filename)
```

**Status**: ✅ Working correctly when given valid UUIDs

---

## Failure Mode Analysis

### Scenario 1: User Runs Query with --show-sources

**Command**: `fileintel graphrag query test "what is scrum?" --show-sources`

**Execution Path**:
1. Query executes successfully ✓
2. Answer is displayed ✓
3. Reports info is shown ✓
4. `_display_source_documents()` is called ✓
5. **FAILURE POINT**: `extract_source_chunks()` begins

**What Happens**:
```
Step 1: API call to get workspace_path ✓
Step 2: workspace_path = "/path/to/workspace" (missing /output) ✗
Step 3: extract_source_chunks(context, workspace_path, storage)
Step 4: _load_parquet_safe(workspace_path, "entities.parquet")
        → Tries to open "/path/to/workspace/entities.parquet"
        → File not found
        → Returns None ✗
Step 5: Early return with empty text_unit_ids ✗
Step 6: lookup_chunks([]) returns []
Step 7: Display "No source documents found"
```

**Result**: Empty sources, misleading message

---

### Scenario 2: If Workspace Path Was Fixed

**Assume**: workspace_path correctly points to `/path/to/workspace/output`

**What Would Happen**:
```
Step 1-3: Same as above ✓
Step 4: _load_parquet_safe succeeds, loads entities.parquet ✓
Step 5: Traversal extracts text_unit IDs (SHA256 hashes) ✓
Step 6: lookup_chunks({"5f6363b0...", "1d76c9..."}, storage)
Step 7: storage.get_chunk_by_id("5f6363b0...")
        → Query: SELECT * FROM document_chunks WHERE id = '5f6363b0...'
        → No match (chunk IDs are UUIDs, not SHA256 hashes) ✗
        → Returns None
Step 8: Loop through all text_unit IDs, all return None ✗
Step 9: sources = [] (empty list)
Step 10: Display "No source documents found"
```

**Result**: Still empty sources, but for different reason

---

### Scenario 3: Timeout Possibility

**If**: Database query is slow or has many text_unit IDs

**What Happens**:
```
Step 1-6: Same as Scenario 2
Step 7: Loop through 100+ text_unit IDs
        Each one queries PostgreSQL with invalid SHA256 hash
        Each query returns no results but takes time
Step 8: CLI times out after N seconds
```

**Result**: Timeout error, no sources displayed

---

## Recommended Fixes

### Fix #1: Correct the Data Flow Logic

**File**: `/src/fileintel/rag/graph_rag/utils/source_tracer.py`

**Change 1**: Update function signature (line 168)
```python
# BEFORE
def lookup_chunks(text_unit_ids: Set[str], storage) -> List[Dict[str, Any]]:

# AFTER
def lookup_chunks(text_unit_ids: Set[str], workspace_path: str, storage) -> List[Dict[str, Any]]:
```

**Change 2**: Rewrite lookup_chunks function (lines 168-210)
```python
def lookup_chunks(
    text_unit_ids: Set[str], workspace_path: str, storage
) -> List[Dict[str, Any]]:
    """
    Look up FileIntel chunks by text unit IDs.

    IMPORTANT: text_unit_ids are SHA256 hashes, NOT chunk UUIDs.
    We must load text_units.parquet to get the actual chunk UUIDs.

    Args:
        text_unit_ids: Set of text unit SHA256 hashes from GraphRAG
        workspace_path: Path to GraphRAG parquet files
        storage: PostgreSQLStorage instance

    Returns:
        List of dicts with document, page_number, text_preview
    """
    sources = []

    # Load text_units.parquet to map text_unit IDs to chunk UUIDs
    text_units_df = _load_parquet_safe(workspace_path, "text_units.parquet")
    if text_units_df is None:
        return sources

    # Extract chunk UUIDs from text_units
    chunk_uuids = set()
    for unit_id in text_unit_ids:
        # Find the text_unit row by SHA256 hash ID
        tu_row = text_units_df[text_units_df['id'] == unit_id]
        if not tu_row.empty:
            # Get document_ids (these ARE the FileIntel chunk UUIDs)
            doc_ids = tu_row.iloc[0]['document_ids']
            if isinstance(doc_ids, list):
                chunk_uuids.update(doc_ids)
            elif hasattr(doc_ids, '__iter__'):  # numpy array
                chunk_uuids.update(doc_ids.tolist())

    # Now query PostgreSQL with actual chunk UUIDs
    for chunk_uuid in chunk_uuids:
        try:
            chunk = storage.get_chunk_by_id(str(chunk_uuid))

            if chunk:
                # Extract metadata
                page_number = None
                if chunk.chunk_metadata:
                    page_number = chunk.chunk_metadata.get("page_number")

                document_name = "Unknown"
                if chunk.document:
                    document_name = chunk.document.original_filename

                sources.append(
                    {
                        "chunk_id": str(chunk.id),
                        "document": document_name,
                        "document_id": str(chunk.document_id) if chunk.document_id else None,
                        "page_number": page_number,
                        "text_preview": chunk.chunk_text[:200] if chunk.chunk_text else "",
                    }
                )
        except Exception as e:
            # Skip chunks that can't be loaded
            continue

    return sources
```

**Change 3**: Update function call (line 37)
```python
# BEFORE
sources = lookup_chunks(text_unit_ids, storage)

# AFTER
sources = lookup_chunks(text_unit_ids, workspace_path, storage)
```

---

### Fix #2: Correct the Workspace Path

**File**: `/src/fileintel/cli/graphrag.py`

**Change**: Line 391-402
```python
# BEFORE
workspace_path = index_data.get("index_path")

if not workspace_path:
    cli_handler.console.print("\n[yellow]No GraphRAG index found for source tracing[/yellow]")
    return

# Initialize storage
config = get_config()
storage = PostgreSQLStorage(config)

# Extract sources using tracer utility
sources = extract_source_chunks(context, workspace_path, storage)

# AFTER
workspace_path = index_data.get("index_path")

if not workspace_path:
    cli_handler.console.print("\n[yellow]No GraphRAG index found for source tracing[/yellow]")
    return

# GraphRAG stores parquet files in the 'output' subdirectory
import os
parquet_path = os.path.join(workspace_path, "output")

# Initialize storage
config = get_config()
storage = PostgreSQLStorage(config)

# Extract sources using tracer utility
sources = extract_source_chunks(context, parquet_path, storage)
```

---

### Fix #3: Update Documentation

**File**: `/src/fileintel/rag/graph_rag/utils/source_tracer.py`

**Change**: Update docstrings to reflect correct understanding

```python
# Lines 14-29
def extract_source_chunks(
    graphrag_context: Dict[str, Any],
    workspace_path: str,
    storage,
) -> List[Dict[str, Any]]:
    """
    Trace GraphRAG context to source document chunks.

    GraphRAG stores text content with SHA256 hash IDs (text_units.parquet)
    but preserves original FileIntel chunk UUIDs in the document_ids field.
    This function traverses the knowledge graph and maps back to source chunks.

    Args:
        graphrag_context: The 'context' dict from GraphRAG response
        workspace_path: Path to GraphRAG parquet files directory (contains *.parquet)
        storage: PostgreSQLStorage instance for chunk lookups

    Returns:
        List of source chunks with document, page_number, text_preview
    """
```

---

## Testing Recommendations

### Test Case 1: Verify Parquet File Loading

```python
import os
from fileintel.rag.graph_rag.utils.source_tracer import _load_parquet_safe

workspace = "/path/to/graphrag/workspace/output"

# Should succeed
entities = _load_parquet_safe(workspace, "entities.parquet")
assert entities is not None
assert len(entities) > 0

# Should fail gracefully
missing = _load_parquet_safe(workspace, "nonexistent.parquet")
assert missing is None
```

### Test Case 2: Verify Text Unit to Chunk UUID Mapping

```python
import pandas as pd

workspace = "/path/to/graphrag/workspace/output"
text_units = pd.read_parquet(os.path.join(workspace, "text_units.parquet"))

# Get a text_unit
tu_id = text_units.iloc[0]['id']
tu_doc_ids = text_units.iloc[0]['document_ids']

# Verify doc_ids are UUIDs
import re
uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
for doc_id in tu_doc_ids:
    assert re.match(uuid_pattern, doc_id), f"Invalid UUID: {doc_id}"
```

### Test Case 3: End-to-End Source Tracing

```python
from fileintel.rag.graph_rag.utils.source_tracer import extract_source_chunks
from fileintel.storage.postgresql_storage import PostgreSQLStorage
from fileintel.core.config import get_config
import pandas as pd

# Setup
config = get_config()
storage = PostgreSQLStorage(config)
workspace = "/path/to/graphrag/workspace/output"

# Create mock context
context = {
    "reports": pd.read_parquet(os.path.join(workspace, "community_reports.parquet")).head(5)
}

# Execute
sources = extract_source_chunks(context, workspace, storage)

# Verify
assert len(sources) > 0, "Should find source documents"
for source in sources:
    assert "document" in source
    assert "chunk_id" in source
    assert "text_preview" in source
    # Verify chunk_id is a valid UUID
    assert re.match(uuid_pattern, source["chunk_id"])
```

---

## Performance Considerations

### Current Issues

1. **N+1 Query Problem**: Each text_unit requires a separate PostgreSQL query
   - With 100+ text_units, this becomes 100+ database queries
   - Each query searches for non-existent SHA256 hash (current bug)

2. **DataFrame Filtering**: O(n) filtering for each text_unit ID
   - Could be optimized with proper indexing

### Optimization Recommendations

1. **Batch Database Queries**:
```python
# Instead of:
for chunk_uuid in chunk_uuids:
    chunk = storage.get_chunk_by_id(chunk_uuid)

# Use bulk query:
chunks = storage.db.query(DocumentChunk).filter(
    DocumentChunk.id.in_(list(chunk_uuids))
).all()
```

2. **Use DataFrame Indexing**:
```python
# Set text_unit ID as index for O(1) lookup
text_units_df = text_units_df.set_index('id')

# Then use .loc instead of filtering
for unit_id in text_unit_ids:
    if unit_id in text_units_df.index:
        doc_ids = text_units_df.loc[unit_id, 'document_ids']
```

3. **Cache Parquet Data**: text_units.parquet could be cached in memory if frequently accessed

---

## Security Considerations

### Current Status: ✅ SECURE

1. **No SQL Injection Risk**: Uses ORM with parameterized queries
2. **No Path Traversal Risk**: workspace_path from database, not user input
3. **No Data Exposure**: Only returns data user already has access to (their collection)

### Recommendations

1. **Add Validation**: Verify workspace_path is within expected directory
2. **Add Logging**: Log source tracing attempts for audit
3. **Add Rate Limiting**: Prevent excessive database queries

---

## Conclusion

The GraphRAG source provenance pipeline has **fundamental design flaws** that prevent it from functioning:

1. ❌ **Wrong ID Type**: Uses SHA256 hashes instead of chunk UUIDs
2. ❌ **Wrong Directory**: Missing `/output` subdirectory in path
3. ❌ **Wrong Data Flow**: Doesn't map text_unit IDs to chunk UUIDs

**Current State**: Pipeline is **completely non-functional**

**After Fixes**: Pipeline will work correctly and provide source attribution

**Estimated Fix Time**:
- Code changes: 30 minutes
- Testing: 1 hour
- Total: ~1.5 hours

**Testing Priority**: HIGH - This feature is user-facing and currently fails silently

---

## Appendix A: GraphRAG Data Model Reference

### File: text_units.parquet
```
Columns:
- id: str (SHA256 hash, 128 chars)
- human_readable_id: int
- text: str (chunk text content)
- n_tokens: int
- document_ids: ndarray[str] (FileIntel chunk UUIDs)  ← KEY FIELD
- entity_ids: ndarray[str]
- relationship_ids: ndarray[str]
- covariate_ids: ndarray[str]
```

### File: documents.parquet
```
Columns:
- id: str (FileIntel chunk UUID, 36 chars)  ← Preserves original chunk.id
- human_readable_id: int
- title: str (document filename)
- text: str (chunk text)
- text_unit_ids: ndarray[str] (SHA256 hashes)
- creation_date: str
- metadata: dict
```

### File: entities.parquet
```
Columns:
- id: str (UUID)
- title: str (entity name)
- type: str (entity type)
- description: str
- text_unit_ids: ndarray[str] (SHA256 hashes)  ← Links to text_units
- frequency: int
- degree: int
```

### File: communities.parquet
```
Columns:
- id: str (UUID)
- community: int
- entity_ids: ndarray[str] (UUIDs)  ← Links to entities
- text_unit_ids: ndarray[str] (SHA256 hashes)
- ... other fields
```

### File: community_reports.parquet
```
Columns:
- id: str (UUID)
- community: int  ← Links to communities
- title: str
- summary: str
- rank: float
- ... other fields
```

---

## Appendix B: Complete File Listing

### Files Analyzed

1. `/src/fileintel/cli/graphrag.py` (429 lines)
2. `/src/fileintel/rag/graph_rag/utils/source_tracer.py` (235 lines)
3. `/src/fileintel/storage/document_storage.py` (436 lines)
4. `/src/fileintel/storage/postgresql_storage.py` (351 lines)
5. `/src/fileintel/api/routes/query.py` (453 lines)
6. `/src/fileintel/api/routes/graphrag_v2.py` (435 lines)
7. `/src/fileintel/rag/graph_rag/services/graphrag_service.py` (430 lines)
8. `/src/fileintel/rag/graph_rag/adapters/data_adapter.py` (49 lines)

### Parquet Files Examined

```
/home/tuomo/code/fileintel/graphrag_indices/graphrag_indices/8bb30b16-817d-4572-9a45-903cbdf43086/output/
├── community_reports.parquet (718 rows, 15 columns)
├── communities.parquet (718 rows, 12 columns)
├── documents.parquet (530 rows, 7 columns)
├── entities.parquet (4310 rows, 10 columns)
├── relationships.parquet (analyzed structure)
└── text_units.parquet (530 rows, 8 columns)
```

---

**End of Analysis**
