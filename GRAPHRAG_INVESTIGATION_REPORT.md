# GraphRAG Implementation Investigation Report

## Executive Summary

The FileIntel GraphRAG implementation uses a **file-based caching and checkpoint system** with intermediate result persistence, but **does NOT have automatic resumption on interruption**. Progress tracking shows item-level progress (e.g., "extract graph progress: 15389/42894") but this only reflects the current extraction batch, not resumable state.

---

## 1. How GraphRAG Stores Intermediate Results

### File-Based Cache System
- **Location**: `/home/tuomo/code/fileintel/src/graphrag/cache/` 
- **Type**: JSON Pipeline Cache with File Storage Backend
- **Key Files**:
  - `/home/tuomo/code/fileintel/src/graphrag/cache/json_pipeline_cache.py` (lines 1-66)
  - `/home/tuomo/code/fileintel/src/graphrag/cache/pipeline_cache.py` (lines 1-68)
  - `/home/tuomo/code/fileintel/src/graphrag/cache/factory.py` (lines 80-105)

### Cache Configuration Hierarchy

```python
# From cache_config.py (lines 12-39)
class CacheConfig(BaseModel):
    type: CacheType | str = "file"  # Default cache type
    base_dir: str = "cache"          # Base directory for cache
    connection_string: Optional[str] = None  # Optional cloud storage
    container_name: Optional[str] = None     # For blob storage
    cosmosdb_account_url: Optional[str] = None  # For CosmosDB
```

### Cache Backend Options (Registered Factory)
From `/home/tuomo/code/fileintel/src/graphrag/cache/factory.py` (lines 100-105):
- **File Cache** (Default): `CacheFactory.register(CacheType.file.value, create_file_cache)`
- **Memory Cache**: `CacheFactory.register(CacheType.memory.value, InMemoryCache)`
- **Blob Storage**: `CacheFactory.register(CacheType.blob.value, create_blob_cache)`
- **CosmosDB**: `CacheFactory.register(CacheType.cosmosdb.value, create_cosmosdb_cache)`

### FileIntel's Cache Integration

In GraphRAG service configuration, caching is **NOT explicitly enabled** in the current setup:
- `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/adapters/config_adapter.py` (lines 227-234)
  - Creates GraphRagConfig with explicit `output_path` and `storage` but no explicit cache configuration
  - This means GraphRAG uses **default cache behavior** (file-based in cache/ subdirectory)

---

## 2. Checkpoint and Resume Mechanism

### Current State: **NO AUTOMATIC RESUMPTION**

**Critical Finding**: There is **NO checkpoint/resume logic** in the FileIntel GraphRAG implementation.

#### Evidence:

1. **Task Definition** (`/home/tuomo/code/fileintel/src/fileintel/tasks/graphrag_tasks.py`, lines 625-730):
   - `build_graphrag_index_task()` has soft timeout of 86400s (24 hours) and hard limit of 90000s (25 hours)
   - **NO restart or resume logic**
   - Uses `asyncio.run(graphrag_service.build_index(...))` with no checkpointing
   - Line 704: Simple call with no state recovery

2. **Build Index Service** (`/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/graphrag_service.py`, lines 84-147):
   - Calls `build_index(config, input_documents)` directly
   - **No checkpoint saving or recovery**
   - No intermediate state persistence between phases

3. **GraphRAG API** (`/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/_graphrag_imports.py`, lines 9-10):
   - Imports `from graphrag.api.index import build_index`
   - This is the upstream GraphRAG library function, not a custom resumable wrapper

4. **Task Failure Handling** (`/home/tuomo/code/fileintel/src/fileintel/celery_config.py`, lines 498-518):
   - Task retry handler exists (lines 498-518) but only marks task as 'RETRY'
   - No checkpoint recovery on retry
   - **Retries restart from beginning**, losing all intermediate work

### What DOES Exist: Progress Callbacks

GraphRAG has a **progress callback system** but it's for monitoring only:
- `/home/tuomo/code/fileintel/src/graphrag/callbacks/workflow_callbacks.py` (lines 12-37)
- `/home/tuomo/code/fileintel/src/graphrag/logger/progress.py` (lines 15-98)

```python
class Progress:
    description: str | None = None
    total_items: int | None = None
    completed_items: int | None = None

class ProgressTicker:
    def __call__(self, num_ticks: int = 1) -> None:
        """Emit progress (for monitoring only)"""
        self._num_complete += num_ticks
        if self._callback is not None:
            self._callback(Progress(...))
```

**This is monitoring only**, not checkpointing.

---

## 3. Storage Backends Used

### Primary Storage: File System
- **Type**: File-based JSON cache
- **Location**: `{workspace_path}/cache/` subdirectory
- **Implementation**: `FilePipelineStorage` (lines 27-38 in `/home/tuomo/code/fileintel/src/graphrag/storage/file_pipeline_storage.py`)
- **Operations**:
  - `get(key)` - Read JSON from file
  - `set(key, value)` - Write JSON to file
  - `has(key)` - Check file existence
  - `find(pattern)` - Search files by pattern

### Secondary Storage: PostgreSQL Database
- **Location**: `/home/tuomo/code/fileintel/src/fileintel/celery_config.py` (lines 50-77)
- **Purpose**: Store task metadata and status
- **Connection Pool**: 
  - `pool_size`: configurable from YAML
  - `max_overflow`: configurable from YAML
  - `pool_recycle`: 3600 seconds (line 62)
  - `pool_timeout`: configurable

### Parquet Files for Query Results
- **Location**: `{workspace_path}/output/*.parquet`
- **Files**:
  - `entities.parquet`
  - `relationships.parquet`
  - `communities.parquet`
  - `community_reports.parquet`
  - `documents.parquet`
  - `covariates.parquet` (optional)

From `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/parquet_loader.py` (lines 27-31):
```python
files_to_load = {
    "entities": "entities.parquet",
    "communities": "communities.parquet",
    "community_reports": "community_reports.parquet",
}
```

### In-Memory Caching (Session-level)
- **Location**: `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/dataframe_cache.py` (lines 11-115)
- **Backend Options**:
  - **LRU Cache** (primary): `LRUCache(maxsize=max_lru_size)` (line 24)
  - **Redis** (optional fallback): `redis.from_url(...)` (lines 28-30)
- **Hit Rate Tracking**: Lines 78-88
- **Warmup Capability**: Lines 90-114

```python
# Cache with dual backend (LRU + Redis)
self.lru_cache = LRUCache(maxsize=max_lru_size)
if settings.rag.cache.enabled and settings.rag.cache.redis_host:
    self.redis_client = redis.from_url(f"redis://{...}")
```

---

## 4. Progress Tracking Analysis

### "Extract graph progress: 15389/42894" Source

**Location**: `/home/tuomo/code/fileintel/src/graphrag/index/operations/extract_graph/extract_graph.py` (lines 69-70):

```python
results = await derive_from_rows(
    text_units,
    run_strategy,
    callbacks,
    async_type=async_mode,
    num_threads=num_threads,
    progress_msg="extract graph progress: ",  # <-- Source of message
)
```

### What This Progress Means

From `/home/tuomo/code/fileintel/src/graphrag/logger/progress.py` (lines 33-62):

```python
class ProgressTicker:
    def __init__(self, callback: ProgressHandler | None, num_total: int, 
                 description: str = ""):
        self._num_total = num_total  # Total items to process
        self._num_complete = 0        # Completed items

    def __call__(self, num_ticks: int = 1) -> None:
        """Emit progress."""
        self._num_complete += num_ticks
        if self._callback is not None:
            p = Progress(
                total_items=self._num_total,
                completed_items=self._num_complete,
                description=self._description,  # "extract graph progress: "
            )
            logger.info("%s%s/%s", p.description, str(p.completed_items), 
                       str(p.total_items))
            self._callback(p)
```

### Critical Finding: **NOT Resumable State**

The progress message indicates:
- **15389 items completed** out of **42894 total**
- This is **within a single extraction batch**, not across the whole index build
- **No intermediate checkpointing** - just progress logging
- If the task fails at 15389/42894, restarting will begin from 0

The full GraphRAG indexing pipeline has multiple phases:
1. **Chunk Text** - Split input into chunks
2. **Create Base Text Units** - Initial text unit creation
3. **Extract Graph** - Entity/relationship extraction (THIS is where we see progress)
4. **Resolve Communities** - Community detection
5. **Summarize Descriptions** - Entity description summarization
6. **Summarize Communities** - Community report generation
7. **Create Final Entities/Relationships/Communities** - Output finalization

Only phase 3 shows progress. If any phase fails, entire index build restarts.

---

## 5. Caching and State Management Summary

### What IS Cached (Persistent)

| Component | Type | Location | Resumable? |
|-----------|------|----------|-----------|
| LLM Responses | JSON File | `cache/` | No |
| Embeddings | JSON File | `cache/` | No |
| Extract Operations | JSON File | `cache/` | No |
| Query Results | Parquet | `output/*.parquet` | Yes (for queries only) |
| Task Metadata | PostgreSQL | Database | Yes (status only) |
| DataFrame Cache | LRU + Redis | Memory + Network | No |

### What is NOT Cached/Checkpointed

- **Intermediate entity/relationship extractions** (only final results in parquet)
- **Community detection state**
- **Summarization progress**
- **Individual chunk processing state**

### Failure Scenarios

**Scenario 1: Task Interruption During Extraction**
- Progress: "extract graph progress: 15389/42894"
- On restart: **ALL 42894 items re-processed from scratch**
- Cached LLM calls may avoid recomputation, but no guarantee

**Scenario 2: Worker Crash**
- Task marked as 'REVOKED' in database (lines 619-623 in celery_config.py)
- Manual restart of task: **Starts from beginning**
- No checkpoint recovery

**Scenario 3: Task Timeout (24+ hours)**
- Celery task killed after hard limit (25 hours)
- Task retry triggered if configured
- **Retry starts from scratch**

---

## 6. GraphRAG Configuration

### Current Configuration in FileIntel
From `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/adapters/config_adapter.py` (lines 227-234):

```python
config = GraphRagConfig(
    root_dir=workspace_path,
    models=models,  # Chat and embedding models
    storage=StorageConfig(base_dir=output_path),  # Output storage
    input=InputConfig(base_dir=input_path),        # Input storage
    output=OutputConfig(base_dir=output_path),     # Output storage
    embed_text=embed_text_config,                  # Embedding config
)
```

**Missing**: No explicit `cache` configuration means using GraphRAG defaults (file cache in `cache/` dir).

---

## 7. Key Files and Line References

### Core Implementation Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `graphrag_service.py` | GraphRAG API service | 84-147 (build_index), 149-181 (count results) |
| `graphrag_tasks.py` | Celery task wrappers | 625-730 (build_graphrag_index_task), 136-141 (time limits) |
| `celery_config.py` | Task registry & stale task cleanup | 236-245 (get_task_status), 550-661 (_do_stale_task_cleanup) |
| `dataframe_cache.py` | Session-level caching | 11-115 (GraphRAGDataFrameCache) |
| `parquet_loader.py` | Parquet file loading with cache | 18-53 (load_parquet_files) |
| `config_adapter.py` | GraphRAG config creation | 227-234 (final config creation) |
| `extract_graph.py` | Entity extraction with progress | 27-82 (extract_graph), 69-70 (progress message) |
| `progress.py` | Progress tracking callbacks | 33-62 (ProgressTicker) |
| `pipeline_cache.py` | Cache interface | 12-68 (PipelineCache abstract) |
| `json_pipeline_cache.py` | JSON cache implementation | 24-48 (get/set operations) |
| `factory.py` | Cache factory pattern | 80-105 (create_file_cache registration) |

### API Routes

| Endpoint | File | Purpose |
|----------|------|---------|
| `POST /graphrag/index` | `graphrag_v2.py:35-110` | Start indexing task |
| `GET /graphrag/{id}/status` | `graphrag_v2.py:113-145` | Check index status |
| `GET /graphrag/{id}/entities` | `graphrag_v2.py:148-243` | Load entities from parquet |
| `GET /graphrag/{id}/communities` | `graphrag_v2.py:246-349` | Load communities from parquet |

---

## 8. Recommendations for Improvements

### If Resumable State is Needed:

1. **Implement Phase-level Checkpointing**
   - Save phase completion status to database
   - On restart, check database and skip completed phases
   - Requires modification to GraphRAG wrapper layer

2. **Use GraphRAG's Built-in Caching**
   - Ensure explicit `cache: {type: "file", base_dir: "..."}` config
   - Implement cache cleanup policy (never discard during build)
   - Document cache retention requirements

3. **Add State Recovery Layer**
   - Create checkpoint table in PostgreSQL
   - Track: phase, timestamp, input state, output state
   - Validate consistency before resuming

4. **Monitor Cache Effectiveness**
   - Currently no monitoring of cache hit rates during indexing
   - Add metrics to `celery_config.py` task tracking

### Current Workarounds:

1. **Increase Time Limits** (Already done)
   - Soft limit: 86400s (24 hours)
   - Hard limit: 90000s (25 hours)
   - See `/home/tuomo/code/fileintel/src/fileintel/tasks/graphrag_tasks.py:138-140`

2. **Use Redis Cache** (Optional)
   - Configure Redis in settings
   - Enables cross-worker cache sharing
   - See `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/services/dataframe_cache.py:26-33`

3. **Batch by Collection Size**
   - Smaller collections complete faster
   - Less risk of timeout

---

## Conclusion

FileIntel's GraphRAG implementation stores intermediate results in a **file-based cache** and persists final results as **Parquet files**, but provides **NO automatic resumption on interruption**. Progress tracking (e.g., "extract graph progress: 15389/42894") is for **monitoring only and not resumable**. If the process interrupts, the entire index build restarts from scratch, though cached LLM responses may avoid recomputation.

For production use cases with very large datasets (>100k documents), consider:
1. Implementing explicit phase-level checkpointing
2. Verifying your cache backend (file vs. memory vs. Redis)
3. Adjusting time limits or batch sizes based on data size
4. Monitoring task progress via the Flower UI integration

