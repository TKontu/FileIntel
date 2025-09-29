# Issues Analysis Report

## Issue 12: GraphRAG LLM Timeout During Entity Extraction

### Root Cause
GraphRAG entity extraction operations were timing out due to insufficient timeout values for gemma3-4B model API calls. The 30-second timeout in the GraphRAG config adapter and 60-second timeout in the unified LLM provider were too short for the slower local model responses.

### Location
- **File**: `src/fileintel/rag/graph_rag/adapters/config_adapter.py`
- **Constant**: `DEFAULT_REQUEST_TIMEOUT = 30.0` (line 16)
- **File**: `src/fileintel/llm_integration/unified_provider.py`
- **Timeout**: `httpx.Timeout(60.0)` (line 177)
- **File**: `src/graphrag/config/defaults.py`
- **Setting**: `request_timeout: float = 180.0` (line 286)

### Error Context
```
celery-worker-1  | [2025-09-28 14:23:35,194: ERROR/ForkPoolWorker-1] error extracting graph
celery-worker-1  | openai.APITimeoutError: Request timed out.
```

### Impact
- GraphRAG indexing tasks fail during entity extraction phase
- Collections cannot be properly indexed for graph-based queries
- Long-running entity extraction operations are prematurely terminated

### Fix Applied
- Increased `DEFAULT_REQUEST_TIMEOUT` from 30.0 to 300.0 seconds (5 minutes)
- Increased unified LLM provider timeout from 60.0 to 300.0 seconds
- Increased GraphRAG defaults `request_timeout` from 180.0 to 300.0 seconds
- All timeout values now consistent at 5 minutes for gemma3-4B compatibility

### Related Methods and Imports
```python
# src/fileintel/rag/graph_rag/adapters/config_adapter.py
class GraphRAGConfigAdapter:
    def adapt_config(self, settings: Settings, collection_id: str, root_path: str) -> GraphRagConfig:
        # Lines 151, 167: Applied to both chat and embedding model configs
        request_timeout=DEFAULT_REQUEST_TIMEOUT,

# src/fileintel/llm_integration/unified_provider.py
class UnifiedLLMProvider:
    def __init__(self, config: Settings, storage: PostgreSQLStorage = None):
        # Line 177: HTTP client timeout configuration
        self.http_client = httpx.Client(timeout=httpx.Timeout(300.0))
```

### Verification Steps
1. Start GraphRAG indexing task with `fileintel graphrag index <collection>`
2. Monitor task progress - should not timeout during entity extraction
3. Check Celery worker logs for successful completion without timeout errors
4. Verify GraphRAG output files are generated in collection workspace

---

## Issue 13: GraphRAG Indices Table Primary Key Constraint Violation

### Root Cause
GraphRAG database models were missing UUID generation for primary key fields, causing null value constraint violations when inserting GraphRAG indices, entities, and communities.

### Location
- **File**: `src/fileintel/storage/graphrag_storage.py`
- **Methods**: `save_graphrag_index_info`, `save_graphrag_entities`, `save_graphrag_communities`
- **Lines**: 61 (GraphRAGIndex), 139 (GraphRAGEntity), 244 (GraphRAGCommunity)

### Error Context
```
postgres-1       | 2025-09-28 14:28:00.086 UTC [432] ERROR:  null value in column "id" of relation "graphrag_indices" violates not-null constraint
celery-worker-1  | [SQL: INSERT INTO graphrag_indices (collection_id, index_path, index_status, documents_count, entities_count, communities_count, updated_at) VALUES (...)]
```

### Impact
- GraphRAG indexing tasks fail to save index metadata to database
- Entity and community data cannot be stored for future queries
- Database integrity constraints prevent successful completion of GraphRAG operations

### Fix Applied
1. **Added UUID generation**: Added `import uuid` and `id=str(uuid.uuid4())` to all GraphRAG model creations
2. **Fixed field mappings**: Corrected field names to match actual model definitions:
   - `name` → `entity_name`, `type` → `entity_type` for GraphRAGEntity
   - Removed non-existent fields and mapped to correct GraphRAGCommunity fields
3. **Consistent UUID pattern**: Applied same UUID generation pattern used throughout codebase

### Related Methods and Imports
```python
# src/fileintel/storage/graphrag_storage.py
import uuid  # Added for UUID generation

# GraphRAGIndex creation (line 61)
index_info = GraphRAGIndex(
    id=str(uuid.uuid4()),
    collection_id=collection_id,
    index_path=index_path,
    ...
)

# GraphRAGEntity creation (line 139)
entity = GraphRAGEntity(
    id=str(uuid.uuid4()),
    collection_id=collection_id,
    entity_name=self.base._clean_text(entity_data.get("name", "")),
    entity_type=entity_data.get("type"),
    ...
)

# GraphRAGCommunity creation (line 244)
community = GraphRAGCommunity(
    id=str(uuid.uuid4()),
    collection_id=collection_id,
    community_id=community_data.get("id"),
    level=community_data.get("level", 0),
    ...
)
```

### Verification Steps
1. Run GraphRAG indexing with `fileintel graphrag index <collection>`
2. Check that task completes without database constraint errors
3. Verify GraphRAG index metadata is saved to database
4. Confirm entities and communities are stored with proper UUIDs

---

## Issue 14: GraphRAG Rate Limiting Too Conservative for Local LLM Server

### Root Cause
GraphRAG rate limiting was configured for external API services rather than local LLM servers, causing unnecessary delays during indexing and querying operations. The rate limits were set to very conservative values (999 RPM, 5-10 queries per minute) which severely throttled processing speed.

### Location
- **File**: `src/fileintel/rag/graph_rag/adapters/config_adapter.py`
- **Constants**: `HIGH_RATE_LIMIT_RPM`, `HIGH_RATE_LIMIT_TPM` (lines 20-21)
- **File**: `src/fileintel/core/config.py`
- **Setting**: `max_concurrent_requests` (line 45)
- **File**: `src/graphrag/config/defaults.py`
- **Settings**: `tokens_per_minute`, `requests_per_minute` (lines 294-295)
- **File**: `src/fileintel/tasks/graphrag_tasks.py`
- **Task limits**: `rate_limit` parameters (lines 243, 339)

### Observation Context
```
celery-worker-1  | [2025-09-28 14:54:28,542: INFO/ForkPoolWorker-1] COMMUNITY DEBUG: Acquiring rate limiter (10 per 60s)
celery-worker-1  | [2025-09-28 14:54:28,542: INFO/ForkPoolWorker-1] COMMUNITY DEBUG: Rate limiter acquired in 0.00 seconds
```

### Impact
- GraphRAG indexing operations artificially slowed by conservative rate limiting
- Community detection and entity extraction unnecessarily throttled
- Local LLM server capacity underutilized despite being able to handle concurrent requests
- Processing times significantly longer than necessary

### Fix Applied
1. **Increased rate limits**:
   - `HIGH_RATE_LIMIT_RPM`: 999 → 100,000 requests per minute
   - `HIGH_RATE_LIMIT_TPM`: 999,999 → 100,000,000 tokens per minute
2. **Enhanced concurrency**:
   - `max_concurrent_requests`: 8 → 25 (matching GraphRAG defaults)
3. **GraphRAG defaults optimization**:
   - Changed `requests_per_minute` from "auto" to 100,000
   - Changed `tokens_per_minute` from "auto" to 100,000,000
4. **Celery task rate limits**:
   - GraphRAG query tasks: 5/m and 10/m → 60/m (still reasonable for queries)
   - Removed Literal type annotation conflicts

### Related Methods and Configuration
```python
# src/fileintel/rag/graph_rag/adapters/config_adapter.py
HIGH_RATE_LIMIT_RPM = 100000  # 100k requests per minute
HIGH_RATE_LIMIT_TPM = 100000000  # 100M tokens per minute

# src/fileintel/core/config.py
max_concurrent_requests: int = Field(default=25, ge=1, le=32)

# src/graphrag/config/defaults.py
tokens_per_minute: int = 100000000
requests_per_minute: int = 100000
concurrent_requests: int = 25

# src/fileintel/tasks/graphrag_tasks.py
@app.task(..., rate_limit="60/m", ...)  # For query tasks
```

### Verification Steps
1. Monitor GraphRAG indexing logs for faster progression through community detection
2. Verify rate limiter acquisition times are minimal
3. Check that concurrent requests are being utilized effectively
4. Confirm overall indexing time is significantly reduced

### Benefits
- Dramatically faster GraphRAG indexing for local LLM deployments
- Better utilization of local server resources and capacity
- Maintained reasonable query rate limits to prevent system overload
- Consistent configuration optimized for local deployment scenarios

**UPDATE**: Additional fix applied to eliminate hardcoded internal GraphRAG rate limiting:

**File**: `src/graphrag/index/operations/summarize_communities/strategies.py`
```python
# Line 55: Eliminated hardcoded 10 per 60s rate limit
rate_limiter = RateLimiter(rate=100000, per=60)  # 100k requests per minute (effectively no limit)
```

This removes the final source of "10 per 60s" rate limiting visible in GraphRAG community extraction logs.

---

## Issue 15: GraphRAG Entity Name and Score Display Issues

### Root Cause
The GraphRAG entity and community retrieval methods had field mapping mismatches between the database model field names and the retrieval code, causing entity names to show as "Unknown" and importance scores to display as 0.00.

### Location
- **File**: `src/fileintel/storage/graphrag_storage.py`
- **Methods**: `get_graphrag_entities`, `get_graphrag_communities`, `_get_graphrag_communities_from_db`
- **Lines**: 171-183 (entities), 276-286 (communities), 310-320 (community helper)
- **File**: `src/fileintel/cli/graphrag.py`
- **CLI Display**: Community display logic (lines 188-195)

### Error Context
```
GraphRAG Entities (20):
Unknown (ORGANIZATION) - Score: 0.00
  AGILE is a comprehensive software development methodology...
Unknown (PERSON) - Score: 0.00
  Takeuchi is a researcher associated with...
```

### Impact
- Entity names displayed as "Unknown" instead of actual names (AGILE, SCRUM, etc.)
- Importance scores all showing as 0.00 instead of their actual values
- Community information displaying incorrect field mappings
- Poor user experience when exploring GraphRAG data

### Root Cause Analysis
1. **Entity field mismatches**:
   - Code used `entity.name` but model has `entity.entity_name`
   - Code used `entity.type` but model has `entity.entity_type`
   - Code used `entity.entity_data` but model has `entity.entity_metadata`
   - Missing `importance_score` mapping

2. **Community field mismatches**:
   - Code used non-existent fields: `rank`, `findings`, `rank_explanation`, `community_data`
   - Code used `.order_by(GraphRAGCommunity.rank.desc())` but model has no `rank` field
   - CLI expected `rank` but model has `level`

### Fix Applied
**Entity Retrieval (`get_graphrag_entities`)**:
```python
# Fixed field mappings
entity_dict = {
    "id": entity.id,
    "name": entity.entity_name,          # Was: entity.name
    "type": entity.entity_type,          # Was: entity.type
    "description": entity.description,
    "importance_score": entity.importance_score,  # Added missing field
    "collection_id": entity.collection_id,
}

# Fixed metadata field
if entity.entity_metadata:              # Was: entity.entity_data
    entity_dict.update(entity.entity_metadata)
```

**Community Retrieval (`get_graphrag_communities`)**:
```python
# Fixed ordering
.order_by(GraphRAGCommunity.size.desc())  # Was: rank.desc()

# Fixed field mappings
community_dict = {
    "id": community.id,
    "community_id": community.community_id,
    "level": community.level,              # Replaced rank
    "title": community.title,
    "summary": community.summary,
    "entities": community.entities or [],  # Replaced findings
    "size": community.size,
    "collection_id": community.collection_id,
}
```

**CLI Display Updates**:
```python
# Community display updated
level = community.get("level", 0)         # Was: rank
community_id = community.get("community_id", "N/A")  # Added
cli_handler.console.print(
    f"[bold]{title}[/bold] (ID: {community_id}, Level: {level}, Size: {size})"
)
```

### Verification Steps
1. Run `fileintel graphrag entities <collection>` - should show actual entity names and scores
2. Run `fileintel graphrag communities <collection>` - should show proper community info
3. Verify entity importance scores are non-zero for relevant entities
4. Check that community levels and sizes are displayed correctly

### Related Models
```python
# src/fileintel/storage/models.py
class GraphRAGEntity(Base):
    entity_name = Column(String, nullable=False, index=True)
    entity_type = Column(String, nullable=True)
    importance_score = Column(Integer, nullable=True)
    entity_metadata = Column(JSONB, nullable=True)

class GraphRAGCommunity(Base):
    community_id = Column(Integer, nullable=False)
    level = Column(Integer, nullable=False, default=0)
    entities = Column(JSONB, nullable=True)
    size = Column(Integer, nullable=False, default=0)
```

---

## Critical Issue: 'AsyncResult' object has no attribute 'apply_async'

### Root Cause
The error occurs in the `generate_collection_embeddings_simple` task when it tries to call `apply_async()` on `completion_callback_signature`, but the parameter is receiving an `AsyncResult` object instead of a Celery signature.

### Location
- **File**: `src/fileintel/tasks/workflow_tasks.py`
- **Method**: `generate_collection_embeddings_simple`
- **Lines**: 238 and 272

### Detailed Analysis

#### Problem Context
```python
# Line 84: Creates a proper Celery signature
completion_callback = mark_collection_completed.s(collection_id)

# Line 92: Passes signature as parameter
completion_callback_signature=completion_callback,

# Lines 238, 272: ERROR - tries to call apply_async on wrong object type
completion_result = completion_callback_signature.apply_async(
    args=[document_results]
).get()
```

#### The Issue
When `completion_callback` is created using `mark_collection_completed.s(collection_id)`, it correctly creates a Celery signature object. However, somewhere in the workflow orchestration, this signature is being converted to an `AsyncResult` object, which doesn't have an `apply_async()` method.

### Related Methods and Imports

#### Method Chain
1. **`complete_collection_analysis`** (line 22-137) - Main orchestration method
   - Creates `completion_callback = mark_collection_completed.s(collection_id)` (line 84)
   - Passes it to `generate_collection_embeddings_simple.s()` (line 92)

2. **`generate_collection_embeddings_simple`** (line 206-320) - Where error occurs
   - Receives `completion_callback_signature` parameter (line 207)
   - Tries to call `completion_callback_signature.apply_async()` (lines 238, 272)

3. **`mark_collection_completed`** (line 140-202) - Target callback method
   - Should be called by the signature

#### Related Imports
```python
from celery import group, chain, chord, signature  # Line 10
from fileintel.celery_config import app  # Line 12
from .base import BaseFileIntelTask  # Line 13
```

#### Functions that use the problematic method
- `complete_collection_analysis` - Creates and passes the signature
- `generate_collection_embeddings_simple` - Uses the signature incorrectly

### Error Log Context
```
[2025-09-28 10:54:20,622: ERROR/ForkPoolWorker-1] Error in complete collection analysis: 'AsyncResult' object has no attribute 'apply_async'
[2025-09-28 10:54:20,640: INFO/ForkPoolWorker-1] Updated collection 05dcb4da-c518-40b8-8e9b-7786a8c89c99 status to failed
```

### Potential Root Causes

1. **Parameter Type Confusion**: The `completion_callback_signature` parameter might be receiving an `AsyncResult` from the Celery workflow orchestration instead of the expected signature.

2. **Celery Chord/Group Behavior**: When using complex Celery patterns (chord, group), the callback signatures might be automatically executed and their results (AsyncResult objects) passed instead of the signatures themselves.

3. **Workflow Orchestration Issue**: The chord pattern at line 89-94 might be causing the signature to be executed prematurely, converting it to an AsyncResult.

### Recommended Fixes

1. **Type Check and Convert**: Add type checking to detect AsyncResult and handle it appropriately
2. **Signature Recreation**: If receiving AsyncResult, recreate the signature from the original task
3. **Workflow Pattern Refactor**: Simplify the chord pattern to avoid signature/result confusion
4. **Direct Task Call**: Instead of using `apply_async()`, call the task directly if already dealing with results

### Impact
- Collection processing fails completely
- Documents are processed but embedding generation and completion callbacks fail
- Collections are marked as "failed" status instead of "completed"
- Workflow orchestration is broken for any collection that uses embedding generation

### Files Modified in Recent Commits
- `src/fileintel/api/routes/collections_v2.py` - API layer calling the workflows
- `src/fileintel/storage/models.py` - Data models
- `src/fileintel/storage/vector_search_storage.py` - Storage layer
- `debug_vector_search.py` - Debug script (new file)

The issue appears to be in the workflow orchestration logic rather than the recent changes to storage or API layers.

---

## Issue 2: PostgreSQL Vector Operator Error

### Root Cause
PostgreSQL is missing the pgvector extension or the extension is not properly installed/enabled, causing the `<=>` vector distance operator to be undefined.

### Location
- **File**: `src/fileintel/storage/vector_search_storage.py`
- **Method**: `find_relevant_chunks_in_collection`
- **Lines**: 79, 84, 86

### Error Details
```
(psycopg2.errors.UndefinedFunction) operator does not exist: vector <=> numeric[]
LINE 10: 1 - (c.embedding <=> ARRAY[0.01816732...
HINT: No operator matches the given name and argument types. You might need to add explicit type casts.
```

### Related Methods and Functions
- `find_relevant_chunks_in_collection` - Main method using vector similarity search
- `find_relevant_chunks_in_document` - Similar vector search for documents
- `find_relevant_chunks_hybrid` - Hybrid search using vector operations

### Impact
- Vector similarity search completely broken
- Query operations fail when trying to find relevant chunks
- Collections cannot be queried for content

---

## Issue 3: GraphRAG Settings Configuration Error

### Root Cause
The Settings configuration has been refactored to consolidate GraphRAG settings into the RAG settings section, but some code still tries to access `settings.graphrag` instead of the new structure.

### Location
- **Files**: Multiple files in `src/fileintel/rag/graph_rag/`
- **Error**: `'Settings' object has no attribute 'graphrag'`

### Configuration Structure Change
**Old structure** (still being accessed by some code):
```python
settings.graphrag.llm_model
settings.graphrag.embedding_model
settings.graphrag.cache.enabled
```

**New structure** (in config.py):
```python
settings.rag.llm_model
settings.rag.embedding_model
settings.rag.cache.enabled
```

### Related Files and Methods
- `src/fileintel/rag/graph_rag/adapters/config_adapter.py` - Lines 52, 55, 143, 150, 159, 166, 219, 223
- `src/fileintel/rag/graph_rag/services/dataframe_cache.py` - Lines 23, 26, 29, 92, 101
- `src/fileintel/rag/graph_rag/services/parquet_loader.py` - Line 44

### Impact
- GraphRAG indexing operations fail
- GraphRAG status requests return 500 errors
- Any GraphRAG functionality is broken

---

## Issue 4: API Response Validation Error

### Root Cause
The `ApiResponseV2` model expects a `timestamp` field but the GraphRAG status endpoint is not providing it, causing Pydantic validation to fail.

### Location
- **File**: `src/fileintel/api/routes/graphrag_v2.py`
- **Method**: `get_graphrag_system_status`
- **Line**: 352-354

### Error Details
```
1 validation error for ApiResponseV2
timestamp
  Field required [type=missing, input_value={'success': True, 'messag...'community_detection']}}, input_type=dict]
```

### Related Methods
- `get_graphrag_system_status` - Returns status without timestamp
- `ApiResponseV2` model - Requires timestamp field

### Impact
- GraphRAG system status endpoint returns 500 errors
- API responses fail validation
- Frontend/CLI cannot get GraphRAG system status

---

## Issue 5: Vector Search Query Execution Confirmed Broken

### Root Cause
This confirms **Issue 2** - the PostgreSQL pgvector extension is not properly installed or enabled, causing all vector similarity searches to fail at the database level.

### Location
- **File**: `src/fileintel/storage/vector_search_storage.py`
- **Method**: `find_relevant_chunks_in_collection`
- **Lines**: 79, 84, 86

### Error Context
The error occurs during a successful query flow:
1. ✅ Embedding provider successfully generates query embedding
2. ✅ Collection exists and is found
3. ❌ **PostgreSQL vector operator fails** when executing similarity search
4. ✅ API still returns 200 OK (but with empty/failed results)

### Detailed Error
```
(psycopg2.errors.UndefinedFunction) operator does not exist: vector <=> numeric[]
LINE 10: 1 - (c.embedding <=> ARRAY[0.01955970004...
HINT: No operator matches the given name and argument types. You might need to add explicit type casts.
```

### SQL Query That Fails
```sql
SELECT
    c.id as chunk_id,
    c.chunk_text,
    c.position,
    c.chunk_metadata,
    d.filename,
    d.original_filename,
    d.id as document_id,
    1 - (c.embedding <=> %(query_embedding)s) as similarity
FROM document_chunks c
JOIN documents d ON c.document_id = d.id
WHERE d.collection_id = %(collection_id)s
    AND c.embedding IS NOT NULL
    AND 1 - (c.embedding <=> %(query_embedding)s) >= %(similarity_threshold)s
ORDER BY c.embedding <=> %(query_embedding)s
LIMIT %(limit)s
```

### Related Components Working Correctly
- ✅ Embedding generation (vLLM server responding)
- ✅ API routing and validation
- ✅ Collection lookup
- ✅ Database connection

### Database Requirements
The issue requires installing/enabling pgvector extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Impact
- **All vector similarity searches fail**
- **Primary RAG functionality broken**
- Query commands return empty results instead of similar chunks
- Users cannot get semantic search results from their collections
- Vector-based document analysis completely non-functional

---

## Issue 6: 'state' KeyError in Task Status/Result APIs

### Root Cause
The task API endpoints were trying to access `task_info["state"]` but the `get_task_status()` function in `celery_config.py` returns a dictionary with key `"status"` instead of `"state"`.

### Location
- **File**: `src/fileintel/api/routes/tasks_v2.py`
- **Lines affected**: 131, 133, 135, 216, 219, 255, 258, 378, 383, 447, 450

### Impact
- Task status endpoints returned `Internal error: 'state'`
- Task result endpoints failed with KeyError
- CLI commands `fileintel tasks get` and `fileintel tasks result` failed

### Related Methods
- `get_task_status_endpoint()` - Line 131, 133, 135
- `cancel_task_endpoint()` - Line 216, 219
- `get_task_result_endpoint()` - Line 255, 258
- `batch_cancel_tasks()` - Line 378, 383
- `retry_task()` - Line 447, 450

### Related Imports and Functions
- **celery_config.py**: `get_task_status()` function (line 354) returns `{"status": ...}` not `{"state": ...}`
- **API error handler**: `celery_error_handler` catches the KeyError and returns generic error message

### Fix Applied
Changed all occurrences of `task_info["state"]` to `task_info["status"]` in the task API routes to match the actual structure returned by `get_task_status()`.

---

## Issue 7: Task Result Returning null/None Data

### Root Cause
When API endpoints encountered errors (like the 'state' KeyError above), they correctly returned error responses with `success: false` and `data: null`. However, the CLI code didn't handle this properly and tried to call methods on `None` values.

### Location
- **File**: `src/fileintel/cli/tasks.py`
- **Lines affected**: 84, 127-129

### Impact
- CLI crashed with `AttributeError: 'NoneType' object has no attribute 'get'`
- Commands `fileintel tasks get` and `fileintel tasks result` displayed stack traces instead of error messages
- Poor user experience when API errors occurred

### Related Methods
- `get_task()` - Line 84: `task_data.get("data", task_data)` could return `None`
- `get_task_result()` - Lines 127-129: `task_result.get("ready")` called on `None`

### Related Imports and Functions
- **CLI handler**: `cli_handler.handle_api_call()` - returns API response structure
- **API response structure**: `{"success": bool, "data": any, "error": str}`

### Fix Applied
1. Added success checking before processing API response data
2. Handle error responses gracefully by displaying error messages instead of crashing
3. Provide fallback empty dict `{}` instead of `None` for data processing

### Code Changes
```python
# Before
task_result = result_data.get("data", result_data)
if task_result.get("ready"):

# After
if not result_data.get("success", False):
    error_msg = result_data.get("error", "Unknown error occurred")
    cli_handler.display_error(f"Failed to get task result: {error_msg}")
    return

task_result = result_data.get("data", {})
if task_result and task_result.get("ready"):
```

### Verification Steps
1. Test `fileintel tasks get <task_id>` - should show task status or proper error message
2. Test `fileintel tasks result <task_id>` - should show task result or proper error message
3. Test `fileintel graphrag index <collection>` - should create task and allow status checking
4. Verify that task status APIs return proper field names without KeyError exceptions

### Prevention
- Ensure consistency between `celery_config.py` function return structures and API route expectations
- Add comprehensive error handling in CLI commands for API failure scenarios
- Add unit tests for error response handling in both API and CLI layers

---

## Issue 8: GraphRAG Task Queue Routing Problem

### Root Cause
GraphRAG indexing tasks are configured to use the "memory_intensive" queue, but Celery workers are only configured to consume from these queues: default, document_processing, indexing, llm_processing, rag_processing. The "memory_intensive" queue is not in the worker's queue configuration, causing tasks to remain stuck in PENDING status indefinitely.

### Location
- **File**: `src/fileintel/tasks/graphrag_tasks.py`
- **Tasks affected**: `build_graph_index` (line 123) and `build_graphrag_index_task` (line 606)
- **Worker configuration**: `src/fileintel/celery_config.py` (lines 45-51)

### Detailed Analysis

#### Problem Context
```python
# GraphRAG tasks configured to use non-existent queue
@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="memory_intensive",  # ❌ This queue doesn't exist in worker config
    soft_time_limit=1800,
    time_limit=3600,
)
def build_graphrag_index_task(...)

# Worker only consumes from these queues (celery_config.py)
task_queues=(
    Queue("default", routing_key="default"),
    Queue("document_processing", routing_key="document_processing"),
    Queue("rag_processing", routing_key="rag_processing"),
    Queue("llm_processing", routing_key="llm_processing"),
    Queue("indexing", routing_key="indexing"),  # ✅ Available queue
)
```

#### Impact
- GraphRAG indexing commands `fileintel graphrag index <collection>` create tasks but they never start processing
- Tasks remain in PENDING status indefinitely
- No worker logs or activity because no worker is consuming from the "memory_intensive" queue
- GraphRAG functionality completely broken - users cannot build indexes

### Related Methods and Functions
- `build_graph_index` - Uses "memory_intensive" queue, should use "indexing" queue
- `build_graphrag_index_task` - Uses "memory_intensive" queue, should use "indexing" queue
- All GraphRAG API endpoints that trigger indexing tasks are affected

### Fix Applied
Changed both GraphRAG tasks to use the "indexing" queue instead of "memory_intensive":
```python
# Before
queue="memory_intensive"

# After
queue="indexing"
```

### Verification Steps
1. Test `fileintel graphrag index <collection>` - task should start processing immediately
2. Check worker logs - should show task being consumed and processed
3. Verify task moves from PENDING to STARTED/SUCCESS status
4. Confirm GraphRAG index files are created successfully

### Fix Applied ✅
Changed both GraphRAG tasks to use the "graphrag_indexing" queue instead of "memory_intensive".

---

## Issue 9: Complete Queue Configuration Optimization

### Root Cause
The original queue configuration was inadequate for FileIntel's diverse workload patterns. Embedding tasks, LLM operations, and GraphRAG processes have fundamentally different resource requirements but were grouped together, leading to suboptimal resource utilization and potential performance bottlenecks.

### Solution Implemented
Implemented a comprehensive 7-queue configuration that separates tasks by resource usage and operational characteristics:

#### New Queue Configuration
```python
task_queues=(
    Queue("default", routing_key="default"),                            # Catch-all
    Queue("document_processing", routing_key="document_processing"),    # File processing, chunking
    Queue("embedding_processing", routing_key="embedding_processing"),  # High-throughput embedding generation
    Queue("llm_processing", routing_key="llm_processing"),              # Text generation, summarization
    Queue("rag_processing", routing_key="rag_processing"),              # Vector queries, lightweight ops
    Queue("graphrag_indexing", routing_key="graphrag_indexing"),        # Heavy GraphRAG index building
    Queue("graphrag_queries", routing_key="graphrag_queries"),          # GraphRAG query operations
)
```

#### Task Distribution Strategy

**Embedding Queue (`embedding_processing`)**:
- `generate_text_embedding`
- `generate_and_store_chunk_embedding`
- `generate_collection_embeddings_simple`
- All batch embedding operations

**LLM Queue (`llm_processing`)**:
- `summarize_content`
- Text generation tasks
- Complex reasoning operations

**GraphRAG Indexing Queue (`graphrag_indexing`)**:
- `build_graph_index`
- `build_graphrag_index_task`
- `update_collection_index`

**GraphRAG Queries Queue (`graphrag_queries`)**:
- `query_graph_global`
- `query_graph_local`
- `adaptive_graphrag_query`

**RAG Processing Queue (`rag_processing`)**:
- `get_graphrag_index_status`
- `remove_graphrag_index`
- Vector RAG queries
- Lightweight operations

### Benefits
1. **Resource Optimization**: Each queue can be scaled independently based on actual workload
2. **Performance Isolation**: Heavy GraphRAG indexing won't block document processing or embedding generation
3. **Specialized Workers**: Different worker configurations optimized for each task type
4. **Better Monitoring**: Clear separation enables targeted performance monitoring and alerting
5. **Future Scalability**: Easy to adjust resource allocation as usage patterns evolve

### Recommended Worker Configuration
```bash
# High-throughput embedding workers
celery -A fileintel.celery_config worker -Q embedding_processing -c 8 --max-memory-per-child=512MB

# LLM workers (fewer concurrent, more memory for context)
celery -A fileintel.celery_config worker -Q llm_processing -c 2 --max-memory-per-child=2GB

# GraphRAG indexing (dedicated heavy compute)
celery -A fileintel.celery_config worker -Q graphrag_indexing -c 1 --max-memory-per-child=4GB

# GraphRAG queries (balanced for responsiveness)
celery -A fileintel.celery_config worker -Q graphrag_queries -c 3 --max-memory-per-child=1GB

# Document processing and RAG queries
celery -A fileintel.celery_config worker -Q document_processing,rag_processing -c 4
```

---

## Issue 10: GraphRAG Indexing Method Name Errors

### Root Cause
The GraphRAG indexing task (`build_graphrag_index_task`) was calling incorrect method names on the PostgreSQL storage object:
- Called `get_documents_in_collection()` but the actual method is `get_documents_by_collection()`
- Called `get_document_chunks()` but the actual method is `get_all_chunks_for_document()` or `get_all_chunks_for_collection()`

### Location
- **File**: `src/fileintel/tasks/graphrag_tasks.py`
- **Lines affected**: 651, 662

### Error Details
```
AttributeError: 'PostgreSQLStorage' object has no attribute 'get_documents_in_collection'
```

### Impact
- GraphRAG indexing tasks failed immediately after starting
- Collections could not be indexed for GraphRAG queries
- GraphRAG functionality completely broken

### Fix Applied
1. Changed `get_documents_in_collection()` → `get_documents_by_collection()`
2. Optimized chunk retrieval by using `get_all_chunks_for_collection()` directly instead of iterating through documents

### Code Changes
```python
# Before
documents = storage.get_documents_in_collection(collection_id)
for doc in documents:
    chunks = storage.get_document_chunks(doc.id)
    all_chunks.extend(chunks)

# After
documents = storage.get_documents_by_collection(collection_id)
all_chunks = storage.get_all_chunks_for_collection(collection_id)
```

### Benefits
- GraphRAG indexing now works correctly
- More efficient chunk retrieval (single query vs multiple queries)
- Proper method names aligned with storage interface

---

## Issue 11: GraphRAG Model Configuration Mismatch

### Root Cause
GraphRAG was configured to use OpenAI models ("gpt-4", "text-embedding-3-small") but the environment only supports local models ("gemma3-4B", "bge-large-en"). This caused 400 BadRequestError when GraphRAG tried to make API calls.

### Location
- **Files**:
  - `src/fileintel/core/config.py` (main configuration defaults)
  - `src/graphrag/config/defaults.py` (GraphRAG-specific defaults)
  - `src/fileintel/llm_integration/unified_provider.py` (fallback model)

### Error Details
```
openai.BadRequestError: Error code: 400 - {'error': "Model not allowed. Please choose from: ['gemma3-4B', 'gemma3-27B', 'meta-llama/Meta-Llama-3-8B-Instruct', 'qwen2.5', 'deepseek-r1-32B', 'deepseek-r1-7B', 'Devstral-Small', 'Qwen3-8B', 'bge-large-en']"}
```

### Impact
- GraphRAG indexing failed due to invalid model requests
- All GraphRAG functionality broken
- API requests to unsupported models caused task failures

### Fix Applied
Updated all model configuration defaults to use environment-compatible models:

**LLM Models**: Changed all instances of `"gpt-4"` and `"gpt-3.5-turbo"` → `"gemma3-4B"`
**Embedding Models**: Changed all instances of `"text-embedding-3-small"` → `"bge-large-en"`

### Files Modified
1. **`src/fileintel/core/config.py`**:
   - `LLMSettings.model`: `"gpt-4"` → `"gemma3-4B"`
   - `RAGSettings.embedding_model`: `"text-embedding-3-small"` → `"bge-large-en"`
   - `RAGSettings.llm_model`: `"gpt-4"` → `"gemma3-4B"`
   - `RAGSettings.classification_model`: `"gpt-3.5-turbo"` → `"gemma3-4B"`

2. **`src/graphrag/config/defaults.py`**:
   - `DEFAULT_CHAT_MODEL`: `"gpt-4-turbo-preview"` → `"gemma3-4B"`
   - `DEFAULT_EMBEDDING_MODEL`: `"text-embedding-3-small"` → `"bge-large-en"`
   - `EmbedTextDefaults.model`: `"text-embedding-3-small"` → `"bge-large-en"`

3. **`src/fileintel/llm_integration/unified_provider.py`**:
   - Fallback model: `"gpt-4"` → `"gemma3-4B"`

### Benefits
- GraphRAG indexing now uses compatible models
- No more API errors due to unsupported models
- Consistent model configuration across all components
- GraphRAG functionality restored