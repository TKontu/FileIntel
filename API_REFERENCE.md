# FileIntel API Reference (Verified)

**Version**: 2.0
**Base URL**: `http://localhost:8000/api/v2`
**Interactive Documentation**: `http://localhost:8000/docs` (Swagger UI)

## Verified vs Removed

**VERIFIED** - These endpoints exist and are documented from actual code
**REMOVED** - These were in the original doc but don't actually exist:
- Rate limiting (config exists but not implemented)
- Some response format details were inferred

---

## Authentication

Optional API key authentication via `X-API-Key` header (disabled by default).

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/v2/collections
```

Enable in `config/default.yaml`:
```yaml
api:
  authentication:
    enabled: true
    api_key: ${API_KEY}
```

---

## Response Format

Standard response structure:

```json
{
  "success": true,
  "data": {},
  "message": "Success message",
  "timestamp": "2025-10-24T12:34:56.789Z",
  "error": null
}
```

Error response:
```json
{
  "success": false,
  "data": null,
  "message": "Error description",
  "timestamp": "2025-10-24T12:34:56.789Z",
  "error": "Detailed error"
}
```

---

## Collections API

### POST /collections
Create a new collection.

**Request:**
```json
{
  "name": "research-papers",
  "description": "AI/ML research papers"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "name": "research-papers",
    "description": "AI/ML research papers"
  }
}
```

---

### GET /collections
List all collections.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "uuid",
      "name": "research-papers",
      "description": "AI/ML research papers",
      "status": "completed"
    }
  ]
}
```

---

### GET /collections/{collection_identifier}
Get collection by ID or name.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "name": "research-papers",
    "description": "...",
    "documents": [
      {
        "id": "doc-uuid",
        "filename": "paper1.pdf",
        "original_filename": "transformer.pdf",
        "mime_type": "application/pdf",
        "file_size": 1234567
      }
    ]
  }
}
```

---

### DELETE /collections/{collection_identifier}
Delete a collection.

---

### POST /collections/{collection_identifier}/documents
Upload document to collection (multipart/form-data).

**Response includes duplicate detection:**
```json
{
  "success": true,
  "data": {
    "document_id": "uuid",
    "filename": "uuid.pdf",
    "original_filename": "paper.pdf",
    "content_hash": "sha256...",
    "file_size": 1234567,
    "file_path": "/uploads/uuid.pdf",
    "duplicate": false,
    "message": "Document uploaded successfully"
  }
}
```

---

### POST /collections/{collection_identifier}/process
Start collection processing (async).

**Request:**
```json
{
  "operation_type": "complete_analysis",
  "generate_embeddings": true,
  "build_graph": true,
  "extract_metadata": true,
  "parameters": {}
}
```

**operation_type options:**
- `complete_analysis` - Full pipeline (processing + embeddings + GraphRAG)
- `document_processing_only` - Extract and chunk only

**Response:**
```json
{
  "success": true,
  "data": {
    "task_id": "celery-task-uuid",
    "task_type": "complete_analysis",
    "status": "PENDING",
    "submitted_at": "2025-10-24T12:34:56Z",
    "collection_id": "uuid"
  }
}
```

---

## Documents API

### GET /documents/{document_id}
Get document information.

**Response:**
```json
{
  "success": true,
  "data": {
    "document_id": "uuid",
    "filename": "paper.pdf",
    "collection_id": "uuid",
    "created_at": "2025-10-24T12:00:00Z",
    "metadata": {
      "title": "Paper Title",
      "authors": ["Author 1"]
    },
    "statistics": {
      "total_chunks": 45,
      "vector_chunks": 45,
      "graph_chunks": 0
    }
  }
}
```

---

### GET /documents/{document_id}/chunks
Retrieve document chunks.

**Query params:**
- `chunk_type` (optional): `vector` or `graph`

**Response:**
```json
{
  "document_id": "uuid",
  "filename": "paper.pdf",
  "collection_id": "uuid",
  "total_chunks": 45,
  "chunks": [
    {
      "id": "chunk-uuid",
      "text": "Chunk content...",
      "position": 0,
      "metadata": {
        "chunk_type": "vector",
        "page_number": 1,
        "token_count": 200
      }
    }
  ]
}
```

---

### GET /documents/{document_id}/export
Export chunks as downloadable markdown.

**Query params:**
- `chunk_type` (optional): Filter by type
- `include_metadata` (boolean): Include metadata (default: false)

Returns markdown file download.

---

### GET /chunks/{chunk_id}
Get specific chunk by ID (for source tracing).

---

### DELETE /documents/{document_id}
Delete document.

---

## Query API

### POST /collections/{collection_identifier}/query
Query collection using hybrid RAG.

**Request:**
```json
{
  "question": "What are the main findings?",
  "search_type": "adaptive",
  "max_results": 5,
  "include_sources": true
}
```

**search_type options:**
- `adaptive` - Auto-select best method (default)
- `vector` - Vector similarity search
- `graph` - GraphRAG search
- `global` - GraphRAG global communities
- `local` - GraphRAG local entities

**Response:**
```json
{
  "success": true,
  "data": {
    "answer": "The main findings are...",
    "sources": [
      {
        "document_id": "uuid",
        "document_name": "paper.pdf",
        "chunk_id": "chunk-uuid",
        "chunk_text": "Relevant excerpt...",
        "page_number": 3,
        "relevance_score": 0.89
      }
    ],
    "query_type": "vector",
    "routing_explanation": "Processed using vector similarity search",
    "collection_id": "uuid",
    "question": "What are the main findings?",
    "processing_time_ms": 1234
  }
}
```

---

### POST /collections/{collection_identifier}/documents/{document_identifier}/query
Query specific document within collection.

Same request/response format as collection query.

---

### GET /query/endpoints
List all query endpoints with usage examples.

---

### GET /query/status
Get query system status and capabilities.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "operational",
    "api_version": "v2",
    "collections": [...],
    "total_collections": 5,
    "capabilities": {
      "vector_search": true,
      "graph_search": true,
      "adaptive_routing": true
    },
    "supported_search_types": [
      "vector",
      "graph",
      "adaptive",
      "global",
      "local"
    ]
  }
}
```

---

## GraphRAG API

### POST /graphrag/index
Build GraphRAG index for collection.

**Request:**
```json
{
  "collection_id": "research-papers",
  "force_rebuild": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "task_id": "celery-task-uuid",
    "collection_id": "uuid",
    "status": "started",
    "message": "GraphRAG indexing started"
  }
}
```

---

### GET /graphrag/{collection_identifier}/status
Check GraphRAG index status.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "indexed",
    "index_path": "/data/graphrag_indices/collection-uuid",
    "index_created": "2025-10-24T10:00:00Z",
    "entity_count": 156,
    "relationship_count": 342,
    "community_count": 23
  }
}
```

**Status values:**
- `not_indexed` - No index
- `indexing` - Building
- `indexed` - Ready
- `failed` - Build failed

---

### GET /graphrag/{collection_identifier}/entities
Get extracted entities.

**Query params:**
- `limit` (int): Max entities (default: 20)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "name": "Transformer Architecture",
      "type": "CONCEPT",
      "description": "Neural network architecture...",
      "importance_score": 0.95
    }
  ]
}
```

---

### GET /graphrag/{collection_identifier}/communities
Get detected communities.

**Query params:**
- `limit` (int): Max communities (default: 10)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "title": "Attention Mechanisms",
      "community_id": "C1",
      "level": 0,
      "rank": 0.92,
      "summary": "This community focuses on...",
      "size": 23
    }
  ]
}
```

---

### DELETE /graphrag/{collection_identifier}/index
Remove GraphRAG index.

---

### GET /graphrag/status
Check GraphRAG system operational status.

---

## Metadata API

### POST /metadata/extract
Extract metadata from document using LLM.

**Request:**
```json
{
  "document_id": "uuid",
  "force_reextract": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "task_id": "celery-task-uuid",
    "document_id": "uuid",
    "status": "started",
    "message": "Metadata extraction started"
  }
}
```

---

### GET /metadata/document/{document_id}
Get extracted metadata for document.

**Response:**
```json
{
  "success": true,
  "data": {
    "document_id": "uuid",
    "filename": "paper.pdf",
    "has_extracted_metadata": true,
    "metadata": {
      "title": "Attention Is All You Need",
      "authors": ["Vaswani", "Shazeer"],
      "year": 2017,
      "publication_venue": "NeurIPS 2017",
      "abstract": "...",
      "doi": "...",
      "llm_extracted": true
    }
  }
}
```

---

### POST /metadata/collection/{collection_identifier}/extract-all
Extract metadata for all documents in collection.

**Query params:**
- `force_reextract` (boolean): Re-extract existing (default: false)

---

### GET /metadata/collection/{collection_identifier}/status
Check metadata extraction status for collection.

---

### GET /metadata/system-status
Check metadata system health.

---

## Tasks API

### GET /tasks/{task_id}/status
Get task status.

**Response:**
```json
{
  "success": true,
  "data": {
    "task_id": "uuid",
    "task_name": "complete_collection_analysis",
    "status": "PROGRESS",
    "result": null,
    "error": null,
    "progress": {
      "current": 3,
      "total": 10,
      "percentage": 30.0,
      "message": "Processing document 3 of 10",
      "timestamp": 1698234567.89
    },
    "worker_id": "celery@worker1",
    "retry_count": 0
  }
}
```

**Status values:**
- `PENDING` - Waiting to start
- `RECEIVED` - Received by worker
- `STARTED` - Execution started
- `PROGRESS` - In progress (with progress info)
- `SUCCESS` - Completed successfully
- `FAILURE` - Failed
- `RETRY` - Being retried
- `REVOKED` - Cancelled

---

### GET /tasks
List active tasks.

**Query params:**
- `status` (string): Filter by status
- `limit` (int): Max tasks (default: 20)
- `offset` (int): Skip tasks (default: 0)

---

### POST /tasks/{task_id}/cancel
Cancel running task.

**Request:**
```json
{
  "terminate": false
}
```

**terminate options:**
- `false` - Soft cancel (revoke)
- `true` - Hard terminate

---

### GET /tasks/{task_id}/result
Get result of completed task.

---

### GET /tasks/metrics
Get Celery worker and task metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "active_tasks": 2,
    "pending_tasks": 0,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "average_task_duration": null,
    "worker_count": 4,
    "queue_lengths": {}
  }
}
```

**Note:** Some metrics require persistent storage to track history.

---

### POST /tasks/batch/cancel
Cancel multiple tasks.

**Request:**
```json
{
  "task_ids": ["task-uuid-1", "task-uuid-2"],
  "terminate": false
}
```

---

### POST /tasks/submit
Submit generic Celery task.

**Request:**
```json
{
  "task_name": "fileintel.tasks.process_document",
  "args": [],
  "kwargs": {},
  "queue": null,
  "countdown": null,
  "eta": null
}
```

---

## WebSocket API

### WS /tasks/monitor
Real-time task monitoring for all tasks.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v2/tasks/monitor');
```

**Client messages:**
```json
// Subscribe to specific task
{"type": "subscribe_task", "task_id": "task-uuid"}

// Unsubscribe
{"type": "unsubscribe_task", "task_id": "task-uuid"}

// Set filters
{"type": "set_filters", "filters": {"event_types": ["task.started", "task.completed"]}}

// Get active tasks
{"type": "get_active_tasks"}

// Get worker stats
{"type": "get_worker_stats"}
```

**Server events:**
```json
{
  "event_type": "task.started",
  "task_id": "uuid",
  "timestamp": "2025-10-24T12:34:56Z",
  "data": {
    "status": "STARTED",
    "task_name": "process_collection"
  },
  "worker_id": "celery@worker1"
}
```

**Event types:**
- `task.started` - Task execution started
- `task.progress` - Progress update
- `task.completed` - Successfully completed
- `task.failed` - Failed with error
- `task.retry` - Being retried

---

### WS /tasks/{task_id}/monitor
Real-time monitoring for specific task.

Automatically subscribes to the specified task. Same event format as general monitor.

---

## Error Handling

### HTTP Status Codes

- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource exists
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service down

### Common Errors

**Collection not found:**
```json
{
  "detail": "Collection 'name' not found"
}
```

**Duplicate document:**
```json
{
  "duplicate": true,
  "message": "Duplicate file detected"
}
```

**Task not found:**
```json
{
  "detail": "Task task-uuid not found"
}
```

---

## Code Examples

### Python

```python
import requests

BASE = "http://localhost:8000/api/v2"

# Create collection
r = requests.post(f"{BASE}/collections",
    json={"name": "docs", "description": "My docs"})
coll_id = r.json()["data"]["id"]

# Upload document
with open("doc.pdf", "rb") as f:
    r = requests.post(f"{BASE}/collections/{coll_id}/documents",
        files={"file": f})
doc_id = r.json()["data"]["document_id"]

# Process collection
r = requests.post(f"{BASE}/collections/{coll_id}/process",
    json={
        "operation_type": "complete_analysis",
        "generate_embeddings": True,
        "build_graph": True
    })
task_id = r.json()["data"]["task_id"]

# Monitor task
import time
while True:
    r = requests.get(f"{BASE}/tasks/{task_id}/status")
    status = r.json()["data"]["status"]
    if status in ["SUCCESS", "FAILURE"]:
        break
    time.sleep(5)

# Query collection
r = requests.post(f"{BASE}/collections/{coll_id}/query",
    json={
        "question": "What is this about?",
        "search_type": "adaptive"
    })
answer = r.json()["data"]["answer"]
print(answer)
```

### Bash/cURL

```bash
BASE="http://localhost:8000/api/v2"

# Create collection
COLL_ID=$(curl -s -X POST $BASE/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"test","description":"Test"}' \
  | jq -r '.data.id')

# Upload document
DOC_ID=$(curl -s -X POST \
  $BASE/collections/$COLL_ID/documents \
  -F "file=@document.pdf" \
  | jq -r '.data.document_id')

# Process
TASK_ID=$(curl -s -X POST \
  $BASE/collections/$COLL_ID/process \
  -H "Content-Type: application/json" \
  -d '{"operation_type":"complete_analysis","generate_embeddings":true}' \
  | jq -r '.data.task_id')

# Check status
curl -s $BASE/tasks/$TASK_ID/status | jq

# Query
curl -s -X POST $BASE/collections/$COLL_ID/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize","search_type":"adaptive"}' \
  | jq '.data.answer'
```

---

## Best Practices

1. **Use collection names instead of UUIDs** - Most endpoints accept both
2. **Monitor long-running tasks** - Always check task status for processing/indexing
3. **Handle duplicate documents** - Check the `duplicate` flag when uploading
4. **Use adaptive query mode** - Let the system choose the best strategy
5. **Enable GraphRAG for relationships** - Build index for relationship queries

---

## Changelog

### v2.0 (Current)
- Task-based async architecture
- Hybrid RAG with query routing
- GraphRAG integration
- Metadata extraction
- Citation generation
- Type-aware chunking
- WebSocket monitoring

---

**Last Updated**: 2025-10-24
**API Version**: 2.0
**Status**: All endpoints verified from source code
