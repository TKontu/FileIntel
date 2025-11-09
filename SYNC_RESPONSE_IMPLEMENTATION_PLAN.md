# Synchronous Response Implementation Plan

## Executive Summary

Implement a non-blocking synchronous request-response layer for FileIntel API v2 that:
- Keeps API layer fully non-blocking (supports thousands of concurrent requests)
- Maintains async Celery worker architecture
- Provides synchronous "feel" to clients via long-polling with Redis pub/sub
- Allows clients to get instant responses for fast queries without polling

## Current Architecture Analysis

### Existing Components

```
┌─────────────────────────────────────────────────────────────┐
│  Client                                                      │
│  POST /api/v2/collections/{id}/query                         │
│  {"question": "...", "query_mode": "sync"}                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI (src/fileintel/api/routes/query.py)                │
│  - CURRENTLY: Forces all to async mode (line 153-158)       │
│  - Returns: {"task_id": "abc-123"}                           │
│  - Client must poll: GET /api/v2/tasks/{task_id}             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Celery Worker (src/fileintel/tasks/graphrag_tasks.py)      │
│  - Queues: graphrag_queries, rag_processing, etc.           │
│  - Broker: Redis (redis://redis:6379/1)                     │
│  - Result Backend: Redis (redis://redis:6379/1)             │
└─────────────────────────────────────────────────────────────┘
```

### Key Infrastructure (Already Available)

1. **Redis**: Already configured for Celery broker + result backend
   - URL: `redis://redis:6379/1` (from `CelerySettings.broker_url`)
   - Shared between Celery and can be used for pub/sub

2. **Celery Task Queues**:
   - `graphrag_queries`: GraphRAG query operations
   - `rag_processing`: Vector queries, lightweight ops
   - All tasks inherit from `BaseFileIntelTask`

3. **FastAPI**: Async-native, supports long-polling via `asyncio`

4. **Query Tasks**:
   - `query_graph_global()`: Global graph queries (~30-60s)
   - `query_graph_local()`: Local graph queries (~20-40s)
   - `adaptive_graphrag_query()`: Adaptive routing
   - `query_vector()`: Vector queries (~1-3s)

## Proposed Architecture

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Client                                                      │
│  POST /api/v2/collections/{id}/query                         │
│  {"question": "...", "query_mode": "sync", "timeout": 120}   │
│  ↓                                                            │
│  HTTP connection stays OPEN, waiting for response...         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Handler (NON-BLOCKING)                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. Submit task to Celery queue                         │ │
│  │    task = query_vector.delay(...)                      │ │
│  │    task_id = task.id                                   │ │
│  │                                                         │ │
│  │ 2. Create SyncResponseHandler                          │ │
│  │    handler = SyncResponseHandler(redis_url)            │ │
│  │                                                         │ │
│  │ 3. Wait for completion (ASYNC, non-blocking)           │ │
│  │    result = await handler.wait_for_task_completion(    │ │
│  │        task_id=task_id,                                │ │
│  │        timeout=120                                     │ │
│  │    )                                                   │ │
│  │                                                         │ │
│  │    ↓ Event loop free to handle other requests          │ │
│  │    ↓ Subscribed to: "task:complete:{task_id}"          │ │
│  └────────────────────────────────────────────────────────┘ │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ (task submitted to queue)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Celery Worker (ASYNC PROCESSING)                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ @app.task                                              │ │
│  │ def query_vector(query, collection_id, ...):           │ │
│  │     # 1. Execute query                                 │ │
│  │     result = execute_vector_query(...)                 │ │
│  │                                                         │ │
│  │     # 2. Notify API (NEW CODE)                         │ │
│  │     notify_task_completion_sync(                       │ │
│  │         task_id=self.request.id,                       │ │
│  │         redis_url=config.celery.broker_url             │ │
│  │     )                                                   │ │
│  │     # Publishes to: "task:complete:{task_id}"          │ │
│  │                                                         │ │
│  │     # 3. Return result (stored in Celery backend)      │ │
│  │     return result                                      │ │
│  └────────────────────────────────────────────────────────┘ │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ PUBLISH task:complete:{task_id} → "ready"
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Redis Pub/Sub                                               │
│  Channel: "task:complete:abc-123"                            │
│  Message: "ready"                                            │
│  ↓                                                            │
│  Notification delivered to waiting API handler               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Handler (WAKES UP)                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. Receives pub/sub notification                       │ │
│  │                                                         │ │
│  │ 2. Fetch result from Celery backend                    │ │
│  │    task_result = AsyncResult(task_id)                  │ │
│  │    result = task_result.result                         │ │
│  │                                                         │ │
│  │ 3. Return HTTP response to client                      │ │
│  │    return ApiResponseV2(                               │ │
│  │        success=True,                                   │ │
│  │        data=result,                                    │ │
│  │        message="Query completed"                       │ │
│  │    )                                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Client                                                      │
│  Receives: {                                                 │
│    "success": true,                                          │
│    "data": {"answer": "...", "sources": [...]},              │
│    "message": "Query completed"                              │
│  }                                                            │
└─────────────────────────────────────────────────────────────┘
```

### Timeout Handling

```
┌─────────────────────────────────────────────────────────────┐
│  If timeout (e.g., 120s) expires before notification:        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ API Handler returns:                                   │ │
│  │ {                                                       │ │
│  │   "success": false,                                    │ │
│  │   "data": {"task_id": "abc-123"},                      │ │
│  │   "message": "Query timeout - task still running.      │ │
│  │              Use task_id to poll /api/v2/tasks/abc-123"│ │
│  │ }                                                       │ │
│  │                                                         │ │
│  │ Client can then:                                       │ │
│  │ - Poll GET /api/v2/tasks/abc-123 manually              │ │
│  │ - Or retry with longer timeout                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Components

### 1. SyncResponseHandler (NEW)

**File**: `src/fileintel/api/sync_response_handler.py` (CREATED)

**Responsibilities**:
- Subscribe to Redis pub/sub channel for task completion
- Wait asynchronously for notification (non-blocking)
- Fetch result from Celery backend when notified
- Handle timeouts gracefully

**Key Methods**:
```python
class SyncResponseHandler:
    async def wait_for_task_completion(
        task_id: str,
        timeout: int = 120
    ) -> Optional[Dict]:
        """
        Wait for task completion notification.

        - Creates Redis pub/sub subscription
        - Waits with asyncio.timeout (non-blocking)
        - Returns result or None on timeout
        """
```

### 2. Task Notification Helper (NEW)

**File**: `src/fileintel/api/sync_response_handler.py` (CREATED)

**Function**: `notify_task_completion_sync(task_id, redis_url)`

**Called from**: Celery tasks after completion

**Purpose**: Publish to Redis channel `task:complete:{task_id}` to wake up waiting API handlers

### 3. Modified Query API Endpoint (UPDATE)

**File**: `src/fileintel/api/routes/query.py`

**Changes**:
1. Remove "force async" logic (lines 147-158)
2. Add routing based on `query_mode`:
   - `query_mode="sync"` → Submit task + wait for notification
   - `query_mode="async"` → Submit task + return task_id immediately
3. Add new function `_submit_and_wait_for_task()`

### 4. Modified Celery Tasks (UPDATE)

**Files**:
- `src/fileintel/tasks/graphrag_tasks.py`
- Any other task files with query operations

**Changes**:
Add notification call after task completion:
```python
@app.task(bind=True, ...)
def query_vector(self, query, collection_id, ...):
    # Execute query
    result = do_work()

    # NEW: Notify waiting API handlers
    notify_task_completion_sync(
        task_id=self.request.id,
        redis_url=config.celery.broker_url
    )

    return result
```

## Technical Specifications

### Redis Pub/Sub Channels

**Channel naming**: `task:complete:{task_id}`

Example: `task:complete:a1b2c3d4-5678-90ab-cdef-1234567890ab`

**Message payload**: Simple string `"ready"` (minimal overhead)

**Why pub/sub?**
- Low latency (~1ms)
- Multiple API instances can each subscribe to different channels
- Automatic cleanup when subscriber disconnects
- No polling overhead

### Concurrency Characteristics

**API Layer**:
- Each sync request creates ONE async task (awaiting notification)
- FastAPI event loop can handle 10,000+ concurrent awaits
- Memory per waiting request: ~10KB (asyncio task + Redis connection)
- 1000 concurrent sync requests ≈ 10MB RAM overhead

**Worker Layer**:
- No changes to concurrency model
- Still processes tasks asynchronously
- Notification adds <1ms overhead per task

**Redis**:
- Pub/sub is separate from Celery queue
- No impact on broker performance
- Auto-cleanup of channels when no subscribers

### Timeout Behavior

**Default timeout**: 120 seconds (configurable per request)

**Timeout strategies**:
1. **Vector queries** (fast): timeout=30s recommended
2. **Graph queries** (slow): timeout=120s recommended
3. **Adaptive queries**: timeout=60s (fallback to async on timeout)

**What happens on timeout**:
- API returns task_id to client
- Task continues running in background
- Client can poll `/api/v2/tasks/{task_id}` for result

## Performance Characteristics

### Latency Comparison

| Mode | Current (Always Async) | Proposed (Sync Option) |
|------|------------------------|------------------------|
| **Vector Query** | Task submit (50ms) + Poll 1 (50ms) + Poll 2 (50ms) = **150ms overhead** | Task submit (50ms) + Notification (1ms) = **51ms overhead** |
| **Graph Query** | Task submit (50ms) + Poll 1-10 (500ms) = **550ms overhead** | Task submit (50ms) + Notification (1ms) = **51ms overhead** |

**Improvement**: ~3-10x reduction in overhead for synchronous mode

### Scalability

**API instances**: Multiple API containers can run concurrently
- Each subscribes to different task channels
- No shared state between instances
- Redis pub/sub handles routing automatically

**Load testing target**:
- 1000 concurrent sync requests
- 100 requests/sec sustained
- <100ms overhead per request

## Implementation Phases

### Phase 1: Core Infrastructure (Files Created)
- [x] `src/fileintel/api/sync_response_handler.py` - Handler + notification functions

### Phase 2: API Integration (Modify)
- [ ] `src/fileintel/api/routes/query.py` - Add sync mode routing
- [ ] Add `_submit_and_wait_for_task()` function
- [ ] Update OpenAPI schema documentation

### Phase 3: Worker Integration (Modify)
- [ ] `src/fileintel/tasks/graphrag_tasks.py`:
  - [ ] `query_graph_global()` - Add notification
  - [ ] `query_graph_local()` - Add notification
  - [ ] `adaptive_graphrag_query()` - Add notification
  - [ ] `query_vector()` - Add notification (if exists as separate task)

### Phase 4: Testing
- [ ] Unit tests for `SyncResponseHandler`
- [ ] Integration tests for sync query flow
- [ ] Load testing for concurrent sync requests
- [ ] Timeout behavior verification

### Phase 5: Documentation
- [ ] Update API documentation
- [ ] Add usage examples
- [ ] Update OpenAPI spec
- [ ] Add troubleshooting guide

## Configuration

### New Environment Variables (Optional)

```bash
# Query API Configuration
API_QUERY_SYNC_DEFAULT_TIMEOUT=120  # Default timeout for sync mode (seconds)
API_QUERY_SYNC_MAX_TIMEOUT=300      # Maximum allowed timeout
API_QUERY_SYNC_ENABLED=true         # Feature flag to enable/disable sync mode
```

### Redis Configuration (Existing)

Uses existing Celery Redis configuration:
```bash
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/1
```

## Benefits

### For Users
1. **Simpler client code**: Single HTTP request instead of submit + poll loop
2. **Lower latency**: ~3-10x faster response delivery
3. **Better UX**: Immediate results for fast queries
4. **Backward compatible**: Async mode still available

### For System
1. **API stays non-blocking**: Uses asyncio, not threads
2. **Scalable**: Supports thousands of concurrent sync requests
3. **Efficient**: Minimal overhead (1ms notification vs 500ms polling)
4. **Resilient**: Graceful timeout handling with fallback to async

## Risks and Mitigations

### Risk 1: Redis Pub/Sub Reliability
**Concern**: What if notification is lost?

**Mitigation**:
- Timeout handling returns task_id for manual polling
- Celery result backend provides fallback
- Health check monitors Redis connectivity

### Risk 2: API Connection Limits
**Concern**: Too many waiting connections

**Mitigation**:
- FastAPI handles 10,000+ concurrent connections easily
- Configurable max timeout prevents indefinite waits
- Load balancer distributes across multiple API instances

### Risk 3: Worker Notification Failure
**Concern**: Worker crashes before sending notification

**Mitigation**:
- API timeout returns task_id
- Celery task state tracking (PENDING/SUCCESS/FAILURE)
- Dead letter queue for failed tasks

## Success Criteria

1. **Functional**: Sync mode returns results without polling
2. **Performance**: <100ms overhead for notification delivery
3. **Scalability**: 1000+ concurrent sync requests supported
4. **Reliability**: <0.1% notification failures
5. **Backward Compatibility**: Existing async mode unchanged

## Rollout Plan

### Stage 1: Deploy Infrastructure
- Deploy `sync_response_handler.py`
- No user-facing changes yet

### Stage 2: Beta Testing
- Enable sync mode with feature flag
- Test with internal users
- Monitor Redis pub/sub metrics

### Stage 3: Gradual Rollout
- Enable for vector queries only (fast, low risk)
- Monitor for 1 week
- Enable for all query types

### Stage 4: Documentation
- Update API docs
- Publish usage examples
- Announce feature

## Monitoring and Metrics

### Key Metrics to Track

1. **Sync request count**: Gauge of feature adoption
2. **Average wait time**: Latency from submit to notification
3. **Timeout rate**: % of sync requests timing out
4. **Notification failures**: Redis pub/sub delivery failures
5. **Concurrent waiting requests**: API connection pressure

### Alerts

- Notification failure rate >1%
- Average wait time >5 seconds
- Timeout rate >10%
- Redis pub/sub connection errors

## Questions for Review

1. **Timeout values**: Are 120s default / 300s max appropriate?
2. **Feature flag**: Should sync mode be opt-in initially?
3. **Redis database**: Use separate Redis DB for pub/sub? (currently shares DB 1 with Celery)
4. **Error handling**: How should we handle partial worker failures?
5. **Monitoring**: What additional metrics should we track?

---

## Next Steps

After plan approval:
1. Review and approve implementation phases
2. Set up monitoring infrastructure
3. Implement Phase 2 (API integration)
4. Implement Phase 3 (Worker integration)
5. Deploy to staging for testing
