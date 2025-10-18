# API Blocking Issue Analysis

## Problem Statement

When uploading documents while a collection is processing, the **entire API server becomes unresponsive**. The CLI hangs indefinitely with no response.

## Root Cause

The `upload_document_to_collection` async endpoint contains **synchronous blocking operations** that freeze the FastAPI event loop:

### Blocking Operations in `collections_service.py:93-138`

```python
async def upload_document_to_collection(...):
    # ❌ BLOCKING #1: Synchronous file I/O (lines 110-111)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # ❌ BLOCKING #2: CPU-bound hashing (line 116)
    content_hash = hashlib.sha256(file_content).hexdigest()

    # ❌ BLOCKING #3: Synchronous database call (lines 119-127)
    document = self.storage.create_document(...)
```

When these operations run in an async function, they **block the entire event loop**, preventing the API from handling ANY other requests.

## Impact

- **Symptom**: API completely unresponsive during file uploads
- **User Experience**: CLI hangs with no feedback (not even timeouts)
- **Severity**: Critical - prevents concurrent operations

## Existing System Patterns

### Already Using `asyncio.to_thread`

The codebase ALREADY uses `asyncio.to_thread` with storage operations in multiple places:

**GraphRAG Service** (`graphrag_service.py`):
```python
collection = await asyncio.to_thread(self.storage.get_collection, collection_id)
index_info = await asyncio.to_thread(self.storage.get_graphrag_index_info, ...)
```

**Query Routes** (`query.py`):
```python
result = await asyncio.to_thread(vector_service.query, question, collection_id)
```

### Dependencies Available

- ✅ `aiofiles >= 24.1.0` already in `pyproject.toml`
- ✅ `asyncio.to_thread` built into Python 3.9+

## Potential Solutions

### Solution 1: Full Async Conversion (Safest)

```python
async def upload_document_to_collection(...):
    import asyncio
    import aiofiles
    import hashlib

    # ✅ Async file I/O
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_content)

    # ✅ CPU-bound in thread pool
    content_hash = await asyncio.to_thread(
        lambda: hashlib.sha256(file_content).hexdigest()
    )

    # ✅ DB operation in thread pool
    document = await asyncio.to_thread(
        self.storage.create_document,
        filename=unique_filename,
        ...
    )
```

**Pros**:
- Fully non-blocking
- Follows existing patterns in codebase
- Uses available dependencies

**Cons**:
- Need to verify SQLAlchemy session thread-safety
- More extensive changes

### Solution 2: Partial Async (File I/O Only)

```python
async def upload_document_to_collection(...):
    import aiofiles

    # ✅ Async file I/O
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_content)

    # Keep sync (already fast for small files)
    content_hash = hashlib.sha256(file_content).hexdigest()

    # Keep sync DB call
    document = self.storage.create_document(...)
```

**Pros**:
- Solves most blocking (file I/O typically slowest)
- Minimal changes
- Lower risk

**Cons**:
- Still some blocking for hash + DB operations
- Not fully optimal

### Solution 3: Add HTTP Timeouts (Band-Aid)

Add timeout to CLI client requests:

```python
# task_client.py
def _request(...):
    if 'timeout' not in kwargs:
        kwargs['timeout'] = (30, 300)  # 30s connect, 300s read
```

**Pros**:
- Prevents indefinite hangs
- Provides user feedback
- Simple change

**Cons**:
- Doesn't fix root cause
- API still blocks
- Poor user experience

## SQLAlchemy Session Thread Safety Concerns

### Current Setup

```python
# dependencies.py
def get_storage():
    db = SessionLocal()  # Creates session in request thread
    try:
        yield PostgreSQLStorage(db)
    finally:
        db.close()
```

**Problem**: SQLAlchemy sessions are **NOT thread-safe**. If we use `asyncio.to_thread()` to run `storage.create_document()` in a thread pool worker, we're accessing a session created in a different thread.

### Why Existing Code Works

The codebase already uses `asyncio.to_thread` with storage operations, which suggests:

1. **Read operations might be safe enough**: Most existing uses are reads (`get_collection`, `get_graphrag_index_info`)
2. **Short-lived operations**: Operations complete before thread switches
3. **SQLAlchemy engine is thread-safe**: Connection pool handles thread-local connections

However, **write operations** (`create_document`) are riskier:
- Require transactions
- Modify session state
- May trigger flushes/commits

### Recommended Approach

If using `asyncio.to_thread` for database writes:

1. **Create session inside the thread function**:
   ```python
   def _create_document():
       db = SessionLocal()
       try:
           storage = PostgreSQLStorage(db)
           return storage.create_document(...)
       finally:
           db.close()

   document = await asyncio.to_thread(_create_document)
   ```

2. **OR use dependency injection differently** for background operations

## Recommendations

### Immediate Fix (Low Risk)

1. ✅ Add HTTP timeouts to CLI client (Solution 3)
2. ✅ Convert file I/O to async using `aiofiles` (Solution 2)

This provides user feedback and eliminates the largest blocking operation.

### Long-Term Fix (Optimal)

1. Full async conversion (Solution 1) with proper session handling
2. Consider async SQLAlchemy (if upgrading to SQLAlchemy 2.0+)
3. Document async patterns for future development

## Testing Requirements

Before deploying any fix:

1. **Concurrent upload test**: Upload files while collection processes
2. **Session safety test**: Verify no race conditions in DB writes
3. **Performance test**: Measure blocking reduction
4. **Error handling**: Ensure proper cleanup on failures

## Related Issues

- Other async endpoints may have similar blocking issues
- Need comprehensive audit of all FastAPI endpoints
- Consider adding event loop monitoring/alerting
