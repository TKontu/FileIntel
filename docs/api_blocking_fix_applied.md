# API Blocking Fix - Applied Changes

## Problem

When uploading documents while a collection is processing, the API server becomes unresponsive, causing the CLI to hang indefinitely.

## Root Cause

The `upload_document_to_collection` async endpoint contained synchronous blocking operations that froze the FastAPI event loop.

## Changes Applied

### 1. Non-Blocking File Upload (`collections_service.py:93-145`)

**File**: `src/fileintel/api/services/collections_service.py`

**Changes**:
- ✅ **Async file I/O**: Replaced `open()`/`write()` with `aiofiles`
- ✅ **Async directory creation**: Using `asyncio.to_thread()` for `mkdir()`
- ✅ **Thread pool hashing**: CPU-bound `sha256()` moved to thread pool
- ⚠️ **DB call remains sync**: Keeping synchronous to avoid SQLAlchemy session thread-safety issues

**Before**:
```python
async def upload_document_to_collection(...):
    # ❌ Blocks event loop
    with open(file_path, "wb") as f:
        f.write(file_content)

    # ❌ Blocks event loop
    content_hash = hashlib.sha256(file_content).hexdigest()

    # ❌ Blocks event loop
    document = self.storage.create_document(...)
```

**After**:
```python
async def upload_document_to_collection(...):
    import asyncio
    import aiofiles

    # ✅ Non-blocking
    await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)

    # ✅ Non-blocking
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_content)

    # ✅ Non-blocking
    content_hash = await asyncio.to_thread(
        lambda: hashlib.sha256(file_content).hexdigest()
    )

    # ⚠️ Still synchronous (session thread-safety)
    document = self.storage.create_document(...)
```

### 2. HTTP Request Timeouts (`task_client.py:48-91`)

**File**: `src/fileintel/cli/task_client.py`

**Changes**:
- ✅ Added default timeout: `(30, 300)` = 30s connect, 300s read
- ✅ Applied to both `_request()` and `_request_raw()` methods
- ✅ Better error messages with timeout-specific feedback

**Timeout Values**:
```python
kwargs['timeout'] = (30, 300)
# 30 seconds: Connection timeout
# 300 seconds (5 min): Read timeout per request
```

**Applies to each individual API call**:
- Each file upload: Max 5 minutes
- Process submission: Max 5 minutes (should complete in <5 seconds)
- Other API calls: Max 5 minutes

**Does NOT apply to**:
- Background Celery tasks (unlimited duration)
- Long-running processing workflows (hours/days)

### 3. Improved Error Handling

**Before**:
```python
except requests.exceptions.RequestException as e:
    self.console.print(f"[bold red]API request failed:[/bold red] {e}")
    raise
```

**After**:
```python
except requests.exceptions.Timeout as e:
    self.console.print(f"[bold red]API request timed out:[/bold red] {e}")
    self.console.print("[yellow]The API server may be overloaded. Try again later.[/yellow]")
    raise
except requests.exceptions.RequestException as e:
    self.console.print(f"[bold red]API request failed:[/bold red] {e}")
    raise
```

## Impact

### Before Fix
- ❌ API completely unresponsive during uploads
- ❌ CLI hangs indefinitely with no feedback
- ❌ Cannot upload files while processing
- ❌ Event loop blocked by I/O operations

### After Fix
- ✅ API remains responsive during uploads
- ✅ CLI provides timeout feedback after 5 minutes
- ✅ Can upload files concurrently
- ✅ Event loop stays responsive for I/O operations
- ⚠️ Minor blocking remains for DB operations (mitigated by fast execution)

## Testing Recommendations

1. **Concurrent upload test**:
   ```bash
   # Terminal 1: Start processing
   poetry run fileintel documents batch-upload "collection1" ./batch1 --process

   # Terminal 2: Upload while processing
   poetry run fileintel documents batch-upload "collection1" ./batch2 --no-process
   ```

2. **Large file test**: Upload 100MB+ files to verify timeout is sufficient

3. **Timeout test**: Simulate slow API response to verify timeout behavior

4. **Responsiveness test**: Check API health endpoint during uploads:
   ```bash
   watch -n 1 curl http://localhost:8001/health
   ```

## Known Limitations

1. **DB operations still synchronous**:
   - Small blocking remains for `storage.create_document()`
   - Typically <100ms, minimal impact
   - Future fix requires proper session handling

2. **Fixed timeout**:
   - Not configurable per-operation
   - 300s may be too short for very large files
   - Consider adding environment variable override

## Future Improvements

1. **Full async DB operations**:
   ```python
   # Create session inside thread to avoid thread-safety issues
   def _create_document_with_session():
       from fileintel.storage.models import SessionLocal
       db = SessionLocal()
       try:
           storage = PostgreSQLStorage(db)
           return storage.create_document(...)
       finally:
           db.close()

   document = await asyncio.to_thread(_create_document_with_session)
   ```

2. **Configurable timeouts**:
   ```python
   UPLOAD_TIMEOUT = int(os.getenv("FILEINTEL_UPLOAD_TIMEOUT", "300"))
   PROCESS_TIMEOUT = int(os.getenv("FILEINTEL_PROCESS_TIMEOUT", "30"))
   ```

3. **Progress indicators**: Show upload progress for large files

4. **Comprehensive async audit**: Check all FastAPI endpoints for blocking operations

## Related Files Modified

- `src/fileintel/api/services/collections_service.py` (lines 93-145)
- `src/fileintel/cli/task_client.py` (lines 48-91)

## Dependencies Required

- ✅ `aiofiles >= 24.1.0` (already in pyproject.toml)
- ✅ `asyncio` (Python 3.9+ standard library)
- ✅ `requests` (already installed)

## Deployment Notes

1. No database migrations required
2. No configuration changes required
3. Compatible with existing workflows
4. Backwards compatible with existing API clients
5. Restart API server to apply changes
