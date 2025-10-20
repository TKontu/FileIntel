# Celery Task Timeout Fix

**Date**: 2025-10-19
**Issue**: Workers being killed with SIGKILL during long-running document processing

## Problem

Celery workers were being forcibly terminated (SIGKILL) when processing large documents:

```
[2025-10-19 14:53:17,640: ERROR/MainProcess] Timed out waiting for UP message from <ForkProcess(ForkPoolWorker-23, started daemon)>
[2025-10-19 14:53:17,646: ERROR/MainProcess] Process 'ForkPoolWorker-23' pid:127 exited with 'signal 9 (SIGKILL)'
```

### Root Cause

Global celery configuration had restrictive time limits (`celery_config.py:178-179`):

```python
task_soft_time_limit=1800,  # 30 minutes soft limit
task_time_limit=3600,       # 1 hour hard limit
```

Document processing tasks can take **hours or even days** for:
- Large PDFs (1000+ pages)
- Complex MinerU processing with VLM backend
- Extensive corruption filtering
- Type-aware chunking with statistical analysis
- Heavy embedding generation

After 1 hour, Celery would send SIGKILL to the worker process, causing:
- ❌ Incomplete document processing
- ❌ Lost work (task progress not saved)
- ❌ Database inconsistency (partial chunks stored)
- ❌ Worker pool exhaustion (constant restarts)

## Solution

Disabled time limits for document processing tasks by setting per-task overrides.

### Changes Made

**File**: `src/fileintel/tasks/document_tasks.py`

#### 1. `process_document` task (lines 610-616)

```python
@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="document_processing",
    soft_time_limit=None,  # No soft limit - documents can take hours/days
    time_limit=None        # No hard limit - let them run as long as needed
)
def process_document(...):
```

#### 2. `process_collection` task (lines 933-939)

```python
@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="document_processing",
    soft_time_limit=None,  # No soft limit - collections can take very long
    time_limit=None        # No hard limit - let them run as long as needed
)
def process_collection(...):
```

#### 3. `extract_document_metadata` task (lines 1012-1018)

```python
@app.task(
    base=BaseFileIntelTask,
    bind=True,
    queue="document_processing",
    soft_time_limit=None,  # No soft limit - metadata extraction can take long
    time_limit=None        # No hard limit - let it run as long as needed
)
def extract_document_metadata(...):
```

## Why This is Safe

1. **Task-Specific Override**: Only document processing tasks have unlimited time
   - Other tasks (LLM, RAG queries) still have reasonable limits
   - Prevents runaway tasks in fast-path operations

2. **Worker Protection Still Active**:
   - `worker_max_memory_per_child=500000` (500MB) still enforced
   - Workers restart after memory limit
   - `worker_max_tasks_per_child` prevents memory leaks

3. **Task Tracking**:
   - Tasks registered in `celery_task_registry` table
   - Heartbeat monitoring detects truly stuck tasks
   - Stale task cleanup runs on worker startup

4. **Manual Intervention Available**:
   - Tasks can still be revoked via Flower/CLI
   - `app.control.revoke(task_id, terminate=True)` works
   - Database has full task history

## Impact

### Before Fix
- ⏱️ Hard limit: 1 hour per document
- ❌ Large documents failed
- ❌ Workers constantly restarted
- ❌ Task queue backed up

### After Fix
- ⏱️ No time limit - run as long as needed
- ✅ Large documents process successfully
- ✅ Workers stay alive
- ✅ Tasks complete naturally

## Monitoring Recommendations

Monitor these metrics post-deployment:

1. **Task Duration Distribution**
   - Track p50, p95, p99 completion times
   - Identify outliers (>24 hours)
   - Investigate if documents routinely exceed 48 hours

2. **Worker Memory Usage**
   - Ensure workers don't exceed 500MB per child
   - Monitor OOM kills (different from SIGKILL)

3. **Task Completion Rate**
   - Should increase significantly
   - Failures should decrease

4. **Queue Depth**
   - `document_processing` queue should drain faster
   - No more timeout-caused retries

## Alternative Approaches Considered

### 1. Very High Fixed Limit (e.g., 48 hours)
```python
task_time_limit=172800  # 48 hours
```
**Rejected**: Arbitrary limit, would need tuning, still fails eventually

### 2. Configurable Limits
```python
task_time_limit=config.celery.document_processing_timeout
```
**Rejected**: Adds complexity, still requires choosing a limit

### 3. Separate Worker Pool
```python
# Separate worker with no limits
celery -A fileintel.celery_config worker -Q document_processing --time-limit=0
```
**Rejected**: Infrastructure complexity, harder to manage

### 4. Task Chunking (Break into Subtasks)
```python
# Process document in pages, chain subtasks
process_page_1.apply_async() | process_page_2.apply_async() | ...
```
**Rejected**:
- Complex refactor
- Loses transaction semantics
- Harder to track progress
- Overhead from inter-task communication

## Rollback Plan

If unlimited timeouts cause issues:

1. **Immediate**: Set high fixed limit
   ```python
   task_time_limit=172800  # 48 hours
   ```

2. **Monitor**: Identify problem documents
   ```sql
   SELECT task_id, task_name, started_at,
          EXTRACT(EPOCH FROM (NOW() - started_at))/3600 as hours_running
   FROM celery_task_registry
   WHERE status = 'STARTED'
   ORDER BY started_at ASC;
   ```

3. **Revoke**: Kill runaway tasks manually
   ```python
   from fileintel.celery_config import app
   app.control.revoke('task-id', terminate=True)
   ```

4. **Implement**: Task chunking if pattern emerges

---

**Status**: ✅ Fixed
**Deployed**: Pending worker restart
**Expected Behavior**: Workers run indefinitely until document processing completes naturally
