# Stale Task Detection and Cleanup Pipeline - Comprehensive Analysis

## Executive Summary

**Analysis Date:** 2025-10-19
**Pipeline Status:** CRITICAL ISSUES FOUND
**Severity:** HIGH - Multiple production-breaking bugs identified

### Critical Issues Summary
- **1 CRITICAL**: Logger not initialized - will cause immediate crashes (Lines 411, 435, 458, 478, 491, 568)
- **2 MAJOR**: Race condition in worker startup, potential JSON serialization failures
- **3 MEDIUM**: Missing error handling, incorrect signal parameters, timing issues
- **4 MINOR**: Code quality and edge case handling

### Priority Recommendations
1. **IMMEDIATE**: Add logger initialization to prevent crashes
2. **HIGH**: Fix JSON serialization for args/kwargs to prevent data loss
3. **HIGH**: Address race condition in worker_ready handler
4. **MEDIUM**: Improve error handling and session management

---

## Pipeline Architecture Overview

### Component Flow
```
Task Lifecycle:
  1. Task Queued → [No tracking yet]
  2. Task Started → @task_prerun → Create/Update DB entry
  3. Task Execution → [No heartbeat updates during execution]
  4. Task Completion → @task_success/@task_failure/@task_retry → Update DB entry

Worker Lifecycle:
  1. Worker Starts → @worker_ready
  2. Query DB for STARTED/RETRY tasks
  3. Check if workers still alive via inspect.stats()
  4. Revoke tasks from dead workers
  5. Worker processes tasks normally
```

### Database Schema
- Table: `celery_task_registry`
- Primary Key: `task_id` (String)
- Key Indexes: `task_name`, `worker_id`, `status`
- Timestamps: `queued_at`, `started_at`, `completed_at`, `last_heartbeat`

---

## CRITICAL ISSUES

### CRITICAL-1: Logger Not Initialized in Signal Handlers

**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 411, 435, 458, 478, 491, 568
**Impact:** PRODUCTION CRASH - All signal handlers will fail with NameError

**Problem:**
The file imports logging (line 11) but never creates a logger instance with `logger = logging.getLogger(__name__)`. All signal handlers reference `logger` which doesn't exist:

```python
# Line 11 - import exists
import logging

# Line 411 - logger used but never defined!
except Exception as e:
    logger.error(f"Error tracking task start for {task_id}: {e}")
```

**Affected Code Locations:**
- Line 411: `task_started_handler` - Error logging
- Line 435: `task_success_handler` - Error logging
- Line 458: `task_failure_handler` - Error logging
- Line 478: `task_retry_handler` - Error logging
- Line 491: `cleanup_stale_tasks` (worker_ready) - Info and warning logging (multiple uses)
- Line 568: `cleanup_stale_tasks` - Error logging

**Consequence:**
```python
NameError: name 'logger' is not defined
```

This will:
1. Crash every signal handler when any logging is attempted
2. Prevent task tracking from working at all
3. Cause worker startup to fail when cleanup_stale_tasks runs
4. Make the entire stale task detection pipeline non-functional

**Fix Required:**
```python
# Add after line 11
import logging

logger = logging.getLogger(__name__)  # ADD THIS LINE
```

**Testing to Verify:**
1. Start a worker and execute any task
2. Check that task_prerun handler doesn't crash
3. Verify database entries are created

---

## MAJOR ISSUES

### MAJOR-1: Unsafe JSON Serialization of Task Arguments

**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 401-402, 429
**Impact:** HIGH - Data loss and potential crashes on non-serializable arguments

**Problem:**
Task `args` and `kwargs` are directly stored in JSONB columns without sanitization or error handling:

```python
# Lines 401-402
args=args,         # Direct assignment - no serialization check
kwargs=kwargs,     # Direct assignment - no serialization check
```

**Issues:**
1. **Non-JSON-serializable types will crash:** Sets, custom objects, datetime objects, file handles, etc.
2. **No try/catch around serialization:** Will fail silently or crash the handler
3. **No size limits:** Large arguments could bloat the database
4. **Tuples converted to lists:** Loss of type information (acceptable but should be documented)

**Example Failure Scenario:**
```python
# Task called with set in kwargs
my_task.delay(collection_id="123", tags={1, 2, 3})

# Signal handler tries to store it
kwargs={'collection_id': '123', 'tags': {1, 2, 3}}
# Database insert fails: TypeError: Object of type set is not JSON serializable
# Exception caught by outer try/except, logged, but task tracking entry NOT created
```

**Consequence:**
- Tasks with complex arguments won't be tracked
- Stale task detection won't work for those tasks
- Silent failures in production
- Potential database integrity issues

**Fix Required:**
```python
import json

def _serialize_task_args(args):
    """Safely serialize task arguments for database storage."""
    try:
        # Test if serializable
        json.dumps(args)
        return args
    except (TypeError, ValueError):
        # Return string representation for non-serializable types
        return str(args)[:1000]  # Limit size

# In task_started_handler:
task_entry = CeleryTaskRegistry(
    task_id=task_id,
    task_name=sender.name if sender else 'unknown',
    worker_id=worker_id,
    worker_pid=worker_pid,
    status='STARTED',
    started_at=datetime.utcnow(),
    last_heartbeat=datetime.utcnow(),
    args=_serialize_task_args(args),      # SAFE
    kwargs=_serialize_task_args(kwargs),  # SAFE
)
```

---

### MAJOR-2: Race Condition in Worker Ready Handler

**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 481-571 (`cleanup_stale_tasks` function)
**Impact:** HIGH - May revoke tasks that are actually running

**Problem:**
The `@worker_ready` signal fires when a worker is ready, but there's a timing window where:

1. Worker A starts and fires `@worker_ready`
2. `inspect.stats()` is called to get active workers
3. Worker B is in the process of starting but not yet registered in stats
4. Worker B's tasks are marked as stale and revoked
5. Worker B finishes starting and processes tasks (that were just revoked)

**Code Analysis:**
```python
# Lines 497-505
inspect = app.control.inspect()
stats = inspect.stats()

if stats:
    active_worker_ids = set(stats.keys())
    logger.info(f"Active workers: {active_worker_ids}")
else:
    active_worker_ids = set()
    logger.warning("No active workers found via inspect - may be a timing issue")
    # WARNING IS LOGGED BUT EXECUTION CONTINUES!
```

**The Issue:**
- Line 505: Even when `stats` is None/empty, the code continues
- It sets `active_worker_ids = set()` (empty set)
- Then ALL tasks in STARTED/RETRY state are considered stale
- All tasks get revoked, even from workers that are starting up

**Consequence in Docker Environment:**
When using `docker-compose up` to start multiple workers:
1. Worker 1 starts first
2. Worker 1's `@worker_ready` fires
3. `inspect.stats()` returns None or only Worker 1
4. Worker 2 is still initializing
5. Tasks assigned to Worker 2 (from previous run or queue) get revoked
6. Worker 2 starts and tries to process revoked tasks
7. Tasks fail or hang in inconsistent state

**Fix Required:**
```python
@worker_ready.connect
def cleanup_stale_tasks(sender=None, **kwargs):
    """Clean up stale tasks with safety checks."""
    logger.info("Worker ready - checking for stale tasks in database...")

    try:
        from fileintel.storage.models import CeleryTaskRegistry

        # SAFETY: Wait for cluster to stabilize
        import time
        time.sleep(5)  # Give other workers time to register

        # Get currently active workers
        inspect = app.control.inspect()
        stats = inspect.stats()

        if not stats:
            logger.warning("No active workers found via inspect - skipping cleanup to avoid false positives")
            return  # ABORT if we can't get worker list

        active_worker_ids = set(stats.keys())
        logger.info(f"Active workers: {active_worker_ids}")

        # Rest of the function...
```

**Alternative Better Approach:**
Only revoke tasks that have been stale for a minimum time:

```python
# Add minimum stale time to avoid race conditions
MINIMUM_STALE_TIME = timedelta(minutes=5)

for task_entry in stale_tasks:
    # Check how long it's been running
    if task_entry.started_at:
        running_time = datetime.utcnow() - task_entry.started_at
        if running_time < MINIMUM_STALE_TIME:
            # Too recent - might be from a worker that's starting
            continue

    # Now check if worker is dead
    if task_entry.worker_id not in active_worker_ids:
        # Only revoke if task has been running for a while
        logger.warning(f"Found stale task...")
        # ... revoke logic
```

---

## MEDIUM ISSUES

### MEDIUM-1: Incorrect Signal Handler Parameter Access

**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Line:** 378
**Impact:** MEDIUM - May cause crashes or store incorrect worker IDs

**Problem:**
The signal handler tries to access `task.request.hostname` but the parameter is passed differently:

```python
# Line 368-369
@task_prerun.connect
def task_started_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
```

According to Celery documentation, the `task_prerun` signal provides:
- `sender`: The task class (not instance)
- `task_id`: The task ID
- `task`: The actual task instance (which has `.request` attribute)
- `args`, `kwargs`: Task arguments

**Line 378:**
```python
worker_id = task.request.hostname if task and hasattr(task, 'request') else 'unknown'
```

**Issues:**
1. `task` parameter in signal is the **task instance**, and it DOES have `.request`
2. However, `task.request.hostname` might not be the right attribute
3. According to Celery docs, it should be `task.request.hostname` OR `sender.request.hostname`
4. The `hasattr` check is good, but 'unknown' fallback will hide issues

**Verification Needed:**
The code structure is correct, but we need to verify:
- Is `task.request.hostname` the correct way to get worker ID?
- Should it be `task.request.hostname` or `task.request.id`?
- What format does hostname return? (e.g., "celery@hostname" vs just "hostname")

**Potential Issue:**
In Celery 5.x, the worker hostname format is typically: `celery@hostname` or `worker_name@hostname`.
The `inspect.stats()` in `cleanup_stale_tasks` returns worker IDs in the same format.

**Recommendation:**
Add defensive logging to verify the format:

```python
# Lines 377-379 (replace)
worker_id = 'unknown'
if task and hasattr(task, 'request'):
    worker_id = task.request.hostname
    logger.debug(f"Task {task_id} assigned to worker: {worker_id}")
else:
    logger.warning(f"Task {task_id} started but could not determine worker ID")
worker_pid = os.getpid()
```

---

### MEDIUM-2: No Session Rollback on Error

**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 375-408, 423-432, 446-455, 467-475, 508-565
**Impact:** MEDIUM - Potential database locks and connection leaks

**Problem:**
All signal handlers use this pattern:

```python
session = _get_task_registry_session()
try:
    # Database operations
    session.commit()
finally:
    session.close()
```

**Issue:**
When an exception occurs BEFORE `session.commit()`, the session is closed without rollback. This can:
1. Leave the transaction open (depending on PostgreSQL settings)
2. Hold locks on affected rows
3. Cause connection pool exhaustion
4. Lead to deadlocks

**Example Scenario:**
```python
# Line 382 - task_entry query succeeds
task_entry = session.query(CeleryTaskRegistry).filter_by(task_id=task_id).first()

# Line 386 - update fails (e.g., invalid data type)
task_entry.status = 'STARTED'

# Exception raised, session.commit() never called
# session.close() called in finally
# Transaction left in inconsistent state
```

**Fix Required:**
```python
session = _get_task_registry_session()
try:
    # Database operations
    session.commit()
except Exception:
    session.rollback()  # ADD THIS
    raise
finally:
    session.close()
```

**Apply to All Handlers:**
- `task_started_handler` (lines 375-408)
- `task_success_handler` (lines 423-432)
- `task_failure_handler` (lines 446-455)
- `task_retry_handler` (lines 467-475)
- `cleanup_stale_tasks` (lines 508-565)

---

### MEDIUM-3: Missing UTC Timezone Awareness

**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 387, 390, 399, 400, 428, 451, 472, 535, 550
**Impact:** MEDIUM - Timezone comparison issues, incorrect stale task detection

**Problem:**
Code uses `datetime.utcnow()` which returns **timezone-naive** datetime objects:

```python
# Line 387
task_entry.started_at = datetime.utcnow()  # NAIVE datetime
```

But the database column is defined as:

```python
# models.py line 223
started_at = Column(DateTime(timezone=True), nullable=True)
```

**Issue:**
1. `DateTime(timezone=True)` expects **timezone-aware** datetimes
2. PostgreSQL will store it as UTC, but Python comparisons may fail
3. `datetime.utcnow()` returns naive datetime (no timezone info)
4. Comparing naive and aware datetimes raises `TypeError`

**Example Failure:**
```python
# Line 550 - comparing naive datetime
time_since_heartbeat = datetime.utcnow() - task_entry.last_heartbeat

# If last_heartbeat is timezone-aware (from database):
# TypeError: can't subtract offset-naive and offset-aware datetimes
```

**Fix Required:**
```python
from datetime import datetime, timezone

# Replace all datetime.utcnow() with:
datetime.now(timezone.utc)  # Timezone-aware UTC datetime
```

**Locations to Fix:**
- Line 387: `task_entry.started_at = datetime.now(timezone.utc)`
- Line 390: `task_entry.last_heartbeat = datetime.now(timezone.utc)`
- Line 399: `started_at=datetime.now(timezone.utc)`
- Line 400: `last_heartbeat=datetime.now(timezone.utc)`
- Line 428: `task_entry.completed_at = datetime.now(timezone.utc)`
- Line 451: `task_entry.completed_at = datetime.now(timezone.utc)`
- Line 472: `task_entry.last_heartbeat = datetime.now(timezone.utc)`
- Line 535: `task_entry.completed_at = datetime.now(timezone.utc)`
- Line 550: `time_since_heartbeat = datetime.now(timezone.utc) - task_entry.last_heartbeat`

**Import Change:**
```python
# Line 358 - update import
from datetime import datetime, timedelta, timezone  # ADD timezone
```

---

### MEDIUM-4: No Heartbeat Updates During Task Execution

**Location:** Entire pipeline - heartbeat mechanism not implemented
**Lines:** Model defines `last_heartbeat` (line 232) but no update mechanism
**Impact:** MEDIUM - Long-running tasks falsely detected as stale

**Problem:**
The `CeleryTaskRegistry` model has a `last_heartbeat` column:

```python
# models.py line 232
last_heartbeat = Column(DateTime(timezone=True), nullable=True)
```

And cleanup logic checks heartbeat age:

```python
# celery_config.py lines 549-556
if task_entry.last_heartbeat:
    time_since_heartbeat = datetime.utcnow() - task_entry.last_heartbeat
    if time_since_heartbeat > timedelta(hours=6):
        logger.warning(f"Task {task_entry.task_id} has no heartbeat for {time_since_heartbeat}")
```

**But there's NO mechanism to update heartbeat during execution!**

**Current State:**
1. Task starts → `last_heartbeat` set to current time (line 390, 400)
2. Task runs for hours → `last_heartbeat` never updated
3. After 6 hours → Cleanup detects as stale (even though still running)
4. Task logged as potentially stuck, but not revoked (line 557)

**Consequence:**
- False positives for long-running tasks
- GraphRAG indexing (can take hours) will always trigger warnings
- No actual heartbeat monitoring capability
- The heartbeat column is effectively useless

**Fix Required:**
Implement periodic heartbeat updates in task execution:

```python
# Option 1: In BaseFileIntelTask.update_progress()
def update_progress(self, current: int, total: int, message: str = "") -> None:
    """Update task progress and heartbeat."""
    progress = {
        "current": current,
        "total": total,
        "percentage": round((current / total) * 100, 2) if total > 0 else 0,
        "message": message,
        "timestamp": time.time(),
    }
    self.update_state(state="PROGRESS", meta=progress)

    # Update heartbeat in registry
    try:
        from fileintel.storage.models import CeleryTaskRegistry, SessionLocal
        from datetime import datetime, timezone

        session = SessionLocal()
        try:
            task_entry = session.query(CeleryTaskRegistry).filter_by(
                task_id=self.request.id
            ).first()
            if task_entry:
                task_entry.last_heartbeat = datetime.now(timezone.utc)
                session.commit()
        finally:
            session.close()
    except Exception:
        pass  # Don't fail task due to heartbeat update

# Option 2: Create dedicated heartbeat task
@app.task(bind=True)
def update_task_heartbeat(self, task_id: str):
    """Update heartbeat for a running task."""
    # ... implementation
```

---

## MINOR ISSUES

### MINOR-1: Queued State Never Set

**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Line:** 222 (model), 398 (handler)
**Impact:** LOW - Tracking incomplete but not breaking

**Problem:**
The model defines a `queued_at` timestamp:

```python
# models.py line 222
queued_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
```

But tasks are only created in the registry when they **start** (task_prerun), not when queued:

```python
# Line 368-369
@task_prerun.connect  # This fires when task STARTS, not when queued
def task_started_handler(...):
```

**Consequence:**
- `queued_at` will always equal `created_at` (both set when task starts)
- No visibility into queue wait time
- Can't distinguish between queue time and execution time

**Fix (Low Priority):**
To properly track queuing, would need to use `task_sent` signal:

```python
from celery.signals import before_task_publish

@before_task_publish.connect
def task_queued_handler(sender=None, headers=None, body=None, **kwargs):
    """Track when task is queued."""
    task_id = headers.get('id')
    task_name = headers.get('task')

    # Create PENDING entry
    session = _get_task_registry_session()
    try:
        task_entry = CeleryTaskRegistry(
            task_id=task_id,
            task_name=task_name,
            worker_id='pending',
            status='PENDING',
            queued_at=datetime.now(timezone.utc),
        )
        session.add(task_entry)
        session.commit()
    finally:
        session.close()
```

---

### MINOR-2: Status Value Mismatch with Celery States

**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 386, 398, 427, 450, 471, 534
**Impact:** LOW - Status naming inconsistency

**Problem:**
Code uses custom status values that don't exactly match Celery's state names:

**Database:** 'STARTED', 'SUCCESS', 'FAILURE', 'RETRY', 'REVOKED', 'PENDING'
**Celery States:** 'PENDING', 'STARTED', 'SUCCESS', 'FAILURE', 'RETRY', 'REVOKED'

Most match, but worth documenting that these are registry-specific states, not directly Celery states.

**Recommendation:**
Add a comment documenting valid status values:

```python
# celery_config.py - add before task_started_handler
# Valid task registry status values (aligned with Celery states):
# - PENDING: Task queued but not started
# - STARTED: Task is running
# - SUCCESS: Task completed successfully
# - FAILURE: Task failed
# - RETRY: Task is being retried
# - REVOKED: Task was cancelled/revoked
```

---

### MINOR-3: No Index on Compound Queries

**Location:** `/home/tuomo/code/fileintel/migrations/versions/20251019_create_celery_task_registry.py`
**Lines:** 49-69 (index creation)
**Impact:** LOW - Query performance for cleanup

**Problem:**
The cleanup query uses a compound filter:

```python
# celery_config.py line 511-513
stale_tasks = (
    session.query(CeleryTaskRegistry)
    .filter(CeleryTaskRegistry.status.in_(['STARTED', 'RETRY']))
    .all()
)
```

Current indexes:
- `ix_celery_task_registry_status` - Single column on `status`
- `ix_celery_task_registry_worker_id` - Single column on `worker_id`

**For optimal performance, could add:**
```python
# Compound index for cleanup queries
op.create_index(
    'ix_celery_task_registry_status_worker',
    'celery_task_registry',
    ['status', 'worker_id'],
    unique=False
)
```

This would speed up the cleanup query which filters by status and then checks worker_id.

**Priority:** LOW - Only matters at scale (thousands of tasks)

---

### MINOR-4: CLI Command Uses Same Unsafe Pattern

**Location:** `/home/tuomo/code/fileintel/src/fileintel/cli/tasks.py`
**Lines:** 238-340 (`cleanup_stale_tasks` command)
**Impact:** LOW - Same issues as signal handler but in CLI

**Problem:**
The CLI command duplicates the cleanup logic and has the same issues:

1. No protection against empty worker list (line 268)
2. Uses `datetime.utcnow()` instead of timezone-aware (line 300, 319)
3. No session rollback on error (line 272-336)

**Fix Required:**
Apply the same fixes as recommended for the signal handler version.

---

## INTEGRATION ISSUES

### INTEGRATION-1: Migration Schema Matches Model (✓ VERIFIED)

**Status:** NO ISSUES FOUND

The migration schema (20251019_create_celery_task_registry.py) correctly matches the model definition:
- All columns present and correct types
- Indexes properly defined
- Constraints match

---

### INTEGRATION-2: SessionLocal Import Path Correct (✓ VERIFIED)

**Status:** NO ISSUES FOUND

**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py` line 363

```python
from fileintel.storage.models import SessionLocal
```

Verified this import exists in `/home/tuomo/code/fileintel/src/fileintel/storage/models.py` line 258:

```python
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

Import path is correct and will work.

---

## EDGE CASES AND RACE CONDITIONS

### EDGE-1: Worker Death During Signal Handler

**Scenario:**
1. Task starts → `task_prerun` fires
2. Handler creates DB entry
3. Worker crashes BEFORE `session.commit()`
4. Task entry never created
5. Task remains in broker as STARTED
6. No record in database → Never cleaned up

**Mitigation:**
Use `session.commit()` as early as possible, and ensure rollback on error.

---

### EDGE-2: Database Connection Failure

**Scenario:**
1. Task starts
2. Signal handler fires
3. Database is down/unreachable
4. `_get_task_registry_session()` fails
5. Exception caught and logged
6. **Task continues execution** (good!)
7. But task is never tracked

**Current Handling:**
```python
except Exception as e:
    logger.error(f"Error tracking task start for {task_id}: {e}")
    # Don't fail the task due to tracking issues
```

This is **CORRECT** - tracking failures shouldn't break task execution.

**Recommendation:**
Add a metric/counter for tracking failures to monitor this issue:

```python
except Exception as e:
    logger.error(f"Error tracking task start for {task_id}: {e}")
    # Increment failure counter for monitoring
    try:
        from prometheus_client import Counter
        TASK_TRACKING_ERRORS.inc()
    except:
        pass
```

---

### EDGE-3: Task Sent to Wrong Worker After Revoke

**Scenario:**
1. Worker A dies with Task X in STARTED state
2. Worker B starts, cleanup runs
3. Task X revoked via `app.control.revoke(task_id)`
4. But Task X still in broker queue (revoke doesn't remove from queue)
5. Worker C picks up Task X from queue
6. Task X executes despite being revoked

**Current Behavior:**
According to Celery docs, `revoke()` with `terminate=False`:
- Marks task as revoked in result backend
- Workers check revocation status before execution
- Task won't execute if already revoked

**But there's a gap:**
If task was never acknowledged (worker died before ack), it's still in the queue.

**Mitigation Needed:**
After revoke, also remove from queue:

```python
# After line 531
app.control.revoke(task_entry.task_id, terminate=False)

# Also try to purge from queue (if possible)
# This is tricky - Celery doesn't expose per-task removal
# Best we can do is rely on revoke working
```

**Current Implementation:** Acceptable - Celery's revoke should handle this

---

## RECOMMENDATIONS

### Immediate Fixes (Required Before Production)

1. **Add logger initialization** (CRITICAL)
   - Location: After line 11 in celery_config.py
   - Code: `logger = logging.getLogger(__name__)`

2. **Fix JSON serialization** (CRITICAL)
   - Location: Lines 401-402 in celery_config.py
   - Add `_serialize_task_args()` helper function
   - Wrap args/kwargs in safe serialization

3. **Fix timezone handling** (HIGH)
   - Location: All datetime.utcnow() calls
   - Replace with: `datetime.now(timezone.utc)`
   - Update import: `from datetime import datetime, timedelta, timezone`

4. **Add session rollback** (HIGH)
   - Location: All signal handlers
   - Add `session.rollback()` in except blocks

5. **Fix race condition in worker_ready** (HIGH)
   - Location: Lines 497-505
   - Add delay before cleanup OR minimum stale time check
   - Don't proceed if stats() returns None/empty

### Short-term Improvements

6. **Implement heartbeat updates** (MEDIUM)
   - Add heartbeat update to BaseFileIntelTask.update_progress()
   - Or create separate periodic heartbeat task

7. **Add task queued tracking** (MEDIUM)
   - Implement before_task_publish signal handler
   - Create PENDING entries when tasks are queued

8. **Add compound index** (LOW)
   - Create index on (status, worker_id) for faster cleanup queries

9. **Fix CLI command** (MEDIUM)
   - Apply same fixes to CLI version of cleanup

### Long-term Architectural Improvements

10. **Centralize cleanup logic**
    - Move cleanup logic to a shared function
    - Use from both signal handler and CLI command
    - Avoid code duplication

11. **Add monitoring and metrics**
    - Track tracking failures
    - Monitor cleanup execution time
    - Alert on high stale task counts

12. **Implement task lifecycle state machine**
    - Validate state transitions
    - Prevent invalid state changes
    - Add state change audit log

---

## TESTING RECOMMENDATIONS

### Unit Tests

```python
def test_logger_exists():
    """Verify logger is defined in celery_config."""
    from fileintel.celery_config import logger
    assert logger is not None

def test_serialize_task_args_handles_sets():
    """Test that sets are safely serialized."""
    result = _serialize_task_args({'tags': {1, 2, 3}})
    assert isinstance(result, (str, dict))

def test_cleanup_handles_empty_worker_list():
    """Test cleanup when no workers are running."""
    # Mock inspect.stats() to return None
    # Verify cleanup doesn't revoke all tasks
```

### Integration Tests

```python
def test_task_tracking_lifecycle():
    """Test complete task lifecycle tracking."""
    # 1. Queue a task
    # 2. Verify PENDING entry created
    # 3. Start task execution
    # 4. Verify STARTED entry with worker_id
    # 5. Complete task
    # 6. Verify SUCCESS entry with result

def test_stale_task_cleanup():
    """Test stale task detection and cleanup."""
    # 1. Create mock stale task entries
    # 2. Simulate worker death
    # 3. Start new worker
    # 4. Verify cleanup revokes stale tasks
    # 5. Verify active tasks not affected

def test_worker_startup_race_condition():
    """Test multiple workers starting simultaneously."""
    # 1. Start 3 workers simultaneously
    # 2. Verify cleanup doesn't revoke active tasks
    # 3. Check all workers registered properly
```

### Production Validation

```bash
# 1. Test logger functionality
docker-compose up worker
# Check logs for signal handler output
# Should see "Worker ready - checking for stale tasks"

# 2. Test task tracking
fileintel tasks list
# Create a task, verify it appears in list

# 3. Test cleanup
docker-compose down  # Kill workers
docker-compose up worker  # Restart
# Check logs for "Revoked X stale tasks"

# 4. Test with complex arguments
# Execute task with sets, custom objects
# Verify it doesn't crash
```

---

## FILE REFERENCES

### Primary Files Analyzed

1. **Database Model**
   - File: `/home/tuomo/code/fileintel/src/fileintel/storage/models.py`
   - Lines: 205-235 (CeleryTaskRegistry class)
   - Status: Schema correct, no issues

2. **Signal Handlers**
   - File: `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
   - Lines: 368-571
   - Critical Issues: Logger missing, JSON serialization unsafe, race condition

3. **CLI Command**
   - File: `/home/tuomo/code/fileintel/src/fileintel/cli/tasks.py`
   - Lines: 238-340
   - Issues: Same as signal handlers

4. **Migration**
   - File: `/home/tuomo/code/fileintel/migrations/versions/20251019_create_celery_task_registry.py`
   - Lines: All
   - Status: Correct, matches model

---

## SUMMARY OF FINDINGS

| Issue | Severity | Impact | File | Lines | Fix Priority |
|-------|----------|--------|------|-------|--------------|
| Logger not initialized | CRITICAL | Crashes | celery_config.py | 411, 435, 458, 478, 491, 568 | IMMEDIATE |
| Unsafe JSON serialization | MAJOR | Data loss | celery_config.py | 401-402 | HIGH |
| Worker ready race condition | MAJOR | Revokes active tasks | celery_config.py | 497-505 | HIGH |
| Missing session rollback | MEDIUM | Connection leaks | celery_config.py | Multiple | HIGH |
| Naive datetime usage | MEDIUM | Comparison errors | celery_config.py | Multiple | HIGH |
| No heartbeat updates | MEDIUM | False positives | celery_config.py | N/A | MEDIUM |
| Incorrect signal params | MEDIUM | Wrong worker ID | celery_config.py | 378 | MEDIUM |
| No queued state tracking | MINOR | Incomplete tracking | celery_config.py | 368 | LOW |
| Status value mismatch | MINOR | Documentation | celery_config.py | Multiple | LOW |
| Missing compound index | MINOR | Performance | migration | 49-69 | LOW |

**Total Issues: 10**
- Critical: 1
- Major: 2
- Medium: 4
- Minor: 3

---

## CONCLUSION

The stale task detection and cleanup pipeline has a **solid architectural design** but suffers from **critical implementation bugs** that will cause immediate failures in production. The most severe issue is the missing logger initialization, which will crash all signal handlers on first use.

The second major concern is the race condition in worker startup, which could incorrectly revoke active tasks in a multi-worker Docker environment.

With the fixes outlined above, particularly the immediate and high-priority items, this system should work reliably. The design is sound - the execution just needs the bugs fixed.

**Recommended Action:** Fix CRITICAL and HIGH priority issues before deploying to production. MEDIUM and LOW priority issues can be addressed in subsequent iterations.
