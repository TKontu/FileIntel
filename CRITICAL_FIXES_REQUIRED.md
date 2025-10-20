# CRITICAL FIXES REQUIRED - Stale Task Pipeline

## STOP - READ THIS FIRST

The stale task detection pipeline has **critical bugs** that will cause **immediate production failures**. Do not deploy without applying these fixes.

---

## CRITICAL FIX 1: Logger Not Initialized (CRASHES ON FIRST USE)

**File:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Line:** After line 11

**Problem:** All signal handlers reference `logger` but it's never created. First task execution will crash with:
```
NameError: name 'logger' is not defined
```

**Fix:**
```python
# Line 11
import logging

logger = logging.getLogger(__name__)  # ADD THIS LINE

# Rest of file...
```

**Impact if not fixed:** Complete pipeline failure - no tasks will be tracked, workers may fail to start

---

## CRITICAL FIX 2: Unsafe JSON Serialization (DATA LOSS)

**File:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 401-402

**Problem:** Task args/kwargs stored directly in JSONB without serialization checks. Tasks with sets, custom objects, or other non-JSON types will fail to be tracked.

**Fix:**
```python
# Add this function before task_started_handler (around line 367)
def _serialize_task_args(data):
    """Safely serialize task arguments for JSONB storage."""
    import json
    try:
        # Test if directly serializable
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        # Convert to string representation for non-serializable types
        try:
            return str(data)[:1000]  # Limit size to prevent bloat
        except Exception:
            return "<unserializable>"

# Then update lines 401-402 and 429:
args=_serialize_task_args(args),
kwargs=_serialize_task_args(kwargs),

# And line 429:
task_entry.result = _serialize_task_args({'success': True, 'result': str(result)[:1000]})
```

**Impact if not fixed:** Tasks with complex arguments silently not tracked, stale detection won't work for them

---

## CRITICAL FIX 3: Race Condition in Worker Startup (REVOKES ACTIVE TASKS)

**File:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 497-505

**Problem:** When `inspect.stats()` returns None/empty (during multi-worker startup), cleanup proceeds anyway and revokes ALL tasks.

**Fix:**
```python
# Replace lines 497-505 with:
inspect = app.control.inspect()
stats = inspect.stats()

if not stats:
    logger.warning(
        "No active workers found via inspect - skipping cleanup to avoid "
        "revoking tasks from workers that are still starting up"
    )
    return  # ABORT cleanup if we can't get worker list

active_worker_ids = set(stats.keys())
logger.info(f"Active workers: {active_worker_ids}")
```

**Additional Safety - Add minimum stale time check (line 521):**
```python
# Add after line 520 (for task_entry in stale_tasks:)
from datetime import timedelta

MINIMUM_STALE_TIME = timedelta(minutes=5)

for task_entry in stale_tasks:
    # Skip tasks that started very recently (might be from starting workers)
    if task_entry.started_at:
        running_time = datetime.utcnow() - task_entry.started_at
        if running_time < MINIMUM_STALE_TIME:
            continue  # Too recent to be stale

    # Check if worker is still alive
    if task_entry.worker_id not in active_worker_ids:
        # ... existing revoke logic
```

**Impact if not fixed:** During `docker-compose up`, active tasks from newly-starting workers will be revoked

---

## HIGH PRIORITY FIX 4: Timezone-Naive Datetime Usage (COMPARISON ERRORS)

**File:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 358, 387, 390, 399, 400, 428, 451, 472, 535, 550

**Problem:** Code uses `datetime.utcnow()` which returns timezone-naive datetime. Database columns are `DateTime(timezone=True)`, causing potential comparison errors.

**Fix:**
```python
# Line 358 - Update import
from datetime import datetime, timedelta, timezone  # Add timezone

# Replace ALL instances of datetime.utcnow() with:
datetime.now(timezone.utc)

# Specific lines to change:
# Line 387: task_entry.started_at = datetime.now(timezone.utc)
# Line 390: task_entry.last_heartbeat = datetime.now(timezone.utc)
# Line 399: started_at=datetime.now(timezone.utc),
# Line 400: last_heartbeat=datetime.now(timezone.utc),
# Line 428: task_entry.completed_at = datetime.now(timezone.utc)
# Line 451: task_entry.completed_at = datetime.now(timezone.utc)
# Line 472: task_entry.last_heartbeat = datetime.now(timezone.utc)
# Line 535: task_entry.completed_at = datetime.now(timezone.utc)
# Line 550: time_since_heartbeat = datetime.now(timezone.utc) - task_entry.last_heartbeat
```

**Impact if not fixed:** Timezone comparison errors, incorrect stale task age calculations

---

## HIGH PRIORITY FIX 5: Missing Session Rollback (CONNECTION LEAKS)

**File:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
**Lines:** 375-408, 423-432, 446-455, 467-475, 508-565

**Problem:** No rollback on database errors - can cause connection pool exhaustion and locks.

**Fix Pattern (apply to ALL signal handlers):**
```python
session = _get_task_registry_session()
try:
    # Database operations
    session.commit()
except Exception as e:
    session.rollback()  # ADD THIS
    logger.error(f"Error: {e}")
    raise  # or just log, depending on handler
finally:
    session.close()
```

**Apply to these functions:**
1. `task_started_handler` (lines 375-408)
2. `task_success_handler` (lines 423-432)
3. `task_failure_handler` (lines 446-455)
4. `task_retry_handler` (lines 467-475)
5. `cleanup_stale_tasks` (lines 508-565)

**Impact if not fixed:** Database connection leaks, potential deadlocks under heavy load

---

## VERIFICATION STEPS

After applying fixes:

### 1. Test Logger
```bash
# Start worker and check for crashes
docker-compose up worker

# Look for this in logs (no NameError):
# "Worker ready - checking for stale tasks in database..."
```

### 2. Test Serialization
```bash
# Run a task with complex arguments
fileintel collections process <collection> --build-graph

# Check database:
docker-compose exec postgres psql -U user -d fileintel -c \
  "SELECT task_id, args, kwargs FROM celery_task_registry LIMIT 5;"

# Should see JSON data, not errors
```

### 3. Test Race Condition
```bash
# Start multiple workers simultaneously
docker-compose up --scale worker=3

# Check logs - should NOT see:
# "No active workers found"
# OR if you do, cleanup should abort

# Should see:
# "Active workers: {celery@worker1, celery@worker2, celery@worker3}"
```

### 4. Test Timezone Handling
```bash
# Check database timestamps
docker-compose exec postgres psql -U user -d fileintel -c \
  "SELECT task_id, started_at, completed_at FROM celery_task_registry LIMIT 5;"

# All timestamps should have timezone info (+00)
```

### 5. Test Session Management
```bash
# Monitor database connections during heavy load
docker-compose exec postgres psql -U user -d fileintel -c \
  "SELECT COUNT(*) FROM pg_stat_activity WHERE datname='fileintel';"

# Connection count should remain stable, not grow continuously
```

---

## DEPLOYMENT CHECKLIST

- [ ] Fix 1: Logger initialized (add after line 11)
- [ ] Fix 2: JSON serialization helper added (before line 367)
- [ ] Fix 2: All args/kwargs use serialization helper (lines 401-402, 429)
- [ ] Fix 3: Worker ready handler aborts if no workers found (lines 497-505)
- [ ] Fix 3: Minimum stale time check added (after line 520)
- [ ] Fix 4: Import timezone added (line 358)
- [ ] Fix 4: All datetime.utcnow() replaced with datetime.now(timezone.utc)
- [ ] Fix 5: Session rollback added to all 5 handlers
- [ ] All verification steps passed
- [ ] Integration test run (start worker, execute task, kill worker, restart, verify cleanup)

---

## ESTIMATED TIME TO FIX

- Fix 1 (Logger): 2 minutes
- Fix 2 (Serialization): 15 minutes
- Fix 3 (Race condition): 10 minutes
- Fix 4 (Timezone): 10 minutes
- Fix 5 (Session rollback): 15 minutes
- Testing: 30 minutes

**Total: ~90 minutes**

---

## RELATED FILES

See `/home/tuomo/code/fileintel/stale_task_pipeline_analysis.md` for complete analysis with medium and low priority issues.
