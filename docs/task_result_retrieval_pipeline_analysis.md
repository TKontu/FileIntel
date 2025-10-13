# Task Result Retrieval and Wait/Monitoring Pipeline Analysis

**Date:** 2025-10-13
**System:** FileIntel Task Result Retrieval & Monitoring
**Status:** CRITICAL ISSUES FOUND

## Executive Summary

**Critical Issues Found:** 12
**High Priority Issues:** 8
**Medium Priority Issues:** 4

### Most Critical Problems:
1. **Result retrieval returns incorrect data structure** - API endpoint returns task status instead of task result
2. **Field name mismatches** between API response and CLI expectations (ready/successful vs status)
3. **Timeout parameter ignored** - CLI accepts timeout but doesn't use it
4. **Missing result serialization validation** - Large or non-serializable results will fail
5. **No result expiration handling** - Results expire after 1 hour but no user notification
6. **Progress information extraction fragile** - Assumes specific result structure without validation

---

## Pipeline Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLI ENTRY POINTS                                   │
│  src/fileintel/cli/tasks.py                                                 │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │ result cmd   │  │  wait cmd    │  │   get cmd    │                     │
│  │ (line 120)   │  │ (line 156)   │  │ (line 74)    │                     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                     │
│         │                  │                  │                              │
└─────────┼──────────────────┼──────────────────┼──────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLIENT API LAYER                                      │
│  src/fileintel/cli/task_client.py (TaskAPIClient)                          │
│                                                                              │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌─────────────────┐       │
│  │ get_task_result  │  │ wait_for_task_      │  │ get_task_status │       │
│  │    (line 172)    │  │   completion        │  │   (line 168)    │       │
│  │                  │  │    (line 229)       │  │                 │       │
│  └────────┬─────────┘  └──────────┬──────────┘  └────────┬────────┘       │
│           │                       │                       │                 │
└───────────┼───────────────────────┼───────────────────────┼─────────────────┘
            │                       │                       │
            │ GET /tasks/{id}/result│ GET /tasks/{id}/status│
            ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API ENDPOINTS                                       │
│  src/fileintel/api/routes/tasks_v2.py                                       │
│                                                                              │
│  ┌────────────────────┐  ┌──────────────────────┐                          │
│  │ get_task_result    │  │ get_task_status_     │                          │
│  │   (line 242)       │  │    endpoint          │                          │
│  │                    │  │   (line 111)         │                          │
│  └─────────┬──────────┘  └──────────┬───────────┘                          │
│            │                        │                                       │
└────────────┼────────────────────────┼───────────────────────────────────────┘
             │                        │
             ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CELERY BACKEND INTEGRATION                              │
│  src/fileintel/celery_config.py                                             │
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │ get_task_status  │  ← Returns: {task_id, state, result, traceback,      │
│  │   (line 232)     │             info}                                     │
│  └────────┬─────────┘                                                       │
│           │                                                                  │
└───────────┼──────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CELERY RESULT BACKEND                                 │
│  Redis/Database Backend (configured in celery_config)                       │
│                                                                              │
│  - Result Expiration: 3600 seconds (1 hour)                                │
│  - Result Serialization: gzip compression                                   │
│  - Result Persistence: True                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROGRESS REPORTING FLOW                                 │
│                                                                              │
│  Task Execution → update_progress() → Celery State Update                  │
│  (base.py:65)      (PROGRESS meta)     (Redis/Backend)                     │
│                                             │                                │
│                                             ▼                                │
│                            API: get_task_status → CLI: wait_for_task_       │
│                            (extracts progress)     completion               │
│                                                    (displays progress bar)   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     WEBSOCKET INTEGRATION (OPTIONAL)                         │
│  src/fileintel/api/routes/websocket_v2.py                                   │
│                                                                              │
│  Real-time Updates: Polls Celery active tasks every 2 seconds              │
│  NOT USED BY CLI - CLI uses polling via REST API                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Critical Issues

### ISSUE #1: Result Retrieval Returns Wrong Data Structure
**Severity:** CRITICAL
**Impact:** HIGH - Users cannot retrieve actual task results
**Location:** `/home/tuomo/code/fileintel/src/fileintel/api/routes/tasks_v2.py:242-269`

**Problem:**
The `/tasks/{task_id}/result` endpoint returns a minimal wrapper instead of the actual result data structure that tasks return.

**Code:**
```python
# Line 242-269
@router.get("/tasks/{task_id}/result", response_model=ApiResponseV2)
@celery_error_handler("get task result")
async def get_task_result(task_id: str) -> ApiResponseV2:
    """
    Get the result of a completed task.

    Returns the full result data for successful tasks.
    """
    task_info = get_task_status(task_id)

    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task_info["state"] != "SUCCESS":
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} is not completed successfully (current state: {task_info['state']})",
        )

    result = _format_task_result(task_info.get("result"))

    return create_success_response(
        {
            "task_id": task_id,
            "result": result,
            "completed_at": datetime.utcnow().isoformat(),  # WRONG: should be actual completion time
        }
    )
```

**Issues:**
1. Returns a wrapper `{task_id, result, completed_at}` instead of the actual task result
2. `completed_at` uses `datetime.utcnow()` instead of actual completion timestamp from Celery
3. No validation that result is serializable
4. Missing result metadata (execution time, worker info, etc.)

**Expected Behavior:**
Should return the actual task result structure directly:
```python
{
    "collection_id": "...",
    "workflow_task_id": "...",
    "status": "processing_with_metadata_and_embeddings",
    "message": "..."
}
```

**Recommendation:**
```python
@router.get("/tasks/{task_id}/result", response_model=ApiResponseV2)
@celery_error_handler("get task result")
async def get_task_result(task_id: str) -> ApiResponseV2:
    """Get the result of a completed task."""
    async_result = app.AsyncResult(task_id)

    if not async_result:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if not async_result.ready():
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} is not completed (state: {async_result.state})",
        )

    if async_result.failed():
        raise HTTPException(
            status_code=500,
            detail=f"Task {task_id} failed: {str(async_result.result)}",
        )

    # Return actual result from task
    result = async_result.result

    # Validate result is serializable
    try:
        json.dumps(result)
    except (TypeError, ValueError) as e:
        logger.error(f"Task result not serializable: {e}")
        raise HTTPException(
            status_code=500,
            detail="Task result cannot be serialized"
        )

    return create_success_response(result)
```

---

### ISSUE #2: Field Name Mismatches Between Layers
**Severity:** CRITICAL
**Impact:** HIGH - CLI commands fail due to missing expected fields
**Location:** `/home/tuomo/code/fileintel/src/fileintel/cli/tasks.py:142-153`

**Problem:**
CLI expects `ready` and `successful` fields but API returns `status` field with enum values.

**Code (CLI expectations):**
```python
# Line 142-153
task_result = result_data.get("data", {})

if task_result and task_result.get("ready"):  # ← Field doesn't exist in API response
    if task_result.get("successful"):          # ← Field doesn't exist in API response
        cli_handler.display_success("Task completed successfully")
        if "result" in task_result:
            cli_handler.display_json(task_result["result"], "Task Result")
    else:
        error = task_result.get("error", "Unknown error")
        cli_handler.display_error(f"Task failed: {error}")
else:
    cli_handler.console.print(f"[yellow]Task {task_id} is still running[/yellow]")
```

**Actual API Response Structure:**
```python
# From tasks_v2.py:242-269
{
    "success": True,
    "data": {
        "task_id": "...",
        "result": {...},  # ← This is the wrapped result, not direct result
        "completed_at": "..."
    }
}
```

**Issue:**
- CLI checks `task_result.get("ready")` which will always be falsy
- CLI checks `task_result.get("successful")` which will always be falsy
- This means successful tasks will display "Task is still running" message

**Recommendation:**
Fix CLI to match API response structure:
```python
# Line 142-153 - CORRECTED
task_result = result_data.get("data", {})

if task_result and "result" in task_result:
    # Task has completed and has a result
    cli_handler.display_success("Task completed successfully")
    cli_handler.display_json(task_result["result"], "Task Result")
elif result_data.get("success") == False:
    # API returned error
    error = result_data.get("error", "Unknown error")
    cli_handler.display_error(f"Task failed: {error}")
else:
    # Task is still running or result not available
    cli_handler.console.print(f"[yellow]Task {task_id} is still running or result not available[/yellow]")
    cli_handler.console.print("Use 'fileintel tasks wait' to monitor progress")
```

---

### ISSUE #3: Timeout Parameter Ignored
**Severity:** HIGH
**Impact:** MEDIUM - Users cannot control wait timeout
**Location:** `/home/tuomo/code/fileintel/src/fileintel/cli/tasks.py:120-154`

**Problem:**
The `result` command accepts a `--timeout` parameter but never uses it.

**Code:**
```python
# Line 120-127
@app.command("result")
def get_task_result(
    task_id: str = typer.Argument(..., help="The ID of the task to get result for."),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", "-t", help="Timeout in seconds to wait for result."
    ),
):
    """Get the result of a completed task."""

    def _get_result(api):
        # Use the dedicated get_task_result method (timeout not currently supported by API)
        return api.get_task_result(task_id)  # ← timeout parameter not passed
```

**Issue:**
- User specifies timeout but it's silently ignored
- Comment acknowledges timeout is not supported but parameter is still accepted
- No error message or warning to user

**Recommendation:**
Either implement timeout or remove the parameter:

**Option 1: Implement timeout with polling**
```python
@app.command("result")
def get_task_result(
    task_id: str = typer.Argument(..., help="The ID of the task to get result for."),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", "-t", help="Timeout in seconds to wait for result."
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for task to complete before retrieving result."
    ),
):
    """Get the result of a completed task."""

    if wait and timeout:
        # Wait with timeout
        try:
            monitor_task_with_progress(task_id, "Task execution")
        except TimeoutError:
            cli_handler.display_error(f"Task did not complete within {timeout} seconds")
            raise typer.Exit(CLI_ERROR)

    # Then get result
    def _get_result(api):
        return api.get_task_result(task_id)

    result_data = cli_handler.handle_api_call(_get_result, "get task result")
    # ... rest of code
```

**Option 2: Remove timeout parameter**
```python
@app.command("result")
def get_task_result(
    task_id: str = typer.Argument(..., help="The ID of the task to get result for."),
):
    """Get the result of a completed task. Use 'wait' command to wait for completion."""
    # ... rest of code
```

---

### ISSUE #4: Missing Result Serialization Validation
**Severity:** HIGH
**Impact:** HIGH - Large results or non-serializable objects cause silent failures
**Location:** Multiple locations

**Problem:**
No validation that task results can be serialized before returning to clients. Tasks can return complex objects, file handles, or extremely large data structures.

**Affected Locations:**
1. `/home/tuomo/code/fileintel/src/fileintel/api/routes/tasks_v2.py:81-93` - `_format_task_result`
2. Task implementations in `/home/tuomo/code/fileintel/src/fileintel/tasks/workflow_tasks.py`

**Code:**
```python
# Line 81-93 - tasks_v2.py
def _format_task_result(result: Any) -> Optional[Dict[str, Any]]:
    """Format task result for API response."""
    if result is None:
        return None

    if isinstance(result, dict):
        return result  # ← No validation that dict is serializable

    try:
        # Try to convert to dict if possible
        return {"result": str(result)}  # ← Converts to string as fallback
    except (TypeError, ValueError):
        return {"result": "Unable to format result"}
```

**Issues:**
1. No size limit checking - results could be gigabytes
2. No deep serialization validation - dict may contain non-serializable objects
3. Fallback to `str()` may produce useless output
4. No logging of serialization failures

**Example Failure Scenario:**
```python
# Task returns this
{
    "collection_id": "test",
    "storage": <PostgreSQLStorage object>,  # Not serializable!
    "file_handle": <open file>,             # Not serializable!
    "chunks": [large_list_with_10k_items],  # May be too large
}
```

**Recommendation:**
```python
import json
import sys

def _format_task_result(result: Any, max_size_mb: int = 10) -> Optional[Dict[str, Any]]:
    """
    Format task result for API response with validation.

    Args:
        result: Task result to format
        max_size_mb: Maximum result size in MB (default 10MB)

    Returns:
        Formatted and validated result

    Raises:
        ValueError: If result is not serializable or too large
    """
    if result is None:
        return None

    # Try to serialize result
    try:
        serialized = json.dumps(result)
    except (TypeError, ValueError) as e:
        logger.error(f"Task result not JSON serializable: {e}")
        # Try to extract serializable parts
        if isinstance(result, dict):
            cleaned_result = {}
            for key, value in result.items():
                try:
                    json.dumps(value)
                    cleaned_result[key] = value
                except (TypeError, ValueError):
                    cleaned_result[key] = f"<Non-serializable: {type(value).__name__}>"
            return cleaned_result
        else:
            return {"error": "Result not serializable", "type": type(result).__name__}

    # Check size
    size_bytes = sys.getsizeof(serialized)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_size_mb:
        logger.warning(f"Task result too large: {size_mb:.2f}MB (limit: {max_size_mb}MB)")
        return {
            "error": "Result too large for inline return",
            "size_mb": round(size_mb, 2),
            "suggestion": "Result should be stored and returned as reference"
        }

    # Result is valid
    if isinstance(result, dict):
        return result
    else:
        return {"result": result}
```

---

### ISSUE #5: Result Expiration Not Communicated
**Severity:** HIGH
**Impact:** MEDIUM - Users don't know results have expired
**Location:** `/home/tuomo/code/fileintel/src/fileintel/celery_config.py:139`

**Problem:**
Results expire after 1 hour (3600 seconds) but users are never notified when results have expired.

**Code:**
```python
# Line 139 - celery_config.py
result_expires=3600,  # Results expire after 1 hour
```

**Issue:**
When a user tries to retrieve results after expiration:
1. `get_task_status()` returns `PENDING` state (default for missing tasks)
2. No distinction between "task never existed" and "result expired"
3. No user-friendly error message

**Current Behavior:**
```bash
$ fileintel tasks result <old_task_id>
# Returns: Task <id> not found
```

**Recommendation:**
Add result expiration tracking and better error messages:

```python
# In celery_config.py - add metadata tracking
def get_task_status_with_metadata(task_id: str) -> dict:
    """Get task status with expiration metadata."""
    result = app.AsyncResult(task_id)

    status = {
        "task_id": task_id,
        "state": result.state,
        "result": result.result,
        "traceback": result.traceback,
        "info": result.info,
    }

    # Check if task ID format suggests it's old (timestamp-based IDs)
    # Or track task creation time in Redis with longer TTL
    if result.state == "PENDING":
        # Could be new task or expired result
        # Check if task was ever registered (requires separate tracking)
        status["possibly_expired"] = True
        status["expiration_note"] = "Results expire after 1 hour. Task may have expired or never existed."

    return status

# In tasks_v2.py endpoint
@router.get("/tasks/{task_id}/result", response_model=ApiResponseV2)
async def get_task_result(task_id: str) -> ApiResponseV2:
    task_info = get_task_status_with_metadata(task_id)

    if not task_info or task_info["state"] == "PENDING":
        if task_info.get("possibly_expired"):
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found. Results expire after 1 hour. "
                       "Task may have expired or never existed."
            )
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
```

**Better Solution: Store completion timestamp**
```python
# In base.py - track completion time
class BaseFileIntelTask(Task):
    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called when the task succeeds."""
        logger.info(f"Task {self.name}[{task_id}] completed successfully")

        # Store completion metadata with longer TTL
        try:
            from fileintel.celery_config import get_celery_app
            app = get_celery_app()

            # Store in Redis with 7 day TTL
            redis_client = app.backend.client
            metadata_key = f"task_metadata:{task_id}"
            metadata = {
                "completed_at": time.time(),
                "task_name": self.name,
                "result_expiry": time.time() + 3600,  # 1 hour from now
            }
            redis_client.setex(
                metadata_key,
                7 * 24 * 3600,  # 7 days
                json.dumps(metadata)
            )
        except Exception as e:
            logger.warning(f"Failed to store task metadata: {e}")
```

---

### ISSUE #6: Progress Information Extraction Fragile
**Severity:** MEDIUM
**Impact:** MEDIUM - Progress display breaks with different task structures
**Location:** `/home/tuomo/code/fileintel/src/fileintel/api/routes/tasks_v2.py:96-108`

**Problem:**
Progress extraction assumes specific structure without validation.

**Code:**
```python
# Line 96-108
def _extract_progress_info(task_info: Dict[str, Any]) -> Optional[TaskProgressInfo]:
    """Extract progress information from Celery task info."""
    if task_info.get("state") == "PROGRESS" and "result" in task_info:
        progress_data = task_info["result"]
        if isinstance(progress_data, dict):  # ← Only check for dict
            return TaskProgressInfo(
                current=progress_data.get("current", 0),     # ← No validation
                total=progress_data.get("total", 1),         # ← No validation
                percentage=progress_data.get("percentage", 0.0),
                message=progress_data.get("message", ""),
                timestamp=progress_data.get("timestamp", datetime.utcnow().timestamp()),
            )
    return None
```

**Issues:**
1. No validation that `current` and `total` are integers
2. No validation that `percentage` is a float
3. No validation that values are in valid ranges (current <= total, 0 <= percentage <= 100)
4. `timestamp` defaults to current time if missing (misleading)

**Failure Scenario:**
```python
# Task sends malformed progress
self.update_state(state="PROGRESS", meta={
    "current": "50",      # String instead of int
    "total": None,        # None instead of int
    "percentage": 150.0,  # Out of range
})
```

**Recommendation:**
```python
def _extract_progress_info(task_info: Dict[str, Any]) -> Optional[TaskProgressInfo]:
    """Extract and validate progress information from Celery task info."""
    if task_info.get("state") != "PROGRESS" or "result" not in task_info:
        return None

    progress_data = task_info["result"]
    if not isinstance(progress_data, dict):
        logger.warning(f"Invalid progress data type: {type(progress_data)}")
        return None

    try:
        # Validate and extract with proper types
        current = int(progress_data.get("current", 0))
        total = int(progress_data.get("total", 1))

        # Validate ranges
        if total <= 0:
            logger.warning(f"Invalid total value: {total}")
            total = 1
        if current < 0:
            current = 0
        if current > total:
            logger.warning(f"Current ({current}) > Total ({total}), capping")
            current = total

        # Calculate or validate percentage
        calculated_percentage = (current / total) * 100 if total > 0 else 0
        provided_percentage = float(progress_data.get("percentage", calculated_percentage))

        # Use calculated percentage if provided one is out of range
        if not (0 <= provided_percentage <= 100):
            logger.warning(f"Invalid percentage: {provided_percentage}, using calculated")
            percentage = calculated_percentage
        else:
            percentage = provided_percentage

        # Validate message
        message = str(progress_data.get("message", ""))
        if len(message) > 500:
            message = message[:497] + "..."

        # Validate timestamp
        timestamp = progress_data.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            logger.warning("Missing or invalid timestamp in progress data")
            timestamp = datetime.utcnow().timestamp()

        return TaskProgressInfo(
            current=current,
            total=total,
            percentage=round(percentage, 2),
            message=message,
            timestamp=timestamp,
        )

    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error extracting progress info: {e}")
        return None
```

---

### ISSUE #7: Wait Command Timeout Not Configurable
**Severity:** MEDIUM
**Impact:** MEDIUM - Long-running tasks can't be monitored
**Location:** `/home/tuomo/code/fileintel/src/fileintel/cli/tasks.py:156-169`

**Problem:**
The `wait` command accepts a timeout parameter but doesn't pass it to the monitoring function.

**Code:**
```python
# Line 156-169
@app.command("wait")
def wait_for_task(
    task_id: str = typer.Argument(..., help="The ID of the task to wait for."),
    timeout: Optional[int] = typer.Option(
        None, "--timeout", "-t", help="Maximum time to wait in seconds."
    ),
):
    """Wait for a task to complete and show progress."""
    try:
        monitor_task_with_progress(task_id, "Task execution")  # ← timeout not passed!
    except KeyboardInterrupt:
        cli_handler.console.print(
            f"\n[yellow]Stopped monitoring task {task_id}[/yellow]"
        )
```

**Downstream Code:**
```python
# src/fileintel/cli/shared.py:137-165
def monitor_task_with_progress(task_id: str, task_description: str = "Processing"):
    """Monitor task progress with live updates."""
    api = cli_handler.get_api_client()

    try:
        # Use the existing monitoring functionality from task_client
        result = api.wait_for_task_completion(
            task_id, timeout=None, show_progress=True  # ← timeout hardcoded to None!
        )
```

**Issue:**
- User specifies timeout but it's ignored
- `wait_for_task_completion` is called with `timeout=None` (infinite wait)
- Long-running tasks will hang indefinitely unless user hits Ctrl+C

**Recommendation:**
```python
# Fix: tasks.py
@app.command("wait")
def wait_for_task(
    task_id: str = typer.Argument(..., help="The ID of the task to wait for."),
    timeout: Optional[int] = typer.Option(
        300, "--timeout", "-t", help="Maximum time to wait in seconds (default: 300)."
    ),
):
    """Wait for a task to complete and show progress."""
    try:
        monitor_task_with_progress(task_id, "Task execution", timeout=timeout)
    except TimeoutError:
        cli_handler.display_error(f"Task did not complete within {timeout} seconds")
        cli_handler.console.print(
            f"Use 'fileintel tasks get {task_id}' to check current status"
        )
        raise typer.Exit(1)
    except KeyboardInterrupt:
        cli_handler.console.print(
            f"\n[yellow]Stopped monitoring task {task_id}[/yellow]"
        )

# Fix: shared.py
def monitor_task_with_progress(
    task_id: str,
    task_description: str = "Processing",
    timeout: Optional[int] = None
):
    """Monitor task progress with live updates."""
    api = cli_handler.get_api_client()

    # Use provided timeout or default from constants
    actual_timeout = timeout if timeout is not None else DEFAULT_TASK_TIMEOUT

    try:
        result = api.wait_for_task_completion(
            task_id, timeout=actual_timeout, show_progress=True
        )
        # ... rest of code
```

---

### ISSUE #8: Race Condition in Task Status Check
**Severity:** MEDIUM
**Impact:** LOW - Rare edge case causing incorrect status
**Location:** `/home/tuomo/code/fileintel/src/fileintel/cli/task_client.py:272-286`

**Problem:**
Task state can change between status check and progress update, causing stale information to be displayed.

**Code:**
```python
# Line 272-286
while time.time() - start_time < timeout:
    status_response = self._check_task_status(task_id)  # ← Check 1
    task_data = status_response["data"]
    task_status = task_data["status"]

    self._update_progress_display(
        progress, task_progress, task_id, task_data       # ← Use data from Check 1
    )

    if task_status in ["SUCCESS", "FAILURE", "REVOKED"]:  # ← Check status from Check 1
        progress.update(task_progress, completed=PROGRESS_BAR_TOTAL)
        return status_response                             # ← Return data from Check 1

    time.sleep(poll_interval)  # ← Task could complete here
```

**Race Condition Scenario:**
1. Status check shows task at 90% progress with PROGRESS state
2. Task completes before `_update_progress_display` is called
3. Progress bar shows 90% and then suddenly 100% without showing final task result
4. Final status check may miss task result if it transitions again

**Impact:**
- Progress display may skip final progress updates
- Users might not see final task output
- State transitions between PROGRESS → SUCCESS are lost

**Recommendation:**
```python
def _wait_with_progress(
    self, task_id: str, timeout: int, poll_interval: float, start_time: float
) -> Dict[str, Any]:
    """Wait for task completion with progress display."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=self.console,
    )

    last_status = None
    with progress:
        task_progress = progress.add_task(
            f"Task {task_id[:TASK_ID_DISPLAY_LENGTH]}...", total=PROGRESS_BAR_TOTAL
        )

        while time.time() - start_time < timeout:
            # Atomic status check
            status_response = self._check_task_status(task_id)
            task_data = status_response["data"]
            task_status = task_data["status"]

            # Check if status changed
            if task_status != last_status:
                logger.debug(f"Task {task_id} status changed: {last_status} → {task_status}")
                last_status = task_status

            # Update display with fresh data
            self._update_progress_display(
                progress, task_progress, task_id, task_data
            )

            # Check terminal states
            if task_status in ["SUCCESS", "FAILURE", "REVOKED"]:
                # Ensure progress bar shows 100% for completed tasks
                if task_status == "SUCCESS":
                    progress.update(task_progress, completed=PROGRESS_BAR_TOTAL)

                # Log completion
                logger.info(f"Task {task_id} completed with status: {task_status}")

                # Perform final status check to ensure we have latest data
                final_status = self._check_task_status(task_id)
                return final_status

            time.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
```

---

## Additional High-Priority Issues

### ISSUE #9: No Result Size Validation in Tasks
**Severity:** HIGH
**Impact:** MEDIUM - Large results cause memory issues
**Location:** `/home/tuomo/code/fileintel/src/fileintel/tasks/workflow_tasks.py` (multiple locations)

**Problem:**
Tasks return results without validating size, potentially causing:
- Memory exhaustion in workers
- Redis/backend storage issues
- Slow serialization/deserialization
- Network timeouts

**Example:**
```python
# Line 98-113 in workflow_tasks.py
return {
    "collection_id": collection_id,
    "workflow_task_id": workflow_result.id,
    "status": "processing_with_metadata_and_embeddings",
    "message": f"Started processing {len(file_paths)} documents...",
    # Could potentially include:
    # "chunks": all_chunks,  # ← Could be thousands of chunks
    # "documents": all_docs,  # ← Could be hundreds of documents
    # "embeddings": embeddings,  # ← Could be megabytes of data
}
```

**Recommendation:**
- Implement result size validation in `BaseFileIntelTask`
- Store large results in database and return references
- Add configuration for max result size

```python
# In base.py
class BaseFileIntelTask(Task):
    MAX_RESULT_SIZE_MB = 5  # Configurable limit

    def validate_result_size(self, result: Any) -> bool:
        """Validate that result is not too large."""
        try:
            import sys
            size_bytes = sys.getsizeof(json.dumps(result))
            size_mb = size_bytes / (1024 * 1024)

            if size_mb > self.MAX_RESULT_SIZE_MB:
                logger.warning(
                    f"Task result too large: {size_mb:.2f}MB (limit: {self.MAX_RESULT_SIZE_MB}MB)"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating result size: {e}")
            return False
```

---

### ISSUE #10: Missing Error Context in Failed Tasks
**Severity:** MEDIUM
**Impact:** MEDIUM - Debugging failures is difficult
**Location:** `/home/tuomo/code/fileintel/src/fileintel/api/routes/tasks_v2.py:132-141`

**Problem:**
Error information returned to clients lacks context about what operation failed and why.

**Code:**
```python
# Line 132-141
status_response = TaskStatusResponse(
    task_id=task_id,
    status=_map_celery_state_to_task_state(task_info["state"]),
    result=_format_task_result(task_info.get("result"))
    if task_info["state"] == "SUCCESS"
    else None,
    error=str(task_info.get("result")) if task_info["state"] == "FAILURE" else None,  # ← Just str(exception)
    progress=progress,
    started_at=None,
    completed_at=None,
    worker_id=task_info.get("worker_id"),
    retry_count=task_info.get("retries", 0),
)
```

**Issue:**
- `error` field is just the exception string
- No stack trace for debugging
- No information about which operation failed
- No structured error data (error codes, categories)

**Example Current Error:**
```
"error": "list index out of range"
```

**Recommendation:**
```python
def _format_task_error(task_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format task error with full context."""
    if task_info["state"] != "FAILURE":
        return None

    error_data = {
        "message": str(task_info.get("result", "Unknown error")),
        "type": type(task_info.get("result")).__name__ if task_info.get("result") else "UnknownError",
    }

    # Include traceback if available
    if task_info.get("traceback"):
        error_data["traceback"] = task_info["traceback"]

    # Include task metadata
    if task_info.get("info"):
        error_data["context"] = task_info["info"]

    return error_data

# Use in status response
status_response = TaskStatusResponse(
    task_id=task_id,
    status=_map_celery_state_to_task_state(task_info["state"]),
    result=_format_task_result(task_info.get("result"))
    if task_info["state"] == "SUCCESS"
    else None,
    error=_format_task_error(task_info),  # ← Use structured error
    progress=progress,
    started_at=None,
    completed_at=None,
    worker_id=task_info.get("worker_id"),
    retry_count=task_info.get("retries", 0),
)
```

---

### ISSUE #11: Polling Inefficiency
**Severity:** MEDIUM
**Impact:** LOW - Unnecessary load on backend
**Location:** `/home/tuomo/code/fileintel/src/fileintel/cli/task_client.py:289-303`

**Problem:**
Fixed polling interval doesn't adapt to task status, causing unnecessary requests.

**Code:**
```python
# Line 289-303
def _wait_simple(
    self, task_id: str, timeout: int, poll_interval: float, start_time: float
) -> Dict[str, Any]:
    """Wait for task completion without progress display."""
    while time.time() - start_time < timeout:
        status_response = self._check_task_status(task_id)
        task_data = status_response["data"]
        task_status = task_data["status"]

        if task_status in ["SUCCESS", "FAILURE", "REVOKED"]:
            return status_response

        time.sleep(poll_interval)  # ← Always 2 seconds, regardless of task progress
```

**Issue:**
- Polls every 2 seconds even for long-running tasks (hours)
- No adaptive polling based on task progress or estimated duration
- Creates unnecessary load when task is in early stages

**Recommendation:**
Implement adaptive polling:

```python
def _calculate_adaptive_interval(
    self,
    task_data: Dict[str, Any],
    base_interval: float = 2.0,
    max_interval: float = 30.0
) -> float:
    """Calculate adaptive polling interval based on task progress."""
    progress_info = task_data.get("progress")

    if not progress_info:
        # No progress info, use base interval
        return base_interval

    percentage = progress_info.get("percentage", 0)

    if percentage < 10:
        # Task just started, poll frequently
        return base_interval
    elif percentage < 50:
        # Task in early stages, moderate polling
        return min(base_interval * 2, max_interval)
    elif percentage < 90:
        # Task in middle stages, slower polling
        return min(base_interval * 3, max_interval)
    else:
        # Task almost done, poll frequently again
        return base_interval

def _wait_simple(
    self, task_id: str, timeout: int, poll_interval: float, start_time: float
) -> Dict[str, Any]:
    """Wait for task completion without progress display."""
    while time.time() - start_time < timeout:
        status_response = self._check_task_status(task_id)
        task_data = status_response["data"]
        task_status = task_data["status"]

        if task_status in ["SUCCESS", "FAILURE", "REVOKED"]:
            return status_response

        # Use adaptive polling
        adaptive_interval = self._calculate_adaptive_interval(task_data, poll_interval)
        time.sleep(adaptive_interval)

    raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
```

---

### ISSUE #12: WebSocket Integration Not Used by CLI
**Severity:** LOW
**Impact:** MEDIUM - CLI uses inefficient polling instead of real-time updates
**Location:** `/home/tuomo/code/fileintel/src/fileintel/cli/task_client.py:229-303`

**Problem:**
WebSocket infrastructure exists for real-time task monitoring but CLI uses polling instead.

**Code:**
WebSocket endpoint exists at `/home/tuomo/code/fileintel/src/fileintel/api/routes/websocket_v2.py` but is never used by CLI.

**Current Implementation:**
- WebSocket provides real-time updates every 2 seconds
- CLI polls REST API every 2 seconds
- Duplicate monitoring infrastructure

**Impact:**
- Higher latency for CLI users (REST roundtrip vs. push)
- More backend load (every CLI user polls independently)
- No real-time progress updates

**Recommendation:**
Implement WebSocket-based monitoring for CLI:

```python
# In task_client.py
import websockets
import asyncio
import json

async def _wait_with_websocket(
    self, task_id: str, timeout: int
) -> Dict[str, Any]:
    """Wait for task completion using WebSocket for real-time updates."""
    ws_url = self.base_url_v2.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/tasks/{task_id}/monitor"

    try:
        async with websockets.connect(ws_url) as websocket:
            # Subscribe to task
            await websocket.send(json.dumps({
                "type": "subscribe_task",
                "task_id": task_id
            }))

            start_time = time.time()

            while time.time() - start_time < timeout:
                # Wait for message with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=30
                    )
                    data = json.loads(message)

                    # Handle different event types
                    if data.get("event_type") == "task_completed":
                        return {
                            "success": True,
                            "data": data.get("data", {})
                        }
                    elif data.get("event_type") == "task_failed":
                        return {
                            "success": False,
                            "error": data.get("data", {}).get("error", "Unknown error")
                        }
                    elif data.get("event_type") == "task_progress":
                        # Update progress display
                        # ... update progress bar
                        pass

                except asyncio.TimeoutError:
                    # No message received, check if still connected
                    await websocket.ping()
                    continue

    except Exception as e:
        logger.warning(f"WebSocket monitoring failed, falling back to polling: {e}")
        # Fall back to polling
        return self._wait_with_progress(task_id, timeout, 2.0, time.time())
```

---

## Configuration Issues

### Issue: Result Expiration Configuration Not Documented
**Location:** `/home/tuomo/code/fileintel/config/default.yaml`

**Problem:**
Result expiration (1 hour) is hardcoded in `/home/tuomo/code/fileintel/src/fileintel/celery_config.py:139` but not exposed in configuration.

**Recommendation:**
Add to config:

```yaml
celery:
  result_expires: 3600  # Results expire after 1 hour (in seconds)
  result_extended_expires: 86400  # Keep metadata for 24 hours
```

And update celery_config.py to use config value:

```python
# In configure_celery_app()
result_expires=celery_settings.result_expires,
```

---

## Testing Gaps

Based on the analysis, the following scenarios are **NOT** covered by existing tests:

1. **Result retrieval after expiration** - No test for expired results
2. **Non-serializable result handling** - No test for tasks returning file handles, objects, etc.
3. **Large result handling** - No test for multi-megabyte results
4. **Race conditions in status polling** - No concurrent test cases
5. **Timeout behavior** - No tests for timeout scenarios
6. **Progress update validation** - No tests for malformed progress data
7. **Field name mismatches** - No integration test covering CLI → API → Backend flow
8. **WebSocket monitoring** - No tests for WebSocket vs. polling equivalence

**Recommendation:**
Create comprehensive integration tests in `/home/tuomo/code/fileintel/tests/integration/test_task_result_retrieval.py`

---

## Recommendations Summary

### Immediate Fixes (Critical)
1. **Fix result endpoint** to return actual task results, not wrapper
2. **Fix field name mismatches** between CLI and API
3. **Add result serialization validation** to prevent runtime failures
4. **Implement timeout parameters** in CLI commands

### Short-term Improvements (High Priority)
5. **Add result expiration notifications** with better error messages
6. **Improve progress information validation** with proper error handling
7. **Add result size limits** to prevent memory issues
8. **Enhance error context** in failed task responses

### Long-term Enhancements (Medium Priority)
9. **Implement adaptive polling** to reduce backend load
10. **Add WebSocket support to CLI** for real-time updates
11. **Add comprehensive integration tests** for all scenarios
12. **Document result expiration** and configuration options

---

## Severity Ratings Explained

- **CRITICAL**: Will cause runtime failures or complete feature non-functionality
- **HIGH**: Causes poor user experience or potential failures in edge cases
- **MEDIUM**: Performance issues or missing features that have workarounds
- **LOW**: Minor inconveniences or inefficiencies

---

## Files Requiring Changes

1. `/home/tuomo/code/fileintel/src/fileintel/api/routes/tasks_v2.py` - Result endpoint rewrite
2. `/home/tuomo/code/fileintel/src/fileintel/cli/tasks.py` - Field name fixes, timeout handling
3. `/home/tuomo/code/fileintel/src/fileintel/cli/task_client.py` - Timeout passing, adaptive polling
4. `/home/tuomo/code/fileintel/src/fileintel/cli/shared.py` - Timeout parameter support
5. `/home/tuomo/code/fileintel/src/fileintel/tasks/base.py` - Result size validation
6. `/home/tuomo/code/fileintel/src/fileintel/celery_config.py` - Metadata tracking
7. `/home/tuomo/code/fileintel/config/default.yaml` - Add result expiration config

---

## Conclusion

The task result retrieval and monitoring pipeline has **significant architectural and implementation issues** that will cause runtime failures and poor user experience. The most critical issue is the mismatch between what the API returns and what the CLI expects, which means the `result` command **currently does not work correctly**.

Priority should be given to fixing the critical issues (#1-#5) as they affect basic functionality. The remaining issues, while important, can be addressed in subsequent iterations.
