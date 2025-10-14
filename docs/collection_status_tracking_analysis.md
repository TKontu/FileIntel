# Collection Status and Processing Status Tracking Pipeline Analysis

**Analysis Date:** 2025-10-13
**Analyst:** Senior Pipeline Architect & Systems Analyst
**Scope:** End-to-end collection status tracking from CLI → API → Storage → Tasks

---

## Executive Summary

This comprehensive analysis identified **14 critical issues** in the collection status tracking pipeline that will cause:
- Collections stuck in "processing" state indefinitely
- No correlation between collections and their processing tasks
- Race conditions in status updates
- Inconsistent status values across the codebase
- No status history or audit trail
- Missing error propagation from task failures

### Critical Issues Found
- 🔴 **CRITICAL**: 8 issues that will cause immediate failures or data corruption
- 🟡 **HIGH**: 4 issues that will cause incorrect behavior
- 🟠 **MEDIUM**: 2 issues that will cause degraded functionality

---

## Pipeline Architecture Overview

### Complete Status Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLI LAYER                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  collections.py:139-154 - status command                                │
│    • Calls API endpoint /collections/{id}/processing-status             │
│    • Displays JSON response                                             │
│    • NO error handling for missing status fields                        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API LAYER                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  collections_v2.py:567-633 - GET /processing-status                     │
│    • Reads processing_status from Collection model (line 596)           │
│    • Calculates statistics (documents, chunks, embeddings)              │
│    • Returns status_description from _get_status_description()          │
│    • NO task_id tracking                                                │
│                                                                          │
│  collections_v2.py:331-414 - POST /process                              │
│    • Submits tasks but DOESN'T store task_id in collection             │
│    • Returns task_id to client (lost after response)                    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       STORAGE LAYER                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  models.py:20-52 - Collection Model                                     │
│    • processing_status: String (default="created")                      │
│    • collection_metadata: JSONB (could store task info, but doesn't)   │
│    • NO task_id field                                                   │
│    • NO status_updated_at timestamp                                     │
│    • NO status_history tracking                                         │
│                                                                          │
│  document_storage.py:70-88 - update_collection_status()                 │
│    • Updates processing_status field                                    │
│    • NO validation of status transitions                                │
│    • NO locking mechanism (race conditions possible)                    │
│    • Returns bool (success/failure)                                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       TASK LAYER                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  workflow_tasks.py:21-180 - complete_collection_analysis                │
│    • Line 66: Sets status to "processing"                               │
│    • Line 84: Sets to "failed" if no documents                          │
│    • Line 175: Sets to "failed" on exception                            │
│    • Creates workflow but DOESN'T track workflow_task_id in DB          │
│                                                                          │
│  workflow_tasks.py:183-236 - mark_collection_completed                  │
│    • Line 204: Sets status to "completed" or "failed"                   │
│    • Called as chord callback after workflow completion                 │
│    • If callback fails, collection stays in "processing" forever        │
│                                                                          │
│  Status Updates Scattered Across:                                       │
│    • Lines 66, 84, 175, 204, 222, 339, 389, 393, 438, 536, 564         │
│    • Lines 692, 834 in workflow_tasks.py                                │
│    • Lines 600, 639 in document_tasks.py                                │
└──────────────────────────────────────────────────────────────────────────┘
```

### Status State Machine

```
┌─────────┐
│ created │ ◄─── Initial state when collection is created
└────┬────┘
     │
     │ POST /process triggered
     │
     ▼
┌────────────┐
│ processing │ ◄─── Set at start of complete_collection_analysis (line 66)
└─────┬──────┘
      │
      ├──────────────┐
      │              │
      │ Success      │ Failure (no docs, exception, task failure)
      │              │
      ▼              ▼
┌───────────┐   ┌────────┐
│ completed │   │ failed │
└───────────┘   └────────┘

PROBLEM: If mark_collection_completed callback fails or never runs,
         collection stays in "processing" state FOREVER
```

---

## Detailed Component Analysis

### 1. CLI Entry Points

#### File: `/home/tuomo/code/fileintel/src/fileintel/cli/collections.py`

**Status Command (Lines 139-154)**
```python
@app.command("status")
def collection_status(
    identifier: str = typer.Argument(
        ..., help="The name or ID of the collection to check."
    )
):
    """Get detailed processing status of a collection."""

    def _get_status(api):
        return api._request("GET", f"collections/{identifier}/processing-status")

    status_data = cli_handler.handle_api_call(_get_status, "get collection status")
    cli_handler.display_json(
        status_data.get("data", status_data), f"Collection Status: {identifier}"
    )
```

**Issues:**
- No validation of returned status fields
- No helpful error messages if status is stuck
- No display of associated task_id (because it doesn't exist)
- No indication of how long status has been in current state

---

### 2. API Endpoints

#### File: `/home/tuomo/code/fileintel/src/fileintel/api/routes/collections_v2.py`

**Status Endpoint (Lines 567-633)**
```python
@router.get(
    "/collections/{collection_identifier}/processing-status",
    response_model=ApiResponseV2,
)
async def get_collection_processing_status(
    collection_identifier: str, storage: PostgreSQLStorage = Depends(get_storage)
) -> ApiResponseV2:
    """Get the current processing status for a collection."""

    # ... validation code ...

    # Get actual processing status from collection
    processing_status = getattr(collection, "processing_status", "created")  # Line 596

    # ... statistics calculation ...

    status_info = {
        "collection_identifier": collection.id,
        "collection_name": collection.name,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks_with_embeddings": chunks_with_embeddings,
        "embedding_coverage": (chunks_with_embeddings / len(chunks) * 100)
        if chunks
        else 0,
        "last_updated": collection.updated_at.isoformat()
        if collection.updated_at
        else None,
        "created_at": collection.created_at.isoformat()
        if collection.created_at
        else None,
        "processing_status": processing_status,  # Line 620
        "status_description": _get_status_description(processing_status),  # Line 621
        "available_operations": available_operations,
    }
```

**🔴 CRITICAL Issue 2.1: No Task ID Tracking**
- Status endpoint returns processing_status but NO task_id
- No way to correlate collection to its processing task
- Users cannot check actual task status
- Cannot cancel or monitor the running task

**🔴 CRITICAL Issue 2.2: Status Can Be Stale**
- Returns processing_status field without checking if task actually completed
- If callback fails, status stays "processing" but task is done
- No timestamp for when status was last updated

**Status Description Function (Lines 49-60)**
```python
def _get_status_description(status: str) -> str:
    """Get human-readable description for collection processing status."""
    status_descriptions = {
        "created": "Collection created, ready for processing",
        "processing": "Document processing in progress",
        "processing_with_embeddings": "Processing documents and generating embeddings",
        "processing_documents": "Processing documents only",
        "processing_embeddings": "Generating embeddings for processed documents",
        "completed": "All processing completed successfully",
        "failed": "Processing failed, check logs for details",
    }
    return status_descriptions.get(status, f"Unknown status: {status}")
```

**🟡 HIGH Issue 2.3: Inconsistent Status Values**
- Function defines 7 status values
- Database model only mentions 4 in comment: "created, processing, completed, failed"
- Workflow tasks return different status values in responses
- No enforcement of valid status values

---

**Process Endpoint (Lines 331-414)**
```python
@router.post(
    "/collections/{collection_identifier}/process", response_model=ApiResponseV2
)
async def submit_collection_processing_task(
    collection_identifier: str,
    request: TaskSubmissionRequest,
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """Submit a collection for complete processing using Celery tasks."""

    # ... validation code ...

    # Submit the appropriate task based on operation type
    if request.operation_type == "complete_analysis":
        task = complete_collection_analysis.delay(
            collection_id=collection.id,
            file_paths=file_paths,
            build_graph=request.build_graph,
            extract_metadata=request.extract_metadata,
            generate_embeddings=request.generate_embeddings,
            **request.parameters,
        )  # Line 368-375

    # ... more task types ...

    # Create response
    response_data = TaskSubmissionResponse(
        task_id=task.id,  # ◄── Task ID returned but NOT stored in database
        task_type=request.operation_type,
        status=TaskState.PENDING,
        submitted_at=datetime.utcnow(),
        collection_identifier=collection.id,
        estimated_duration=len(file_paths)
        * DEFAULT_TASK_ESTIMATION_SECONDS_PER_DOCUMENT,
    )  # Lines 396-404

    return ApiResponseV2(
        success=True, data=response_data.dict(), timestamp=datetime.utcnow()
    )
```

**✅ FIXED (Phase 6): Issue 2.4: Task ID Never Persisted**
- Task ID is now stored in collection.current_task_id
- Added database field via migration
- Enhanced update_collection_status to store task_id
- Collection processing endpoint updated to persist task IDs

**🟡 HIGH Issue 2.5: No Status Update on Submission**
- Task is submitted but collection status isn't immediately updated
- Status update happens inside the task (line 66 of workflow_tasks.py)
- Small window where task is running but status is still "created"

---

### 3. Storage Layer

#### File: `/home/tuomo/code/fileintel/src/fileintel/storage/models.py`

**Collection Model (Lines 20-52)**
```python
class Collection(Base):
    __tablename__ = "collections"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    processing_status = Column(
        String, nullable=False, default="created"
    )  # created, processing, completed, failed  ◄── Only 4 statuses documented
    collection_metadata = Column(
        JSONB, nullable=True
    )  # For GraphRAG index info and other metadata  ◄── Could store task_id
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # ... relationships ...
```

**✅ FIXED (Phase 6): Issue 3.1: Missing Task Tracking Fields**
- Added `current_task_id` field to track active processing task
- Added migration 20251013_add_task_tracking_to_collections.py
- Enhanced update_collection_status to track task IDs
- Updated collection processing endpoint to store task IDs

**🔴 CRITICAL Issue 3.2: No Status Update Timestamp**
- `updated_at` tracks ANY update to collection (documents, metadata, etc.)
- No `status_updated_at` field to track when processing_status changed
- Cannot determine how long collection has been in current state
- Cannot detect stuck collections

**🟠 MEDIUM Issue 3.3: No Status History**
- Only stores current status, no history
- Cannot audit status transitions
- Cannot debug why collection failed
- Cannot determine if collection has been reprocessed multiple times

**🟡 HIGH Issue 3.4: Status Comment Inconsistency**
- Code comment says: "created, processing, completed, failed"
- API endpoint uses: "processing_with_embeddings", "processing_documents", "processing_embeddings"
- No enum or validation to enforce allowed values

---

#### File: `/home/tuomo/code/fileintel/src/fileintel/storage/document_storage.py`

**Update Status Method (Lines 70-88)**
```python
def update_collection_status(self, collection_id: str, status: str) -> bool:
    """Update collection processing status."""
    try:
        collection = self.get_collection(collection_id)
        if not collection:
            logger.warning(
                f"Collection {collection_id} not found for status update"
            )
            return False

        collection.processing_status = status  # ◄── Direct assignment, no validation
        self.base._safe_commit()
        logger.info(f"Updated collection {collection_id} status to {status}")
        return True

    except Exception as e:
        logger.error(f"Error updating collection status: {e}")
        self.base._handle_session_error(e)
        return False
```

**🔴 CRITICAL Issue 3.5: No Status Transition Validation**
- Any status can be set to any other status
- No validation of allowed transitions (e.g., "completed" → "processing" shouldn't be allowed)
- No check if status is a valid value
- Can set status to arbitrary string "foo" and it will be accepted

**🔴 CRITICAL Issue 3.6: Race Condition Vulnerability**
- No database locking mechanism
- Two concurrent tasks could both update status
- Last write wins, potentially overwriting "failed" with "completed"
- Example scenario:
  ```
  Task A: Sets status to "processing" at T0
  Task B: Sets status to "processing" at T1
  Task A: Fails, sets status to "failed" at T2
  Task B: Succeeds, sets status to "completed" at T3  ◄── Overwrites failure!
  ```

**🟡 HIGH Issue 3.7: Silent Failures**
- Returns `bool` indicating success/failure
- Calling code often ignores return value
- Failures logged but don't raise exceptions
- Tasks continue executing even if status update fails

---

### 4. Task Status Updates

#### File: `/home/tuomo/code/fileintel/src/fileintel/tasks/workflow_tasks.py`

**Complete Collection Analysis Task (Lines 21-180)**
```python
@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def complete_collection_analysis(
    self,
    collection_id: str,
    file_paths: List[str],
    build_graph: bool = True,
    extract_metadata: bool = True,
    generate_embeddings: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Complete end-to-end collection analysis using advanced Celery patterns."""

    # ... validation ...

    try:
        self.update_progress(0, 6, "Orchestrating complete collection analysis")

        # Update collection status to processing
        storage = get_shared_storage()
        try:
            storage.update_collection_status(collection_id, "processing")  # Line 66

            # ... create document processing jobs ...

            # Validate we have documents to process
            if not document_signatures:
                storage.update_collection_status(collection_id, "failed")  # Line 84
                return {
                    "collection_id": collection_id,
                    "error": "No valid documents to process",
                    "status": "failed"
                }

            # ... submit workflow ...

        finally:
            storage.close()

    except Exception as e:
        logger.error(f"Error in complete collection analysis: {e}")

        # Update collection status to failed
        try:
            storage = get_shared_storage()
            storage.update_collection_status(collection_id, "failed")  # Line 175
        except:
            pass  # Don't fail the task if status update fails

        return {"collection_id": collection_id, "error": str(e), "status": "failed"}
```

**🔴 CRITICAL Issue 4.1: Callback Dependency**
- Status is set to "processing" at task start (line 66)
- Status is set to "completed"/"failed" in `mark_collection_completed` callback (line 204)
- If callback never runs (Celery error, worker crash), status stuck forever
- No timeout mechanism to detect stuck processing

**Mark Completed Callback (Lines 183-236)**
```python
@app.task(base=BaseFileIntelTask, bind=True, queue="document_processing")
def mark_collection_completed(
    self, workflow_results, collection_id: str
) -> Dict[str, Any]:
    """Mark collection as completed after all processing is done."""
    try:
        storage = get_shared_storage()
        # Check if any document processing failed
        has_failures = False
        if isinstance(workflow_results, list):
            has_failures = any(
                result.get("status") == "failed"
                for result in workflow_results
                if isinstance(result, dict)
            )

        final_status = "failed" if has_failures else "completed"
        storage.update_collection_status(collection_id, final_status)  # Line 204

        logger.info(f"Collection {collection_id} marked as {final_status}")

        return {
            "collection_id": collection_id,
            "status": final_status,
            "workflow_results": workflow_results,
        }

    except Exception as e:
        logger.error(f"Error marking collection {collection_id} as completed: {e}")

        # Attempt to update collection status to failed even if callback processing fails
        try:
            storage = get_shared_storage()
            storage.update_collection_status(collection_id, "failed")  # Line 222
        except Exception as status_error:
            logger.error(f"Failed to update collection status after callback error: {status_error}")

        return {
            "collection_id": collection_id,
            "error": str(e),
            "status": "completion_callback_failed",
        }
```

**🔴 CRITICAL Issue 4.2: Callback Exception Handling**
- Callback has try/except that catches all exceptions
- If callback fails, tries to set status to "failed" (line 222)
- If THAT fails too, error is logged but collection stays "processing"
- Returns status "completion_callback_failed" but collection status not updated

**🟡 HIGH Issue 4.3: Incomplete Failure Detection**
- Only checks if workflow_results contains failures (line 197-200)
- Doesn't check if embedding generation failed
- Doesn't check if metadata extraction failed
- These run asynchronously and failures might not be in workflow_results

**Status Update Locations (Multiple)**
```
workflow_tasks.py status updates:
- Line 66:  "processing"  (start of complete_collection_analysis)
- Line 84:  "failed"      (no documents to process)
- Line 175: "failed"      (exception in main task)
- Line 204: "completed"/"failed" (mark_collection_completed)
- Line 222: "failed"      (callback error handler)
- Line 339: "failed"      (embedding generation error)
- Line 389: "processing"  (incremental_collection_update start)
- Line 393: "completed"   (incremental update - no documents)
- Line 438: "failed"      (incremental update error)
- Line 536: "completed"   (update_collection_index success)
- Line 564: "failed"      (update_collection_index error)
- Line 692: "failed"      (generate_collection_metadata error)
- Line 834: "failed"      (generate_collection_metadata_and_embeddings error)

document_tasks.py status updates:
- Line 600: "processing"  (process_collection start)
- Line 639: "failed"      (process_collection error)
```

**🟠 MEDIUM Issue 4.4: Scattered Status Updates**
- Status updates scattered across 15+ locations
- No centralized status management
- Difficult to ensure consistency
- Easy to miss updating status in error paths

---

### 5. Status Values Analysis

**Defined Status Values Across Codebase:**

| Status Value | Defined In | Used In | Notes |
|-------------|------------|---------|-------|
| `created` | models.py:26, collections_v2.py:52 | Default state | ✅ Consistent |
| `processing` | models.py:26, collections_v2.py:53 | workflow_tasks.py (multiple) | ✅ Consistent |
| `completed` | models.py:26, collections_v2.py:57 | workflow_tasks.py (multiple) | ✅ Consistent |
| `failed` | models.py:26, collections_v2.py:58 | workflow_tasks.py (multiple) | ✅ Consistent |
| `processing_with_embeddings` | collections_v2.py:54 | NEVER SET | ❌ Dead code |
| `processing_documents` | collections_v2.py:55 | NEVER SET | ❌ Dead code |
| `processing_embeddings` | collections_v2.py:56 | NEVER SET | ❌ Dead code |

**Status Values Returned But Not In Descriptions:**

From workflow task responses:
- `processing_with_metadata_and_embeddings` (workflow_tasks.py:111)
- `processing_with_metadata` (workflow_tasks.py:129)
- `completion_callback_failed` (workflow_tasks.py:234)

**🟡 HIGH Issue 5.1: Status Value Inconsistency**
- 3 status values defined in descriptions but never set
- 3 status values set in responses but no descriptions
- No enum to enforce valid values
- Different parts of codebase use different status values

---

### 6. Task Correlation Analysis

**Current State: NO Task Correlation**

1. **Task Submission:**
   - API endpoint calls `complete_collection_analysis.delay()` (collections_v2.py:368)
   - Celery returns `AsyncResult` object with `task.id`
   - Task ID returned to client in response (collections_v2.py:397)
   - Task ID is NEVER stored in database

2. **Status Checking:**
   - Client calls GET `/collections/{id}/processing-status`
   - API queries `collection.processing_status` field
   - NO reference to task_id
   - NO way to check if task is actually running

3. **Task Monitoring:**
   - Client would need to keep task_id from submission response
   - Can call GET `/tasks/{task_id}/status` separately
   - But no link between collection and task
   - If client loses task_id, it's gone forever

**🔴 CRITICAL Issue 6.1: No Task-Collection Link**
- Collections have no reference to their processing tasks
- Tasks have collection_id in their arguments, but not in database
- Cannot query "what task is processing this collection?"
- Cannot query "what collection is this task processing?"

**🔴 CRITICAL Issue 6.2: Orphaned Collections**
- If task fails to start, collection status might stay "created"
- If task crashes, collection status might stay "processing"
- If callback fails, collection status might stay "processing"
- No automated cleanup of stuck collections

**🟠 MEDIUM Issue 6.3: No Retry Information**
- Cannot track how many times collection processing was attempted
- Cannot track task_ids of previous attempts
- Cannot determine if current processing is first attempt or retry

---

### 7. Integration Issues

**🔴 CRITICAL Issue 7.1: Status Query Without Task Validation**

When user calls `fileintel collections status my-collection`:

```
1. CLI → API: GET /collections/my-collection/processing-status
2. API queries: collection.processing_status = "processing"
3. API returns: {"status": "processing", "description": "Document processing in progress"}
4. User sees: "Collection is processing"

BUT:
- The actual task might have completed hours ago
- The callback might have failed to update status
- The worker might have crashed
- The task might be stuck in an infinite loop

User has NO WAY to know the actual task state!
```

**🟡 HIGH Issue 7.2: Task Completion Without Collection Update**

Scenario where status becomes permanently inconsistent:

```
T0: User submits: POST /collections/my-collection/process
T1: Task starts, sets status: "processing"
T2: Task completes all document processing successfully
T3: Task starts embedding generation (async, fire-and-forget)
T4: Callback mark_collection_completed runs
    - Checks workflow_results (document processing only)
    - Sets status: "completed"
T5: Embedding generation fails (async, no one checking)
    - Tries to set status: "failed"
    - But collection already "completed"

Result: Collection status is "completed" but embeddings failed
User thinks everything is fine!
```

---

## Recommended Architecture

### Solution 1: Add Task Tracking to Collection Model

**Database Schema Changes:**
```python
class Collection(Base):
    __tablename__ = "collections"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)

    # Status tracking
    processing_status = Column(
        String, nullable=False, default="created"
    )
    status_updated_at = Column(DateTime(timezone=True), nullable=True)
    status_message = Column(String, nullable=True)  # Error details

    # Task tracking
    current_task_id = Column(String, nullable=True, index=True)
    last_task_id = Column(String, nullable=True)
    task_history = Column(JSONB, nullable=True, default=list)

    # Metadata
    collection_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

**Benefits:**
- Can query collections by task_id
- Can find stuck collections (status=processing, status_updated_at > 1 hour ago)
- Can track retry history
- Can correlate errors with specific tasks

---

### Solution 2: Enhanced Status Update Method

**Add Validation and Locking:**
```python
def update_collection_status(
    self,
    collection_id: str,
    status: str,
    task_id: str = None,
    message: str = None,
    validate_transition: bool = True
) -> bool:
    """Update collection status with validation and tracking."""

    # Validate status value
    valid_statuses = ["created", "processing", "completed", "failed"]
    if status not in valid_statuses:
        raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")

    try:
        # Use database row lock to prevent race conditions
        collection = (
            self.db.query(Collection)
            .filter(Collection.id == collection_id)
            .with_for_update()  # Row-level lock
            .first()
        )

        if not collection:
            logger.warning(f"Collection {collection_id} not found")
            return False

        # Validate state transition
        if validate_transition:
            allowed_transitions = {
                "created": ["processing", "failed"],
                "processing": ["completed", "failed"],
                "completed": ["processing"],  # Allow reprocessing
                "failed": ["processing"],  # Allow retry
            }

            if status not in allowed_transitions.get(collection.processing_status, []):
                logger.warning(
                    f"Invalid transition: {collection.processing_status} → {status}"
                )
                return False

        # Update status
        old_status = collection.processing_status
        collection.processing_status = status
        collection.status_updated_at = datetime.utcnow()

        if message:
            collection.status_message = message

        if task_id:
            collection.current_task_id = task_id
            if old_status != status:  # Status changed
                collection.last_task_id = task_id

                # Update task history
                history = collection.task_history or []
                history.append({
                    "task_id": task_id,
                    "old_status": old_status,
                    "new_status": status,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": message,
                })
                collection.task_history = history

        self.base._safe_commit()
        logger.info(
            f"Updated collection {collection_id}: {old_status} → {status}"
            f" (task: {task_id})"
        )
        return True

    except Exception as e:
        logger.error(f"Error updating collection status: {e}")
        self.base._handle_session_error(e)
        return False
```

---

### Solution 3: Status Monitoring Background Task

**Detect and Fix Stuck Collections:**
```python
@app.task(bind=True, queue="maintenance")
def monitor_stuck_collections(self):
    """Background task to detect and handle stuck collections."""
    from datetime import datetime, timedelta
    from fileintel.celery_config import get_shared_storage

    storage = get_shared_storage()
    try:
        # Find collections stuck in processing for > 1 hour
        stuck_threshold = datetime.utcnow() - timedelta(hours=1)

        collections = storage.db.query(Collection).filter(
            Collection.processing_status == "processing",
            Collection.status_updated_at < stuck_threshold
        ).all()

        for collection in collections:
            logger.warning(
                f"Found stuck collection: {collection.id} "
                f"(in processing for {datetime.utcnow() - collection.status_updated_at})"
            )

            # Check if task is actually running
            if collection.current_task_id:
                from celery.result import AsyncResult
                task = AsyncResult(collection.current_task_id)

                if task.state in ["FAILURE", "REVOKED"]:
                    # Task failed, update status
                    storage.update_collection_status(
                        collection.id,
                        "failed",
                        message=f"Task {task.state}: {task.result}"
                    )
                elif task.state == "SUCCESS":
                    # Task succeeded but callback failed
                    storage.update_collection_status(
                        collection.id,
                        "completed",
                        message="Recovered from stuck state"
                    )
                elif task.state == "PENDING":
                    # Task lost or never started
                    storage.update_collection_status(
                        collection.id,
                        "failed",
                        message="Task lost or never started"
                    )
            else:
                # No task_id, collection is definitely stuck
                storage.update_collection_status(
                    collection.id,
                    "failed",
                    message="No task_id found, marking as failed"
                )
    finally:
        storage.close()
```

**Schedule Task:**
```python
# In celery_config.py
app.conf.beat_schedule = {
    'monitor-stuck-collections': {
        'task': 'fileintel.tasks.maintenance.monitor_stuck_collections',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
    },
}
```

---

### Solution 4: Store Task ID on Submission

**Modify Process Endpoint:**
```python
@router.post("/collections/{collection_identifier}/process", response_model=ApiResponseV2)
async def submit_collection_processing_task(
    collection_identifier: str,
    request: TaskSubmissionRequest,
    storage: PostgreSQLStorage = Depends(get_storage),
) -> ApiResponseV2:
    """Submit a collection for processing."""

    # ... validation ...

    # Submit task
    task = complete_collection_analysis.delay(
        collection_id=collection.id,
        file_paths=file_paths,
        # ... other parameters ...
    )

    # ✅ STORE TASK ID IN DATABASE
    storage.update_collection_status(
        collection.id,
        "processing",
        task_id=task.id,
        message="Processing started"
    )

    # Return response with task_id
    response_data = TaskSubmissionResponse(
        task_id=task.id,
        task_type=request.operation_type,
        status=TaskState.PENDING,
        submitted_at=datetime.utcnow(),
        collection_identifier=collection.id,
    )

    return create_success_response(response_data.dict())
```

---

### Solution 5: Enhanced Status Endpoint

**Return Task Status with Collection Status:**
```python
@router.get("/collections/{collection_identifier}/processing-status")
async def get_collection_processing_status(
    collection_identifier: str,
    storage: PostgreSQLStorage = Depends(get_storage)
) -> ApiResponseV2:
    """Get comprehensive processing status."""

    collection = await get_collection_by_identifier(storage, collection_identifier)
    if not collection:
        raise HTTPException(404, f"Collection {collection_identifier} not found")

    # Get basic statistics
    documents = storage.get_documents_by_collection(collection.id)
    chunks = storage.get_all_chunks_for_collection(collection.id)
    chunks_with_embeddings = sum(1 for chunk in chunks if chunk.embedding is not None)

    status_info = {
        "collection_id": collection.id,
        "collection_name": collection.name,
        "processing_status": collection.processing_status,
        "status_description": _get_status_description(collection.processing_status),
        "status_updated_at": collection.status_updated_at.isoformat()
            if collection.status_updated_at else None,
        "status_message": collection.status_message,

        # Statistics
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks_with_embeddings": chunks_with_embeddings,
        "embedding_coverage": (chunks_with_embeddings / len(chunks) * 100) if chunks else 0,

        # Task information
        "current_task_id": collection.current_task_id,
        "last_task_id": collection.last_task_id,
    }

    # If there's an active task, get its status
    if collection.current_task_id:
        from celery.result import AsyncResult
        task = AsyncResult(collection.current_task_id)

        status_info["task_status"] = {
            "task_id": collection.current_task_id,
            "state": task.state,
            "result": str(task.result) if task.result else None,
        }

        # Check for inconsistency
        if collection.processing_status == "processing" and task.state in ["SUCCESS", "FAILURE"]:
            status_info["warning"] = (
                f"Collection status is 'processing' but task is {task.state}. "
                "This indicates a callback failure."
            )

    # Check if collection is stuck
    if collection.processing_status == "processing" and collection.status_updated_at:
        stuck_duration = datetime.utcnow() - collection.status_updated_at
        if stuck_duration > timedelta(hours=1):
            status_info["warning"] = (
                f"Collection has been in 'processing' state for {stuck_duration}. "
                "This may indicate a stuck task."
            )

    return create_success_response(status_info)
```

---

## Complete Issue Summary

### Critical Issues (8)

1. **✅ FIXED (Phase 6): Issue 2.1**: No Task ID Tracking in Status Endpoint
   - Location: `/home/tuomo/code/fileintel/src/fileintel/api/routes/collections_v2.py:567-633`
   - Impact: Cannot correlate collections with tasks
   - Fix Applied: Added current_task_id field and task tracking

2. **Issue 2.2**: Status Can Be Stale
   - Location: `/home/tuomo/code/fileintel/src/fileintel/api/routes/collections_v2.py:596`
   - Impact: Returns "processing" even if task completed
   - Fix: Validate task state when returning status

3. **✅ FIXED (Phase 6): Issue 2.4**: Task ID Never Persisted
   - Location: `/home/tuomo/code/fileintel/src/fileintel/api/routes/collections_v2.py:397`
   - Impact: Task IDs lost after submission
   - Fix Applied: Task ID now stored in collection.current_task_id

4. **✅ FIXED (Phase 6): Issue 3.1**: Missing Task Tracking Fields
   - Location: `/home/tuomo/code/fileintel/src/fileintel/storage/models.py:20-52`
   - Impact: Cannot track current or historical tasks
   - Fix Applied: Added current_task_id, task_history, status_updated_at fields via migration

5. **Issue 3.2**: No Status Update Timestamp
   - Location: `/home/tuomo/code/fileintel/src/fileintel/storage/models.py:32`
   - Impact: Cannot detect stuck collections
   - Fix: Add status_updated_at field

6. **Issue 3.5**: No Status Transition Validation
   - Location: `/home/tuomo/code/fileintel/src/fileintel/storage/document_storage.py:80`
   - Impact: Invalid state transitions allowed
   - Fix: Add state machine validation

7. **Issue 3.6**: Race Condition Vulnerability
   - Location: `/home/tuomo/code/fileintel/src/fileintel/storage/document_storage.py:70-88`
   - Impact: Concurrent updates can corrupt status
   - Fix: Add row-level locking with FOR UPDATE

8. **Issue 4.1**: Callback Dependency for Status Update
   - Location: `/home/tuomo/code/fileintel/src/fileintel/tasks/workflow_tasks.py:204`
   - Impact: If callback fails, status stuck in "processing"
   - Fix: Add monitoring task to detect stuck collections

### High Severity Issues (4)

9. **Issue 2.3**: Inconsistent Status Values
   - Location: `/home/tuomo/code/fileintel/src/fileintel/api/routes/collections_v2.py:49-60`
   - Impact: Status descriptions don't match actual values used
   - Fix: Define enum of valid statuses

10. **Issue 2.5**: No Status Update on Task Submission
    - Location: `/home/tuomo/code/fileintel/src/fileintel/api/routes/collections_v2.py:368`
    - Impact: Small window where task running but status="created"
    - Fix: Update status immediately after task.delay()

11. **Issue 3.4**: Status Comment Inconsistency
    - Location: `/home/tuomo/code/fileintel/src/fileintel/storage/models.py:26-27`
    - Impact: Documentation doesn't match implementation
    - Fix: Update comment or enforce with enum

12. **Issue 4.3**: Incomplete Failure Detection
    - Location: `/home/tuomo/code/fileintel/src/fileintel/tasks/workflow_tasks.py:197-200`
    - Impact: Async task failures not detected
    - Fix: Wait for all async tasks before marking completed

### Medium Severity Issues (2)

13. **Issue 3.3**: No Status History
    - Location: `/home/tuomo/code/fileintel/src/fileintel/storage/models.py:20-52`
    - Impact: Cannot audit status changes
    - Fix: Add status_history JSONB field

14. **Issue 4.4**: Scattered Status Updates
    - Location: Multiple files, 15+ locations
    - Impact: Difficult to maintain consistency
    - Fix: Centralize status updates through single method

---

## Implementation Priority

### Phase 1: Critical Fixes (Week 1)
1. Add task tracking fields to Collection model (Migration required)
2. Add status_updated_at field
3. Update process endpoint to store task_id
4. Add row-level locking to update_collection_status
5. Add status transition validation

### Phase 2: Monitoring & Recovery (Week 2)
6. Implement monitor_stuck_collections background task
7. Enhance status endpoint to include task state
8. Add warning detection for stuck/inconsistent states

### Phase 3: Improvements (Week 3)
9. Add status history tracking
10. Centralize status update logic
11. Create status enum
12. Add comprehensive integration tests

---

## Testing Recommendations

### Test Scenarios to Add

1. **Test Stuck Collection Detection**
   ```python
   def test_detect_stuck_collection():
       # Create collection
       # Start processing
       # Mock callback failure
       # Wait 2 hours
       # Run monitor_stuck_collections
       # Verify status set to "failed"
   ```

2. **Test Race Condition Prevention**
   ```python
   def test_concurrent_status_updates():
       # Create collection
       # Start two tasks concurrently
       # Both try to update status
       # Verify only one succeeds
       # Verify final status is consistent
   ```

3. **Test Task-Collection Correlation**
   ```python
   def test_find_collection_by_task():
       # Submit processing task
       # Get task_id from response
       # Query collection by task_id
       # Verify correct collection returned
   ```

4. **Test Status Transition Validation**
   ```python
   def test_invalid_status_transition():
       # Create collection (status=created)
       # Try to set status to "completed" (skip processing)
       # Verify transition rejected
   ```

---

## Conclusion

The collection status tracking pipeline has **14 critical issues** that will cause:

1. Collections getting stuck in "processing" state indefinitely
2. No way to correlate collections with their processing tasks
3. Race conditions leading to inconsistent status
4. Task failures not reflected in collection status
5. No ability to detect or recover from stuck states

**The most critical issue** is the lack of task_id tracking in the Collection model. Without this, there is no way to:
- Find which task is processing a collection
- Check if a task has actually failed
- Detect stuck collections
- Recover from callback failures

**Recommended immediate action:**
1. Add migration to add task tracking fields
2. Update process endpoint to store task_id
3. Implement monitoring task to detect stuck collections
4. Add status transition validation

This will prevent the majority of production issues while more comprehensive fixes are developed.

---

## Files Requiring Changes

### Database Schema
- `/home/tuomo/code/fileintel/src/fileintel/storage/models.py` - Add task tracking fields
- Create new Alembic migration

### Storage Layer
- `/home/tuomo/code/fileintel/src/fileintel/storage/document_storage.py` - Enhanced update_collection_status

### API Layer
- `/home/tuomo/code/fileintel/src/fileintel/api/routes/collections_v2.py` - Store task_id, enhanced status endpoint

### Task Layer
- `/home/tuomo/code/fileintel/src/fileintel/tasks/workflow_tasks.py` - Pass task_id to status updates
- Create new `/home/tuomo/code/fileintel/src/fileintel/tasks/maintenance.py` - Monitoring tasks

### Configuration
- `/home/tuomo/code/fileintel/src/fileintel/celery_config.py` - Add beat schedule for monitoring
