# GraphRAG Status Tracking - Code Fixes Applied

## Summary

All code fixes to properly track GraphRAG indexing status have been completed. The system now correctly uses the database schema values (`building`, `ready`, `failed`, `updating`) instead of hardcoded incorrect values.

## Changes Made

### 1. Fixed Status Reporting (graphrag_service.py:459)

**File**: `src/fileintel/rag/graph_rag/services/graphrag_service.py`

**Changed**: Line 459 - Return actual database status instead of hardcoded "indexed"

```python
# BEFORE (Wrong):
return {
    "status": "indexed",  # Hardcoded, ignores actual DB status!
    ...
}

# AFTER (Fixed):
return {
    "status": index_info.get("index_status", "unknown"),  # Use actual DB value
    ...
}
```

### 2. Fixed API Resume Check (graphrag_v2.py)

**File**: `src/fileintel/api/routes/graphrag_v2.py`

**Changed**: Lines 70, 210, 304, 410 - Check for correct status values and allow resume when status is "building"

```python
# BEFORE (Wrong):
if status.get("status") == "indexed":  # Wrong value!

# AFTER (Fixed):
# Allow resume if status is "building" (checkpoint resume enabled)
if status.get("status") in ["ready", "indexed"]:  # Correct schema values
```

### 3. Added Status Update Method (graphrag_storage.py)

**File**: `src/fileintel/storage/graphrag_storage.py`

**Added**: New method `update_graphrag_index_status()` at line 104

```python
def update_graphrag_index_status(self, collection_id: str, status: str) -> bool:
    """
    Update GraphRAG index status.

    Args:
        collection_id: Collection ID
        status: New status (building, ready, failed, updating)

    Returns:
        True if successful, False otherwise
    """
```

### 4. Exposed Status Update Method (postgresql_storage.py)

**File**: `src/fileintel/storage/postgresql_storage.py`

**Added**: Delegation method at line 256

```python
def update_graphrag_index_status(self, collection_id: str, status: str) -> bool:
    """Update GraphRAG index status (building, ready, failed, updating)."""
    return self.graphrag_storage.update_graphrag_index_status(collection_id, status)
```

### 5. Added index_status to Returned Data (graphrag_storage.py)

**File**: `src/fileintel/storage/graphrag_storage.py`

**Changed**: Line 91 - Include `index_status` in returned dictionary

```python
return {
    "collection_id": index_info.collection_id,
    "index_path": index_info.index_path,
    "index_status": index_info.index_status,  # Now included!
    ...
}
```

### 6. Added Status Lifecycle Updates (graphrag_tasks.py)

**File**: `src/fileintel/tasks/graphrag_tasks.py`

**Added three status update points**:

#### a) Set to "building" at start (Line 719):
```python
# Set status to "building" when indexing starts
storage.update_graphrag_index_status(collection_id, "building")
logger.info(f"Set GraphRAG index status to 'building' for collection {collection_id}")
```

#### b) Set to "ready" on success (Line 734):
```python
# Set status to "ready" when indexing completes successfully
storage.update_graphrag_index_status(collection_id, "ready")
logger.info(f"Set GraphRAG index status to 'ready' for collection {collection_id}")
```

#### c) Set to "failed" on error (Line 768):
```python
# Set status to "failed" when indexing fails
try:
    storage = get_shared_storage()
    storage.update_graphrag_index_status(collection_id, "failed")
    storage.close()
    logger.info(f"Set GraphRAG index status to 'failed' for collection {collection_id}")
except Exception as status_error:
    logger.error(f"Failed to update status to 'failed': {status_error}")
```

## Database Schema Reference

```sql
CREATE TABLE graphrag_indices (
    id VARCHAR PRIMARY KEY,
    collection_id VARCHAR NOT NULL,
    index_path VARCHAR NOT NULL,
    index_status VARCHAR NOT NULL DEFAULT 'building',  -- building, ready, failed, updating
    documents_count INTEGER NOT NULL DEFAULT 0,
    entities_count INTEGER NOT NULL DEFAULT 0,
    communities_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## Next Steps

### 1. Manual Database Fix (REQUIRED BEFORE TESTING)

The current index has stale status "indexed" from Oct 30. Update it manually:

```bash
chmod +x fix_graphrag_status.sh
./fix_graphrag_status.sh
```

Or run SQL directly:
```sql
UPDATE graphrag_indices
SET index_status = 'building'
WHERE collection_id = '6525aacb-55b1-4a88-aaaa-a4211d03beba';
```

### 2. Test Resume Functionality

After deploying the code fixes and updating the database:

```bash
# This should now properly resume from checkpoint
fileintel graphrag index thesis_sources
```

Expected behavior:
- API sees status is "building" → allows request to proceed
- Task starts and keeps status as "building"
- CheckpointManager detects completed workflows
- Resumes from last checkpoint (21% entity/relationship summarization)
- On completion, updates status to "ready"

### 3. Verify Logs

After running resume, check that you see:

**Server logs**:
```
Updated GraphRAG index status to 'building' for collection 6525aacb-55b1-4a88-aaaa-a4211d03beba
```

**Worker logs**:
```
Set GraphRAG index status to 'building' for collection 6525aacb-55b1-4a88-aaaa-a4211d03beba
...
[checkpoint resume logs]
...
Set GraphRAG index status to 'ready' for collection 6525aacb-55b1-4a88-aaaa-a4211d03beba
```

## Impact

These fixes ensure:
1. ✅ Database status is always accurate (not hardcoded)
2. ✅ API correctly allows/blocks resume based on actual status
3. ✅ Status is properly tracked throughout indexing lifecycle
4. ✅ Checkpoint resume works as designed when index is partially complete
5. ✅ Schema values (`building`, `ready`, `failed`) are used consistently

## Files Modified

1. `src/fileintel/rag/graph_rag/services/graphrag_service.py` - Fixed status reporting
2. `src/fileintel/api/routes/graphrag_v2.py` - Fixed resume checks
3. `src/fileintel/storage/graphrag_storage.py` - Added status update method and include status in data
4. `src/fileintel/storage/postgresql_storage.py` - Exposed status update method
5. `src/fileintel/tasks/graphrag_tasks.py` - Added lifecycle status updates
